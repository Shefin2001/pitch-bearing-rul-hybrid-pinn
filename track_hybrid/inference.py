"""track_hybrid/inference.py — Single-track Hybrid inference with MC Dropout.

Used by the unified API in Hybrid_PINN_ParisRUL/inference.py and by
compare_v2.py. Loads ``results/hybrid/best_model.pt`` and exports a
TorchScript build to ``results/hybrid/model_jit.pt``.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from scipy import signal as sp_signal

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from common.config import Config  # noqa: E402
from Hybrid_PINN_ParisRUL.track_hybrid.model import HybridParisModel  # noqa: E402

CKPT_PATH = ROOT / "Hybrid_PINN_ParisRUL" / "results" / "hybrid" / "best_model.pt"


# ---------------------------------------------------------------------------
# Preprocessing — same pipeline as dataset_v2 but for live signals
# ---------------------------------------------------------------------------

def _design_bandpass(cfg: Config):
    nyq = cfg.sampling_freq / 2.0
    lo = max(cfg.bandpass_low / nyq, 0.001)
    hi = min(cfg.bandpass_high / nyq, 0.999)
    return sp_signal.butter(cfg.filter_order, [lo, hi], btype="band")


def _segment_and_normalise(signal_NxC: np.ndarray, ws: int, stride: int):
    """Return (windows_z (N, C, ws), windows_raw (N, ws, C))."""
    n = max(1, (signal_NxC.shape[0] - ws) // stride + 1)
    raw_wins = np.empty((n, ws, signal_NxC.shape[1]), dtype=np.float32)
    z_wins = np.empty((n, signal_NxC.shape[1], ws), dtype=np.float32)
    for i in range(n):
        seg = signal_NxC[i * stride: i * stride + ws]
        raw_wins[i] = seg
        mu = seg.mean(axis=0, keepdims=True)
        sd = seg.std(axis=0, keepdims=True) + 1e-8
        z_wins[i] = ((seg - mu) / sd).T
    return z_wins, raw_wins


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_hybrid(ckpt_path: str | Path = CKPT_PATH,
                device: Optional[torch.device] = None) -> HybridParisModel:
    cfg = Config()
    if device is None:
        device = cfg.get_device()
    model = HybridParisModel(n_classes=cfg.n_classes).to(device)
    if Path(ckpt_path).exists():
        ck = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        state = ck.get("state_dict", ck)
        # Strip torch.compile / DDP prefixes
        state = {k.replace("_orig_mod.", "").replace("module.", ""): v
                 for k, v in state.items()}
        model.load_state_dict(state, strict=False)
    model.eval()
    return model


def export_jit(model: HybridParisModel, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        scripted = torch.jit.script(model)
    except Exception:
        # Fallback to trace
        x_raw = torch.zeros(1, 5, 2048)
        x_feat = torch.zeros(1, 160)
        scripted = torch.jit.trace(model, (x_raw, x_feat), strict=False)
    torch.jit.save(scripted, str(out_path))


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_hybrid(raw_signal: np.ndarray, speed: str = "1rpm",
                   ckpt_path: str | Path = CKPT_PATH,
                   mc_passes: int = 30,
                   device: Optional[torch.device] = None) -> Dict:
    """Run Hybrid track on a raw vibration block.

    Args:
        raw_signal: (N, 5) float32 — at least 2048 samples of 5-channel data
        speed:      "1rpm" or "3rpm"
        ckpt_path:  path to best_model.pt
        mc_passes:  number of MC Dropout passes for uncertainty (1 = deterministic)
        device:     torch device or None for auto

    Returns:
        dict with rul, log_ttf, fault_logits, prog_logits and uncertainty bounds
    """
    cfg = Config()
    if device is None:
        device = cfg.get_device()
    model = load_hybrid(ckpt_path, device)

    # Preprocess
    b, a = _design_bandpass(cfg)
    sig = np.where(np.isfinite(raw_signal), raw_signal, 0.0).astype(np.float32)
    filt = np.empty_like(sig)
    for c in range(sig.shape[1]):
        filt[:, c] = sp_signal.filtfilt(b, a, sig[:, c])
    z_wins, raw_wins = _segment_and_normalise(filt, cfg.window_size, cfg.window_stride)

    # Features — fail loudly. The Hybrid model's 160-D head was trained on
    # real engineered features; zero-filling produces silently-wrong predictions.
    try:
        from approach_2_wave_features.feature_extractor import FeatureExtractor
    except ImportError as e:
        raise ImportError(
            f"[track_hybrid.inference] approach_2_wave_features.feature_extractor "
            f"unavailable ({e}). Required for Hybrid inference."
        ) from e
    extractor = FeatureExtractor(cfg, speed=speed)
    feats = extractor.extract(raw_wins)

    x_raw = torch.from_numpy(z_wins).to(device)
    x_feat = torch.from_numpy(feats).to(device)

    # MC Dropout
    if mc_passes > 1:
        model.enable_mc_dropout()
    else:
        model.eval()

    all_rul, all_log_ttf, all_fault, all_prog = [], [], [], []
    for _ in range(max(1, mc_passes)):
        out = model(x_raw, x_feat)
        all_rul.append(out["rul"].cpu().numpy())
        all_log_ttf.append(out["log_ttf"].cpu().numpy())
        all_fault.append(torch.softmax(out["fault_logits"], dim=-1).cpu().numpy())
        all_prog.append(torch.sigmoid(out["prog_logits"]).cpu().numpy())

    rul_stack = np.stack(all_rul, axis=0)        # (M, N)
    log_ttf_stack = np.stack(all_log_ttf, axis=0)
    fault_stack = np.stack(all_fault, axis=0)    # (M, N, 12)
    prog_stack = np.stack(all_prog, axis=0)

    return {
        "rul_per_window": rul_stack.mean(axis=0),
        "rul_std_per_window": rul_stack.std(axis=0),
        "log_ttf_per_window": log_ttf_stack.mean(axis=0),
        "log_ttf_std_per_window": log_ttf_stack.std(axis=0),
        "fault_proba_per_window": fault_stack.mean(axis=0),
        "prog_proba_per_window": prog_stack.mean(axis=0),
        "n_windows": z_wins.shape[0],
    }


if __name__ == "__main__":
    import time
    cfg = Config()
    print("[hybrid:inference] smoke test")
    sig = np.random.randn(8192, 5).astype(np.float32)
    t0 = time.time()
    out = predict_hybrid(sig, speed="1rpm", mc_passes=1)
    dt = time.time() - t0
    print(f"  windows: {out['n_windows']}")
    print(f"  rul[median]: {np.median(out['rul_per_window']):.3f}")
    print(f"  log_ttf[median]: {np.median(out['log_ttf_per_window']):.3f}")
    print(f"  elapsed: {dt * 1000:.1f} ms ({dt * 1000 / out['n_windows']:.2f} ms/window)")
