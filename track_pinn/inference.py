"""track_pinn/inference.py — Single-track PINN inference."""
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
from Hybrid_PINN_ParisRUL.track_pinn.model import PINNModel  # noqa: E402

CKPT_PATH = ROOT / "Hybrid_PINN_ParisRUL" / "results" / "pinn" / "best_model.pt"


def _design_bandpass(cfg: Config):
    nyq = cfg.sampling_freq / 2.0
    lo = max(cfg.bandpass_low / nyq, 0.001)
    hi = min(cfg.bandpass_high / nyq, 0.999)
    return sp_signal.butter(cfg.filter_order, [lo, hi], btype="band")


def load_pinn(ckpt_path: str | Path = CKPT_PATH,
              device: Optional[torch.device] = None) -> PINNModel:
    cfg = Config()
    if device is None:
        device = cfg.get_device()
    model = PINNModel(n_classes=cfg.n_classes).to(device)
    if Path(ckpt_path).exists():
        ck = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        state = ck.get("state_dict", ck)
        state = {k.replace("_orig_mod.", "").replace("module.", ""): v
                 for k, v in state.items()}
        model.load_state_dict(state, strict=False)
    model.eval()
    return model


@torch.no_grad()
def predict_pinn(raw_signal: np.ndarray, speed: str = "1rpm",
                 ckpt_path: str | Path = CKPT_PATH,
                 mc_passes: int = 30,
                 device: Optional[torch.device] = None) -> Dict:
    cfg = Config()
    if device is None:
        device = cfg.get_device()
    model = load_pinn(ckpt_path, device)

    b, a = _design_bandpass(cfg)
    sig = np.where(np.isfinite(raw_signal), raw_signal, 0.0).astype(np.float32)
    filt = np.empty_like(sig)
    for c in range(sig.shape[1]):
        filt[:, c] = sp_signal.filtfilt(b, a, sig[:, c])

    ws, stride = cfg.window_size, cfg.window_stride
    n = max(1, (filt.shape[0] - ws) // stride + 1)
    z_wins = np.empty((n, sig.shape[1], ws), dtype=np.float32)
    raw_wins = np.empty((n, ws, sig.shape[1]), dtype=np.float32)
    for i in range(n):
        seg = filt[i * stride: i * stride + ws]
        raw_wins[i] = seg
        mu = seg.mean(axis=0, keepdims=True)
        sd = seg.std(axis=0, keepdims=True) + 1e-8
        z_wins[i] = ((seg - mu) / sd).T

    try:
        from approach_2_wave_features.feature_extractor import FeatureExtractor
    except ImportError as e:
        raise ImportError(
            f"[track_pinn.inference] approach_2_wave_features.feature_extractor "
            f"unavailable ({e}). Required for PINN inference."
        ) from e
    feats = FeatureExtractor(cfg, speed=speed).extract(raw_wins)

    x_raw = torch.from_numpy(z_wins).to(device)
    x_feat = torch.from_numpy(feats).to(device)

    if mc_passes > 1:
        model.enable_mc_dropout()

    all_rul, all_log_ttf, all_a, all_ds, all_fault, all_prog = [], [], [], [], [], []
    for _ in range(max(1, mc_passes)):
        out = model(x_raw, x_feat)
        all_rul.append(out["rul"].cpu().numpy())
        all_log_ttf.append(out["log_ttf"].cpu().numpy())
        all_a.append(out["crack_a_mm"].cpu().numpy())
        all_ds.append(out["delta_sigma_MPa"].cpu().numpy())
        all_fault.append(torch.softmax(out["fault_logits"], dim=-1).cpu().numpy())
        all_prog.append(torch.sigmoid(out["prog_logits"]).cpu().numpy())

    return {
        "rul_per_window": np.stack(all_rul).mean(axis=0),
        "rul_std_per_window": np.stack(all_rul).std(axis=0),
        "log_ttf_per_window": np.stack(all_log_ttf).mean(axis=0),
        "log_ttf_std_per_window": np.stack(all_log_ttf).std(axis=0),
        "crack_a_mm_per_window": np.stack(all_a).mean(axis=0),
        "delta_sigma_MPa_per_window": np.stack(all_ds).mean(axis=0),
        "fault_proba_per_window": np.stack(all_fault).mean(axis=0),
        "prog_proba_per_window": np.stack(all_prog).mean(axis=0),
        "C_paris": float(model.C_paris().item()),
        "m_paris": float(model.m_paris().item()),
        "n_windows": z_wins.shape[0],
    }


if __name__ == "__main__":
    import time
    sig = np.random.randn(8192, 5).astype(np.float32)
    t0 = time.time()
    out = predict_pinn(sig, speed="1rpm", mc_passes=1)
    dt = time.time() - t0
    print(f"[pinn:inference] {out['n_windows']} windows in {dt*1000:.1f} ms")
    print(f"  rul[median] = {np.median(out['rul_per_window']):.3f}")
    print(f"  crack_a[median] = {np.median(out['crack_a_mm_per_window']):.3f} mm")
    print(f"  Δσ[median] = {np.median(out['delta_sigma_MPa_per_window']):.1f} MPa")
    print(f"  C = {out['C_paris']:.2e}, m = {out['m_paris']:.2f}")
