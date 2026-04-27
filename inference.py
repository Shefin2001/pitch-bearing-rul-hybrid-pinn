"""Hybrid_PINN_ParisRUL/inference.py — Unified real-time prediction API.

Top-level ``predict()`` accepts raw vibration + acoustic samples and returns
RUL (relative + absolute time) plus fault diagnosis with uncertainty bounds.

Mode dispatch:
    mode="cloud" → load cloud_fp16 model (or distilled student in fp32),
                   run MC Dropout × 30 → uncertainty bounds.
    mode="edge"  → load edge_int8 model, single deterministic pass.
    mode="ensemble" → run Hybrid + PINN teachers, weighted average.

Honest limitations (printed to stdout once per process):
    * TTF supervision is Paris-law-synthesised — no real run-to-failure data
      for pitch bearings exists.
    * Paris constants C, m are population-level for 42CrMo4. Real bearings
      vary; the PINN learned batch-effective values which mitigates this.
    * K_dyn ≈ 1.8 is the conservative-typical value (Harris & Kotzalas).
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from scipy import signal as sp_signal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from common.config import Config  # noqa: E402
from common.rul_labels import INDEX_FAULT, get_progression_timeline, rul_category  # noqa: E402

FUSION_DIR = ROOT / "Hybrid_PINN_ParisRUL" / "results" / "fusion"
HYBRID_CKPT = ROOT / "Hybrid_PINN_ParisRUL" / "results" / "hybrid" / "best_model.pt"
PINN_CKPT = ROOT / "Hybrid_PINN_ParisRUL" / "results" / "pinn" / "best_model.pt"

_BANNER_PRINTED = False


def _print_banner_once() -> None:
    global _BANNER_PRINTED
    if _BANNER_PRINTED:
        return
    print("=" * 72)
    print("Hybrid_PINN_ParisRUL inference — honest limitations")
    print("-" * 72)
    print("  * TTF supervision is Paris-law-synthesised (no real")
    print("    run-to-failure data exists for pitch bearings).")
    print("  * Paris constants are population-level for 42CrMo4 bearing steel.")
    print("  * K_dyn = 1.8 (Harris & Kotzalas conservative-typical).")
    print("  * Absolute time predictions are physics-grounded but not validated")
    print("    against real run-to-failure measurements.")
    print("=" * 72)
    _BANNER_PRINTED = True


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def _design_bandpass(cfg: Config):
    nyq = cfg.sampling_freq / 2.0
    lo = max(cfg.bandpass_low / nyq, 0.001)
    hi = min(cfg.bandpass_high / nyq, 0.999)
    return sp_signal.butter(cfg.filter_order, [lo, hi], btype="band")


def _segment(filt: np.ndarray, ws: int, stride: int):
    n = max(1, (filt.shape[0] - ws) // stride + 1)
    raw_wins = np.empty((n, ws, filt.shape[1]), dtype=np.float32)
    z_wins = np.empty((n, filt.shape[1], ws), dtype=np.float32)
    for i in range(n):
        seg = filt[i * stride: i * stride + ws]
        raw_wins[i] = seg
        mu = seg.mean(axis=0, keepdims=True)
        sd = seg.std(axis=0, keepdims=True) + 1e-8
        z_wins[i] = ((seg - mu) / sd).T
    return z_wins, raw_wins


def _features_or_zeros(raw_wins: np.ndarray, speed: str, cfg: Config) -> np.ndarray:
    # Despite the name, this no longer silently returns zeros. Inference paths
    # that consume the Hybrid model need the real 160-D features; missing them
    # would produce confidently wrong RUL/TTF predictions.
    try:
        from approach_2_wave_features.feature_extractor import FeatureExtractor
    except ImportError as e:
        raise ImportError(
            f"[inference] approach_2_wave_features.feature_extractor unavailable ({e}). "
            f"The Hybrid / fusion / cloud modes require the 160-D engineered features. "
            f"Install approach_2_wave_features alongside Hybrid_PINN_ParisRUL, or call "
            f"the PINN-only inference path explicitly."
        ) from e
    return FeatureExtractor(cfg, speed=speed).extract(raw_wins)


# ---------------------------------------------------------------------------
# Model loading dispatch
# ---------------------------------------------------------------------------

def _load_descriptor() -> Dict:
    desc_path = FUSION_DIR / "fusion_descriptor.json"
    if desc_path.exists():
        with open(desc_path) as f:
            return json.load(f)
    return {"mode": "ensemble", "w_hybrid": 0.6, "w_pinn": 0.4}


def _load_jit_or_eager(jit_path: Path, eager_loader, device):
    if jit_path.exists():
        try:
            return torch.jit.load(str(jit_path), map_location=device)
        except Exception:
            pass
    return eager_loader(device=device)


# ---------------------------------------------------------------------------
# Aggregation utilities
# ---------------------------------------------------------------------------

def _aggregate_window_predictions(
    rul_per_win: np.ndarray, log_ttf_per_win: np.ndarray,
    fault_per_win: np.ndarray, prog_per_win: np.ndarray,
    rul_std: Optional[np.ndarray] = None,
    log_ttf_std: Optional[np.ndarray] = None,
) -> Dict:
    rul = float(np.median(rul_per_win))
    rul_lo, rul_hi = float(np.percentile(rul_per_win, 2.5)), float(np.percentile(rul_per_win, 97.5))
    log_ttf = float(np.median(log_ttf_per_win))
    ttf_seconds = float(np.exp(log_ttf))
    if log_ttf_std is not None:
        ttf_std = float(np.median(log_ttf_std))
        ttf_lo = float(np.exp(log_ttf - 1.96 * ttf_std))
        ttf_hi = float(np.exp(log_ttf + 1.96 * ttf_std))
    else:
        ttf_lo, ttf_hi = ttf_seconds, ttf_seconds

    # Fault: average probabilities across windows, then argmax
    fault_proba_mean = fault_per_win.mean(axis=0)
    dom_idx = int(np.argmax(fault_proba_mean))
    dom_label = INDEX_FAULT[dom_idx]
    fault_proba = {INDEX_FAULT[i]: float(fault_proba_mean[i])
                   for i in range(fault_proba_mean.shape[0])}

    # Progression
    prog_mean = prog_per_win.mean(axis=0)
    prog_risk = {INDEX_FAULT[i]: float(prog_mean[i]) for i in range(prog_mean.shape[0])}

    return {
        "rul_relative": rul,
        "rul_relative_ci95": (rul_lo, rul_hi),
        "time_to_failure_seconds": ttf_seconds,
        "time_to_failure_hours": ttf_seconds / 3600.0,
        "time_to_failure_ci95_hours": (ttf_lo / 3600.0, ttf_hi / 3600.0),
        "rul_category": rul_category(rul),
        "dominant_fault": dom_label,
        "fault_probabilities": fault_proba,
        "progression_timeline": get_progression_timeline(dom_label),
        "progression_risk": prog_risk,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict(raw_signal: np.ndarray, speed: str = "1rpm",
            mode: str = "cloud", mc_passes: Optional[int] = None) -> Dict:
    """Predict RUL + fault diagnosis for a real-time vibration block.

    Args:
        raw_signal: (N, 5) float32 — N ≥ 2048 samples of (vib_y_A, vib_x_A,
                    vib_y_B, vib_x_B, acoustic) at 38.5 kHz.
        speed:      "1rpm" or "3rpm" — pitch bearing motion speed
        mode:       "cloud" | "edge" | "ensemble"
        mc_passes:  override default MC Dropout passes (cloud=30, edge=1)

    Returns:
        dict — see field list in module docstring
    """
    _print_banner_once()
    assert raw_signal.ndim == 2 and raw_signal.shape[1] == 5, \
        f"raw_signal must be (N, 5), got {raw_signal.shape}"
    cfg = Config()
    device = cfg.get_device()

    b, a = _design_bandpass(cfg)
    sig = np.where(np.isfinite(raw_signal), raw_signal, 0.0).astype(np.float32)
    filt = np.empty_like(sig)
    for c in range(sig.shape[1]):
        filt[:, c] = sp_signal.filtfilt(b, a, sig[:, c])
    z_wins, raw_wins = _segment(filt, cfg.window_size, cfg.window_stride)
    feats = _features_or_zeros(raw_wins, speed, cfg)
    x_raw = torch.from_numpy(z_wins).to(device)
    x_feat = torch.from_numpy(feats).to(device)

    desc = _load_descriptor()
    t0 = time.time()

    if mode == "ensemble" or desc["mode"] == "ensemble":
        return _predict_ensemble(x_raw, x_feat, z_wins.shape[0], device, desc, t0)

    if mode == "edge":
        edge_path = Path(desc.get("edge_path", FUSION_DIR / "model_edge_int8.pt"))
        return _predict_distilled(x_raw, x_feat, z_wins.shape[0], device,
                                  edge_path, mc_passes=mc_passes or 1, t0=t0,
                                  fp16=False)
    # cloud
    cloud_path = Path(desc.get("cloud_path", FUSION_DIR / "model_cloud_fp16.pt"))
    return _predict_distilled(x_raw, x_feat, z_wins.shape[0], device,
                              cloud_path, mc_passes=mc_passes or 30, t0=t0,
                              fp16=True)


@torch.no_grad()
def _predict_distilled(x_raw, x_feat, n_win, device, jit_path: Path,
                       mc_passes: int, t0: float, fp16: bool) -> Dict:
    from Hybrid_PINN_ParisRUL.track_fusion.distill import StudentModel

    if jit_path.exists():
        model = torch.jit.load(str(jit_path), map_location=device)
    else:
        # Fallback to eager student checkpoint
        student_path = FUSION_DIR / "student_best.pt"
        model = StudentModel().to(device)
        if student_path.exists():
            ck = torch.load(str(student_path), map_location=device, weights_only=False)
            model.load_state_dict(ck["state_dict"], strict=False)
        model.eval()

    if fp16 and not isinstance(model, torch.jit.ScriptModule):
        model = model.half()
        x_raw = x_raw.half()
        x_feat = x_feat.half()

    if mc_passes > 1 and hasattr(model, "enable_mc_dropout"):
        model.enable_mc_dropout()

    rul_list, log_ttf_list, fault_list, prog_list = [], [], [], []
    for _ in range(max(1, mc_passes)):
        out = model(x_raw, x_feat)
        rul_list.append(out["rul"].float().cpu().numpy())
        log_ttf_list.append(out["log_ttf"].float().cpu().numpy())
        fault_list.append(torch.softmax(out["fault_logits"].float(), dim=-1).cpu().numpy())
        prog_list.append(torch.sigmoid(out["prog_logits"].float()).cpu().numpy())
    rul_arr = np.stack(rul_list)
    log_ttf_arr = np.stack(log_ttf_list)
    fault_arr = np.stack(fault_list)
    prog_arr = np.stack(prog_list)
    rul_std = rul_arr.std(0) if mc_passes > 1 else None
    log_ttf_std = log_ttf_arr.std(0) if mc_passes > 1 else None

    res = _aggregate_window_predictions(
        rul_arr.mean(0), log_ttf_arr.mean(0),
        fault_arr.mean(0), prog_arr.mean(0),
        rul_std, log_ttf_std,
    )
    res["windows_processed"] = n_win
    res["mode"] = "edge" if not fp16 else "cloud"
    res["mc_passes"] = mc_passes
    res["inference_ms_per_window"] = (time.time() - t0) * 1000 / max(n_win, 1)
    return res


@torch.no_grad()
def _predict_ensemble(x_raw, x_feat, n_win, device, desc, t0) -> Dict:
    from Hybrid_PINN_ParisRUL.track_hybrid.inference import load_hybrid
    from Hybrid_PINN_ParisRUL.track_pinn.inference import load_pinn

    w_h = float(desc.get("w_hybrid", 0.6))
    w_p = float(desc.get("w_pinn", 0.4))

    h = load_hybrid(device=device)
    p = load_pinn(device=device)

    h_out = h(x_raw, x_feat)
    p_out = p(x_raw, x_feat)
    rul = (w_h * h_out["rul"] + w_p * p_out["rul"]).cpu().numpy()
    log_ttf = (w_h * h_out["log_ttf"] + w_p * p_out["log_ttf"]).cpu().numpy()
    fault_proba = (w_h * torch.softmax(h_out["fault_logits"], dim=-1)
                   + w_p * torch.softmax(p_out["fault_logits"], dim=-1)).cpu().numpy()
    prog_proba = (w_h * torch.sigmoid(h_out["prog_logits"])
                  + w_p * torch.sigmoid(p_out["prog_logits"])).cpu().numpy()

    res = _aggregate_window_predictions(rul, log_ttf, fault_proba, prog_proba)
    res["windows_processed"] = n_win
    res["mode"] = "ensemble"
    res["ensemble_weights"] = {"hybrid": w_h, "pinn": w_p}
    res["pinn_C_paris"] = float(p.C_paris().item())
    res["pinn_m_paris"] = float(p.m_paris().item())
    res["inference_ms_per_window"] = (time.time() - t0) * 1000 / max(n_win, 1)
    return res


if __name__ == "__main__":
    sig = np.random.randn(8192, 5).astype(np.float32)
    print("\n--- ENSEMBLE ---")
    out = predict(sig, speed="1rpm", mode="ensemble")
    for k, v in out.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for kk, vv in list(v.items())[:5]:
                print(f"    {kk}: {vv}")
        else:
            print(f"  {k}: {v}")
