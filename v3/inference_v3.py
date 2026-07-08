"""inference_v3.py -- torch-free deployable predictor (ONNX Runtime + NumPy).

    from Hybrid_PINN_ParisRUL.v3.inference_v3 import predict
    out = predict(raw, speed="1rpm")      # raw: (N >= 2048, 5) float32

Pipeline (all NumPy/SciPy):
    bandpass -> windows (2048/1024, z-scored) -> 160-D features (pywt path)
    -> run-level physics features -> ONNX DamageNet -> aggregate windows
    -> Paris MC RUL distribution -> propagation forecast

The 160-D FeatureExtractor transitively imports common.config which needs
torch; when torch is absent we inject a minimal config stub so the pywt
code path still works. If the extractor is unavailable entirely, x_feat
falls back to zeros (raw + phys branches still carry the prediction) with
an explicit warning in the output dict.

Model default is fp32 (1.4 MB, ~2 ms/window CPU -- faster AND exact vs
int8 on x86; pass mode="int8" for the 0.4 MB file on tiny edge targets).
"""
from __future__ import annotations

import json
import math
import sys
import time
import types
import warnings
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from scipy import signal as sp_signal

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from common.rul_labels import (  # noqa: E402  (numpy-only module)
    FAULT_INDEX,
    INDEX_FAULT,
    N_CLASSES,
)
from Hybrid_PINN_ParisRUL.common.paris_labels import A_FAIL_M, A_MAP_M  # noqa: E402
from Hybrid_PINN_ParisRUL.v3.paris_engine import mc_rul  # noqa: E402
from Hybrid_PINN_ParisRUL.v3.propagation import propagation_forecast  # noqa: E402
from Hybrid_PINN_ParisRUL.v3.severity_axis import (  # noqa: E402
    LEVEL_NAMES,
    level_from_log_a,
)
from Hybrid_PINN_ParisRUL.v3.signal_physics import PhysFeatureExtractor  # noqa: E402

# Signal constants (mirror common/config.py without importing torch)
FS = 38_500.0
WIN, STRIDE = 2048, 1024
BP_LO, BP_HI, BP_ORDER = 500.0, 15_000.0, 4

_V3_DIR = ROOT / "Hybrid_PINN_ParisRUL" / "results" / "v3"
DEFAULT_EXPORT_DIR = _V3_DIR / "export"


def _feature_extractor(speed: str):
    """FeatureExtractor with a torch-free config stub when torch is absent."""
    try:
        import torch  # noqa: F401
    except ImportError:
        if "common.config" not in sys.modules:
            stub_cfg = types.SimpleNamespace(
                sampling_freq=FS, n_channels=5, window_size=WIN,
                nb=16, bd_mm=22.0, pd_mm=120.0, contact_angle_deg=15.0)
            mod = types.ModuleType("common.config")
            mod.Config = lambda **kw: stub_cfg
            sys.modules["common.config"] = mod
    from approach_2_wave_features.feature_extractor import FeatureExtractor
    try:
        from common.config import Config
        cfg = Config()
    except Exception:
        cfg = sys.modules["common.config"].Config()
    return FeatureExtractor(cfg, speed=speed)


class V3Predictor:
    def __init__(self, model_dir: str | Path = DEFAULT_EXPORT_DIR,
                 mode: str = "fp32") -> None:
        import onnxruntime as ort
        model_dir = Path(model_dir)
        self.model_path = model_dir / f"damage_net_{mode}.onnx"
        if not self.model_path.exists():
            raise FileNotFoundError(f"{self.model_path} -- run stage 14 first")
        meta_path = model_dir / "model_meta.json"
        self.meta = (json.loads(meta_path.read_text())
                     if meta_path.exists() else {})
        self.sigma_scale = float(self.meta.get("sigma_scale", 1.0))
        pm = self.meta.get("phys_norm_mean")
        ps = self.meta.get("phys_norm_std")
        self.phys_mean = (np.asarray(pm, np.float32) if pm is not None else None)
        self.phys_std = (np.asarray(ps, np.float32) if ps is not None else None)
        self.sess = ort.InferenceSession(str(self.model_path),
                                         providers=["CPUExecutionProvider"])
        self.mode = mode
        self._fx_cache: Dict[str, object] = {}
        b, a = sp_signal.butter(BP_ORDER,
                                [BP_LO / (FS / 2), BP_HI / (FS / 2)],
                                btype="band")
        self._bp = (b, a)

    # ── preprocessing ────────────────────────────────────────────────────
    def _windows(self, raw_filt: np.ndarray):
        n = max(1, (raw_filt.shape[0] - WIN) // STRIDE + 1)
        w = np.lib.stride_tricks.sliding_window_view(
            raw_filt, WIN, axis=0)[::STRIDE][:n].copy().astype(np.float32)
        # w: (n, C, WIN); z-score per window/channel for the raw branch
        mu = w.mean(axis=2, keepdims=True)
        sd = w.std(axis=2, keepdims=True) + 1e-8
        return (w - mu) / sd, w.transpose(0, 2, 1)  # (n,C,WIN), (n,WIN,C)

    def _features_160(self, raw_wins_twc: np.ndarray, speed: str,
                      notes: list) -> np.ndarray:
        try:
            if speed not in self._fx_cache:
                self._fx_cache[speed] = _feature_extractor(speed)
            return np.asarray(self._fx_cache[speed].extract(raw_wins_twc),
                              dtype=np.float32)
        except Exception as e:  # extractor unavailable -> degrade explicitly
            warnings.warn(f"160-D features unavailable ({e}); using zeros")
            notes.append(f"feat_branch_disabled: {e}")
            return np.zeros((raw_wins_twc.shape[0], 160), dtype=np.float32)

    # ── main entry ───────────────────────────────────────────────────────
    def predict(self, raw: np.ndarray, speed: str = "1rpm",
                n_mc: int = 20_000, seed: int = 0) -> Dict:
        t_start = time.perf_counter()
        raw = np.ascontiguousarray(raw, dtype=np.float32)
        assert raw.ndim == 2 and raw.shape[1] >= 5 and raw.shape[0] >= WIN, \
            f"raw must be (N>={WIN}, 5); got {raw.shape}"
        raw = raw[:, :5]
        np.nan_to_num(raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        notes: list = []

        filt = np.empty_like(raw)
        for c in range(5):
            filt[:, c] = sp_signal.filtfilt(*self._bp, raw[:, c])

        x_raw, raw_wins_twc = self._windows(filt)
        x_feat = self._features_160(raw_wins_twc, speed, notes)
        fx = PhysFeatureExtractor(fs=FS, speed=speed, use_gpu=False)
        phys = fx.extract_run(filt)
        if self.phys_mean is not None:
            phys = (phys - self.phys_mean) / self.phys_std
        x_phys = np.tile(phys[None, :], (x_raw.shape[0], 1)).astype(np.float32)

        # ── ONNX forward (batched) ───────────────────────────────────────
        fl, cl, mu, lv = [], [], [], []
        for i in range(0, x_raw.shape[0], 256):
            o = self.sess.run(None, {
                "x_raw": x_raw[i:i + 256], "x_feat": x_feat[i:i + 256],
                "x_phys": x_phys[i:i + 256]})
            fl.append(o[0]); cl.append(o[1]); mu.append(o[2]); lv.append(o[3])
        fault_logits = np.concatenate(fl)
        corn_logits = np.concatenate(cl)
        log_a_mu = np.concatenate(mu)
        log_a_sigma = np.exp(0.5 * np.clip(np.concatenate(lv), -8, 4))

        # ── window aggregation (damage is constant per recording) ────────
        z = fault_logits - fault_logits.max(axis=1, keepdims=True)
        probs = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
        probs = probs.mean(axis=0)
        mu_agg = float(np.median(log_a_mu))
        sigma_agg = float(math.sqrt(np.median(log_a_sigma) ** 2
                                    + log_a_mu.var())) * self.sigma_scale

        exceed = 1.0 / (1.0 + np.exp(-corn_logits))
        level = int(np.median((np.cumprod(exceed, axis=1) > 0.5).sum(axis=1)))
        level_from_a = int(level_from_log_a(np.array([mu_agg]))[0])

        # ── physics: RUL + propagation ───────────────────────────────────
        rul = mc_rul(mu_agg, max(sigma_agg, 1e-3), probs, n_samples=n_mc,
                     seed=seed, backend="numpy")
        prop = propagation_forecast(probs, mu_agg, max(sigma_agg, 1e-3))

        ln_h, ln_f = math.log(A_MAP_M["Health"]), math.log(A_FAIL_M)
        health_index = float(np.clip(1.0 - (mu_agg - ln_h) / (ln_f - ln_h),
                                     0.0, 1.0))
        dominant = INDEX_FAULT[int(probs.argmax())]
        ms = (time.perf_counter() - t_start) * 1e3 / x_raw.shape[0]

        return {
            "rul_hours": rul.hours(),
            "rul_ig_params": {"mu_s": rul.ig_mu, "lambda_s": rul.ig_lambda},
            "health_index": health_index,
            "crack_length_mm": {
                "p50": float(np.exp(mu_agg)) * 1e3,
                "p5": float(np.exp(mu_agg - 1.6449 * sigma_agg)) * 1e3,
                "p95": float(np.exp(mu_agg + 1.6449 * sigma_agg)) * 1e3,
            },
            "dominant_fault": dominant,
            "fault_probabilities": {INDEX_FAULT[i]: float(round(probs[i], 4))
                                    for i in range(N_CLASSES)},
            "severity_level": level,
            "severity_stage": LEVEL_NAMES[level],
            "severity_level_from_crack": level_from_a,
            "propagation": prop,
            "windows_processed": int(x_raw.shape[0]),
            "inference_ms_per_window": round(ms, 2),
            "mode": self.mode,
            "notes": notes,
        }


def predict(raw: np.ndarray, speed: str = "1rpm",
            model_dir: str | Path = DEFAULT_EXPORT_DIR,
            mode: str = "fp32", **kw) -> Dict:
    return V3Predictor(model_dir, mode).predict(raw, speed=speed, **kw)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--npy", help="(N, 5) float32 .npy recording")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--speed", default="1rpm")
    parser.add_argument("--mode", default="fp32", choices=["fp32", "int8"])
    parser.add_argument("--model-dir", default=str(DEFAULT_EXPORT_DIR))
    args = parser.parse_args()

    if args.demo:
        rng = np.random.default_rng(0)
        raw = rng.standard_normal((WIN * 8, 5)).astype(np.float32)
    elif args.npy:
        raw = np.load(args.npy)
    else:
        parser.error("--demo or --npy required")

    out = predict(raw, speed=args.speed, model_dir=args.model_dir,
                  mode=args.mode)
    print(json.dumps(out, indent=2))
