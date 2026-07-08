"""test_v3.py -- v3 track unit + small-scale end-to-end tests.

Fast unit tests always run; the end-to-end test builds a tiny REAL-layout
synthetic parquet (sample-per-row -- the list-per-row fixture in conftest
cannot be streamed by dataset_v2) and drives stages 10 -> 11 -> 12 -> 14 ->
inference on CPU in a few minutes.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Hybrid_PINN_ParisRUL"))


# ---------------------------------------------------------------------------
# Unit: severity axis
# ---------------------------------------------------------------------------

def test_severity_axis_invariants():
    from Hybrid_PINN_ParisRUL.v3.severity_axis import (
        LOG_A_HI, LOG_A_LO, LOG_A_MID, N_LEVELS, SEV_LEVEL, corn_targets,
        level_from_log_a)
    assert N_LEVELS == 9
    assert SEV_LEVEL.min() == 0 and SEV_LEVEL.max() == N_LEVELS - 1
    assert (LOG_A_LO < LOG_A_MID).all() and (LOG_A_MID < LOG_A_HI).all()
    assert (level_from_log_a(LOG_A_MID) == SEV_LEVEL).all()
    t = corn_targets(SEV_LEVEL)
    assert t.shape == (12, N_LEVELS - 1)
    # CORN targets must be monotone non-increasing along thresholds
    assert (np.diff(t, axis=1) <= 0).all()


# ---------------------------------------------------------------------------
# Unit: signal physics
# ---------------------------------------------------------------------------

def test_signal_physics_shapes_and_finiteness():
    from Hybrid_PINN_ParisRUL.v3.signal_physics import (
        PHYS_DIM, PHYS_NAMES, PhysFeatureExtractor, fault_frequencies)
    assert len(PHYS_NAMES) == PHYS_DIM
    f = fault_frequencies(1.0 / 60.0)
    assert 0 < f["FTF"] < f["BSF"] < f["BPFO"] < f["BPFI"]
    fx = PhysFeatureExtractor(speed="1rpm", use_gpu=False)
    rng = np.random.default_rng(0)
    out = fx.extract_run(rng.standard_normal((8192, 5)).astype(np.float32))
    assert out.shape == (PHYS_DIM,) and np.isfinite(out).all()


def test_dual_impulse_recovery():
    from Hybrid_PINN_ParisRUL.v3.signal_physics import PhysFeatureExtractor
    fs = 38_500.0
    fx = PhysFeatureExtractor(fs=fs, speed="1rpm", use_gpu=False)
    rng = np.random.default_rng(1)
    T = int(40 * fs)
    sig = 0.1 * rng.standard_normal((T, 5)).astype(np.float32)
    t = np.arange(300) / fs
    burst = (np.exp(-np.arange(300) / 60.0)
             * np.sin(2 * np.pi * 6000 * t)).astype(np.float32) * 3.0
    dt_true, bpfo = 0.4, fx.freqs["BPFO"]
    for k in range(int(40 * bpfo)):
        for off in (0.0, dt_true):
            i = int((k / bpfo + off) * fs)
            if i + 300 < T:
                sig[i:i + 300, 0] += burst
    from Hybrid_PINN_ParisRUL.v3.signal_physics import PHYS_NAMES
    named = dict(zip(PHYS_NAMES, fx.extract_run(sig)))
    assert abs(named["dual_impulse_dt_s"] - dt_true) < 0.05
    assert named["dual_impulse_conf"] > 0.5


# ---------------------------------------------------------------------------
# Unit: losses + model
# ---------------------------------------------------------------------------

def test_corn_loss_and_probs():
    import torch
    from Hybrid_PINN_ParisRUL.v3.losses import (
        corn_level_probs, corn_loss, corn_predict_level,
        interval_censored_nll)
    torch.manual_seed(0)
    B, K = 32, 9
    logits = torch.randn(B, K - 1)
    levels = torch.randint(0, K, (B,))
    assert corn_loss(logits, levels).item() > 0
    p = corn_level_probs(logits)
    assert torch.allclose(p.sum(1), torch.ones(B), atol=1e-5)
    lv = corn_predict_level(logits)
    assert lv.min() >= 0 and lv.max() <= K - 1
    mu, lo, hi = torch.zeros(B), -torch.ones(B), torch.ones(B)
    assert interval_censored_nll(mu, torch.full((B,), -6.0), lo, hi) \
        < interval_censored_nll(mu + 5, torch.full((B,), -6.0), lo, hi)


def test_damage_net_forward():
    import torch
    from Hybrid_PINN_ParisRUL.v3.damage_net import DamageNet
    from Hybrid_PINN_ParisRUL.v3.signal_physics import PHYS_DIM
    m = DamageNet()
    out = m(torch.randn(2, 5, 2048), torch.randn(2, 160),
            torch.randn(2, PHYS_DIM))
    assert out["fault_logits"].shape == (2, 12)
    assert out["corn_logits"].shape == (2, 8)
    assert out["log_a_mu"].shape == (2,)
    assert m.count_parameters() < 5_000_000


# ---------------------------------------------------------------------------
# Unit: physics engine + propagation
# ---------------------------------------------------------------------------

def test_paris_closed_form_matches_loop():
    from Hybrid_PINN_ParisRUL.common.paris_labels import (
        A_MAP_M, paris_cycles_to_failure)
    from Hybrid_PINN_ParisRUL.v3.paris_engine import (
        DELTA_SIGMA_MPA, paris_cycles_closed_form)
    from common.rul_labels import FAULT_INDEX
    for cond in ("IRC", "ORS", "IORW"):
        n_loop = paris_cycles_to_failure(cond)
        n_cf = float(paris_cycles_closed_form(
            A_MAP_M[cond], DELTA_SIGMA_MPA[FAULT_INDEX[cond]]))
        assert abs(n_cf - n_loop) / max(n_loop, 1) < 0.05


def test_mc_rul_and_ig():
    from common.rul_labels import FAULT_INDEX
    from Hybrid_PINN_ParisRUL.v3.paris_engine import ig_quantile, mc_rul
    probs = np.eye(12)[FAULT_INDEX["IRS"]]
    r = mc_rul(math.log(1.5e-3), 0.25, probs, n_samples=20_000, seed=0,
               backend="numpy")
    assert 0 < r.ttf_seconds_p5 < r.ttf_seconds_p50 < r.ttf_seconds_p95
    q50 = ig_quantile(r.ig_mu, r.ig_lambda, 0.5)
    assert abs(q50 - r.ttf_seconds_p50) / r.ttf_seconds_p50 < 0.25


def test_propagation_forecast():
    from common.rul_labels import FAULT_INDEX
    from Hybrid_PINN_ParisRUL.v3.propagation import (
        propagation_forecast, shortest_time_to_terminal)
    sec, path = shortest_time_to_terminal("Health")
    assert math.isfinite(sec) and path[-1] == "IORW"
    probs = np.zeros(12)
    probs[FAULT_INDEX["IRS"]] = 1.0
    fc = propagation_forecast(probs, math.log(1.6e-3), 0.3)
    assert fc["current"] == "IRS"
    assert abs(sum(s["risk"] for s in fc["next_stages"]) - 1.0) < 0.01
    json.dumps(fc)  # must be JSON-serialisable


# ---------------------------------------------------------------------------
# End-to-end: stages 10 -> 11 -> 12 -> 14 -> inference (CPU, minutes)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def flat_synth_parquet(tmp_path_factory) -> Path:
    """REAL-layout (sample-per-row) synthetic parquet: 12 runs, 4 windows."""
    import pandas as pd
    rng = np.random.default_rng(0)
    frames = []
    for speed in ("1rpm", "3rpm"):
        for cond in ("Health", "IRC", "ORS"):
            for fidx in range(2):
                sig = rng.standard_normal((2048 * 4, 5)).astype(np.float32)
                frames.append(pd.DataFrame({
                    "speed": speed, "condition": cond, "file_idx": fidx,
                    "vib_y_A": sig[:, 0], "vib_x_A": sig[:, 1],
                    "vib_y_B": sig[:, 2], "vib_x_B": sig[:, 3],
                    "acoustic": sig[:, 4]}))
    out = tmp_path_factory.mktemp("v3data") / "flat_synth.parquet"
    pd.concat(frames, ignore_index=True).to_parquet(str(out), index=False)
    return out


@pytest.mark.slow
def test_v3_pipeline_end_to_end(flat_synth_parquet, tmp_path):
    v3_res = tmp_path / "v3_results"
    env = {**os.environ,
           "PARQUET_PATH": str(flat_synth_parquet),
           "RUL_FORCE_SERIAL_PARQUET": "1",
           "PYTHONIOENCODING": "utf-8",
           "PYTHONPATH": str(ROOT),
           "V3_RESULTS_DIR": str(v3_res),
           "OUTPUT_DIR": str(tmp_path / "out")}

    def run(mod, *cli):
        r = subprocess.run([sys.executable, "-m", mod, *cli], env=env,
                           capture_output=True, text=True, cwd=str(ROOT),
                           encoding="utf-8", errors="replace", timeout=900)
        assert r.returncode == 0, f"{mod} failed:\n{r.stdout}\n{r.stderr}"
        return r

    phys_dir = v3_res / "physfeat"
    run("Hybrid_PINN_ParisRUL.v3.build_physfeat", "--workers", "1",
        "--out-dir", str(phys_dir))
    assert (phys_dir / "physfeat.parquet").exists()

    phys = str(phys_dir / "physfeat.parquet")
    run("Hybrid_PINN_ParisRUL.v3.train", "--epochs", "2", "--batch", "16",
        "--workers", "0", "--physfeat", phys)
    assert (v3_res / "damage_net" / "best_model.pt").exists()
    assert (v3_res / "damage_net" / "checkpoint_last.pt").exists()

    run("Hybrid_PINN_ParisRUL.v3.calibrate", "--workers", "0",
        "--physfeat", phys)
    assert (v3_res / "calibration.json").exists()

    run("Hybrid_PINN_ParisRUL.v3.export_onnx")
    export = v3_res / "export"
    assert (export / "damage_net_fp32.onnx").exists()
    assert (export / "damage_net_int8.onnx").exists()
    assert (export / "damage_net_fp32.onnx").stat().st_size < 10e6

    # torch-free-style inference through the public API
    from Hybrid_PINN_ParisRUL.v3.inference_v3 import predict
    rng = np.random.default_rng(0)
    raw = rng.standard_normal((2048 * 6, 5)).astype(np.float32)
    out = predict(raw, speed="1rpm", model_dir=export)
    for key in ("rul_hours", "health_index", "dominant_fault",
                "fault_probabilities", "severity_stage", "propagation",
                "crack_length_mm"):
        assert key in out, f"missing {key}"
    assert out["rul_hours"]["p5"] <= out["rul_hours"]["p50"] \
        <= out["rul_hours"]["p95"]
    assert 0.0 <= out["health_index"] <= 1.0
    json.dumps(out)
