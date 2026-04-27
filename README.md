# Hybrid + PINN Pitch-Bearing RUL with Paris-Law Absolute Time-to-Failure

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A **dual-track** Remaining-Useful-Life system for wind-turbine pitch bearings
that combines a deep TCN-Transformer-Mixed-MLP **Hybrid** model with a
**Physics-Informed Neural Network (PINN)** supervised by the Paris fatigue
crack-growth law. Outputs both a relative health score and a *physically grounded
absolute time-to-failure in hours* — without requiring run-to-failure ground truth,
which does not exist for pitch bearings in the public domain.

This codebase is the second-generation rebuild of the dual-NN system. The
first-generation MSTCAN + FAN build was reviewed against a 12-source benchmark
(see `track_hybrid/` & `track_pinn/`) and found to suffer from three
disqualifying flaws: constant class-based RUL labels, window-level data leakage,
and a broken FAN training. Every flaw was diagnosed, documented, and fixed here.

---

## Why this project

The first-generation system reported `RUL R² = 0.986` and `Fault accuracy = 99.6%`.
Those numbers were *wrong-because-degenerate*, not wrong-because-noisy:

| Flaw in v1 | What actually happened | v2 fix |
|---|---|---|
| Class-constant RUL labels | Model learned `fault_class → fixed RUL value` (a lookup table) | **FPT-piecewise** per-recording labels (Lin et al. 2021) |
| Window-level train/test split | Same bearing run appeared in both splits | **Run-level** split keyed on `file_idx` |
| FAN training broken | R² = −0.04 — worse than predicting the mean | Replaced with **Channel-Temporal Mixed-MLP + cross-attention** |
| No monotonicity prior | Predicted RUL could increase over time | Loss term penalises `Δrul > 0` |
| Multi-label progression scored as accuracy | 12-class always-zero baseline scored well | Switched to **F1-macro** |
| Different test windows per model | Silent comparison bug across models | Shared, pre-materialised `test_windows.npz` |
| Output capped at `[0, 1]` only | No actionable absolute time | **Paris-law TTF supervision** → hours-to-failure |

The v2 system is calibrated to realistic, post-leakage targets:
RUL RMSE ≤ 0.08, fault F1-macro 0.85–0.92, monotonicity violation < 1%.

---

## Architecture

```
                       ┌────────────────────────┐
                       │     CUMTB parquet      │
                       └────────────┬───────────┘
                                    │
                ┌───────────────────┼────────────────────┐
                │                   │                    │
                ▼                   ▼                    ▼
         FPT-piecewise       Run-level split    Paris-law TTF labels
         RUL labels          (file_idx)         (Hertzian → ΔK → da/dN)
                │                   │                    │
                └───────────────────┼────────────────────┘
                                    │
                  ┌─────────────────┴─────────────────┐
                  │                                   │
                  ▼                                   ▼
         ┌─────────────────┐                ┌─────────────────┐
         │  Track Hybrid   │                │   Track PINN    │
         │  TCN-Transformer│                │  TCN + Paris    │
         │  + Mixed-MLP    │                │  residual loss  │
         │  + Cross-Attn   │                │  + learnable    │
         │  ~30M params    │                │    C, m         │
         └────────┬────────┘                │  ~10M params    │
                  │                         └────────┬────────┘
                  └────────────┬────────────────────┘
                               │
                               ▼
                    ┌───────────────────────┐
                    │   Fusion / Distill    │
                    │  Student ~5M params   │
                    └────────────┬──────────┘
                                 │
                ┌────────────────┴────────────────┐
                ▼                                 ▼
        ┌──────────────┐                  ┌────────────────┐
        │ Edge: INT8   │                  │ Cloud: FP16    │
        │ TorchScript  │                  │ + MC Dropout   │
        │ <10 MB       │                  │ uncertainty    │
        └──────────────┘                  └────────────────┘
```

### Hybrid track (~30 M params)
- **Raw branch** — 4 dilated TCN blocks (dilations 1, 2, 4, 8) → positional encoding → 2-layer Transformer encoder → mean-pool ⇒ (B, 256).
- **Feature branch** — 160-D engineered features split into 4 groups (time, frequency, time-frequency, acoustic) → group embeddings → 3× **Channel-Temporal Mixed-MLP** blocks → projection ⇒ (B, 256).
- **Fusion** — cross-attention (query = raw, kv = features) + SE channel attention.
- **Heads (4)** — RUL relative (sigmoid), log-TTF (linear), fault logits (12), progression logits (12).

### PINN track (~10 M params)
- TCN backbone (3 blocks, smaller) + small feature MLP → fuse.
- **6 heads** including intermediate physics states `crack_a_mm` and `delta_sigma_MPa`.
- **Two learnable global scalars** for the Paris constants:
  - `C_paris ≈ 6.9 × 10⁻¹²` (initial), parameterised in log-space, clamped.
  - `m_paris ∈ [2, 5]` via affine sigmoid.
- Residual loss enforces `da/dN = C · ΔK^m` on every window.

### Fusion / distillation (~5 M-param student)
Knowledge distillation: `student ≈ scaled-down Hybrid`, teachers = `Hybrid + PINN`. Soft-target loss = 0.7·KL/MSE on teacher mean + 0.3·hard-label loss. Two final exports:
- `model_edge_int8.pt` — `torch.ao.quantization.quantize_dynamic` on Linear layers, traced TorchScript.
- `model_cloud_fp16.pt` — `model.half()`, MC Dropout × 30 enabled.

---

## The Paris-law derivation chain

How the system produces *hours-to-failure* without any run-to-failure data:

```
Rated axial load (8e5 N)
  └→ Stribeck distribution (4-pt contact):  Q_max = 5·F / (Z·sin α)
       └→ Hamrock-Brewe Hertzian peak:       σ_H ≈ 0.6·√(Q_max·E*/R*)
            └→ Dynamic amplification:        × K_dyn (1.8, Harris & Kotzalas)
                 └→ Per-class K_t:           × K_t(Healthy ... IORW)  → Δσ_local
                      └→ ΔK = Y · Δσ · √(πa)  with Y = 1.12
                           └→ da/dN = C · ΔK^m   (42CrMo4 fatigue)
                                └→ ∫ a from a_class → 8 mm  →  N_cycles
                                     └→ × 2 s/cycle  →  TTF (seconds)
```

`paris_labels.py` runs this chain offline once and stores per-window
`(rul_relative, ttf_seconds, log_ttf_seconds)` tied to FPT-piecewise RUL.

---

## Inference API

```python
from Hybrid_PINN_ParisRUL.inference import predict
import numpy as np

raw = read_8192_samples_5_channels()           # shape (N≥2048, 5), float32
out = predict(raw, speed="1rpm", mode="cloud") # MC Dropout × 30
print(out["time_to_failure_hours"], out["dominant_fault"])
```

| Mode | Path | Latency target | Use case |
|---|---|---|---|
| `cloud` | distilled FP16 student, MC Dropout × 30 | < 100 ms/window | full uncertainty bounds |
| `edge` | distilled INT8 student, single deterministic pass | < 50 ms/window | on-turbine inference |
| `ensemble` | Hybrid + PINN teachers weighted (0.6 / 0.4) | n/a | best accuracy, no distillation |

Output dict (units always explicit):

| Key | Type | Meaning |
|---|---|---|
| `rul_relative` | float | health score [0, 1] |
| `rul_relative_ci95` | (lo, hi) | 95 % CI from MC Dropout |
| `time_to_failure_seconds` | float | physics-grounded synthetic estimate |
| `time_to_failure_hours` | float | seconds / 3600 |
| `time_to_failure_ci95_hours` | (lo, hi) | 95 % CI |
| `rul_category` | str | Healthy / Early / Moderate / Advanced / Critical |
| `dominant_fault` | str | argmax of fault probabilities |
| `fault_probabilities` | dict[str, float] | all 12 classes |
| `progression_timeline` | list[list[str]] | BFS-ordered future fault stages |
| `progression_risk` | dict[str, float] | per-class risk score |
| `windows_processed` | int | input segmentation count |
| `inference_ms_per_window` | float | wall-time latency |
| `mode` | str | which path was used |

---

## End-to-end pipeline

The pipeline is broken into eight numbered shell scripts under `scripts/`. Each
stage is overrideable per env-var (`EPOCHS=20 BATCH=64 NPROC=2 bash 03_train_hybrid.sh`).

```
scripts/00_setup.sh           env check, dirs, git tag v1-pre-novel
scripts/01_build_labels.sh    FPT labels + Paris-law TTF labels
scripts/02_build_dataset.sh   discover runs, materialise shared test index
scripts/03_train_hybrid.sh    torchrun DDP/AMP — 30 M-param Hybrid (~8 hr A100)
scripts/04_train_pinn.sh      single-GPU PINN (~6 hr A100)
scripts/05_fuse_models.sh     distill student + INT8/FP16 export
scripts/06_evaluate.sh        compare Hybrid / PINN / Ensemble on shared test
scripts/07_inference_smoke.sh API smoke test (edge + cloud + ensemble)
scripts/run_all.sh            master orchestrator
```

### Quickstart

```bash
# 1. Set up env (Python ≥ 3.10, PyTorch 2.x with CUDA)
pip install torch pyarrow scipy numpy pywavelets numba

# 2. Point to your CUMTB parquet (not bundled — multi-GB)
export PARQUET_PATH=/path/to/pitch_bearing_dataset.parquet

# 3. Run the full pipeline
bash scripts/run_all.sh
```

The CUMTB *Vibration and acoustic data of pitch bearing in wind turbines under
time-varying load for fault diagnosis* dataset is **not bundled** with this repo
(several GB). Place the materialised parquet wherever you like and point
`PARQUET_PATH` at it.

---

## Honest limitations

- **No real run-to-failure ground truth for pitch bearings** in the public
  domain (confirmed via literature review, April 2026). The TTF output is
  therefore *physics-grounded synthetic supervision* — internally consistent but
  not externally validated against fielded turbine failures.
- **Paris constants `C, m`** are population-level for 42CrMo4. The PINN learns
  batch-effective values that adapt to CUMTB's load conditions.
- `K_dyn ≈ 1.8` is the conservative-typical value (Harris & Kotzalas). Real
  turbines need a single multiplicative correction once their actual load
  spectrum is known — no retraining required.
- CUMTB is class-based (each recording = one fault state), not run-to-failure.
  The FPT detection within a single recording is a partial proxy for in-run
  degradation; it does not substitute for true bearing-life data.
- This is a research / portfolio build, not production-hardened.

---

## Repository layout

```
common/                shared dataset, label, and feature extraction code
track_hybrid/          Hybrid track — model, trainer, inference
track_pinn/            PINN track — model, trainer, Paris-law residuals
track_fusion/          knowledge distillation + edge/cloud export
scripts/               numbered pipeline shell scripts (00 → 07)
inference.py           top-level predict() entry point
compare_v2.py          Hybrid vs PINN vs Ensemble evaluation
DOCUMENTATION_v2.md    full technical write-up
```

---

## References

- **Lin et al. 2021** — *A Novel Approach of Label Construction for Predicting Remaining Useful Life of Machinery*, Shock and Vibration. (FPT-piecewise RUL labels)
- **Lu Ren et al. 2026** — Pitch-bearing FEM under extreme loads.
- **Harris & Kotzalas** — *Rolling Bearing Analysis* (5th ed.) for `K_dyn`.
- **Forman & Mettu, NASA TM-104519** — 42CrMo4 Paris constants.
- **TCN-Transformer for Bearing RUL**, Sensors 2025 — backbone choice.
- **Channel attention for pitch-bearing AE+vibration**, Measurement 2024 — sensor fusion.
- **Channel-Temporal Mixed-MLP**, 2024 — feature-track architecture.

---

## License

MIT — see [LICENSE](LICENSE).
