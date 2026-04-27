# Hybrid_PINN_ParisRUL — Novel Pitch Bearing RUL with Absolute Time

## 1. Problem & Goal

Predict **remaining useful life of wind-turbine pitch bearings** in real time
from raw vibration + acoustic signals. Output:

* `rul_relative` ∈ [0, 1] — health score (1 = healthy, 0 = failed)
* `time_to_failure_hours` — absolute time-to-failure (physics-grounded)
* `dominant_fault` + `fault_probabilities` — 12-class diagnosis
* `progression_timeline` + `progression_risk` — likely future fault stages

The system supersedes the v1 dual-NN (MSTCAN + FAN) build, which the
benchmark report (`BENCHMARKS.md`, 2026-04-16) demonstrated was unreliable
due to constant-class RUL labels, window-level data leakage, and a broken
FAN training. v2 fixes all flagged issues and adds physics-grounded absolute-
time output via Paris-law supervision.

## 2. Architecture (dual-track, parallel-then-fuse)

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

### Hybrid track
- **Raw branch:** 4 dilated TCN blocks (dilations 1, 2, 4, 8) → positional
  encoding → 2-layer Transformer encoder → mean-pool ⇒ (B, 256).
- **Feature branch:** 160-D engineered features split into 4 groups (time,
  freq, time-freq, acoustic) → group embeddings → 3× Channel-Temporal
  Mixed-MLP blocks → projection ⇒ (B, 256).
- **Fusion:** cross-attention (query = raw, kv = features) + SE channel
  attention.
- **Heads (4):** RUL relative (sigmoid), log-TTF (linear), fault logits (12),
  progression logits (12).

### PINN track
- TCN backbone (3 blocks, smaller) + small feature MLP → fuse.
- 6 heads including intermediate physics states **crack_a_mm** and
  **delta_sigma_MPa**.
- Two **learnable global scalars** for Paris constants:
  - `C_paris` ≈ 6.9 × 10⁻¹² (initial), parameterised in log-space, clamped.
  - `m_paris` ∈ [2, 5] via affine sigmoid.

### Fusion
- Knowledge distillation: student ≈ scaled-down Hybrid; teachers = Hybrid + PINN.
- Soft-target loss = 0.7 KL/MSE on teacher mean + 0.3 hard-label loss.
- Two final exports:
  - `model_edge_int8.pt` — `torch.ao.quantization.quantize_dynamic` on Linear
    layers, traced TorchScript.
  - `model_cloud_fp16.pt` — `model.half()`, MC Dropout × 30 enabled.

## 3. The Paris-law derivation chain

Why this gives "time to failure" without any run-to-failure data:

```
Rated axial load (8e5 N)
  └→ Stribeck distribution (4-pt contact): Q_max = 5·F / (Z·sin α)
       └→ Hamrock-Brewe Hertzian peak:    σ_H ≈ 0.6·√(Q_max·E*/R*)
            └→ Dynamic amplification:     × K_dyn (1.8, Harris & Kotzalas)
                 └→ Per-class K_t:        × K_t(Healthy ... IORW)  → Δσ_local
                      └→ ΔK = Y · Δσ · √(πa)  with Y = 1.12
                           └→ da/dN = C · ΔK^m   (42CrMo4 fatigue)
                                └→ ∫ a from a_class → 8 mm  →  N_cycles
                                     └→ × 2 s/cycle  →  TTF (seconds)
```

`paris_labels.py` runs this offline once and stores per-window
`(rul_relative, ttf_seconds, log_ttf_seconds)` tied to FPT-piecewise RUL.

## 4. Why the v1 numbers were unreliable

| Issue | v1 symptom | v2 fix |
|---|---|---|
| Constant class-based RUL labels | RMSE 0.032 / R² 0.986 (degenerate) | FPT-piecewise per-recording labels |
| Window-level split | Fault acc 99.6% (leakage) | Run-level (file_idx) split |
| FAN broken | R²=−0.04, fault acc=12% | Mixed-MLP + cross-attention |
| No monotonicity | Predicted RUL could increase | Loss term penalises Δrul > 0 |
| Progression accuracy | Uninterpretable | F1-macro instead |
| Different test windows per model | Silent comparison bug | Shared `test_windows.npz` |
| No absolute time | Output = 0–1 only | Paris-law TTF supervision |

## 5. Honest limitations

* **No real run-to-failure ground truth for pitch bearings.** Confirmed via
  literature search 2026-04-26 and 2026-04-27. TTF is therefore physics-
  grounded synthetic supervision — internally consistent but not externally
  validated.
* **Paris constants C, m** are population-level for 42CrMo4. The PINN learns
  batch-effective values that adapt to CUMTB's load conditions.
* **`K_dyn ≈ 1.8`** is the conservative-typical value (Harris & Kotzalas).
  Real turbines need a single multiplicative correction once their actual
  load spectrum is known — no retraining required.
* **CUMTB is class-based** (each recording = one fault state), not run-to-
  failure. The FPT detection within a single recording is a partial proxy
  for in-run degradation; it does not substitute for true bearing-life data.

## 6. End-to-end pipeline

```
scripts/00_setup.sh        env check, dirs, git tag v1-pre-novel
scripts/01_build_labels.sh FPT labels + Paris-law TTF labels
scripts/02_build_dataset.sh discover runs, materialise shared test index
scripts/03_train_hybrid.sh torchrun DDP/AMP — 30M-param Hybrid (~8 hr A100)
scripts/04_train_pinn.sh   single-GPU PINN (~6 hr A100)
scripts/05_fuse_models.sh  distill student + INT8/FP16 export
scripts/06_evaluate.sh     compare Hybrid / PINN / Ensemble on shared test
scripts/07_inference_smoke.sh API smoke test (edge + cloud + ensemble)
scripts/run_all.sh         master orchestrator
```

Override per stage via env: `EPOCHS=20 BATCH=64 NPROC=2 bash 03_train_hybrid.sh`.

## 7. Inference API

```python
from Hybrid_PINN_ParisRUL.inference import predict
import numpy as np

raw = read_8192_samples_5_channels()      # shape (N≥2048, 5), float32
out = predict(raw, speed="1rpm", mode="cloud")    # MC Dropout × 30
print(out["time_to_failure_hours"], out["dominant_fault"])
```

Modes:
- `"cloud"` — distilled FP16 student, MC Dropout × 30, full uncertainty bounds
- `"edge"`  — distilled INT8 student, single deterministic pass, <50 ms/window
- `"ensemble"` — Hybrid + PINN teachers weighted (0.6 / 0.4), no distillation

Output dict (all units explicit):

| Key | Type | Meaning |
|---|---|---|
| `rul_relative` | float | health score [0,1] |
| `rul_relative_ci95` | (lo, hi) | 95% CI from MC Dropout |
| `time_to_failure_seconds` | float | physics-grounded synthetic estimate |
| `time_to_failure_hours` | float | seconds / 3600 |
| `time_to_failure_ci95_hours` | (lo, hi) | 95% CI |
| `rul_category` | str | Healthy / Early / Moderate / Advanced / Critical |
| `dominant_fault` | str | argmax of fault probabilities |
| `fault_probabilities` | dict[str, float] | all 12 classes |
| `progression_timeline` | list[list[str]] | BFS-ordered future fault stages |
| `progression_risk` | dict[str, float] | per-class risk score |
| `windows_processed` | int | input segmentation count |
| `inference_ms_per_window` | float | wall-time latency |
| `mode` | str | which path was used |

## 8. Targets (post-fix realistic)

| Metric | Target | Source |
|---|---|---|
| RUL RMSE | ≤ 0.08 | FEMTO benchmark equivalent (Source 5) |
| Fault F1-macro | 0.85–0.92 | post-leakage realistic (Source 7) |
| Monotonicity violation | < 1% | Paris-law inspired (Source 9) |
| Edge model size | < 10 MB | dual-export constraint |
| Edge latency | < 50 ms/window | dual-export constraint |
| Cloud latency | < 100 ms/window FP16 | dual-export constraint |

If RUL RMSE > 0.08 → enable Phase 2 of the plan (CycleGAN signal synthesis).

## 9. References

- BENCHMARKS.md — full source list and prior art.
- Lin et al. 2021 — FPT-piecewise RUL labels.
- Lu Ren et al. 2026 — pitch-bearing FEM under extreme loads.
- Harris & Kotzalas — *Rolling Bearing Analysis* (5th ed.) for K_dyn.
- Forman & Mettu, NASA TM-104519 — 42CrMo4 Paris constants.
- Sources 4, 6, 10 in BENCHMARKS.md — TCN-Transformer, SE channel attention,
  Channel-Temporal Mixed MLP architectures.
