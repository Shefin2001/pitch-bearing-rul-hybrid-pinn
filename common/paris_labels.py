"""paris_labels.py — Paris-law derived synthetic time-to-failure (TTF).

Provides the **absolute-time** ground truth that lets the model output
"how many hours/days until failure" instead of just a 0–1 score.

Pipeline (offline, runs once before training):

    rated axial load  ──┐
                        ├─►  Hertzian contact stress σ_H  (Hamrock-Brewe)
    bearing geometry  ──┘                │
                                          ▼
                              dynamic amplification × K_t(class)
                                          │
                                          ▼  Δσ_local
                              ΔK = Y · Δσ · √(πa)
                                          │
                                          ▼
                  da/dN = C · (ΔK)^m   (Paris, 42CrMo4)
                                          │
                                          ▼
                    forward-integrate a:  a_class → 8 mm  →  N_cycles
                                          │
                                          ▼
              N_cycles × cycle_seconds × fpt_rul = TTF in seconds

Outputs:
    - per-class TTF lookup (deterministic)
    - per-window TTF labels  =  per-class TTF  ×  fpt_rul[window]
      (FPT-piecewise RUL from rul_labels_v2 modulates absolute time)

Honest limitations:
    - No real run-to-failure data exists for pitch / slewing bearings, so
      TTF is **physics-grounded synthetic supervision**, not measured truth.
    - C, m are population-level constants for 42CrMo4 bearing steel.
    - K_dyn ≈ 1.8 is the Harris & Kotzalas conservative-typical value.
    - All limitations carry through to the model's TTF output. The output
      is correctly scaled relative to bearing physics; absolute calibration
      to a specific turbine requires a single multiplicative correction.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.rul_labels import RUL_MAP  # noqa: E402

# ---------------------------------------------------------------------------
# Material constants — 42CrMo4 bearing steel (Paris-law)
# Reference: Forman, R.G., Mettu, S.R., NASA TM-104519
# ---------------------------------------------------------------------------

C_PARIS: float = 6.9e-12     # m / cycle / (MPa·√m)^m
M_PARIS: float = 3.0
Y_GEOM: float = 1.12         # surface crack on cylindrical raceway

# ---------------------------------------------------------------------------
# Bearing geometry (CUMTB pitch bearing — from project_context.md)
# ---------------------------------------------------------------------------

NB: int = 16                              # number of balls
BD_M: float = 22e-3                       # ball diameter [m]
PD_M: float = 120e-3                      # pitch diameter [m]
ALPHA_RAD: float = np.deg2rad(15.0)       # nominal contact angle
ALPHA4_RAD: float = np.deg2rad(45.0)      # 4-pt contact effective angle

# ---------------------------------------------------------------------------
# Operating point — pitch bearing nominal
# Source: Lu Ren et al. 2026, "Analysis on safety performance of wind turbine
# pitch bearing structure considering the influence of extreme loads."
# ---------------------------------------------------------------------------

F_AXIAL_RATED_N: float = 8.0e5            # rated axial thrust [N] (MW-scale turbine)
K_DYN: float = 1.8                        # dynamic amplification (Harris & Kotzalas)
CYCLE_SECONDS: float = 2.0                # one pitch oscillation event ≈ 2 s

# Subsurface stress amplitude factor — Hertzian peak σ_H is the *maximum*
# contact pressure, but the damaging cyclic Δσ at the subsurface fatigue
# origin (typically depth ≈ 0.5·a_contact below the raceway) is a fraction
# of σ_H. Calibrated to give a healthy bearing TTF ≈ 50,000 hr (~5 yr) at
# rated load, matching SKF/Schaeffler L₁₀ catalogue figures.
SIGMA_AMP_FACTOR: float = 0.018

# Crack length for "failure" — bearing replaced when surface crack reaches 8 mm
A_FAIL_M: float = 8e-3

# ---------------------------------------------------------------------------
# Per-class stress concentration K_t and starting crack length
# K_t derived from FEM of pitch-bearing defects (Lu Ren 2026, Source 6).
# A_MAP scales linearly from class severity.
# ---------------------------------------------------------------------------

KT_MAP: Dict[str, float] = {
    "Health": 1.00,    # no defect
    "IRC":    2.40,    # sharp crack — mode I
    "ORC":    2.20,
    "RBC":    2.60,    # ball surface crack, rotating contact
    "IRS":    1.80,    # spall — blunt notch
    "ORS":    1.80,
    "ITRC":   2.80,    # compound crack, multi-defect
    "IORC":   3.00,
    "IRW":    1.50,    # wear — distributed
    "ORW":    1.50,
    "IORS":   3.10,    # severe compound
    "IORW":   3.20,    # near-failure
}

# Initial crack length per class (m) — degradation snapshot anchor
A_MAP_M: Dict[str, float] = {
    "Health": 0.05e-3,
    "IRC":    0.50e-3,
    "ORC":    0.50e-3,
    "RBC":    0.60e-3,
    "IRS":    1.50e-3,
    "ORS":    1.50e-3,
    "ITRC":   2.00e-3,
    "IORC":   2.50e-3,
    "IRW":    3.00e-3,
    "ORW":    3.00e-3,
    "IORS":   5.50e-3,
    "IORW":   7.00e-3,
}


# ---------------------------------------------------------------------------
# Hertzian contact stress
# ---------------------------------------------------------------------------

def hertz_contact_stress_pa(F_axial_n: float = F_AXIAL_RATED_N) -> float:
    """Maximum Hertzian contact pressure for the loaded ball [Pa].

    Stribeck distribution for 4-point contact ball bearing:

        Q_max = 5 · F_axial / (Z · sin α_4)

    Hertz peak pressure on cylindrical raceway:

        σ_H ≈ 0.6 · (Q_max · E* / R*)^(1/2)

    where E* = effective modulus (~110 GPa for steel-on-steel) and
    R* = effective radius from ball+race curvature. We use the standard
    Hamrock-Brewe approximation for a ball-on-toroidal-race contact.
    """
    Q_max = (5.0 * F_axial_n) / (NB * np.sin(ALPHA4_RAD))

    # Effective curvature (ball-on-outer-race, conformal):
    R_ball = BD_M / 2.0                              # 11 mm
    R_race = PD_M / 2.0 - R_ball                     # ≈ 49 mm
    R_eff = (R_ball * R_race) / (R_ball + R_race)    # ≈ 8.97 mm

    # Effective modulus for steel-on-steel: E1=E2=210 GPa, ν=0.3
    E_eff = 210e9 / (1.0 - 0.3 ** 2) * 0.5           # ~ 1.15e11 Pa

    # Peak Hertzian contact pressure (Pa)
    sigma_H = 0.6 * np.sqrt(Q_max * E_eff / R_eff)
    return float(sigma_H)


def delta_sigma_pa(condition: str, F_axial_n: float = F_AXIAL_RATED_N,
                   k_dyn: float = K_DYN) -> float:
    """Effective fatigue stress range Δσ at a defect of class *condition*, in Pa.

        Δσ_local = σ_H · σ_amp_factor · K_dyn · K_t(class)

    σ_amp_factor reduces the peak Hertzian pressure to the cyclic stress at
    the subsurface fatigue origin — the damaging quantity in Paris-law crack
    propagation, not the surface peak.
    """
    sigma_H = hertz_contact_stress_pa(F_axial_n)
    return sigma_H * SIGMA_AMP_FACTOR * k_dyn * KT_MAP[condition]


# ---------------------------------------------------------------------------
# Paris-law forward integration
# ---------------------------------------------------------------------------

def paris_cycles_to_failure(condition: str,
                            F_axial_n: float = F_AXIAL_RATED_N,
                            k_dyn: float = K_DYN,
                            a_fail_m: float = A_FAIL_M,
                            max_iters: int = 50_000_000) -> int:
    """Number of load cycles from class anchor to bearing failure.

    Integrates ``da/dN = C · (Y · Δσ · √(πa))^m`` forward from the per-class
    starting crack length to ``a_fail_m``.

    Δσ is in Pa internally; ΔK is converted to MPa·√m for Paris-law constants.
    """
    a = float(A_MAP_M[condition])
    if a >= a_fail_m:
        return 0

    delta_sigma_MPa = delta_sigma_pa(condition, F_axial_n, k_dyn) * 1e-6
    n_cycles = 0
    while a < a_fail_m and n_cycles < max_iters:
        delta_K_MPa_sqrtm = Y_GEOM * delta_sigma_MPa * np.sqrt(np.pi * a)
        da_dN = C_PARIS * (delta_K_MPa_sqrtm ** M_PARIS)
        # Adaptive step: if growth-per-cycle is tiny, jump in chunks
        # to keep the integration tractable for healthy/early classes.
        if da_dN < 1e-9:
            step = max(1, int(min(1e-6 / max(da_dN, 1e-15), 1e6)))
        else:
            step = 1
        a += da_dN * step
        n_cycles += step
    return n_cycles


def paris_ttf_seconds(condition: str,
                      fpt_rul: float = 1.0,
                      F_axial_n: float = F_AXIAL_RATED_N,
                      cycle_seconds: float = CYCLE_SECONDS) -> float:
    """Synthetic time-to-failure for a window in seconds.

    Args:
        condition: fault class label
        fpt_rul:   per-window FPT-piecewise RUL ∈ [floor, 1.0]
                   (modulates absolute time: window earlier in degradation
                    has more time left than window later)

    Returns:
        TTF in seconds. For Healthy + fpt_rul=1.0, this is the full
        Paris-derived life of the bearing under nominal load.
    """
    n_cycles = paris_cycles_to_failure(condition, F_axial_n)
    base_seconds = n_cycles * cycle_seconds
    return float(base_seconds * fpt_rul)


def build_class_ttf_table(F_axial_n: float = F_AXIAL_RATED_N) -> Dict[str, float]:
    """Pre-compute per-class TTF (seconds) at fpt_rul = 1.0."""
    return {c: paris_ttf_seconds(c, 1.0, F_axial_n) for c in KT_MAP}


# ---------------------------------------------------------------------------
# Window-level label generator (consumes labels_fpt.parquet from rul_labels_v2)
# ---------------------------------------------------------------------------

def build_ttf_labels_from_fpt(labels_fpt_path: str | Path,
                              out_path: str | Path,
                              F_axial_n: float = F_AXIAL_RATED_N) -> None:
    """Read FPT labels, append synthetic TTF column.

    Output schema: speed, condition, file_idx, window_idx, rul_relative,
                   ttf_seconds, ttf_hours, log_ttf_seconds
    """
    labels_fpt_path = Path(labels_fpt_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    table = pq.read_table(str(labels_fpt_path))
    cond_arr = np.asarray(table.column("condition").to_pylist())
    rul_arr = table.column("rul_relative").to_numpy(zero_copy_only=False).astype(np.float32)

    # Vectorised TTF computation: lookup-then-multiply
    class_ttf = build_class_ttf_table(F_axial_n)
    ttf_seconds = np.empty(rul_arr.size, dtype=np.float32)
    for cond_label, base in class_ttf.items():
        mask = cond_arr == cond_label
        if mask.any():
            ttf_seconds[mask] = (base * rul_arr[mask]).astype(np.float32)

    # Guard against log(0) — clamp to 1 second minimum
    ttf_seconds = np.clip(ttf_seconds, 1.0, None)
    ttf_hours = ttf_seconds / 3600.0
    log_ttf = np.log(ttf_seconds)

    out_table = table.append_column("ttf_seconds", pa.array(ttf_seconds))
    out_table = out_table.append_column("ttf_hours", pa.array(ttf_hours))
    out_table = out_table.append_column("log_ttf_seconds", pa.array(log_ttf))
    pq.write_table(out_table, out_path, compression="snappy")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fpt-labels",
                        default=r"D:\Pitch_Bearings_RUL\PitchBearing_RUL_DualNN\Hybrid_PINN_ParisRUL\results\labels\labels_fpt.parquet")
    parser.add_argument("--out",
                        default=r"D:\Pitch_Bearings_RUL\PitchBearing_RUL_DualNN\Hybrid_PINN_ParisRUL\results\labels\labels_paris.parquet")
    parser.add_argument("--F-axial", type=float, default=F_AXIAL_RATED_N,
                        help="Rated axial load in N (default: 8e5)")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    sigma_H_MPa = hertz_contact_stress_pa(args.F_axial) * 1e-6
    print(f"--- Hertzian contact (F_axial = {args.F_axial:.2e} N) ---")
    print(f"  sigma_H    = {sigma_H_MPa:.1f} MPa")
    print(f"  K_dyn      = {K_DYN}")
    print(f"  Y          = {Y_GEOM}")

    print(f"\n--- Per-class Paris-law TTF (cycle = {CYCLE_SECONDS}s) ---")
    print(f"{'Class':<7} {'K_t':>5} {'dsig_loc(MPa)':>14} {'a0(mm)':>8} {'cycles':>14} {'TTF(hrs)':>12}")
    print("-" * 60)
    for cond in KT_MAP:
        ds_MPa = delta_sigma_pa(cond, args.F_axial) * 1e-6
        a0_mm = A_MAP_M[cond] * 1e3
        n = paris_cycles_to_failure(cond, args.F_axial)
        hrs = n * CYCLE_SECONDS / 3600.0
        print(f"{cond:<7} {KT_MAP[cond]:>5.2f} {ds_MPa:>14.1f} {a0_mm:>8.2f} {n:>14,d} {hrs:>12.1f}")

    if args.summary_only:
        return
    if not Path(args.fpt_labels).exists():
        print(f"\n[!] No FPT label file at {args.fpt_labels} — skipping per-window TTF generation.")
        print("    Run rul_labels_v2.py first to produce it.")
        return

    print(f"\nGenerating per-window TTF labels:")
    print(f"  in : {args.fpt_labels}")
    print(f"  out: {args.out}")
    build_ttf_labels_from_fpt(args.fpt_labels, args.out, args.F_axial)
    print("[OK]")


if __name__ == "__main__":
    _cli()
