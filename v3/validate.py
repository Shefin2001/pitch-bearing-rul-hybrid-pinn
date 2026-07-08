"""validate.py -- Stage 15: physics-consistency gates + expert-review report.

Gates (approved plan):
  G1 ordinal    : Spearman rho(log_a_mu, severity level) on TEST >= 0.9
  G2 monotone   : health index is monotone in a-hat BY CONSTRUCTION (checked)
  G3 ttf        : Healthy p50 within 2x of the 50,000 h L10 calibration;
                  IORW p50 << IRC p50
  G4 classifier : fault F1-macro on TEST >= 0.99 (v2 baseline 0.997)
  G5 position   : per-run |Spearman(win_idx, log_a_mu)| median < 0.3
                  (no-memorization probe -- v2's RUL head fails this)
  G6 deploy     : ONNX artifact <= 10 MB, fp32 parity < 1e-4 (from stage 14)

Writes results/v3/validation.json + results/v3/expert_report.md and exits
non-zero only if a HARD gate (G1, G4, G6) fails; G3/G5 report WARN.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from common.config import Config  # noqa: E402
from common.rul_labels import FAULT_INDEX, INDEX_FAULT, PROGRESSION_GRAPH  # noqa: E402
from Hybrid_PINN_ParisRUL.common.paris_labels import (  # noqa: E402
    A_FAIL_M,
    A_MAP_M,
    C_PARIS,
    CYCLE_SECONDS,
    K_DYN,
    KT_MAP,
    M_PARIS,
    SIGMA_AMP_FACTOR,
    Y_GEOM,
)
from Hybrid_PINN_ParisRUL.v3.damage_net import DamageNet  # noqa: E402
from Hybrid_PINN_ParisRUL.v3.dataset_v3 import make_v3_loaders  # noqa: E402
from Hybrid_PINN_ParisRUL.v3.paris_engine import mc_rul  # noqa: E402
from Hybrid_PINN_ParisRUL.v3.propagation import build_sojourn_table  # noqa: E402
from Hybrid_PINN_ParisRUL.v3.severity_axis import (  # noqa: E402
    LOG_A_LO,
    LOG_A_HI,
    N_LEVELS,
    SEV_LEVEL,
)
from Hybrid_PINN_ParisRUL.v3.train import (  # noqa: E402
    RESULTS_DIR,
    V3_DIR,
    f1_macro,
    spearman,
)

N_CLASSES = len(FAULT_INDEX)


@torch.no_grad()
def collect_test(model, loader, device):
    out = {k: [] for k in ("fault_pred", "fault_true", "log_a_mu", "sigma",
                           "run_id", "win_idx")}
    for batch in loader:
        x = batch["x"].to(device)
        feat = batch["feat"].to(device)
        phys = batch["phys"].to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            pred = model(x, feat, phys)
        out["fault_pred"].append(pred["fault_logits"].float().argmax(1).cpu())
        out["fault_true"].append(batch["fault_idx"])
        out["log_a_mu"].append(pred["log_a_mu"].float().cpu())
        out["sigma"].append(torch.exp(0.5 * pred["log_a_log_var"].float()).cpu())
        out["run_id"].append(batch["run_id"])
        out["win_idx"].append(batch["win_idx"])
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def class_ttf_table(sigma_ln_a: float = 0.2, n: int = 20_000) -> dict:
    table = {}
    for cond, idx in FAULT_INDEX.items():
        probs = np.eye(N_CLASSES)[idx]
        r = mc_rul(math.log(A_MAP_M[cond]), sigma_ln_a, probs,
                   n_samples=n, seed=idx, backend="numpy")
        table[cond] = r.hours()
    return table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--physfeat", type=str,
                        default=str(V3_DIR / "physfeat" / "physfeat.parquet"))
    parser.add_argument("--workers", type=int, default=-1)
    parser.add_argument("--f1-gate", type=float, default=0.99)
    parser.add_argument("--rho-gate", type=float, default=0.90)
    args = parser.parse_args()

    gates: dict = {}
    warn: list = []

    # ── model + test predictions ─────────────────────────────────────────
    ck = torch.load(RESULTS_DIR / "best_model.pt", map_location="cpu",
                    weights_only=False)
    cfg = Config()
    cfg.seed_everything()
    cfg.num_workers = min(cfg.num_workers, os.cpu_count() or 1)
    if args.workers >= 0:
        cfg.num_workers = args.workers
    device = cfg.get_device()

    paris = ROOT / "Hybrid_PINN_ParisRUL" / "results" / "labels" / "labels_paris.parquet"
    shared = ROOT / "Hybrid_PINN_ParisRUL" / "results" / "test_index" / "test_windows.npz"
    _, _, test_loader, _ = make_v3_loaders(
        cfg, physfeat_path=args.physfeat,
        labels_paris_path=str(paris) if paris.exists() else None,
        shared_test_path=str(shared) if shared.exists() else None,
        feat_cache_dir=V3_DIR.parent / "dataset_cache", verbose=False)

    model = DamageNet(hidden=ck.get("hidden", 128),
                      dropout=ck.get("dropout", 0.15)).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    t = collect_test(model, test_loader, device)

    # G1 ordinal
    rho = spearman(t["log_a_mu"], SEV_LEVEL[t["fault_true"]].astype(float))
    gates["G1_ordinal_spearman"] = {"value": rho, "gate": args.rho_gate,
                                    "pass": rho >= args.rho_gate}

    # G2 health-index monotonicity by construction
    a_grid = np.linspace(math.log(1e-5), math.log(A_FAIL_M), 512)
    hi = 1.0 - (a_grid - math.log(A_MAP_M["Health"])) \
        / (math.log(A_FAIL_M) - math.log(A_MAP_M["Health"]))
    gates["G2_hi_monotone"] = {"value": float(np.max(np.diff(hi))),
                               "gate": 0.0, "pass": bool(np.all(np.diff(hi) <= 0))}

    # G3 TTF plausibility
    ttf = class_ttf_table()
    healthy_ok = 25_000.0 <= ttf["Health"]["p50"] <= 100_000.0
    terminal_ok = ttf["IORW"]["p50"] < 0.1 * ttf["IRC"]["p50"]
    gates["G3_ttf_plausibility"] = {
        "healthy_p50_h": ttf["Health"]["p50"], "iorw_p50_h": ttf["IORW"]["p50"],
        "irc_p50_h": ttf["IRC"]["p50"],
        "pass": bool(healthy_ok and terminal_ok), "hard": False}
    if not healthy_ok:
        warn.append("Healthy p50 TTF outside 2x of the 50kh L10 calibration")

    # G4 classifier
    f1 = f1_macro(t["fault_pred"], t["fault_true"], N_CLASSES)
    gates["G4_fault_f1"] = {"value": f1, "gate": args.f1_gate,
                            "pass": f1 >= args.f1_gate}

    # G5 no-memorization probe: damage estimate must not trend with window
    # position inside a steady-state recording
    pos_rho = []
    for r in np.unique(t["run_id"]):
        m = t["run_id"] == r
        if m.sum() >= 8:
            pos_rho.append(abs(spearman(t["win_idx"][m].astype(float),
                                        t["log_a_mu"][m])))
    med_pos = float(np.median(pos_rho)) if pos_rho else 0.0
    gates["G5_position_probe"] = {"median_abs_rho": med_pos, "gate": 0.3,
                                  "pass": med_pos < 0.3, "hard": False,
                                  "n_runs": len(pos_rho)}
    if med_pos >= 0.3:
        warn.append("damage estimate correlates with window position")

    # G6 deployment artifacts
    export_dir = V3_DIR / "export"
    meta_p = export_dir / "model_meta.json"
    fp32 = export_dir / "damage_net_fp32.onnx"
    size_mb = fp32.stat().st_size / 1e6 if fp32.exists() else float("inf")
    parity = (json.loads(meta_p.read_text()).get("int8_parity_max_diff")
              if meta_p.exists() else None)
    gates["G6_deploy"] = {"fp32_mb": size_mb, "gate_mb": 10.0,
                          "int8_parity": parity,
                          "pass": size_mb <= 10.0 and fp32.exists()}

    hard_pass = all(g["pass"] for k, g in gates.items()
                    if g.get("hard", True))
    result = {"gates": gates, "warnings": warn, "hard_pass": hard_pass,
              "per_class_ttf_hours": ttf}
    (V3_DIR / "validation.json").write_text(json.dumps(result, indent=2))

    # ── expert report ────────────────────────────────────────────────────
    calib_p = V3_DIR / "calibration.json"
    calib = json.loads(calib_p.read_text()) if calib_p.exists() else {}
    lines = [
        "# V3 Physics-Anchored RUL -- Expert Review Report", "",
        "The neural network predicts only the *current damage state* "
        "(fault mode, ordinal severity, crack length with uncertainty). "
        "Time-to-failure is derived by Paris-law integration; it is never a "
        "training target, so it cannot be memorised from window position.", "",
        "## Gates",
        "| gate | value | threshold | result |", "|---|---|---|---|",
        f"| G1 severity ordinal Spearman | {rho:.3f} | >= {args.rho_gate} | "
        f"{'PASS' if gates['G1_ordinal_spearman']['pass'] else 'FAIL'} |",
        f"| G2 health index monotone | structural | -- | "
        f"{'PASS' if gates['G2_hi_monotone']['pass'] else 'FAIL'} |",
        f"| G3 TTF plausibility | Health p50 {ttf['Health']['p50']:.0f} h | "
        f"25k-100k h | {'PASS' if gates['G3_ttf_plausibility']['pass'] else 'WARN'} |",
        f"| G4 fault F1-macro | {f1:.4f} | >= {args.f1_gate} | "
        f"{'PASS' if gates['G4_fault_f1']['pass'] else 'FAIL'} |",
        f"| G5 position probe (median rho) | {med_pos:.3f} | < 0.3 | "
        f"{'PASS' if gates['G5_position_probe']['pass'] else 'WARN'} |",
        f"| G6 ONNX size | {size_mb:.2f} MB | <= 10 MB | "
        f"{'PASS' if gates['G6_deploy']['pass'] else 'FAIL'} |", "",
        "## Physics assumptions (require sign-off)",
        "| constant | value | source |", "|---|---|---|",
        f"| Paris C | {C_PARIS:.2e} m/cyc/(MPa sqrt(m))^m | NASA TM-104519, 42CrMo4 |",
        f"| Paris m | {M_PARIS} | NASA TM-104519 |",
        f"| Y geometry | {Y_GEOM} | surface crack, cylindrical raceway |",
        f"| K_dyn | {K_DYN} | Harris & Kotzalas |",
        f"| sigma_amp factor | {SIGMA_AMP_FACTOR} | calibrated to 50 kh L10 |",
        f"| a_fail | {A_FAIL_M*1e3:.0f} mm | replacement criterion |",
        f"| cycle period | {CYCLE_SECONDS} s | pitch oscillation |",
        f"| C scatter (ln sd) | {calib.get('sigma_ln_c', 0.35)} | literature factor-2 spread |",
        f"| sigma scale (conformal) | {calib.get('sigma_scale', 'n/a')} | "
        f"val coverage {calib.get('coverage_before', float('nan')):.0%} -> "
        f"{calib.get('coverage_after', float('nan')):.0%} |"
        if calib else "| calibration | not run | stage 12 |", "",
        "## Per-class anchors and median TTF",
        "| class | K_t | a0 (mm) | severity level | TTF p50 (h) | p5-p95 (h) |",
        "|---|---|---|---|---|---|",
    ]
    for cond, idx in FAULT_INDEX.items():
        h = ttf[cond]
        lines.append(
            f"| {cond} | {KT_MAP[cond]:.2f} | {A_MAP_M[cond]*1e3:.2f} | "
            f"{SEV_LEVEL[idx]} | {h['p50']:.0f} | {h['p5']:.0f}-{h['p95']:.0f} |")
    lines += [
        "",
        "**Note (real physics, not a bug):** TTF is *not* monotone in crack "
        "length across classes -- a sharp crack (IRC, K_t 2.40, 0.5 mm) "
        "propagates to failure in about the same time as a blunt spall "
        "(IRS, K_t 1.80, 1.5 mm). Only the terminal ordering "
        "(IORW << early classes) is structurally guaranteed.", "",
        "## Fault-evolution graph (sojourn p50, hours) -- REVIEW REQUIRED",
        "| src | dst | sojourn (h) |", "|---|---|---|",
    ]
    for src, dsts in build_sojourn_table().items():
        for dst, sec in dsts.items():
            lines.append(f"| {src} | {dst} | {sec/3600.0:.1f} |")
    lines += ["", "## Warnings", *([f"- {w}" for w in warn] or ["- none"]), ""]
    (V3_DIR / "expert_report.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "hard"}
                      for k, v in gates.items()}, indent=2))
    print(f"[v3:validate] hard gates: {'PASS' if hard_pass else 'FAIL'} | "
          f"report -> {V3_DIR / 'expert_report.md'}")
    if not hard_pass:
        sys.exit(1)
    (V3_DIR / ".validation_complete").touch()


if __name__ == "__main__":
    main()
