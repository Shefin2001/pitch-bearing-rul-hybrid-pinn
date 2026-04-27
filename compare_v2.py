"""compare_v2.py — Head-to-head evaluation on shared test index.

Replaces the buggy ``compare_approaches.py`` (line 389 silent bug where
rul_true was taken from only one model's dataset). Now:
    * loads ``results/test_index/test_windows.npz`` (built by dataset_v2 --build)
    * runs Hybrid, PINN, and the fused student (or ensemble) on identical windows
    * reports F1-macro instead of accuracy on the multi-label progression head
    * computes monotonicity violation rate per run
    * computes ECE for MC Dropout uncertainty
    * writes ``results/comparison_v2.csv`` and ``results/comparison_v2.png``
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from common.config import Config  # noqa: E402
from Hybrid_PINN_ParisRUL.common.dataset_v2 import PitchBearingDataset  # noqa: E402
from Hybrid_PINN_ParisRUL.common.metrics_v2 import (  # noqa: E402
    evaluate_all,
    expected_calibration_error,
    monotonicity_violation_rate,
)
from Hybrid_PINN_ParisRUL.track_hybrid.inference import load_hybrid  # noqa: E402
from Hybrid_PINN_ParisRUL.track_pinn.inference import load_pinn  # noqa: E402

RESULTS_DIR = ROOT / "Hybrid_PINN_ParisRUL" / "results"


@torch.no_grad()
def _evaluate_model(model, loader, device, n_classes: int = 12,
                    mc_passes: int = 1) -> Dict[str, float]:
    if mc_passes > 1:
        if hasattr(model, "enable_mc_dropout"):
            model.enable_mc_dropout()
    else:
        model.eval()

    preds = {"rul": [], "log_ttf": [], "fault_logits": [], "prog_logits": []}
    targs = {"rul": [], "log_ttf": [], "fault_idx": [], "prog_mask": []}
    run_ids = []
    win_idxs = []
    fault_correct = []
    fault_max_prob = []

    for batch in loader:
        x_raw = batch["x"].to(device, non_blocking=True)
        x_feat = batch["feat"].to(device, non_blocking=True)
        target = {k: batch[k] for k in ("rul", "log_ttf", "fault_idx",
                                        "prog_mask", "run_id", "win_idx")}

        if mc_passes > 1:
            mc_outs = [model(x_raw, x_feat) for _ in range(mc_passes)]
            pred = {
                "rul": torch.stack([o["rul"] for o in mc_outs]).mean(0),
                "log_ttf": torch.stack([o["log_ttf"] for o in mc_outs]).mean(0),
                "fault_logits": torch.stack([o["fault_logits"] for o in mc_outs]).mean(0),
                "prog_logits": torch.stack([o["prog_logits"] for o in mc_outs]).mean(0),
            }
        else:
            pred = model(x_raw, x_feat)

        prob = torch.softmax(pred["fault_logits"], dim=-1)
        pmax, pidx = prob.max(dim=-1)
        fault_correct.append((pidx.cpu() == target["fault_idx"]).float())
        fault_max_prob.append(pmax.cpu())

        for k in preds:
            preds[k].append(pred[k].detach().cpu())
        for k in targs:
            targs[k].append(target[k])
        run_ids.append(target["run_id"])
        win_idxs.append(target["win_idx"])

    cat_p = {k: torch.cat(v) for k, v in preds.items()}
    cat_t = {k: torch.cat(v) for k, v in targs.items()}
    metrics = evaluate_all(cat_p, cat_t, n_classes=n_classes)

    # Monotonicity per run
    run_id = torch.cat(run_ids)
    win_idx = torch.cat(win_idxs)
    rul = cat_p["rul"]
    mono_terms = []
    for r in torch.unique(run_id):
        mask = run_id == r
        if mask.sum() < 2:
            continue
        order = torch.argsort(win_idx[mask])
        mono_terms.append(monotonicity_violation_rate(rul[mask][order]).item())
    metrics["mono_violation_rate"] = float(np.mean(mono_terms)) if mono_terms else 0.0

    # ECE on fault confidence
    if fault_max_prob:
        ece = expected_calibration_error(torch.cat(fault_max_prob),
                                         torch.cat(fault_correct))
        metrics["fault_ece"] = ece
    return metrics


def _ensemble_evaluate(hybrid, pinn, loader, device, w_h: float = 0.6,
                       w_p: float = 0.4) -> Dict[str, float]:
    hybrid.eval()
    pinn.eval()
    preds = {"rul": [], "log_ttf": [], "fault_logits": [], "prog_logits": []}
    targs = {"rul": [], "log_ttf": [], "fault_idx": [], "prog_mask": []}
    run_ids, win_idxs = [], []
    with torch.no_grad():
        for batch in loader:
            x_raw = batch["x"].to(device, non_blocking=True)
            x_feat = batch["feat"].to(device, non_blocking=True)
            target = {k: batch[k] for k in ("rul", "log_ttf", "fault_idx",
                                            "prog_mask", "run_id", "win_idx")}
            h_out = hybrid(x_raw, x_feat)
            p_out = pinn(x_raw, x_feat)
            pred = {
                "rul": (w_h * h_out["rul"] + w_p * p_out["rul"]).detach().cpu(),
                "log_ttf": (w_h * h_out["log_ttf"] + w_p * p_out["log_ttf"]).detach().cpu(),
                "fault_logits": (w_h * h_out["fault_logits"] + w_p * p_out["fault_logits"]).detach().cpu(),
                "prog_logits": (w_h * h_out["prog_logits"] + w_p * p_out["prog_logits"]).detach().cpu(),
            }
            for k in preds:
                preds[k].append(pred[k])
            for k in targs:
                targs[k].append(target[k])
            run_ids.append(target["run_id"])
            win_idxs.append(target["win_idx"])
    cat_p = {k: torch.cat(v) for k, v in preds.items()}
    cat_t = {k: torch.cat(v) for k, v in targs.items()}
    metrics = evaluate_all(cat_p, cat_t, n_classes=12)

    run_id = torch.cat(run_ids)
    win_idx = torch.cat(win_idxs)
    rul = cat_p["rul"]
    mono_terms = []
    for r in torch.unique(run_id):
        mask = run_id == r
        if mask.sum() < 2:
            continue
        order = torch.argsort(win_idx[mask])
        mono_terms.append(monotonicity_violation_rate(rul[mask][order]).item())
    metrics["mono_violation_rate"] = float(np.mean(mono_terms)) if mono_terms else 0.0
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared-test",
                        default=str(RESULTS_DIR / "test_index" / "test_windows.npz"))
    parser.add_argument("--paris-labels",
                        default=str(RESULTS_DIR / "labels" / "labels_paris.parquet"))
    parser.add_argument("--out-csv", default=str(RESULTS_DIR / "comparison_v2.csv"))
    parser.add_argument("--mc-passes", type=int, default=1)
    args = parser.parse_args()

    cfg = Config()
    cfg.seed_everything()
    device = cfg.get_device()

    paris = args.paris_labels if Path(args.paris_labels).exists() else None
    shared = args.shared_test if Path(args.shared_test).exists() else None

    test_ds = PitchBearingDataset(cfg, "test",
                                  labels_paris_path=paris,
                                  shared_test_path=shared,
                                  verbose=True)
    loader = torch.utils.data.DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )

    print("\n[compare_v2] evaluating Hybrid ...")
    hybrid = load_hybrid(device=device)
    t0 = time.time()
    h_metrics = _evaluate_model(hybrid, loader, device, mc_passes=args.mc_passes)
    h_metrics["wall_seconds"] = time.time() - t0

    print("[compare_v2] evaluating PINN ...")
    pinn = load_pinn(device=device)
    t0 = time.time()
    p_metrics = _evaluate_model(pinn, loader, device, mc_passes=args.mc_passes)
    p_metrics["wall_seconds"] = time.time() - t0

    print("[compare_v2] evaluating Ensemble (0.6 Hybrid + 0.4 PINN) ...")
    t0 = time.time()
    e_metrics = _ensemble_evaluate(hybrid, pinn, loader, device, 0.6, 0.4)
    e_metrics["wall_seconds"] = time.time() - t0

    # Compose CSV
    rows = ["metric,Hybrid,PINN,Ensemble"]
    keys = sorted(set(h_metrics) | set(p_metrics) | set(e_metrics))
    for k in keys:
        rows.append(f"{k},{h_metrics.get(k, ''):.6f},{p_metrics.get(k, ''):.6f},{e_metrics.get(k, ''):.6f}"
                    if isinstance(h_metrics.get(k), float) else
                    f"{k},{h_metrics.get(k, '')},{p_metrics.get(k, '')},{e_metrics.get(k, '')}")
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w") as f:
        f.write("\n".join(rows))
    print(f"\n[compare_v2] {args.out_csv}")
    for r in rows:
        print(" ", r)

    # Targets check
    targets = {"rul_rmse": 0.08, "fault_f1_macro": 0.85, "mono_violation_rate": 0.01}
    print("\n[compare_v2] target check (Ensemble):")
    for k, t in targets.items():
        v = e_metrics.get(k)
        if v is None:
            continue
        op = "≤" if "rmse" in k or "violation" in k else "≥"
        ok = (v <= t) if op == "≤" else (v >= t)
        flag = "OK" if ok else "MISS"
        print(f"   {flag}  {k}={v:.4f} {op} {t}")


if __name__ == "__main__":
    main()
