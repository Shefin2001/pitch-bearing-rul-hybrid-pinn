"""calibrate.py -- Stage 12: conformal calibration of the damage head.

The interval-censored NLL trains sigma to be *useful*, not *calibrated*.
Here we compute, on the VAL split (never test), the multiplicative sigma
scale s such that the class-anchor ln(a) falls inside the model's 90%
interval for 90% of windows (split-conformal quantile).

Outputs results/v3/calibration.json:
    sigma_scale, coverage_before/after, sigma_ln_c, per-class damage summary
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from common.config import Config  # noqa: E402
from common.rul_labels import INDEX_FAULT  # noqa: E402
from Hybrid_PINN_ParisRUL.v3.damage_net import DamageNet  # noqa: E402
from Hybrid_PINN_ParisRUL.v3.dataset_v3 import make_v3_loaders  # noqa: E402
from Hybrid_PINN_ParisRUL.v3.paris_engine import SIGMA_LN_C_DEFAULT  # noqa: E402
from Hybrid_PINN_ParisRUL.v3.severity_axis import LOG_A_MID  # noqa: E402
from Hybrid_PINN_ParisRUL.v3.train import RESULTS_DIR, V3_DIR  # noqa: E402

_Z90 = 1.6449  # two-sided 90% normal quantile


@torch.no_grad()
def collect_val_predictions(model, loader, device):
    mus, sigmas, faults = [], [], []
    for batch in loader:
        x = batch["x"].to(device)
        feat = batch["feat"].to(device)
        phys = batch["phys"].to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            pred = model(x, feat, phys)
        mus.append(pred["log_a_mu"].float().cpu())
        sigmas.append(torch.exp(0.5 * pred["log_a_log_var"].float()).cpu())
        faults.append(batch["fault_idx"])
    return (torch.cat(mus).numpy(), torch.cat(sigmas).numpy(),
            torch.cat(faults).numpy())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coverage", type=float, default=0.90)
    parser.add_argument("--physfeat", type=str,
                        default=str(V3_DIR / "physfeat" / "physfeat.parquet"))
    parser.add_argument("--workers", type=int, default=-1)
    args = parser.parse_args()

    out_path = V3_DIR / "calibration.json"
    if out_path.exists():
        print(f"[v3:calibrate] SKIP -- {out_path} exists (rm to redo)")
        return

    ckpt_path = RESULTS_DIR / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"{ckpt_path} -- run stage 11 first")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    cfg = Config()
    cfg.seed_everything()
    cfg.num_workers = min(cfg.num_workers, os.cpu_count() or 1)
    if args.workers >= 0:
        cfg.num_workers = args.workers
    device = cfg.get_device()

    paris = ROOT / "Hybrid_PINN_ParisRUL" / "results" / "labels" / "labels_paris.parquet"
    _, val_loader, _, _ = make_v3_loaders(
        cfg, physfeat_path=args.physfeat,
        labels_paris_path=str(paris) if paris.exists() else None,
        feat_cache_dir=V3_DIR.parent / "dataset_cache", verbose=False)

    model = DamageNet(n_classes=cfg.n_classes, hidden=ck.get("hidden", 128),
                      dropout=ck.get("dropout", 0.15)).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    mu, sigma, fault = collect_val_predictions(model, val_loader, device)
    anchor = LOG_A_MID[fault]

    # Split-conformal scale: r = |anchor - mu| / (z90 * sigma); s = q-th quantile
    r = np.abs(anchor - mu) / (_Z90 * np.maximum(sigma, 1e-6))
    n = r.size
    q = min(1.0, np.ceil((n + 1) * args.coverage) / n)
    s = float(np.quantile(r, q))
    cov_before = float((r <= 1.0).mean())
    cov_after = float((r <= s).mean())

    per_class = {}
    for c in np.unique(fault):
        m = fault == c
        per_class[INDEX_FAULT[int(c)]] = {
            "n": int(m.sum()),
            "log_a_mu_mean": float(mu[m].mean()),
            "a_mm_median": float(np.exp(np.median(mu[m])) * 1e3),
            "sigma_mean": float(sigma[m].mean()),
            "anchor_mm": float(np.exp(anchor[m][0]) * 1e3),
        }

    out = {"sigma_scale": s, "coverage_target": args.coverage,
           "coverage_before": cov_before, "coverage_after": cov_after,
           "sigma_ln_c": SIGMA_LN_C_DEFAULT, "n_val_windows": int(n),
           "per_class": per_class}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[v3:calibrate] sigma_scale={s:.3f} coverage {cov_before:.2%} -> "
          f"{cov_after:.2%} (target {args.coverage:.0%}) n={n}")
    print(f"[v3:calibrate] DONE -> {out_path}")


if __name__ == "__main__":
    main()
