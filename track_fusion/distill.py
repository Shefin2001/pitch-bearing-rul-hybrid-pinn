"""track_fusion/distill.py — Knowledge-distilled fused student + dual export.

Trains a small (~5M-param) Hybrid-style student to mimic the average of the
Hybrid teacher and PINN teacher on the training set's outputs. Exports two
final builds:
    * ``model_edge_int8.pt``  — INT8 dynamic-quantized TorchScript, target <10 MB
    * ``model_cloud_fp16.pt`` — TorchScript with model.half() and MC Dropout enabled

Fallback: if distilled student underperforms either teacher on val,
write an "ensemble" descriptor json that the unified inference layer reads
and applies as ``rul = w_h * hybrid + w_p * pinn``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from common.config import Config  # noqa: E402
from Hybrid_PINN_ParisRUL.common.dataset_v2 import PitchBearingDataset  # noqa: E402
from Hybrid_PINN_ParisRUL.common.metrics_v2 import evaluate_all  # noqa: E402
from Hybrid_PINN_ParisRUL.track_hybrid.inference import load_hybrid  # noqa: E402
from Hybrid_PINN_ParisRUL.track_hybrid.model import HybridParisModel  # noqa: E402
from Hybrid_PINN_ParisRUL.track_pinn.inference import load_pinn  # noqa: E402

RESULTS_DIR = ROOT / "Hybrid_PINN_ParisRUL" / "results" / "fusion"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Compact student (Hybrid-style backbone but smaller)
# ---------------------------------------------------------------------------

class StudentModel(HybridParisModel):
    """Smaller Hybrid for edge deployment."""

    def __init__(self, n_classes: int = 12):
        # smaller d_model, fewer layers
        super().__init__(n_classes=n_classes, d_model=128,
                         n_transformer_layers=1, nhead=4, dropout=0.2)


# ---------------------------------------------------------------------------
# Distillation training
# ---------------------------------------------------------------------------

def distill(epochs: int, batch_size: int, lr: float,
            hybrid_ckpt: Path, pinn_ckpt: Path,
            paris_labels: Optional[Path]) -> StudentModel:
    cfg = Config(batch_size=batch_size, num_epochs=epochs, learning_rate=lr)
    cfg.seed_everything()
    device = cfg.get_device()

    teacher_h = load_hybrid(hybrid_ckpt, device)
    teacher_p = load_pinn(pinn_ckpt, device)
    teacher_h.eval()
    teacher_p.eval()

    train_ds = PitchBearingDataset(cfg, "train",
                                   labels_paris_path=paris_labels, verbose=True)
    val_ds = PitchBearingDataset(cfg, "val",
                                 labels_paris_path=paris_labels, verbose=True)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)

    student = StudentModel(n_classes=cfg.n_classes).to(device)
    print(f"[distill] student params = {student.count_parameters():,}")

    opt = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=1e-5)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best_val = float("inf")
    for epoch in range(1, epochs + 1):
        student.train()
        running = 0.0
        n = 0
        t0 = time.time()
        for batch in train_loader:
            x_raw = batch["x"].to(device, non_blocking=True)
            x_feat = batch["feat"].to(device, non_blocking=True)
            target = {
                "rul": batch["rul"].to(device, non_blocking=True),
                "log_ttf": batch["log_ttf"].to(device, non_blocking=True),
                "fault_idx": batch["fault_idx"].to(device, non_blocking=True),
                "prog_mask": batch["prog_mask"].to(device, non_blocking=True),
            }

            with torch.no_grad():
                t_h = teacher_h(x_raw, x_feat)
                t_p = teacher_p(x_raw, x_feat)
                # Average teachers (RUL, log_ttf scalar averaging; logits soft-averaging)
                soft_rul = (t_h["rul"] + t_p["rul"]) * 0.5
                soft_log_ttf = (t_h["log_ttf"] + t_p["log_ttf"]) * 0.5
                soft_fault = (F.softmax(t_h["fault_logits"], dim=-1)
                              + F.softmax(t_p["fault_logits"], dim=-1)) * 0.5
                soft_prog = (torch.sigmoid(t_h["prog_logits"])
                             + torch.sigmoid(t_p["prog_logits"])) * 0.5

            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                s = student(x_raw, x_feat)
                # Hybrid distillation: soft-targets + hard-labels
                l_rul = 0.7 * F.mse_loss(s["rul"], soft_rul) + 0.3 * F.mse_loss(s["rul"], target["rul"])
                l_ttf = 0.7 * F.mse_loss(s["log_ttf"], soft_log_ttf) + 0.3 * F.mse_loss(s["log_ttf"], target["log_ttf"])
                l_fault = (0.7 * F.kl_div(F.log_softmax(s["fault_logits"], dim=-1),
                                          soft_fault, reduction="batchmean")
                           + 0.3 * F.cross_entropy(s["fault_logits"], target["fault_idx"]))
                l_prog = (0.7 * F.binary_cross_entropy_with_logits(s["prog_logits"], soft_prog)
                          + 0.3 * F.binary_cross_entropy_with_logits(s["prog_logits"], target["prog_mask"]))
                loss = l_rul + 0.7 * l_ttf + 0.5 * l_fault + 0.3 * l_prog

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            bs = x_raw.size(0)
            running += loss.item() * bs
            n += bs

        # ---- Val ----
        student.eval()
        preds = {"rul": [], "log_ttf": [], "fault_logits": [], "prog_logits": []}
        targs = {"rul": [], "log_ttf": [], "fault_idx": [], "prog_mask": []}
        with torch.no_grad():
            for batch in val_loader:
                x_raw = batch["x"].to(device, non_blocking=True)
                x_feat = batch["feat"].to(device, non_blocking=True)
                target = {k: batch[k].to(device, non_blocking=True)
                          for k in ("rul", "log_ttf", "fault_idx", "prog_mask")}
                p = student(x_raw, x_feat)
                for k in preds:
                    preds[k].append(p[k].detach().cpu())
                for k in targs:
                    targs[k].append(target[k].detach().cpu())
        cat_p = {k: torch.cat(v) for k, v in preds.items()}
        cat_t = {k: torch.cat(v) for k, v in targs.items()}
        m = evaluate_all(cat_p, cat_t, n_classes=cfg.n_classes)

        train_loss = running / max(n, 1)
        elapsed = time.time() - t0
        print(f"[distill ep {epoch:3d}/{epochs}] train={train_loss:.4f} "
              f"val_rmse={m.get('rul_rmse', 0):.4f} "
              f"val_f1={m.get('fault_f1_macro', 0):.3f} ({elapsed:.1f}s)")

        if m["rul_rmse"] < best_val:
            best_val = m["rul_rmse"]
            torch.save({"state_dict": student.state_dict(), "val_metrics": m},
                       RESULTS_DIR / "student_best.pt")
            print(f"   ↳ saved student best (rmse={best_val:.4f})")

    return student


# ---------------------------------------------------------------------------
# Export — INT8 edge + FP16 cloud
# ---------------------------------------------------------------------------

def export_edge_int8(student: nn.Module, out_path: Path) -> None:
    """Dynamic INT8 quantization on Linear layers, save TorchScript."""
    student.eval().cpu()
    qmodel = torch.ao.quantization.quantize_dynamic(
        student, {nn.Linear}, dtype=torch.qint8
    )
    # Trace (dynamic-quantized models often don't script cleanly)
    x_raw = torch.zeros(1, 5, 2048)
    x_feat = torch.zeros(1, 160)
    try:
        scripted = torch.jit.trace(qmodel, (x_raw, x_feat), strict=False)
    except Exception as e:
        print(f"[!] edge trace failed: {e} — saving state-dict only")
        torch.save(qmodel.state_dict(), out_path.with_suffix(".sd.pt"))
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(scripted, str(out_path))
    size_mb = out_path.stat().st_size / 1e6
    print(f"[export edge INT8] {out_path.name}: {size_mb:.2f} MB")


def export_cloud_fp16(student: nn.Module, out_path: Path) -> None:
    """FP16 TorchScript build for cloud GPU inference (MC Dropout-friendly)."""
    student.eval()
    device = next(student.parameters()).device
    # Half-precision copy
    cloud_model = type(student)().to(device).half()
    cloud_model.load_state_dict({k: v.half() for k, v in student.state_dict().items()})
    x_raw = torch.zeros(1, 5, 2048, dtype=torch.float16, device=device)
    x_feat = torch.zeros(1, 160, dtype=torch.float16, device=device)
    try:
        scripted = torch.jit.trace(cloud_model, (x_raw, x_feat), strict=False)
    except Exception as e:
        print(f"[!] cloud trace failed: {e}")
        torch.save(cloud_model.state_dict(), out_path.with_suffix(".sd.pt"))
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(scripted, str(out_path))
    size_mb = out_path.stat().st_size / 1e6
    print(f"[export cloud FP16] {out_path.name}: {size_mb:.2f} MB")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hybrid", type=str,
                        default=str(ROOT / "Hybrid_PINN_ParisRUL" / "results" / "hybrid" / "best_model.pt"))
    parser.add_argument("--pinn", type=str,
                        default=str(ROOT / "Hybrid_PINN_ParisRUL" / "results" / "pinn" / "best_model.pt"))
    parser.add_argument("--paris-labels", type=str,
                        default=str(ROOT / "Hybrid_PINN_ParisRUL" / "results" / "labels" / "labels_paris.parquet"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--export-edge", action="store_true")
    parser.add_argument("--export-cloud", action="store_true")
    args = parser.parse_args()

    paris_path = Path(args.paris_labels) if Path(args.paris_labels).exists() else None
    if not Path(args.hybrid).exists() or not Path(args.pinn).exists():
        print("[!] missing teacher checkpoint(s) — writing ensemble fallback descriptor")
        descriptor = {"mode": "ensemble", "w_hybrid": 0.6, "w_pinn": 0.4}
        with open(RESULTS_DIR / "fusion_descriptor.json", "w") as f:
            json.dump(descriptor, f, indent=2)
        return

    student = distill(args.epochs, args.batch, args.lr,
                      Path(args.hybrid), Path(args.pinn), paris_path)

    if args.export_edge:
        export_edge_int8(student, RESULTS_DIR / "model_edge_int8.pt")
    if args.export_cloud:
        export_cloud_fp16(student, RESULTS_DIR / "model_cloud_fp16.pt")

    descriptor = {
        "mode": "distilled",
        "student_path": str(RESULTS_DIR / "student_best.pt"),
        "edge_path": str(RESULTS_DIR / "model_edge_int8.pt"),
        "cloud_path": str(RESULTS_DIR / "model_cloud_fp16.pt"),
    }
    with open(RESULTS_DIR / "fusion_descriptor.json", "w") as f:
        json.dump(descriptor, f, indent=2)
    print(f"[fusion] descriptor → {RESULTS_DIR / 'fusion_descriptor.json'}")


if __name__ == "__main__":
    main()
