"""track_pinn/train.py — DDP/AMP training loop for the PINN track.

Run via torchrun:
    torchrun --standalone --nproc_per_node=1 track_pinn/train.py [args]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from common.config import Config  # noqa: E402
from common.distributed import (  # noqa: E402
    barrier,
    cleanup,
    init_distributed,
    is_main_process,
    wrap_model_ddp,
)
from Hybrid_PINN_ParisRUL.common.dataset_v2 import make_loaders  # noqa: E402
from Hybrid_PINN_ParisRUL.common.metrics_v2 import evaluate_all  # noqa: E402
from Hybrid_PINN_ParisRUL.track_pinn.loss import PINNLoss, PINNLossWeights  # noqa: E402
from Hybrid_PINN_ParisRUL.track_pinn.model import PINNModel  # noqa: E402

RESULTS_DIR = ROOT / "Hybrid_PINN_ParisRUL" / "results" / "pinn"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def train_one_epoch(model, loader, loss_fn, optimizer, scaler, device,
                    grad_clip: float, accum_steps: int) -> Dict[str, float]:
    model.train()
    running: Dict[str, float] = {}
    n = 0
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader):
        x_raw = batch["x"].to(device, non_blocking=True)
        x_feat = batch["feat"].to(device, non_blocking=True)
        target = {k: batch[k].to(device, non_blocking=True)
                  for k in ("rul", "log_ttf", "fault_idx", "prog_mask",
                            "run_id", "win_idx")}
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            pred = model(x_raw, x_feat)
            losses = loss_fn(pred, target)
            loss = losses["total"] / accum_steps
        scaler.scale(loss).backward()
        if (step + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        bs = x_raw.size(0)
        n += bs
        for k, v in losses.items():
            running[k] = running.get(k, 0.0) + float(v.item()) * bs
    return {k: v / max(n, 1) for k, v in running.items()}


@torch.no_grad()
def validate(model, loader, loss_fn, device, n_classes: int = 12) -> Dict[str, float]:
    model.eval()
    preds = {"rul": [], "log_ttf": [], "fault_logits": [], "prog_logits": []}
    targets = {"rul": [], "log_ttf": [], "fault_idx": [], "prog_mask": []}
    running_loss = 0.0
    n = 0
    for batch in loader:
        x_raw = batch["x"].to(device, non_blocking=True)
        x_feat = batch["feat"].to(device, non_blocking=True)
        target = {k: batch[k].to(device, non_blocking=True)
                  for k in ("rul", "log_ttf", "fault_idx", "prog_mask",
                            "run_id", "win_idx")}
        pred = model(x_raw, x_feat)
        losses = loss_fn(pred, target)
        bs = x_raw.size(0)
        running_loss += float(losses["total"].item()) * bs
        n += bs
        for k in preds:
            preds[k].append(pred[k].detach().cpu())
        for k in targets:
            targets[k].append(target[k].detach().cpu())
    cat_pred = {k: torch.cat(v) for k, v in preds.items()}
    cat_targ = {k: torch.cat(v) for k, v in targets.items()}
    metrics = evaluate_all(cat_pred, cat_targ, n_classes=n_classes)
    metrics["val_loss"] = running_loss / max(n, 1)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--accum-steps", type=int, default=1)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--paris-labels", type=str,
                        default=str(ROOT / "Hybrid_PINN_ParisRUL" / "results" / "labels" / "labels_paris.parquet"))
    args = parser.parse_args()

    rank, world_size, device = init_distributed()
    cfg = Config(batch_size=args.batch, num_epochs=args.epochs,
                 learning_rate=args.lr, weight_decay=args.weight_decay,
                 grad_clip_norm=args.grad_clip, accum_steps=args.accum_steps,
                 patience=args.patience)
    cfg.seed_everything()
    cfg.apply_cudnn_settings()

    if is_main_process():
        print(f"[pinn:train] rank={rank} world_size={world_size} device={device}")

    paris_path = args.paris_labels if Path(args.paris_labels).exists() else None
    if rank == 0 and paris_path is None:
        print("[!] No paris-labels found — using class-constant fallback.")

    def _ddp_sampler(ds, shuffle):
        if world_size > 1:
            return torch.utils.data.distributed.DistributedSampler(
                ds, num_replicas=world_size, rank=rank, shuffle=shuffle
            )
        return None

    train_loader, val_loader, _ = make_loaders(
        cfg, labels_paris_path=paris_path,
        ddp_sampler_fn=_ddp_sampler, verbose=is_main_process(),
    )

    model = PINNModel(n_classes=cfg.n_classes, dropout=0.2).to(device)
    if is_main_process():
        print(f"[pinn:train] params = {model.count_parameters():,}")
    if cfg.compile_model:
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except Exception:
            pass
    model = wrap_model_ddp(model, device)

    loss_fn = PINNLoss(PINNLossWeights())
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.effective_lr(),
                                  weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    writer = None
    if is_main_process():
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(str(RESULTS_DIR / "tensorboard"))
        except Exception:
            writer = None

    best_val = float("inf")
    patience_left = cfg.patience
    history: List[Dict] = []
    for epoch in range(1, cfg.num_epochs + 1):
        if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        t0 = time.time()
        train_stats = train_one_epoch(
            model, train_loader, loss_fn, optimizer, scaler, device,
            grad_clip=cfg.grad_clip_norm, accum_steps=cfg.accum_steps,
        )
        val_stats = validate(model, val_loader, loss_fn, device, n_classes=cfg.n_classes)
        scheduler.step(val_stats["val_loss"])

        if is_main_process():
            elapsed = time.time() - t0
            # Pull effective C, m from the model
            inner = model.module if hasattr(model, "module") else model
            C_now = float(inner.C_paris().item())
            m_now = float(inner.m_paris().item())
            print(f"[ep {epoch:3d}/{cfg.num_epochs}] "
                  f"train_loss={train_stats.get('total', 0):.4f} "
                  f"val_loss={val_stats['val_loss']:.4f} "
                  f"rmse={val_stats.get('rul_rmse', 0):.4f} "
                  f"f1={val_stats.get('fault_f1_macro', 0):.3f} "
                  f"C={C_now:.2e} m={m_now:.2f} "
                  f"({elapsed:.1f}s)")
            if writer is not None:
                for k, v in {**train_stats, **val_stats,
                              "C_paris": C_now, "m_paris": m_now}.items():
                    writer.add_scalar(k, v, epoch)
            history.append({"epoch": epoch, **train_stats, **val_stats,
                            "C_paris": C_now, "m_paris": m_now})

            if val_stats["val_loss"] < best_val:
                best_val = val_stats["val_loss"]
                patience_left = cfg.patience
                state = inner.state_dict()
                torch.save({
                    "epoch": epoch,
                    "state_dict": state,
                    "config": cfg.__dict__,
                    "val_metrics": val_stats,
                    "C_paris": C_now,
                    "m_paris": m_now,
                }, RESULTS_DIR / "best_model.pt")
                print(f"   ↳ saved best model")
            else:
                patience_left -= 1
                if patience_left <= 0:
                    print(f"   ↳ early stop at epoch {epoch}")
                    break
        barrier()

    if is_main_process() and history:
        import json
        with open(RESULTS_DIR / "history.json", "w") as f:
            json.dump(history, f, indent=2, default=str)
        print(f"[pinn:train] DONE. best val_loss = {best_val:.4f}")
        if writer is not None:
            writer.close()
    cleanup()


if __name__ == "__main__":
    main()
