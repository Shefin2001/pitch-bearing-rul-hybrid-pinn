"""track_hybrid/train.py — DDP/AMP training loop for the Hybrid track.

Run via torchrun:
    torchrun --standalone --nproc_per_node=auto track_hybrid/train.py [args]

Single-GPU / single-process (Lightning AI, containers):
    python -m Hybrid_PINN_ParisRUL.track_hybrid.train --epochs 2 --batch 16
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader

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
from Hybrid_PINN_ParisRUL.track_hybrid.loss import (  # noqa: E402
    HybridLossWeights,
    HybridMultiTaskLoss,
)
from Hybrid_PINN_ParisRUL.track_hybrid.model import HybridParisModel  # noqa: E402

RESULTS_DIR = ROOT / "Hybrid_PINN_ParisRUL" / "results" / "hybrid"
DATASET_CACHE_DIR = ROOT / "Hybrid_PINN_ParisRUL" / "results" / "dataset_cache"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATASET_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Training stability constants (sourced from LLM pipeline best practices)
# Refs: ZClip (arXiv 2504.02507), PyTorch Lightning EarlyStopping,
#       Better-ML loss-spike guide, DMaS-LLaMa (arXiv 2412.13335)
# ---------------------------------------------------------------------------
_EMA_ALPHA      = 0.98   # loss EMA smoothing — high value = slow to forget
_SPIKE_FACTOR   = 3.0    # skip optimizer step if batch_loss > factor × EMA
_GRAD_WARN      = 100.0  # warn (don't abort) if global grad norm exceeds this
_MIN_DELTA      = 1e-4   # min val_loss improvement to count as "better" (noise filter)
_DIVERGE_FACTOR = 3.0    # abort run if val_loss > factor × best_val (after epoch 10)
_CKPT_EVERY     = 10     # save periodic checkpoint every N epochs (keep last 2)
_LOG_STEPS      = 50     # print within-epoch progress every N optimizer steps
_WARMUP_FRAC    = 0.05   # fraction of total epochs for linear LR warmup


# ---------------------------------------------------------------------------
# Train loop
# ---------------------------------------------------------------------------

def train_one_epoch(
    model, loader, loss_fn, optimizer, scaler, device,
    epoch: int, grad_clip: float, accum_steps: int,
    loss_ema: Optional[float], log_steps: int, is_main: bool,
) -> Tuple[Dict[str, float], float]:
    """Run one training epoch with NaN guards, spike detection, and step logging.

    Returns:
        (stats_dict, updated_loss_ema)
    """
    model.train()
    running: Dict[str, float] = {}
    n = 0
    skipped = 0
    grad_norms: List[float] = []
    t_start = time.time()
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader):
        x_raw  = batch["x"].to(device, non_blocking=True)
        x_feat = batch["feat"].to(device, non_blocking=True)
        target = {k: batch[k].to(device, non_blocking=True)
                  for k in ("rul", "log_ttf", "fault_idx", "prog_mask",
                             "run_id", "win_idx")}

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            pred   = model(x_raw, x_feat)
            losses = loss_fn(pred, target)
            loss   = losses["total"] / accum_steps

        loss_val = float(losses["total"].item())

        # ── Guard 1: NaN / Inf → abort immediately ──────────────────────────
        if not torch.isfinite(losses["total"]):
            raise RuntimeError(
                f"[ABORT] Loss is {loss_val} at step {step+1} / epoch {epoch}. "
                "Check data pipeline, learning rate, and model numerics."
            )

        # ── Guard 2: Spike → skip optimizer step ────────────────────────────
        if loss_ema is not None and loss_val > _SPIKE_FACTOR * loss_ema:
            if is_main:
                print(f"  [SPIKE ep{epoch} s{step+1}] loss={loss_val:.4f} > "
                      f"{_SPIKE_FACTOR}×EMA={loss_ema:.4f} — skipping step")
            optimizer.zero_grad(set_to_none=True)
            skipped += 1
            continue

        # Update EMA with current (non-spike) loss
        loss_ema = (_EMA_ALPHA * loss_ema + (1 - _EMA_ALPHA) * loss_val
                    if loss_ema is not None else loss_val)

        scaler.scale(loss).backward()
        if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            gnorm = float(
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip).item()
            )
            grad_norms.append(gnorm)

            # ── Guard 3: Grad norm warning ───────────────────────────────────
            if is_main and gnorm > _GRAD_WARN:
                print(f"  [GRAD WARN ep{epoch} s{step+1}] "
                      f"grad_norm={gnorm:.1f} > {_GRAD_WARN} — clipped")

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        bs = x_raw.size(0)
        n  += bs
        for k, v in losses.items():
            running[k] = running.get(k, 0.0) + float(v.item()) * bs

        # ── Per-step telemetry ───────────────────────────────────────────────
        if is_main and (step + 1) % log_steps == 0:
            elapsed = time.time() - t_start
            sps     = n / max(elapsed, 1e-6)
            gnorm_s = f"{grad_norms[-1]:.3f}" if grad_norms else "—"
            ema_s   = f"{loss_ema:.4f}" if loss_ema is not None else "—"
            print(f"  step [{step+1:4d}/{len(loader)}]  "
                  f"loss={loss_val:.4f}  ema={ema_s}  "
                  f"gnorm={gnorm_s}  sps={sps:.0f}")

    stats = {k: v / max(n, 1) for k, v in running.items()}
    stats["grad_norm_mean"] = float(np.mean(grad_norms))  if grad_norms else 0.0
    stats["grad_norm_max"]  = float(np.max(grad_norms))   if grad_norms else 0.0
    stats["skipped_steps"]  = float(skipped)
    stats["throughput_sps"] = n / max(time.time() - t_start, 1e-6)
    return stats, loss_ema if loss_ema is not None else 0.0


# ---------------------------------------------------------------------------
# Validation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model, loader, loss_fn, device, n_classes: int = 12,
             mc_passes: int = 1) -> Dict[str, float]:
    if mc_passes > 1 and hasattr(model, "module"):
        model.module.enable_mc_dropout()
    elif mc_passes > 1:
        model.enable_mc_dropout()
    else:
        model.eval()

    preds   = {"rul": [], "log_ttf": [], "fault_logits": [], "prog_logits": []}
    targets = {"rul": [], "log_ttf": [], "fault_idx": [], "prog_mask": []}
    running_loss = 0.0
    n = 0

    for batch in loader:
        x_raw  = batch["x"].to(device, non_blocking=True)
        x_feat = batch["feat"].to(device, non_blocking=True)
        target = {k: batch[k].to(device, non_blocking=True)
                  for k in ("rul", "log_ttf", "fault_idx", "prog_mask",
                             "run_id", "win_idx")}
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            if mc_passes > 1:
                mc_outs: List[Dict[str, torch.Tensor]] = []
                for _ in range(mc_passes):
                    mc_outs.append(model(x_raw, x_feat))
                pred = {
                    "rul":          torch.stack([o["rul"]          for o in mc_outs]).mean(0),
                    "log_ttf":      torch.stack([o["log_ttf"]      for o in mc_outs]).mean(0),
                    "fault_logits": torch.stack([o["fault_logits"] for o in mc_outs]).mean(0),
                    "prog_logits":  torch.stack([o["prog_logits"]  for o in mc_outs]).mean(0),
                    "embedding":    mc_outs[0]["embedding"],
                }
            else:
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",       type=int,   default=100)
    parser.add_argument("--batch",        type=int,   default=4096)  # H200 141GB: fills VRAM, 4× fewer steps
    parser.add_argument("--lr",           type=float, default=8e-4)  # √4 × 4e-4 (4× batch → 2× LR)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip",    type=float, default=1.0)
    parser.add_argument("--accum-steps",  type=int,   default=1)
    parser.add_argument("--patience",     type=int,   default=20)
    parser.add_argument("--log-steps",    type=int,   default=_LOG_STEPS,
                        help="Print within-epoch progress every N optimizer steps")
    parser.add_argument("--paris-labels", type=str,
                        default=str(ROOT / "Hybrid_PINN_ParisRUL" / "results" / "labels" / "labels_paris.parquet"))
    parser.add_argument("--shared-test",  type=str,
                        default=str(ROOT / "Hybrid_PINN_ParisRUL" / "results" / "test_index" / "test_windows.npz"))
    parser.add_argument("--mc-passes",   type=int,   default=1)
    parser.add_argument("--amp",         action="store_true", default=True)
    parser.add_argument("--no-amp",      dest="amp", action="store_false")
    parser.add_argument("--resume",      action="store_true",
                        help="Resume from best_model.pt if it exists")
    args = parser.parse_args()

    try:
        rank, world_size, device = init_distributed()
        cfg = Config(batch_size=args.batch, num_epochs=args.epochs,
                     learning_rate=args.lr, weight_decay=args.weight_decay,
                     grad_clip_norm=args.grad_clip, accum_steps=args.accum_steps,
                     patience=args.patience)
        cfg.seed_everything()
        cfg.apply_cudnn_settings()

        # ── Data ─────────────────────────────────────────────────────────────
        paris_path = args.paris_labels if Path(args.paris_labels).exists() else None
        if is_main_process() and paris_path is None:
            print("[!] No paris-labels — using class-constant fallback (TTF will be coarse).")

        # Use pre-built shared test npz (fast load) instead of re-extracting features
        shared_test = args.shared_test if Path(args.shared_test).exists() else None
        if is_main_process() and shared_test is None:
            print("[!] shared-test npz not found — test set will re-extract features.")

        def _ddp_sampler(ds, shuffle):
            if world_size > 1:
                return torch.utils.data.distributed.DistributedSampler(
                    ds, num_replicas=world_size, rank=rank, shuffle=shuffle)
            return None

        train_loader, val_loader, _ = make_loaders(
            cfg, labels_paris_path=paris_path, shared_test_path=shared_test,
            ddp_sampler_fn=_ddp_sampler, verbose=is_main_process(),
            feat_cache_dir=DATASET_CACHE_DIR,
        )

        # ── Model ─────────────────────────────────────────────────────────────
        model = HybridParisModel(n_classes=cfg.n_classes, dropout=0.2).to(device)
        n_params = model.count_parameters()

        if cfg.compile_model:
            try:
                # Raise dynamo's recompilation cache so max-autotune doesn't
                # retrigger on minor input shape variations (larger batch = fewer shapes).
                import torch._dynamo as _dynamo
                _dynamo.config.cache_size_limit = 64
                _dynamo.config.optimize_ddp = False  # avoid recompile on DDP buckets
            except (ImportError, AttributeError):
                pass
            try:
                model = torch.compile(model, mode="max-autotune")
            except Exception as e:
                if is_main_process():
                    print(f"[hybrid:train] torch.compile unavailable: {e}")

        model = wrap_model_ddp(model, device)

        # ── Loss / optim / scalers ────────────────────────────────────────────
        loss_fn   = HybridMultiTaskLoss(HybridLossWeights())
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.effective_lr(),
                                      weight_decay=cfg.weight_decay)
        scaler    = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

        warmup_epochs = max(1, int(cfg.num_epochs * _WARMUP_FRAC))
        warmup_sched  = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
        plateau_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-7)

        # ── Resume ───────────────────────────────────────────────────────────
        start_epoch  = 1
        best_val     = float("inf")
        patience_left = cfg.patience
        loss_ema: Optional[float] = None
        ckpt_path    = RESULTS_DIR / "best_model.pt"

        if args.resume and ckpt_path.exists():
            ckpt      = torch.load(ckpt_path, map_location=device)
            raw_model = model.module if hasattr(model, "module") else model
            raw_model = getattr(raw_model, "_orig_mod", raw_model)
            raw_model.load_state_dict(ckpt["state_dict"])
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            if "scheduler" in ckpt:
                plateau_sched.load_state_dict(ckpt["scheduler"])
            best_val    = ckpt.get("best_val", float("inf"))
            start_epoch = ckpt["epoch"] + 1
            loss_ema    = ckpt.get("loss_ema")
            if is_main_process():
                print(f"[hybrid:train] RESUMED from epoch {ckpt['epoch']} "
                      f"(best_val={best_val:.4f}) → continuing from epoch "
                      f"{start_epoch}/{cfg.num_epochs}")

        # ── Startup map ───────────────────────────────────────────────────────
        if is_main_process():
            n_train = len(train_loader.dataset)
            n_val   = len(val_loader.dataset)
            steps_per_epoch = len(train_loader)
            print()
            print("╔══════════════════════════════════════════════════════════════╗")
            print("║  HYBRID TRAINING PLAN                                        ║")
            print("╠══════════════════════════════════════════════════════════════╣")
            print(f"║  Device     : {str(device):<47} ║")
            print(f"║  Train data : {n_train:,} windows ({steps_per_epoch} steps/epoch){'':>10} ║"[:67] + "║")
            print(f"║  Val data   : {n_val:,} windows{'':>30} ║"[:67] + "║")
            print(f"║  Model      : HybridParisModel — {n_params:,} params{'':>10} ║"[:67] + "║")
            print(f"║  Epochs     : {start_epoch}→{cfg.num_epochs}  Batch={cfg.batch_size}  "
                  f"LR={cfg.effective_lr():.1e}  WD={cfg.weight_decay:.0e}{'':>5} ║"[:67] + "║")
            print(f"║  Warmup     : {warmup_epochs} epochs (linear 0.1→1.0 LR scale){'':>12} ║"[:67] + "║")
            print(f"║  Grad clip  : {cfg.grad_clip_norm}  Accum={cfg.accum_steps} steps{'':>20} ║"[:67] + "║")
            print(f"║  Checkpoints: best_model.pt + every {_CKPT_EVERY} epochs (keep last 2){'':>5} ║"[:67] + "║")
            print(f"║  Early stop : patience={cfg.patience} min_delta={_MIN_DELTA}{'':>25} ║"[:67] + "║")
            print(f"║  Guards     : NaN→abort | spike>{_SPIKE_FACTOR}×EMA→skip | "
                  f"diverge>{_DIVERGE_FACTOR}×best→stop ║")
            print(f"║  Step log   : every {args.log_steps} steps{'':>40} ║"[:67] + "║")
            print("╚══════════════════════════════════════════════════════════════╝")
            print()

        # ── TensorBoard ───────────────────────────────────────────────────────
        writer = None
        if is_main_process():
            try:
                from torch.utils.tensorboard import SummaryWriter
                writer = SummaryWriter(str(RESULTS_DIR / "tensorboard"))
            except Exception:
                writer = None

        # ── Training loop ─────────────────────────────────────────────────────
        history: List[Dict] = []
        t_run = time.time()           # wall-clock start for ETA

        for epoch in range(start_epoch, cfg.num_epochs + 1):
            if hasattr(train_loader, "sampler") and \
               hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)

            t0 = time.time()
            train_stats, loss_ema = train_one_epoch(
                model, train_loader, loss_fn, optimizer, scaler, device,
                epoch=epoch, grad_clip=cfg.grad_clip_norm,
                accum_steps=cfg.accum_steps, loss_ema=loss_ema,
                log_steps=args.log_steps, is_main=is_main_process(),
            )
            val_stats = validate(model, val_loader, loss_fn, device,
                                 n_classes=cfg.n_classes, mc_passes=args.mc_passes)

            # ── LR schedule ─────────────────────────────────────────────────
            if epoch <= warmup_epochs:
                warmup_sched.step()
            else:
                plateau_sched.step(val_stats["val_loss"])
            current_lr = optimizer.param_groups[0]["lr"]

            if is_main_process():
                elapsed     = time.time() - t0
                epochs_done = epoch - start_epoch + 1
                epochs_left = cfg.num_epochs - epoch
                secs_per_ep = (time.time() - t_run) / max(epochs_done, 1)
                eta_h       = (secs_per_ep * epochs_left) / 3600

                phase = "warmup" if epoch <= warmup_epochs else "train"
                print(
                    f"[ep {epoch:3d}/{cfg.num_epochs}|{phase}] "
                    f"train={train_stats.get('total', 0):.4f}  "
                    f"val={val_stats['val_loss']:.4f}  "
                    f"rmse={val_stats.get('rul_rmse', 0):.4f}  "
                    f"f1={val_stats.get('fault_f1_macro', 0):.3f}  "
                    f"ttf_mape={val_stats.get('ttf_mape', 0):.3f}  "
                    f"gnorm={train_stats.get('grad_norm_mean', 0):.3f}(max={train_stats.get('grad_norm_max', 0):.1f})  "
                    f"lr={current_lr:.1e}  "
                    f"skip={int(train_stats.get('skipped_steps', 0))}  "
                    f"sps={train_stats.get('throughput_sps', 0):.0f}  "
                    f"ETA={eta_h:.1f}h  ({elapsed:.1f}s)"
                )

                if writer is not None:
                    for k, v in {**train_stats, **val_stats,
                                 "lr": current_lr, "loss_ema": loss_ema}.items():
                        writer.add_scalar(k, float(v), epoch)

                row = {"epoch": epoch, "lr": current_lr, "loss_ema": loss_ema,
                       **train_stats, **val_stats}
                history.append(row)

                # ── Guard 4: Divergence brake (after warmup) ─────────────────
                if epoch > warmup_epochs + 10 and best_val < float("inf"):
                    if val_stats["val_loss"] > _DIVERGE_FACTOR * best_val:
                        print(f"   ↳ [DIVERGE] val_loss={val_stats['val_loss']:.4f} > "
                              f"{_DIVERGE_FACTOR}×best ({best_val:.4f}) — stopping (model diverged)")
                        break

                # ── Checkpoint: best model (with min_delta noise filter) ──────
                improvement = best_val - val_stats["val_loss"]
                if improvement > _MIN_DELTA:
                    best_val      = val_stats["val_loss"]
                    patience_left = cfg.patience
                    state = (model.module.state_dict()
                             if hasattr(model, "module") else model.state_dict())
                    torch.save({
                        "epoch":      epoch,
                        "state_dict": state,
                        "optimizer":  optimizer.state_dict(),
                        "scheduler":  plateau_sched.state_dict(),
                        "best_val":   best_val,
                        "loss_ema":   loss_ema,
                        "config":     cfg.__dict__,
                        "val_metrics": val_stats,
                    }, ckpt_path)
                    print(f"   ↳ [BEST CKPT] epoch={epoch} "
                          f"val_loss={best_val:.4f} (Δ={improvement:.4f}) → {ckpt_path}")
                else:
                    patience_left -= 1
                    print(f"   ↳ no improvement ({improvement:+.4f}) — "
                          f"patience {patience_left}/{cfg.patience}")
                    if patience_left <= 0:
                        print(f"   ↳ [EARLY STOP] epoch={epoch} best val={best_val:.4f}")
                        break

                # ── Checkpoint: periodic (every N epochs, keep last 2) ────────
                if epoch % _CKPT_EVERY == 0:
                    periodic = RESULTS_DIR / f"checkpoint_ep{epoch:04d}.pt"
                    state    = (model.module.state_dict()
                                if hasattr(model, "module") else model.state_dict())
                    torch.save({
                        "epoch": epoch, "state_dict": state,
                        "optimizer": optimizer.state_dict(),
                        "scheduler": plateau_sched.state_dict(),
                        "best_val": best_val, "loss_ema": loss_ema,
                        "config": cfg.__dict__, "val_metrics": val_stats,
                    }, periodic)
                    old = RESULTS_DIR / f"checkpoint_ep{epoch - 2 * _CKPT_EVERY:04d}.pt"
                    if old.exists():
                        old.unlink()
                    print(f"   ↳ [PERIODIC CKPT] → {periodic.name}")

            barrier()

        # ── Wrap-up ───────────────────────────────────────────────────────────
        if is_main_process() and history:
            with open(RESULTS_DIR / "history.json", "w") as f:
                json.dump(history, f, indent=2, default=str)
            if writer is not None:
                writer.close()
            total_h = (time.time() - t_run) / 3600
            print()
            print(f"[hybrid:train] DONE — best val_loss={best_val:.4f} "
                  f"in {len(history)} epochs ({total_h:.2f}h)")
            (RESULTS_DIR / ".training_complete").touch()
            print(f"[hybrid:train] Sentinel → {RESULTS_DIR / '.training_complete'}")
    finally:
        cleanup()


if __name__ == "__main__":
    main()
