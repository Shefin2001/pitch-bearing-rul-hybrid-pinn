"""train.py -- v3 DamageNet trainer (single-GPU, checkpointed, resumable).

    python -m Hybrid_PINN_ParisRUL.v3.train --epochs 40 --batch 8192

Reuses the v2 conventions: bf16 autocast (no GradScaler), torch.compile
reduce-overhead, tqdm epoch/batch bars, NaN-abort / spike-skip / divergence
guards, sqrt LR scaling for batch changes. Adds crash-safe resume via
checkpoint_last.pt written EVERY epoch (model+optim+sched+RNG).
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from common.config import Config  # noqa: E402
from Hybrid_PINN_ParisRUL.v3.damage_net import DamageNet  # noqa: E402
from Hybrid_PINN_ParisRUL.v3.dataset_v3 import make_v3_loaders  # noqa: E402
from Hybrid_PINN_ParisRUL.v3.losses import (  # noqa: E402
    DamageLoss,
    corn_predict_level,
)

V3_DIR = Path(os.environ.get("V3_RESULTS_DIR",
                             str(ROOT / "Hybrid_PINN_ParisRUL" / "results" / "v3")))
RESULTS_DIR = V3_DIR / "damage_net"

_EMA_ALPHA = 0.98
_SPIKE_FACTOR = 3.0
_MIN_DELTA = 1e-4
_DIVERGE_FACTOR = 3.0
_WARMUP_FRAC = 0.05


def _tqdm_or_id():
    try:
        from tqdm import tqdm
        return tqdm
    except ImportError:
        return lambda it, **kw: it


def f1_macro(pred: np.ndarray, true: np.ndarray, n_classes: int) -> float:
    scores = []
    for c in range(n_classes):
        tp = int(((pred == c) & (true == c)).sum())
        fp = int(((pred == c) & (true != c)).sum())
        fn = int(((pred != c) & (true == c)).sum())
        if tp + fp + fn == 0:
            continue
        scores.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(scores)) if scores else 0.0


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    try:
        from scipy.stats import spearmanr
        r = spearmanr(a, b).statistic
        return float(r) if np.isfinite(r) else 0.0
    except Exception:
        ra = np.argsort(np.argsort(a)).astype(np.float64)
        rb = np.argsort(np.argsort(b)).astype(np.float64)
        c = np.corrcoef(ra, rb)[0, 1]
        return float(c) if np.isfinite(c) else 0.0


def train_one_epoch(model, loader, loss_fn, optimizer, device, epoch: int,
                    grad_clip: float, loss_ema: Optional[float]
                    ) -> Tuple[Dict[str, float], float]:
    tqdm = _tqdm_or_id()
    model.train()
    running: Dict[str, torch.Tensor | float] = {}
    n, skipped = 0, 0
    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)

    bar = tqdm(loader, desc=f"  ep{epoch:03d} train", unit="b", leave=False,
               ncols=110)
    for step, batch in enumerate(bar):
        x = batch["x"].to(device, non_blocking=True)
        feat = batch["feat"].to(device, non_blocking=True)
        phys = batch["phys"].to(device, non_blocking=True)
        target = {k: batch[k].to(device, non_blocking=True)
                  for k in ("fault_idx", "sev_level", "log_a_lo", "log_a_hi")}
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            pred = model(x, feat, phys)
            losses = loss_fn(pred, target)

        loss_val = float(losses["total"].item())
        if not np.isfinite(loss_val):
            raise RuntimeError(f"[ABORT] non-finite loss at ep{epoch} s{step+1}")
        if loss_ema is not None and loss_val > _SPIKE_FACTOR * loss_ema:
            optimizer.zero_grad(set_to_none=True)
            skipped += 1
            continue
        loss_ema = (_EMA_ALPHA * loss_ema + (1 - _EMA_ALPHA) * loss_val
                    if loss_ema is not None else loss_val)
        if hasattr(bar, "set_postfix"):
            bar.set_postfix(loss=f"{loss_val:.4f}", ema=f"{loss_ema:.4f}",
                            skip=skipped)

        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        bs = x.size(0)
        n += bs
        for k, v in losses.items():
            running[k] = running.get(k, 0.0) + v.detach() * bs

    stats = {k: float(v.item() if torch.is_tensor(v) else v) / max(n, 1)
             for k, v in running.items()}
    stats["skipped_steps"] = float(skipped)
    stats["throughput_sps"] = n / max(time.time() - t0, 1e-6)
    return stats, loss_ema if loss_ema is not None else 0.0


@torch.no_grad()
def validate(model, loader, loss_fn, device, n_classes: int) -> Dict[str, float]:
    tqdm = _tqdm_or_id()
    model.eval()
    tot_loss, n = 0.0, 0
    fp, ft, lp, lt, am = [], [], [], [], []
    for batch in tqdm(loader, desc="  validate  ", unit="b", leave=False,
                      ncols=110):
        x = batch["x"].to(device, non_blocking=True)
        feat = batch["feat"].to(device, non_blocking=True)
        phys = batch["phys"].to(device, non_blocking=True)
        target = {k: batch[k].to(device, non_blocking=True)
                  for k in ("fault_idx", "sev_level", "log_a_lo", "log_a_hi")}
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            pred = model(x, feat, phys)
            losses = loss_fn(pred, target)
        bs = x.size(0)
        tot_loss += float(losses["total"].item()) * bs
        n += bs
        fp.append(pred["fault_logits"].argmax(1).cpu())
        ft.append(target["fault_idx"].cpu())
        lp.append(corn_predict_level(pred["corn_logits"].float()).cpu())
        lt.append(target["sev_level"].cpu())
        am.append(pred["log_a_mu"].float().cpu())

    fp, ft = torch.cat(fp).numpy(), torch.cat(ft).numpy()
    lp, lt = torch.cat(lp).numpy(), torch.cat(lt).numpy()
    am = torch.cat(am).numpy()
    return {
        "val_loss": tot_loss / max(n, 1),
        "fault_f1_macro": f1_macro(fp, ft, n_classes),
        "sev_mae": float(np.abs(lp - lt).mean()),
        "sev_spearman": spearman(am, lt),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=5.6e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--physfeat", type=str,
                        default=str(V3_DIR / "physfeat" / "physfeat.parquet"))
    parser.add_argument("--paris-labels", type=str,
                        default=str(ROOT / "Hybrid_PINN_ParisRUL" / "results"
                                    / "labels" / "labels_paris.parquet"))
    parser.add_argument("--workers", type=int, default=-1,
                        help="DataLoader workers (-1 = Config default)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Ignore checkpoint_last.pt and start fresh")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sentinel = RESULTS_DIR / ".training_complete"
    if sentinel.exists():
        print(f"[v3:train] SKIP -- sentinel exists ({sentinel}); rm to retrain")
        return

    cfg = Config(batch_size=args.batch, num_epochs=args.epochs,
                 learning_rate=args.lr, weight_decay=args.weight_decay,
                 grad_clip_norm=args.grad_clip, patience=args.patience)
    cfg.seed_everything()
    cfg.apply_cudnn_settings()
    try:
        torch.backends.cuda.enable_cudnn_sdp(False)
    except AttributeError:
        pass
    cfg.num_workers = min(cfg.num_workers, os.cpu_count() or 1)
    if args.workers >= 0:
        cfg.num_workers = args.workers
    device = cfg.get_device()

    paris = args.paris_labels if Path(args.paris_labels).exists() else None
    shared_test = ROOT / "Hybrid_PINN_ParisRUL" / "results" / "test_index" / "test_windows.npz"
    cache_dir = V3_DIR.parent / "dataset_cache"  # shared with the v2 tracks
    train_loader, val_loader, _, phys_norm = make_v3_loaders(
        cfg, physfeat_path=args.physfeat, labels_paris_path=paris,
        shared_test_path=str(shared_test) if shared_test.exists() else None,
        feat_cache_dir=cache_dir,
        phys_norm_out=RESULTS_DIR / "phys_norm.json")

    model = DamageNet(n_classes=cfg.n_classes, hidden=args.hidden,
                      dropout=args.dropout).to(device)
    n_params = model.count_parameters()
    if cfg.compile_model:
        try:
            model = torch.compile(model, mode="reduce-overhead",
                                  fullgraph=False, dynamic=False)
            print("[v3:train] torch.compile mode=reduce-overhead active")
        except Exception:
            pass

    loss_fn = DamageLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    warmup_epochs = max(1, int(args.epochs * _WARMUP_FRAC))
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=6, min_lr=1e-7)

    def _inner(m):
        return getattr(m, "_orig_mod", m)

    # ── Resume from last checkpoint (crash-safe, every epoch) ────────────
    last_ckpt = RESULTS_DIR / "checkpoint_last.pt"
    best_ckpt = RESULTS_DIR / "best_model.pt"
    start_epoch, best_val, loss_ema = 1, float("inf"), None
    patience_left = args.patience
    if last_ckpt.exists() and not args.no_resume:
        ck = torch.load(last_ckpt, map_location=device, weights_only=False)
        _inner(model).load_state_dict(ck["state_dict"])
        optimizer.load_state_dict(ck["optimizer"])
        plateau.load_state_dict(ck["scheduler"])
        best_val = ck.get("best_val", float("inf"))
        loss_ema = ck.get("loss_ema")
        patience_left = ck.get("patience_left", args.patience)
        start_epoch = ck["epoch"] + 1
        torch.set_rng_state(ck["rng_torch"].cpu())
        np.random.set_state(ck["rng_numpy"])
        if device.type == "cuda" and ck.get("rng_cuda") is not None:
            torch.cuda.set_rng_state_all([s.cpu() for s in ck["rng_cuda"]])
        print(f"[v3:train] RESUMED at epoch {start_epoch} "
              f"(best_val={best_val:.4f})")

    n_train = len(train_loader.dataset)
    print(f"[v3:train] device={device} params={n_params:,} "
          f"train={n_train:,} windows batch={args.batch} lr={args.lr:.1e} "
          f"epochs={start_epoch}->{args.epochs}")

    history: List[Dict] = []
    hist_path = RESULTS_DIR / "history.json"
    if hist_path.exists() and start_epoch > 1:
        history = json.loads(hist_path.read_text())

    tqdm = _tqdm_or_id()
    epoch_iter = tqdm(range(start_epoch, args.epochs + 1), desc="  v3 epochs",
                      unit="ep", ncols=110, initial=start_epoch - 1,
                      total=args.epochs)
    for epoch in epoch_iter:
        train_stats, loss_ema = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device, epoch,
            args.grad_clip, loss_ema)
        val_stats = validate(model, val_loader, loss_fn, device, cfg.n_classes)

        if epoch <= warmup_epochs:
            warmup.step()
        else:
            plateau.step(val_stats["val_loss"])

        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
               **{f"train_{k}": v for k, v in train_stats.items()},
               **val_stats}
        history.append(row)
        hist_path.write_text(json.dumps(history, indent=2, default=str))
        print(f"[ep {epoch:3d}/{args.epochs}] "
              f"train={train_stats.get('total', 0):.4f} "
              f"val={val_stats['val_loss']:.4f} "
              f"f1={val_stats['fault_f1_macro']:.3f} "
              f"sev_mae={val_stats['sev_mae']:.3f} "
              f"rho={val_stats['sev_spearman']:.3f} "
              f"sps={train_stats['throughput_sps']:.0f}")

        improvement = best_val - val_stats["val_loss"]
        if improvement > _MIN_DELTA:
            best_val = val_stats["val_loss"]
            patience_left = args.patience
            torch.save({"epoch": epoch,
                        "state_dict": _inner(model).state_dict(),
                        "best_val": best_val, "val_metrics": val_stats,
                        "hidden": args.hidden, "dropout": args.dropout,
                        "phys_norm_mean": phys_norm[0].tolist(),
                        "phys_norm_std": phys_norm[1].tolist()},
                       best_ckpt)
            print(f"   -> [BEST] val={best_val:.4f} -> {best_ckpt.name}")
        else:
            patience_left -= 1

        # Crash-safe resume point, every epoch
        torch.save({"epoch": epoch, "state_dict": _inner(model).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": plateau.state_dict(),
                    "best_val": best_val, "loss_ema": loss_ema,
                    "patience_left": patience_left,
                    "rng_torch": torch.get_rng_state(),
                    "rng_numpy": np.random.get_state(),
                    "rng_cuda": (torch.cuda.get_rng_state_all()
                                 if device.type == "cuda" else None)},
                   last_ckpt)

        if epoch > warmup_epochs + 5 and best_val < float("inf") \
                and val_stats["val_loss"] > _DIVERGE_FACTOR * best_val:
            print(f"   -> [DIVERGE] stopping (val > {_DIVERGE_FACTOR}x best)")
            break
        if patience_left <= 0:
            print(f"   -> [EARLY STOP] best val={best_val:.4f}")
            break

    sentinel.touch()
    print(f"[v3:train] DONE best_val={best_val:.4f} sentinel -> {sentinel}")


if __name__ == "__main__":
    main()
