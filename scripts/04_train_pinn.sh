#!/usr/bin/env bash
# 04_train_pinn.sh — Train PINN track.
set -euo pipefail

ROOT="${ROOT:-$HOME}"
HYBRID="${HYBRID:-$ROOT/Hybrid_PINN_ParisRUL}"
DATA="${DATA:-$ROOT/pitch_bearing_dataset.parquet}"
export PARQUET_PATH="${PARQUET_PATH:-$DATA}"
echo "=== 04_train_pinn ==="

EPOCHS="${EPOCHS:-100}"
BATCH="${BATCH:-256}"        # H100 80GB: 4× default; TensorCore-aligned
LR="${LR:-1e-4}"             # √4 × base LR (scales with √batch_size from 64→256)
NPROC="${NPROC:-auto}"

SENTINEL="$HYBRID/results/pinn/.training_complete"
CKPT="$HYBRID/results/pinn/best_model.pt"

# Already completed — skip
if [ -f "$SENTINEL" ]; then
    echo "[04_train_pinn] SKIP — training already complete"
    echo "  Sentinel : $SENTINEL"
    echo "  (delete sentinel to re-run from scratch)"
    echo "[04_train_pinn] OK"
    exit 0
fi

# Same torchrun bypass as 03_train_hybrid.sh — see comment there.
NGPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 1)
if [ "$NPROC" = "auto" ]; then NPROC="$NGPU"; fi

RESUME_FLAG=""
if [ -f "$CKPT" ]; then
    echo "[04_train_pinn] Checkpoint found — resuming training"
    echo "  Checkpoint : $CKPT"
    RESUME_FLAG="--resume"
else
    echo "[04_train_pinn] No checkpoint found — starting fresh"
fi

echo "[04_train_pinn] epochs=$EPOCHS  batch=$BATCH  lr=$LR  GPUs=$NPROC"

cd "$ROOT"
if [ "$NPROC" -le 1 ]; then
    echo "[04_train_pinn] 1 GPU — bypassing torchrun (avoids c10d segfault)"
    python -m Hybrid_PINN_ParisRUL.track_pinn.train \
        --epochs "$EPOCHS" --batch "$BATCH" --lr "$LR" $RESUME_FLAG
else
    torchrun --standalone --nproc_per_node="$NPROC" \
        Hybrid_PINN_ParisRUL/track_pinn/train.py \
        --epochs "$EPOCHS" --batch "$BATCH" --lr "$LR" $RESUME_FLAG
fi

echo "[04_train_pinn] OK"
