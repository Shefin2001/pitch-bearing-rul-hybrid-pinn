#!/usr/bin/env bash
# 03_train_hybrid.sh — Train Hybrid track (DDP/AMP).
set -euo pipefail

ROOT="${ROOT:-$HOME}"
HYBRID="${HYBRID:-$ROOT/Hybrid_PINN_ParisRUL}"
DATA="${DATA:-$ROOT/pitch_bearing_dataset.parquet}"
export PARQUET_PATH="${PARQUET_PATH:-$DATA}"
echo "=== 03_train_hybrid ==="

EPOCHS="${EPOCHS:-100}"
BATCH="${BATCH:-4096}"       # H200 141GB: 4096 fills VRAM; 4× fewer steps vs 1024
LR="${LR:-8e-4}"             # √4 × 4e-4 (4× batch increase → 2× LR via sqrt-scaling rule)
NPROC="${NPROC:-auto}"

SENTINEL="$HYBRID/results/hybrid/.training_complete"
CKPT="$HYBRID/results/hybrid/best_model.pt"

# Already completed — skip
if [ -f "$SENTINEL" ]; then
    echo "[03_train_hybrid] SKIP — training already complete"
    echo "  Sentinel : $SENTINEL"
    echo "  (delete sentinel to re-run from scratch)"
    echo "[03_train_hybrid] OK"
    exit 0
fi

# Resolve 'auto' to actual GPU count.
# Bypass torchrun for single-GPU: torchrun initialises a c10d TCPStore rendezvous
# even with nproc=1, which segfaults in container environments (e.g. Lightning AI).
NGPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 1)
if [ "$NPROC" = "auto" ]; then NPROC="$NGPU"; fi

# Build the Python args list — add --resume if a checkpoint exists
RESUME_FLAG=""
if [ -f "$CKPT" ]; then
    echo "[03_train_hybrid] Checkpoint found — resuming training"
    echo "  Checkpoint : $CKPT"
    RESUME_FLAG="--resume"
else
    echo "[03_train_hybrid] No checkpoint found — starting fresh"
fi

echo "[03_train_hybrid] epochs=$EPOCHS  batch=$BATCH  lr=$LR  GPUs=$NPROC"

cd "$ROOT"
if [ "$NPROC" -le 1 ]; then
    echo "[03_train_hybrid] 1 GPU — bypassing torchrun (avoids c10d segfault)"
    python -m Hybrid_PINN_ParisRUL.track_hybrid.train \
        --epochs "$EPOCHS" --batch "$BATCH" --lr "$LR" --amp $RESUME_FLAG
else
    torchrun --standalone --nproc_per_node="$NPROC" \
        Hybrid_PINN_ParisRUL/track_hybrid/train.py \
        --epochs "$EPOCHS" --batch "$BATCH" --lr "$LR" --amp $RESUME_FLAG
fi

echo "[03_train_hybrid] OK"
