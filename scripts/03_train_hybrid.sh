#!/usr/bin/env bash
# 03_train_hybrid.sh — Train Hybrid track (DDP/AMP).
set -euo pipefail

ROOT="${ROOT:-$HOME}"
DATA="${DATA:-$ROOT/pitch_bearing_dataset.parquet}"
export PARQUET_PATH="${PARQUET_PATH:-$DATA}"
echo "=== 03_train_hybrid ==="

EPOCHS="${EPOCHS:-100}"
BATCH="${BATCH:-128}"
NPROC="${NPROC:-auto}"

cd "$ROOT"
torchrun --standalone --nproc_per_node="$NPROC" \
    Hybrid_PINN_ParisRUL/track_hybrid/train.py \
    --epochs "$EPOCHS" --batch "$BATCH" --amp

echo "[03_train_hybrid] OK"
