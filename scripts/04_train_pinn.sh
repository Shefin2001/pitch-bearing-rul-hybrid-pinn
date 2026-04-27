#!/usr/bin/env bash
# 04_train_pinn.sh — Train PINN track.
set -euo pipefail

ROOT="${ROOT:-$HOME}"
DATA="${DATA:-$ROOT/pitch_bearing_dataset.parquet}"
export PARQUET_PATH="${PARQUET_PATH:-$DATA}"
echo "=== 04_train_pinn ==="

EPOCHS="${EPOCHS:-100}"
BATCH="${BATCH:-64}"
NPROC="${NPROC:-1}"

cd "$ROOT"
torchrun --standalone --nproc_per_node="$NPROC" \
    Hybrid_PINN_ParisRUL/track_pinn/train.py \
    --epochs "$EPOCHS" --batch "$BATCH"

echo "[04_train_pinn] OK"
