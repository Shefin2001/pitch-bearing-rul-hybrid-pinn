#!/usr/bin/env bash
# 11_v3_train.sh — Stage 11: train DamageNet (v3 track, single GPU).
set -euo pipefail

ROOT="${ROOT:-$HOME}"
HYBRID="${HYBRID:-$ROOT/Hybrid_PINN_ParisRUL}"
DATA="${DATA:-$ROOT/pitch_bearing_dataset.parquet}"
export PARQUET_PATH="${PARQUET_PATH:-$DATA}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
echo "=== 11_v3_train ==="

EPOCHS="${EPOCHS:-40}"
BATCH="${BATCH:-8192}"      # H200 141GB; DamageNet is tiny so this is loader-bound
LR="${LR:-5.6e-4}"          # sqrt-scaling convention (entries 9/10/15)

SENTINEL="$HYBRID/results/v3/damage_net/.training_complete"
if [ -f "$SENTINEL" ]; then
    echo "[11_v3_train] SKIP — training already complete ($SENTINEL)"
    exit 0
fi

PHYS="$HYBRID/results/v3/physfeat/physfeat.parquet"
if [ ! -f "$PHYS" ]; then
    echo "[11_v3_train] ERROR: $PHYS missing — run 10_v3_physfeat.sh first" >&2
    exit 1
fi

cd "$ROOT"
# checkpoint_last.pt is written every epoch; train.py auto-resumes from it
python -m Hybrid_PINN_ParisRUL.v3.train \
    --epochs "$EPOCHS" --batch "$BATCH" --lr "$LR"

echo "[11_v3_train] OK"
