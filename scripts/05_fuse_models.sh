#!/usr/bin/env bash
# 05_fuse_models.sh — Distill student from teachers, export INT8 + FP16 builds.
set -euo pipefail

ROOT="${ROOT:-$HOME}"
DATA="${DATA:-$ROOT/pitch_bearing_dataset.parquet}"
export PARQUET_PATH="${PARQUET_PATH:-$DATA}"
echo "=== 05_fuse_models ==="

EPOCHS="${EPOCHS:-30}"
BATCH="${BATCH:-64}"

cd "$ROOT"
python -m Hybrid_PINN_ParisRUL.track_fusion.distill \
    --hybrid       "$ROOT/Hybrid_PINN_ParisRUL/results/hybrid/best_model.pt" \
    --pinn         "$ROOT/Hybrid_PINN_ParisRUL/results/pinn/best_model.pt" \
    --paris-labels "$ROOT/Hybrid_PINN_ParisRUL/results/labels/labels_paris.parquet" \
    --epochs "$EPOCHS" --batch "$BATCH" \
    --export-edge --export-cloud

echo "[05_fuse_models] OK"
