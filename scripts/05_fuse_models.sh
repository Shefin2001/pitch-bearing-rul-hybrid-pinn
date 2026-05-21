#!/usr/bin/env bash
# 05_fuse_models.sh — Distill student from teachers, export INT8 + FP16 builds.
set -euo pipefail

ROOT="${ROOT:-$HOME}"
HYBRID="${HYBRID:-$ROOT/Hybrid_PINN_ParisRUL}"
DATA="${DATA:-$ROOT/pitch_bearing_dataset.parquet}"
export PARQUET_PATH="${PARQUET_PATH:-$DATA}"
echo "=== 05_fuse_models ==="

EPOCHS="${EPOCHS:-30}"
BATCH="${BATCH:-64}"

SENTINEL="$HYBRID/results/fusion/.fuse_complete"

if [ -f "$SENTINEL" ]; then
    echo "[05_fuse_models] SKIP — fusion already complete"
    echo "  Sentinel : $SENTINEL"
    echo "  (delete sentinel to re-run)"
    echo "[05_fuse_models] OK"
    exit 0
fi

echo "[05_fuse_models] Starting distillation (epochs=$EPOCHS  batch=$BATCH)..."
cd "$ROOT"
python -m Hybrid_PINN_ParisRUL.track_fusion.distill \
    --hybrid       "$HYBRID/results/hybrid/best_model.pt" \
    --pinn         "$HYBRID/results/pinn/best_model.pt" \
    --paris-labels "$HYBRID/results/labels/labels_paris.parquet" \
    --epochs "$EPOCHS" --batch "$BATCH" \
    --export-edge --export-cloud

# Write sentinel only after distill.py exits cleanly (set -e ensures this)
touch "$SENTINEL"
echo "[05_fuse_models] DONE — sentinel written → $SENTINEL"
echo "[05_fuse_models] OK"
