#!/usr/bin/env bash
# 01_build_labels.sh — Build FPT-piecewise labels + Paris-law TTF labels.
set -euo pipefail

ROOT="${ROOT:-$HOME}"
DATA="${DATA:-$ROOT/pitch_bearing_dataset.parquet}"
export PARQUET_PATH="${PARQUET_PATH:-$DATA}"
echo "=== 01_build_labels ==="

cd "$ROOT"

# Step 1: FPT-piecewise labels (per recording)
python -m Hybrid_PINN_ParisRUL.common.rul_labels_v2 \
    --parquet "$DATA" \
    --out "$ROOT/Hybrid_PINN_ParisRUL/results/labels/labels_fpt.parquet"

# Step 2: Paris-law TTF labels (consumes FPT labels above)
python -m Hybrid_PINN_ParisRUL.common.paris_labels \
    --fpt-labels "$ROOT/Hybrid_PINN_ParisRUL/results/labels/labels_fpt.parquet" \
    --out        "$ROOT/Hybrid_PINN_ParisRUL/results/labels/labels_paris.parquet"

echo "[01_build_labels] OK"
