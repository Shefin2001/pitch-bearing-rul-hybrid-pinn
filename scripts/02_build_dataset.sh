#!/usr/bin/env bash
# 02_build_dataset.sh — Discover runs, materialise shared test index.
set -euo pipefail

ROOT="${ROOT:-$HOME}"
DATA="${DATA:-$ROOT/pitch_bearing_dataset.parquet}"
export PARQUET_PATH="${PARQUET_PATH:-$DATA}"
echo "=== 02_build_dataset ==="

cd "$ROOT"
python -m Hybrid_PINN_ParisRUL.common.dataset_v2 --build \
    --paris-labels "$ROOT/Hybrid_PINN_ParisRUL/results/labels/labels_paris.parquet" \
    --out          "$ROOT/Hybrid_PINN_ParisRUL/results/test_index/test_windows.npz"

echo "[02_build_dataset] OK"
