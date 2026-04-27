#!/usr/bin/env bash
# 06_evaluate.sh — Evaluate Hybrid, PINN, Ensemble on shared test index.
set -euo pipefail

ROOT="${ROOT:-$HOME}"
DATA="${DATA:-$ROOT/pitch_bearing_dataset.parquet}"
export PARQUET_PATH="${PARQUET_PATH:-$DATA}"
echo "=== 06_evaluate ==="

MC_PASSES="${MC_PASSES:-1}"

cd "$ROOT"
python -m Hybrid_PINN_ParisRUL.compare_v2 \
    --shared-test  "$ROOT/Hybrid_PINN_ParisRUL/results/test_index/test_windows.npz" \
    --paris-labels "$ROOT/Hybrid_PINN_ParisRUL/results/labels/labels_paris.parquet" \
    --out-csv      "$ROOT/Hybrid_PINN_ParisRUL/results/comparison_v2.csv" \
    --mc-passes "$MC_PASSES"

echo "[06_evaluate] OK"
