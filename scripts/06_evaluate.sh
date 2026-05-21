#!/usr/bin/env bash
# 06_evaluate.sh — Evaluate Hybrid, PINN, Ensemble on shared test index.
set -euo pipefail

ROOT="${ROOT:-$HOME}"
HYBRID="${HYBRID:-$ROOT/Hybrid_PINN_ParisRUL}"
DATA="${DATA:-$ROOT/pitch_bearing_dataset.parquet}"
export PARQUET_PATH="${PARQUET_PATH:-$DATA}"
echo "=== 06_evaluate ==="

MC_PASSES="${MC_PASSES:-1}"
CSV_OUT="$HYBRID/results/comparison_v2.csv"

if [ -f "$CSV_OUT" ]; then
    echo "[06_evaluate] SKIP — evaluation already complete"
    echo "  Results : $CSV_OUT"
    echo "  (delete this file to re-evaluate)"
    echo "[06_evaluate] OK"
    exit 0
fi

echo "[06_evaluate] Running evaluation (mc_passes=$MC_PASSES)..."
cd "$ROOT"
python -m Hybrid_PINN_ParisRUL.compare_v2 \
    --shared-test  "$HYBRID/results/test_index/test_windows.npz" \
    --paris-labels "$HYBRID/results/labels/labels_paris.parquet" \
    --out-csv      "$CSV_OUT" \
    --mc-passes "$MC_PASSES"

echo "[06_evaluate] DONE → $CSV_OUT"
echo "[06_evaluate] OK"
