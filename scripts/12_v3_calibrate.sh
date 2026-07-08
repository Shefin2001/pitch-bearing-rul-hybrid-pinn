#!/usr/bin/env bash
# 12_v3_calibrate.sh — Stage 12: conformal calibration of the damage head.
set -euo pipefail

ROOT="${ROOT:-$HOME}"
HYBRID="${HYBRID:-$ROOT/Hybrid_PINN_ParisRUL}"
DATA="${DATA:-$ROOT/pitch_bearing_dataset.parquet}"
export PARQUET_PATH="${PARQUET_PATH:-$DATA}"
echo "=== 12_v3_calibrate ==="

OUT="$HYBRID/results/v3/calibration.json"
if [ -f "$OUT" ]; then
    echo "[12_v3_calibrate] SKIP — already calibrated ($OUT)"
    exit 0
fi

cd "$ROOT"
python -m Hybrid_PINN_ParisRUL.v3.calibrate

echo "[12_v3_calibrate] OK"
