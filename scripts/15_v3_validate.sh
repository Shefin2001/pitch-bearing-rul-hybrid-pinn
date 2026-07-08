#!/usr/bin/env bash
# 15_v3_validate.sh — Stage 15: physics-consistency gates + expert report.
set -euo pipefail

ROOT="${ROOT:-$HOME}"
HYBRID="${HYBRID:-$ROOT/Hybrid_PINN_ParisRUL}"
DATA="${DATA:-$ROOT/pitch_bearing_dataset.parquet}"
export PARQUET_PATH="${PARQUET_PATH:-$DATA}"
echo "=== 15_v3_validate ==="

SENTINEL="$HYBRID/results/v3/.validation_complete"
if [ -f "$SENTINEL" ]; then
    echo "[15_v3_validate] SKIP — already validated ($SENTINEL)"
    exit 0
fi

cd "$ROOT"
python -m Hybrid_PINN_ParisRUL.v3.validate

echo "[15_v3_validate] OK — report: $HYBRID/results/v3/expert_report.md"
