#!/usr/bin/env bash
# 13_v3_propagation.sh — Stage 13: materialise the propagation sojourn table.
set -euo pipefail

ROOT="${ROOT:-$HOME}"
HYBRID="${HYBRID:-$ROOT/Hybrid_PINN_ParisRUL}"
echo "=== 13_v3_propagation ==="

OUT="$HYBRID/results/v3/propagation_table.json"
if [ -f "$OUT" ]; then
    echo "[13_v3_propagation] SKIP — table exists ($OUT)"
    exit 0
fi

cd "$ROOT"
python -m Hybrid_PINN_ParisRUL.v3.propagation --out "$OUT"

echo "[13_v3_propagation] OK"
