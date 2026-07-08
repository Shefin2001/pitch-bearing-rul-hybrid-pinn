#!/usr/bin/env bash
# 14_v3_export.sh — Stage 14: ONNX export (fp32 + INT8) + inference smoke.
set -euo pipefail

ROOT="${ROOT:-$HOME}"
HYBRID="${HYBRID:-$ROOT/Hybrid_PINN_ParisRUL}"
echo "=== 14_v3_export ==="

SENTINEL="$HYBRID/results/v3/export/.export_complete"
cd "$ROOT"

if [ -f "$SENTINEL" ]; then
    echo "[14_v3_export] SKIP — export already complete ($SENTINEL)"
else
    python -m Hybrid_PINN_ParisRUL.v3.export_onnx
fi

# Torch-free smoke: must run without CUDA and print a full prediction dict
echo "[14_v3_export] inference smoke (fp32) ..."
python -m Hybrid_PINN_ParisRUL.v3.inference_v3 --demo > /dev/null
echo "[14_v3_export] inference smoke (int8) ..."
python -m Hybrid_PINN_ParisRUL.v3.inference_v3 --demo --mode int8 > /dev/null

echo "[14_v3_export] OK"
