#!/usr/bin/env bash
# 01_build_labels.sh — Build FPT-piecewise labels + Paris-law TTF labels.
set -euo pipefail

# ROOT = workspace root (contains parquet + Hybrid_PINN_ParisRUL/)
ROOT="${ROOT:-$HOME}"
# REPO = folder to cd into for python -m imports; same as ROOT when no wrapper dir exists
REPO="${REPO:-$ROOT}"
HYBRID="${HYBRID:-$ROOT/Hybrid_PINN_ParisRUL}"
DATA="${DATA:-$ROOT/pitch_bearing_dataset.parquet}"
export PARQUET_PATH="${PARQUET_PATH:-$DATA}"
echo "=== 01_build_labels ==="

FPT_OUT="$HYBRID/results/labels/labels_fpt.parquet"
PARIS_OUT="$HYBRID/results/labels/labels_paris.parquet"

# Skip entirely if both outputs already exist
if [ -f "$FPT_OUT" ] && [ -f "$PARIS_OUT" ]; then
    echo "[01_build_labels] SKIP — labels already built"
    echo "  FPT   : $FPT_OUT"
    echo "  Paris : $PARIS_OUT"
    echo "  (delete these files to force rebuild)"
    echo "[01_build_labels] OK"
    exit 0
fi

cd "$REPO"

# Step 1a: FPT-piecewise labels
if [ -f "$FPT_OUT" ]; then
    echo "[01_build_labels] SKIP step 1a — labels_fpt.parquet already exists"
else
    echo "[01_build_labels] Step 1a: building FPT-piecewise RUL labels..."
    python -m Hybrid_PINN_ParisRUL.common.rul_labels_v2 \
        --parquet "$DATA" \
        --out "$FPT_OUT"
    echo "[01_build_labels] Step 1a DONE → $FPT_OUT"
fi

# Step 1b: Paris-law TTF labels
if [ -f "$PARIS_OUT" ]; then
    echo "[01_build_labels] SKIP step 1b — labels_paris.parquet already exists"
else
    echo "[01_build_labels] Step 1b: building Paris-law TTF labels..."
    python -m Hybrid_PINN_ParisRUL.common.paris_labels \
        --fpt-labels "$FPT_OUT" \
        --out        "$PARIS_OUT"
    echo "[01_build_labels] Step 1b DONE → $PARIS_OUT"
fi

echo "[01_build_labels] OK"
