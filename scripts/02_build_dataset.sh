#!/usr/bin/env bash
# 02_build_dataset.sh — Discover runs, materialise shared test index.
set -euo pipefail

ROOT="${ROOT:-$HOME}"
REPO="${REPO:-$ROOT}"
HYBRID="${HYBRID:-$ROOT/Hybrid_PINN_ParisRUL}"
DATA="${DATA:-$ROOT/pitch_bearing_dataset.parquet}"
export PARQUET_PATH="${PARQUET_PATH:-$DATA}"
echo "=== 02_build_dataset ==="

NPZ_OUT="$HYBRID/results/test_index/test_windows.npz"
PARIS_LABELS="$HYBRID/results/labels/labels_paris.parquet"

if [ -f "$NPZ_OUT" ]; then
    echo "[02_build_dataset] SKIP — shared test index already exists"
    echo "  $NPZ_OUT"
    echo "  (delete this file to force rebuild)"
    echo "[02_build_dataset] OK"
    exit 0
fi

echo "[02_build_dataset] Building shared test index..."
cd "$REPO"
python -m Hybrid_PINN_ParisRUL.common.dataset_v2 --build \
    --paris-labels "$PARIS_LABELS" \
    --out          "$NPZ_OUT"
echo "[02_build_dataset] DONE → $NPZ_OUT"
echo "[02_build_dataset] OK"
