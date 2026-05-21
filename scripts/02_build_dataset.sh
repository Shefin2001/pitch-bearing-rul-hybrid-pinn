#!/usr/bin/env bash
# 02_build_dataset.sh — Discover runs, materialise shared test index,
#                        pre-extract train/val features to disk cache.
#
# Running this once (takes 60-90 min on first run) means Stage 3 training
# never waits for feature extraction — it loads the cache in ~2 min instead.
set -euo pipefail

ROOT="${ROOT:-$HOME}"
REPO="${REPO:-$ROOT}"
HYBRID="${HYBRID:-$ROOT/Hybrid_PINN_ParisRUL}"
DATA="${DATA:-$ROOT/pitch_bearing_dataset.parquet}"
export PARQUET_PATH="${PARQUET_PATH:-$DATA}"
echo "=== 02_build_dataset ==="

NPZ_OUT="$HYBRID/results/test_index/test_windows.npz"
PARIS_LABELS="$HYBRID/results/labels/labels_paris.parquet"
CACHE_DIR="$HYBRID/results/dataset_cache"

# ── Step 1: shared test index ─────────────────────────────────────────────
if [ -f "$NPZ_OUT" ]; then
    echo "[02_build_dataset] SKIP test-index — already exists: $NPZ_OUT"
else
    echo "[02_build_dataset] Building shared test index..."
    cd "$REPO"
    python -m Hybrid_PINN_ParisRUL.common.dataset_v2 --build \
        --paris-labels "$PARIS_LABELS" \
        --out          "$NPZ_OUT"
    echo "[02_build_dataset] Test index saved → $NPZ_OUT"
fi

# ── Step 2: pre-extract train + val features to disk cache ────────────────
# Checks if any train_*.npz already exists in the cache dir; if so, skips.
# This is the heavy step (60-90 min) but only runs once — subsequent Stage 3
# runs load from cache in ~2 min.
if ls "$CACHE_DIR"/train_*.npz &>/dev/null 2>&1; then
    echo "[02_build_dataset] SKIP feature-precompute — train cache exists: $CACHE_DIR/train_*.npz"
else
    echo "[02_build_dataset] Pre-extracting train+val features (one-time, ~60-90 min)..."
    echo "  After this, Stage 3 (training) will skip feature extraction entirely."
    cd "$REPO"
    python -m Hybrid_PINN_ParisRUL.common.dataset_v2 --precompute-splits \
        --paris-labels "$PARIS_LABELS" \
        --cache-dir    "$CACHE_DIR"
    echo "[02_build_dataset] Feature cache ready → $CACHE_DIR"
fi

echo "[02_build_dataset] OK"
