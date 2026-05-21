#!/usr/bin/env bash
# 00_setup.sh — Verify env, create dirs, tag pre-novel state.
set -euo pipefail

# ROOT = workspace root (contains pitch_bearing_dataset.parquet and Hybrid_PINN_ParisRUL/)
ROOT="${ROOT:-$HOME}"
# HYBRID = the Hybrid_PINN_ParisRUL subfolder
HYBRID="${HYBRID:-$ROOT/Hybrid_PINN_ParisRUL}"
DATA="${DATA:-$ROOT/pitch_bearing_dataset.parquet}"
export PARQUET_PATH="${PARQUET_PATH:-$DATA}"

echo "=== 00_setup ==="
echo "  ROOT=$ROOT"
echo "  HYBRID=$HYBRID"

mkdir -p "$HYBRID/results/labels"
mkdir -p "$HYBRID/results/test_index"
mkdir -p "$HYBRID/results/hybrid/tensorboard"
mkdir -p "$HYBRID/results/pinn/tensorboard"
mkdir -p "$HYBRID/results/fusion"
mkdir -p "$HYBRID/results/plots"

# Tag pre-novel state if not present
( cd "$ROOT" && git tag -f v1-pre-novel 2>/dev/null || true )

# Verify Python deps
python -c "
import torch, pyarrow, scipy, numpy
print(f'  torch    : {torch.__version__} (cuda={torch.cuda.is_available()})')
print(f'  pyarrow  : {pyarrow.__version__}')
print(f'  scipy    : {scipy.__version__}')
print(f'  numpy    : {numpy.__version__}')
try:
    import pywt; print(f'  pywavelets: {pywt.__version__}')
except ImportError: print('  pywavelets: MISSING — wavelet features will be 0')
try:
    import numba; print(f'  numba    : {numba.__version__}')
except ImportError: print('  numba    : MISSING — fallback to numpy')
"

# GPU check
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    echo "  no GPU detected (CPU mode)"
fi

echo "[00_setup] OK"
