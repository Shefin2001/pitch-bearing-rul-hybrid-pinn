#!/usr/bin/env bash
# 00_setup.sh — Verify env, create dirs, tag pre-novel state.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME}"
ROOT="${ROOT:-$PROJECT_ROOT/Hybrid_PINN_ParisRUL}"
PROJECT="${PROJECT:-$PROJECT_ROOT}"
DATA="${DATA:-$PROJECT_ROOT/pitch_bearing_dataset.parquet}"
export PARQUET_PATH="${PARQUET_PATH:-$DATA}"

echo "=== 00_setup ==="
echo "  ROOT=$ROOT"

mkdir -p "$ROOT/results/labels"
mkdir -p "$ROOT/results/test_index"
mkdir -p "$ROOT/results/hybrid/tensorboard"
mkdir -p "$ROOT/results/pinn/tensorboard"
mkdir -p "$ROOT/results/fusion"
mkdir -p "$ROOT/results/plots"

# Tag pre-novel state if not present
( cd "$PROJECT" && git tag -f v1-pre-novel 2>/dev/null || true )

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
