#!/usr/bin/env bash
# 10_v3_physfeat.sh — Stage 10: run-level physics features (v3 track).
# Backends: CuPy GPU (auto), CPU pool (WORKERS=N), or MPI (USE_MPI=1, NPROC ranks).
set -euo pipefail

ROOT="${ROOT:-$HOME}"
HYBRID="${HYBRID:-$ROOT/Hybrid_PINN_ParisRUL}"
DATA="${DATA:-$ROOT/pitch_bearing_dataset.parquet}"
export PARQUET_PATH="${PARQUET_PATH:-$DATA}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"   # numpy/MKL kernels per worker
echo "=== 10_v3_physfeat ==="

OUT_DIR="$HYBRID/results/v3/physfeat"
SENTINEL="$OUT_DIR/.physfeat_complete"

if [ -f "$SENTINEL" ]; then
    echo "[10_v3_physfeat] SKIP — already complete ($SENTINEL)"
    exit 0
fi

WORKERS="${WORKERS:-0}"
USE_MPI="${USE_MPI:-0}"
NPROC="${NPROC:-16}"

cd "$ROOT"
if [ "$USE_MPI" = "1" ]; then
    echo "[10_v3_physfeat] MPI mode — $NPROC ranks"
    mpirun -np "$NPROC" python -m Hybrid_PINN_ParisRUL.v3.build_physfeat --mpi
else
    # auto: CuPy GPU if present, else CPU pool
    python -m Hybrid_PINN_ParisRUL.v3.build_physfeat --workers "$WORKERS"
fi

echo "[10_v3_physfeat] OK"
