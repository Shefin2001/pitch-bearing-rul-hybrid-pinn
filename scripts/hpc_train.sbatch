#!/bin/bash
# ============================================================
# SLURM job script — Hybrid + PINN dual-track training
# ============================================================
# Usage:
#   sbatch hpc_train.sbatch
#
# Customise the SBATCH directives below for your cluster.
# Common presets are commented — uncomment the right block.
# ============================================================

# ---- Resource request (adjust per cluster) ----------------
#SBATCH --job-name=pitchbearing_rul
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=24:00:00

# -- H100 single-node (4× or 8× H100 80 GB SXM5) ---
# Most H100 clusters: 4 or 8 GPUs / node, 96 CPU cores, 384–768 GB RAM.
# Check with: sinfo -o "%P %G %c %m" or your HPC docs.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=256G

# -- 8-GPU H100 node (uncomment if your cluster has 8-way nodes) ---
##SBATCH --gres=gpu:8
##SBATCH --cpus-per-task=12
##SBATCH --mem=256G

# -- Multi-node alternative (uncomment to use) ---
##SBATCH --nodes=2
##SBATCH --ntasks-per-node=1
##SBATCH --gres=gpu:4
##SBATCH --cpus-per-task=24
##SBATCH --mem=256G

# ---- Partition (uncomment the one that matches your HPC) --
##SBATCH --partition=gpu
##SBATCH --partition=h100
##SBATCH --partition=h100_80gb
##SBATCH --partition=hgx

# ---- Email notifications (optional) -----------------------
##SBATCH --mail-type=BEGIN,END,FAIL
##SBATCH --mail-user=your@email.com

# ===========================================================
# Environment setup
# ===========================================================

set -euo pipefail

# Activate conda / venv — edit to match your environment name
# conda activate pitch_rul
# source $HOME/venvs/pitch_rul/bin/activate

# Module loads — uncomment and edit for your HPC module system
# module load cuda/12.1
# module load cudnn/8.9
# module load python/3.11

# ---- Paths (override Windows defaults in config.py) -------
export PARQUET_PATH="${PARQUET_PATH:-$SCRATCH/pitch_bearing_dataset.parquet}"
export OUTPUT_DIR="${OUTPUT_DIR:-$SCRATCH/PitchBearing_RUL_DualNN}"

# Repo root — set to wherever you cloned the project
REPO_ROOT="${REPO_ROOT:-$HOME/PitchBearing_RUL_DualNN}"

# PYTHONPATH must include both repo root (for common/) and the
# Hybrid_PINN_ParisRUL parent so package imports resolve correctly.
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/Hybrid_PINN_ParisRUL:${PYTHONPATH:-}"

# ---- NCCL / runtime tuning for H100 ──────────────────────
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0                    # keep InfiniBand enabled
export NCCL_SOCKET_IFNAME=^lo,docker
export NCCL_NVLS_ENABLE=1                   # H100 NVLink SHARP collective acceleration
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / 2))
# expandable_segments avoids OOM from memory fragmentation on large batches
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- Detect number of GPUs available ----------------------
NGPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 1)
echo "[sbatch] Using ${NGPU} GPU(s) on $(hostname)"
echo "[sbatch] PARQUET_PATH = ${PARQUET_PATH}"
echo "[sbatch] OUTPUT_DIR   = ${OUTPUT_DIR}"
echo "[sbatch] PYTHONPATH   = ${PYTHONPATH}"

# Create log directory
mkdir -p logs

# ===========================================================
# Stage 1 — Build labels: FPT-piecewise first, then Paris-law TTF
# ===========================================================
echo "[sbatch] Stage 1a: building FPT-piecewise RUL labels..."
python -m Hybrid_PINN_ParisRUL.common.rul_labels_v2 \
    --parquet "${PARQUET_PATH}" \
    --out     "${OUTPUT_DIR}/Hybrid_PINN_ParisRUL/results/labels/labels_fpt.parquet"

echo "[sbatch] Stage 1b: building Paris-law TTF labels..."
python -m Hybrid_PINN_ParisRUL.common.paris_labels \
    --fpt-labels "${OUTPUT_DIR}/Hybrid_PINN_ParisRUL/results/labels/labels_fpt.parquet" \
    --out        "${OUTPUT_DIR}/Hybrid_PINN_ParisRUL/results/labels/labels_paris.parquet"

# ===========================================================
# Stage 2 — Build shared test index + pre-extract train/val features
#
# Pre-extracting features here (one-time, 60-90 min) means Stage 3 loads
# from cache in ~2 min instead of re-extracting on every training run.
# PARQUET_PATH env var is already exported above; dataset_v2 reads it via
# cfg.parquet_path — no --parquet CLI arg needed.
# ===========================================================
_PARIS="${OUTPUT_DIR}/Hybrid_PINN_ParisRUL/results/labels/labels_paris.parquet"
_NPZ="${OUTPUT_DIR}/Hybrid_PINN_ParisRUL/results/test_index/test_windows.npz"
_CACHE="${OUTPUT_DIR}/Hybrid_PINN_ParisRUL/results/dataset_cache"

echo "[sbatch] Stage 2a: building shared test index..."
python -m Hybrid_PINN_ParisRUL.common.dataset_v2 --build \
    --paris-labels "${_PARIS}" \
    --out          "${_NPZ}"

echo "[sbatch] Stage 2b: pre-extracting train+val features (one-time ~60-90 min)..."
if ls "${_CACHE}"/train_*.npz &>/dev/null 2>&1; then
    echo "[sbatch] Feature cache exists — skipping precompute"
else
    python -m Hybrid_PINN_ParisRUL.common.dataset_v2 --precompute-splits \
        --paris-labels "${_PARIS}" \
        --cache-dir    "${_CACHE}"
fi

# ===========================================================
# Stage 3 — Train Hybrid track (multi-GPU DDP)
# Bypass torchrun when only 1 GPU is available: torchrun initialises a
# c10d TCPStore rendezvous even with nproc=1, which segfaults in container
# environments (Lightning AI, some Singularity setups). python -m runs the
# same code without any distributed plumbing.
# ===========================================================
echo "[sbatch] Stage 3: training Hybrid track (${NGPU} GPUs)..."
cd "${REPO_ROOT}"
if [ "${NGPU}" -le 1 ]; then
    echo "[sbatch] 1 GPU — bypassing torchrun for Hybrid track"
    python -m Hybrid_PINN_ParisRUL.track_hybrid.train \
        --epochs  "${HYBRID_EPOCHS:-100}" \
        --batch   "${HYBRID_BATCH:-4096}" \
        --lr      "${HYBRID_LR:-8e-4}" \
        --accum-steps "${ACCUM_STEPS:-1}" \
        --paris-labels "${_PARIS}" \
        --shared-test  "${_NPZ}"
else
    torchrun \
        --standalone \
        --nproc_per_node="${NGPU}" \
        "${REPO_ROOT}/Hybrid_PINN_ParisRUL/track_hybrid/train.py" \
        --epochs  "${HYBRID_EPOCHS:-100}" \
        --batch   "${HYBRID_BATCH:-4096}" \
        --lr      "${HYBRID_LR:-8e-4}" \
        --accum-steps "${ACCUM_STEPS:-1}" \
        --paris-labels "${_PARIS}" \
        --shared-test  "${_NPZ}"
fi

# ===========================================================
# Stage 4 — Train PINN track (single GPU — always bypasses torchrun)
# ===========================================================
echo "[sbatch] Stage 4: training PINN track (1 GPU)..."
cd "${REPO_ROOT}"
python -m Hybrid_PINN_ParisRUL.track_pinn.train \
    --epochs  "${PINN_EPOCHS:-100}" \
    --batch   "${PINN_BATCH:-4096}" \
    --lr      "${PINN_LR:-4e-4}" \
    --paris-labels "${_PARIS}"

# ===========================================================
# Stage 5 — Distil + export (optional)
# ===========================================================
if [ "${RUN_FUSION:-0}" = "1" ]; then
    echo "[sbatch] Stage 5: distillation + export..."
    python "${REPO_ROOT}/Hybrid_PINN_ParisRUL/track_fusion/distill.py" \
        --hybrid "${OUTPUT_DIR}/Hybrid_PINN_ParisRUL/results/hybrid/best_model.pt" \
        --pinn   "${OUTPUT_DIR}/Hybrid_PINN_ParisRUL/results/pinn/best_model.pt" \
        --paris-labels "${OUTPUT_DIR}/Hybrid_PINN_ParisRUL/results/labels/labels_paris.parquet" \
        --epochs "${FUSION_EPOCHS:-30}" \
        --export-edge \
        --export-cloud
fi

echo "[sbatch] All stages complete."
