#!/usr/bin/env bash
# run_all.sh — Master orchestrator. Chronological 00 → 07 pipeline.
set -euo pipefail

cd "$(dirname "$0")"

echo "############################################################"
echo "# Hybrid_PINN_ParisRUL — FULL PIPELINE"
echo "# Started: $(date)"
echo "############################################################"

bash 00_setup.sh
bash 01_build_labels.sh
bash 02_build_dataset.sh
bash 03_train_hybrid.sh
bash 04_train_pinn.sh
bash 05_fuse_models.sh
bash 06_evaluate.sh
bash 07_inference_smoke.sh

echo "############################################################"
echo "# DONE: $(date)"
echo "# Results in Hybrid_PINN_ParisRUL/results/"
echo "############################################################"
