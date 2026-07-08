#!/usr/bin/env bash
# run_v3.sh — v3 physics-anchored RUL track orchestrator (stages 10 → 15).
#
#   bash scripts/run_v3.sh              resume-safe full run
#   bash scripts/run_v3.sh --fresh      wipe all v3 state, start over
#
# Every stage is sentinel-guarded; kill/resubmit at any point and the run
# continues from the last incomplete stage (training additionally resumes
# from checkpoint_last.pt inside stage 11).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$HOME}"
HYBRID="${HYBRID:-$ROOT/Hybrid_PINN_ParisRUL}"
export ROOT HYBRID

if [ "${1:-}" = "--fresh" ]; then
    echo "[run_v3] --fresh: wiping $HYBRID/results/v3"
    rm -rf "$HYBRID/results/v3"
fi

T0=$(date +%s)
for stage in 10_v3_physfeat 11_v3_train 12_v3_calibrate \
             13_v3_propagation 14_v3_export 15_v3_validate; do
    echo ""
    echo "──────────────────────────────────────────────"
    echo "[run_v3] stage $stage  ($((($(date +%s) - T0) / 60)) min elapsed)"
    echo "──────────────────────────────────────────────"
    bash "$SCRIPT_DIR/$stage.sh"
done

echo ""
echo "[run_v3] ALL STAGES OK in $((($(date +%s) - T0) / 60)) min"
echo "[run_v3] deployable model : $HYBRID/results/v3/export/damage_net_fp32.onnx"
echo "[run_v3] expert report    : $HYBRID/results/v3/expert_report.md"
