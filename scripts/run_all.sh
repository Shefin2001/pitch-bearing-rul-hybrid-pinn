#!/usr/bin/env bash
# run_all.sh — Master orchestrator. Chronological 00 → 07 pipeline.
# Checks which stages are already done and starts only from what is needed.
# Re-run anytime — completed stages are never re-executed.
set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"

export ROOT="${ROOT:-$HOME}"
export HYBRID="${HYBRID:-$ROOT/Hybrid_PINN_ParisRUL}"

# ── Artifact / sentinel paths ────────────────────────────────────────────────
FPT_OUT="$HYBRID/results/labels/labels_fpt.parquet"
PARIS_OUT="$HYBRID/results/labels/labels_paris.parquet"
NPZ_OUT="$HYBRID/results/test_index/test_windows.npz"
HYBRID_CKPT="$HYBRID/results/hybrid/best_model.pt"
HYBRID_SENTINEL="$HYBRID/results/hybrid/.training_complete"
PINN_CKPT="$HYBRID/results/pinn/best_model.pt"
PINN_SENTINEL="$HYBRID/results/pinn/.training_complete"
FUSE_SENTINEL="$HYBRID/results/fusion/.fuse_complete"
EVAL_CSV="$HYBRID/results/comparison_v2.csv"

# ── Stage completion checks ──────────────────────────────────────────────────
stage01_done()   { [ -f "$FPT_OUT" ] && [ -f "$PARIS_OUT" ]; }
stage02_done()   { [ -f "$NPZ_OUT" ]; }
stage03_done()   { [ -f "$HYBRID_SENTINEL" ]; }
stage03_partial(){ [ -f "$HYBRID_CKPT" ] && ! [ -f "$HYBRID_SENTINEL" ]; }
stage04_done()   { [ -f "$PINN_SENTINEL" ]; }
stage04_partial(){ [ -f "$PINN_CKPT" ]   && ! [ -f "$PINN_SENTINEL" ]; }
stage05_done()   { [ -f "$FUSE_SENTINEL" ]; }
stage06_done()   { [ -f "$EVAL_CSV" ]; }

# ── Status dashboard ─────────────────────────────────────────────────────────
print_status() {
    echo "┌─────────────────────────────────────────────────────────┐"
    echo "│  Stage  │ Status                                        │"
    echo "├─────────────────────────────────────────────────────────┤"

    if stage01_done; then
        echo "│  01     │ DONE    — labels (fpt + paris)                │"
    else
        echo "│  01     │ PENDING — build labels                        │"
    fi

    if stage02_done; then
        echo "│  02     │ DONE    — shared test index (91 k windows)    │"
    else
        echo "│  02     │ PENDING — build dataset index                 │"
    fi

    if stage03_done; then
        echo "│  03     │ DONE    — hybrid training                     │"
    elif stage03_partial; then
        echo "│  03     │ PARTIAL — hybrid checkpoint found (resuming)  │"
    else
        echo "│  03     │ PENDING — hybrid training                     │"
    fi

    if stage04_done; then
        echo "│  04     │ DONE    — pinn training                       │"
    elif stage04_partial; then
        echo "│  04     │ PARTIAL — pinn checkpoint found (resuming)    │"
    else
        echo "│  04     │ PENDING — pinn training                       │"
    fi

    if stage05_done; then
        echo "│  05     │ DONE    — fusion / distillation               │"
    else
        echo "│  05     │ PENDING — fusion / distillation               │"
    fi

    if stage06_done; then
        echo "│  06     │ DONE    — evaluation                          │"
    else
        echo "│  06     │ PENDING — evaluation                          │"
    fi

    echo "│  07     │ ALWAYS  — inference smoke test                │"
    echo "└─────────────────────────────────────────────────────────┘"
}

# ── Header ───────────────────────────────────────────────────────────────────
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  Hybrid_PINN_ParisRUL — FULL PIPELINE                               ║"
echo "╠══════════════════════════════════════════════════════════════════════╣"
echo "║  Started : $(date)"
echo "║  ROOT    : $ROOT"
echo "║  HYBRID  : $HYBRID"
echo "╠══════════════════════════════════════════════════════════════════════╣"
echo "║  PIPELINE MAP                                                        ║"
echo "║  00 setup      : verify deps, create result dirs, git tag            ║"
echo "║  01 labels     : FPT-piecewise RUL (01a) + Paris-law TTF (01b)      ║"
echo "║  02 dataset    : run-level train/val/test split + shared test npz    ║"
echo "║  03 hybrid     : CNN-Transformer + 160-D features, DDP/AMP, 100 ep  ║"
echo "║                  guards: NaN abort, spike skip, diverge stop         ║"
echo "║                  logs:   per-step loss/gnorm/sps + per-epoch ETA     ║"
echo "║                  ckpts:  best_model.pt + every 10 epochs             ║"
echo "║  04 pinn       : Physics-Informed NN (Paris C,m learnable), 100 ep  ║"
echo "║                  same guards + prints C/m each epoch                 ║"
echo "║  05 fusion     : knowledge distillation → edge INT8 + cloud FP16     ║"
echo "║  06 evaluate   : Hybrid vs PINN vs Ensemble on shared test index     ║"
echo "║  07 smoke      : end-to-end inference API test (always runs)         ║"
echo "╠══════════════════════════════════════════════════════════════════════╣"
echo "║  CURRENT STATUS                                                      ║"
print_status
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

# ── Stage runner — skips if done, calls script if needed ────────────────────

cd "$SCRIPTS_DIR"

# Stage 00 — setup (always run: fast dir-create + env check, idempotent)
echo "──────────────── Stage 00 : setup ──────────────────────────"
bash 00_setup.sh

# Stage 01 — build labels
echo "──────────────── Stage 01 : labels ─────────────────────────"
if stage01_done; then
    echo "[run_all] Stage 01 DONE — skipping"
    echo "  fpt   : $FPT_OUT"
    echo "  paris : $PARIS_OUT"
else
    bash 01_build_labels.sh
fi

# Stage 02 — shared test index
echo "──────────────── Stage 02 : dataset index ───────────────────"
if stage02_done; then
    echo "[run_all] Stage 02 DONE — skipping"
    echo "  npz : $NPZ_OUT"
else
    bash 02_build_dataset.sh
fi

# Stage 03 — hybrid training
echo "──────────────── Stage 03 : hybrid training ─────────────────"
if stage03_done; then
    echo "[run_all] Stage 03 DONE — skipping"
    echo "  sentinel : $HYBRID_SENTINEL"
else
    if stage03_partial; then
        echo "[run_all] Stage 03 PARTIAL — checkpoint found, will resume"
    else
        echo "[run_all] Stage 03 PENDING — starting fresh"
    fi
    bash 03_train_hybrid.sh
fi

# Stage 04 — pinn training
echo "──────────────── Stage 04 : pinn training ───────────────────"
if stage04_done; then
    echo "[run_all] Stage 04 DONE — skipping"
    echo "  sentinel : $PINN_SENTINEL"
else
    if stage04_partial; then
        echo "[run_all] Stage 04 PARTIAL — checkpoint found, will resume"
    else
        echo "[run_all] Stage 04 PENDING — starting fresh"
    fi
    bash 04_train_pinn.sh
fi

# Stage 05 — fusion / distillation
echo "──────────────── Stage 05 : fusion ─────────────────────────"
if stage05_done; then
    echo "[run_all] Stage 05 DONE — skipping"
    echo "  sentinel : $FUSE_SENTINEL"
else
    bash 05_fuse_models.sh
fi

# Stage 06 — evaluation
echo "──────────────── Stage 06 : evaluation ─────────────────────"
if stage06_done; then
    echo "[run_all] Stage 06 DONE — skipping"
    echo "  csv : $EVAL_CSV"
else
    bash 06_evaluate.sh
fi

# Stage 07 — smoke test (always runs)
echo "──────────────── Stage 07 : smoke test ─────────────────────"
bash 07_inference_smoke.sh

# ── Final summary ────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Pipeline complete : $(date)"
echo "  Results in        : $HYBRID/results/"
echo "============================================================"
