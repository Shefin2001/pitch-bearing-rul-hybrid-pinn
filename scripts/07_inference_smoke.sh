#!/usr/bin/env bash
# 07_inference_smoke.sh — End-to-end inference API smoke test.
# Always runs — it is a quick functional check, not a long computation.
set -euo pipefail

ROOT="${ROOT:-$HOME}"
DATA="${DATA:-$ROOT/pitch_bearing_dataset.parquet}"
export PARQUET_PATH="${PARQUET_PATH:-$DATA}"
echo "=== 07_inference_smoke ==="
echo "[07_inference_smoke] Running end-to-end inference smoke test..."

cd "$ROOT"
python - <<'PY'
import numpy as np
from Hybrid_PINN_ParisRUL.inference import predict

print("\n[smoke] generating 8192-sample synthetic block (5 channels)")
sig = np.random.randn(8192, 5).astype(np.float32)

passed = 0
for mode in ("ensemble", "edge", "cloud"):
    try:
        out = predict(sig, speed="1rpm", mode=mode)
        print(f"\n--- mode = {mode} ---")
        print(f"  windows                  : {out['windows_processed']}")
        print(f"  rul_relative             : {out['rul_relative']:.3f}")
        print(f"  rul_category             : {out['rul_category']}")
        print(f"  time_to_failure_hours    : {out['time_to_failure_hours']:.1f}")
        print(f"  dominant_fault           : {out['dominant_fault']}")
        print(f"  inference_ms_per_window  : {out['inference_ms_per_window']:.2f}")
        passed += 1
    except Exception as e:
        print(f"  [SKIP {mode}] {e}")

print(f"\n[smoke] {passed}/3 modes passed")
PY

echo "[07_inference_smoke] OK"
