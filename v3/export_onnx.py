"""export_onnx.py -- Stage 14: DamageNet -> portable ONNX (fp32 + INT8).

Exports the trained checkpoint to
    results/v3/export/damage_net_fp32.onnx      (parity-checked vs torch)
    results/v3/export/damage_net_int8.onnx      (onnxruntime dynamic quant)
    results/v3/export/model_meta.json           (phys_norm, sigma_scale, dims)

The ONNX graph has a dynamic batch axis and tuple outputs
(fault_logits, corn_logits, log_a_mu, log_a_log_var) so the torch-free
wrapper (inference_v3.py) needs no dict-name guessing.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from Hybrid_PINN_ParisRUL.v3.damage_net import DamageNet  # noqa: E402
from Hybrid_PINN_ParisRUL.v3.severity_axis import N_LEVELS  # noqa: E402
from Hybrid_PINN_ParisRUL.v3.signal_physics import PHYS_DIM  # noqa: E402
from Hybrid_PINN_ParisRUL.v3.train import RESULTS_DIR, V3_DIR  # noqa: E402

EXPORT_DIR = V3_DIR / "export"


class _TupleWrap(torch.nn.Module):
    def __init__(self, net: DamageNet) -> None:
        super().__init__()
        self.net = net

    def forward(self, x_raw, x_feat, x_phys):
        out = self.net(x_raw, x_feat, x_phys)
        return (out["fault_logits"], out["corn_logits"],
                out["log_a_mu"], out["log_a_log_var"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=str(RESULTS_DIR / "best_model.pt"))
    parser.add_argument("--out-dir", default=str(EXPORT_DIR))
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sentinel = out_dir / ".export_complete"
    if sentinel.exists():
        print(f"[v3:export] SKIP -- {sentinel} exists (rm to re-export)")
        return

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = DamageNet(hidden=ck.get("hidden", 128),
                      dropout=ck.get("dropout", 0.15))
    model.load_state_dict(ck["state_dict"])
    model.eval()
    wrapped = _TupleWrap(model).eval()

    B = 4
    ex = (torch.randn(B, 5, 2048), torch.randn(B, 160),
          torch.randn(B, PHYS_DIM))
    fp32_path = out_dir / "damage_net_fp32.onnx"
    torch.onnx.export(
        wrapped, ex, str(fp32_path), opset_version=args.opset,
        input_names=["x_raw", "x_feat", "x_phys"],
        output_names=["fault_logits", "corn_logits", "log_a_mu",
                      "log_a_log_var"],
        dynamic_axes={n: {0: "batch"} for n in
                      ("x_raw", "x_feat", "x_phys", "fault_logits",
                       "corn_logits", "log_a_mu", "log_a_log_var")},
        dynamo=False,
    )
    print(f"[v3:export] fp32 -> {fp32_path} "
          f"({fp32_path.stat().st_size/1e6:.2f} MB)")

    # ── Parity check fp32 ────────────────────────────────────────────────
    import onnxruntime as ort
    sess = ort.InferenceSession(str(fp32_path),
                                providers=["CPUExecutionProvider"])
    feeds = {"x_raw": ex[0].numpy(), "x_feat": ex[1].numpy(),
             "x_phys": ex[2].numpy()}
    t0 = time.perf_counter()
    ort_out = sess.run(None, feeds)
    ms = (time.perf_counter() - t0) * 1e3 / B
    with torch.no_grad():
        ref = wrapped(*ex)
    max_diff = max(float(np.abs(o - r.numpy()).max())
                   for o, r in zip(ort_out, ref))
    print(f"[v3:export] fp32 parity max|diff|={max_diff:.2e} "
          f"({ms:.2f} ms/window CPU)")
    assert max_diff < 1e-4, "fp32 ONNX diverges from torch"

    # ── INT8 dynamic quantisation ────────────────────────────────────────
    from onnxruntime.quantization import QuantType, quantize_dynamic
    int8_path = out_dir / "damage_net_int8.onnx"
    quantize_dynamic(str(fp32_path), str(int8_path),
                     weight_type=QuantType.QInt8)
    sess8 = ort.InferenceSession(str(int8_path),
                                 providers=["CPUExecutionProvider"])
    out8 = sess8.run(None, feeds)
    int8_diff = max(float(np.abs(o8 - r.numpy()).max())
                    for o8, r in zip(out8, ref))
    t0 = time.perf_counter()
    sess8.run(None, feeds)
    ms8 = (time.perf_counter() - t0) * 1e3 / B
    print(f"[v3:export] int8 -> {int8_path} "
          f"({int8_path.stat().st_size/1e6:.2f} MB) "
          f"max|diff|={int8_diff:.3f} ({ms8:.2f} ms/window CPU)")

    # ── Metadata for the torch-free wrapper ──────────────────────────────
    calib_path = V3_DIR / "calibration.json"
    calib = json.loads(calib_path.read_text()) if calib_path.exists() else {}
    meta = {
        "hidden": ck.get("hidden", 128),
        "n_levels": N_LEVELS, "phys_dim": PHYS_DIM,
        "phys_norm_mean": ck.get("phys_norm_mean"),
        "phys_norm_std": ck.get("phys_norm_std"),
        "sigma_scale": calib.get("sigma_scale", 1.0),
        "sigma_ln_c": calib.get("sigma_ln_c", 0.35),
        "val_metrics": ck.get("val_metrics", {}),
        "int8_parity_max_diff": int8_diff,
        "opset": args.opset,
    }
    (out_dir / "model_meta.json").write_text(json.dumps(meta, indent=2))
    sentinel.touch()
    print(f"[v3:export] DONE -> {out_dir}")


if __name__ == "__main__":
    main()
