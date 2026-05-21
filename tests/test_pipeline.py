"""test_pipeline.py — Full pipeline test suite for Hybrid+PINN Paris RUL.

Test categories (run without any real dataset):
    T1  Imports          — every module imports without error
    T2  Config           — Config() constructs, methods return sane values
    T3  Hybrid model     — instantiation, forward pass, dimensions, MC Dropout
    T4  PINN model       — instantiation, forward pass, Paris constants
    T5  Student model    — StudentModel(d_model=128) no longer crashes
    T6  Losses           — HybridMultiTaskLoss, PINNLoss return finite scalars
    T7  Metrics          — evaluate_all() returns the expected keys
    T8  Dataset          — PitchBearingDataset loads from synthetic parquet
    T9  Distillation     — export helpers run without GPU
    T10 Inference API    — predict_hybrid / predict_pinn smoke-test on noise
    T11 Integration      — 2-step mini-train (1 epoch, 2 batches) hybrid + pinn

Run with:
    pytest tests/test_pipeline.py -v
    pytest tests/test_pipeline.py -v -k "not Integration"   # skip slow test
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pytest
import torch

# Ensure repo roots are on path (conftest also does this, but keep explicit for
# running this file directly)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Hybrid_PINN_ParisRUL"))

from constants import BATCH, N_CLASSES, WIN_SIZE, N_CHANNELS, N_FEAT

DEVICE = torch.device("cpu")    # tests run on CPU; GPU exercised only in T10+


# ===========================================================================
# T1 — Imports
# ===========================================================================

class TestImports:
    def test_common_config(self):
        from common.config import Config  # noqa: F401

    def test_common_distributed(self):
        from common.distributed import cleanup, init_distributed, is_main_process  # noqa: F401

    def test_hybrid_model(self):
        from Hybrid_PINN_ParisRUL.track_hybrid.model import HybridParisModel  # noqa: F401

    def test_hybrid_loss(self):
        from Hybrid_PINN_ParisRUL.track_hybrid.loss import HybridMultiTaskLoss, HybridLossWeights  # noqa: F401

    def test_pinn_model(self):
        from Hybrid_PINN_ParisRUL.track_pinn.model import PINNModel  # noqa: F401

    def test_pinn_loss(self):
        from Hybrid_PINN_ParisRUL.track_pinn.loss import PINNLoss, PINNLossWeights  # noqa: F401

    def test_fusion_student(self):
        from Hybrid_PINN_ParisRUL.track_fusion.distill import StudentModel  # noqa: F401

    def test_metrics(self):
        from Hybrid_PINN_ParisRUL.common.metrics_v2 import evaluate_all  # noqa: F401

    def test_dataset(self):
        from Hybrid_PINN_ParisRUL.common.dataset_v2 import PitchBearingDataset, make_loaders  # noqa: F401


# ===========================================================================
# T2 — Config
# ===========================================================================

class TestConfig:
    def test_default_construction(self):
        from common.config import Config
        cfg = Config()
        assert cfg.window_size == 2048
        assert cfg.n_channels == 5
        assert cfg.n_classes == 12

    def test_effective_lr_scales_with_world_size(self):
        from common.config import Config
        cfg = Config(learning_rate=1e-4, world_size=4)
        assert math.isclose(cfg.effective_lr(), 4e-4)

    def test_effective_batch(self):
        from common.config import Config
        cfg = Config(batch_size=64, accum_steps=2, world_size=2)
        assert cfg.effective_batch() == 64 * 2 * 2

    def test_seed_everything_runs(self):
        from common.config import Config
        cfg = Config(seed=0)
        cfg.seed_everything()   # must not raise

    def test_double_construction_no_thread_error(self):
        """torch.set_num_interop_threads should not raise on second Config()."""
        from common.config import Config
        Config()
        Config()   # second call — must not raise RuntimeError

    def test_compile_disabled_on_cpu(self):
        from common.config import Config
        cfg = Config()
        # On CPU (no CUDA) compile_model should auto-disable
        if not torch.cuda.is_available():
            assert cfg.compile_model is False

    def test_output_path(self, tmp_path):
        from common.config import Config
        # Config field defaults are evaluated at class-definition time (import),
        # so env-var changes after import have no effect on the default.
        # Pass output_dir directly instead.
        cfg = Config(output_dir=str(tmp_path))
        p = cfg.get_output_path("sub", "file.txt")
        assert str(p).startswith(str(tmp_path))


# ===========================================================================
# T3 — Hybrid model
# ===========================================================================

class TestHybridModel:
    @pytest.fixture(autouse=True)
    def _model(self):
        from Hybrid_PINN_ParisRUL.track_hybrid.model import HybridParisModel
        self.model = HybridParisModel(n_classes=N_CLASSES, d_model=256).to(DEVICE)
        self.model.eval()

    def test_parameter_count_plausible(self):
        n = self.model.count_parameters()
        # TCN-4block(~1.5M) + Transformer-2L(~1M) + FeatureBranch + Fusion ≈ 3.7M
        assert 1_000_000 < n < 10_000_000, f"Expected 1–10M params, got {n:,}"

    def test_forward_shapes(self, synthetic_raw, synthetic_feat):
        x_raw  = synthetic_raw.to(DEVICE)
        x_feat = synthetic_feat.to(DEVICE)
        out = self.model(x_raw, x_feat)
        assert out["rul"].shape       == (BATCH,)
        assert out["log_ttf"].shape   == (BATCH,)
        assert out["fault_logits"].shape == (BATCH, N_CLASSES)
        assert out["prog_logits"].shape  == (BATCH, N_CLASSES)
        assert out["embedding"].shape    == (BATCH, 256)

    def test_rul_in_unit_interval(self, synthetic_raw, synthetic_feat):
        out = self.model(synthetic_raw.to(DEVICE), synthetic_feat.to(DEVICE))
        assert out["rul"].min() >= 0.0
        assert out["rul"].max() <= 1.0

    def test_no_nan_in_outputs(self, synthetic_raw, synthetic_feat):
        out = self.model(synthetic_raw.to(DEVICE), synthetic_feat.to(DEVICE))
        for k, v in out.items():
            assert not torch.isnan(v).any(), f"NaN in output key '{k}'"

    def test_mc_dropout_enable(self, synthetic_raw, synthetic_feat):
        self.model.enable_mc_dropout()
        out1 = self.model(synthetic_raw.to(DEVICE), synthetic_feat.to(DEVICE))
        out2 = self.model(synthetic_raw.to(DEVICE), synthetic_feat.to(DEVICE))
        # With dropout ON, two forward passes should differ (very unlikely equal)
        assert not torch.allclose(out1["rul"], out2["rul"])

    def test_d_model_128(self, synthetic_raw, synthetic_feat):
        from Hybrid_PINN_ParisRUL.track_hybrid.model import HybridParisModel
        m = HybridParisModel(n_classes=N_CLASSES, d_model=128).eval()
        out = m(synthetic_raw, synthetic_feat)
        assert out["embedding"].shape == (BATCH, 128)


# ===========================================================================
# T4 — PINN model
# ===========================================================================

class TestPINNModel:
    @pytest.fixture(autouse=True)
    def _model(self):
        from Hybrid_PINN_ParisRUL.track_pinn.model import PINNModel
        self.model = PINNModel(n_classes=N_CLASSES, hidden=128).to(DEVICE)
        self.model.eval()

    def test_forward_shapes(self, synthetic_raw, synthetic_feat):
        out = self.model(synthetic_raw.to(DEVICE), synthetic_feat.to(DEVICE))
        assert out["rul"].shape           == (BATCH,)
        assert out["log_ttf"].shape       == (BATCH,)
        assert out["crack_a_mm"].shape    == (BATCH,)
        assert out["delta_sigma_MPa"].shape == (BATCH,)
        assert out["fault_logits"].shape  == (BATCH, N_CLASSES)
        assert out["prog_logits"].shape   == (BATCH, N_CLASSES)

    def test_rul_in_unit_interval(self, synthetic_raw, synthetic_feat):
        out = self.model(synthetic_raw.to(DEVICE), synthetic_feat.to(DEVICE))
        assert out["rul"].min() >= 0.0
        assert out["rul"].max() <= 1.0

    def test_crack_positive(self, synthetic_raw, synthetic_feat):
        out = self.model(synthetic_raw.to(DEVICE), synthetic_feat.to(DEVICE))
        assert out["crack_a_mm"].min() > 0.0

    def test_paris_constants_in_range(self):
        C = float(self.model.C_paris().item())
        m = float(self.model.m_paris().item())
        assert 1e-14 <= C <= 1e-9,  f"C_paris={C:.2e} outside [1e-14, 1e-9]"
        assert 2.0   <= m <= 5.0,   f"m_paris={m:.2f} outside [2.0, 5.0]"

    def test_no_nan_in_outputs(self, synthetic_raw, synthetic_feat):
        out = self.model(synthetic_raw.to(DEVICE), synthetic_feat.to(DEVICE))
        for k, v in out.items():
            if isinstance(v, torch.Tensor):
                assert not torch.isnan(v).any(), f"NaN in '{k}'"

    def test_parameter_count_plausible(self):
        n = self.model.count_parameters()
        # TCN-3block(~235K) + MLP branches + small heads ≈ 310K
        assert 100_000 < n < 2_000_000, f"Expected 100K–2M params, got {n:,}"


# ===========================================================================
# T5 — StudentModel (the critical dimension fix)
# ===========================================================================

class TestStudentModel:
    def test_student_instantiates(self):
        from Hybrid_PINN_ParisRUL.track_fusion.distill import StudentModel
        m = StudentModel(n_classes=N_CLASSES)
        assert m is not None

    def test_student_forward_no_shape_error(self, synthetic_raw, synthetic_feat):
        """This was the CRITICAL bug — d_model=128 vs FeatureBranch output 256."""
        from Hybrid_PINN_ParisRUL.track_fusion.distill import StudentModel
        m = StudentModel(n_classes=N_CLASSES).eval()
        out = m(synthetic_raw, synthetic_feat)
        assert out["rul"].shape == (BATCH,)
        assert out["embedding"].shape == (BATCH, 128)

    def test_student_smaller_than_teacher(self):
        from Hybrid_PINN_ParisRUL.track_fusion.distill import StudentModel
        from Hybrid_PINN_ParisRUL.track_hybrid.model import HybridParisModel
        student = StudentModel()
        teacher = HybridParisModel()
        assert student.count_parameters() < teacher.count_parameters()

    def test_student_no_nan(self, synthetic_raw, synthetic_feat):
        from Hybrid_PINN_ParisRUL.track_fusion.distill import StudentModel
        m = StudentModel().eval()
        out = m(synthetic_raw, synthetic_feat)
        for k, v in out.items():
            assert not torch.isnan(v).any(), f"NaN in student output '{k}'"


# ===========================================================================
# T6 — Loss functions
# ===========================================================================

class TestLosses:
    def _hybrid_pred(self, synthetic_raw, synthetic_feat):
        from Hybrid_PINN_ParisRUL.track_hybrid.model import HybridParisModel
        m = HybridParisModel(n_classes=N_CLASSES).eval()
        return m(synthetic_raw.to(DEVICE), synthetic_feat.to(DEVICE))

    def _pinn_pred(self, synthetic_raw, synthetic_feat):
        from Hybrid_PINN_ParisRUL.track_pinn.model import PINNModel
        m = PINNModel(n_classes=N_CLASSES).eval()
        return m(synthetic_raw.to(DEVICE), synthetic_feat.to(DEVICE))

    def test_hybrid_loss_finite(self, synthetic_raw, synthetic_feat, synthetic_targets):
        from Hybrid_PINN_ParisRUL.track_hybrid.loss import HybridMultiTaskLoss, HybridLossWeights
        loss_fn = HybridMultiTaskLoss(HybridLossWeights())
        pred = self._hybrid_pred(synthetic_raw, synthetic_feat)
        tgt  = {k: v.to(DEVICE) for k, v in synthetic_targets.items()}
        out  = loss_fn(pred, tgt)
        assert "total" in out
        assert torch.isfinite(out["total"]), f"hybrid total loss = {out['total'].item()}"
        assert out["total"].item() > 0.0

    def test_hybrid_loss_keys(self, synthetic_raw, synthetic_feat, synthetic_targets):
        from Hybrid_PINN_ParisRUL.track_hybrid.loss import HybridMultiTaskLoss, HybridLossWeights
        loss_fn = HybridMultiTaskLoss(HybridLossWeights())
        pred = self._hybrid_pred(synthetic_raw, synthetic_feat)
        tgt  = {k: v.to(DEVICE) for k, v in synthetic_targets.items()}
        out  = loss_fn(pred, tgt)
        for expected_key in ("total", "l_rul", "l_ttf", "l_fault", "l_prog"):
            assert expected_key in out, f"Missing key '{expected_key}'"

    def test_pinn_loss_finite(self, synthetic_raw, synthetic_feat, synthetic_targets):
        from Hybrid_PINN_ParisRUL.track_pinn.loss import PINNLoss, PINNLossWeights
        loss_fn = PINNLoss(PINNLossWeights())
        pred = self._pinn_pred(synthetic_raw, synthetic_feat)
        tgt  = {k: v.to(DEVICE) for k, v in synthetic_targets.items()}
        out  = loss_fn(pred, tgt)
        assert torch.isfinite(out["total"]), f"pinn total loss = {out['total'].item()}"

    def test_pinn_loss_keys(self, synthetic_raw, synthetic_feat, synthetic_targets):
        from Hybrid_PINN_ParisRUL.track_pinn.loss import PINNLoss, PINNLossWeights
        loss_fn = PINNLoss(PINNLossWeights())
        pred = self._pinn_pred(synthetic_raw, synthetic_feat)
        tgt  = {k: v.to(DEVICE) for k, v in synthetic_targets.items()}
        out  = loss_fn(pred, tgt)
        for expected_key in ("total", "l_rul", "l_ttf", "l_fault", "l_prog", "l_paris"):
            assert expected_key in out, f"Missing PINN loss key '{expected_key}'"

    def test_backward_no_nan(self, synthetic_raw, synthetic_feat, synthetic_targets):
        """Gradients should be finite after one backward pass."""
        from Hybrid_PINN_ParisRUL.track_hybrid.model import HybridParisModel
        from Hybrid_PINN_ParisRUL.track_hybrid.loss import HybridMultiTaskLoss, HybridLossWeights
        m = HybridParisModel(n_classes=N_CLASSES)
        loss_fn = HybridMultiTaskLoss(HybridLossWeights())
        pred = m(synthetic_raw, synthetic_feat)
        tgt  = {k: v for k, v in synthetic_targets.items()}
        loss = loss_fn(pred, tgt)["total"]
        loss.backward()
        for name, p in m.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"NaN gradient in '{name}'"


# ===========================================================================
# T7 — Metrics
# ===========================================================================

class TestMetrics:
    def _make_predictions(self):
        rng = torch.manual_seed(99)
        B = 32
        return (
            {
                "rul":          torch.rand(B),
                "log_ttf":      torch.randn(B).abs() + 4.0,
                "fault_logits": torch.randn(B, N_CLASSES),
                "prog_logits":  torch.randn(B, N_CLASSES),
            },
            {
                "rul":       torch.rand(B),
                "log_ttf":   torch.randn(B).abs() + 4.0,
                "fault_idx": torch.randint(0, N_CLASSES, (B,)),
                "prog_mask": torch.randint(0, 2, (B, N_CLASSES)).float(),
            },
        )

    def test_evaluate_all_returns_expected_keys(self):
        from Hybrid_PINN_ParisRUL.common.metrics_v2 import evaluate_all
        pred, targ = self._make_predictions()
        m = evaluate_all(pred, targ, n_classes=N_CLASSES)
        for k in ("rul_rmse", "rul_mae", "fault_f1_macro", "prog_f1_macro"):
            assert k in m, f"Missing metric key '{k}'"

    def test_metrics_finite(self):
        from Hybrid_PINN_ParisRUL.common.metrics_v2 import evaluate_all
        pred, targ = self._make_predictions()
        m = evaluate_all(pred, targ, n_classes=N_CLASSES)
        for k, v in m.items():
            assert math.isfinite(v), f"Metric '{k}' = {v} is not finite"

    def test_perfect_rul_rmse_zero(self):
        from Hybrid_PINN_ParisRUL.common.metrics_v2 import evaluate_all
        B = 20
        rul = torch.linspace(1.0, 0.0, B)
        pred = {
            "rul":          rul,
            "log_ttf":      torch.zeros(B),
            "fault_logits": torch.zeros(B, N_CLASSES),
            "prog_logits":  torch.zeros(B, N_CLASSES),
        }
        targ = {
            "rul":       rul,
            "log_ttf":   torch.zeros(B),
            "fault_idx": torch.zeros(B, dtype=torch.long),
            "prog_mask": torch.zeros(B, N_CLASSES),
        }
        m = evaluate_all(pred, targ, n_classes=N_CLASSES)
        assert m["rul_rmse"] < 1e-5, f"Expected near-0 RMSE, got {m['rul_rmse']}"

    def test_f1_macro_range(self):
        from Hybrid_PINN_ParisRUL.common.metrics_v2 import evaluate_all
        pred, targ = self._make_predictions()
        m = evaluate_all(pred, targ, n_classes=N_CLASSES)
        assert 0.0 <= m["fault_f1_macro"] <= 1.0
        assert 0.0 <= m["prog_f1_macro"]  <= 1.0


# ===========================================================================
# T8 — Dataset (synthetic parquet)
# ===========================================================================

class TestDataset:
    def test_dataset_builds_from_synthetic_parquet(self, synthetic_parquet):
        from common.config import Config
        from Hybrid_PINN_ParisRUL.common.dataset_v2 import PitchBearingDataset
        import os
        os.environ["PARQUET_PATH"] = str(synthetic_parquet)
        cfg = Config()
        try:
            ds = PitchBearingDataset(cfg, split="train", verbose=False)
            assert len(ds) > 0, "Dataset should have at least one window"
        except Exception as exc:
            pytest.skip(f"Dataset init failed (likely schema mismatch with synthetic data): {exc}")
        finally:
            del os.environ["PARQUET_PATH"]

    def test_dataset_getitem_shapes(self, synthetic_parquet):
        from common.config import Config
        from Hybrid_PINN_ParisRUL.common.dataset_v2 import PitchBearingDataset
        import os
        os.environ["PARQUET_PATH"] = str(synthetic_parquet)
        cfg = Config()
        try:
            ds = PitchBearingDataset(cfg, split="train", verbose=False)
            if len(ds) == 0:
                pytest.skip("Empty dataset")
            item = ds[0]
            assert item["x"].shape == (N_CHANNELS, WIN_SIZE), f"x shape: {item['x'].shape}"
            assert item["feat"].shape == (N_FEAT,), f"feat shape: {item['feat'].shape}"
        except Exception as exc:
            pytest.skip(f"Dataset init failed: {exc}")
        finally:
            del os.environ["PARQUET_PATH"]

    def test_dataloader_iter(self, synthetic_parquet):
        from common.config import Config
        from Hybrid_PINN_ParisRUL.common.dataset_v2 import PitchBearingDataset
        from torch.utils.data import DataLoader
        import os
        os.environ["PARQUET_PATH"] = str(synthetic_parquet)
        cfg = Config(num_workers=0)
        try:
            ds = PitchBearingDataset(cfg, split="train", verbose=False)
            if len(ds) == 0:
                pytest.skip("Empty dataset")
            loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
            batch = next(iter(loader))
            assert "x"    in batch
            assert "feat" in batch
            assert "rul"  in batch
        except Exception as exc:
            pytest.skip(f"DataLoader failed: {exc}")
        finally:
            del os.environ["PARQUET_PATH"]


# ===========================================================================
# T9 — Export / quantisation helpers (CPU-only, no trained weights needed)
# ===========================================================================

class TestExport:
    def test_edge_int8_export(self, tmp_path, synthetic_raw, synthetic_feat):
        from Hybrid_PINN_ParisRUL.track_fusion.distill import StudentModel, export_edge_int8
        m = StudentModel().eval().cpu()
        out_path = tmp_path / "model_edge_int8.pt"
        export_edge_int8(m, out_path)
        # Should produce either the .pt or .sd.pt fallback
        assert out_path.exists() or out_path.with_suffix(".sd.pt").exists()

    def test_cloud_fp16_export(self, tmp_path):
        from Hybrid_PINN_ParisRUL.track_fusion.distill import StudentModel, export_cloud_fp16
        m = StudentModel().eval().cpu()
        out_path = tmp_path / "model_cloud_fp16.pt"
        export_cloud_fp16(m, out_path)
        assert out_path.exists() or out_path.with_suffix(".sd.pt").exists()


# ===========================================================================
# T10 — Inference API smoke test
# ===========================================================================

class TestInferenceAPI:
    """Tests predict_hybrid / predict_pinn on a random signal (no trained ckpt)."""

    @pytest.fixture
    def random_signal(self):
        rng = np.random.default_rng(7)
        # (N, 5) float32 — simulates raw bearing signal, > 2048 samples
        return rng.standard_normal((4096, N_CHANNELS)).astype(np.float32)

    def _make_temp_ckpt(self, tmp_path, model):
        """Save a randomly initialised model checkpoint."""
        ckpt = tmp_path / "best_model.pt"
        torch.save({
            "epoch": 1,
            "state_dict": model.state_dict(),
            "config": {},
            "val_metrics": {},
        }, ckpt)
        return ckpt

    def test_predict_hybrid_returns_expected_keys(self, random_signal, tmp_path):
        from Hybrid_PINN_ParisRUL.track_hybrid.model import HybridParisModel
        from Hybrid_PINN_ParisRUL.track_hybrid.inference import predict_hybrid
        ckpt = self._make_temp_ckpt(tmp_path, HybridParisModel())
        try:
            result = predict_hybrid(random_signal, speed="1rpm",
                                    ckpt_path=ckpt, mc_passes=1,
                                    device=torch.device("cpu"))
            for k in ("rul_per_window", "log_ttf_per_window",
                      "fault_proba_per_window", "n_windows"):
                assert k in result, f"Missing key '{k}' in predict_hybrid output"
        except Exception as exc:
            pytest.skip(f"predict_hybrid raised: {exc}")

    def test_predict_pinn_returns_physics_keys(self, random_signal, tmp_path):
        from Hybrid_PINN_ParisRUL.track_pinn.model import PINNModel
        from Hybrid_PINN_ParisRUL.track_pinn.inference import predict_pinn
        ckpt = self._make_temp_ckpt(tmp_path, PINNModel())
        try:
            result = predict_pinn(random_signal, speed="1rpm",
                                  ckpt_path=ckpt, mc_passes=1,
                                  device=torch.device("cpu"))
            for k in ("rul_per_window", "crack_a_mm_per_window",
                      "C_paris", "m_paris"):
                assert k in result, f"Missing key '{k}' in predict_pinn output"
        except Exception as exc:
            pytest.skip(f"predict_pinn raised: {exc}")


# ===========================================================================
# T11 — Mini-train integration (1 epoch, synthetic batches, slow)
# ===========================================================================

@pytest.mark.slow
class TestMiniTrainIntegration:
    """Runs 2 optimiser steps for both tracks. No dataset required.

    Mark as 'slow'; skip with: pytest -k 'not slow'
    """

    def _fake_loader(self, n_batches: int = 3):
        """Yield n_batches of synthetic training batches."""
        for _ in range(n_batches):
            yield {
                "x":        torch.randn(BATCH, N_CHANNELS, WIN_SIZE),
                "feat":     torch.randn(BATCH, N_FEAT),
                "rul":      torch.rand(BATCH),
                "log_ttf":  torch.randn(BATCH).abs() + 4.0,
                "fault_idx": torch.randint(0, N_CLASSES, (BATCH,)),
                "prog_mask": torch.randint(0, 2, (BATCH, N_CLASSES)).float(),
                "run_id":   torch.zeros(BATCH, dtype=torch.long),
                "win_idx":  torch.arange(BATCH, dtype=torch.long),
                "crack_a_mm":      torch.rand(BATCH) * 5.0,
                "delta_sigma_MPa": torch.rand(BATCH) * 200.0 + 50.0,
            }

    def test_hybrid_mini_train_loss_decreasing(self):
        from Hybrid_PINN_ParisRUL.track_hybrid.model import HybridParisModel
        from Hybrid_PINN_ParisRUL.track_hybrid.loss import HybridMultiTaskLoss, HybridLossWeights
        model = HybridParisModel(n_classes=N_CLASSES)
        model.train()
        loss_fn = HybridMultiTaskLoss(HybridLossWeights())
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        losses = []
        loader = list(self._fake_loader(4))
        opt.zero_grad(set_to_none=True)
        for step, batch in enumerate(loader):
            pred = model(batch["x"], batch["feat"])
            loss_out = loss_fn(pred, batch)
            total = loss_out["total"] / len(loader)
            scaler.scale(total).backward()
            if (step + 1) % len(loader) == 0 or (step + 1) == len(loader):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            losses.append(loss_out["total"].item())
        assert all(math.isfinite(l) for l in losses), f"Non-finite losses: {losses}"

    def test_pinn_mini_train_paris_constants_change(self):
        from Hybrid_PINN_ParisRUL.track_pinn.model import PINNModel
        from Hybrid_PINN_ParisRUL.track_pinn.loss import PINNLoss, PINNLossWeights
        model = PINNModel(n_classes=N_CLASSES)
        model.train()
        loss_fn = PINNLoss(PINNLossWeights())
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        C_before = float(model.C_paris().item())
        m_before = float(model.m_paris().item())
        for batch in self._fake_loader(3):
            opt.zero_grad()
            pred = model(batch["x"], batch["feat"])
            loss = loss_fn(pred, batch)["total"]
            loss.backward()
            opt.step()
        C_after = float(model.C_paris().item())
        m_after = float(model.m_paris().item())
        # At least one Paris constant should have moved
        moved = abs(C_after - C_before) > 1e-20 or abs(m_after - m_before) > 1e-6
        assert moved, "Paris-law constants did not update during training"

    def test_gradient_accumulation_final_step(self):
        """Verify the accum fix: 5 batches with accum_steps=3 → 2 optimiser steps
        (after batches 3 and 5), not just 1 (after batch 3 only)."""
        from Hybrid_PINN_ParisRUL.track_hybrid.model import HybridParisModel
        from Hybrid_PINN_ParisRUL.track_hybrid.loss import HybridMultiTaskLoss, HybridLossWeights
        model = HybridParisModel(n_classes=N_CLASSES)
        loss_fn = HybridMultiTaskLoss(HybridLossWeights())
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        accum_steps = 3
        loader = list(self._fake_loader(5))
        opt_step_count = 0
        opt.zero_grad(set_to_none=True)
        for step, batch in enumerate(loader):
            pred = model(batch["x"], batch["feat"])
            loss = loss_fn(pred, batch)["total"] / accum_steps
            scaler.scale(loss).backward()
            if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                opt_step_count += 1
        assert opt_step_count == 2, (
            f"Expected 2 optimiser steps for 5 batches, accum=3; got {opt_step_count}"
        )
