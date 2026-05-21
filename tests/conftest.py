"""conftest.py — Shared pytest fixtures for the Hybrid+PINN pipeline tests.

All fixtures that need the real dataset are skipped automatically when
PARQUET_PATH is not set or the file does not exist.
"""
from __future__ import annotations

import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import pytest

# Resolve repo roots so imports work regardless of where pytest is invoked
ROOT = Path(__file__).resolve().parents[2]          # PitchBearing_RUL_DualNN
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Hybrid_PINN_ParisRUL"))
# Make tests/ itself importable (so test_pipeline can import constants.py)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from constants import BATCH, N_CHANNELS, N_CLASSES, N_FEAT, WIN_SIZE  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic tensor helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def synthetic_raw():
    """(BATCH, 5, 2048) float32 tensor — bandpass-like white noise."""
    import torch
    torch.manual_seed(42)
    return torch.randn(BATCH, N_CHANNELS, WIN_SIZE)


@pytest.fixture(scope="session")
def synthetic_feat():
    """(BATCH, 160) float32 tensor — normalised engineered features."""
    import torch
    torch.manual_seed(43)
    return torch.randn(BATCH, N_FEAT)


@pytest.fixture(scope="session")
def synthetic_targets():
    """Dict of target tensors matching the training batch format."""
    import torch
    rng = torch.Generator()
    rng.manual_seed(44)
    return {
        "rul":       torch.rand(BATCH, generator=rng),
        "log_ttf":   torch.randn(BATCH, generator=rng).abs() + 5.0,
        "fault_idx": torch.randint(0, N_CLASSES, (BATCH,), generator=rng),
        "prog_mask": torch.randint(0, 2, (BATCH, N_CLASSES), generator=rng).float(),
        "run_id":    torch.zeros(BATCH, dtype=torch.long),
        "win_idx":   torch.arange(BATCH, dtype=torch.long),
        # PINN extras
        "crack_a_mm":       torch.rand(BATCH, generator=rng) * 5.0,
        "delta_sigma_MPa":  torch.rand(BATCH, generator=rng) * 200.0 + 50.0,
    }


# ---------------------------------------------------------------------------
# Synthetic parquet (tiny — no real dataset needed)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def synthetic_parquet(tmp_path_factory) -> Path:
    """Write a minimal synthetic parquet with 2 speeds × 2 conditions × 5 files.

    Each row is one 2048-sample window flattened into 5 columns.
    """
    try:
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        pytest.skip("pyarrow / pandas not installed")

    rng = np.random.default_rng(0)
    rows = []
    speeds = ["1rpm", "3rpm"]
    conditions = ["Health", "IRC", "ORS"]
    n_files = 3
    n_windows = 6   # small — just enough to exercise data paths

    for speed in speeds:
        for cond in conditions:
            for fidx in range(n_files):
                for widx in range(n_windows):
                    sig = rng.standard_normal((WIN_SIZE, N_CHANNELS)).astype(np.float32)
                    row = {
                        "speed":      speed,
                        "condition":  cond,
                        "file_idx":   fidx,
                        "window_idx": widx,
                        "vib_y_A":    sig[:, 0].tolist(),
                        "vib_x_A":    sig[:, 1].tolist(),
                        "vib_y_B":    sig[:, 2].tolist(),
                        "vib_x_B":    sig[:, 3].tolist(),
                        "acoustic":   sig[:, 4].tolist(),
                    }
                    rows.append(row)

    df = pd.DataFrame(rows)
    out = tmp_path_factory.mktemp("data") / "synthetic_bearing.parquet"
    df.to_parquet(str(out), index=False)
    return out


# ---------------------------------------------------------------------------
# Real-dataset guard
# ---------------------------------------------------------------------------

def real_parquet_path() -> Path | None:
    p = os.environ.get("PARQUET_PATH", "")
    if p and Path(p).exists():
        return Path(p)
    return None


requires_real_data = pytest.mark.skipif(
    real_parquet_path() is None,
    reason="PARQUET_PATH not set or file missing — skipping real-data test",
)
