"""rul_labels_v2.py — FPT-piecewise-linear RUL per recording.

Replaces the broken class-constant ``RUL_MAP`` from ``common/rul_labels.py``.
Each (condition, file_idx) recording gets its own degradation curve based on
its own RMS trend, eliminating the lookup-table shortcut that produced the
degenerate R²=0.986 in v1.

Algorithm (per recording):
    1. Compute rolling RMS of vib_y_A over windows of 2048 samples, stride 1024.
    2. Estimate baseline from first 100 windows (healthy reference).
    3. First Prediction Time (FPT) = first index where RMS > k_fpt × baseline.
    4. RUL = 1.0 before FPT, then linearly decays to a class-dependent floor.
       Floor = old RUL_MAP value, so a Healthy run ends at 1.0 (no decay) and
       a near-failure run (IORW) ends at 0.05.

The floor preserves class-conditioning (model still benefits from "this is an
IORW recording → expect heavy decay") while removing the constant-output
shortcut that caused window leakage to read as 99% accuracy.

Reference: Lin et al. 2021, "A Novel Approach of Label Construction for
Predicting Remaining Useful Life of Machinery", Shock and Vibration.
DOI: 10.1155/2021/6806319
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pyarrow.parquet as pq

# Re-export DAG helpers from v1 unchanged — they are correct.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.rul_labels import (  # noqa: E402  (parent import)
    FAULT_INDEX,
    INDEX_FAULT,
    N_CLASSES,
    PROGRESSION_GRAPH,
    RUL_MAP,
    get_fault_risk_vector,
    get_progression_mask,
    get_progression_timeline,
    get_reachable_indices,
    rul_category,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WINDOW_SIZE = 2048
WINDOW_STRIDE = 1024
RMS_CHANNEL = "vib_y_A"            # primary channel for FPT detection
FPT_THRESHOLD_K = 2.0              # k × baseline RMS triggers FPT
BASELINE_NWINDOWS = 100            # number of leading windows for baseline


# ---------------------------------------------------------------------------
# RMS computation (vectorised over windows)
# ---------------------------------------------------------------------------

def rolling_rms(signal: np.ndarray, window: int = WINDOW_SIZE,
                stride: int = WINDOW_STRIDE) -> np.ndarray:
    """Compute RMS over sliding windows.

    Args:
        signal: 1-D array, shape (T,)
        window: window size in samples
        stride: hop size in samples

    Returns:
        1-D array of RMS values, shape (n_windows,) where
        n_windows = (T - window) // stride + 1
    """
    if signal.size < window:
        return np.array([float(np.sqrt(np.mean(signal ** 2)))], dtype=np.float32)

    n = (signal.size - window) // stride + 1
    # Vectorised stride trick — no Python loop
    shape = (n, window)
    strides = (signal.strides[0] * stride, signal.strides[0])
    framed = np.lib.stride_tricks.as_strided(signal, shape=shape, strides=strides,
                                             writeable=False)
    return np.sqrt((framed.astype(np.float32) ** 2).mean(axis=1))


def detect_fpt(rms_series: np.ndarray,
               baseline_n: int = BASELINE_NWINDOWS,
               k: float = FPT_THRESHOLD_K) -> int:
    """Return First Prediction Time index.

    FPT = first index where RMS exceeds k × baseline mean.
    Falls back to the midpoint of the run if no exceedance is found
    (this happens for healthy or very-stable recordings).
    """
    if rms_series.size <= baseline_n:
        return rms_series.size // 2

    baseline = rms_series[:baseline_n].mean() + 1e-12
    threshold = k * baseline
    exceed = np.where(rms_series[baseline_n:] > threshold)[0]
    if exceed.size == 0:
        return rms_series.size // 2
    return baseline_n + int(exceed[0])


def build_rul_curve(rms_series: np.ndarray, condition: str) -> np.ndarray:
    """Build per-window RUL curve for a single recording.

    Args:
        rms_series: rolling RMS, shape (n_windows,)
        condition: fault label (e.g., "Health", "IORW")

    Returns:
        rul: float32 array, shape (n_windows,), values in [floor, 1.0]
    """
    n = rms_series.size
    floor = float(RUL_MAP[condition])
    fpt_idx = detect_fpt(rms_series)

    rul = np.ones(n, dtype=np.float32)
    if fpt_idx < n:
        decay_len = n - fpt_idx
        # Linear decay 1.0 → floor over the post-FPT segment
        rul[fpt_idx:] = np.linspace(1.0, floor, decay_len, dtype=np.float32)
    return rul


# ---------------------------------------------------------------------------
# Per-recording driver (reads from parquet)
# ---------------------------------------------------------------------------

def iterate_recordings(parquet_path: str | Path) -> Iterable[Tuple[str, str, int, np.ndarray]]:
    """Stream (speed, condition, file_idx, vib_y_A) tuples from parquet.

    Uses PyArrow row-group streaming — never loads full 4.5 GB.
    Groups are reconstructed by (speed, condition, file_idx).
    """
    parquet_path = Path(parquet_path)
    pf = pq.ParquetFile(str(parquet_path))

    # Build group index from metadata only (no data loaded)
    group_buffer: Dict[Tuple[str, str, int], list] = {}
    for rg_idx in range(pf.num_row_groups):
        table = pf.read_row_group(rg_idx, columns=[RMS_CHANNEL, "speed",
                                                    "condition", "file_idx"])
        speed_col = table.column("speed").to_pylist()
        cond_col = table.column("condition").to_pylist()
        file_col = table.column("file_idx").to_pylist()
        sig_col = table.column(RMS_CHANNEL).to_numpy(zero_copy_only=False)

        # Within a row-group, all rows usually share the same group key,
        # but be defensive.
        cur_key = None
        cur_buf: list = []
        for i, (sp, co, fi) in enumerate(zip(speed_col, cond_col, file_col)):
            key = (sp, co, int(fi))
            if cur_key is None:
                cur_key = key
            if key != cur_key:
                # Flush buffer
                acc = group_buffer.setdefault(cur_key, [])
                acc.append(np.asarray(cur_buf, dtype=np.float32))
                cur_buf = []
                cur_key = key
            cur_buf.append(sig_col[i])
        if cur_key is not None:
            acc = group_buffer.setdefault(cur_key, [])
            acc.append(np.asarray(cur_buf, dtype=np.float32))

    # Yield concatenated per-recording signal
    for (speed, cond, fi), chunks in group_buffer.items():
        yield speed, cond, fi, np.concatenate(chunks)


# ---------------------------------------------------------------------------
# Offline label table builder
# ---------------------------------------------------------------------------

def build_label_table(parquet_path: str | Path) -> Dict[Tuple[str, str, int], np.ndarray]:
    """Build {(speed, condition, file_idx): rul_curve} for every recording.

    Returns a dict mapping recording key → per-window RUL array.
    """
    table: Dict[Tuple[str, str, int], np.ndarray] = {}
    for speed, cond, fi, signal in iterate_recordings(parquet_path):
        rms = rolling_rms(signal)
        rul = build_rul_curve(rms, cond)
        table[(speed, cond, fi)] = rul
    return table


def save_label_table(table: Dict[Tuple[str, str, int], np.ndarray],
                     out_path: str | Path) -> None:
    """Save label table as a flat parquet file.

    Schema: speed, condition, file_idx, window_idx, rul_relative
    """
    import pyarrow as pa
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    speed_col, cond_col, file_col, win_col, rul_col = [], [], [], [], []
    for (sp, co, fi), rul in table.items():
        n = rul.size
        speed_col.extend([sp] * n)
        cond_col.extend([co] * n)
        file_col.extend([fi] * n)
        win_col.extend(range(n))
        rul_col.extend(rul.tolist())

    arrow_table = pa.table({
        "speed": speed_col,
        "condition": cond_col,
        "file_idx": file_col,
        "window_idx": win_col,
        "rul_relative": rul_col,
    })
    pq.write_table(arrow_table, out_path, compression="snappy")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet",
                        default=r"D:\Pitch_Bearings_RUL\pitch_bearing_dataset.parquet")
    parser.add_argument("--out",
                        default=r"D:\Pitch_Bearings_RUL\PitchBearing_RUL_DualNN\Hybrid_PINN_ParisRUL\results\labels\labels_fpt.parquet")
    parser.add_argument("--demo", action="store_true",
                        help="Skip parquet read; show synthetic FPT curve.")
    args = parser.parse_args()

    if args.demo or not Path(args.parquet).exists():
        # Synthetic demonstration — useful for CI / smoke tests
        print("[DEMO MODE] generating synthetic RMS series for 'IRC' run")
        t = np.arange(500)
        baseline = 1.0
        defect_rise = np.where(t > 200, np.exp((t - 200) / 100.0), 0.0)
        rms = baseline + defect_rise + 0.1 * np.random.randn(500)
        fpt = detect_fpt(rms.astype(np.float32))
        rul = build_rul_curve(rms.astype(np.float32), "IRC")
        print(f"  Detected FPT index: {fpt}/{len(rms)}")
        print(f"  RUL[start]={rul[0]:.3f}, RUL[FPT]={rul[fpt]:.3f}, RUL[end]={rul[-1]:.3f}")
        print(f"  Class floor (RUL_MAP['IRC']) = {RUL_MAP['IRC']}")
        return

    print(f"Building FPT labels from: {args.parquet}")
    table = build_label_table(args.parquet)
    print(f"Built {len(table)} recording labels")
    print(f"Saving → {args.out}")
    save_label_table(table, args.out)
    print("[OK]")


if __name__ == "__main__":
    _cli()
