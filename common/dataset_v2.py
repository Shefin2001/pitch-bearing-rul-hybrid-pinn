"""dataset_v2.py — Run-level split + shared test index + sequence-aware loader.

Fixes the v1 leakage bug where ``_stratified_split`` shuffled all windows
across runs, allowing the model to see the temporal neighbourhood of every
test window during training (Sources 7, 11 in BENCHMARKS.md).

Key changes vs v1:
    1. **Run-level split** — entire (speed, condition, file_idx) recordings
       are assigned to one of train/val/test. No window from one run is ever
       split across two splits.
    2. **Per-window labels** — RUL & TTF are read from labels_paris.parquet
       (FPT-piecewise + Paris-law derived), not from a class-constant lookup.
    3. **Shared test index** — the test split is materialised once to disk so
       Hybrid, PINN and the fused model all evaluate on identical windows.
    4. **Sequence loader** — windows within a run are kept in temporal order
       so the monotonicity loss can compare consecutive predictions.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq
import torch
from scipy import signal as sp_signal
from torch.utils.data import Dataset

# Project-root imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.config import Config  # noqa: E402
from common.rul_labels import (  # noqa: E402
    FAULT_INDEX,
    RUL_MAP,
    get_progression_mask,
)

# Reuse Numba-accelerated kernels from approach_1 (already JIT-warmed).
# Optional: NumPy fallback below is functionally equivalent, just ~5× slower.
# Warn once so a missing install doesn't go unnoticed in production.
try:
    from approach_1_raw_signal.numba_kernels import (  # noqa: E402
        nb_extract_normalise_windows,
        nb_nan_guard_block,
    )
    HAS_NUMBA = True
except ImportError as _e:
    import warnings as _warnings
    _warnings.warn(
        f"[dataset_v2] approach_1_raw_signal.numba_kernels unavailable ({_e}); "
        f"falling back to NumPy preprocessing (~5× slower, results identical).",
        RuntimeWarning,
        stacklevel=2,
    )
    HAS_NUMBA = False


# ---------------------------------------------------------------------------
# Bandpass filter (reused from approach_1)
# ---------------------------------------------------------------------------

def design_bandpass(cfg: Config):
    nyq = cfg.sampling_freq / 2.0
    lo = max(cfg.bandpass_low / nyq, 0.001)
    hi = min(cfg.bandpass_high / nyq, 0.999)
    return sp_signal.butter(cfg.filter_order, [lo, hi], btype="band")


def apply_bandpass(x: np.ndarray, b, a) -> np.ndarray:
    out = np.empty_like(x)
    for ch in range(x.shape[1]):
        out[:, ch] = sp_signal.filtfilt(b, a, x[:, ch])
    return out


# ---------------------------------------------------------------------------
# Run-level split
# ---------------------------------------------------------------------------

def split_runs_run_level(
    runs: List[Tuple[str, str, int]],
    cfg: Config,
    seed: int = 42,
) -> Dict[str, List[Tuple[str, str, int]]]:
    """Assign each run to train/val/test, stratified by condition.

    Args:
        runs: list of (speed, condition, file_idx) tuples
        cfg:  Config (uses train_ratio, val_ratio, test_ratio)
        seed: RNG seed

    Returns:
        dict {"train": [...], "val": [...], "test": [...]}
    """
    rng = np.random.default_rng(seed)
    by_cond: Dict[str, List[Tuple[str, str, int]]] = {}
    for run in runs:
        by_cond.setdefault(run[1], []).append(run)

    splits: Dict[str, List[Tuple[str, str, int]]] = {"train": [], "val": [], "test": []}
    for cond, cond_runs in by_cond.items():
        arr = list(cond_runs)
        rng.shuffle(arr)
        n = len(arr)
        n_train = max(1, int(n * cfg.train_ratio))
        n_val = max(1, int(n * cfg.val_ratio)) if n > 2 else 0
        n_test = n - n_train - n_val
        if n_test < 1 and n >= 3:
            n_test = 1
            n_val = max(0, n - n_train - n_test)
        splits["train"].extend(arr[:n_train])
        splits["val"].extend(arr[n_train: n_train + n_val])
        splits["test"].extend(arr[n_train + n_val:])
    return splits


def discover_runs(parquet_path: str | Path) -> List[Tuple[str, str, int]]:
    """Enumerate every unique (speed, condition, file_idx) in the parquet."""
    pf = pq.ParquetFile(str(parquet_path))
    seen = set()
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg, columns=["speed", "condition", "file_idx"])
        sp = t.column("speed").to_pylist()
        co = t.column("condition").to_pylist()
        fi = t.column("file_idx").to_pylist()
        for s, c, f in zip(sp, co, fi):
            seen.add((s, c, int(f)))
    return sorted(seen)


# ---------------------------------------------------------------------------
# Per-window label lookup
# ---------------------------------------------------------------------------

class LabelLookup:
    """In-memory lookup for (speed, cond, file_idx, win_idx) → labels.

    Loads from ``labels_paris.parquet`` if present; otherwise falls back to
    class-constant RUL_MAP and a typical-life TTF estimate (with a warning).
    """

    def __init__(self, labels_paris_path: Optional[str | Path]) -> None:
        self.have_paris_labels = False
        self._table: Dict[Tuple[str, str, int, int], Tuple[float, float, float]] = {}
        if labels_paris_path is not None and Path(labels_paris_path).exists():
            self._load(labels_paris_path)
            self.have_paris_labels = True

    def _load(self, path: str | Path) -> None:
        t = pq.read_table(str(path))
        sp = t.column("speed").to_pylist()
        co = t.column("condition").to_pylist()
        fi = t.column("file_idx").to_pylist()
        wi = t.column("window_idx").to_pylist()
        rul = t.column("rul_relative").to_numpy(zero_copy_only=False)
        ttf = t.column("ttf_seconds").to_numpy(zero_copy_only=False)
        log_ttf = t.column("log_ttf_seconds").to_numpy(zero_copy_only=False)
        for i in range(len(sp)):
            self._table[(sp[i], co[i], int(fi[i]), int(wi[i]))] = (
                float(rul[i]), float(ttf[i]), float(log_ttf[i])
            )

    def get(self, speed: str, cond: str, file_idx: int, win_idx: int
            ) -> Tuple[float, float, float]:
        if self.have_paris_labels:
            key = (speed, cond, int(file_idx), int(win_idx))
            if key in self._table:
                return self._table[key]
        # Fallback — class-constant RUL, nominal life
        rul = float(RUL_MAP.get(cond, 0.5))
        ttf_seconds = max(rul * 1.5e8, 1.0)  # ~5 years × rul, very rough
        return rul, ttf_seconds, float(np.log(ttf_seconds))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PitchBearingDataset(Dataset):
    """Run-level-split dataset with per-window FPT+Paris labels.

    Each item:
        x          : (C=5, T=2048) float32  — bandpass-filtered, z-scored
        rul        : float32 ∈ [floor, 1.0]
        ttf_log    : float32 — log(seconds-to-failure)
        fault_idx  : int64
        prog_mask  : (12,) float32
        run_id     : int64 — index into self.runs (for monotonicity batching)
        win_idx    : int64 — position within run

    Args:
        cfg              : Config
        split            : "train" | "val" | "test"
        labels_paris_path: path to labels_paris.parquet (None → fallback)
        shared_test_path : if given AND split=="test", load test windows
                           from this materialised parquet instead of streaming
    """

    def __init__(
        self,
        cfg: Config,
        split: str = "train",
        labels_paris_path: Optional[str | Path] = None,
        shared_test_path: Optional[str | Path] = None,
        precompute_features: bool = True,
        verbose: bool = True,
    ) -> None:
        super().__init__()
        assert split in ("train", "val", "test")
        self.cfg = cfg
        self.split = split
        self.precompute_features = precompute_features
        self.labels = LabelLookup(labels_paris_path)
        self.runs: List[Tuple[str, str, int]] = []
        self._x: List[np.ndarray] = []
        self._feat: List[np.ndarray] = []
        self._meta: List[Tuple[float, float, int, np.ndarray, int, int, str]] = []
        # meta = (rul, log_ttf, fault_idx, prog_mask, run_id, win_idx, speed)

        if split == "test" and shared_test_path is not None and Path(shared_test_path).exists():
            self._load_from_shared(shared_test_path, verbose)
        else:
            self._stream_from_parquet(verbose)

    def _stream_from_parquet(self, verbose: bool) -> None:
        cfg = self.cfg
        all_runs = discover_runs(cfg.parquet_path)
        splits = split_runs_run_level(all_runs, cfg, seed=cfg.seed)
        my_runs = splits[self.split]
        my_run_set = set(my_runs)
        if verbose:
            print(f"[dataset_v2:{self.split}] runs assigned: {len(my_runs)} / {len(all_runs)}")

        b, a = design_bandpass(cfg)
        col_names = cfg.column_names

        # Lazy-import the feature extractor so PINN-only runs don't pay the cost.
        # Hard-fail when precompute_features=True: silently filling with zeros would
        # train the Hybrid model's 160-D feature head on garbage and the failure
        # would only surface as poor metrics.
        feat_extractors: Dict[str, "object"] = {}
        if self.precompute_features:
            try:
                from approach_2_wave_features.feature_extractor import FeatureExtractor
            except ImportError as e:
                raise ImportError(
                    f"[dataset_v2] cannot import approach_2_wave_features.feature_extractor "
                    f"({e}). The Hybrid track requires the 160-D engineered features; "
                    f"silently zero-filling would corrupt training. "
                    f"Either upload approach_2_wave_features/ next to Hybrid_PINN_ParisRUL/, "
                    f"or instantiate this dataset with precompute_features=False for PINN-only runs."
                ) from e
            for sp in cfg.speeds:
                feat_extractors[sp] = FeatureExtractor(cfg, speed=sp)

        # Buffer signal per run before processing (PyArrow streams may split a run)
        run_signal: Dict[Tuple[str, str, int], List[np.ndarray]] = {}

        pf = pq.ParquetFile(str(cfg.parquet_path))
        for rg in range(pf.num_row_groups):
            t = pf.read_row_group(rg, columns=col_names + ["speed", "condition", "file_idx"])
            sp = t.column("speed").to_pylist()
            co = t.column("condition").to_pylist()
            fi = t.column("file_idx").to_pylist()
            cols = [t.column(c).to_numpy(zero_copy_only=False) for c in col_names]
            sig = np.stack(cols, axis=1).astype(np.float32)  # (T, 5)

            # Group rows in this RG by run key (assumes contiguous)
            cur_key = None
            cur_start = 0
            for i in range(len(sp)):
                key = (sp[i], co[i], int(fi[i]))
                if cur_key is None:
                    cur_key = key
                if key != cur_key:
                    if cur_key in my_run_set:
                        run_signal.setdefault(cur_key, []).append(sig[cur_start:i].copy())
                    cur_key = key
                    cur_start = i
            if cur_key in my_run_set:
                run_signal.setdefault(cur_key, []).append(sig[cur_start:].copy())

        # Process each assigned run end-to-end (preserves window ordering)
        for run_id, run_key in enumerate(my_runs):
            if run_key not in run_signal:
                continue
            speed, cond, fi = run_key
            raw = np.concatenate(run_signal[run_key], axis=0)  # (T, 5)
            if HAS_NUMBA:
                nb_nan_guard_block(raw)
            else:
                np.nan_to_num(raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            if raw.shape[0] > cfg.filter_order * 3:
                raw_filt = apply_bandpass(raw, b, a)
            else:
                raw_filt = raw

            ws, stride = cfg.window_size, cfg.window_stride
            # Un-normalised windows (T, C) for feature extraction
            n_w = max(1, (raw_filt.shape[0] - ws) // stride + 1)
            raw_wins = np.zeros((n_w, ws, raw_filt.shape[1]), dtype=np.float32)
            for i in range(n_w):
                raw_wins[i] = raw_filt[i * stride: i * stride + ws]

            # Compute 160-D features once per run (vectorised, no Python loop over windows)
            if self.precompute_features:
                if speed not in feat_extractors:
                    raise KeyError(
                        f"[dataset_v2] no FeatureExtractor for speed={speed!r}; "
                        f"available speeds: {sorted(feat_extractors)}. Add this speed "
                        f"to cfg.speeds or set precompute_features=False."
                    )
                features_arr = feat_extractors[speed].extract(raw_wins)  # (N, 160)
            else:
                features_arr = np.zeros((n_w, 160), dtype=np.float32)

            # Z-scored channel-first windows for the raw branch
            if HAS_NUMBA:
                wins = nb_extract_normalise_windows(raw_filt, ws, stride)  # (N, C, ws)
            else:
                wins = self._extract_windows_numpy(raw_filt, ws, stride)
            n = wins.shape[0]
            self.runs.append(run_key)

            prog_mask = get_progression_mask(cond)
            fault_idx = FAULT_INDEX[cond]
            for w_idx in range(min(n, features_arr.shape[0])):
                rul, ttf_s, log_ttf = self.labels.get(speed, cond, fi, w_idx)
                self._x.append(wins[w_idx])
                self._feat.append(features_arr[w_idx])
                self._meta.append((float(rul), float(log_ttf), int(fault_idx),
                                   prog_mask, int(run_id), int(w_idx), speed))

        if verbose:
            print(f"[dataset_v2:{self.split}] windows: {len(self._x)} | runs: {len(self.runs)}"
                  f" | paris-labels: {self.labels.have_paris_labels}")

    @staticmethod
    def _extract_windows_numpy(raw: np.ndarray, ws: int, stride: int) -> np.ndarray:
        """Numpy-only fallback for window extraction + per-channel z-score.

        Returns (N, C, ws) float32.
        """
        n = max(1, (raw.shape[0] - ws) // stride + 1)
        out = np.empty((n, raw.shape[1], ws), dtype=np.float32)
        for i in range(n):
            seg = raw[i * stride: i * stride + ws]
            mu = seg.mean(axis=0, keepdims=True)
            std = seg.std(axis=0, keepdims=True) + 1e-8
            out[i] = ((seg - mu) / std).T
        return out

    def _load_from_shared(self, shared_path: str | Path, verbose: bool) -> None:
        """Load pre-materialised test windows from disk."""
        data = np.load(str(shared_path), allow_pickle=True)
        self._x = list(data["x"])
        if "feat" not in data.files:
            raise KeyError(
                f"[dataset_v2] shared test index at {shared_path} has no 'feat' array. "
                f"This .npz was built before engineered features were required — rebuild "
                f"it with `python -m Hybrid_PINN_ParisRUL.common.dataset_v2 --build ...`."
            )
        self._feat = list(data["feat"])
        rul_arr = data["rul"]
        log_ttf_arr = data["log_ttf"]
        fault_arr = data["fault_idx"]
        prog_arr = data["prog_mask"]
        run_id_arr = data["run_id"]
        win_idx_arr = data["win_idx"]
        speed_arr = data["speed"] if "speed" in data.files else np.array(["1rpm"] * len(self._x))
        runs_unique = data["runs"].tolist() if "runs" in data.files else []
        self.runs = [tuple(r) for r in runs_unique]
        for i in range(len(self._x)):
            self._meta.append((float(rul_arr[i]), float(log_ttf_arr[i]),
                               int(fault_arr[i]), prog_arr[i],
                               int(run_id_arr[i]), int(win_idx_arr[i]),
                               str(speed_arr[i])))
        if verbose:
            print(f"[dataset_v2:{self.split}] loaded shared test index: {len(self._x)} windows")

    def export_shared_test_index(self, out_path: str | Path) -> None:
        """Save the test split as an .npz for sharing across tracks."""
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        x = np.stack(self._x, axis=0).astype(np.float32)
        feat = np.stack(self._feat, axis=0).astype(np.float32) if self._feat else np.zeros((x.shape[0], 160), dtype=np.float32)
        rul = np.array([m[0] for m in self._meta], dtype=np.float32)
        log_ttf = np.array([m[1] for m in self._meta], dtype=np.float32)
        fault_idx = np.array([m[2] for m in self._meta], dtype=np.int64)
        prog_mask = np.stack([m[3] for m in self._meta], axis=0).astype(np.float32)
        run_id = np.array([m[4] for m in self._meta], dtype=np.int64)
        win_idx = np.array([m[5] for m in self._meta], dtype=np.int64)
        speed = np.array([m[6] for m in self._meta], dtype=object)
        runs = np.array(self.runs, dtype=object)
        np.savez_compressed(out_path, x=x, feat=feat, rul=rul, log_ttf=log_ttf,
                            fault_idx=fault_idx, prog_mask=prog_mask,
                            run_id=run_id, win_idx=win_idx, speed=speed, runs=runs)

    # -------------------- torch.utils.data.Dataset interface --------------------

    def __len__(self) -> int:
        return len(self._x)

    def __getitem__(self, idx: int):
        rul, log_ttf, fault_idx, prog_mask, run_id, win_idx, _speed = self._meta[idx]
        feat = self._feat[idx] if idx < len(self._feat) else np.zeros(160, dtype=np.float32)
        return {
            "x": torch.from_numpy(self._x[idx]),
            "feat": torch.from_numpy(feat),
            "rul": torch.tensor(rul, dtype=torch.float32),
            "log_ttf": torch.tensor(log_ttf, dtype=torch.float32),
            "fault_idx": torch.tensor(fault_idx, dtype=torch.long),
            "prog_mask": torch.from_numpy(prog_mask),
            "run_id": torch.tensor(run_id, dtype=torch.long),
            "win_idx": torch.tensor(win_idx, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def make_loaders(cfg: Config,
                 labels_paris_path: Optional[str | Path] = None,
                 shared_test_path: Optional[str | Path] = None,
                 ddp_sampler_fn=None,
                 verbose: bool = True):
    """Return (train_loader, val_loader, test_loader)."""
    train = PitchBearingDataset(cfg, "train", labels_paris_path,
                                shared_test_path=None, verbose=verbose)
    val = PitchBearingDataset(cfg, "val", labels_paris_path,
                              shared_test_path=None, verbose=verbose)
    test = PitchBearingDataset(cfg, "test", labels_paris_path,
                               shared_test_path=shared_test_path, verbose=verbose)

    def _make(ds: PitchBearingDataset, shuffle: bool):
        sampler = ddp_sampler_fn(ds, shuffle=shuffle) if ddp_sampler_fn else None
        return torch.utils.data.DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=(sampler is None) and shuffle,
            sampler=sampler,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
            prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
        )

    return _make(train, True), _make(val, False), _make(test, False)


# ---------------------------------------------------------------------------
# CLI — build & cache shared test index
# ---------------------------------------------------------------------------

def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true",
                        help="Build & save shared test index npz")
    parser.add_argument("--paris-labels",
                        default=r"D:\Pitch_Bearings_RUL\PitchBearing_RUL_DualNN\Hybrid_PINN_ParisRUL\results\labels\labels_paris.parquet")
    parser.add_argument("--out",
                        default=r"D:\Pitch_Bearings_RUL\PitchBearing_RUL_DualNN\Hybrid_PINN_ParisRUL\results\test_index\test_windows.npz")
    args = parser.parse_args()

    cfg = Config()
    cfg.seed_everything()
    print("[dataset_v2] discovering runs ...")
    runs = discover_runs(cfg.parquet_path)
    splits = split_runs_run_level(runs, cfg, seed=cfg.seed)
    for k, v in splits.items():
        cnt_by_cond: Dict[str, int] = {}
        for _, c, _ in v:
            cnt_by_cond[c] = cnt_by_cond.get(c, 0) + 1
        print(f"  {k:5s}: {len(v):4d} runs | by condition: {cnt_by_cond}")

    if args.build:
        labels = args.paris_labels if Path(args.paris_labels).exists() else None
        if labels is None:
            print("[!] No paris labels found — using class-constant fallback.")
        ds = PitchBearingDataset(cfg, "test", labels, shared_test_path=None, verbose=True)
        ds.export_shared_test_index(args.out)
        print(f"[OK] shared test index saved → {args.out}")


if __name__ == "__main__":
    _cli()
