"""dataset_v3.py -- wraps the proven v2 PitchBearingDataset with v3 targets.

Adds per item:
    phys      (PHYS_DIM,)  run-level physics features, z-scored on train stats
    sev_level ()           ordinal severity level from the class anchor
    log_a_lo/hi ()         interval-censoring bounds for the damage head

Reuses the v2 feature cache untouched (same base dataset, same md5 key), so
the expensive extraction is shared between v2 and v3 training.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from common.config import Config  # noqa: E402
from common.rul_labels import INDEX_FAULT  # noqa: E402
from Hybrid_PINN_ParisRUL.common.dataset_v2 import PitchBearingDataset  # noqa: E402
from Hybrid_PINN_ParisRUL.v3.severity_axis import (  # noqa: E402
    LOG_A_HI,
    LOG_A_LO,
    SEV_LEVEL,
)
from Hybrid_PINN_ParisRUL.v3.signal_physics import PHYS_DIM  # noqa: E402

RunKey = Tuple[str, str, int]


class V3Dataset(Dataset):
    """Attach run-level phys features + severity targets to a v2 dataset.

    Exact run keys come from base.runs[run_id]; if a key is missing from the
    phys table (e.g. rebuilt subset), falls back to the (speed, condition)
    group mean so training never crashes on a stale table.
    """

    def __init__(self, base: PitchBearingDataset,
                 phys_table: Dict[RunKey, np.ndarray],
                 phys_mean: np.ndarray, phys_std: np.ndarray) -> None:
        self.base = base
        self.phys_mean = phys_mean.astype(np.float32)
        self.phys_std = phys_std.astype(np.float32)

        group_mean: Dict[Tuple[str, str], np.ndarray] = {}
        for (sp, co, _fi), v in phys_table.items():
            group_mean.setdefault((sp, co), []).append(v)
        group_mean = {k: np.mean(v, axis=0) for k, v in group_mean.items()}
        zero = np.zeros(PHYS_DIM, dtype=np.float32)

        # Pre-resolve one normalized phys vector per run_id
        self._phys_by_run: Dict[int, torch.Tensor] = {}
        self._n_fallback = 0
        for run_id, key in enumerate(base.runs):
            key = (key[0], key[1], int(key[2]))
            v = phys_table.get(key)
            if v is None:
                v = group_mean.get((key[0], key[1]))
                self._n_fallback += 1
            if v is None:
                v = zero
            z = (v.astype(np.float32) - self.phys_mean) / self.phys_std
            self._phys_by_run[run_id] = torch.from_numpy(z)
        self._phys_zero = torch.from_numpy(
            (zero - self.phys_mean) / self.phys_std)

        self._sev = torch.from_numpy(SEV_LEVEL)
        self._lo = torch.from_numpy(LOG_A_LO)
        self._hi = torch.from_numpy(LOG_A_HI)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        item = self.base[idx]
        run_id = int(item["run_id"])
        fault = item["fault_idx"]
        item["phys"] = self._phys_by_run.get(run_id, self._phys_zero)
        item["sev_level"] = self._sev[fault]
        item["log_a_lo"] = self._lo[fault]
        item["log_a_hi"] = self._hi[fault]
        return item


def compute_phys_norm(phys_table: Dict[RunKey, np.ndarray],
                      train_runs) -> Tuple[np.ndarray, np.ndarray]:
    """Mean/std over TRAIN-split runs only (no test-set peeking)."""
    keys = [(r[0], r[1], int(r[2])) for r in train_runs]
    vals = [phys_table[k] for k in keys if k in phys_table]
    if not vals:
        vals = list(phys_table.values()) or [np.zeros(PHYS_DIM, np.float32)]
    arr = np.asarray(vals, dtype=np.float64)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def save_phys_norm(path: str | Path, mean: np.ndarray, std: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"mean": mean.tolist(), "std": std.tolist()}, f)


def load_phys_norm(path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    with open(path) as f:
        d = json.load(f)
    return (np.asarray(d["mean"], dtype=np.float32),
            np.asarray(d["std"], dtype=np.float32))


def make_v3_loaders(cfg: Config,
                    physfeat_path: str | Path,
                    labels_paris_path: Optional[str | Path] = None,
                    shared_test_path: Optional[str | Path] = None,
                    feat_cache_dir: Optional[Path] = None,
                    phys_norm_out: Optional[str | Path] = None,
                    verbose: bool = True):
    """(train_loader, val_loader, test_loader, phys_norm) for the v3 track."""
    from Hybrid_PINN_ParisRUL.v3.build_physfeat import load_physfeat_table

    phys_table = load_physfeat_table(physfeat_path)

    train_b = PitchBearingDataset(cfg, "train", labels_paris_path,
                                  shared_test_path=None, verbose=verbose,
                                  feat_cache_dir=feat_cache_dir)
    val_b = PitchBearingDataset(cfg, "val", labels_paris_path,
                                shared_test_path=None, verbose=verbose,
                                feat_cache_dir=feat_cache_dir)
    test_b = PitchBearingDataset(cfg, "test", labels_paris_path,
                                 shared_test_path=shared_test_path,
                                 verbose=verbose)

    mean, std = compute_phys_norm(phys_table, train_b.runs)
    if phys_norm_out is not None:
        save_phys_norm(phys_norm_out, mean, std)

    wrap = [V3Dataset(b, phys_table, mean, std)
            for b in (train_b, val_b, test_b)]
    if verbose:
        for name, w in zip(("train", "val", "test"), wrap):
            if w._n_fallback:
                print(f"[v3:dataset:{name}] WARN {w._n_fallback} runs missing "
                      f"from phys table (group-mean fallback)")

    def _make(ds, shuffle: bool, drop_last: bool = False):
        return torch.utils.data.DataLoader(
            ds, batch_size=cfg.batch_size, shuffle=shuffle,
            num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
            persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
            prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
            drop_last=drop_last)

    return (_make(wrap[0], True, drop_last=getattr(cfg, "drop_last", False)),
            _make(wrap[1], False), _make(wrap[2], False), (mean, std))
