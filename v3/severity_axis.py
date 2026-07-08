"""severity_axis.py — physics severity ordinal axis + interval-censored targets.

The 12 CUMTB classes are anchored to crack lengths (A_MAP_M, paris_labels.py).
Several classes share an anchor (IRC/ORC at 0.5 mm, IRS/ORS at 1.5 mm,
IRW/ORW at 3.0 mm), so the ordinal axis has 9 distinct severity LEVELS, not 12.

The continuous damage head is supervised with *interval-censored* targets:
class k only tells us the crack length lies between the geometric midpoints
to the neighbouring anchors — we never pretend to know `a` more precisely
than the class structure supports. The heteroscedastic head absorbs the
class-width uncertainty (survival-analysis view: arXiv 2405.01614).

Everything here is pure NumPy — importable by the torch-free inference path.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from common.rul_labels import FAULT_INDEX, INDEX_FAULT, N_CLASSES  # noqa: E402
from Hybrid_PINN_ParisRUL.common.paris_labels import A_FAIL_M, A_MAP_M  # noqa: E402

# ---------------------------------------------------------------------------
# Ordinal severity levels — unique crack-length anchors, ascending
# ---------------------------------------------------------------------------

_UNIQUE_A: List[float] = sorted(set(A_MAP_M.values()))
N_LEVELS: int = len(_UNIQUE_A)                       # 9 for the CUMTB map

#: class label → ordinal severity level (0 = Health … N_LEVELS-1 = IORW)
LEVEL_OF: Dict[str, int] = {
    cond: _UNIQUE_A.index(a) for cond, a in A_MAP_M.items()
}

#: human-readable stage name per level (used in inference output)
LEVEL_NAMES: List[str] = [
    "/".join(sorted(c for c, lv in LEVEL_OF.items() if lv == i))
    for i in range(N_LEVELS)
]

# ---------------------------------------------------------------------------
# Interval-censored bounds in log-crack-length space
# ---------------------------------------------------------------------------
# Bounds between consecutive levels are geometric midpoints (arithmetic in
# log-space). The lowest level opens down to half its anchor; the highest
# closes at the geometric midpoint to the failure length A_FAIL_M.

_log_a = np.log(np.asarray(_UNIQUE_A, dtype=np.float64))
_lo = np.empty(N_LEVELS)
_hi = np.empty(N_LEVELS)
_lo[0] = _log_a[0] + np.log(0.5)
_lo[1:] = 0.5 * (_log_a[:-1] + _log_a[1:])
_hi[:-1] = _lo[1:]
_hi[-1] = 0.5 * (_log_a[-1] + np.log(A_FAIL_M))

# ---------------------------------------------------------------------------
# Lookup tables indexed by FAULT_INDEX (torch-friendly gather targets)
# ---------------------------------------------------------------------------

SEV_LEVEL = np.zeros(N_CLASSES, dtype=np.int64)      # (12,) ordinal level
LOG_A_MID = np.zeros(N_CLASSES, dtype=np.float32)    # (12,) ln(anchor a [m])
LOG_A_LO = np.zeros(N_CLASSES, dtype=np.float32)     # (12,) censoring lower
LOG_A_HI = np.zeros(N_CLASSES, dtype=np.float32)     # (12,) censoring upper
for cond, idx in FAULT_INDEX.items():
    lv = LEVEL_OF[cond]
    SEV_LEVEL[idx] = lv
    LOG_A_MID[idx] = np.log(A_MAP_M[cond])
    LOG_A_LO[idx] = _lo[lv]
    LOG_A_HI[idx] = _hi[lv]

#: per-level anchor crack length [m], index = severity level
LEVEL_A_M = np.asarray(_UNIQUE_A, dtype=np.float64)


def corn_targets(sev_level: np.ndarray) -> np.ndarray:
    """CORN binary targets: (N, N_LEVELS-1) where col j = 1 if level > j."""
    lv = np.asarray(sev_level).reshape(-1, 1)
    return (lv > np.arange(N_LEVELS - 1)[None, :]).astype(np.float32)


def level_from_log_a(log_a: np.ndarray) -> np.ndarray:
    """Map continuous ln(a) back to the nearest severity level (for reports)."""
    return np.abs(np.log(LEVEL_A_M)[None, :]
                  - np.asarray(log_a).reshape(-1, 1)).argmin(axis=1)


if __name__ == "__main__":
    print(f"N_LEVELS = {N_LEVELS}")
    print(f"{'idx':>3} {'class':<7} {'lvl':>3} {'a(mm)':>7} {'lo(mm)':>8} {'hi(mm)':>8}")
    for idx in range(N_CLASSES):
        c = INDEX_FAULT[idx]
        print(f"{idx:>3} {c:<7} {SEV_LEVEL[idx]:>3} "
              f"{np.exp(LOG_A_MID[idx])*1e3:>7.2f} "
              f"{np.exp(LOG_A_LO[idx])*1e3:>8.3f} {np.exp(LOG_A_HI[idx])*1e3:>8.3f}")
    print("levels:", {i: n for i, n in enumerate(LEVEL_NAMES)})
    t = corn_targets(SEV_LEVEL)
    assert t.shape == (N_CLASSES, N_LEVELS - 1)
    assert (level_from_log_a(LOG_A_MID) == SEV_LEVEL).all()
    print("[OK] severity_axis self-test passed")
