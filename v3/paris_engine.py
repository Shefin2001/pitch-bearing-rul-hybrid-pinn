"""paris_engine.py -- damage state -> RUL distribution. Pure physics, no NN.

Chain (benchmark: DARPA ESP / Li et al. Paris-law bearing prognostics):

    P(ln a | window)  x  P(C)  x  P(class)        [from DamageNet + literature]
        -> closed-form Paris integration a0 -> A_FAIL   (per MC sample)
        -> TTF samples -> percentiles + inverse-Gaussian fit
           (Wiener-process first-passage form, deployable without sampling)

Closed form (m > 2), with K = Y * dsigma_MPa * sqrt(pi):

    N = (a0^(1-m/2) - af^(1-m/2)) / ((m/2 - 1) * C * K^m)

Backends: NumPy (default), CuPy (--gpu, H200), mpi4py (rank-sharded MC for
the 16 vCPUs -- embarrassingly parallel). All torch-free by design so the
deployed ONNX wrapper can import this module directly.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from common.rul_labels import FAULT_INDEX, N_CLASSES  # noqa: E402
from Hybrid_PINN_ParisRUL.common.paris_labels import (  # noqa: E402
    A_FAIL_M,
    C_PARIS,
    CYCLE_SECONDS,
    M_PARIS,
    Y_GEOM,
    delta_sigma_pa,
)

#: per-class fatigue stress range [MPa], FAULT_INDEX order (physics constants)
DELTA_SIGMA_MPA = np.zeros(N_CLASSES, dtype=np.float64)
for _cond, _idx in FAULT_INDEX.items():
    DELTA_SIGMA_MPA[_idx] = delta_sigma_pa(_cond) * 1e-6

#: lognormal scatter of the Paris C constant. Literature reports roughly a
#: factor-2 spread for 42CrMo4 batches; ln(2)/2 puts +-2 sd at x4 / x0.25.
SIGMA_LN_C_DEFAULT = 0.35


def paris_cycles_closed_form(a0_m, delta_sigma_MPa, C=C_PARIS, m=M_PARIS,
                             a_fail_m: float = A_FAIL_M, xp=np):
    """Vectorised cycles-to-failure. Any argument may be an array."""
    a0 = xp.clip(a0_m, 1e-6, a_fail_m * 0.999)
    K = Y_GEOM * delta_sigma_MPa * math.sqrt(math.pi)
    e = 1.0 - m / 2.0                                  # < 0 for m > 2
    return (a0 ** e - a_fail_m ** e) / ((m / 2.0 - 1.0) * C * K ** m)


@dataclass
class RULResult:
    ttf_seconds_p5: float
    ttf_seconds_p50: float
    ttf_seconds_p95: float
    ttf_seconds_mean: float
    ig_mu: float          # inverse-Gaussian mean (seconds)
    ig_lambda: float      # inverse-Gaussian shape (seconds)
    n_samples: int

    def hours(self) -> Dict[str, float]:
        return {"p5": self.ttf_seconds_p5 / 3600.0,
                "p50": self.ttf_seconds_p50 / 3600.0,
                "p95": self.ttf_seconds_p95 / 3600.0,
                "mean": self.ttf_seconds_mean / 3600.0}


def _summarise(ttf_s: np.ndarray) -> RULResult:
    ttf_s = np.asarray(ttf_s, dtype=np.float64)
    ttf_s = ttf_s[np.isfinite(ttf_s) & (ttf_s > 0)]
    if ttf_s.size == 0:
        return RULResult(0, 0, 0, 0, 0, 0, 0)
    p5, p50, p95 = np.percentile(ttf_s, [5, 50, 95])
    mean = float(ttf_s.mean())
    var = float(ttf_s.var())
    # Inverse-Gaussian moment fit: mean = mu, var = mu^3 / lambda
    ig_lambda = mean ** 3 / max(var, 1e-12)
    return RULResult(float(p5), float(p50), float(p95), mean,
                     mean, float(ig_lambda), int(ttf_s.size))


def mc_rul(log_a_mu: float, log_a_sigma: float,
           class_probs: np.ndarray,
           n_samples: int = 100_000,
           sigma_ln_c: float = SIGMA_LN_C_DEFAULT,
           cycle_seconds: float = CYCLE_SECONDS,
           seed: int = 0,
           backend: str = "auto") -> RULResult:
    """Monte-Carlo RUL for ONE window/aggregate damage estimate.

    Samples (a0, class -> dsigma, C) jointly, integrates Paris in closed
    form, returns the TTF distribution summary.
    """
    xp = np
    if backend in ("auto", "cupy"):
        try:
            import cupy as cp
            if cp.cuda.runtime.getDeviceCount() > 0:
                xp = cp
        except Exception:
            if backend == "cupy":
                raise
    if backend == "mpi":
        return _mc_rul_mpi(log_a_mu, log_a_sigma, class_probs, n_samples,
                           sigma_ln_c, cycle_seconds, seed)

    rng = xp.random.default_rng(seed) if xp is np else None
    if xp is np:
        z = rng.standard_normal(n_samples)
        zc = rng.standard_normal(n_samples)
        p = np.clip(np.asarray(class_probs, dtype=np.float64), 0.0, None)
        p /= max(p.sum(), 1e-12)
        cls = rng.choice(N_CLASSES, size=n_samples, p=p)
    else:
        xp.random.seed(seed)
        z = xp.random.standard_normal(n_samples)
        zc = xp.random.standard_normal(n_samples)
        p = xp.asarray(class_probs, dtype=xp.float64)
        p = p / p.sum()
        cls = xp.searchsorted(xp.cumsum(p), xp.random.uniform(size=n_samples))
        cls = xp.clip(cls, 0, N_CLASSES - 1)

    a0 = xp.exp(log_a_mu + log_a_sigma * z)
    C = C_PARIS * xp.exp(sigma_ln_c * zc)
    dsig = xp.asarray(DELTA_SIGMA_MPA)[cls]
    n_cycles = paris_cycles_closed_form(a0, dsig, C=C, xp=xp)
    ttf_s = n_cycles * cycle_seconds
    if xp is not np:
        ttf_s = xp.asnumpy(ttf_s)
    return _summarise(ttf_s)


def _mc_rul_mpi(log_a_mu, log_a_sigma, class_probs, n_samples,
                sigma_ln_c, cycle_seconds, seed) -> Optional[RULResult]:
    """Rank-sharded MC:  mpirun -np 16 python -m ...paris_engine --mpi.
    Non-root ranks return None."""
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank, world = comm.Get_rank(), comm.Get_size()
    local = mc_rul(log_a_mu, log_a_sigma, class_probs,
                   n_samples=max(1, n_samples // world),
                   sigma_ln_c=sigma_ln_c, cycle_seconds=cycle_seconds,
                   seed=seed + 7919 * rank, backend="numpy")
    # Gather the raw percentile-defining samples is overkill; gather summaries
    # weighted by sample count (percentile-of-merged approximated by median of
    # rank percentiles -- adequate at >= 6k samples/rank).
    all_s = comm.gather((local.ttf_seconds_p5, local.ttf_seconds_p50,
                         local.ttf_seconds_p95, local.ttf_seconds_mean,
                         local.n_samples), root=0)
    if rank != 0:
        return None
    arr = np.asarray(all_s)
    w = arr[:, 4] / arr[:, 4].sum()
    mean = float((arr[:, 3] * w).sum())
    p5, p50, p95 = (float(np.median(arr[:, i])) for i in range(3))
    var_proxy = ((p95 - p5) / 3.29) ** 2  # normal-approx var from CI width
    return RULResult(p5, p50, p95, mean, mean,
                     mean ** 3 / max(var_proxy, 1e-12), int(arr[:, 4].sum()))


def ig_quantile(mu: float, lam: float, q: float, tol: float = 1e-6) -> float:
    """Inverse-Gaussian quantile by bisection on the closed-form CDF
    (deployment path: percentiles without re-sampling)."""
    from math import erf, exp, sqrt

    def _ndtr(x):
        return 0.5 * (1.0 + erf(x / sqrt(2.0)))

    def cdf(x):
        if x <= 0:
            return 0.0
        s = sqrt(lam / x)
        # guard exp overflow for large lam/mu
        e = 2.0 * lam / mu
        second = exp(min(e, 700.0)) * _ndtr(-s * (x / mu + 1.0)) if e < 700 else 0.0
        return _ndtr(s * (x / mu - 1.0)) + second

    lo, hi = 1e-6, mu * 100 + 1e6
    while cdf(hi) < q and hi < 1e18:
        hi *= 10
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if cdf(mid) < q:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol * max(1.0, mid):
            break
    return 0.5 * (lo + hi)


if __name__ == "__main__":
    import argparse
    from Hybrid_PINN_ParisRUL.common.paris_labels import (
        A_MAP_M,
        paris_cycles_to_failure,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--mpi", action="store_true")
    parser.add_argument("--samples", type=int, default=100_000)
    args = parser.parse_args()

    # 1) Closed form vs the v2 loop integrator (adaptive chunking => few % off)
    for cond in ("IRC", "IRS", "IORW"):
        n_loop = paris_cycles_to_failure(cond)
        idx = FAULT_INDEX[cond]
        n_cf = float(paris_cycles_closed_form(A_MAP_M[cond],
                                              DELTA_SIGMA_MPA[idx]))
        err = abs(n_cf - n_loop) / max(n_loop, 1)
        print(f"{cond:5s} loop={n_loop:>12,d} closed={n_cf:>14,.0f} "
              f"rel.err={err:.3%}")
        assert err < 0.05, f"closed form deviates {err:.1%} for {cond}"

    # 2) MC for a moderately damaged bearing (IRS-ish, a ~ 1.5 mm +- class width)
    probs = np.zeros(N_CLASSES)
    probs[FAULT_INDEX["IRS"]] = 0.8
    probs[FAULT_INDEX["ORS"]] = 0.2
    backend = "mpi" if args.mpi else "auto"
    res = mc_rul(math.log(1.5e-3), 0.25, probs, n_samples=args.samples,
                 seed=1, backend=backend)
    if res is not None:
        h = res.hours()
        print(f"IRS MC RUL: p5={h['p5']:.0f}h p50={h['p50']:.0f}h "
              f"p95={h['p95']:.0f}h (n={res.n_samples})")
        assert h["p5"] < h["p50"] < h["p95"]
        # 3) IG quantiles should approximate the MC percentiles
        q50 = ig_quantile(res.ig_mu, res.ig_lambda, 0.5) / 3600
        print(f"IG fit: mu={res.ig_mu/3600:.0f}h lambda={res.ig_lambda/3600:.0f}h "
              f"q50={q50:.0f}h (MC p50={h['p50']:.0f}h)")
        # 4) terminal ordering: near-failure IORW is far shorter than any
        # early state. NOTE: IRC vs IRS is NOT monotone in crack length --
        # a sharp crack (K_t 2.40) at 0.5 mm outpaces a blunt spall (K_t 1.80)
        # at 1.5 mm under the FEM K_t map. That inversion is real physics and
        # is surfaced in the expert report, not asserted away here.
        r_irc = mc_rul(math.log(0.5e-3), 0.25,
                       np.eye(N_CLASSES)[FAULT_INDEX["IRC"]], seed=2)
        r_iorw = mc_rul(math.log(7.0e-3), 0.25,
                        np.eye(N_CLASSES)[FAULT_INDEX["IORW"]], seed=3)
        assert r_iorw.ttf_seconds_p50 < 0.1 * res.ttf_seconds_p50
        assert r_iorw.ttf_seconds_p50 < 0.1 * r_irc.ttf_seconds_p50
        print(f"terminal ordering OK: IORW {r_iorw.hours()['p50']:.0f}h << "
              f"IRS {h['p50']:.0f}h ~ IRC {r_irc.hours()['p50']:.0f}h")
        print("[OK] paris_engine self-test passed")
