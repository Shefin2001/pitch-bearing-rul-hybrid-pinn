"""signal_physics.py — run-level physical damage evidence from vibration.

Why RUN-level, not window-level: at 1–3 rpm the fault characteristic
frequencies are sub-Hz (BPFO ≈ 6.6 × f_shaft ≈ 0.11 Hz @ 1 rpm), so a
2048-sample / 53 ms window can never resolve them. The full recording can
(Δf = 1/T_rec). Each recording is one steady-state damage snapshot, so one
physics vector per run, broadcast to all its windows, is both resolvable
and positionally-safe (cannot encode window position).

Features per run (PHYS_DIM total):
  per channel-group (vib = max over 4 vib channels, ac = acoustic):
    - envelope-spectrum SNR (log1p) at harmonics 1–3 of BPFO/BPFI/BSF/FTF (12)
    - kurtosis, crest factor, envelope kurtosis, high-band ratio, log-RMS (5)
  global:
    - dual-impulse lag Δt [s], implied spall length ln(mm), confidence (3)

Dual-impulse basis: Sawalhi & Randall entry/exit events — the lag between
the entry step response and exit impulse scales with spall length via the
rolling-contact sweep speed. Known noisy on its own; used here as auxiliary
evidence for the NN, never as the sole damage estimate.

GPU: CuPy path (H200) with transparent NumPy fallback — same code, `xp` swap.
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

try:
    import cupy as _cupy
    _HAS_CUPY = _cupy.cuda.runtime.getDeviceCount() > 0
except Exception:
    _cupy = None
    _HAS_CUPY = False

# ---------------------------------------------------------------------------
# Bearing geometry defaults (CUMTB pitch bearing — mirrors common/config.py)
# ---------------------------------------------------------------------------

NB_DEFAULT = 16
BD_MM_DEFAULT = 22.0
PD_MM_DEFAULT = 120.0
CONTACT_ANGLE_DEG_DEFAULT = 15.0

SPEED_HZ: Dict[str, float] = {"1rpm": 1.0 / 60.0, "3rpm": 3.0 / 60.0}

_HARMONICS = (1, 2, 3)
_CHAR_FREQS = ("BPFO", "BPFI", "BSF", "FTF")
_GROUP_STATS = ("kurtosis", "crest", "env_kurtosis", "high_band_ratio", "log_rms")

PHYS_NAMES: List[str] = (
    [f"vib_env_snr_{f}_h{h}" for f in _CHAR_FREQS for h in _HARMONICS]
    + [f"vib_{s}" for s in _GROUP_STATS]
    + [f"ac_env_snr_{f}_h{h}" for f in _CHAR_FREQS for h in _HARMONICS]
    + [f"ac_{s}" for s in _GROUP_STATS]
    + ["dual_impulse_dt_s", "dual_impulse_len_log_mm", "dual_impulse_conf"]
)
PHYS_DIM: int = len(PHYS_NAMES)  # 37


def fault_frequencies(f_shaft_hz: float,
                      nb: int = NB_DEFAULT,
                      bd_mm: float = BD_MM_DEFAULT,
                      pd_mm: float = PD_MM_DEFAULT,
                      contact_angle_deg: float = CONTACT_ANGLE_DEG_DEFAULT,
                      ) -> Dict[str, float]:
    """Characteristic defect frequencies [Hz] for a rotating inner race."""
    g = (bd_mm / pd_mm) * math.cos(math.radians(contact_angle_deg))
    return {
        "BPFO": 0.5 * nb * f_shaft_hz * (1.0 - g),
        "BPFI": 0.5 * nb * f_shaft_hz * (1.0 + g),
        "BSF": (pd_mm / (2.0 * bd_mm)) * f_shaft_hz * (1.0 - g * g),
        "FTF": 0.5 * f_shaft_hz * (1.0 - g),
    }


def rolling_sweep_speed_mps(f_shaft_hz: float,
                            bd_mm: float = BD_MM_DEFAULT,
                            pd_mm: float = PD_MM_DEFAULT,
                            contact_angle_deg: float = CONTACT_ANGLE_DEG_DEFAULT,
                            ) -> float:
    """Approximate speed at which the rolling contact sweeps across a race
    defect [m/s] — cage-relative race surface speed. Used to convert the
    entry/exit lag into a spall length. First-order kinematics; documented
    approximation, adequate for an auxiliary feature."""
    g = (bd_mm / pd_mm) * math.cos(math.radians(contact_angle_deg))
    return 0.5 * math.pi * (pd_mm * 1e-3) * f_shaft_hz * (1.0 - g * g)


# ---------------------------------------------------------------------------
# xp-generic kernels (xp = numpy | cupy)
# ---------------------------------------------------------------------------

def _envelope(x, xp):
    """Analytic-signal magnitude via FFT Hilbert transform. x: (..., T)."""
    n = x.shape[-1]
    X = xp.fft.fft(x, axis=-1)
    h = xp.zeros(n, dtype=X.real.dtype)
    if n % 2 == 0:
        h[0] = h[n // 2] = 1.0
        h[1:n // 2] = 2.0
    else:
        h[0] = 1.0
        h[1:(n + 1) // 2] = 2.0
    return xp.abs(xp.fft.ifft(X * h, axis=-1))


def _mean_pool(x, factor: int, xp):
    """Anti-alias-lite decimation of a slowly-varying envelope: block mean."""
    if factor <= 1:
        return x
    t = (x.shape[-1] // factor) * factor
    if t == 0:
        return x
    return x[..., :t].reshape(*x.shape[:-1], t // factor, factor).mean(axis=-1)


def _kurtosis(x, xp, axis=-1):
    mu = x.mean(axis=axis, keepdims=True)
    d = x - mu
    var = (d * d).mean(axis=axis)
    m4 = (d ** 4).mean(axis=axis)
    return m4 / (var * var + 1e-24)


def _env_snr_features(env_pooled, fs_pool: float, freqs_hz: Dict[str, float], xp
                      ) -> List[float]:
    """log1p SNR of the envelope spectrum at harmonics of each char freq."""
    n = env_pooled.shape[-1]
    out: List[float] = []
    if n < 16:
        return [0.0] * (len(_CHAR_FREQS) * len(_HARMONICS))
    win = xp.hanning(n)
    e = env_pooled - env_pooled.mean()
    A = xp.abs(xp.fft.rfft(e * win))
    fbins = xp.fft.rfftfreq(n, d=1.0 / fs_pool)
    df = fs_pool / n
    A_np = A  # stays on device until scalar extraction
    for name in _CHAR_FREQS:
        f0 = freqs_hz[name]
        for h in _HARMONICS:
            f = f0 * h
            if f <= df or f >= float(fbins[-1]):
                out.append(0.0)
                continue
            band = xp.abs(fbins - f) <= max(2.0 * df, 0.02 * f)
            local = (xp.abs(fbins - f) <= max(20.0 * df, 0.25 * f)) & ~band
            if not bool(band.any()) or not bool(local.any()):
                out.append(0.0)
                continue
            peak = float(A_np[band].max())
            noise = float(xp.median(A_np[local]))
            out.append(math.log1p(peak / (noise + 1e-12)))
    return out


def _dual_impulse(env_pooled, fs_pool: float, t_ballpass_s: float, xp
                  ) -> Tuple[float, float]:
    """Dominant envelope-autocorrelation lag inside (2 ms, 0.9·T_ballpass).

    Returns (lag seconds, lag lag-domain confidence, raw autocorr trace is
    discarded). Lag ≈ entry→exit separation when a spall dominates.
    """
    n = env_pooled.shape[-1]
    lag_min = max(2, int(0.002 * fs_pool))
    lag_max = min(n - 2, int(0.9 * t_ballpass_s * fs_pool))
    if lag_max <= lag_min + 4:
        return 0.0, 0.0
    e = env_pooled - env_pooled.mean()
    S = xp.abs(xp.fft.rfft(e, n=2 * n)) ** 2
    r = xp.fft.irfft(S)[:n]
    r0 = float(r[0]) + 1e-24
    seg = r[lag_min:lag_max] / r0
    k = int(xp.argmax(seg))
    peak = float(seg[k])
    med = float(xp.median(seg))
    mad = float(xp.median(xp.abs(seg - med))) + 1e-12
    conf = max(0.0, math.tanh((peak - med) / (10.0 * mad)))
    return (lag_min + k) / fs_pool, conf


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class PhysFeatureExtractor:
    """Per-run physics features. `raw` is the full recording (T, C) float32
    with channels [vib_y_A, vib_x_A, vib_y_B, vib_x_B, acoustic]."""

    def __init__(self,
                 fs: float = 38_500.0,
                 speed: str = "1rpm",
                 nb: int = NB_DEFAULT,
                 bd_mm: float = BD_MM_DEFAULT,
                 pd_mm: float = PD_MM_DEFAULT,
                 contact_angle_deg: float = CONTACT_ANGLE_DEG_DEFAULT,
                 use_gpu: str = "auto",
                 target_env_rate_hz: float = 150.0) -> None:
        self.fs = float(fs)
        self.f_shaft = SPEED_HZ.get(speed)
        if self.f_shaft is None:
            raise KeyError(f"unknown speed {speed!r}; known: {sorted(SPEED_HZ)}")
        self.freqs = fault_frequencies(self.f_shaft, nb, bd_mm, pd_mm,
                                       contact_angle_deg)
        self.v_sweep = rolling_sweep_speed_mps(self.f_shaft, bd_mm, pd_mm,
                                               contact_angle_deg)
        self.pool = max(1, int(self.fs / target_env_rate_hz))
        self.fs_pool = self.fs / self.pool
        if use_gpu == "auto":
            self.gpu = _HAS_CUPY
        else:
            self.gpu = bool(use_gpu) and _HAS_CUPY

    @property
    def xp(self):
        return _cupy if self.gpu else np

    def extract_run(self, raw: np.ndarray) -> np.ndarray:
        """(T, C≥5) float → (PHYS_DIM,) float32. Non-finite → 0."""
        xp = self.xp
        x = xp.asarray(np.ascontiguousarray(raw, dtype=np.float32).T)  # (C, T)
        feats: List[float] = []

        env = _envelope(x, xp)                            # (C, T)
        env_pooled = _mean_pool(env, self.pool, xp)       # (C, Tp)

        # High-band energy ratio needs a raw spectrum split at 5 kHz
        n = x.shape[-1]
        A2 = xp.abs(xp.fft.rfft(x, axis=-1)) ** 2
        fbins = xp.fft.rfftfreq(n, d=1.0 / self.fs)
        hi_mask = fbins >= 5_000.0

        for group in ((0, 1, 2, 3), (4,)):
            idx = list(g for g in group if g < x.shape[0])
            # SNR features: best (max) channel of the group per feature
            snr_rows = [
                _env_snr_features(env_pooled[c], self.fs_pool, self.freqs, xp)
                for c in idx
            ]
            feats.extend(np.max(np.asarray(snr_rows), axis=0).tolist())
            xg = x[idx]
            envg = env_pooled[idx]
            kurt = float(xp.asarray(_kurtosis(xg, xp)).max())
            rms = xp.sqrt((xg * xg).mean(axis=-1))
            crest = float((xp.abs(xg).max(axis=-1) / (rms + 1e-12)).max())
            ek = float(xp.asarray(_kurtosis(envg, xp)).max())
            hbr = float((A2[idx][:, hi_mask].sum(axis=-1)
                         / (A2[idx].sum(axis=-1) + 1e-24)).max())
            lrms = float(xp.log(rms + 1e-12).max())
            feats.extend([kurt, crest, ek, hbr, lrms])

        # Dual-impulse on the best vib channel (highest envelope kurtosis)
        vib_idx = [g for g in (0, 1, 2, 3) if g < x.shape[0]]
        ek_per = _kurtosis(env_pooled[vib_idx], xp)
        best = vib_idx[int(xp.asarray(ek_per).argmax())]
        t_bp = 1.0 / max(self.freqs["BPFO"], 1e-9)
        dt, conf = _dual_impulse(env_pooled[best], self.fs_pool, t_bp, xp)
        spall_len_mm = self.v_sweep * dt * 1e3
        feats.extend([dt, math.log(spall_len_mm + 1e-3), conf])

        out = np.asarray(feats, dtype=np.float32)
        out[~np.isfinite(out)] = 0.0
        assert out.shape == (PHYS_DIM,), f"{out.shape} != ({PHYS_DIM},)"
        return out


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    fs = 38_500.0
    fx = PhysFeatureExtractor(fs=fs, speed="1rpm", use_gpu="auto")
    print(f"backend: {'cupy' if fx.gpu else 'numpy'} | PHYS_DIM={PHYS_DIM}")
    print("char freqs @1rpm:", {k: f"{v:.4f} Hz" for k, v in fx.freqs.items()})
    print(f"sweep speed: {fx.v_sweep*1e3:.2f} mm/s")

    # Synthetic 60 s recording: BPFO-periodic dual impulses on channel 0
    T = int(60 * fs)
    t = np.arange(T) / fs
    sig = 0.1 * rng.standard_normal((T, 5)).astype(np.float32)
    bpfo = fx.freqs["BPFO"]
    dt_true = 0.4  # entry→exit separation [s] → ~1.2 mm spall at 1 rpm
    for k in range(int(60 * bpfo)):
        for off in (0.0, dt_true):
            i = int((k / bpfo + off) * fs)
            if i + 300 < T:
                sig[i:i + 300, 0] += (np.exp(-np.arange(300) / 60.0)
                                      * np.sin(2 * np.pi * 6_000 * t[:300])
                                      ).astype(np.float32) * 3.0
    f = fx.extract_run(sig)
    named = dict(zip(PHYS_NAMES, f.tolist()))
    print(f"vib BPFO h1 SNR : {named['vib_env_snr_BPFO_h1']:.3f} (expect >> ac)")
    print(f"ac  BPFO h1 SNR : {named['ac_env_snr_BPFO_h1']:.3f}")
    print(f"dual-impulse dt : {named['dual_impulse_dt_s']:.3f}s "
          f"(true {dt_true}s) conf={named['dual_impulse_conf']:.2f}")
    print(f"kurtosis vib/ac : {named['vib_kurtosis']:.1f} / {named['ac_kurtosis']:.1f}")
    # Short-signal guard (test-suite scale)
    f2 = PhysFeatureExtractor(fs=fs, speed="3rpm").extract_run(
        rng.standard_normal((4096, 5)).astype(np.float32))
    assert f2.shape == (PHYS_DIM,) and np.isfinite(f2).all()
    print("[OK] signal_physics self-test passed")
