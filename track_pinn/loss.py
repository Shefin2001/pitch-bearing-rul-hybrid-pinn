"""track_pinn/loss.py — PINN loss with Paris-law residual + monotonicity + BC.

Loss components:
    L_data     = MSE(rul) + MSE(log_ttf) + CE(fault) + BCE(prog)
    L_paris    = MSE(da/dN_pred,  da/dN_obs)
                 with da/dN_pred  = C · (Y · Δσ · √(πa))^m
                      da/dN_obs   = (a[t+1] − a[t]) / cycles_between
    L_mono     = mean(clamp(rul[t]   − rul[t-1],   min=0))   # RUL non-increasing
               + mean(clamp(a  [t-1] − a  [t],     min=0))   # crack non-decreasing
    L_BC       = MSE(a[run_start], a_init_class) + MSE(a[run_end], a_fail)

Total = L_data + λ_phys·L_paris + λ_mono·L_mono + λ_BC·L_BC
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from common.rul_labels import FAULT_INDEX, INDEX_FAULT  # noqa: E402
from Hybrid_PINN_ParisRUL.common.paris_labels import (  # noqa: E402
    A_FAIL_M, A_MAP_M, Y_GEOM,
)


@dataclass
class PINNLossWeights:
    alpha: float = 1.0       # rul mse
    beta: float = 0.5        # log_ttf mse
    gamma: float = 0.5       # fault CE
    delta: float = 0.3       # progression BCE
    lambda_phys: float = 0.4
    lambda_mono: float = 0.3
    lambda_BC: float = 0.2


# ---------------------------------------------------------------------------
# Per-class boundary-condition tensors
# ---------------------------------------------------------------------------

def _a_init_lookup() -> torch.Tensor:
    """Return (n_classes,) tensor of starting crack length in mm."""
    n = len(FAULT_INDEX)
    out = torch.zeros(n)
    for label, idx in FAULT_INDEX.items():
        out[idx] = float(A_MAP_M[label] * 1e3)  # m → mm
    return out


class PINNLoss(nn.Module):
    def __init__(self, w: PINNLossWeights = PINNLossWeights()):
        super().__init__()
        self.w = w
        self.register_buffer("a_init_per_class_mm", _a_init_lookup())
        self.a_fail_mm = float(A_FAIL_M * 1e3)
        self.cycle_seconds = 2.0

    def forward(self, pred: Dict[str, torch.Tensor],
                target: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # ---- Data losses ----
        l_rul = F.mse_loss(pred["rul"], target["rul"])
        l_ttf = F.mse_loss(pred["log_ttf"], target["log_ttf"])
        l_fault = F.cross_entropy(pred["fault_logits"], target["fault_idx"])
        l_prog = F.binary_cross_entropy_with_logits(pred["prog_logits"], target["prog_mask"])

        # ---- Boundary conditions on crack length ----
        # Predicted a should match per-class a_init for the START of each run
        # (window_idx == 0). We approximate this by anchoring every window's a
        # to the class-derived a_init weighted by (1 − rul) — i.e. early in
        # degradation, a should be near a_init; late, near a_fail.
        a_pred = pred["crack_a_mm"]
        a_init_target = self.a_init_per_class_mm.to(a_pred.device)[target["fault_idx"]]
        # Smooth interpolation between a_init and a_fail driven by 1 − rul
        a_target = a_init_target + (self.a_fail_mm - a_init_target) * (1.0 - target["rul"])
        l_BC = F.mse_loss(a_pred, a_target)

        # ---- Paris-law residual ----
        l_paris = torch.zeros((), device=a_pred.device)
        l_mono = torch.zeros((), device=a_pred.device)
        if "run_id" in target and "win_idx" in target:
            run_id = target["run_id"]
            win_idx = target["win_idx"]
            unique_runs = torch.unique(run_id)
            paris_terms, mono_terms = [], []
            C = pred["C_paris"]
            m = pred["m_paris"]
            for r in unique_runs:
                mask = run_id == r
                if mask.sum() < 2:
                    continue
                w_idx = win_idx[mask]
                order = torch.argsort(w_idx)
                a_seq = a_pred[mask][order]                # (T,) mm
                ds_seq = pred["delta_sigma_MPa"][mask][order]
                rul_seq = pred["rul"][mask][order]

                # Paris: convert mm → m for ΔK = Y · Δσ · √(πa)
                a_m = a_seq * 1e-3
                delta_K = Y_GEOM * ds_seq * torch.sqrt(np.pi * a_m + 1e-12)
                da_dN_pred = C * (delta_K + 1e-12) ** m  # m/cycle

                # Observed: (a[t+1] − a[t]) per cycle assuming each window is
                # one "macro cycle" of pitch motion (≈ 2s real-time)
                cycles_between = max(1.0, float(self.cycle_seconds))
                da_obs = (a_m[1:] - a_m[:-1]) / cycles_between
                paris_terms.append(F.mse_loss(da_dN_pred[:-1], da_obs))

                # Monotonicity: a non-decreasing, rul non-increasing
                mono_terms.append(torch.clamp(a_seq[:-1] - a_seq[1:], min=0).mean()
                                  + torch.clamp(rul_seq[1:] - rul_seq[:-1], min=0).mean())
            if paris_terms:
                l_paris = torch.stack(paris_terms).mean()
            if mono_terms:
                l_mono = torch.stack(mono_terms).mean()

        w = self.w
        total = (w.alpha * l_rul +
                 w.beta * l_ttf +
                 w.gamma * l_fault +
                 w.delta * l_prog +
                 w.lambda_phys * l_paris +
                 w.lambda_mono * l_mono +
                 w.lambda_BC * l_BC)

        return {
            "total": total,
            "l_rul": l_rul.detach(),
            "l_ttf": l_ttf.detach(),
            "l_fault": l_fault.detach(),
            "l_prog": l_prog.detach(),
            "l_paris": l_paris.detach(),
            "l_mono": l_mono.detach(),
            "l_BC": l_BC.detach(),
        }
