"""track_hybrid/loss.py — Multi-task loss with monotonicity & PHM penalty."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HybridLossWeights:
    alpha: float = 1.0      # RUL relative MSE
    beta: float = 0.7       # log-TTF MSE
    gamma: float = 0.5      # fault CE
    delta: float = 0.3      # progression BCE
    lambda_mono: float = 0.2
    lambda_phm: float = 0.1


class HybridMultiTaskLoss(nn.Module):
    def __init__(self, w: HybridLossWeights = HybridLossWeights()):
        super().__init__()
        self.w = w

    def forward(self, pred: Dict[str, torch.Tensor],
                target: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # ---- Data-driven losses ----
        l_rul = F.mse_loss(pred["rul"], target["rul"])
        l_ttf = F.mse_loss(pred["log_ttf"], target["log_ttf"])
        l_fault = F.cross_entropy(pred["fault_logits"], target["fault_idx"])
        l_prog = F.binary_cross_entropy_with_logits(pred["prog_logits"], target["prog_mask"])

        # ---- PHM asymmetric penalty (penalises late predictions more) ----
        err = pred["rul"] - target["rul"]
        pen = torch.where(err < 0,
                          torch.exp(-err / 13.0) - 1.0,
                          torch.exp(err / 10.0) - 1.0)
        l_phm = pen.mean()

        # ---- Monotonicity (requires consecutive same-run windows in batch) ----
        # If batch has run_id + win_idx metadata, sort and compute Δrul on consecutive pairs.
        l_mono = torch.zeros((), device=pred["rul"].device)
        if "run_id" in target and "win_idx" in target:
            run_id = target["run_id"]
            win_idx = target["win_idx"]
            rul = pred["rul"]
            # Build per-run sequences: for each unique run in batch, sort by win_idx
            # and compute clamped positive jumps.
            unique_runs = torch.unique(run_id)
            mono_terms = []
            for r in unique_runs:
                mask = run_id == r
                if mask.sum() < 2:
                    continue
                w = win_idx[mask]
                v = rul[mask]
                order = torch.argsort(w)
                vs = v[order]
                d = vs[1:] - vs[:-1]
                mono_terms.append(torch.clamp(d, min=0).mean())
            if mono_terms:
                l_mono = torch.stack(mono_terms).mean()

        # ---- Combined ----
        w = self.w
        total = (w.alpha * l_rul +
                 w.beta * l_ttf +
                 w.gamma * l_fault +
                 w.delta * l_prog +
                 w.lambda_mono * l_mono +
                 w.lambda_phm * l_phm)

        return {
            "total": total,
            "l_rul": l_rul.detach(),
            "l_ttf": l_ttf.detach(),
            "l_fault": l_fault.detach(),
            "l_prog": l_prog.detach(),
            "l_mono": l_mono.detach(),
            "l_phm": l_phm.detach(),
        }
