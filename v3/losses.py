"""losses.py -- v3 DamageNet losses. No time-like target anywhere.

  fault  : cross-entropy over the 12 classes (the proven v2 task)
  corn   : CORN conditional ordinal loss over the 9 severity levels
           (Cao & Raschka -- rank-consistent ordinal regression)
  damage : interval-censored Gaussian NLL on ln(a). Class k only bounds the
           crack length inside [LOG_A_LO[k], LOG_A_HI[k]]; the model is free
           to place mu anywhere inside and must widen sigma when unsure.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F

_LOG_VAR_MIN, _LOG_VAR_MAX = -8.0, 4.0


def corn_loss(logits: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    """CORN loss. logits: (B, K-1); levels: (B,) int in [0, K-1].

    Task j predicts P(level > j | level > j-1); its training subset is the
    samples with level >= j, with binary target 1{level > j}.
    """
    n_tasks = logits.shape[1]
    total = logits.new_zeros(())
    n_terms = logits.new_zeros(())
    for j in range(n_tasks):
        subset = levels >= j
        if not bool(subset.any()):
            continue
        tgt = (levels[subset] > j).float()
        total = total + F.binary_cross_entropy_with_logits(
            logits[subset, j], tgt, reduction="sum")
        n_terms = n_terms + subset.sum()
    return total / n_terms.clamp_min(1)


def corn_level_probs(logits: torch.Tensor) -> torch.Tensor:
    """(B, K-1) logits -> (B, K) level probabilities via the CORN chain rule.

    P(level > j) = prod_{k<=j} sigmoid(logit_k);  P(level = j) is the
    difference of consecutive exceedance probabilities.
    """
    exceed = torch.cumprod(torch.sigmoid(logits), dim=1)      # (B, K-1)
    ones = exceed.new_ones(exceed.shape[0], 1)
    upper = torch.cat([ones, exceed], dim=1)                  # P(level > j-1)
    lower = torch.cat([exceed, exceed.new_zeros(exceed.shape[0], 1)], dim=1)
    return (upper - lower).clamp_min(0.0)


def corn_predict_level(logits: torch.Tensor) -> torch.Tensor:
    """Rank-consistent point prediction: count of exceedance probs > 0.5."""
    return (torch.cumprod(torch.sigmoid(logits), dim=1) > 0.5).sum(dim=1)


def interval_censored_nll(mu: torch.Tensor, log_var: torch.Tensor,
                          lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    """-log P(lo < X < hi) for X ~ N(mu, sigma^2), mean over the batch."""
    log_var = log_var.clamp(_LOG_VAR_MIN, _LOG_VAR_MAX)
    sigma = torch.exp(0.5 * log_var)
    p = torch.special.ndtr((hi - mu) / sigma) - torch.special.ndtr((lo - mu) / sigma)
    return -torch.log(p.clamp_min(1e-12)).mean()


@dataclass
class V3LossWeights:
    fault: float = 1.0
    corn: float = 1.0
    damage: float = 1.0


class DamageLoss(torch.nn.Module):
    """Total = w_f * CE + w_c * CORN + w_d * interval-censored NLL."""

    def __init__(self, weights: V3LossWeights | None = None) -> None:
        super().__init__()
        self.w = weights or V3LossWeights()

    def forward(self, pred: Dict[str, torch.Tensor],
                target: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        l_fault = F.cross_entropy(pred["fault_logits"], target["fault_idx"])
        l_corn = corn_loss(pred["corn_logits"], target["sev_level"])
        l_dmg = interval_censored_nll(pred["log_a_mu"], pred["log_a_log_var"],
                                      target["log_a_lo"], target["log_a_hi"])
        total = (self.w.fault * l_fault + self.w.corn * l_corn
                 + self.w.damage * l_dmg)
        return {"total": total, "fault": l_fault, "corn": l_corn,
                "damage": l_dmg}


if __name__ == "__main__":
    torch.manual_seed(0)
    B, K = 64, 9
    levels = torch.randint(0, K, (B,))
    logits = torch.randn(B, K - 1)
    l = corn_loss(logits, levels)
    probs = corn_level_probs(logits)
    assert torch.allclose(probs.sum(1), torch.ones(B), atol=1e-5)
    lv = corn_predict_level(logits)
    assert lv.min() >= 0 and lv.max() <= K - 1

    # ICNLL sanity: mu inside the interval with small sigma -> near-zero loss
    mu = torch.zeros(B)
    lo, hi = -torch.ones(B), torch.ones(B)
    good = interval_censored_nll(mu, torch.full((B,), -6.0), lo, hi)
    bad = interval_censored_nll(mu + 5.0, torch.full((B,), -6.0), lo, hi)
    assert good < 1e-3 < bad, (good.item(), bad.item())

    # Gradient flows through a full DamageLoss call
    pred = {"fault_logits": torch.randn(B, 12, requires_grad=True),
            "corn_logits": logits.clone().requires_grad_(True),
            "log_a_mu": mu.clone().requires_grad_(True),
            "log_a_log_var": torch.zeros(B, requires_grad=True)}
    target = {"fault_idx": torch.randint(0, 12, (B,)), "sev_level": levels,
              "log_a_lo": lo, "log_a_hi": hi}
    out = DamageLoss()(pred, target)
    out["total"].backward()
    assert pred["log_a_mu"].grad is not None
    print(f"corn={l.item():.4f} icnll(good)={good.item():.2e} "
          f"icnll(bad)={bad.item():.2f} total={out['total'].item():.4f}")
    print("[OK] losses self-test passed")
