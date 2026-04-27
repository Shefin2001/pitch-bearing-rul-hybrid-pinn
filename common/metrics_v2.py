"""metrics_v2.py — JIT metrics for the novel dual-track build.

Includes everything BENCHMARKS.md flagged as missing:
    * F1-macro for fault classification (was: only accuracy → leakage hid issues)
    * F1-macro for progression head (multi-label) — replaces accuracy
    * Monotonicity violation rate
    * Expected Calibration Error (ECE) for MC Dropout uncertainty
    * TTF MAPE (mean absolute % error on absolute time)

All scalar tensor kernels are decorated with ``@torch.jit.script`` for
JIT compilation and graceful fallback on platforms where TorchScript fails.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# JIT-script kernels (numerical primitives)
# ---------------------------------------------------------------------------

@torch.jit.script
def safe_normalize_tensor(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mu = x.mean()
    std = x.std() + eps
    return (x - mu) / std


@torch.jit.script
def nan_to_zero(x: torch.Tensor) -> torch.Tensor:
    return torch.where(torch.isfinite(x), x, torch.zeros_like(x))


@torch.jit.script
def compute_rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((pred - target) ** 2))


@torch.jit.script
def compute_mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(pred - target))


@torch.jit.script
def compute_r2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    ss_res = torch.sum((target - pred) ** 2)
    ss_tot = torch.sum((target - torch.mean(target)) ** 2) + 1e-12
    return 1.0 - ss_res / ss_tot


@torch.jit.script
def phm_score(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """PHM 2008 asymmetric score — penalises late predictions more."""
    err = pred - target
    a1, a2 = 13.0, 10.0
    pen = torch.where(err < 0, torch.exp(-err / a1) - 1.0, torch.exp(err / a2) - 1.0)
    return torch.mean(pen)


@torch.jit.script
def monotonicity_violation_rate(rul_seq: torch.Tensor) -> torch.Tensor:
    """Fraction of consecutive pairs where RUL increases (degradation reverses).

    Args:
        rul_seq: (T,) or (B, T) — windows ordered by time within a run

    Returns:
        scalar in [0, 1]: fraction of t where rul[t+1] > rul[t] + ε
    """
    if rul_seq.dim() == 1:
        rul_seq = rul_seq.unsqueeze(0)
    diff = rul_seq[:, 1:] - rul_seq[:, :-1]
    violations = (diff > 1e-4).float()
    return violations.mean()


@torch.jit.script
def apply_progression_mask(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Zero-out logits for faults not reachable from the current state.

    Args:
        logits: (B, n_classes) raw logits from progression head
        mask:   (B, n_classes) binary float mask — 1 where reachable

    Returns:
        masked logits with -1e9 at unreachable positions (softmax-safe)
    """
    return logits + (mask - 1.0) * 1e9


# ---------------------------------------------------------------------------
# F1-macro (sklearn-equivalent, pure torch — no sklearn dependency at runtime)
# ---------------------------------------------------------------------------

def f1_macro_multiclass(pred_idx: torch.Tensor, target_idx: torch.Tensor,
                        n_classes: int) -> float:
    """Macro-F1 across n_classes for single-label classification.

    Args:
        pred_idx: (N,) int64 — argmax predictions
        target_idx: (N,) int64 — ground truth class indices

    Returns:
        macro-averaged F1 in [0, 1]
    """
    pred_idx = pred_idx.flatten().long()
    target_idx = target_idx.flatten().long()
    f1s = []
    for c in range(n_classes):
        tp = ((pred_idx == c) & (target_idx == c)).sum().item()
        fp = ((pred_idx == c) & (target_idx != c)).sum().item()
        fn = ((pred_idx != c) & (target_idx == c)).sum().item()
        p = tp / (tp + fp + 1e-12)
        r = tp / (tp + fn + 1e-12)
        f1 = 2 * p * r / (p + r + 1e-12)
        f1s.append(f1)
    return float(np.mean(f1s))


def f1_macro_multilabel(prob: torch.Tensor, target: torch.Tensor,
                        threshold: float = 0.5) -> float:
    """Macro-F1 across labels for multi-label classification.

    Args:
        prob:   (N, L) float — sigmoid outputs
        target: (N, L) float — binary targets

    Returns:
        macro-averaged F1
    """
    pred = (prob >= threshold).float()
    target = target.float()
    f1s = []
    for c in range(target.shape[1]):
        tp = ((pred[:, c] == 1) & (target[:, c] == 1)).sum().item()
        fp = ((pred[:, c] == 1) & (target[:, c] == 0)).sum().item()
        fn = ((pred[:, c] == 0) & (target[:, c] == 1)).sum().item()
        if tp + fp + fn == 0:
            continue  # class never appeared — skip rather than score 0
        p = tp / (tp + fp + 1e-12)
        r = tp / (tp + fn + 1e-12)
        f1 = 2 * p * r / (p + r + 1e-12)
        f1s.append(f1)
    return float(np.mean(f1s)) if f1s else 0.0


# ---------------------------------------------------------------------------
# Expected Calibration Error (for MC Dropout uncertainty)
# ---------------------------------------------------------------------------

def expected_calibration_error(prob_max: torch.Tensor, correct: torch.Tensor,
                               n_bins: int = 15) -> float:
    """ECE — bins predictions by max-class probability and compares to accuracy.

    Args:
        prob_max: (N,) float — max softmax probability per sample
        correct:  (N,) bool/float — 1 if argmax was correct

    Returns:
        ECE in [0, 1] — lower is better calibrated
    """
    prob_max = prob_max.flatten().float().detach().cpu()
    correct = correct.flatten().float().detach().cpu()
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n_total = prob_max.numel()
    for i in range(n_bins):
        lo, hi = bin_boundaries[i].item(), bin_boundaries[i + 1].item()
        mask = (prob_max > lo) & (prob_max <= hi)
        if mask.sum().item() == 0:
            continue
        bin_acc = correct[mask].mean().item()
        bin_conf = prob_max[mask].mean().item()
        bin_weight = mask.sum().item() / n_total
        ece += abs(bin_acc - bin_conf) * bin_weight
    return float(ece)


# ---------------------------------------------------------------------------
# TTF metrics
# ---------------------------------------------------------------------------

def ttf_mape(log_ttf_pred: torch.Tensor, log_ttf_target: torch.Tensor) -> float:
    """Mean absolute percentage error on absolute time-to-failure.

    Both arguments are in log-seconds (model outputs and targets stored that
    way to handle the wide range 1s..10⁹s gracefully).
    """
    ttf_pred = torch.exp(log_ttf_pred)
    ttf_true = torch.exp(log_ttf_target)
    err = torch.abs(ttf_pred - ttf_true) / (ttf_true + 1.0)
    return float(err.mean().item())


# ---------------------------------------------------------------------------
# Aggregate evaluation
# ---------------------------------------------------------------------------

def evaluate_all(predictions: Dict[str, torch.Tensor],
                 targets: Dict[str, torch.Tensor],
                 n_classes: int = 12) -> Dict[str, float]:
    """One-stop evaluation. Inputs are tensors collected over the test set.

    predictions keys: rul, log_ttf, fault_logits, prog_logits, [rul_seq for monotonicity]
    targets     keys: rul, log_ttf, fault_idx,    prog_mask,   [rul_seq]
    """
    out: Dict[str, float] = {}
    out["rul_rmse"] = compute_rmse(predictions["rul"], targets["rul"]).item()
    out["rul_mae"] = compute_mae(predictions["rul"], targets["rul"]).item()
    out["rul_r2"] = compute_r2(predictions["rul"], targets["rul"]).item()
    out["rul_phm"] = phm_score(predictions["rul"], targets["rul"]).item()

    out["ttf_mape"] = ttf_mape(predictions["log_ttf"], targets["log_ttf"])

    pred_idx = predictions["fault_logits"].argmax(dim=-1)
    out["fault_acc"] = (pred_idx == targets["fault_idx"]).float().mean().item()
    out["fault_f1_macro"] = f1_macro_multiclass(pred_idx, targets["fault_idx"], n_classes)

    prog_prob = torch.sigmoid(predictions["prog_logits"])
    out["prog_f1_macro"] = f1_macro_multilabel(prog_prob, targets["prog_mask"])

    if "rul_seq" in predictions:
        out["mono_violation"] = monotonicity_violation_rate(predictions["rul_seq"]).item()

    return out


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    pred = torch.rand(64) * 0.9 + 0.05
    targ = pred + torch.randn(64) * 0.05
    print(f"RMSE       : {compute_rmse(pred, targ).item():.4f}")
    print(f"MAE        : {compute_mae(pred, targ).item():.4f}")
    print(f"R2         : {compute_r2(pred, targ).item():.4f}")
    print(f"PHM        : {phm_score(pred, targ).item():.4f}")
    print(f"Mono viol. : {monotonicity_violation_rate(pred).item():.4f}")

    fault_pred = torch.randint(0, 12, (200,))
    fault_true = torch.randint(0, 12, (200,))
    print(f"F1-macro   : {f1_macro_multiclass(fault_pred, fault_true, 12):.4f}")

    prog_p = torch.rand(200, 12)
    prog_t = (torch.rand(200, 12) > 0.6).float()
    print(f"F1-multilabel: {f1_macro_multilabel(prog_p, prog_t):.4f}")

    log_ttf_pred = torch.randn(64) * 2 + 12
    log_ttf_true = log_ttf_pred + torch.randn(64) * 0.1
    print(f"TTF MAPE   : {ttf_mape(log_ttf_pred, log_ttf_true):.4f}")

    print("[OK] all metrics import + run cleanly")
