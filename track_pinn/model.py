"""track_pinn/model.py — Physics-Informed Paris-law NN.

Smaller TCN backbone (~10M params) feeding 6 heads. Two of the heads predict
intermediate physics states (crack length a, stress range Δσ). Two global
learnable scalars hold the population-effective Paris constants C and m.
The training loss enforces the Paris residual on consecutive windows in a run:

    da/dN_predicted  =  C · (Y · Δσ · √(πa))^m
    da/dN_observed   =  (a[t+1] − a[t]) / cycles_between

This lets the model learn the material's effective fatigue law from CUMTB
data while staying physically consistent.
"""
from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Conv1d):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int = 1):
        self._left_pad = (kernel - 1) * dilation
        super().__init__(in_ch, out_ch, kernel, padding=0, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(F.pad(x, (self._left_pad, 0)))


class TCNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 7, dilation: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        self.conv1 = CausalConv1d(in_ch, out_ch, kernel, dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = CausalConv1d(out_ch, out_ch, kernel, dilation)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.gelu(self.bn1(self.conv1(x)))
        y = self.drop(y)
        y = self.bn2(self.conv2(y))
        return F.gelu(y + self.skip(x))


class PINNModel(nn.Module):
    """Physics-informed PINN for pitch-bearing RUL.

    Heads:
      rul             ∈ [0, 1]
      log_ttf         in seconds (log)
      crack_a_mm      ∈ (0, ∞)  — predicted crack length in mm (physics state)
      delta_sigma_MPa ∈ (0, ∞)  — predicted stress range (physics state)
      fault_logits    (12,)
      prog_logits     (12,)

    Globals (learnable scalars):
      C_paris   in [1e-14, 1e-9] via softplus
      m_paris   in [2.0, 5.0]   via sigmoid affine
    """

    def __init__(self, n_classes: int = 12, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.n_classes = n_classes
        self.dropout_p = dropout

        # ---- TCN backbone (smaller than Hybrid) ----
        widths = [32, 64, hidden]
        dilations = [1, 2, 4]
        layers = []
        prev = 5
        for w, d in zip(widths, dilations):
            layers.append(TCNBlock(prev, w, kernel=7, dilation=d, dropout=dropout))
            layers.append(nn.AvgPool1d(2, 2))
            prev = w
        self.tcn = nn.Sequential(*layers)

        # ---- Feature mixer for engineered features (small) ----
        self.feat_mlp = nn.Sequential(
            nn.Linear(160, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, hidden),
        )

        # ---- Fusion (concat + projection) ----
        self.fuse = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
        )

        # ---- Heads ----
        self.head_rul = nn.Linear(hidden, 1)
        self.head_log_ttf = nn.Linear(hidden, 1)
        self.head_a = nn.Linear(hidden, 1)
        self.head_ds = nn.Linear(hidden, 1)
        self.head_fault = nn.Linear(hidden, n_classes)
        self.head_prog = nn.Linear(hidden, n_classes)

        # ---- Learnable Paris constants ----
        # log(C) parameterisation — C = 10^(-12) * 10^(log_C_offset)
        # Initialise near literature value C ≈ 6.9e-12 for 42CrMo4
        self.log_C_offset = nn.Parameter(torch.tensor(math.log10(6.9), dtype=torch.float32))
        # m bounded to [2, 5] via sigmoid(2 + 3 * sigmoid(raw))
        self.m_raw = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def C_paris(self) -> torch.Tensor:
        return 10.0 ** (-12.0 + torch.clamp(self.log_C_offset, -2.0, 2.0))

    def m_paris(self) -> torch.Tensor:
        return 2.0 + 3.0 * torch.sigmoid(self.m_raw)

    def forward(self, x_raw: torch.Tensor, x_feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Raw branch
        z = self.tcn(x_raw)              # (B, hidden, T')
        e_raw = z.mean(dim=-1)            # (B, hidden)
        # Feature branch
        e_feat = self.feat_mlp(x_feat)    # (B, hidden)
        # Fuse
        e = self.fuse(torch.cat([e_raw, e_feat], dim=-1))

        rul = torch.sigmoid(self.head_rul(e)).squeeze(-1)
        log_ttf = self.head_log_ttf(e).squeeze(-1)
        crack_a_mm = F.softplus(self.head_a(e)).squeeze(-1) + 0.05  # min 0.05 mm
        delta_sigma_MPa = F.softplus(self.head_ds(e)).squeeze(-1) + 1.0
        fault_logits = self.head_fault(e)
        prog_logits = self.head_prog(e)

        return {
            "rul": rul,
            "log_ttf": log_ttf,
            "crack_a_mm": crack_a_mm,
            "delta_sigma_MPa": delta_sigma_MPa,
            "fault_logits": fault_logits,
            "prog_logits": prog_logits,
            "embedding": e,
            "C_paris": self.C_paris(),
            "m_paris": self.m_paris(),
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def enable_mc_dropout(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()


if __name__ == "__main__":
    torch.manual_seed(0)
    m = PINNModel()
    n = m.count_parameters()
    print(f"PINNModel params: {n:,} ({n / 1e6:.1f} M)")
    out = m(torch.randn(4, 5, 2048), torch.randn(4, 160))
    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:18s}: {tuple(v.shape) if v.dim() else 'scalar'} = {v.detach().mean().item():.4g}")
    print(f"  C_paris init = {m.C_paris().item():.3e}")
    print(f"  m_paris init = {m.m_paris().item():.3f}")
