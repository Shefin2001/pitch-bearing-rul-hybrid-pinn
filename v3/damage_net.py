"""damage_net.py -- the only trained model in the v3 track (~0.5-2 M params).

Predicts the *current physical damage state* of the bearing from one window:
    fault_logits   (12,)  -- which failure mode (proven learnable, v2 F1 0.997)
    corn_logits    (K-1,) -- ordinal severity level (rank-consistent CORN)
    log_a_mu/var   ()     -- ln(crack length [m]) with heteroscedastic variance

No RUL / TTF head: time is computed downstream by the Paris engine, so the
network cannot memorise window position (v2's failure mode).

Inputs:
    x_raw  (B, 5, 2048)  -- band-passed z-scored window (raw branch)
    x_feat (B, 160)      -- v2 engineered features
    x_phys (B, PHYS_DIM) -- run-level physics evidence (z-scored)

Backbone reuses the proven TCNBlock from track_pinn.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from Hybrid_PINN_ParisRUL.track_pinn.model import TCNBlock  # noqa: E402
from Hybrid_PINN_ParisRUL.v3.severity_axis import N_LEVELS  # noqa: E402
from Hybrid_PINN_ParisRUL.v3.signal_physics import PHYS_DIM  # noqa: E402


class DamageNet(nn.Module):
    def __init__(self, n_classes: int = 12, n_levels: int = N_LEVELS,
                 phys_dim: int = PHYS_DIM, hidden: int = 128,
                 dropout: float = 0.15) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.n_levels = n_levels
        self.phys_dim = phys_dim

        widths = [32, 64, hidden]
        dilations = [1, 2, 4]
        layers = []
        prev = 5
        for w, d in zip(widths, dilations):
            layers.append(TCNBlock(prev, w, kernel=7, dilation=d, dropout=dropout))
            layers.append(nn.AvgPool1d(2, 2))
            prev = w
        self.tcn = nn.Sequential(*layers)

        self.feat_mlp = nn.Sequential(
            nn.Linear(160, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, hidden),
        )
        self.phys_mlp = nn.Sequential(
            nn.Linear(phys_dim, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, hidden),
        )
        self.fuse = nn.Sequential(
            nn.Linear(3 * hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Dropout(dropout),
        )

        self.head_fault = nn.Linear(hidden, n_classes)
        self.head_corn = nn.Linear(hidden, n_levels - 1)
        self.head_log_a = nn.Linear(hidden, 2)  # [mu, log_var]

        # Centre the damage head at the geometric middle of the anchor range
        # (ln of ~1 mm) so early training starts inside plausible intervals.
        nn.init.zeros_(self.head_log_a.weight)
        with torch.no_grad():
            self.head_log_a.bias.copy_(torch.tensor([-6.9, 0.0]))

    def forward(self, x_raw: torch.Tensor, x_feat: torch.Tensor,
                x_phys: torch.Tensor) -> Dict[str, torch.Tensor]:
        e_raw = self.tcn(x_raw).mean(dim=-1)
        e_feat = self.feat_mlp(x_feat)
        e_phys = self.phys_mlp(x_phys)
        e = self.fuse(torch.cat([e_raw, e_feat, e_phys], dim=-1))
        log_a = self.head_log_a(e)
        return {
            "fault_logits": self.head_fault(e),
            "corn_logits": self.head_corn(e),
            "log_a_mu": log_a[:, 0],
            "log_a_log_var": log_a[:, 1],
            "embedding": e,
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    torch.manual_seed(0)
    m = DamageNet()
    n = m.count_parameters()
    out = m(torch.randn(4, 5, 2048), torch.randn(4, 160), torch.randn(4, PHYS_DIM))
    for k, v in out.items():
        print(f"  {k:15s}: {tuple(v.shape)}")
    a_mm = torch.exp(out["log_a_mu"]).mean().item() * 1e3
    print(f"DamageNet params: {n:,} ({n/1e6:.2f} M) | init a ~ {a_mm:.2f} mm")
    assert out["fault_logits"].shape == (4, 12)
    assert out["corn_logits"].shape == (4, N_LEVELS - 1)
    print("[OK] damage_net self-test passed")
