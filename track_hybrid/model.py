"""track_hybrid/model.py — Hybrid TCN-Transformer + Mixed-MLP + Cross-Attention.

Replaces v1 MSTCAN entirely. Architecture:

    Raw signal (B, 5, 2048)
        → TCN encoder (4 dilated blocks: 1, 2, 4, 8)
        → Positional encoding
        → Transformer encoder (2 layers, d_model=256)
        → mean-pool over time → e_raw (B, 256)

    160-D engineered features (B, 160) [grouped]
        → Group embeddings (time / freq / TF / acoustic)
        → Channel-Temporal Mixed MLP (factorised mixing)
        → e_feat (B, 256)

    Fusion
        → Cross-attention: query = e_raw, key/value = e_feat
        → SE channel attention
        → e_fused (B, 256)

    Heads (4)
        RUL_relative  : Linear → Sigmoid              (B, 1) ∈ [0,1]
        log_TTF       : Linear                        (B, 1)  log(seconds)
        fault_logits  : Linear                        (B, 12)
        prog_logits   : Linear                        (B, 12) [DAG-masked at use]

Target: ~30M params on A100/H100.

References:
    * TCN-Transformer for bearing RUL — Sensors 2025 (Source 4 in BENCHMARKS.md)
    * Channel-Temporal Mixed MLP — Springer AIS 2024 (Source 10)
    * SE channel attention — Pitch bearing AE+vib, Measurement 2024 (Source 6)
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# Building blocks
# ===========================================================================

class CausalConv1d(nn.Conv1d):
    """1-D causal convolution — pads only on the left so output[t] depends on
    input[≤ t]. Used by the TCN encoder."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int = 1):
        self._left_pad = (kernel - 1) * dilation
        super().__init__(in_ch, out_ch, kernel_size=kernel, padding=0,
                         dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(F.pad(x, (self._left_pad, 0)))


class TCNBlock(nn.Module):
    """One TCN residual block: dilated causal conv → BN → GELU → conv → BN → +residual."""

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
        y = y + self.skip(x)
        return F.gelu(y)


class TCNEncoder(nn.Module):
    """4 TCN blocks with exponentially-growing dilation, plus stride downsampling.

    Input  : (B, 5, 2048)
    Output : (B, 256, 256)   — time downsampled 8×, channels 5→256
    """

    def __init__(self, in_ch: int = 5, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        widths = [64, 128, 192, hidden]
        dilations = [1, 2, 4, 8]
        layers = []
        prev = in_ch
        for w, d in zip(widths, dilations):
            layers.append(TCNBlock(prev, w, kernel=7, dilation=d, dropout=dropout))
            layers.append(nn.AvgPool1d(kernel_size=2, stride=2))  # downsample 2×
            prev = w
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding, fixed."""

    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


# ---------------------------------------------------------------------------
# Channel-Temporal Mixed MLP (Source 10 — replaces FAN attention)
# ---------------------------------------------------------------------------

class MixedMLPBlock(nn.Module):
    """Factorised mixing: across channels then across feature groups.

    Input/Output: (B, n_groups, d_group)
    """

    def __init__(self, n_groups: int, d_group: int, mlp_ratio: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_group)
        self.token_mix = nn.Sequential(
            nn.Linear(n_groups, n_groups * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(n_groups * mlp_ratio, n_groups),
        )
        self.norm2 = nn.LayerNorm(d_group)
        self.channel_mix = nn.Sequential(
            nn.Linear(d_group, d_group * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_group * mlp_ratio, d_group),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Token mixing across groups (transpose, mix, transpose back)
        z = self.norm1(x).transpose(1, 2)         # (B, d_group, n_groups)
        z = self.token_mix(z).transpose(1, 2)     # (B, n_groups, d_group)
        x = x + z
        # Channel mixing within each group
        x = x + self.channel_mix(self.norm2(x))
        return x


class FeatureBranch(nn.Module):
    """160-D engineered features → group-embed → 3× MixedMLPBlocks → (B, 256).

    Group sizes match feature_extractor.py from approach_2:
        time     :  55 features (11 stats × 5 channels)
        freq     :  60 features (12 stats × 5 channels)
        timefreq :  40 features (8 stats × 5 channels — STFT + DWT)
        acoustic :   5 features (acoustic-only)
    Total: 160
    """

    def __init__(self, d_group: int = 64, mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        self.time_emb = nn.Sequential(nn.Linear(55, d_group), nn.LayerNorm(d_group), nn.GELU())
        self.freq_emb = nn.Sequential(nn.Linear(60, d_group), nn.LayerNorm(d_group), nn.GELU())
        self.tf_emb = nn.Sequential(nn.Linear(40, d_group), nn.LayerNorm(d_group), nn.GELU())
        self.acou_emb = nn.Sequential(nn.Linear(5, d_group), nn.LayerNorm(d_group), nn.GELU())
        self.blocks = nn.Sequential(
            MixedMLPBlock(4, d_group, mlp_ratio, dropout),
            MixedMLPBlock(4, d_group, mlp_ratio, dropout),
            MixedMLPBlock(4, d_group, mlp_ratio, dropout),
        )
        self.proj = nn.Linear(4 * d_group, 256)

    def forward(self, feat160: torch.Tensor) -> torch.Tensor:
        # Split 160 → (55, 60, 40, 5)
        t, f, tf, a = torch.split(feat160, [55, 60, 40, 5], dim=-1)
        e = torch.stack([self.time_emb(t), self.freq_emb(f),
                         self.tf_emb(tf), self.acou_emb(a)], dim=1)  # (B, 4, d_group)
        e = self.blocks(e)
        e = e.flatten(1)                      # (B, 4 * d_group)
        return self.proj(e)                   # (B, 256)


# ---------------------------------------------------------------------------
# SE channel attention (Source 6 — pitch bearing AE+vib)
# ---------------------------------------------------------------------------

class SEAttention(nn.Module):
    def __init__(self, dim: int, reduction: int = 16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x)


# ---------------------------------------------------------------------------
# Cross-attention fusion
# ---------------------------------------------------------------------------

class CrossAttnFusion(nn.Module):
    """One layer of multi-head cross-attention: query=e_raw, kv=e_feat."""

    def __init__(self, dim: int = 256, nhead: int = 8, dropout: float = 0.1):
        super().__init__()
        self.q_norm = nn.LayerNorm(dim)
        self.kv_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, nhead, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, e_raw: torch.Tensor, e_feat: torch.Tensor) -> torch.Tensor:
        q = self.q_norm(e_raw).unsqueeze(1)
        kv = self.kv_norm(e_feat).unsqueeze(1)
        a, _ = self.attn(q, kv, kv, need_weights=False)
        a = a.squeeze(1)
        return e_raw + a + self.ff(e_raw + a)


# ===========================================================================
# Full Hybrid model
# ===========================================================================

class HybridParisModel(nn.Module):
    """End-to-end Hybrid network with 4 heads.

    Args:
        n_classes: number of fault classes (default 12)
        d_model:   shared embedding dimension (default 256)
        nhead:     transformer / fusion heads (default 8)
        dropout:   global dropout rate (default 0.2 — also drives MC Dropout
                   uncertainty at inference)
    """

    def __init__(self, n_classes: int = 12, d_model: int = 256,
                 n_transformer_layers: int = 2, nhead: int = 8,
                 dropout: float = 0.2):
        super().__init__()
        self.d_model = d_model
        self.n_classes = n_classes
        self.dropout_p = dropout

        # ---- Raw branch ----
        self.tcn = TCNEncoder(in_ch=5, hidden=d_model, dropout=dropout)
        self.pos_enc = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_transformer_layers)

        # ---- Feature branch ----
        self.feat_branch = FeatureBranch(d_group=64, mlp_ratio=4, dropout=dropout)

        # ---- Fusion ----
        self.fusion = CrossAttnFusion(d_model, nhead, dropout)
        self.se = SEAttention(d_model)

        # ---- Heads ----
        self.head_rul = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        self.head_log_ttf = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        self.head_fault = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes),
        )
        self.head_prog = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes),
        )

    def forward(self, x_raw: torch.Tensor, x_feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x_raw:  (B, 5, 2048) bandpass-filtered, z-scored windows
            x_feat: (B, 160) engineered features

        Returns:
            dict with rul, log_ttf, fault_logits, prog_logits, embedding
        """
        # Raw branch
        z = self.tcn(x_raw)                           # (B, 256, 256)
        z = z.transpose(1, 2)                         # (B, T=256, d=256)
        z = self.pos_enc(z)
        z = self.transformer(z)                       # (B, T, d)
        e_raw = z.mean(dim=1)                         # (B, d)

        # Feature branch
        e_feat = self.feat_branch(x_feat)             # (B, d)

        # Fusion
        e = self.fusion(e_raw, e_feat)
        e = self.se(e)

        return {
            "rul": torch.sigmoid(self.head_rul(e)).squeeze(-1),
            "log_ttf": self.head_log_ttf(e).squeeze(-1),
            "fault_logits": self.head_fault(e),
            "prog_logits": self.head_prog(e),
            "embedding": e,
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def enable_mc_dropout(self) -> None:
        """Keep dropout ON during eval — for MC Dropout uncertainty estimation."""
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()


# ===========================================================================
# Self-test
# ===========================================================================

if __name__ == "__main__":
    torch.manual_seed(0)
    model = HybridParisModel()
    n_params = model.count_parameters()
    print(f"HybridParisModel parameters: {n_params:,} ({n_params / 1e6:.1f} M)")

    x_raw = torch.randn(4, 5, 2048)
    x_feat = torch.randn(4, 160)
    out = model(x_raw, x_feat)
    for k, v in out.items():
        print(f"  {k:14s}: {tuple(v.shape)}")

    # JIT export smoke test
    try:
        scripted = torch.jit.script(model)
        print(f"[OK] torch.jit.script export succeeded ({type(scripted).__name__})")
    except Exception as e:
        print(f"[!] torch.jit.script failed: {e}")
