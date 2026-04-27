"""Hybrid_PINN_ParisRUL — Novel dual-track pitch bearing RUL with absolute-time output.

Tracks:
    track_hybrid : TCN-Transformer + Mixed-MLP + cross-attention fusion (~30M params)
    track_pinn   : Physics-informed Paris-law constrained network (~10M params)
    track_fusion : Distillation student + dual edge/cloud export

Data:
    common/rul_labels_v2  : FPT-piecewise-linear RUL per recording
    common/paris_labels   : Hertzian + Paris-law synthetic absolute-time labels
    common/dataset_v2     : Run-level split, shared test index
    common/metrics_v2     : F1-macro, monotonicity, ECE, TTF MAPE
"""
__version__ = "2.0.0"
