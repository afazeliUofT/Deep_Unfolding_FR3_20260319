"""TWC-grade FR3 coexistence extension package.

This package is additive to the existing ``fr3_sim`` baseline package and provides:
- proportional-fair (PF) weighted WMMSE baselines
- topology-aware cognitive tone masking for FS protection
- trainable deep-unfolded WMMSE layers
- Sionna-based 5G NR FER evaluation helpers
- flat-vs-selective diagnostics and publication-grade plotting helpers
"""

__all__ = [
    "config",
    "common",
    "fs_masks",
    "metrics_ext",
    "solver",
    "unfolding",
    "fer",
    "selectivity",
    "plotting",
    "reporting",
]
