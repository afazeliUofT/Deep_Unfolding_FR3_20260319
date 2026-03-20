from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import tensorflow as tf

from fr3_sim.config import ResolvedConfig


def coherence_bandwidth_rules(tau_rms_ns: float) -> Dict[str, float]:
    """Rule-of-thumb coherence bandwidth estimates for diagnostics only."""
    tau_s = float(tau_rms_ns) * 1e-9
    return {
        "tau_rms_ns": float(tau_rms_ns),
        "bc50_hz": float(1.0 / max(5.0 * tau_s, 1e-12)),
        "bc90_hz": float(1.0 / max(50.0 * tau_s, 1e-12)),
    }


def tone_group_bandwidth_hz(cfg: ResolvedConfig, num_tones: Optional[int] = None) -> float:
    total_re = int(cfg.raw["system_model"]["num_re_total"])
    sim_re = int(num_tones or cfg.derived["num_re_sim"])
    scs = float(cfg.raw["system_model"]["subcarrier_spacing_hz"])
    return float(total_re / sim_re) * scs


def _freq_correlation(delta_f_hz: float, tau_rms_ns: float) -> complex:
    tau_s = float(tau_rms_ns) * 1e-9
    x = 2.0 * np.pi * float(delta_f_hz) * tau_s
    return 1.0 / (1.0 + 1j * x)


def _toeplitz_corr(num_tones: int, tone_spacing_hz: float, tau_rms_ns: float) -> np.ndarray:
    c = np.array([_freq_correlation(i * tone_spacing_hz, tau_rms_ns) for i in range(num_tones)], dtype=np.complex64)
    T = np.empty((num_tones, num_tones), dtype=np.complex64)
    for i in range(num_tones):
        for j in range(num_tones):
            T[i, j] = c[abs(i - j)]
    T += 1e-6 * np.eye(num_tones, dtype=np.complex64)
    return T


def make_frequency_selective_channels(
    H_base: tf.Tensor,
    *,
    cfg: ResolvedConfig,
    tau_rms_ns: float,
    num_tones: Optional[int] = None,
    seed: int = 1,
) -> tf.Tensor:
    """Create correlated per-tone channels from a flat base channel.

    The same spatial channel tensor is modulated by a per-link correlated complex
    tone process. This keeps the diagnostic cheap while exposing the flat-vs-
    selective sensitivity requested by the reviewers.
    """
    num_tones = int(num_tones or cfg.derived["num_re_sim"])
    batch, B, U, Nr, M = [int(x) for x in H_base.shape]
    tone_spacing_hz = tone_group_bandwidth_hz(cfg, num_tones=num_tones)

    C = _toeplitz_corr(num_tones=num_tones, tone_spacing_hz=tone_spacing_hz, tau_rms_ns=tau_rms_ns)
    L = np.linalg.cholesky(C)

    rng = np.random.default_rng(seed)
    z = (rng.standard_normal((batch, B, U, num_tones)) + 1j * rng.standard_normal((batch, B, U, num_tones))) / np.sqrt(2.0)
    g = np.einsum("...t,tu->...u", z, L.T, optimize=True)
    g = g / np.sqrt(np.mean(np.abs(g) ** 2, axis=-1, keepdims=True) + 1e-12)

    g_tf = tf.convert_to_tensor(np.transpose(g, [0, 3, 1, 2])[:, :, :, :, None, None], dtype=H_base.dtype)
    H_t = H_base[:, None, ...] * g_tf
    return H_t


def summarize_flat_vs_selective(
    *,
    cfg: ResolvedConfig,
    tau_rms_ns: float,
    num_tones: Optional[int] = None,
) -> Dict[str, float]:
    num_tones = int(num_tones or cfg.derived["num_re_sim"])
    cb = coherence_bandwidth_rules(tau_rms_ns)
    group_bw = tone_group_bandwidth_hz(cfg, num_tones=num_tones)
    return {
        **cb,
        "group_bandwidth_hz": float(group_bw),
        "group_over_bc50": float(group_bw / max(cb["bc50_hz"], 1e-12)),
        "group_over_bc90": float(group_bw / max(cb["bc90_hz"], 1e-12)),
        "flat_ok_bc50": float(group_bw <= cb["bc50_hz"]),
        "flat_ok_bc90": float(group_bw <= cb["bc90_hz"]),
    }
