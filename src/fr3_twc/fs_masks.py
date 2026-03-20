from __future__ import annotations

from dataclasses import replace
from typing import Optional

import numpy as np
import tensorflow as tf

from fr3_sim.channel import FsStats


def apply_delta_mask(fs: FsStats, delta: tf.Tensor) -> FsStats:
    """Return a new ``FsStats`` with a replaced BS-tone activity mask.

    Parameters
    ----------
    delta : [B, T] in {0,1}
    """
    return replace(fs, delta=tf.cast(delta, fs.delta.dtype))


def compute_cognitive_mask(
    fs: FsStats,
    active_fraction: float = 0.75,
    min_active_tones: int = 1,
    temperature: float = 1.0,
    protect_top_l: Optional[int] = None,
) -> tf.Tensor:
    """Topology-aware per-BS tone mask for cognitive coexistence.

    Idea
    ----
    For each BS and tone-group, build a simple *risk score* using the current
    FS overlap weights and large-scale BS->FS couplings. Then keep only the
    lowest-risk fraction of tones active for that BS.

    This directly addresses the reviewer suggestion that different BSs may use
    different sub-bands depending on the geometry w.r.t. the FS receivers.
    """
    bar_beta = tf.cast(fs.bar_beta, tf.float32)  # [S,B,L]
    epsilon = tf.cast(fs.epsilon, tf.float32)    # [T,L]

    beta_mean = tf.reduce_mean(bar_beta, axis=0)  # [B,L]
    if protect_top_l is not None and protect_top_l > 0 and protect_top_l < int(beta_mean.shape[-1]):
        top = tf.math.top_k(beta_mean, k=protect_top_l, sorted=False)
        beta_mask = tf.reduce_sum(tf.one_hot(top.indices, depth=int(beta_mean.shape[-1]), dtype=beta_mean.dtype), axis=-2)
        beta_mean = beta_mean * beta_mask

    risk = tf.einsum("bl,tl->bt", beta_mean, epsilon, optimize=True)  # [B,T]
    if temperature != 1.0:
        risk = tf.pow(tf.maximum(risk, 1e-12), tf.constant(float(temperature), dtype=risk.dtype))

    B = int(risk.shape[0])
    T = int(risk.shape[1])
    keep = max(int(round(float(active_fraction) * T)), int(min_active_tones))
    keep = min(max(keep, 1), T)

    risk_np = risk.numpy()
    mask_np = np.zeros((B, T), dtype=np.float32)
    for b in range(B):
        idx = np.argsort(risk_np[b])[:keep]
        mask_np[b, idx] = 1.0
    return tf.convert_to_tensor(mask_np, dtype=fs.delta.dtype)


def summarize_mask(mask: tf.Tensor) -> dict:
    m = tf.cast(mask, tf.float32)
    active_per_bs = tf.reduce_sum(m, axis=1)
    return {
        "mean_active_tones_per_bs": float(tf.reduce_mean(active_per_bs).numpy()),
        "min_active_tones_per_bs": float(tf.reduce_min(active_per_bs).numpy()),
        "max_active_tones_per_bs": float(tf.reduce_max(active_per_bs).numpy()),
        "active_fraction": float(tf.reduce_mean(m).numpy()),
    }
