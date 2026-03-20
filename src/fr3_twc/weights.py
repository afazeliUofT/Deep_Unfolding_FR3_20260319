from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import tensorflow as tf


def normalize_mean_one(w: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64)
    return w / max(float(np.mean(w)), eps)


def uniform_weights(num_users: int, value: float = 1.0) -> np.ndarray:
    return np.full((int(num_users),), float(value), dtype=np.float64)


def edge_boost_weights_from_channel(H: tf.Tensor, gamma: float = 0.5) -> np.ndarray:
    """Create a static weight profile that boosts weaker users.

    H shape: [batch,B,U,Nr,M] or [batch,T,B,U,Nr,M]
    """
    if len(H.shape) == 6:
        H0 = H[:, 0]
    else:
        H0 = H
    power = tf.reduce_mean(tf.reduce_sum(tf.abs(H0) ** 2, axis=[1, 3, 4]), axis=0)  # [U]
    w = 1.0 / np.maximum(power.numpy(), 1e-12) ** float(gamma)
    return normalize_mean_one(w)


def lognormal_weights(num_users: int, sigma_db: float = 4.0, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sigma = float(sigma_db) / 10.0 * np.log(10.0)
    w = np.exp(rng.normal(loc=0.0, scale=sigma, size=int(num_users)))
    return normalize_mean_one(w)


def proportional_fair_weights(
    avg_rates: np.ndarray,
    eps: float = 1e-3,
    clip_min: float = 0.1,
    clip_max: float = 10.0,
) -> np.ndarray:
    w = 1.0 / np.maximum(np.asarray(avg_rates, dtype=np.float64), eps)
    w = np.clip(w, clip_min, clip_max)
    return normalize_mean_one(w)


def update_ema_rates(avg_rates: np.ndarray, inst_rates: np.ndarray, beta: float) -> np.ndarray:
    avg_rates = np.asarray(avg_rates, dtype=np.float64)
    inst_rates = np.asarray(inst_rates, dtype=np.float64)
    return float(beta) * avg_rates + (1.0 - float(beta)) * inst_rates


def catalog_weight_profiles(num_users: int, H: Optional[tf.Tensor] = None, seed: int = 1) -> Dict[str, np.ndarray]:
    out = {
        "uniform": uniform_weights(num_users),
        "lognormal": lognormal_weights(num_users=num_users, seed=seed),
    }
    if H is not None:
        out["edge_boost"] = edge_boost_weights_from_channel(H)
    else:
        out["edge_boost"] = uniform_weights(num_users)
    return out
