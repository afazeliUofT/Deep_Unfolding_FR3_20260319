from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import tensorflow as tf

from fr3_sim.channel import FsStats
from fr3_sim.processing import MmseOutput


def _log2(x: tf.Tensor) -> tf.Tensor:
    return tf.math.log(x) / tf.math.log(tf.constant(2.0, dtype=x.dtype))


def _jain_index(x: tf.Tensor, eps: float = 1e-12) -> tf.Tensor:
    x = tf.cast(x, tf.float32)
    num = tf.square(tf.reduce_sum(x, axis=-1))
    den = tf.cast(tf.shape(x)[-1], x.dtype) * tf.reduce_sum(tf.square(x), axis=-1)
    return num / tf.maximum(den, tf.cast(eps, x.dtype))


def _watt_to_dbm(x: tf.Tensor, eps: float = 1e-30) -> tf.Tensor:
    x = tf.maximum(tf.cast(x, tf.float32), tf.cast(eps, tf.float32))
    return 10.0 * tf.math.log(x / 1e-3) / tf.math.log(10.0)


def per_user_rates_from_mmse(mmse: MmseOutput, batch: int, num_re_sim: int) -> tf.Tensor:
    """Return [batch, T, U] rate tensor in b/s/Hz."""
    sinr = tf.reshape(mmse.sinr, [batch, num_re_sim, -1])
    rate = _log2(1.0 + tf.cast(sinr, tf.float32))
    return rate


def compute_fs_interference(w: tf.Tensor, fs: Optional[FsStats], re_scaling: float) -> tf.Tensor:
    if fs is None:
        batch = int(w.shape[0])
        return tf.zeros([batch, 0], dtype=tf.float32)

    real_dtype = tf.float32
    corr = getattr(fs, "correlation", "identity")
    if corr == "steering_rank1" and getattr(fs, "a_bs_fs", None) is not None:
        a = tf.cast(fs.a_bs_fs, w.dtype)  # [batch,B,L,M]
        proj = tf.einsum(
            "sblm,stbmu->stblu",
            tf.math.conj(a),
            w,
            optimize=True,
        )
        p_dir = tf.reduce_sum(tf.abs(proj) ** 2, axis=-1)
        delta_tb = tf.transpose(tf.cast(fs.delta, real_dtype), [1, 0])
        p_dir = tf.cast(p_dir, real_dtype) * delta_tb[None, :, :, None]
        return tf.cast(re_scaling, real_dtype) * tf.einsum(
            "stbl,tl,sbl->sl",
            p_dir,
            tf.cast(fs.epsilon, real_dtype),
            tf.cast(fs.bar_beta, real_dtype),
            optimize=True,
        )

    pow_t_b = tf.reduce_sum(tf.abs(w) ** 2, axis=[3, 4])
    delta_tb = tf.transpose(tf.cast(fs.delta, real_dtype), [1, 0])
    pow_t_b = tf.cast(pow_t_b, real_dtype) * delta_tb[None, :, :]
    return tf.cast(re_scaling, real_dtype) * tf.einsum(
        "stb,tl,sbl->sl",
        pow_t_b,
        tf.cast(fs.epsilon, real_dtype),
        tf.cast(fs.bar_beta, real_dtype),
        optimize=True,
    )


def extended_metrics(
    *,
    w: tf.Tensor,
    mmse: MmseOutput,
    fs: Optional[FsStats],
    p_tot_watt: float,
    re_scaling: float,
    num_re_sim: int,
    runtime_sec: float,
    history: Optional[Dict[str, Any]] = None,
    long_term_avg_rates: Optional[np.ndarray] = None,
    current_weights: Optional[np.ndarray] = None,
    coverage_rate_threshold_bpshz: float = 1.0,
    coverage_sinr_threshold_db: float = 0.0,
) -> Dict[str, float]:
    batch = int(w.shape[0])

    rate = per_user_rates_from_mmse(mmse, batch=batch, num_re_sim=num_re_sim)  # [S,T,U]
    per_user = tf.reduce_mean(rate, axis=[0, 1])
    per_sample_user = tf.reduce_mean(rate, axis=1)  # [S,U]

    sinr = tf.reshape(mmse.sinr, [batch, num_re_sim, -1])
    sinr_db = 10.0 * tf.math.log(tf.maximum(tf.cast(sinr, tf.float32), 1e-12)) / tf.math.log(10.0)

    weighted_sum_rate = tf.reduce_sum(per_user)
    if current_weights is not None:
        wgt = tf.convert_to_tensor(current_weights, dtype=tf.float32)
        weighted_sum_rate = tf.reduce_sum(wgt * per_user)

    pow_t_b = tf.reduce_sum(tf.abs(w) ** 2, axis=[3, 4])
    pow_b = tf.cast(re_scaling, tf.float32) * tf.reduce_sum(tf.cast(pow_t_b, tf.float32), axis=1)

    fs_int = compute_fs_interference(w=w, fs=fs, re_scaling=re_scaling)

    out: Dict[str, float] = {
        "sum_rate_bps_per_hz": float(tf.reduce_sum(per_user).numpy()),
        "weighted_sum_rate_bps_per_hz": float(weighted_sum_rate.numpy()),
        "avg_user_rate_bps_per_hz": float(tf.reduce_mean(per_user).numpy()),
        "p05_user_rate_bps_per_hz": float(np.percentile(per_user.numpy(), 5.0)),
        "p50_user_rate_bps_per_hz": float(np.percentile(per_user.numpy(), 50.0)),
        "jain_fairness": float(_jain_index(per_sample_user).numpy().mean()),
        "avg_sinr_db": float(tf.reduce_mean(sinr_db).numpy()),
        "p05_sinr_db": float(np.percentile(sinr_db.numpy().reshape(-1), 5.0)),
        "coverage_rate": float(
            tf.reduce_mean(tf.cast(per_user >= coverage_rate_threshold_bpshz, tf.float32)).numpy()
        ),
        "coverage_sinr": float(
            tf.reduce_mean(tf.cast(sinr_db >= coverage_sinr_threshold_db, tf.float32)).numpy()
        ),
        "max_bs_power_watt": float(tf.reduce_max(pow_b).numpy()),
        "max_bs_power_violation_watt": float(
            tf.reduce_max(tf.maximum(pow_b - p_tot_watt, 0.0)).numpy()
        ),
        "runtime_sec": float(runtime_sec),
    }

    if long_term_avg_rates is not None:
        long_term_avg_rates = np.asarray(long_term_avg_rates, dtype=np.float64)
        out["pf_utility"] = float(np.sum(np.log(np.maximum(long_term_avg_rates, 1e-8))))
        out["pf_avg_rate_bps_per_hz"] = float(np.mean(long_term_avg_rates))
        out["pf_p05_rate_bps_per_hz"] = float(np.percentile(long_term_avg_rates, 5.0))
        out["pf_jain_fairness"] = float(
            (long_term_avg_rates.sum() ** 2)
            / (len(long_term_avg_rates) * np.sum(long_term_avg_rates ** 2) + 1e-12)
        )

    if fs is None or fs_int.shape[-1] == 0:
        out.update(
            {
                "max_fs_interference_watt": 0.0,
                "mean_fs_interference_watt": 0.0,
                "max_fs_interference_dbm": -300.0,
                "mean_fs_interference_dbm": -300.0,
                "max_fs_violation_watt": 0.0,
                "protection_satisfaction": 1.0,
                "protection_satisfaction_strict": 1.0,
                "min_fs_i_max_dbm": -300.0,
                "max_fs_i_max_dbm": -300.0,
                "min_protection_margin_db": 300.0,
                "mean_protection_margin_db": 300.0,
            }
        )
    else:
        i_max = tf.cast(fs.i_max_watt[None, :], tf.float32)
        viol = tf.maximum(fs_int - i_max, 0.0)
        # Small tolerance avoids meaningless failures from machine precision at the threshold.
        tol = tf.maximum(1e-6 * i_max, tf.constant(1e-18, dtype=tf.float32))
        strict_ok = tf.cast(fs_int <= i_max, tf.float32)
        ok = tf.cast(fs_int <= (i_max + tol), tf.float32)

        margin_db = 10.0 * tf.math.log(
            tf.maximum(i_max, 1e-30) / tf.maximum(fs_int, 1e-30)
        ) / tf.math.log(10.0)

        out.update(
            {
                "max_fs_interference_watt": float(tf.reduce_max(fs_int).numpy()),
                "mean_fs_interference_watt": float(tf.reduce_mean(fs_int).numpy()),
                "max_fs_interference_dbm": float(tf.reduce_max(_watt_to_dbm(fs_int)).numpy()),
                "mean_fs_interference_dbm": float(tf.reduce_mean(_watt_to_dbm(fs_int)).numpy()),
                "max_fs_violation_watt": float(tf.reduce_max(viol).numpy()),
                "protection_satisfaction": float(tf.reduce_mean(ok).numpy()),
                "protection_satisfaction_strict": float(tf.reduce_mean(strict_ok).numpy()),
                "min_fs_i_max_dbm": float(tf.reduce_min(_watt_to_dbm(i_max)).numpy()),
                "max_fs_i_max_dbm": float(tf.reduce_max(_watt_to_dbm(i_max)).numpy()),
                "min_protection_margin_db": float(tf.reduce_min(margin_db).numpy()),
                "mean_protection_margin_db": float(tf.reduce_mean(margin_db).numpy()),
            }
        )

    if history:
        try:
            out["final_w_delta"] = float(tf.cast(history["w_delta"][-1], tf.float32).numpy())
        except Exception:
            pass
        try:
            out["final_history_weighted_sum_rate"] = float(
                tf.cast(history["weighted_sum_rate"][-1], tf.float32).numpy()
            )
        except Exception:
            pass

    return out
