from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
import tensorflow as tf

from fr3_sim.channel import FsStats
from fr3_sim.config import ResolvedConfig
from fr3_sim.processing import MmseOutput, mmse_combiners_and_mse


@dataclass(frozen=True)
class SolveOutput:
    w: tf.Tensor
    mmse: MmseOutput
    mu: tf.Tensor
    lam: tf.Tensor
    history: Dict[str, tf.Tensor]
    num_iter: int
    converged: bool


@dataclass(frozen=True)
class LayerHyperParams:
    damping: Sequence[float]
    dual_step_mu: Sequence[float]
    dual_step_lambda: Sequence[float]


def _complex_normal(shape, dtype: tf.dtypes.DType) -> tf.Tensor:
    try:
        from sionna.phy.utils import complex_normal  # type: ignore

        return complex_normal(shape, dtype=dtype)
    except Exception:
        real_dtype = tf.float32 if dtype == tf.complex64 else tf.float64
        re = tf.random.normal(shape, dtype=real_dtype)
        im = tf.random.normal(shape, dtype=real_dtype)
        return tf.complex(re, im) / tf.cast(tf.sqrt(2.0), dtype)


def _extract_self_channels(H: tf.Tensor, u_per_bs: int) -> tf.Tensor:
    S = tf.shape(H)[0]
    B = tf.shape(H)[1]
    Nr = tf.shape(H)[3]
    M = tf.shape(H)[4]
    H_rs = tf.reshape(H, [S, B, B, u_per_bs, Nr, M])
    H_perm = tf.transpose(H_rs, [0, 3, 4, 5, 1, 2])
    H_diag = tf.linalg.diag_part(H_perm)
    return tf.transpose(H_diag, [0, 4, 1, 2, 3])


def _reshape_with_tones(H: tf.Tensor, T: int) -> tf.Tensor:
    if len(H.shape) == 5:
        return tf.tile(H[:, tf.newaxis, ...], [1, T, 1, 1, 1, 1])
    if len(H.shape) == 6:
        return H
    raise ValueError("H must have shape [batch,B,U,Nr,M] or [batch,T,B,U,Nr,M]")


def _prepare_user_weights(
    user_weights: tf.Tensor,
    batch: int,
    T: int,
    U: int,
    real_dtype: tf.DType,
) -> tf.Tensor:
    user_weights = tf.cast(user_weights, real_dtype)
    if len(user_weights.shape) == 1:
        user_weights = tf.broadcast_to(user_weights[tf.newaxis, :], [batch, U])
    elif len(user_weights.shape) != 2:
        raise ValueError("user_weights must have shape [U] or [batch,U]")
    return tf.reshape(tf.tile(user_weights[:, tf.newaxis, :], [1, T, 1]), [batch * T, U])


def _history_weighted_sum_rate(mmse: MmseOutput, xi: tf.Tensor, batch: int, T: int, real_dtype: tf.DType) -> tf.Tensor:
    sinr = tf.reshape(tf.cast(mmse.sinr, real_dtype), [batch, T, -1])
    xi_bt = tf.reshape(xi, [batch, T, -1])
    rate = tf.math.log(1.0 + sinr) / tf.math.log(tf.constant(2.0, dtype=real_dtype))
    return tf.reduce_mean(tf.reduce_sum(xi_bt * rate, axis=-1))


def weighted_wmmse_solve(
    *,
    cfg: ResolvedConfig,
    H: tf.Tensor,
    noise_var_watt: float,
    fs: Optional[FsStats],
    user_weights: tf.Tensor,
    bs_total_tx_power_watt: Optional[float] = None,
    num_iterations: Optional[int] = None,
    init_w: Optional[tf.Tensor] = None,
    fs_mode: str = "budget_dual",
    layer_params: Optional[LayerHyperParams] = None,
) -> SolveOutput:
    """Weighted WMMSE with per-user weights and optional FS protection.

    Supported FS modes
    ------------------
    - none          : ignores FS constraints
    - budget_dual   : soft FS dual updates
    - hard_null     : steering-vector null projection every layer
    - hybrid        : soft dual + null projection when FS violation is high
    """
    precision = cfg.derived.get("tf_precision", "single")
    real_dtype = tf.float32 if str(precision).lower().startswith("single") else tf.float64
    complex_dtype = tf.complex64 if real_dtype == tf.float32 else tf.complex128

    B = int(cfg.derived["num_bs"])
    U = int(cfg.derived["num_ut"])
    u_per_bs = int(cfg.derived["u_per_bs"])
    M = int(cfg.raw["channel_model"]["num_bs_ant"])
    Nr = int(cfg.raw["channel_model"]["num_ut_ant"])
    T = int(cfg.derived["num_re_sim"])
    batch = int(H.shape[0])

    H_t = _reshape_with_tones(H, T=T)
    H_eff = tf.reshape(H_t, [-1, B, U, Nr, M])
    S_eff = int(H_eff.shape[0])

    w_cfg = dict(cfg.raw.get("receiver", {}).get("wmmse", {}) or {})
    K_iter = int(num_iterations or w_cfg.get("num_iterations", 20))
    tol = float(w_cfg.get("convergence_tol", 1e-6))
    ridge = float(w_cfg.get("ridge_regularization", 1e-9))
    rho_mu_base = float(w_cfg.get("dual_step_mu", 0.02))
    rho_lam_base = float(w_cfg.get("dual_step_lambda", 0.2))
    damping_base = float(w_cfg.get("damping_w", 1.0))
    fs_mode = str(fs_mode).lower().strip()

    re_scaling = float(cfg.derived["re_scaling"])
    p_tot_watt = float(bs_total_tx_power_watt) if bs_total_tx_power_watt is not None else float(cfg.derived["bs_total_tx_power_watt"])

    xi = _prepare_user_weights(user_weights, batch=batch, T=T, U=U, real_dtype=real_dtype)

    if init_w is not None:
        w = tf.cast(init_w, complex_dtype)
    else:
        w0 = _complex_normal([S_eff, B, M, u_per_bs], dtype=complex_dtype)
        pow_b = tf.reduce_sum(tf.abs(w0) ** 2, axis=[2, 3])
        target = tf.cast(p_tot_watt / (re_scaling * T), real_dtype)
        scale = tf.sqrt(target / (tf.cast(pow_b, real_dtype) + 1e-12))
        w = w0 * tf.cast(scale[:, :, tf.newaxis, tf.newaxis], complex_dtype)

    mu_init = float(w_cfg.get("mu_init", 1.0))
    lam_init = float(w_cfg.get("lambda_init", 0.0))
    mu = tf.fill([batch, B], tf.cast(mu_init, real_dtype))
    if fs is not None:
        L = int(fs.bar_beta.shape[-1])
        lam = tf.fill([batch, L], tf.cast(lam_init, real_dtype))
        if fs_mode == "none":
            lam = tf.zeros_like(lam)
    else:
        L = 0
        lam = tf.zeros([batch, 0], dtype=real_dtype)

    obj_hist = []
    fs_hist = []
    pow_hist = []
    delta_hist = []

    mmse = mmse_combiners_and_mse(H_eff, w, noise_var_watt=tf.cast(noise_var_watt, real_dtype))

    def _step_param(seq: Sequence[float] | None, k: int, default: float) -> tf.Tensor:
        if seq is None or len(seq) == 0:
            return tf.cast(default, real_dtype)
        idx = min(k, len(seq) - 1)
        val = seq[idx]
        try:
            return tf.cast(val, real_dtype)
        except Exception:
            return tf.cast(float(val), real_dtype)

    hard_null_thresh = float(cfg.raw.get("twc", {}).get("hybrid", {}).get("null_when_ratio_exceeds", 0.98))

    for k in range(K_iter):
        w_prev = w
        q = xi / tf.cast(mmse.mse, real_dtype)

        q_bu = tf.reshape(q, [S_eff, B, u_per_bs])
        v_bu = tf.reshape(tf.cast(mmse.v, complex_dtype), [S_eff, B, u_per_bs, Nr])
        H_self = _extract_self_channels(H_eff, u_per_bs=u_per_bs)
        a_self = tf.einsum("sburm,sbur->sbum", tf.math.conj(H_self), v_bu, optimize=True)

        v_all = tf.cast(mmse.v, complex_dtype)
        a_all = tf.einsum("sburm,sur->sbum", tf.math.conj(H_eff), v_all, optimize=True)
        sqrt_q_all = tf.sqrt(tf.maximum(tf.cast(q, real_dtype), 0.0))
        x_all = tf.cast(sqrt_q_all[:, tf.newaxis, :, tf.newaxis], complex_dtype) * a_all
        D_signal = tf.einsum("sbum,sbun->sbmn", x_all, tf.math.conj(x_all), optimize=True)
        C = tf.einsum("sbu,sbum->sbmu", tf.cast(q_bu, complex_dtype), a_self, optimize=True)

        R_fs = None
        f_eff = None
        if fs is not None and fs_mode in ("budget_dual", "hybrid"):
            bar_beta = tf.cast(fs.bar_beta, real_dtype)
            epsilon = tf.cast(fs.epsilon, real_dtype)
            delta = tf.cast(fs.delta, real_dtype)
            g = tf.cast(re_scaling, real_dtype) * tf.einsum("sl,bt,tl,sbl->sbtl", lam, delta, epsilon, bar_beta, optimize=True)
            g_eff = tf.reshape(tf.transpose(g, [0, 2, 1, 3]), [S_eff, B, L])
            corr = getattr(fs, "correlation", "identity")
            if corr == "steering_rank1" and getattr(fs, "a_bs_fs", None) is not None:
                a = tf.cast(fs.a_bs_fs, complex_dtype)
                a_eff = tf.reshape(tf.tile(a[:, tf.newaxis, ...], [1, T, 1, 1, 1]), [S_eff, B, L, M])
                sqrt_g = tf.sqrt(tf.maximum(g_eff, 0.0))
                u = tf.cast(sqrt_g[..., tf.newaxis], complex_dtype) * a_eff
                R_fs = tf.einsum("sblm,sbln->sbmn", u, tf.math.conj(u), optimize=True)
            else:
                f_eff = tf.reduce_sum(g_eff, axis=-1)

        mu_eff = tf.reshape(tf.tile(mu[:, tf.newaxis, :], [1, T, 1]), [S_eff, B])
        diag_add = tf.cast(re_scaling, real_dtype) * mu_eff + tf.cast(ridge, real_dtype)
        I = tf.eye(M, batch_shape=tf.shape(D_signal)[:-2], dtype=complex_dtype)
        D = D_signal + tf.cast(diag_add[:, :, tf.newaxis, tf.newaxis], complex_dtype) * I
        if R_fs is not None:
            D = D + R_fs
        elif f_eff is not None:
            D = D + tf.cast(f_eff[:, :, tf.newaxis, tf.newaxis], complex_dtype) * I

        w_new = tf.linalg.solve(D, C)

        damping = _step_param(layer_params.damping if layer_params else None, k, damping_base)
        w = tf.cast(damping, complex_dtype) * w_new + tf.cast(1.0 - damping, complex_dtype) * w_prev

        # Optional null projection
        if fs is not None and getattr(fs, "a_bs_fs", None) is not None and fs_mode in ("hard_null", "hybrid"):
            do_null = fs_mode == "hard_null"
            if fs_mode == "hybrid":
                # evaluate current FS ratio quickly
                w_tmp = tf.reshape(w, [batch, T, B, M, u_per_bs])
                a = tf.cast(fs.a_bs_fs, complex_dtype)
                proj = tf.einsum("sblm,stbmu->stblu", tf.math.conj(a), w_tmp, optimize=True)
                p_dir = tf.reduce_sum(tf.abs(proj) ** 2, axis=-1)
                delta_tb = tf.transpose(tf.cast(fs.delta, real_dtype), [1, 0])
                p_dir = tf.cast(p_dir, real_dtype) * delta_tb[None, :, :, None]
                I_fs_tmp = tf.cast(re_scaling, real_dtype) * tf.einsum(
                    "stbl,tl,sbl->sl",
                    p_dir,
                    tf.cast(fs.epsilon, real_dtype),
                    tf.cast(fs.bar_beta, real_dtype),
                    optimize=True,
                )
                ratio = tf.reduce_max(I_fs_tmp / tf.maximum(fs.i_max_watt[None, :], 1e-30))
                do_null = bool((ratio > hard_null_thresh).numpy())

            if do_null:
                a = tf.cast(fs.a_bs_fs, complex_dtype)
                a_eff = tf.reshape(tf.tile(a[:, tf.newaxis, ...], [1, T, 1, 1, 1]), [S_eff, B, L, M])
                G = tf.einsum("sblm,sbkm->sblk", tf.math.conj(a_eff), a_eff, optimize=True)
                reg = tf.cast(float(w_cfg.get("aggressive_fs_nulling_reg", 1e-6)), complex_dtype)
                I_L = tf.eye(L, batch_shape=[S_eff, B], dtype=complex_dtype)
                G_reg = tf.cast(G, complex_dtype) + reg * I_L
                Aw = tf.einsum("sblm,sbmu->sblu", tf.math.conj(a_eff), w, optimize=True)
                x = tf.linalg.lstsq(G_reg, Aw, fast=False)
                correction = tf.einsum("sblm,sblu->sbmu", a_eff, x, optimize=True)
                w = w - correction

        w_full = tf.reshape(w, [batch, T, B, M, u_per_bs])
        pow_bt = tf.reduce_sum(tf.abs(w_full) ** 2, axis=[3, 4])
        pow_b_raw = tf.cast(re_scaling, real_dtype) * tf.reduce_sum(tf.cast(pow_bt, real_dtype), axis=1)

        rho_mu = _step_param(layer_params.dual_step_mu if layer_params else None, k, rho_mu_base)
        mu = tf.maximum(0.0, mu + rho_mu * (pow_b_raw - p_tot_watt))

        scale = tf.sqrt(p_tot_watt / tf.maximum(pow_b_raw, tf.cast(1e-12, real_dtype)))
        scale = tf.minimum(scale, 1.0)
        scale = tf.where(pow_b_raw > p_tot_watt, scale * tf.cast(1.0 - 1e-6, real_dtype), scale)
        w_full = w_full * tf.cast(scale[:, tf.newaxis, :, tf.newaxis, tf.newaxis], complex_dtype)
        w = tf.reshape(w_full, [S_eff, B, M, u_per_bs])
        pow_b = tf.cast(re_scaling, real_dtype) * tf.reduce_sum(tf.cast(tf.reduce_sum(tf.abs(w_full) ** 2, axis=[3, 4]), real_dtype), axis=1)

        if fs is not None and fs_mode in ("budget_dual", "hybrid"):
            corr = getattr(fs, "correlation", "identity")
            if corr == "steering_rank1" and getattr(fs, "a_bs_fs", None) is not None:
                a = tf.cast(fs.a_bs_fs, complex_dtype)
                proj = tf.einsum("sblm,stbmu->stblu", tf.math.conj(a), w_full, optimize=True)
                p_dir = tf.reduce_sum(tf.abs(proj) ** 2, axis=-1)
                delta_tb = tf.transpose(tf.cast(fs.delta, real_dtype), [1, 0])
                p_dir = tf.cast(p_dir, real_dtype) * delta_tb[None, :, :, None]
                I_fs = tf.cast(re_scaling, real_dtype) * tf.einsum(
                    "stbl,tl,sbl->sl",
                    p_dir,
                    tf.cast(fs.epsilon, real_dtype),
                    tf.cast(fs.bar_beta, real_dtype),
                    optimize=True,
                )
            else:
                delta_tb = tf.transpose(tf.cast(fs.delta, real_dtype), [1, 0])
                pow_bt_fs = tf.cast(tf.reduce_sum(tf.abs(w_full) ** 2, axis=[3, 4]), real_dtype) * delta_tb[None, :, :]
                I_fs = tf.cast(re_scaling, real_dtype) * tf.einsum(
                    "stb,tl,sbl->sl",
                    pow_bt_fs,
                    tf.cast(fs.epsilon, real_dtype),
                    tf.cast(fs.bar_beta, real_dtype),
                    optimize=True,
                )
            i_max = tf.cast(fs.i_max_watt[None, :], real_dtype)
            rho_lam = _step_param(layer_params.dual_step_lambda if layer_params else None, k, rho_lam_base)
            viol = (I_fs - i_max) / tf.maximum(i_max, 1e-30)
            lam = tf.maximum(0.0, lam + rho_lam * viol)
        else:
            I_fs = tf.zeros([batch, 0], dtype=real_dtype)
            i_max = tf.zeros([batch, 0], dtype=real_dtype)

        num = tf.abs(tf.linalg.norm(w - w_prev))
        den = tf.maximum(tf.abs(tf.linalg.norm(w_prev)), tf.cast(1e-12, real_dtype))
        w_delta = tf.cast(num / den, real_dtype)

        mmse = mmse_combiners_and_mse(H_eff, w, noise_var_watt=tf.cast(noise_var_watt, real_dtype))
        obj_hist.append(_history_weighted_sum_rate(mmse, xi=xi, batch=batch, T=T, real_dtype=real_dtype))
        pow_hist.append(tf.reduce_max(tf.maximum(pow_b - p_tot_watt, 0.0)))
        fs_hist.append(tf.reduce_max(tf.maximum(I_fs - i_max, 0.0)) if fs is not None else tf.cast(0.0, real_dtype))
        delta_hist.append(w_delta)

    history = {
        "weighted_sum_rate": tf.stack(obj_hist, axis=0),
        "max_power_violation_watt": tf.stack(pow_hist, axis=0),
        "max_fs_violation_watt": tf.stack(fs_hist, axis=0),
        "w_delta": tf.stack(delta_hist, axis=0),
    }
    converged = bool(tf.cast(history["w_delta"][-1] < tol, tf.bool).numpy())
    return SolveOutput(
        w=tf.reshape(w, [batch, T, B, M, u_per_bs]),
        mmse=mmse,
        mu=mu,
        lam=lam,
        history=history,
        num_iter=K_iter,
        converged=converged,
    )


def make_layer_hyperparams(
    num_layers: int,
    damping: float,
    dual_step_mu: float,
    dual_step_lambda: float,
) -> LayerHyperParams:
    return LayerHyperParams(
        damping=[float(damping)] * int(num_layers),
        dual_step_mu=[float(dual_step_mu)] * int(num_layers),
        dual_step_lambda=[float(dual_step_lambda)] * int(num_layers),
    )
