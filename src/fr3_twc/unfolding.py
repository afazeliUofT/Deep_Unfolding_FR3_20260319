from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import tensorflow as tf

from fr3_sim.channel import FsStats
from fr3_sim.config import ResolvedConfig

from .metrics_ext import compute_fs_interference, per_user_rates_from_mmse
from .solver import LayerHyperParams, SolveOutput, weighted_wmmse_solve


@dataclass(frozen=True)
class TrainConfig:
    num_layers: int
    learning_rate: float
    lambda_power: float
    lambda_fs: float
    lambda_smooth: float


class UnfoldedWeightedWMMSE(tf.Module):
    """Trainable unrolled weighted-WMMSE.

    Trainable parameters are intentionally low dimensional:
    - one damping coefficient per layer
    - one BS-power dual step size per layer
    - one FS dual step size per layer

    This preserves the original algorithmic structure and makes the model
    interpretable, stable, and easy to compare to the classical solver.
    """

    def __init__(
        self,
        num_layers: int,
        init_damping: float = 0.85,
        init_dual_step_mu: float = 0.02,
        init_dual_step_lambda: float = 0.20,
        name: str = "unfolded_weighted_wmmse",
    ) -> None:
        super().__init__(name=name)
        self.num_layers = int(num_layers)
        eps = 1e-6

        init_damping = float(np.clip(init_damping, 0.05, 0.98))
        damp_logit = np.log(init_damping / max(1.0 - init_damping, 1e-8))
        mu_raw = np.log(np.exp(max(init_dual_step_mu, eps)) - 1.0)
        lam_raw = np.log(np.exp(max(init_dual_step_lambda, eps)) - 1.0)

        self.raw_damping = tf.Variable(np.full([self.num_layers], damp_logit, dtype=np.float32), trainable=True, name="raw_damping")
        self.raw_dual_step_mu = tf.Variable(np.full([self.num_layers], mu_raw, dtype=np.float32), trainable=True, name="raw_dual_step_mu")
        self.raw_dual_step_lambda = tf.Variable(np.full([self.num_layers], lam_raw, dtype=np.float32), trainable=True, name="raw_dual_step_lambda")

    @property
    def damping(self) -> tf.Tensor:
        return 0.02 + 0.96 * tf.math.sigmoid(self.raw_damping)

    @property
    def dual_step_mu(self) -> tf.Tensor:
        return 1e-5 + tf.nn.softplus(self.raw_dual_step_mu)

    @property
    def dual_step_lambda(self) -> tf.Tensor:
        return 1e-5 + tf.nn.softplus(self.raw_dual_step_lambda)

    def layer_hyperparams(self) -> LayerHyperParams:
        return LayerHyperParams(
            damping=list(tf.unstack(self.damping)),
            dual_step_mu=list(tf.unstack(self.dual_step_mu)),
            dual_step_lambda=list(tf.unstack(self.dual_step_lambda)),
        )

    def __call__(
        self,
        *,
        cfg: ResolvedConfig,
        H: tf.Tensor,
        noise_var_watt: float,
        fs: Optional[FsStats],
        user_weights: tf.Tensor,
        bs_total_tx_power_watt: Optional[float] = None,
        fs_mode: str = "budget_dual",
        init_w: Optional[tf.Tensor] = None,
    ) -> SolveOutput:
        return weighted_wmmse_solve(
            cfg=cfg,
            H=H,
            noise_var_watt=noise_var_watt,
            fs=fs,
            user_weights=user_weights,
            bs_total_tx_power_watt=bs_total_tx_power_watt,
            num_iterations=self.num_layers,
            init_w=init_w,
            fs_mode=fs_mode,
            layer_params=self.layer_hyperparams(),
        )

    def export_state(self) -> Dict[str, Any]:
        return {
            "num_layers": self.num_layers,
            "damping": self.damping.numpy().tolist(),
            "dual_step_mu": self.dual_step_mu.numpy().tolist(),
            "dual_step_lambda": self.dual_step_lambda.numpy().tolist(),
        }

    def save_npz(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            p,
            num_layers=self.num_layers,
            raw_damping=self.raw_damping.numpy(),
            raw_dual_step_mu=self.raw_dual_step_mu.numpy(),
            raw_dual_step_lambda=self.raw_dual_step_lambda.numpy(),
        )
        return p

    @classmethod
    def load_npz(cls, path: str | Path) -> "UnfoldedWeightedWMMSE":
        data = np.load(Path(path), allow_pickle=True)
        model = cls(num_layers=int(data["num_layers"]))
        model.raw_damping.assign(data["raw_damping"])
        model.raw_dual_step_mu.assign(data["raw_dual_step_mu"])
        model.raw_dual_step_lambda.assign(data["raw_dual_step_lambda"])
        return model


def differentiable_loss(
    *,
    cfg: ResolvedConfig,
    result: SolveOutput,
    fs: Optional[FsStats],
    user_weights: tf.Tensor,
    noise_var_watt: float,
    lambda_power: float,
    lambda_fs: float,
    lambda_smooth: float,
    model: UnfoldedWeightedWMMSE,
) -> tf.Tensor:
    batch = int(result.w.shape[0])
    num_re_sim = int(cfg.derived["num_re_sim"])
    re_scaling = float(cfg.derived["re_scaling"])
    p_tot_watt = float(cfg.derived["bs_total_tx_power_watt"])

    rates = per_user_rates_from_mmse(result.mmse, batch=batch, num_re_sim=num_re_sim)
    user_rate = tf.reduce_mean(rates, axis=[0, 1])

    user_weights = tf.cast(user_weights, tf.float32)
    if len(user_weights.shape) == 2:
        user_weights = tf.reduce_mean(user_weights, axis=0)
    weighted_sum_rate = tf.reduce_sum(user_weights * user_rate)

    pow_t_b = tf.reduce_sum(tf.abs(result.w) ** 2, axis=[3, 4])
    pow_b = tf.cast(re_scaling, tf.float32) * tf.reduce_sum(tf.cast(pow_t_b, tf.float32), axis=1)
    power_penalty = tf.reduce_mean(tf.square(tf.nn.relu(pow_b - p_tot_watt)))

    if fs is None or int(result.lam.shape[-1]) == 0:
        fs_penalty = tf.cast(0.0, tf.float32)
    else:
        I_fs = compute_fs_interference(result.w, fs=fs, re_scaling=re_scaling)
        i_max = tf.cast(fs.i_max_watt[None, :], tf.float32)
        fs_penalty = tf.reduce_mean(tf.square(tf.nn.relu(I_fs - i_max)))

    smooth_penalty = tf.reduce_mean(tf.square(model.damping[1:] - model.damping[:-1]))
    smooth_penalty += tf.reduce_mean(tf.square(model.dual_step_mu[1:] - model.dual_step_mu[:-1]))
    smooth_penalty += tf.reduce_mean(tf.square(model.dual_step_lambda[1:] - model.dual_step_lambda[:-1]))

    return -weighted_sum_rate + lambda_power * power_penalty + lambda_fs * fs_penalty + lambda_smooth * smooth_penalty
