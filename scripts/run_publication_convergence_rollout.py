#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping

import pandas as pd
import tensorflow as tf

import _repo_bootstrap as _rb

_rb.bootstrap()

from fr3_twc.config import get_twc_paths, load_twc_config
from fr3_twc.pipeline import (
    AlgorithmSpec,
    default_baseline_algorithms,
    default_unfolded_algorithms,
    run_suite,
)
from fr3_twc.reporting import load_models, save_grouped_mean
from fr3_twc.solver import LayerHyperParams, weighted_wmmse_solve
from fr3_twc.unfolding import UnfoldedWeightedWMMSE


class ContinuedRolloutUnfoldedModel:
    """Continue a trained unfolded model beyond its trained depth.

    Layers 1..L use the saved learned parameters. Layers L+1..T reuse the
    last learned layer parameters. This is a continuation rollout for figure
    generation, not a newly trained T-layer unfolded network.
    """

    def __init__(self, base_model: UnfoldedWeightedWMMSE, total_layers: int) -> None:
        self.base_model = base_model
        self.total_layers = int(total_layers)
        if self.total_layers <= 0:
            raise ValueError("total_layers must be positive")
        if self.total_layers < int(base_model.num_layers):
            raise ValueError(
                f"total_layers={self.total_layers} is smaller than trained depth={base_model.num_layers}"
            )

    @property
    def num_layers(self) -> int:
        return self.total_layers

    def _extend_vector(self, x: tf.Tensor) -> tf.Tensor:
        x = tf.cast(x, tf.float32)
        n = int(x.shape[0])
        if self.total_layers == n:
            return x
        last = x[n - 1 : n]
        tail = tf.repeat(last, repeats=self.total_layers - n, axis=0)
        return tf.concat([x, tail], axis=0)

    def _layer_hyperparams(self) -> LayerHyperParams:
        return LayerHyperParams(
            damping=list(tf.unstack(self._extend_vector(self.base_model.damping))),
            dual_step_mu=list(tf.unstack(self._extend_vector(self.base_model.dual_step_mu))),
            dual_step_lambda=list(tf.unstack(self._extend_vector(self.base_model.dual_step_lambda))),
        )

    def __call__(
        self,
        *,
        cfg,
        H: tf.Tensor,
        noise_var_watt: float,
        fs,
        user_weights: tf.Tensor,
        bs_total_tx_power_watt: float | None = None,
        fs_mode: str = "budget_dual",
        init_w: tf.Tensor | None = None,
    ):
        return weighted_wmmse_solve(
            cfg=cfg,
            H=H,
            noise_var_watt=noise_var_watt,
            fs=fs,
            user_weights=user_weights,
            bs_total_tx_power_watt=bs_total_tx_power_watt,
            num_iterations=self.total_layers,
            init_w=init_w,
            fs_mode=fs_mode,
            layer_params=self._layer_hyperparams(),
        )


def _pick_algorithms() -> list[AlgorithmSpec]:
    baselines = {a.name: a for a in default_baseline_algorithms()}
    unfolded = {a.name: a for a in default_unfolded_algorithms()}
    wanted = ["du_pf_cognitive", "pf_fs_cognitive", "edge_fs_soft"]
    out: list[AlgorithmSpec] = []
    for name in wanted:
        if name in unfolded:
            out.append(unfolded[name])
        elif name in baselines:
            out.append(baselines[name])
        else:
            raise KeyError(f"Missing algorithm spec for {name}")
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run a lightweight convergence-only suite that continues the trained unfolded cognitive model to a longer rollout length."
    )
    p.add_argument("--config", type=str, default="configs/twc_eval_final.yaml")
    p.add_argument("--suite-name", type=str, default="publication_convergence_rollout")
    p.add_argument("--snr-db", type=float, default=15.0)
    p.add_argument("--rollout-layers", type=int, default=56)
    p.add_argument("--set", dest="overrides", action="append", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_twc_config(args.config, overrides=args.overrides)

    # Convergence figure should use exactly one SNR point.
    cfg.raw.setdefault("sweep", {})["values"] = [float(args.snr_db)]

    twc_paths = get_twc_paths(cfg)
    models: Mapping[str, UnfoldedWeightedWMMSE] = load_models(
        twc_paths.checkpoint_root,
        names=["soft", "cognitive"],
        output_root=twc_paths.output_root,
        repo_root=twc_paths.project_root,
    )
    if "cognitive" not in models:
        raise FileNotFoundError(f"Missing cognitive checkpoint in {twc_paths.checkpoint_root}")

    model_registry = dict(models)
    trained_depth = int(model_registry["cognitive"].num_layers)
    model_registry["cognitive"] = ContinuedRolloutUnfoldedModel(
        base_model=model_registry["cognitive"],
        total_layers=int(args.rollout_layers),
    )

    art = run_suite(
        cfg=cfg,
        suite_name=args.suite_name,
        algorithms=_pick_algorithms(),
        model_registry=model_registry,
    )

    summary_mean = save_grouped_mean(
        art.summary_df,
        art.paths.root / "summary_mean.csv",
        group_cols=["algorithm", "sweep_value"],
    )
    history_mean = save_grouped_mean(
        art.history_df,
        art.paths.root / "history_mean.csv",
        group_cols=["algorithm", "sweep_value", "iteration"],
    )

    meta = pd.DataFrame(
        [
            {
                "trained_unfolded_layers": trained_depth,
                "continued_rollout_layers": int(args.rollout_layers),
                "snr_db": float(args.snr_db),
                "note": "Layers beyond the trained depth reuse the last learned layer parameters. This is a continuation rollout, not a separately trained deeper model.",
            }
        ]
    )
    meta.to_csv(art.paths.root / "rollout_meta.csv", index=False)

    print(f"ROLLOUT_OK root={art.paths.root}")
    print(f"ROLLOUT_INFO trained_layers={trained_depth} continued_layers={int(args.rollout_layers)}")
    print(f"ROLLOUT_ROWS summary_mean={len(summary_mean)} history_mean={len(history_mean)}")


if __name__ == "__main__":
    main()
