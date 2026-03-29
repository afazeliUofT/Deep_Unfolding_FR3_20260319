from __future__ import annotations

import _repo_bootstrap as _rb

_rb.bootstrap()

import argparse

import tensorflow as tf

import train_unfolding as train_job
from fr3_twc.config import load_twc_config
from fr3_twc.unfolding import UnfoldedWeightedWMMSE, differentiable_loss


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fast preflight for unfolded training jobs")
    p.add_argument("--config", type=str, default="configs/twc_train_final.yaml")
    p.add_argument("--variant", type=str, choices=["soft", "cognitive"], required=True)
    return p.parse_args()


def _assert_finite_tensor(x: tf.Tensor, name: str) -> None:
    x = tf.convert_to_tensor(x)
    if x.dtype.is_complex:
        tf.debugging.assert_all_finite(tf.math.real(x), f"{name}.real has NaN/Inf")
        tf.debugging.assert_all_finite(tf.math.imag(x), f"{name}.imag has NaN/Inf")
    else:
        tf.debugging.assert_all_finite(x, f"{name} has NaN/Inf")


def main() -> None:
    args = parse_args()
    cfg = load_twc_config(args.config)

    base_seed = int(cfg.raw["reproducibility"]["seed"])
    train_job.seed_all(base_seed)

    twc = cfg.raw.get("twc", {}) or {}
    unfold_cfg = twc.get("unfolding", {}) or {}
    tr = twc.get("training", {}) or {}

    batch_size_cfg = int(cfg.raw.get("experiment", {}).get("batch_size", 1))
    batch_size = max(1, min(batch_size_cfg, 2))
    num_users = int(cfg.derived["num_ut"])
    p_tot_watt = float(cfg.derived["bs_total_tx_power_watt"])
    snr_choices = [float(x) for x in list(tr.get("snr_db_choices", [-5.0, 0.0, 5.0]))]
    sigma = float(tr.get("random_weight_sigma", 0.7))
    snr_db = float(snr_choices[0])

    case = train_job._build_training_case(
        cfg=cfg,
        twc_cfg=twc,
        variant=args.variant,
        seed=base_seed + 9091,
        snr_db=snr_db,
        sigma=sigma,
        num_users=num_users,
        batch_size=batch_size,
    )

    model = UnfoldedWeightedWMMSE(
        num_layers=int(unfold_cfg.get("num_layers", 8)),
        init_damping=float(unfold_cfg.get("init_damping", 0.85)),
        init_dual_step_mu=float(unfold_cfg.get("init_dual_step_mu", 0.02)),
        init_dual_step_lambda=float(unfold_cfg.get("init_dual_step_lambda", 0.10)),
    )

    weights_tf = tf.convert_to_tensor(case["weights"], dtype=tf.float32)
    with tf.GradientTape() as tape:
        result = model(
            cfg=cfg,
            H=case["H"],
            noise_var_watt=float(case["noise_var_watt"]),
            fs=case["fs"],
            user_weights=weights_tf,
            bs_total_tx_power_watt=p_tot_watt,
            fs_mode="budget_dual",
        )
        loss = differentiable_loss(
            cfg=cfg,
            result=result,
            fs=case["fs"],
            user_weights=weights_tf,
            noise_var_watt=float(case["noise_var_watt"]),
            lambda_power=float(tr.get("lambda_power", 25.0)),
            lambda_fs=float(tr.get("lambda_fs", 25.0)),
            lambda_smooth=float(tr.get("lambda_smooth", 1e-2)),
            model=model,
        )

    raw_grads = tape.gradient(loss, model.trainable_variables)
    grads = train_job._project_grads_to_variable_dtype(raw_grads, model.trainable_variables)

    _assert_finite_tensor(loss, "loss")
    for idx, g in enumerate(grads):
        _assert_finite_tensor(tf.convert_to_tensor(g), f"grad[{idx}]")

    print(
        "TRAIN_PREFLIGHT_OK "
        f"variant={args.variant} batch_size={batch_size} snr_db={snr_db:.2f} "
        f"loss={float(loss.numpy()):.6f} num_vars={len(model.trainable_variables)}"
    )


if __name__ == "__main__":
    main()
