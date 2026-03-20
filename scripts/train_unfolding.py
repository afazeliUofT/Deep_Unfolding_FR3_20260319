from __future__ import annotations

import _repo_bootstrap as _rb

_rb.bootstrap()

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import tensorflow as tf

from fr3_sim.channel import generate_fs_stats, generate_ue_channels
from fr3_sim.topology import generate_fixed_service_locations, generate_hexgrid_topology
from fr3_twc.common import ensure_dir, now_ts, save_json, save_yaml, seed_all
from fr3_twc.config import get_twc_paths, load_twc_config
from fr3_twc.fs_masks import apply_delta_mask, compute_cognitive_mask, summarize_mask
from fr3_twc.metrics_ext import extended_metrics
from fr3_twc.unfolding import UnfoldedWeightedWMMSE, differentiable_loss
from fr3_twc.weights import proportional_fair_weights


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train unfolded weighted-WMMSE")
    p.add_argument("--config", type=str, default="configs/twc_base.yaml")
    p.add_argument("--variant", type=str, choices=["soft", "cognitive"], default="soft")
    p.add_argument("--set", dest="overrides", action="append", default=None)
    return p.parse_args()



def _base_noise(cfg, snr_db: float) -> float:
    return float(cfg.derived["ue_noise_re_watt"]) * (10.0 ** (-float(snr_db) / 10.0))



def _training_row(step: int, snr_db: float, loss: float, grad_norm: float, summary: Dict[str, float], mask_meta: Dict[str, float]) -> Dict[str, float]:
    row: Dict[str, float] = {
        "step": float(step),
        "snr_db": float(snr_db),
        "loss": float(loss),
        "grad_norm": float(grad_norm),
    }
    row.update({k: float(v) for k, v in summary.items()})
    row.update({k: float(v) for k, v in mask_meta.items()})
    return row



def main() -> None:
    args = parse_args()
    cfg = load_twc_config(args.config, overrides=args.overrides)
    seed_all(int(cfg.raw["reproducibility"]["seed"]))

    paths = get_twc_paths(cfg)
    out_root = ensure_dir(Path(paths.output_root) / f"train_{args.variant}_{now_ts()}")
    ckpt_root = ensure_dir(paths.checkpoint_root)
    local_ckpts = ensure_dir(out_root / "checkpoints")
    save_yaml(out_root / "config_resolved.yaml", cfg.to_dict())
    save_json(out_root / "meta.json", {"variant": args.variant})

    twc = cfg.raw.get("twc", {}) or {}
    unfold_cfg = twc.get("unfolding", {}) or {}
    tr = twc.get("training", {}) or {}
    batch_size = int(cfg.raw.get("experiment", {}).get("batch_size", 1))
    num_users = int(cfg.derived["num_ut"])
    p_tot_watt = float(cfg.derived["bs_total_tx_power_watt"])
    rng = np.random.default_rng(int(cfg.raw["reproducibility"]["seed"]) + (11 if args.variant == "cognitive" else 3))

    model = UnfoldedWeightedWMMSE(
        num_layers=int(unfold_cfg.get("num_layers", 8)),
        init_damping=float(unfold_cfg.get("init_damping", 0.85)),
        init_dual_step_mu=float(unfold_cfg.get("init_dual_step_mu", 0.02)),
        init_dual_step_lambda=float(unfold_cfg.get("init_dual_step_lambda", 0.10)),
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=float(tr.get("learning_rate", 5e-3)))

    num_steps = int(tr.get("num_steps", 300))
    log_every = int(tr.get("log_every", 10))
    save_every = int(tr.get("save_every", 25))
    grad_clip_norm = float(tr.get("grad_clip_norm", 5.0))
    snr_choices = list(tr.get("snr_db_choices", [-5.0, 0.0, 5.0]))
    sigma = float(tr.get("random_weight_sigma", 0.7))

    best_loss = np.inf
    rows = []

    for step in range(1, num_steps + 1):
        snr_db = float(rng.choice(snr_choices))
        noise_var_watt = _base_noise(cfg, snr_db)
        topo = generate_hexgrid_topology(cfg, batch_size=batch_size)
        fs_loc = generate_fixed_service_locations(cfg, topo, batch_size=batch_size) if bool(cfg.raw.get("fixed_service", {}).get("enabled", False)) else None
        fs = generate_fs_stats(cfg, topo, fs_loc, batch_size=batch_size) if fs_loc is not None else None
        mask_meta: Dict[str, float] = {}
        if fs is not None and args.variant == "cognitive":
            cg = twc.get("cognitive_mask", {}) or {}
            delta = compute_cognitive_mask(
                fs,
                active_fraction=float(cg.get("active_fraction", 0.5)),
                min_active_tones=int(cg.get("min_active_tones", 1)),
                temperature=float(cg.get("temperature", 1.0)),
                protect_top_l=int(cg.get("protect_top_l", 3)),
            )
            fs = apply_delta_mask(fs, delta)
            mask_meta = summarize_mask(delta)

        H = generate_ue_channels(cfg, topo, batch_size=batch_size)
        avg_rates = rng.lognormal(mean=0.0, sigma=sigma, size=num_users)
        weights = proportional_fair_weights(avg_rates)
        weights_tf = tf.convert_to_tensor(weights, dtype=tf.float32)

        with tf.GradientTape() as tape:
            result = model(
                cfg=cfg,
                H=H,
                noise_var_watt=noise_var_watt,
                fs=fs,
                user_weights=weights_tf,
                bs_total_tx_power_watt=p_tot_watt,
                fs_mode="budget_dual",
            )
            loss = differentiable_loss(
                cfg=cfg,
                result=result,
                fs=fs,
                user_weights=weights_tf,
                noise_var_watt=noise_var_watt,
                lambda_power=float(tr.get("lambda_power", 25.0)),
                lambda_fs=float(tr.get("lambda_fs", 25.0)),
                lambda_smooth=float(tr.get("lambda_smooth", 1e-2)),
                model=model,
            )

        grads = tape.gradient(loss, model.trainable_variables)
        grads = [tf.zeros_like(v) if g is None else g for g, v in zip(grads, model.trainable_variables)]
        grad_norm = float(tf.linalg.global_norm(grads).numpy())
        if grad_clip_norm > 0:
            grads, _ = tf.clip_by_global_norm(grads, grad_clip_norm)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        summary = extended_metrics(
            w=result.w,
            mmse=result.mmse,
            fs=fs,
            p_tot_watt=p_tot_watt,
            re_scaling=float(cfg.derived["re_scaling"]),
            num_re_sim=int(cfg.derived["num_re_sim"]),
            runtime_sec=0.0,
            history=result.history,
            current_weights=weights,
        )
        rows.append(_training_row(step, snr_db, float(loss.numpy()), grad_norm, summary, mask_meta))

        loss_f = float(loss.numpy())
        if loss_f < best_loss:
            best_loss = loss_f
            model.save_npz(ckpt_root / f"{args.variant}.npz")

        if step % save_every == 0:
            model.save_npz(local_ckpts / f"{args.variant}_step{step:04d}.npz")

        if step % log_every == 0 or step == 1 or step == num_steps:
            print(
                f"step={step:04d} variant={args.variant} snr_db={snr_db:+.1f} "
                f"loss={loss_f:.4f} wsr={summary['weighted_sum_rate_bps_per_hz']:.3f} "
                f"fs_violation={summary['max_fs_violation_watt']:.3e}"
            )

    df = pd.DataFrame(rows)
    df.to_csv(out_root / "train_log.csv", index=False)
    model.save_npz(out_root / f"{args.variant}_final.npz")
    print(f"Saved training outputs to: {out_root}")
    print(f"Best checkpoint: {ckpt_root / (args.variant + '.npz')}")


if __name__ == "__main__":
    main()
