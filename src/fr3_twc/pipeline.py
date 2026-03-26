from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import time
import numpy as np
import pandas as pd
import tensorflow as tf

from fr3_sim.channel import FsStats, generate_fs_stats, generate_ue_channels
from fr3_sim.config import ResolvedConfig
from fr3_sim.topology import FixedServiceLocations, TopologyData, generate_fixed_service_locations, generate_hexgrid_topology

from .common import ensure_dir, now_ts, save_csv, save_json, save_yaml, seed_all
from .fs_masks import apply_delta_mask, compute_cognitive_mask, summarize_mask
from .metrics_ext import extended_metrics, per_user_rates_from_mmse
from .plotting import plot_topology
from .solver import SolveOutput, weighted_wmmse_solve
from .unfolding import UnfoldedWeightedWMMSE
from .weights import (
    catalog_weight_profiles,
    edge_boost_weights_from_channel,
    proportional_fair_weights,
    uniform_weights,
    update_ema_rates,
)


@dataclass(frozen=True)
class AlgorithmSpec:
    name: str
    weight_mode: str
    fs_mode: str
    cognitive_mask: bool = False
    use_unfolding: bool = False
    checkpoint_name: Optional[str] = None


@dataclass(frozen=True)
class SuitePaths:
    root: Path
    summary_csv: Path
    history_csv: Path
    config_yaml: Path
    meta_json: Path
    figures_dir: Path


@dataclass(frozen=True)
class RunArtifacts:
    summary_df: pd.DataFrame
    history_df: pd.DataFrame
    paths: SuitePaths
    topo: TopologyData
    fs_loc: Optional[FixedServiceLocations]


def default_baseline_algorithms() -> List[AlgorithmSpec]:
    return [
        AlgorithmSpec(name="ew_no_fs", weight_mode="uniform", fs_mode="none"),
        AlgorithmSpec(name="ew_fs_soft", weight_mode="uniform", fs_mode="budget_dual"),
        AlgorithmSpec(name="edge_fs_soft", weight_mode="edge_boost", fs_mode="budget_dual"),
        AlgorithmSpec(name="pf_fs_soft", weight_mode="pf", fs_mode="budget_dual"),
        AlgorithmSpec(name="pf_fs_hybrid", weight_mode="pf", fs_mode="hybrid"),
        AlgorithmSpec(name="pf_fs_cognitive", weight_mode="pf", fs_mode="budget_dual", cognitive_mask=True),
    ]


def default_unfolded_algorithms() -> List[AlgorithmSpec]:
    return [
        AlgorithmSpec(name="du_pf_soft", weight_mode="pf", fs_mode="budget_dual", use_unfolding=True, checkpoint_name="soft"),
        AlgorithmSpec(name="du_pf_cognitive", weight_mode="pf", fs_mode="budget_dual", cognitive_mask=True, use_unfolding=True, checkpoint_name="cognitive"),
    ]


def _noise_from_snr(cfg: ResolvedConfig, snr_db: float) -> float:
    base = float(cfg.derived["ue_noise_re_watt"])
    return base * (10.0 ** (-float(snr_db) / 10.0))


def _suite_paths(cfg: ResolvedConfig, suite_name: str) -> SuitePaths:
    root = ensure_dir(Path(str(cfg.raw.get("twc", {}).get("output_root", "results_twc"))) / f"{suite_name}_{now_ts()}")
    figures = ensure_dir(root / "figures")
    return SuitePaths(
        root=root,
        summary_csv=root / "summary.csv",
        history_csv=root / "history.csv",
        config_yaml=root / "config_resolved.yaml",
        meta_json=root / "meta.json",
        figures_dir=figures,
    )


def _sweep_values(cfg: ResolvedConfig) -> List[float]:
    sw = cfg.raw.get("sweep", {}) or {}
    return [float(v) for v in list(sw.get("values", [0.0]))]


def _coverage_thresholds(cfg: ResolvedConfig) -> Tuple[float, float]:
    twc = cfg.raw.get("twc", {}) or {}
    return float(twc.get("coverage_rate_threshold_bpshz", 1.0)), float(twc.get("coverage_sinr_threshold_db", 0.0))


def _pf_cfg(cfg: ResolvedConfig) -> Mapping[str, float]:
    pf = cfg.raw.get("twc", {}).get("pf", {}) or {}
    return {
        "num_slots": int(pf.get("num_slots", 8)),
        "ema_beta": float(pf.get("ema_beta", 0.9)),
        "weight_clip_min": float(pf.get("weight_clip_min", 0.1)),
        "weight_clip_max": float(pf.get("weight_clip_max", 10.0)),
        "init_avg_rate": float(pf.get("init_avg_rate", 1.0)),
    }


def _cognitive_cfg(cfg: ResolvedConfig) -> Mapping[str, float]:
    cg = cfg.raw.get("twc", {}).get("cognitive_mask", {}) or {}
    return {
        "active_fraction": float(cg.get("active_fraction", 0.75)),
        "min_active_tones": int(cg.get("min_active_tones", 1)),
        "temperature": float(cg.get("temperature", 1.0)),
        "protect_top_l": int(cg.get("protect_top_l", 0)),
    }



def _scenario_seed(
    *,
    base_seed: int,
    sweep_index: int,
    batch_index: int,
    common_random_numbers_across_sweep: bool,
) -> int:
    if bool(common_random_numbers_across_sweep):
        return int(base_seed + batch_index)
    return int(base_seed + 10000 * sweep_index + batch_index)


def _prepare_fs_variant(cfg: ResolvedConfig, fs: Optional[FsStats], spec: AlgorithmSpec) -> Tuple[Optional[FsStats], Dict[str, float]]:
    if fs is None:
        return None, {}
    if not spec.cognitive_mask:
        return fs, {}
    cg = _cognitive_cfg(cfg)
    protect_top_l = None if int(cg["protect_top_l"]) <= 0 else int(cg["protect_top_l"])
    delta = compute_cognitive_mask(
        fs,
        active_fraction=float(cg["active_fraction"]),
        min_active_tones=int(cg["min_active_tones"]),
        temperature=float(cg["temperature"]),
        protect_top_l=protect_top_l,
    )
    fs_masked = apply_delta_mask(fs, delta)
    meta = summarize_mask(delta)
    return fs_masked, meta


def _weight_vector(spec: AlgorithmSpec, H: tf.Tensor, num_users: int, avg_rates: Optional[np.ndarray]) -> np.ndarray:
    if spec.weight_mode == "uniform":
        return uniform_weights(num_users)
    if spec.weight_mode == "edge_boost":
        return edge_boost_weights_from_channel(H)
    if spec.weight_mode == "pf":
        if avg_rates is None:
            return uniform_weights(num_users)
        return proportional_fair_weights(avg_rates)
    raise ValueError(f"Unsupported weight_mode={spec.weight_mode}")


def _snapshot_run(
    *,
    cfg: ResolvedConfig,
    spec: AlgorithmSpec,
    H: tf.Tensor,
    fs: Optional[FsStats],
    noise_var_watt: float,
    batch_index: int,
    sweep_value: float,
    model_registry: Optional[Mapping[str, UnfoldedWeightedWMMSE]] = None,
    avg_rates: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, float], List[Dict[str, float]], SolveOutput, np.ndarray]:
    num_users = int(cfg.derived["num_ut"])
    weights = _weight_vector(spec, H=H, num_users=num_users, avg_rates=avg_rates)
    fs_used, extra_meta = _prepare_fs_variant(cfg, fs, spec)

    start = time.perf_counter()
    if spec.use_unfolding:
        if model_registry is None or spec.checkpoint_name not in model_registry:
            raise KeyError(f"Missing unfolding model for checkpoint_name={spec.checkpoint_name}")
        solver = model_registry[spec.checkpoint_name]
        res = solver(
            cfg=cfg,
            H=H,
            noise_var_watt=noise_var_watt,
            fs=fs_used,
            user_weights=tf.convert_to_tensor(weights, dtype=tf.float32),
            fs_mode=spec.fs_mode,
        )
    else:
        res = weighted_wmmse_solve(
            cfg=cfg,
            H=H,
            noise_var_watt=noise_var_watt,
            fs=fs_used,
            user_weights=tf.convert_to_tensor(weights, dtype=tf.float32),
            fs_mode=spec.fs_mode,
        )
    runtime_sec = time.perf_counter() - start

    cov_rate_thr, cov_sinr_thr = _coverage_thresholds(cfg)
    metrics = extended_metrics(
        w=res.w,
        mmse=res.mmse,
        fs=fs_used if fs_used is not None else None,
        p_tot_watt=float(cfg.derived["bs_total_tx_power_watt"]),
        re_scaling=float(cfg.derived["re_scaling"]),
        num_re_sim=int(cfg.derived["num_re_sim"]),
        runtime_sec=runtime_sec,
        history=res.history,
        current_weights=weights,
        coverage_rate_threshold_bpshz=cov_rate_thr,
        coverage_sinr_threshold_db=cov_sinr_thr,
    )

    metrics.update(
        {
            "algorithm": spec.name,
            "sweep_value": float(sweep_value),
            "batch_index": float(batch_index),
            "num_iterations_or_layers": float(res.num_iter),
            "converged": float(bool(res.converged)),
            "weight_profile": spec.weight_mode,
            "fs_mode": spec.fs_mode,
            "used_unfolding": float(spec.use_unfolding),
        }
    )
    metrics.update({k: float(v) for k, v in extra_meta.items()})

    rates = per_user_rates_from_mmse(res.mmse, batch=int(H.shape[0]), num_re_sim=int(cfg.derived["num_re_sim"]))
    inst_rates = tf.reduce_mean(rates, axis=[0, 1]).numpy()

    hist_rows: List[Dict[str, float]] = []
    hist_np = {k: tf.cast(v, tf.float32).numpy().tolist() for k, v in res.history.items()}
    num_hist = max((len(v) for v in hist_np.values()), default=0)
    for it in range(num_hist):
        row = {
            "algorithm": spec.name,
            "sweep_value": float(sweep_value),
            "batch_index": float(batch_index),
            "iteration": float(it + 1),
        }
        for key, vals in hist_np.items():
            if it < len(vals):
                row[key] = float(vals[it])
        hist_rows.append(row)
    return metrics, hist_rows, res, np.asarray(inst_rates, dtype=np.float64)


def _pf_run(
    *,
    cfg: ResolvedConfig,
    spec: AlgorithmSpec,
    topo: TopologyData,
    fs: Optional[FsStats],
    noise_var_watt: float,
    batch_size: int,
    batch_index: int,
    sweep_value: float,
    model_registry: Optional[Mapping[str, UnfoldedWeightedWMMSE]] = None,
    H_slots: Optional[Sequence[tf.Tensor]] = None,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    pf = _pf_cfg(cfg)
    num_users = int(cfg.derived["num_ut"])
    avg_rates = np.full((num_users,), float(pf["init_avg_rate"]), dtype=np.float64)
    last_summary: Optional[Dict[str, float]] = None
    last_hist: List[Dict[str, float]] = []
    total_runtime = 0.0

    num_slots = int(pf["num_slots"])
    if H_slots is None:
        H_slots = [generate_ue_channels(cfg, topo, batch_size=batch_size) for _ in range(num_slots)]
    for slot in range(num_slots):
        H = H_slots[slot]
        summary, hist_rows, _res, inst_rates = _snapshot_run(
            cfg=cfg,
            spec=spec,
            H=H,
            fs=fs,
            noise_var_watt=noise_var_watt,
            batch_index=batch_index,
            sweep_value=sweep_value,
            model_registry=model_registry,
            avg_rates=avg_rates,
        )
        avg_rates = update_ema_rates(avg_rates, inst_rates, beta=float(pf["ema_beta"]))
        total_runtime += float(summary["runtime_sec"])
        last_summary = summary
        last_hist = [dict(row, slot=float(slot + 1)) for row in hist_rows]

    assert last_summary is not None
    last_summary["runtime_sec"] = total_runtime
    last_summary["pf_utility"] = float(np.sum(np.log(np.maximum(avg_rates, 1e-8))))
    last_summary["pf_avg_rate_bps_per_hz"] = float(np.mean(avg_rates))
    last_summary["pf_p05_rate_bps_per_hz"] = float(np.percentile(avg_rates, 5.0))
    last_summary["pf_jain_fairness"] = float((avg_rates.sum() ** 2) / (len(avg_rates) * np.sum(avg_rates ** 2) + 1e-12))
    last_summary["pf_slots"] = float(pf["num_slots"])
    return last_summary, last_hist


def run_suite(
    *,
    cfg: ResolvedConfig,
    suite_name: str,
    algorithms: Sequence[AlgorithmSpec],
    model_registry: Optional[Mapping[str, UnfoldedWeightedWMMSE]] = None,
) -> RunArtifacts:
    paths = _suite_paths(cfg, suite_name=suite_name)
    batch_size = int(cfg.raw.get("experiment", {}).get("batch_size", 1))
    num_batches = int(cfg.raw.get("experiment", {}).get("num_batches", 3))
    freeze_topology = bool(cfg.raw.get("experiment", {}).get("freeze_topology", True))
    sweep_values = _sweep_values(cfg)
    repro = cfg.raw.get("reproducibility", {}) or {}
    base_seed = int(repro.get("seed", 0))
    common_random_numbers_across_sweep = bool(repro.get("common_random_numbers_across_sweep", True))

    seed_all(base_seed)
    topo_fixed = generate_hexgrid_topology(cfg, batch_size=batch_size)
    fs_loc_fixed = generate_fixed_service_locations(cfg, topo_fixed, batch_size=batch_size) if bool(cfg.raw.get("fixed_service", {}).get("enabled", False)) else None
    fs_fixed = generate_fs_stats(cfg, topo_fixed, fs_loc_fixed, batch_size=batch_size) if fs_loc_fixed is not None else None

    save_yaml(paths.config_yaml, cfg.to_dict())
    plot_topology(topo_fixed, fs_loc_fixed, paths.figures_dir / "reference_geometry.png")

    summary_rows: List[Dict[str, float]] = []
    history_rows: List[Dict[str, float]] = []

    for sweep_index, sweep_value in enumerate(sweep_values):
        noise_var_watt = _noise_from_snr(cfg, snr_db=float(sweep_value))
        for batch_index in range(num_batches):
            seed_all(
                _scenario_seed(
                    base_seed=base_seed,
                    sweep_index=sweep_index,
                    batch_index=batch_index,
                    common_random_numbers_across_sweep=common_random_numbers_across_sweep,
                )
            )
            topo = topo_fixed if freeze_topology else generate_hexgrid_topology(cfg, batch_size=batch_size)
            fs_loc = fs_loc_fixed if freeze_topology else (
                generate_fixed_service_locations(cfg, topo, batch_size=batch_size) if bool(cfg.raw.get("fixed_service", {}).get("enabled", False)) else None
            )
            fs = fs_fixed if freeze_topology else (generate_fs_stats(cfg, topo, fs_loc, batch_size=batch_size) if fs_loc is not None else None)
            H = generate_ue_channels(cfg, topo, batch_size=batch_size)

            H_slots_pf = None
            if any(spec.weight_mode == "pf" for spec in algorithms):
                num_slots = int(_pf_cfg(cfg)["num_slots"])
                H_slots_pf = [generate_ue_channels(cfg, topo, batch_size=batch_size) for _ in range(num_slots)]

            for spec in algorithms:
                if spec.weight_mode == "pf":
                    summary, hist = _pf_run(
                        cfg=cfg,
                        spec=spec,
                        topo=topo,
                        fs=fs,
                        noise_var_watt=noise_var_watt,
                        batch_size=batch_size,
                        batch_index=batch_index,
                        sweep_value=float(sweep_value),
                        model_registry=model_registry,
                        H_slots=H_slots_pf,
                    )
                else:
                    summary, hist, _res, _inst = _snapshot_run(
                        cfg=cfg,
                        spec=spec,
                        H=H,
                        fs=fs,
                        noise_var_watt=noise_var_watt,
                        batch_index=batch_index,
                        sweep_value=float(sweep_value),
                        model_registry=model_registry,
                        avg_rates=None,
                    )
                summary_rows.append(summary)
                history_rows.extend(hist)

    summary_df = save_csv(paths.summary_csv, summary_rows)
    history_df = save_csv(paths.history_csv, history_rows) if history_rows else pd.DataFrame()
    save_json(
        paths.meta_json,
        {
            "suite_name": suite_name,
            "num_algorithms": len(algorithms),
            "algorithms": [spec.__dict__ for spec in algorithms],
            "num_rows": len(summary_rows),
            "num_history_rows": len(history_rows),
        },
    )
    return RunArtifacts(summary_df=summary_df, history_df=history_df, paths=paths, topo=topo_fixed, fs_loc=fs_loc_fixed)
