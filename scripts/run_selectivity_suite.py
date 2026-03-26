from __future__ import annotations

import _repo_bootstrap as _rb

_rb.bootstrap()

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

from fr3_sim.channel import generate_fs_stats, generate_ue_channels
from fr3_sim.topology import generate_fixed_service_locations, generate_hexgrid_topology
from fr3_twc.common import ensure_dir, now_ts, save_json, save_yaml, seed_all
from fr3_twc.config import get_twc_paths, load_twc_config
from fr3_twc.pipeline import _pf_run, default_baseline_algorithms, default_unfolded_algorithms
from fr3_twc.reporting import load_models, save_grouped_mean
from fr3_twc.selectivity import make_frequency_selective_channels, summarize_flat_vs_selective



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run flat-vs-selective PF/unfolding suite")
    p.add_argument("--config", type=str, default="configs/twc_selectivity.yaml")
    p.add_argument("--set", dest="overrides", action="append", default=None)
    return p.parse_args()



def _noise(cfg, snr_db: float) -> float:
    return float(cfg.derived["ue_noise_re_watt"]) * (10.0 ** (-float(snr_db) / 10.0))



def main() -> None:
    args = parse_args()
    cfg = load_twc_config(args.config, overrides=args.overrides)
    seed_all(int(cfg.raw["reproducibility"]["seed"]))
    twc = cfg.raw.get("twc", {}) or {}
    sel = twc.get("selectivity", {}) or {}
    pf = twc.get("pf", {}) or {}

    twc_paths = get_twc_paths(cfg)
    models = load_models(twc_paths.checkpoint_root, names=["soft", "cognitive"])
    missing = [n for n in ["soft", "cognitive"] if n not in models]
    if missing:
        raise FileNotFoundError(f"Missing checkpoints in {twc_paths.checkpoint_root}: {missing}")

    root = ensure_dir(Path(twc_paths.output_root) / f"selectivity_{now_ts()}")
    save_yaml(root / "config_resolved.yaml", cfg.to_dict())
    save_json(root / "meta.json", {"suite": "selectivity"})

    algs = [
        spec
        for spec in (default_baseline_algorithms() + default_unfolded_algorithms())
        if spec.name in {"pf_fs_soft", "pf_fs_cognitive", "du_pf_soft", "du_pf_cognitive"}
    ]

    batch_size = int(cfg.raw.get("experiment", {}).get("batch_size", 1))
    num_batches = int(sel.get("num_batches", cfg.raw.get("experiment", {}).get("num_batches", 4)))
    freeze_topology = bool(cfg.raw.get("experiment", {}).get("freeze_topology", True))
    num_slots = int(pf.get("num_slots", 6))
    snr_db = float(sel.get("sweep_value_db", 5.0))
    noise_var_watt = _noise(cfg, snr_db)

    topo_fixed = generate_hexgrid_topology(cfg, batch_size=batch_size)
    fs_loc_fixed = generate_fixed_service_locations(cfg, topo_fixed, batch_size=batch_size) if bool(cfg.raw.get("fixed_service", {}).get("enabled", False)) else None
    fs_fixed = generate_fs_stats(cfg, topo_fixed, fs_loc_fixed, batch_size=batch_size) if fs_loc_fixed is not None else None

    summary_rows: List[Dict[str, float | str]] = []
    gap_rows: List[Dict[str, float | str]] = []
    repro = cfg.raw.get("reproducibility", {}) or {}
    common_random_numbers_across_sweep = bool(repro.get("common_random_numbers_across_sweep", True))
    base_seed = int(cfg.raw["reproducibility"]["seed"]) + 100

    for tau in list(sel.get("tau_rms_ns_values", [20.0, 50.0, 100.0])):
        diag = summarize_flat_vs_selective(cfg=cfg, tau_rms_ns=float(tau))
        for batch_index in range(num_batches):
            if common_random_numbers_across_sweep:
                seed_all(base_seed + batch_index)
            topo = topo_fixed if freeze_topology else generate_hexgrid_topology(cfg, batch_size=batch_size)
            fs_loc = fs_loc_fixed if freeze_topology else (
                generate_fixed_service_locations(cfg, topo, batch_size=batch_size) if bool(cfg.raw.get("fixed_service", {}).get("enabled", False)) else None
            )
            fs = fs_fixed if freeze_topology else (generate_fs_stats(cfg, topo, fs_loc, batch_size=batch_size) if fs_loc is not None else None)

            H_slots_flat = [generate_ue_channels(cfg, topo, batch_size=batch_size) for _ in range(num_slots)]
            H_slots_sel = [
                make_frequency_selective_channels(H0, cfg=cfg, tau_rms_ns=float(tau), seed=base_seed + 1000 * batch_index + 13 * slot)
                for slot, H0 in enumerate(H_slots_flat)
            ]

            for spec in algs:
                s_flat, _ = _pf_run(
                    cfg=cfg,
                    spec=spec,
                    topo=topo,
                    fs=fs,
                    noise_var_watt=noise_var_watt,
                    batch_size=batch_size,
                    batch_index=batch_index,
                    sweep_value=snr_db,
                    model_registry=models,
                    H_slots=H_slots_flat,
                )
                s_sel, _ = _pf_run(
                    cfg=cfg,
                    spec=spec,
                    topo=topo,
                    fs=fs,
                    noise_var_watt=noise_var_watt,
                    batch_size=batch_size,
                    batch_index=batch_index,
                    sweep_value=snr_db,
                    model_registry=models,
                    H_slots=H_slots_sel,
                )
                row_flat = dict(s_flat)
                row_flat.update(diag)
                row_flat.update({"channel_modeling": "flat", "tau_rms_ns": float(tau), "batch_index": float(batch_index)})
                row_sel = dict(s_sel)
                row_sel.update(diag)
                row_sel.update({"channel_modeling": "selective", "tau_rms_ns": float(tau), "batch_index": float(batch_index)})
                summary_rows.extend([row_flat, row_sel])

                fairness_key = "pf_jain_fairness" if "pf_jain_fairness" in s_flat else "jain_fairness"
                gap_rows.append(
                    {
                        "algorithm": spec.name,
                        "tau_rms_ns": float(tau),
                        "batch_index": float(batch_index),
                        **diag,
                        "rate_gap_bps_per_hz": float(s_flat["weighted_sum_rate_bps_per_hz"] - s_sel["weighted_sum_rate_bps_per_hz"]),
                        "fairness_gap": float(s_flat.get(fairness_key, float('nan')) - s_sel.get(fairness_key, float('nan'))),
                        "coverage_gap": float(s_flat.get("coverage_rate", float('nan')) - s_sel.get("coverage_rate", float('nan'))),
                        "protection_gap": float(s_flat.get("protection_satisfaction", float('nan')) - s_sel.get("protection_satisfaction", float('nan'))),
                    }
                )
            print(f"finished tau={float(tau):.1f} ns batch={batch_index}")

    summary_df = pd.DataFrame(summary_rows)
    gap_df = pd.DataFrame(gap_rows)
    summary_df.to_csv(root / "summary.csv", index=False)
    gap_df.to_csv(root / "gap.csv", index=False)
    save_grouped_mean(summary_df, root / "summary_mean.csv", group_cols=["algorithm", "tau_rms_ns", "channel_modeling"])
    save_grouped_mean(gap_df, root / "gap_mean.csv", group_cols=["algorithm", "tau_rms_ns"])
    print(f"Saved selectivity suite to: {root}")


if __name__ == "__main__":
    main()
