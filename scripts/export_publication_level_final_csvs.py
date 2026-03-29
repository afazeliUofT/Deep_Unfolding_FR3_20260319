#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from plot_publication_level_final import (
    ABLATION_ALGOS,
    CONV_ALGOS,
    FER_ALGOS,
    MAIN_ALGOS,
    SCALING_ALGOS,
    SELECTIVITY_ALGOS,
    _aggregate_mean_ci,
    _filter_algos,
    _label,
    _latest_prefixed_dir,
    _prepare_fer_display_df,
    _repo_root,
    _select_publication_mcs,
    _xlim_from_transition,
)


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _with_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "algorithm" in out.columns:
        insert_at = 1 if len(out.columns) >= 1 else 0
        out.insert(insert_at, "label", out["algorithm"].map(_label))
    return out


def _write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export CSV tables for Publication_Level_Final figures.")
    p.add_argument("--output-root", type=str, default="results_twc")
    p.add_argument("--eval-dir", type=str, default=None)
    p.add_argument("--scaling-dir", type=str, default=None)
    p.add_argument("--selectivity-dir", type=str, default=None)
    p.add_argument("--out-dir", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = _repo_root()
    root = Path(args.output_root)
    if not root.is_absolute():
        root = repo_root / root

    eval_dir = Path(args.eval_dir) if args.eval_dir else _latest_prefixed_dir(root, "eval_all_")
    scaling_dir = Path(args.scaling_dir) if args.scaling_dir else _latest_prefixed_dir(root, "scaling_")
    selectivity_dir = Path(args.selectivity_dir) if args.selectivity_dir else _latest_prefixed_dir(root, "selectivity_")
    out_dir = Path(args.out_dir) if args.out_dir else root / "Publication_Level_Final"
    csv_dir = _ensure_dir(out_dir / "csv_data")

    eval_summary = pd.read_csv(eval_dir / "summary.csv")
    history_mean = pd.read_csv(eval_dir / "history_mean.csv")
    fer_df = pd.read_csv(eval_dir / "fer.csv")
    scaling_summary = pd.read_csv(scaling_dir / "summary.csv")
    gap_df = pd.read_csv(selectivity_dir / "gap.csv")

    fig1 = _filter_algos(eval_summary, MAIN_ALGOS)
    fig1 = fig1[fig1["sweep_value"] >= 0].copy()
    fig1_wsr = _aggregate_mean_ci(fig1, ["algorithm", "sweep_value"], "weighted_sum_rate_bps_per_hz")
    fig1_wsr.insert(0, "panel", "main_performance")
    fig1_wsr.insert(1, "metric", "weighted_sum_rate_bps_per_hz")
    fig1_prot = _aggregate_mean_ci(fig1, ["algorithm", "sweep_value"], "protection_satisfaction")
    fig1_prot.insert(0, "panel", "incumbent_protection")
    fig1_prot.insert(1, "metric", "protection_satisfaction")
    _write(_with_labels(pd.concat([fig1_wsr, fig1_prot], ignore_index=True)), csv_dir / "Fig01_Main_SNR_Tradeoff.csv")

    fig2 = _filter_algos(eval_summary, ABLATION_ALGOS)
    fig2 = fig2[fig2["sweep_value"] >= 0].copy()
    fig2_wsr = _aggregate_mean_ci(fig2, ["algorithm", "sweep_value"], "weighted_sum_rate_bps_per_hz")
    fig2_wsr.insert(0, "panel", "quality")
    fig2_wsr.insert(1, "metric", "weighted_sum_rate_bps_per_hz")
    fig2_run = _aggregate_mean_ci(fig2, ["algorithm", "sweep_value"], "runtime_sec")
    fig2_run.insert(0, "panel", "runtime")
    fig2_run.insert(1, "metric", "runtime_sec")
    fig2_prot25 = (
        fig2[fig2["sweep_value"] == 25]
        .groupby("algorithm", dropna=False)["protection_satisfaction"]
        .mean()
        .reset_index()
        .rename(columns={"protection_satisfaction": "mean"})
    )
    fig2_prot25.insert(0, "panel", "protection_at_25dB")
    fig2_prot25.insert(1, "metric", "protection_satisfaction")
    fig2_prot25["sweep_value"] = 25.0
    fig2_prot25["std"] = np.nan
    fig2_prot25["count"] = np.nan
    fig2_prot25["half_ci"] = np.nan
    fig2_prot25["lower"] = np.nan
    fig2_prot25["upper"] = np.nan
    _write(_with_labels(pd.concat([fig2_wsr, fig2_run, fig2_prot25], ignore_index=True)), csv_dir / "Fig02_Unfolding_Ablation.csv")

    fig3 = _filter_algos(history_mean, CONV_ALGOS)
    fig3 = fig3[np.isclose(fig3["sweep_value"], 15.0)].copy()
    _write(_with_labels(fig3), csv_dir / "Fig03_Convergence_Quality.csv")

    fig4 = _filter_algos(scaling_summary, SCALING_ALGOS)
    fig4 = _aggregate_mean_ci(fig4, ["algorithm", "num_bs_ant", "num_ut_per_sector"], "weighted_sum_rate_bps_per_hz")
    fig4.insert(0, "metric", "weighted_sum_rate_bps_per_hz")
    _write(_with_labels(fig4), csv_dir / "Fig04_Large_Array_Scaling.csv")

    fig5 = _filter_algos(gap_df, SELECTIVITY_ALGOS)
    fig5 = _aggregate_mean_ci(fig5, ["algorithm", "tau_rms_ns"], "rate_gap_bps_per_hz")
    fig5.insert(0, "metric", "rate_gap_bps_per_hz")
    _write(_with_labels(fig5), csv_dir / "Fig05_Frequency_Selectivity_Robustness.csv")

    fer_sub = _filter_algos(fer_df, FER_ALGOS)
    chosen, mcs_label = _select_publication_mcs(fer_sub)
    chosen, zero_error_floor = _prepare_fer_display_df(chosen, repo_root)
    xlo, xhi = _xlim_from_transition(chosen, x_col="fer_input_sinr_db")
    chosen = _with_labels(chosen)
    _write(chosen, csv_dir / "Fig06_Selected_FER.csv")
    fig6_meta = pd.DataFrame(
        [{
            "selected_case": mcs_label,
            "effective_sinr_min_db": xlo,
            "effective_sinr_max_db": xhi,
            "zero_error_display_floor": zero_error_floor,
            "algorithms": ";".join(sorted(chosen["algorithm"].unique().tolist())),
            "source_eval_dir": str(eval_dir),
            "x_axis_used_for_plot": "fer_input_sinr_db",
            "left_panel_used_for_plot": "sweep_value_to_fer_input_sinr_db",
        }]
    )
    _write(fig6_meta, csv_dir / "Fig06_Selected_FER_metadata.csv")

    meta = pd.DataFrame(
        [
            {"key": "eval_dir", "value": str(eval_dir)},
            {"key": "scaling_dir", "value": str(scaling_dir)},
            {"key": "selectivity_dir", "value": str(selectivity_dir)},
            {"key": "output_dir", "value": str(out_dir)},
            {"key": "du_num_layers", "value": "20"},
            {"key": "wmmse_iterations_eval", "value": "56"},
            {"key": "figure6_x_axis", "value": "Median effective user SINR [dB]"},
            {"key": "figure6_zero_error_floor", "value": str(zero_error_floor)},
            {"key": "note", "value": "Figure 6 now shows the effective-SINR mapping and uses a conservative display floor for zero-error FER points. Deep-unfolding convergence traces stop at layer 20 because the saved unfolded models have 20 layers. Classical iterative baselines continue to the eval receiver iteration limit."},
        ]
    )
    _write(meta, csv_dir / "00_Publication_Level_Final_Metadata.csv")

    print(f"EXPORT_OK csv_dir={csv_dir}")


if __name__ == "__main__":
    main()
