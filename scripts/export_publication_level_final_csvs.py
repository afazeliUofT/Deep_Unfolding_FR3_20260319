#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from plot_publication_level_final import (
    MAIN_ALGOS,
    ABLATION_ALGOS,
    CONV_ALGOS,
    SCALING_ALGOS,
    SELECTIVITY_ALGOS,
    FER_ALGOS,
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
        out.insert(1 if len(out.columns) >= 1 else 0, "label", out["algorithm"].map(_label))
    return out


def _write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _load_convergence_history(root: Path, eval_dir: Path) -> tuple[pd.DataFrame, str, bool]:
    try:
        rollout_dir = _latest_prefixed_dir(root, "publication_convergence_rollout_")
    except FileNotFoundError:
        rollout_dir = None
    if rollout_dir is not None and (rollout_dir / "history_mean.csv").exists():
        df = pd.read_csv(rollout_dir / "history_mean.csv")
        return df, str(rollout_dir), True
    return pd.read_csv(eval_dir / "history_mean.csv"), str(eval_dir), False


def _append_note_to_guide(guide_path: Path, has_rollout_extension: bool) -> None:
    if not guide_path.exists():
        return
    text = guide_path.read_text(encoding="utf-8")
    additions = [
        "CSV subfolder: results_twc/Publication_Level_Final/csv_data contains the tabular data used in each publication-level figure.",
    ]
    if has_rollout_extension:
        additions.append(
            "Fig03 note: the deep-unfolding cognitive-mask curve is extended to 56 steps by continuing the trained 20-layer model with the learned layer-20 update reused for steps 21-56. This is a continued rollout, not a separately retrained 56-layer unfolded network."
        )
    else:
        additions.append(
            "Fig03 note: the deep-unfolding curves stop at layer 20 because the current unfolded models were trained with 20 layers."
        )
    missing = [line for line in additions if line not in text]
    if missing:
        if not text.endswith("\n"):
            text += "\n"
        text += "\n" + "\n".join(missing) + "\n"
        guide_path.write_text(text, encoding="utf-8")


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
    root = Path(args.output_root)
    if not root.is_absolute():
        root = _repo_root() / root

    eval_dir = Path(args.eval_dir) if args.eval_dir else _latest_prefixed_dir(root, "eval_all_")
    scaling_dir = Path(args.scaling_dir) if args.scaling_dir else _latest_prefixed_dir(root, "scaling_")
    selectivity_dir = Path(args.selectivity_dir) if args.selectivity_dir else _latest_prefixed_dir(root, "selectivity_")
    out_dir = Path(args.out_dir) if args.out_dir else root / "Publication_Level_Final"
    csv_dir = _ensure_dir(out_dir / "csv_data")
    repo_root = _repo_root()

    eval_summary = pd.read_csv(eval_dir / "summary.csv")
    history_mean, convergence_source, has_rollout_extension = _load_convergence_history(root, eval_dir)
    fer_df = pd.read_csv(eval_dir / "fer.csv")
    scaling_summary = pd.read_csv(scaling_dir / "summary.csv")
    gap_df = pd.read_csv(selectivity_dir / "gap.csv")

    # Fig01
    fig1 = _filter_algos(eval_summary, MAIN_ALGOS)
    fig1 = fig1[fig1["sweep_value"] >= 0].copy()
    fig1_wsr = _aggregate_mean_ci(fig1, ["algorithm", "sweep_value"], "weighted_sum_rate_bps_per_hz")
    fig1_wsr.insert(0, "panel", "main_performance")
    fig1_wsr.insert(1, "metric", "weighted_sum_rate_bps_per_hz")
    fig1_prot = _aggregate_mean_ci(fig1, ["algorithm", "sweep_value"], "protection_satisfaction")
    fig1_prot.insert(0, "panel", "incumbent_protection")
    fig1_prot.insert(1, "metric", "protection_satisfaction")
    _write(_with_labels(pd.concat([fig1_wsr, fig1_prot], ignore_index=True)), csv_dir / "Fig01_Main_SNR_Tradeoff.csv")

    # Fig02
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

    # Fig03
    fig3 = _filter_algos(history_mean, CONV_ALGOS)
    fig3 = fig3[np.isclose(fig3["sweep_value"], 15.0)].copy()
    _write(_with_labels(fig3), csv_dir / "Fig03_Convergence_Quality.csv")

    # Fig04
    fig4 = _filter_algos(scaling_summary, SCALING_ALGOS)
    fig4 = _aggregate_mean_ci(fig4, ["algorithm", "num_bs_ant", "num_ut_per_sector"], "weighted_sum_rate_bps_per_hz")
    fig4.insert(0, "metric", "weighted_sum_rate_bps_per_hz")
    _write(_with_labels(fig4), csv_dir / "Fig04_Large_Array_Scaling.csv")

    # Fig05
    fig5 = _filter_algos(gap_df, SELECTIVITY_ALGOS)
    fig5 = _aggregate_mean_ci(fig5, ["algorithm", "tau_rms_ns"], "rate_gap_bps_per_hz")
    fig5.insert(0, "metric", "rate_gap_bps_per_hz")
    _write(_with_labels(fig5), csv_dir / "Fig05_Frequency_Selectivity_Robustness.csv")

    # Fig06
    fer_sub = _filter_algos(fer_df, FER_ALGOS)
    chosen, mcs_label = _select_publication_mcs(fer_sub)
    chosen, zero_error_floor = _prepare_fer_display_df(chosen, repo_root)
    xlo, xhi = _xlim_from_transition(chosen, x_col="fer_input_sinr_db")
    chosen = _with_labels(chosen)
    _write(chosen, csv_dir / "Fig06_Selected_FER.csv")
    fig6_meta = pd.DataFrame(
        [{
            "selected_case": mcs_label,
            "fer_input_sinr_min_db": xlo,
            "fer_input_sinr_max_db": xhi,
            "zero_error_display_floor": zero_error_floor,
            "algorithms": ";".join(sorted(chosen["algorithm"].unique().tolist())),
            "source_eval_dir": str(eval_dir),
        }]
    )
    _write(fig6_meta, csv_dir / "Fig06_Selected_FER_metadata.csv")

    meta = pd.DataFrame(
        [
            {"key": "eval_dir", "value": str(eval_dir)},
            {"key": "scaling_dir", "value": str(scaling_dir)},
            {"key": "selectivity_dir", "value": str(selectivity_dir)},
            {"key": "convergence_source", "value": convergence_source},
            {"key": "output_dir", "value": str(out_dir)},
            {"key": "du_num_layers_trained", "value": "20"},
            {"key": "du_fig03_rollout_layers", "value": "56" if has_rollout_extension else "20"},
            {
                "key": "note",
                "value": "Figure 3 can be extended without retraining from scratch by continuing the trained 20-layer deep-unfolding cognitive model with the learned layer-20 update reused after layer 20. This is a continued rollout, not a separately retrained 56-layer unfolded network." if has_rollout_extension else "Figure 3 uses the saved 20-layer unfolded model directly.",
            },
        ]
    )
    _write(meta, csv_dir / "00_Publication_Level_Final_Metadata.csv")

    guide_path = out_dir / "00_Publication_Figure_Guide.txt"
    _append_note_to_guide(guide_path, has_rollout_extension=has_rollout_extension)
    print(f"EXPORT_OK csv_dir={csv_dir}")


if __name__ == "__main__":
    main()
