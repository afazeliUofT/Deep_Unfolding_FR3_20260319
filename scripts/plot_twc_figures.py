from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from fr3_twc.common import ensure_dir, now_ts
from fr3_twc.plotting import plot_convergence, plot_fer, plot_metric_vs_sweep, plot_scaling_heatmap, plot_selectivity_gap
from fr3_twc.reporting import latest_prefixed_dir



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate publication-grade figures from latest TWC outputs")
    p.add_argument("--output-root", type=str, default="results_twc")
    p.add_argument("--eval-dir", type=str, default=None)
    p.add_argument("--baseline-dir", type=str, default=None)
    p.add_argument("--scaling-dir", type=str, default=None)
    p.add_argument("--selectivity-dir", type=str, default=None)
    p.add_argument("--out-dir", type=str, default=None)
    return p.parse_args()



def _pick_csv(root: Path, preferred: str, fallback: str) -> Path | None:
    p = root / preferred
    if p.exists():
        return p
    q = root / fallback
    if q.exists():
        return q
    return None



def _fairness_col(df: pd.DataFrame) -> str | None:
    for c in ["pf_jain_fairness", "jain_fairness"]:
        if c in df.columns:
            return c
    return None



def _copy_geometry(src_dir: Path, out_dir: Path) -> None:
    cand = src_dir / "figures" / "reference_geometry.png"
    if cand.exists():
        shutil.copy2(cand, out_dir / "reference_geometry.png")



def _pareto_plot(df: pd.DataFrame, out_path: Path) -> None:
    fairness = _fairness_col(df)
    if fairness is None or "protection_satisfaction" not in df.columns:
        return
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for algo, sub in df.groupby("algorithm"):
        ax.plot(sub["protection_satisfaction"].to_numpy(), sub[fairness].to_numpy(), marker="o", linestyle="", label=str(algo))
    ax.set_xlabel("Protection satisfaction")
    ax.set_ylabel(fairness)
    ax.set_title("Fairness vs protection")
    ax.grid(True)
    ax.legend(fontsize=8)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)



def main() -> None:
    args = parse_args()
    root = Path(args.output_root)
    eval_dir = Path(args.eval_dir) if args.eval_dir else latest_prefixed_dir(root, "eval_all_")
    baseline_dir = Path(args.baseline_dir) if args.baseline_dir else latest_prefixed_dir(root, "baselines_")
    scaling_dir = Path(args.scaling_dir) if args.scaling_dir else latest_prefixed_dir(root, "scaling_")
    selectivity_dir = Path(args.selectivity_dir) if args.selectivity_dir else latest_prefixed_dir(root, "selectivity_")
    out_dir = ensure_dir(Path(args.out_dir) if args.out_dir else root / f"figures_twc_{now_ts()}")

    source_for_geometry = eval_dir or baseline_dir
    if source_for_geometry is not None:
        _copy_geometry(source_for_geometry, out_dir)

    if eval_dir is not None:
        summary_csv = _pick_csv(eval_dir, "summary_mean.csv", "summary.csv")
        hist_csv = _pick_csv(eval_dir, "history_mean.csv", "history.csv")
        if summary_csv is not None:
            df = pd.read_csv(summary_csv)
            if "weighted_sum_rate_bps_per_hz" in df.columns:
                plot_metric_vs_sweep(df, "weighted_sum_rate_bps_per_hz", out_dir / "01_weighted_sum_rate_vs_snr.png")
            if "sum_rate_bps_per_hz" in df.columns:
                plot_metric_vs_sweep(df, "sum_rate_bps_per_hz", out_dir / "02_sum_rate_vs_snr.png")
            fairness = _fairness_col(df)
            if fairness is not None:
                plot_metric_vs_sweep(df, fairness, out_dir / "03_fairness_vs_snr.png")
            if "protection_satisfaction" in df.columns:
                plot_metric_vs_sweep(df, "protection_satisfaction", out_dir / "04_protection_vs_snr.png")
            if "coverage_rate" in df.columns:
                plot_metric_vs_sweep(df, "coverage_rate", out_dir / "05_coverage_vs_snr.png")
            if "runtime_sec" in df.columns:
                plot_metric_vs_sweep(df, "runtime_sec", out_dir / "06_runtime_vs_snr.png")
            _pareto_plot(df, out_dir / "07_fairness_vs_protection.png")

        if hist_csv is not None:
            h = pd.read_csv(hist_csv)
            if "w_delta" in h.columns:
                plot_convergence(h, "w_delta", out_dir / "08_convergence_w_delta.png")
            if "weighted_sum_rate" in h.columns:
                plot_convergence(h, "weighted_sum_rate", out_dir / "09_convergence_wsr.png")
            elif "final_history_weighted_sum_rate" in h.columns:
                plot_convergence(h, "final_history_weighted_sum_rate", out_dir / "09_convergence_wsr.png")

        fer_csv = eval_dir / "fer.csv"
        if fer_csv.exists():
            fer_df = pd.read_csv(fer_csv)
            plot_fer(fer_df, out_dir / "10_fer.png")

    if scaling_dir is not None:
        scaling_csv = _pick_csv(scaling_dir, "summary_mean.csv", "summary.csv")
        if scaling_csv is not None:
            df = pd.read_csv(scaling_csv)
            for algo in ["pf_fs_soft", "pf_fs_cognitive", "du_pf_soft", "du_pf_cognitive"]:
                sub = df[df["algorithm"] == algo]
                if not sub.empty and "weighted_sum_rate_bps_per_hz" in sub.columns:
                    plot_scaling_heatmap(sub, "weighted_sum_rate_bps_per_hz", out_dir / f"11_scaling_wsr_{algo}.png")
                if not sub.empty and "runtime_sec" in sub.columns:
                    plot_scaling_heatmap(sub, "runtime_sec", out_dir / f"12_scaling_runtime_{algo}.png")

    if selectivity_dir is not None:
        gap_csv = _pick_csv(selectivity_dir, "gap_mean.csv", "gap.csv")
        if gap_csv is not None:
            gap_df = pd.read_csv(gap_csv)
            if "rate_gap_bps_per_hz" in gap_df.columns:
                plot_selectivity_gap(gap_df, out_dir / "13_selectivity_rate_gap.png")
            if "fairness_gap" in gap_df.columns:
                plot_selectivity_gap(gap_df.rename(columns={"fairness_gap": "rate_gap_bps_per_hz"}), out_dir / "14_selectivity_fairness_gap.png")

    print(f"Saved figures to: {out_dir}")


if __name__ == "__main__":
    main()
