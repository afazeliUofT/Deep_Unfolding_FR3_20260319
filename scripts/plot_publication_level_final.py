#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

try:
    from scipy.interpolate import PchipInterpolator  # type: ignore
except Exception:  # pragma: no cover
    PchipInterpolator = None


ALGO_LABELS = {
    "du_pf_cognitive": "DU-Cognitive",
    "du_pf_soft": "DU-Soft",
    "pf_fs_cognitive": "PF-Cognitive",
    "pf_fs_soft": "PF-Soft",
    "pf_fs_hybrid": "PF-Hybrid",
    "edge_fs_soft": "Edge-Soft",
    "ew_fs_soft": "EW-Soft",
    "ew_no_fs": "EW-NoFS",
}

ALGO_ORDER = [
    "du_pf_cognitive",
    "pf_fs_cognitive",
    "du_pf_soft",
    "pf_fs_soft",
    "pf_fs_hybrid",
    "edge_fs_soft",
    "ew_fs_soft",
    "ew_no_fs",
]

MCS_LABELS = {2: "QPSK", 4: "16-QAM", 6: "64-QAM", 8: "256-QAM"}

MAIN_ALGOS = [
    "du_pf_cognitive",
    "pf_fs_cognitive",
    "edge_fs_soft",
    "pf_fs_soft",
    "pf_fs_hybrid",
]
ABLATION_ALGOS = ["du_pf_cognitive", "du_pf_soft", "pf_fs_cognitive"]
CONV_ALGOS = ["du_pf_cognitive", "pf_fs_cognitive", "edge_fs_soft"]
SCALING_ALGOS = ["du_pf_cognitive", "pf_fs_cognitive"]
SELECTIVITY_ALGOS = ["du_pf_cognitive", "pf_fs_cognitive", "pf_fs_soft"]
FER_ALGOS = ["du_pf_cognitive", "pf_fs_cognitive", "edge_fs_soft", "pf_fs_soft"]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _latest_prefixed_dir(root: Path, prefix: str) -> Path:
    dirs = sorted(p for p in root.glob(f"{prefix}*") if p.is_dir())
    if not dirs:
        raise FileNotFoundError(f"Could not find any directory with prefix '{prefix}' under {root}")
    return dirs[-1]


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", dpi=400, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _ordered_unique(values: Iterable[str]) -> list[str]:
    present = list(dict.fromkeys(values))
    rank = {a: i for i, a in enumerate(ALGO_ORDER)}
    return sorted(present, key=lambda x: (rank.get(x, 999), x))


def _label(algo: str) -> str:
    return ALGO_LABELS.get(algo, algo)


def _filter_algos(df: pd.DataFrame, algos: Sequence[str]) -> pd.DataFrame:
    sub = df[df["algorithm"].isin(algos)].copy()
    if sub.empty:
        raise ValueError(f"No rows left after filtering algorithms: {algos}")
    sub["_algo_order"] = sub["algorithm"].map({a: i for i, a in enumerate(ALGO_ORDER)}).fillna(999)
    return sub.sort_values(["_algo_order"] + [c for c in sub.columns if c in {"sweep_value", "num_bs_ant", "tau_rms_ns", "iteration"}]).drop(columns=["_algo_order"])


def _aggregate_mean_ci(df: pd.DataFrame, group_cols: Sequence[str], value_col: str, ci_z: float = 1.96) -> pd.DataFrame:
    grp = df.groupby(list(group_cols), dropna=False)[value_col].agg(["mean", "std", "count"]).reset_index()
    grp["std"] = grp["std"].fillna(0.0)
    grp["count"] = grp["count"].clip(lower=1)
    grp["half_ci"] = ci_z * grp["std"] / np.sqrt(grp["count"])
    grp["lower"] = grp["mean"] - grp["half_ci"]
    grp["upper"] = grp["mean"] + grp["half_ci"]
    return grp


def _smooth_xy(x: np.ndarray, y: np.ndarray, *, logx: bool = False, n_dense: int = 200) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size <= 2:
        return x, y
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    xu, idx = np.unique(x, return_index=True)
    yu = y[idx]
    if xu.size <= 2:
        return xu, yu
    if logx:
        xx = np.log10(xu)
        dense = np.linspace(xx.min(), xx.max(), n_dense)
    else:
        xx = xu
        dense = np.linspace(xx.min(), xx.max(), n_dense)
    if PchipInterpolator is not None:
        interp = PchipInterpolator(xx, yu)
        yd = interp(dense)
    else:  # pragma: no cover
        yd = np.interp(dense, xx, yu)
    xd = 10 ** dense if logx else dense
    return xd, yd


def _plot_curve_with_ci(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str = "mean",
    lower_col: str = "lower",
    upper_col: str = "upper",
    label: str,
    semilogy: bool = False,
    logx: bool = False,
    marker: str = "o",
) -> None:
    x = df[x_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    plot_fun = ax.semilogy if semilogy else ax.plot
    xd, yd = _smooth_xy(x, y, logx=logx)
    line = plot_fun(xd, yd, linewidth=2.2, label=label)[0]
    color = line.get_color()
    plot_fun(x, y, linestyle="", marker=marker, markersize=5.5, color=color)
    if lower_col in df.columns and upper_col in df.columns:
        lo = df[lower_col].to_numpy(dtype=float)[order]
        hi = df[upper_col].to_numpy(dtype=float)[order]
        xdl, lod = _smooth_xy(x, lo, logx=logx)
        xdu, hid = _smooth_xy(x, hi, logx=logx)
        if semilogy:
            lod = np.maximum(lod, 1.0e-6)
            hid = np.maximum(hid, 1.0e-6)
        ax.fill_between(xdl, lod, hid, color=color, alpha=0.12, linewidth=0)


def _apply_common_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "lines.linewidth": 2.0,
            "axes.grid": True,
            "grid.linestyle": ":",
            "grid.alpha": 0.35,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _panel_title(ax: plt.Axes, tag: str, text: str) -> None:
    ax.set_title(f"({tag}) {text}", loc="left", fontweight="bold")


def _write_manifest(out_dir: Path, lines: list[str]) -> None:
    text = "\n".join(lines).rstrip() + "\n"
    (out_dir / "00_Publication_Figure_Guide.txt").write_text(text, encoding="utf-8")


def _select_publication_mcs(fer_df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    sub = _filter_algos(fer_df, FER_ALGOS)
    rows: list[dict[str, float | int]] = []
    for (m, r), df_mcs in sub.groupby(["modulation_order", "code_rate"]):
        metric = pd.to_numeric(df_mcs["fer"], errors="coerce").to_numpy(dtype=float)
        metric = metric[np.isfinite(metric)]
        if metric.size < 8:
            continue
        metric = np.clip(metric, 1.0e-6, 1.0 - 1.0e-6)
        frac_transition = float(np.mean((metric >= 1.0e-4) & (metric <= 5.0e-1)))
        spread = float(np.quantile(np.log10(metric), 0.9) - np.quantile(np.log10(metric), 0.1))
        saturation = float(np.mean(metric <= 1.0e-6) + np.mean(metric >= 0.999))
        score = spread + 1.4 * frac_transition - 1.2 * saturation
        rows.append({"modulation_order": int(m), "code_rate": float(r), "score": score})
    if not rows:
        raise ValueError("Could not select an informative FER MCS.")
    stats = pd.DataFrame(rows).sort_values(["score", "modulation_order", "code_rate"], ascending=[False, True, True])
    best = stats.iloc[0]
    m = int(best["modulation_order"])
    r = float(best["code_rate"])
    chosen = sub[(sub["modulation_order"] == m) & (sub["code_rate"] == r)].copy()
    label = f"{MCS_LABELS.get(m, f'{m}-bit')} (R={r:.2f})"
    return chosen, label


def _xlim_from_transition(fer_df: pd.DataFrame) -> tuple[float, float]:
    work = fer_df.copy()
    work["fer"] = pd.to_numeric(work["fer"], errors="coerce")
    trans = work[(work["fer"] >= 1.0e-4) & (work["fer"] <= 8.0e-1)]
    x_all = sorted(work["sweep_value"].unique())
    if trans.empty:
        return float(min(x_all)), float(max(x_all))
    lo = float(trans["sweep_value"].min()) - 2.5
    hi = float(trans["sweep_value"].max()) + 2.5
    lo = max(lo, float(min(x_all)))
    hi = min(hi, float(max(x_all)))
    if hi - lo < 7.5:
        hi = min(float(max(x_all)), lo + 7.5)
    return lo, hi


def _build_main_tradeoff(eval_summary: pd.DataFrame, out_dir: Path) -> None:
    df = _filter_algos(eval_summary, MAIN_ALGOS)
    df = df[df["sweep_value"] >= 0].copy()
    wsr = _aggregate_mean_ci(df, ["algorithm", "sweep_value"], "weighted_sum_rate_bps_per_hz")
    prot = _aggregate_mean_ci(df, ["algorithm", "sweep_value"], "protection_satisfaction")
    prot["lower"] = prot["lower"].clip(0.0, 1.0)
    prot["upper"] = prot["upper"].clip(0.0, 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0), sharex=True)
    for algo in _ordered_unique(df["algorithm"]):
        _plot_curve_with_ci(axes[0], wsr[wsr["algorithm"] == algo], x_col="sweep_value", label=_label(algo))
        _plot_curve_with_ci(axes[1], prot[prot["algorithm"] == algo], x_col="sweep_value", label=_label(algo))
    axes[0].set_xlim(0, 25)
    axes[0].set_xlabel("SNR [dB]")
    axes[1].set_xlabel("SNR [dB]")
    axes[0].set_ylabel("Weighted sum rate [bit/s/Hz]")
    axes[1].set_ylabel("Protection satisfaction")
    axes[1].set_ylim(-0.02, 1.02)
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    _panel_title(axes[0], "a", "Main weighted-sum-rate comparison")
    _panel_title(axes[1], "b", "Fixed-service protection comparison")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.10), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, out_dir, "Fig01_Main_SNR_Tradeoff")


def _build_unfolding_ablation(eval_summary: pd.DataFrame, out_dir: Path) -> None:
    df = _filter_algos(eval_summary, ABLATION_ALGOS)
    df = df[df["sweep_value"] >= 0].copy()
    wsr = _aggregate_mean_ci(df, ["algorithm", "sweep_value"], "weighted_sum_rate_bps_per_hz")
    run = _aggregate_mean_ci(df, ["algorithm", "sweep_value"], "runtime_sec")
    prot25 = (
        df[df["sweep_value"] == 25]
        .groupby("algorithm")["protection_satisfaction"]
        .mean()
        .reindex(ABLATION_ALGOS)
        .dropna()
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0), sharex=True)
    for algo in _ordered_unique(df["algorithm"]):
        _plot_curve_with_ci(axes[0], wsr[wsr["algorithm"] == algo], x_col="sweep_value", label=_label(algo))
        _plot_curve_with_ci(axes[1], run[run["algorithm"] == algo], x_col="sweep_value", label=_label(algo))
    axes[0].set_xlim(0, 25)
    axes[0].set_xlabel("SNR [dB]")
    axes[1].set_xlabel("SNR [dB]")
    axes[0].set_ylabel("Weighted sum rate [bit/s/Hz]")
    axes[1].set_ylabel("Runtime per realization [s]")
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    _panel_title(axes[0], "a", "Unfolding ablation: quality")
    _panel_title(axes[1], "b", "Unfolding ablation: runtime")
    text_lines = [f"Protection @ 25 dB:"] + [f"{_label(a)} = {prot25[a]:.3f}" for a in prot25.index]
    axes[1].text(
        0.98,
        0.02,
        "\n".join(text_lines),
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="0.7"),
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.10), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, out_dir, "Fig02_Unfolding_Ablation")


def _build_convergence(history_mean: pd.DataFrame, out_dir: Path, select_snr: float = 15.0) -> None:
    df = _filter_algos(history_mean, CONV_ALGOS)
    df = df[np.isclose(df["sweep_value"], float(select_snr))].copy()
    if df.empty:
        raise ValueError(f"No convergence history for SNR={select_snr}")

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0))
    for algo in _ordered_unique(df["algorithm"]):
        sub = df[df["algorithm"] == algo].sort_values("iteration")
        axes[0].plot(sub["iteration"], sub["weighted_sum_rate"], marker="o", markersize=4.5, linewidth=2.2, label=_label(algo))
        axes[1].semilogy(sub["iteration"], np.maximum(sub["w_delta"], 1.0e-6), marker="o", markersize=4.5, linewidth=2.2, label=_label(algo))
    axes[0].set_xlabel("Iteration / unfolded layer")
    axes[1].set_xlabel("Iteration / unfolded layer")
    axes[0].set_ylabel("Weighted sum rate [bit/s/Hz]")
    axes[1].set_ylabel(r"Beam update norm $\|W^{(\ell)}-W^{(\ell-1)}\|$")
    _panel_title(axes[0], "a", f"Convergence of weighted sum rate at {int(select_snr)} dB")
    _panel_title(axes[1], "b", f"Convergence of beam update norm at {int(select_snr)} dB")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.10), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, out_dir, "Fig03_Convergence_Quality")


def _build_scaling(scaling_summary: pd.DataFrame, out_dir: Path) -> None:
    df = _filter_algos(scaling_summary, SCALING_ALGOS)
    wsr = _aggregate_mean_ci(df, ["algorithm", "num_bs_ant", "num_ut_per_sector"], "weighted_sum_rate_bps_per_hz")

    ks = sorted(df["num_ut_per_sector"].unique())
    fig, axes = plt.subplots(1, len(ks), figsize=(12.0, 3.8), sharey=True)
    if len(ks) == 1:
        axes = [axes]
    for ax, k in zip(axes, ks):
        subk = wsr[wsr["num_ut_per_sector"] == k]
        for algo in SCALING_ALGOS:
            sub = subk[subk["algorithm"] == algo].sort_values("num_bs_ant")
            _plot_curve_with_ci(ax, sub, x_col="num_bs_ant", label=_label(algo))
        ax.set_xlabel("Number of BS antennas $M$")
        ax.set_xticks(sorted(subk["num_bs_ant"].unique()))
        _panel_title(ax, chr(97 + len([kk for kk in ks if kk < k])), f"$K={int(k)}$ users / sector")
    axes[0].set_ylabel("Weighted sum rate [bit/s/Hz]")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.12), frameon=False)
    fig.text(0.5, 1.02, "Scaling at SNR = 5 dB", ha="center", va="bottom", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, out_dir, "Fig04_Large_Array_Scaling")


def _build_selectivity(gap_df: pd.DataFrame, out_dir: Path) -> None:
    df = _filter_algos(gap_df, SELECTIVITY_ALGOS)
    agg = _aggregate_mean_ci(df, ["algorithm", "tau_rms_ns"], "rate_gap_bps_per_hz")

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for algo in _ordered_unique(df["algorithm"]):
        sub = agg[agg["algorithm"] == algo].sort_values("tau_rms_ns")
        _plot_curve_with_ci(ax, sub, x_col="tau_rms_ns", label=_label(algo), logx=True)
    ax.set_xscale("log")
    ax.set_xlabel(r"RMS delay spread $\tau_{\mathrm{rms}}$ [ns]")
    ax.set_ylabel("Flat-to-selective WSR gap [bit/s/Hz]")
    ax.set_xticks([10, 20, 50, 100, 200, 400, 800])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    _panel_title(ax, "a", "Frequency-selectivity robustness")
    ax.legend(ncol=1, frameon=False)
    fig.tight_layout()
    _save(fig, out_dir, "Fig05_Frequency_Selectivity_Robustness")


def _build_selected_fer(fer_df: pd.DataFrame, out_dir: Path) -> tuple[str, tuple[float, float]]:
    chosen, mcs_label = _select_publication_mcs(fer_df)
    xlo, xhi = _xlim_from_transition(chosen)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for algo in _ordered_unique(chosen["algorithm"]):
        sub = chosen[chosen["algorithm"] == algo].sort_values("sweep_value")
        x = sub["sweep_value"].to_numpy(dtype=float)
        y = np.maximum(sub["fer"].to_numpy(dtype=float), 1.0e-6)
        xd, yd = _smooth_xy(x, np.log10(y), logx=False)
        line = ax.semilogy(xd, np.maximum(10 ** yd, 1.0e-6), linewidth=2.2, label=_label(algo))[0]
        ax.semilogy(x, y, linestyle="", marker="o", markersize=5.5, color=line.get_color())
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(1.0e-4, 1.0)
    ax.set_xlabel("SNR [dB]")
    ax.set_ylabel("Frame error rate")
    _panel_title(ax, "a", f"Selected FER case: {mcs_label}")
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, out_dir, "Fig06_Selected_FER")
    return mcs_label, (xlo, xhi)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create a compact publication-level TWC figure set.")
    p.add_argument("--output-root", type=str, default="results_twc")
    p.add_argument("--eval-dir", type=str, default=None)
    p.add_argument("--scaling-dir", type=str, default=None)
    p.add_argument("--selectivity-dir", type=str, default=None)
    p.add_argument("--out-dir", type=str, default=None)
    return p.parse_args()


def main() -> None:
    _apply_common_style()
    args = parse_args()
    root = Path(args.output_root)
    eval_dir = Path(args.eval_dir) if args.eval_dir else _latest_prefixed_dir(root, "eval_all_")
    scaling_dir = Path(args.scaling_dir) if args.scaling_dir else _latest_prefixed_dir(root, "scaling_")
    selectivity_dir = Path(args.selectivity_dir) if args.selectivity_dir else _latest_prefixed_dir(root, "selectivity_")
    out_dir = _ensure_dir(Path(args.out_dir) if args.out_dir else root / "Publication_Level_Final")

    eval_summary = pd.read_csv(eval_dir / "summary.csv")
    history_mean = pd.read_csv(eval_dir / "history_mean.csv")
    fer_df = pd.read_csv(eval_dir / "fer.csv")
    scaling_summary = pd.read_csv(scaling_dir / "summary.csv")
    gap_df = pd.read_csv(selectivity_dir / "gap.csv")

    _build_main_tradeoff(eval_summary, out_dir)
    _build_unfolding_ablation(eval_summary, out_dir)
    _build_convergence(history_mean, out_dir, select_snr=15.0)
    _build_scaling(scaling_summary, out_dir)
    _build_selectivity(gap_df, out_dir)
    mcs_label, fer_xlim = _build_selected_fer(fer_df, out_dir)

    manifest_lines = [
        "Publication-Level Final Figure Set",
        f"Created: {datetime.now().isoformat(timespec='seconds')}",
        f"Eval directory: {eval_dir}",
        f"Scaling directory: {scaling_dir}",
        f"Selectivity directory: {selectivity_dir}",
        "",
        "Files:",
        "- Fig01_Main_SNR_Tradeoff.(png|pdf): main weighted-sum-rate and protection comparison.",
        "- Fig02_Unfolding_Ablation.(png|pdf): DU ablation for quality/runtime.",
        "- Fig03_Convergence_Quality.(png|pdf): convergence at 15 dB.",
        "- Fig04_Large_Array_Scaling.(png|pdf): scaling across M for K=4,6,8 at 5 dB.",
        "- Fig05_Frequency_Selectivity_Robustness.(png|pdf): flat-to-selective WSR gap.",
        f"- Fig06_Selected_FER.(png|pdf): selected FER case {mcs_label}, x-range {fer_xlim[0]:.1f} to {fer_xlim[1]:.1f} dB.",
        "",
        "Note: smooth lines are shape-preserving interpolations through the simulated means; circle markers show the actual simulated operating points.",
        "Confidence bands show approximate 95% normal-theory intervals computed from the saved raw batches where available.",
    ]
    _write_manifest(out_dir, manifest_lines)
    print(f"Saved publication-level figures to: {out_dir}")


if __name__ == "__main__":
    main()
