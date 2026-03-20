from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from fr3_sim.topology import FixedServiceLocations, TopologyData


def _prep_out(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def plot_topology(
    topo: TopologyData,
    fs_loc: Optional[FixedServiceLocations],
    out_path: str | Path,
    title: str = "Reference geometry",
    dpi: int = 220,
) -> Path:
    p = _prep_out(out_path)
    bs = tf.cast(topo.bs_loc[0], tf.float32).numpy()
    ut = tf.cast(topo.ut_loc[0], tf.float32).numpy()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(bs[:, 0], bs[:, 1], marker="^", label="BS")
    ax.scatter(ut[:, 0], ut[:, 1], s=8, alpha=0.6, label="UE")
    if fs_loc is not None:
        fs = tf.cast(fs_loc.fs_loc[0], tf.float32).numpy()
        ax.scatter(fs[:, 0], fs[:, 1], marker="x", s=50, label="FS RX")
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True)
    ax.legend()
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return p


def plot_metric_vs_sweep(
    df: pd.DataFrame,
    metric: str,
    out_path: str | Path,
    sweep_col: str = "sweep_value",
    algo_col: str = "algorithm",
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    dpi: int = 220,
) -> Path:
    p = _prep_out(out_path)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for algo, sub in df.groupby(algo_col):
        sub = sub.sort_values(sweep_col)
        ax.plot(sub[sweep_col].to_numpy(), sub[metric].to_numpy(), marker="o", label=str(algo))
    ax.set_title(title or metric)
    ax.set_xlabel(sweep_col)
    ax.set_ylabel(ylabel or metric)
    ax.grid(True)
    ax.legend()
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return p


def plot_convergence(
    history_df: pd.DataFrame,
    metric: str,
    out_path: str | Path,
    algo_col: str = "algorithm",
    iter_col: str = "iteration",
    sweep_col: str = "sweep_value",
    select_sweep: Optional[float] = None,
    dpi: int = 220,
) -> Path:
    p = _prep_out(out_path)
    df = history_df.copy()
    if select_sweep is None and sweep_col in df.columns:
        select_sweep = sorted(df[sweep_col].unique())[len(df[sweep_col].unique()) // 2]
    if sweep_col in df.columns and select_sweep is not None:
        df = df[df[sweep_col] == select_sweep]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for algo, sub in df.groupby(algo_col):
        sub = sub.sort_values(iter_col)
        ax.plot(sub[iter_col].to_numpy(), sub[metric].to_numpy(), marker="o", label=str(algo))
    ax.set_title(f"Convergence: {metric}")
    ax.set_xlabel("iteration / layer")
    ax.set_ylabel(metric)
    ax.grid(True)
    ax.legend()
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return p


def plot_fer(
    fer_df: pd.DataFrame,
    out_path: str | Path,
    sweep_col: str = "sweep_value",
    algo_col: str = "algorithm",
    title: str = "FER",
    dpi: int = 220,
) -> Path:
    p = _prep_out(out_path)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for (algo, m, r), sub in fer_df.groupby([algo_col, "modulation_order", "code_rate"]):
        sub = sub.sort_values(sweep_col)
        ax.semilogy(sub[sweep_col].to_numpy(), np.maximum(sub["fer"].to_numpy(), 1e-6), marker="o", label=f"{algo} | {m}-bit | R={r:.2f}")
    ax.set_title(title)
    ax.set_xlabel(sweep_col)
    ax.set_ylabel("FER")
    ax.grid(True, which="both")
    ax.legend(fontsize=8)
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return p


def plot_scaling_heatmap(
    df: pd.DataFrame,
    metric: str,
    out_path: str | Path,
    x_col: str = "num_bs_ant",
    y_col: str = "num_ut_per_sector",
    dpi: int = 220,
) -> Path:
    p = _prep_out(out_path)
    piv = df.pivot_table(index=y_col, columns=x_col, values=metric, aggfunc="mean")
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(piv.to_numpy(), aspect="auto", origin="lower")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(list(piv.columns))
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(list(piv.index))
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(metric)
    fig.colorbar(im, ax=ax)
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return p


def plot_selectivity_gap(
    df: pd.DataFrame,
    out_path: str | Path,
    x_col: str = "tau_rms_ns",
    y_col: str = "rate_gap_bps_per_hz",
    hue_col: str = "algorithm",
    dpi: int = 220,
) -> Path:
    p = _prep_out(out_path)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for algo, sub in df.groupby(hue_col):
        sub = sub.sort_values(x_col)
        ax.plot(sub[x_col].to_numpy(), sub[y_col].to_numpy(), marker="o", label=str(algo))
    ax.set_title("Flat-to-selective rate gap")
    ax.set_xlabel("RMS delay spread [ns]")
    ax.set_ylabel(y_col)
    ax.grid(True)
    ax.legend()
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return p
