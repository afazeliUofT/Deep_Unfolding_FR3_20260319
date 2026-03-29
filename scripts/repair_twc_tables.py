#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


_DROP_NUMERIC_COLS = {"batch_index", "seed", "scenario_seed"}


def _latest_prefixed_dir(root: Path, prefix: str) -> Path | None:
    cands = sorted([p for p in root.glob(f"{prefix}*") if p.is_dir()])
    return cands[-1] if cands else None


def _read_csv_if_valid(path: Path, required_cols: Iterable[str] | None = None) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty:
        return None
    if required_cols:
        req = set(required_cols)
        if not req.issubset(df.columns):
            return None
    if len(df.columns) == 1:
        only = str(df.columns[0])
        if only == "0" or only.startswith("Unnamed"):
            return None
    return df


def grouped_mean(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    keep_group_cols = [c for c in group_cols if c in df.columns]
    numeric_cols = []
    for c in df.columns:
        if c in keep_group_cols or c in _DROP_NUMERIC_COLS:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)
    if not keep_group_cols:
        out = df.copy()
    elif not numeric_cols:
        out = df[keep_group_cols].drop_duplicates().reset_index(drop=True)
    else:
        out = df.groupby(keep_group_cols, dropna=False, as_index=False)[numeric_cols].mean()
        out = out.sort_values(keep_group_cols).reset_index(drop=True)
    return out


def repair_table(mean_path: Path, raw_path: Path, group_cols: list[str], required_cols: list[str]) -> pd.DataFrame:
    raw_df = _read_csv_if_valid(raw_path, required_cols)
    if raw_df is None:
        raise FileNotFoundError(f"Missing or malformed raw table: {raw_path}")
    mean_df = grouped_mean(raw_df, group_cols)
    mean_path.parent.mkdir(parents=True, exist_ok=True)
    mean_df.to_csv(mean_path, index=False)
    return mean_df


def load_or_repair_table(
    preferred_path: Path,
    raw_path: Path,
    group_cols: list[str],
    required_cols: list[str],
) -> pd.DataFrame | None:
    df = _read_csv_if_valid(preferred_path, required_cols)
    if df is not None:
        return df
    if not raw_path.exists():
        return None
    return repair_table(preferred_path, raw_path, group_cols, required_cols)


def repair_existing_results(
    output_root: Path,
    eval_dir: Path | None = None,
    scaling_dir: Path | None = None,
    selectivity_dir: Path | None = None,
) -> list[Path]:
    repaired: list[Path] = []
    if eval_dir is None:
        eval_dir = _latest_prefixed_dir(output_root, "eval_all_")
    if scaling_dir is None:
        scaling_dir = _latest_prefixed_dir(output_root, "scaling_")
    if selectivity_dir is None:
        selectivity_dir = _latest_prefixed_dir(output_root, "selectivity_")

    specs: list[tuple[Path | None, str, str, list[str], list[str]]] = [
        (eval_dir, "summary_mean.csv", "summary.csv", ["algorithm", "sweep_value"], ["algorithm", "sweep_value"]),
        (eval_dir, "history_mean.csv", "history.csv", ["algorithm", "sweep_value", "iteration"], ["algorithm"]),
        (
            scaling_dir,
            "summary_mean.csv",
            "summary.csv",
            ["algorithm", "num_bs_ant", "num_ut_per_sector", "sweep_value"],
            ["algorithm", "num_bs_ant", "num_ut_per_sector"],
        ),
        (
            scaling_dir,
            "history_mean.csv",
            "history.csv",
            ["algorithm", "num_bs_ant", "num_ut_per_sector", "sweep_value", "iteration"],
            ["algorithm"],
        ),
        (
            selectivity_dir,
            "summary_mean.csv",
            "summary.csv",
            ["algorithm", "tau_rms_ns", "channel_modeling"],
            ["algorithm", "tau_rms_ns"],
        ),
        (
            selectivity_dir,
            "gap_mean.csv",
            "gap.csv",
            ["algorithm", "tau_rms_ns"],
            ["algorithm", "tau_rms_ns"],
        ),
    ]

    for base_dir, mean_name, raw_name, group_cols, required_cols in specs:
        if base_dir is None:
            continue
        raw_path = base_dir / raw_name
        if not raw_path.exists():
            continue
        mean_path = base_dir / mean_name
        repair_table(mean_path, raw_path, group_cols, required_cols)
        repaired.append(mean_path)
    return repaired


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Repair malformed grouped TWC CSV tables from raw result tables")
    p.add_argument("--output-root", type=str, default="results_twc")
    p.add_argument("--eval-dir", type=str, default=None)
    p.add_argument("--scaling-dir", type=str, default=None)
    p.add_argument("--selectivity-dir", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    repaired = repair_existing_results(
        output_root=output_root,
        eval_dir=Path(args.eval_dir) if args.eval_dir else None,
        scaling_dir=Path(args.scaling_dir) if args.scaling_dir else None,
        selectivity_dir=Path(args.selectivity_dir) if args.selectivity_dir else None,
    )
    if not repaired:
        print("No raw result tables found to repair.")
        return
    print("Repaired grouped tables:")
    for path in repaired:
        print(path)


if __name__ == "__main__":
    main()
