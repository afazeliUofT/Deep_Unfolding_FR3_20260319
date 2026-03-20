from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence

import pandas as pd

from .common import ensure_dir, save_csv
from .unfolding import UnfoldedWeightedWMMSE


def aggregate_mean(df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    num_cols = [c for c in df.columns if c not in group_cols]
    out = df.groupby(list(group_cols), dropna=False)[num_cols].mean(numeric_only=True).reset_index()
    return out


def latest_prefixed_dir(root: str | Path, prefix: str) -> Optional[Path]:
    root = Path(root)
    cands = [p for p in root.glob(f"{prefix}*") if p.is_dir()]
    if not cands:
        return None
    return sorted(cands)[-1]


def load_models(model_dir: str | Path, names: Sequence[str]) -> Dict[str, UnfoldedWeightedWMMSE]:
    out: Dict[str, UnfoldedWeightedWMMSE] = {}
    model_dir = Path(model_dir)
    for name in names:
        path = model_dir / f"{name}.npz"
        if path.exists():
            out[str(name)] = UnfoldedWeightedWMMSE.load_npz(path)
    return out


def save_grouped_mean(df: pd.DataFrame, out_path: str | Path, group_cols: Sequence[str]) -> pd.DataFrame:
    agg = aggregate_mean(df, group_cols=group_cols)
    agg.to_csv(Path(out_path), index=False)
    return agg
