from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Sequence

import pandas as pd

from .checkpoints import ensure_checkpoint_files
from .common import save_csv

if TYPE_CHECKING:  # pragma: no cover
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



def load_models(
    model_dir: str | Path,
    names: Sequence[str],
    *,
    output_root: str | Path = "results_twc",
    repo_root: str | Path | None = None,
) -> Dict[str, "UnfoldedWeightedWMMSE"]:
    repo_root = repo_root or os.environ.get("FR3_REPO_ROOT")
    ensure_checkpoint_files(
        checkpoint_root=model_dir,
        names=names,
        output_root=output_root,
        repo_root=repo_root,
        verbose=False,
    )

    # Lazy import so non-training/reporting utilities do not import TensorFlow unnecessarily.
    from .unfolding import UnfoldedWeightedWMMSE

    out: Dict[str, UnfoldedWeightedWMMSE] = {}
    model_dir = Path(model_dir)
    for name in names:
        path = model_dir / f"{name}.npz"
        if path.exists():
            out[str(name)] = UnfoldedWeightedWMMSE.load_npz(path)
    return out



def save_grouped_mean(df: pd.DataFrame, out_path: str | Path, group_cols: Sequence[str]) -> pd.DataFrame:
    agg = aggregate_mean(df, group_cols=group_cols)
    save_csv(Path(out_path), agg)
    return agg
