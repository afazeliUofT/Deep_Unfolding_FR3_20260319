from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Sequence

import json
import math
import os
import random
import time

import numpy as np
import pandas as pd
import yaml

from fr3_sim.seeding import set_global_seed


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    set_global_seed(seed)


def deep_get(d: Mapping[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, Mapping) or k not in cur:
            return default
        cur = cur[k]
    return cur


def save_yaml(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(_to_builtin(obj), f, sort_keys=False)


def save_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(_to_builtin(obj), f, indent=2)


def save_csv(path: str | Path, rows: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(list(rows))
    df.to_csv(p, index=False)
    return df


def _to_builtin(obj: Any) -> Any:
    if is_dataclass(obj):
        return _to_builtin(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    try:
        import tensorflow as tf  # type: ignore

        if isinstance(obj, (tf.Tensor, tf.Variable)):
            return obj.numpy().tolist()
    except Exception:
        pass
    return obj


def safe_ratio(num: float, den: float, eps: float = 1e-12) -> float:
    return float(num) / float(max(abs(den), eps))


def db10(x: np.ndarray | float, eps: float = 1e-12) -> np.ndarray | float:
    arr = np.asarray(x)
    out = 10.0 * np.log10(np.maximum(arr, eps))
    return float(out) if np.isscalar(x) else out


def human_seconds(sec: float) -> str:
    sec = float(sec)
    if sec < 60.0:
        return f"{sec:.1f}s"
    if sec < 3600.0:
        return f"{sec/60.0:.1f}m"
    return f"{sec/3600.0:.2f}h"


def flatten_history_dict(history: Mapping[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in history.items():
        name = f"{prefix}{k}"
        if isinstance(v, Mapping):
            out.update(flatten_history_dict(v, prefix=f"{name}."))
        else:
            out[name] = _to_builtin(v)
    return out
