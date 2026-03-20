from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, MutableMapping, Optional, Tuple

import copy
import yaml

from fr3_sim.config import ResolvedConfig, _derive, _parse_override, _validate_minimum  # type: ignore


@dataclass(frozen=True)
class TWCPaths:
    project_root: Path
    output_root: Path
    checkpoint_root: Path


def _deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in update.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)  # type: ignore[index]
        else:
            out[k] = copy.deepcopy(v)
    return out


def _deep_set(d: MutableMapping[str, Any], keys: List[str], value: Any) -> None:
    cur: MutableMapping[str, Any] = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], MutableMapping):
            cur[k] = {}
        cur = cur[k]  # type: ignore[assignment]
    cur[keys[-1]] = value


def load_twc_config(path: str | Path, overrides: Optional[List[str]] = None) -> ResolvedConfig:
    """Load a TWC config.

    Supports a lightweight inheritance mechanism via

    ``base_config: configs/default.yaml``

    and then recursively merges the local YAML on top of the base config.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"Config file {path} did not parse to a dict")

    cfg = copy.deepcopy(cfg)
    base_config = cfg.pop("base_config", None)
    if base_config is not None:
        base_path = (path.parent / str(base_config)).resolve() if not Path(str(base_config)).is_absolute() else Path(str(base_config))
        with base_path.open("r", encoding="utf-8") as f:
            base_raw = yaml.safe_load(f)
        if not isinstance(base_raw, dict):
            raise ValueError(f"Base config {base_path} did not parse to a dict")
        cfg = _deep_merge(base_raw, cfg)

    if overrides:
        for ov in overrides:
            keys, val = _parse_override(ov)
            _deep_set(cfg, keys, val)

    _validate_minimum(cfg)
    derived = _derive(cfg)
    derived["config_dir"] = str(path.parent.resolve())
    derived["config_path"] = str(path.resolve())

    twc = cfg.get("twc", {}) or {}
    output_root = Path(str(twc.get("output_root", "results_twc")))
    checkpoint_root = Path(str(twc.get("checkpoint_root", output_root / "checkpoints")))
    project_root = Path(str(twc.get("project_root", "."))).resolve()

    derived["twc_paths"] = {
        "project_root": str(project_root),
        "output_root": str(output_root),
        "checkpoint_root": str(checkpoint_root),
    }
    return ResolvedConfig(raw=cfg, derived=derived)


def get_twc_paths(cfg: ResolvedConfig) -> TWCPaths:
    p = cfg.derived.get("twc_paths", {}) or {}
    return TWCPaths(
        project_root=Path(str(p.get("project_root", "."))).resolve(),
        output_root=Path(str(p.get("output_root", "results_twc"))),
        checkpoint_root=Path(str(p.get("checkpoint_root", "results_twc/checkpoints"))),
    )
