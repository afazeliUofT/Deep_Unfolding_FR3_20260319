#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

import yaml


def _prepend(path: Path) -> None:
    s = str(path)
    if path.exists() and s not in sys.path:
        sys.path.insert(0, s)


def _add_repo_paths() -> Path:
    root = Path(__file__).resolve().parents[1]
    for path in (root, root / "src", root / "scripts"):
        _prepend(path)
    return root


REPO_ROOT = _add_repo_paths()

from fr3_twc.checkpoints import (  # noqa: E402
    checkpoint_recovery_report,
    checkpoint_roots_from_cfg,
    ensure_checkpoint_files,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Restore or validate TWC checkpoints locally before submission")
    p.add_argument("--config", type=str, default="configs/twc_eval_final.yaml")
    p.add_argument("--set", dest="overrides", action="append", default=None)
    return p.parse_args()


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, val in update.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} did not parse to a dict")
    return data


def _load_config(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    cfg = _load_yaml(p)
    base_cfg = cfg.pop("base_config", None)
    if base_cfg is None:
        return cfg
    base_path = Path(str(base_cfg))
    if not base_path.is_absolute():
        base_path = (p.parent / base_path).resolve()
    return _deep_merge(_load_config(base_path), cfg)


def _parse_override(override: str) -> tuple[list[str], Any]:
    if "=" not in str(override):
        raise ValueError(f"Override must be KEY=VALUE, got: {override}")
    lhs, rhs = str(override).split("=", 1)
    keys = [k.strip() for k in lhs.split(".") if k.strip()]
    if not keys:
        raise ValueError(f"Invalid override key path: {override}")
    return keys, yaml.safe_load(rhs)


def _deep_set(d: dict[str, Any], keys: list[str], value: Any) -> None:
    cur: dict[str, Any] = d
    for key in keys[:-1]:
        nxt = cur.get(key)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[key] = nxt
        cur = nxt
    cur[keys[-1]] = value


def _apply_overrides(cfg: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    out = copy.deepcopy(cfg)
    for item in overrides or []:
        keys, value = _parse_override(item)
        _deep_set(out, keys, value)
    return out


def _print_report(stage: str, cfg: dict[str, Any]) -> None:
    output_root, ckpt_root = checkpoint_roots_from_cfg(cfg, repo_root=REPO_ROOT)
    reports = checkpoint_recovery_report(
        checkpoint_root=ckpt_root,
        names=["soft", "cognitive"],
        output_root=output_root,
        repo_root=REPO_ROOT,
    )
    for rep in reports:
        train_list = ",".join(str(p) for p in rep.valid_train_candidates[:3]) if rep.valid_train_candidates else "-"
        git_present = ",".join(rep.git_head_candidates_present) if rep.git_head_candidates_present else "-"
        git_candidates = ",".join(rep.git_candidates) if rep.git_candidates else "-"
        print(
            f"CHECKPOINT_STATUS stage={stage} name={rep.name} dest={rep.destination} "
            f"dest_exists={int(rep.destination_exists)} dest_valid={int(rep.destination_valid)} "
            f"valid_train_candidates={len(rep.valid_train_candidates)} sample_train_candidates={train_list} "
            f"git_available={int(rep.git_available)} git_repo_ok={int(rep.git_repo_ok)} "
            f"git_head_present={git_present} git_candidates={git_candidates}"
        )


def main() -> int:
    args = parse_args()
    cfg = _apply_overrides(_load_config(args.config), args.overrides)
    output_root, ckpt_root = checkpoint_roots_from_cfg(cfg, repo_root=REPO_ROOT)

    _print_report("before_repair", cfg)
    repairs = ensure_checkpoint_files(
        checkpoint_root=ckpt_root,
        names=["soft", "cognitive"],
        output_root=output_root,
        repo_root=REPO_ROOT,
        verbose=True,
    )
    for repair in repairs:
        print(
            f"CHECKPOINT_RESTORED name={repair.name} source={repair.source} destination={repair.destination}"
        )
    _print_report("after_repair", cfg)

    reports = checkpoint_recovery_report(
        checkpoint_root=ckpt_root,
        names=["soft", "cognitive"],
        output_root=output_root,
        repo_root=REPO_ROOT,
    )
    missing = [rep.name for rep in reports if not rep.destination_valid]
    if missing:
        print(
            "CHECKPOINT_RESTORE_FAILED "
            f"missing={','.join(missing)} checkpoint_root={ckpt_root} output_root={output_root}"
        )
        return 1
    print(f"CHECKPOINT_RESTORE_OK checkpoint_root={ckpt_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
