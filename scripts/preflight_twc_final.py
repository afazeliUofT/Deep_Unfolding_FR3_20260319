#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import math
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

from fr3_twc.checkpoints import checkpoint_roots_from_cfg, ensure_checkpoint_files  # noqa: E402

_MIN_SIONNA_EFFECTIVE_CODE_RATE = 1.0 / 5.0
_SIONNA_RATE_TOL = 1.0e-12



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fast preflight checks for final TWC runs")
    p.add_argument("--mode", choices=["eval", "scaling", "selectivity", "figures"], required=True)
    p.add_argument("--config", type=str, default="configs/twc_base.yaml")
    p.add_argument("--output-root", type=str, default="results_twc")
    p.add_argument("--eval-dir", type=str, default=None)
    p.add_argument("--scaling-dir", type=str, default=None)
    p.add_argument("--selectivity-dir", type=str, default=None)
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
    merged = _deep_merge(_load_config(base_path), cfg)
    return merged



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



def _latest_prefixed_dir(root: Path, prefix: str) -> Path | None:
    dirs = sorted([p for p in root.glob(f"{prefix}*") if p.is_dir()])
    return dirs[-1] if dirs else None



def _require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")



def _aligned_n_bits(k_bits: int, code_rate: float, modulation_order: int) -> int:
    n_bits = int(math.ceil(int(k_bits) / max(float(code_rate), 1.0e-6)))
    n_bits += (-n_bits) % int(modulation_order)
    return int(n_bits)



def _effective_code_rate(k_bits: int, *, n_bits: int) -> float:
    return float(int(k_bits) / max(int(n_bits), 1))



def _validate_sionna_fer_grid(*, modulation_orders: list[int], code_rates: list[float], k_bits: int) -> None:
    msgs: list[str] = []
    for m in modulation_orders:
        m_int = int(m)
        if m_int <= 0:
            msgs.append(f"invalid modulation order {m!r}")
            continue
        for r in code_rates:
            r_float = float(r)
            if not math.isfinite(r_float) or r_float <= 0.0:
                msgs.append(f"invalid code rate {r!r}")
                continue
            n_bits = _aligned_n_bits(k_bits=int(k_bits), code_rate=r_float, modulation_order=m_int)
            eff_rate = _effective_code_rate(int(k_bits), n_bits=n_bits)
            if eff_rate + _SIONNA_RATE_TOL < _MIN_SIONNA_EFFECTIVE_CODE_RATE:
                msgs.append(
                    "unsupported Sionna FER grid entry: "
                    f"modulation_order={m_int}, requested_code_rate={r_float:.6f}, "
                    f"k_bits={int(k_bits)}, aligned_n_bits={n_bits}, "
                    f"effective_code_rate={eff_rate:.6f} < 0.200000"
                )
    if msgs:
        raise ValueError("Invalid FER grid for strict Sionna evaluation: " + " | ".join(msgs))



def _check_checkpoints(cfg: dict[str, Any]) -> None:
    output_root, ckpt_root = checkpoint_roots_from_cfg(cfg, repo_root=REPO_ROOT)
    repairs = ensure_checkpoint_files(
        checkpoint_root=ckpt_root,
        names=["soft", "cognitive"],
        output_root=output_root,
        repo_root=REPO_ROOT,
        verbose=False,
    )
    for repair in repairs:
        print(
            f"CHECKPOINT_RESTORED name={repair.name} source={repair.source} "
            f"destination={repair.destination}"
        )
    for name in ["soft", "cognitive"]:
        _require_file(ckpt_root / f"{name}.npz", f"{name} checkpoint")



def _check_eval_config(cfg: dict[str, Any]) -> None:
    fer_cfg = cfg.get("twc", {}).get("fer", {}) or {}
    require_sionna = bool(fer_cfg.get("require_sionna", False))
    allow_fallback = bool(fer_cfg.get("allow_fallback", not require_sionna))
    if require_sionna or not allow_fallback:
        _validate_sionna_fer_grid(
            modulation_orders=[int(x) for x in list(fer_cfg.get("modulation_orders", [2, 4]))],
            code_rates=[float(x) for x in list(fer_cfg.get("code_rates", [0.3, 0.5]))],
            k_bits=int(fer_cfg.get("k_bits", 1024)),
        )



def _check_figures_inputs(root: Path, eval_dir: str | None, scaling_dir: str | None, selectivity_dir: str | None) -> None:
    eval_path = Path(eval_dir) if eval_dir else _latest_prefixed_dir(root, "eval_all_")
    scaling_path = Path(scaling_dir) if scaling_dir else _latest_prefixed_dir(root, "scaling_")
    selectivity_path = Path(selectivity_dir) if selectivity_dir else _latest_prefixed_dir(root, "selectivity_")

    if eval_path is None:
        raise FileNotFoundError(f"No eval_all_* directory found under {root}")
    if scaling_path is None:
        raise FileNotFoundError(f"No scaling_* directory found under {root}")
    if selectivity_path is None:
        raise FileNotFoundError(f"No selectivity_* directory found under {root}")

    _require_file(eval_path / "summary_mean.csv", "eval summary_mean.csv")
    _require_file(eval_path / "history_mean.csv", "eval history_mean.csv")
    _require_file(eval_path / "fer.csv", "eval fer.csv")
    _require_file(scaling_path / "summary_mean.csv", "scaling summary_mean.csv")
    _require_file(selectivity_path / "gap_mean.csv", "selectivity gap_mean.csv")



def main() -> None:
    args = parse_args()

    if args.mode in {"eval", "scaling", "selectivity"}:
        cfg = _apply_overrides(_load_config(args.config), args.overrides)
        _check_checkpoints(cfg)
        if args.mode == "eval":
            _check_eval_config(cfg)
        print(f"PREFLIGHT_OK mode={args.mode}")
        return

    _check_figures_inputs(
        root=Path(args.output_root),
        eval_dir=args.eval_dir,
        scaling_dir=args.scaling_dir,
        selectivity_dir=args.selectivity_dir,
    )
    print("PREFLIGHT_OK mode=figures")


if __name__ == "__main__":
    main()
