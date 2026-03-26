from __future__ import annotations

import argparse
from pathlib import Path

import _repo_bootstrap as _rb

_rb.bootstrap()

from fr3_twc.config import get_twc_paths, load_twc_config
from fr3_twc.fer import validate_sionna_fer_grid
from fr3_twc.reporting import load_models


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


def _latest_prefixed_dir(root: Path, prefix: str) -> Path | None:
    dirs = sorted([p for p in root.glob(f"{prefix}*") if p.is_dir()])
    return dirs[-1] if dirs else None


def _require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")


def _check_checkpoints(cfg) -> None:
    twc_paths = get_twc_paths(cfg)
    models = load_models(twc_paths.checkpoint_root, names=["soft", "cognitive"])
    missing = [name for name in ["soft", "cognitive"] if name not in models]
    if missing:
        raise FileNotFoundError(f"Missing checkpoints in {twc_paths.checkpoint_root}: {missing}")


def _check_eval_config(cfg) -> None:
    fer_cfg = cfg.raw.get("twc", {}).get("fer", {}) or {}
    require_sionna = bool(fer_cfg.get("require_sionna", False))
    allow_fallback = bool(fer_cfg.get("allow_fallback", not require_sionna))
    if require_sionna or not allow_fallback:
        validate_sionna_fer_grid(
            modulation_orders=list(fer_cfg.get("modulation_orders", [2, 4])),
            code_rates=list(fer_cfg.get("code_rates", [0.3, 0.5])),
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
        cfg = load_twc_config(args.config, overrides=args.overrides)
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
