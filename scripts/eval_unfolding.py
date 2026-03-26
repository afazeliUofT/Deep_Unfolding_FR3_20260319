from __future__ import annotations

import argparse

import pandas as pd

import _repo_bootstrap as _rb

_rb.bootstrap()

from fr3_twc.config import get_twc_paths, load_twc_config
from fr3_twc.fer import fer_from_algorithm_summary, validate_sionna_fer_grid
from fr3_twc.pipeline import (
    default_baseline_algorithms,
    default_unfolded_algorithms,
    run_suite,
)
from fr3_twc.reporting import load_models, save_grouped_mean


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate baselines and trained unfolded models")
    p.add_argument("--config", type=str, default="configs/twc_base.yaml")
    p.add_argument("--suite-name", type=str, default="eval_all")
    p.add_argument("--set", dest="overrides", action="append", default=None)
    return p.parse_args()


def _prevalidate_eval(cfg) -> None:
    twc_paths = get_twc_paths(cfg)
    models = load_models(twc_paths.checkpoint_root, names=["soft", "cognitive"])
    missing = [n for n in ["soft", "cognitive"] if n not in models]
    if missing:
        raise FileNotFoundError(f"Missing checkpoints in {twc_paths.checkpoint_root}: {missing}")

    fer_cfg = cfg.raw.get("twc", {}).get("fer", {}) or {}
    require_sionna = bool(fer_cfg.get("require_sionna", False))
    allow_fallback = bool(fer_cfg.get("allow_fallback", not require_sionna))
    if require_sionna or not allow_fallback:
        validate_sionna_fer_grid(
            modulation_orders=list(fer_cfg.get("modulation_orders", [2, 4])),
            code_rates=list(fer_cfg.get("code_rates", [0.3, 0.5])),
            k_bits=int(fer_cfg.get("k_bits", 1024)),
        )


def main() -> None:
    args = parse_args()
    cfg = load_twc_config(args.config, overrides=args.overrides)
    _prevalidate_eval(cfg)

    twc_paths = get_twc_paths(cfg)
    models = load_models(twc_paths.checkpoint_root, names=["soft", "cognitive"])

    algs = default_baseline_algorithms() + default_unfolded_algorithms()
    art = run_suite(
        cfg=cfg,
        suite_name=args.suite_name,
        algorithms=algs,
        model_registry=models,
    )

    summary_mean = save_grouped_mean(
        art.summary_df,
        art.paths.root / "summary_mean.csv",
        group_cols=["algorithm", "sweep_value"],
    )
    if not art.history_df.empty:
        save_grouped_mean(
            art.history_df,
            art.paths.root / "history_mean.csv",
            group_cols=["algorithm", "sweep_value", "iteration"],
        )

    fer_cfg = cfg.raw.get("twc", {}).get("fer", {}) or {}
    require_sionna = bool(fer_cfg.get("require_sionna", False))
    allow_fallback = bool(fer_cfg.get("allow_fallback", not require_sionna))
    sinr_metric = str(fer_cfg.get("sinr_metric", "p50_sinr_db"))
    fallback_sinr_cols = list(
        fer_cfg.get(
            "fallback_sinr_cols",
            ["avg_sinr_db", "p05_sinr_db", "avg_user_rate_bps_per_hz"],
        )
    )
    fer_df = fer_from_algorithm_summary(
        summary_df=summary_mean,
        sinr_col=sinr_metric,
        fallback_sinr_cols=fallback_sinr_cols,
        modulation_orders=list(fer_cfg.get("modulation_orders", [2, 4])),
        code_rates=list(fer_cfg.get("code_rates", [0.3, 0.5])),
        k_bits=int(fer_cfg.get("k_bits", 1024)),
        num_frames_per_point=int(fer_cfg.get("num_frames_per_point", 128)),
        max_frame_errors=int(fer_cfg.get("max_frame_errors", 100)),
        decoder_iterations=int(fer_cfg.get("decoder_iterations", 20)),
        require_sionna=require_sionna,
        allow_fallback=allow_fallback,
    )
    fer_path = art.paths.root / "fer.csv"
    fer_df.to_csv(fer_path, index=False)

    if "used_sionna" in fer_df.columns:
        used = fer_df["used_sionna"].astype(bool)
        print(f"FER_STATUS used_sionna_all={bool(used.all())} used_sionna_any={bool(used.any())}")
        if not used.all():
            err_col = fer_df["sionna_error"] if "sionna_error" in fer_df.columns else pd.Series([], dtype=str)
            errors = sorted({str(x) for x in err_col.dropna().tolist() if str(x).strip()})
            if errors:
                print("FER_WARNING unique_errors:")
                for err in errors:
                    print(f" - {err}")
            if require_sionna:
                raise RuntimeError(
                    f"FER fallback detected even though require_sionna=True. See {fer_path}"
                )

    print(f"Saved evaluation suite to: {art.paths.root}")


if __name__ == "__main__":
    main()
