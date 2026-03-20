from __future__ import annotations

import argparse
from pathlib import Path

from fr3_twc.config import get_twc_paths, load_twc_config
from fr3_twc.fer import fer_from_algorithm_summary
from fr3_twc.pipeline import default_baseline_algorithms, default_unfolded_algorithms, run_suite
from fr3_twc.reporting import load_models, save_grouped_mean


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate baselines and trained unfolded models")
    p.add_argument("--config", type=str, default="configs/twc_base.yaml")
    p.add_argument("--suite-name", type=str, default="eval_all")
    p.add_argument("--set", dest="overrides", action="append", default=None)
    return p.parse_args()



def main() -> None:
    args = parse_args()
    cfg = load_twc_config(args.config, overrides=args.overrides)
    twc_paths = get_twc_paths(cfg)
    models = load_models(twc_paths.checkpoint_root, names=["soft", "cognitive"])
    missing = [n for n in ["soft", "cognitive"] if n not in models]
    if missing:
        raise FileNotFoundError(f"Missing checkpoints in {twc_paths.checkpoint_root}: {missing}")

    algs = default_baseline_algorithms() + default_unfolded_algorithms()
    art = run_suite(cfg=cfg, suite_name=args.suite_name, algorithms=algs, model_registry=models)

    summary_mean = save_grouped_mean(art.summary_df, art.paths.root / "summary_mean.csv", group_cols=["algorithm", "sweep_value"])
    if not art.history_df.empty:
        save_grouped_mean(
            art.history_df,
            art.paths.root / "history_mean.csv",
            group_cols=["algorithm", "sweep_value", "iteration"],
        )

    fer_cfg = cfg.raw.get("twc", {}).get("fer", {}) or {}
    fer_df = fer_from_algorithm_summary(
        summary_mean,
        modulation_orders=list(fer_cfg.get("modulation_orders", [2, 4])),
        code_rates=list(fer_cfg.get("code_rates", [0.3, 0.5])),
        k_bits=int(fer_cfg.get("k_bits", 1024)),
        num_frames_per_point=int(fer_cfg.get("num_frames_per_point", 128)),
        max_frame_errors=int(fer_cfg.get("max_frame_errors", 100)),
    )
    fer_df.to_csv(art.paths.root / "fer.csv", index=False)
    print(f"Saved evaluation suite to: {art.paths.root}")


if __name__ == "__main__":
    main()
