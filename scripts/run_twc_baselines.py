from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
from pathlib import Path

from fr3_twc.config import load_twc_config
from fr3_twc.pipeline import default_baseline_algorithms, run_suite
from fr3_twc.reporting import save_grouped_mean


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run TWC PF/FS benchmark suite")
    p.add_argument("--config", type=str, default="configs/twc_base.yaml")
    p.add_argument("--suite-name", type=str, default="baselines")
    p.add_argument("--set", dest="overrides", action="append", default=None)
    return p.parse_args()



def main() -> None:
    args = parse_args()
    cfg = load_twc_config(args.config, overrides=args.overrides)
    art = run_suite(cfg=cfg, suite_name=args.suite_name, algorithms=default_baseline_algorithms())

    save_grouped_mean(art.summary_df, art.paths.root / "summary_mean.csv", group_cols=["algorithm", "sweep_value"])
    if not art.history_df.empty:
        save_grouped_mean(
            art.history_df,
            art.paths.root / "history_mean.csv",
            group_cols=["algorithm", "sweep_value", "iteration"],
        )
    print(f"Saved baseline suite to: {art.paths.root}")


if __name__ == "__main__":
    main()
