from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

import _repo_bootstrap as _rb

_rb.bootstrap()

from fr3_twc.common import ensure_dir, now_ts, save_yaml
from fr3_twc.config import get_twc_paths, load_twc_config
from fr3_twc.pipeline import default_baseline_algorithms, default_unfolded_algorithms, run_suite
from fr3_twc.reporting import load_models, save_grouped_mean


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run scaling suite over antennas and users")
    p.add_argument("--config", type=str, default="configs/twc_scaling.yaml")
    p.add_argument("--set", dest="overrides", action="append", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg0 = load_twc_config(args.config, overrides=args.overrides)
    twc = cfg0.raw.get("twc", {}) or {}
    sc = twc.get("scaling", {}) or {}

    twc_paths = get_twc_paths(cfg0)
    models = load_models(twc_paths.checkpoint_root, names=["soft", "cognitive"])
    missing = [n for n in ["soft", "cognitive"] if n not in models]
    if missing:
        raise FileNotFoundError(f"Missing checkpoints in {twc_paths.checkpoint_root}: {missing}")

    master_root = ensure_dir(Path(twc_paths.output_root) / f"scaling_{now_ts()}")
    sub_root = ensure_dir(master_root / "subruns")
    keep_subruns = bool(sc.get("keep_subruns", False))
    save_yaml(master_root / "base_config_resolved.yaml", cfg0.to_dict())

    algs = [
        spec
        for spec in (default_baseline_algorithms() + default_unfolded_algorithms())
        if spec.name in {"pf_fs_soft", "pf_fs_cognitive", "du_pf_soft", "du_pf_cognitive"}
    ]

    summary_rows = []
    history_rows = []

    for m in list(sc.get("num_bs_ant_values", [16, 32, 64])):
        for u in list(sc.get("num_ut_per_sector_values", [4, 6, 8])):
            ov = list(args.overrides or [])
            ov += [
                f"channel_model.num_bs_ant={int(m)}",
                f"topology.num_ut_per_sector={int(u)}",
                f"experiment.num_batches={int(sc.get('num_batches', 4))}",
                f"sweep.values=[{float(sc.get('sweep_value_db', 5.0))}]",
                f"twc.output_root={repr(str(sub_root))}",
            ]
            cfg = load_twc_config(args.config, overrides=ov)
            art = run_suite(
                cfg=cfg,
                suite_name=f"M{int(m)}_K{int(u)}",
                algorithms=algs,
                model_registry=models,
            )
            df_s = art.summary_df.copy()
            df_s["num_bs_ant"] = int(m)
            df_s["num_ut_per_sector"] = int(u)
            summary_rows.append(df_s)

            if not art.history_df.empty:
                df_h = art.history_df.copy()
                df_h["num_bs_ant"] = int(m)
                df_h["num_ut_per_sector"] = int(u)
                history_rows.append(df_h)

            print(f"finished scaling point M={int(m)} K={int(u)} -> {art.paths.root}")

    summary_df = pd.concat(summary_rows, ignore_index=True) if summary_rows else pd.DataFrame()
    history_df = pd.concat(history_rows, ignore_index=True) if history_rows else pd.DataFrame()

    summary_df.to_csv(master_root / "summary.csv", index=False)
    if not history_df.empty:
        history_df.to_csv(master_root / "history.csv", index=False)

    save_grouped_mean(
        summary_df,
        master_root / "summary_mean.csv",
        group_cols=["algorithm", "num_bs_ant", "num_ut_per_sector", "sweep_value"],
    )
    if not history_df.empty:
        save_grouped_mean(
            history_df,
            master_root / "history_mean.csv",
            group_cols=["algorithm", "num_bs_ant", "num_ut_per_sector", "sweep_value", "iteration"],
        )

    if not keep_subruns:
        shutil.rmtree(sub_root, ignore_errors=True)

    print(f"Saved scaling suite to: {master_root}")


if __name__ == "__main__":
    main()
