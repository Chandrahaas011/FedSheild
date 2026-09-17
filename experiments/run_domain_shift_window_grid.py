"""Run the frozen source-only sentinel window-size sensitivity grid."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml

from experiments.run_cross_netflow import DATASET_KEYS
from experiments.run_domain_shift_validation import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/domain_shift_window_sensitivity.yaml"
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    base = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    settings = base["domain_shift_validation"]
    jobs = [
        (window_size, source, target, seed)
        for window_size in settings["window_sizes"]
        for source, target in (DATASET_KEYS, tuple(reversed(DATASET_KEYS)))
        for seed in base["seeds"]
    ]
    for index, (window_size, source, target, seed) in enumerate(jobs, 1):
        config = copy.deepcopy(base)
        config["domain_shift_validation"]["window_size"] = int(window_size)
        result = run(config, source, target, seed, args.force)
        primary = result["domain_shift_validation"]["methods"]["jsd_mean"]
        print(
            f"[{index}/{len(jobs)}] {result['experiment_id']} "
            f"source-FAR={primary['source_audit']['alarm_rate']:.3f} "
            f"25%-target={primary['mixtures']['0.25']['alarm_rate']:.3f} "
            f"target={primary['target_full']['alarm_rate']:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
