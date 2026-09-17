"""Run the two-direction, five-seed domain-shift sentinel grid."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from experiments.run_cross_netflow import DATASET_KEYS
from experiments.run_domain_shift_sentinel import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/netflow_v2_cross.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    jobs = [
        (source, target, seed)
        for source, target in (DATASET_KEYS, tuple(reversed(DATASET_KEYS)))
        for seed in config["seeds"]
    ]
    for index, (source, target, seed) in enumerate(jobs, 1):
        result = run(config, source, target, seed, args.force)
        primary = result["domain_shift"]["primary"]
        print(
            f"[{index}/{len(jobs)}] {result['experiment_id']} "
            f"source-FAR={primary['source_audit']['alarm_rate']:.3f} "
            f"target={primary['mixed_target']['alarm_rate']:.3f} "
            f"benign-target={primary['benign_target_diagnostic']['alarm_rate']:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
