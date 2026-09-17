"""Run bounded safe-resumption experiments in both directions over five seeds."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from experiments.run_cross_netflow import DATASET_KEYS
from experiments.run_safe_resumption import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/safe_resumption.yaml")
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
        selected = result["safe_resumption"]["selected_policy"]
        print(
            f"[{index}/{len(jobs)}] {result['experiment_id']} "
            f"decision={selected['decision']} action={selected['action']} "
            f"budget={selected['label_budget']} unsafe={selected['unsafe_resumption']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
