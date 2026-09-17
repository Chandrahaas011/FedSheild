"""Run prospective two-stage recovery under frozen review-label noise rates."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml

from experiments.run_cross_netflow import DATASET_KEYS
from experiments.run_safe_resumption import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/label_noise_resumption.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    base = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    jobs = [
        (rate, source, target, seed)
        for rate in base["review_label_noise_rates"]
        for source, target in (DATASET_KEYS, tuple(reversed(DATASET_KEYS)))
        for seed in base["seeds"]
    ]
    for index, (rate, source, target, seed) in enumerate(jobs, 1):
        config = copy.deepcopy(base)
        percentage = int(round(100 * float(rate)))
        prefix_base = base.get("experiment_prefix_base", "label_noise_resumption")
        config["experiment_prefix"] = f"{prefix_base}_{percentage:02d}"
        config["artifact_dir"] = str(Path(base["artifact_dir"]) / f"noise_{percentage:02d}")
        config["recovery"]["review_label_noise_rate"] = float(rate)
        result = run(config, source, target, seed, args.force)
        selected = result["safe_resumption"]["selected_policy"]
        print(
            f"[{index}/{len(jobs)}] {result['experiment_id']} "
            f"noise={rate:.0%} decision={selected['decision']} "
            f"budget={selected['label_budget']} unsafe={selected['unsafe_resumption']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
