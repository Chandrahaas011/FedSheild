"""Run frozen class-conditional review-label-noise recovery experiments."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml

from experiments.run_cross_netflow import DATASET_KEYS
from experiments.run_safe_resumption import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/asymmetric_label_noise_resumption.yaml"
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    base = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    jobs = [
        (profile, source, target, seed)
        for profile in base["noise_profiles"]
        for source, target in (DATASET_KEYS, tuple(reversed(DATASET_KEYS)))
        for seed in base["seeds"]
    ]
    for index, (profile, source, target, seed) in enumerate(jobs, 1):
        config = copy.deepcopy(base)
        config["experiment_prefix"] = (
            f"{base['experiment_prefix_base']}_{profile['name']}"
        )
        config["artifact_dir"] = str(Path(base["artifact_dir"]) / profile["name"])
        config["recovery"]["review_label_noise_rate"] = 0.0
        config["recovery"]["review_label_noise_by_class"] = {
            0: float(profile["benign_to_malicious"]),
            1: float(profile["malicious_to_benign"]),
        }
        result = run(config, source, target, seed, args.force)
        selected = result["safe_resumption"]["selected_policy"]
        print(
            f"[{index}/{len(jobs)}] {result['experiment_id']} "
            f"profile={profile['name']} decision={selected['decision']} "
            f"budget={selected['label_budget']} unsafe={selected['unsafe_resumption']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
