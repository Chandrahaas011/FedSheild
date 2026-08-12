"""Run both NetFlow-v2 transfer directions over two lightweight models and five seeds."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from experiments.run_cross_netflow import DATASET_KEYS, run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/netflow_v2_cross.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    jobs = []
    for source, target in (DATASET_KEYS, tuple(reversed(DATASET_KEYS))):
        for model in config["models"]:
            for seed in config["seeds"]:
                jobs.append((source, target, model, seed))
    for index, (source, target, model, seed) in enumerate(jobs, 1):
        result = run(config, source, target, model, seed, args.force)
        mondrian = result["uncertainty"]["methods"]["mondrian_conformal_alpha_10"]
        print(
            f"[{index}/{len(jobs)}] {result['experiment_id']} "
            f"macro-F1={result['metrics']['macro_f1']:.3f} "
            f"coverage={mondrian['coverage']:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
