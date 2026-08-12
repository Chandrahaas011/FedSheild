"""Run the complete inexpensive RT-IoT2022 replication matrix."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from experiments.run_rt_iot2022 import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rt_iot2022.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    jobs = [(None, model, seed) for model in config["models"] for seed in config["seeds"]]
    jobs += [
        (family, model, seed)
        for family in config["heldout_attack_families"]
        for model in config["models"]
        for seed in config["seeds"]
    ]
    for index, (family, model, seed) in enumerate(jobs, start=1):
        result = run(config, model, seed, heldout=family, force=args.force)
        print(f"[{index}/{len(jobs)}] {result['experiment_id']} macro-F1={result['metrics']['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
