from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/research_baseline.yaml")
    parser.add_argument("--models", nargs="+", default=["decision_tree", "hist_gradient_boosting"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    commands = []
    for family in config["heldout_attack_families"]:
        for model in args.models:
            for seed in config["seeds"]:
                command = [
                    sys.executable,
                    "-m",
                    "experiments.run_heldout_attack",
                    "--config",
                    args.config,
                    "--model",
                    model,
                    "--family",
                    family,
                    "--seed",
                    str(seed),
                ]
                if args.force:
                    command.append("--force")
                commands.append(command)
    failures = []
    for index, command in enumerate(commands, 1):
        print(f"[{index}/{len(commands)}] {' '.join(command[2:])}", flush=True)
        completed = subprocess.run(command, check=False)
        if completed.returncode:
            failures.append((command, completed.returncode))
    if failures:
        for command, code in failures:
            print(f"FAILED ({code}): {' '.join(command)}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

