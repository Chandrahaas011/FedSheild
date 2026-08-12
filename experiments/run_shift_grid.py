"""Run five-seed compound chronological/campaign-shift experiments."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/research_baseline.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    commands = []
    for model in config["models"]:
        for seed in config["seeds"]:
            commands.append(
                [
                    sys.executable,
                    "-m",
                    "experiments.run_research_baseline",
                    "--config",
                    args.config,
                    "--model",
                    model,
                    "--seed",
                    str(seed),
                    "--protocol",
                    "chronological",
                ]
            )
    for seed in config["seeds"]:
        commands.append(
            [
                sys.executable,
                "-m",
                "experiments.run_research_proposed",
                "--config",
                args.config,
                "--seed",
                str(seed),
                "--protocol",
                "chronological",
            ]
        )
    if args.force:
        for command in commands:
            command.append("--force")
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

