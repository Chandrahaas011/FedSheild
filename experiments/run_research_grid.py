"""Run the frozen research grid with each cell isolated in its own process."""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/research_baseline.yaml")
    parser.add_argument("--stage", choices=["baselines", "proposed", "all"], default="all")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    commands = []
    if args.stage in {"baselines", "all"}:
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
                    ]
                )
    if args.stage in {"proposed", "all"}:
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
                ]
            )
    if args.force:
        for command in commands:
            command.append("--force")

    failures = []
    grid_start = time.perf_counter()
    for index, command in enumerate(commands, start=1):
        start = time.perf_counter()
        print(f"[{index}/{len(commands)}] {' '.join(command[2:])}", flush=True)
        completed = subprocess.run(command, check=False)
        elapsed = time.perf_counter() - start
        print(f"completed in {elapsed:.1f}s with exit code {completed.returncode}", flush=True)
        if completed.returncode != 0:
            failures.append((command, completed.returncode))
    print(f"Grid duration: {time.perf_counter() - grid_start:.1f}s")
    if failures:
        for command, code in failures:
            print(f"FAILED ({code}): {' '.join(command)}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

