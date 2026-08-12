"""Run the frozen five-seed uncertainty-aware experiment matrix."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from experiments.run_uncertainty import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cicids-config", default="configs/research_baseline.yaml")
    parser.add_argument("--rt-config", default="configs/rt_iot2022.yaml")
    parser.add_argument("--stage", choices=["cicids", "rt", "compact", "all"], default="all")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cicids = yaml.safe_load(Path(args.cicids_config).read_text(encoding="utf-8"))
    rt = yaml.safe_load(Path(args.rt_config).read_text(encoding="utf-8"))
    jobs = []
    if args.stage in {"cicids", "all"}:
        for seed in cicids["seeds"]:
            jobs.append((cicids, "cicids", "random", seed, None, False))
            jobs.append((cicids, "cicids", "chronological", seed, None, False))
        for family in cicids["heldout_attack_families"]:
            for seed in cicids["seeds"]:
                jobs.append((cicids, "cicids", "heldout", seed, family, False))
    if args.stage in {"rt", "all"}:
        for seed in rt["seeds"]:
            jobs.append((rt, "rt_iot2022", "random", seed, None, False))
        for family in rt["heldout_attack_families"]:
            for seed in rt["seeds"]:
                jobs.append((rt, "rt_iot2022", "heldout", seed, family, False))
    if args.stage in {"compact", "all"}:
        for seed in cicids["seeds"]:
            jobs.append((cicids, "cicids", "random", seed, None, True))
            jobs.append((cicids, "cicids", "chronological", seed, None, True))
    for index, (config, dataset, protocol, seed, family, compact) in enumerate(jobs, 1):
        result = run(
            config,
            dataset,
            protocol,
            seed,
            family=family,
            compact=compact,
            force=args.force,
        )
        method = result["uncertainty"]["methods"]["mondrian_conformal_alpha_10"]
        print(
            f"[{index}/{len(jobs)}] {result['experiment_id']} "
            f"coverage={method['coverage']:.3f} "
            f"confident-FNR={method['confident_false_negative_rate']:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
