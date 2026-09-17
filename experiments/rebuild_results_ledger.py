"""Recover the master index from validated per-experiment JSON files; no retraining."""
import argparse
from pathlib import Path

from experiments.research_common import rebuild_results_ledger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/metrics"))
    args = parser.parse_args()
    count = rebuild_results_ledger(args.output_dir)
    print(f"Rebuilt ledger from {count} validated, uniquely identified result files.")


if __name__ == "__main__":
    main()
