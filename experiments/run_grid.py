"""Experiment grid runner (ROADMAP.md §3), resumable — each cell is skipped if its
result JSON already exists (see run_single's cache check in run_experiment.py).

    python -m experiments.run_grid

This session's grid is deliberately scaled down from ROADMAP.md's full ~100-run grid
(fewer seeds are NOT used here -- all 3 seeds are kept per the "report mean +/- std"
rule -- instead data_frac/n_rounds/n_clients are reduced; see README.md "Scaling up").
"""
import copy
import time
import traceback
from pathlib import Path

import yaml

from experiments.run_experiment import run_single

SEEDS = [0, 1, 2]
ALPHAS_FOR_TABLE2 = [0.1, 1.0, "iid"]  # 0.5 already covered by the Table-1 base configs
DP_NOISE_MULTIPLIERS_FOR_ABLATION = [0.5, 2.0]  # 1.0 already covered by topk_dp base config


def load_config(name: str) -> dict:
    with open(Path("configs") / f"{name}.yaml") as f:
        return yaml.safe_load(f)


def build_grid() -> list[dict]:
    grid = []

    # --- Table 1: method comparison at the default non-IID setting (alpha=0.5) ---
    for method_name in ["centralized", "fedavg", "topk", "ra_topk", "topk_dp", "ra_topk_dp"]:
        base = load_config(method_name)
        for seed in SEEDS:
            cfg = copy.deepcopy(base)
            cfg["seed"] = seed
            grid.append(cfg)

    # --- Table 2: non-IID robustness sweep, FedAvg vs your-method ---
    for method_name in ["fedavg", "topk", "ra_topk"]:
        base = load_config(method_name)
        for alpha in ALPHAS_FOR_TABLE2:
            for seed in SEEDS:
                cfg = copy.deepcopy(base)
                cfg["alpha"] = alpha
                cfg["seed"] = seed
                grid.append(cfg)

    # --- Privacy ablation: DP noise-multiplier sweep at alpha=0.5 ---
    base = load_config("topk_dp")
    for nm in DP_NOISE_MULTIPLIERS_FOR_ABLATION:
        for seed in SEEDS:
            cfg = copy.deepcopy(base)
            cfg["dp_noise_multiplier"] = nm
            cfg["seed"] = seed
            cfg["experiment_name"] = f"privacy_ablation_nm{nm}"
            grid.append(cfg)

    return grid


def main():
    grid = build_grid()
    print(f"Grid has {len(grid)} cells.")
    results, failures = [], []
    for i, cfg in enumerate(grid):
        t0 = time.time()
        try:
            result = run_single(cfg)
            results.append(result)
            print(f"[{i+1}/{len(grid)}] {result['run_id']} -> {result['final_metrics']} ({time.time()-t0:.1f}s)")
        except Exception as e:
            failures.append({"config": cfg, "error": str(e)})
            print(f"[{i+1}/{len(grid)}] FAILED: {cfg.get('experiment_name')} seed={cfg.get('seed')}: {e}")
            traceback.print_exc()

    print(f"\nDone: {len(results)} succeeded, {len(failures)} failed.")
    if failures:
        for f in failures:
            print(" -", f["config"].get("experiment_name"), f["config"].get("seed"), "->", f["error"])


if __name__ == "__main__":
    main()
