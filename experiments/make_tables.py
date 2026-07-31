"""Regenerates results/tables/*.csv from results/runs/*.json — every table cell in the
paper is regenerable by running this after experiments/run_grid.py (ROADMAP.md §3).

    python -m experiments.make_tables
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.metrics import summarize_over_seeds

RUNS_DIR = Path("results/runs")
OUT_DIR = Path("results/tables")
SEEDS = [0, 1, 2]

METHOD_LABELS = {
    "centralized": "Centralized",
    "fedavg": "FedAvg",
    "topk": "Your-method (top-k)",
    "topk_dp": "Your-method+DP (flagship)",
}


def load_run(run_id: str) -> dict | None:
    path = RUNS_DIR / f"{run_id}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def rounds_to_converge(history: dict, tolerance: float = 0.95) -> int | None:
    f1s = history["server_test_macro_f1"]
    rounds = history["rounds"]
    valid = [(r, f) for r, f in zip(rounds, f1s) if f is not None]
    if not valid:
        return None
    final_f1 = valid[-1][1]
    if final_f1 <= 0:
        return valid[-1][0]
    for r, f in valid:
        if f >= tolerance * final_f1:
            return r
    return valid[-1][0]


def table1() -> pd.DataFrame:
    """Accuracy vs privacy/communication tradeoff (ROADMAP.md §3 Table 1). final_metrics
    is a full compute_metrics() dict for every method (centralized and federated alike) —
    federated runs get it from one extra evaluation of the converged global model
    (see experiments/run_experiment.py), so per-class recall/FPR/precision are real for
    every row, not just the centralized one."""
    rows = []
    for method in ["centralized", "fedavg", "topk", "topk_dp"]:
        metric_dicts = []
        bytes_per_round_vals, converge_round_vals = [], []
        for seed in SEEDS:
            run_id = f"{method}_seed{seed}" if method == "centralized" else f"{method}_alpha0.5_seed{seed}"
            run = load_run(run_id)
            if run is None:
                continue
            metric_dicts.append(run["final_metrics"])
            if run["history"] is not None:
                history = run["history"]
                bytes_vals = [b for b in history["bytes_this_round"] if b is not None]
                if bytes_vals:
                    bytes_per_round_vals.append(np.mean(bytes_vals))
                rtc = rounds_to_converge(history)
                if rtc is not None:
                    converge_round_vals.append(rtc)
        if not metric_dicts:
            continue

        class_names = list(metric_dicts[0]["per_class_recall"].keys())
        row = {
            "method": METHOD_LABELS[method],
            "accuracy_mean": np.mean([m["accuracy"] for m in metric_dicts]),
            "accuracy_std": np.std([m["accuracy"] for m in metric_dicts]),
            "macro_f1_mean": np.mean([m["macro_f1"] for m in metric_dicts]),
            "macro_f1_std": np.std([m["macro_f1"] for m in metric_dicts]),
            "macro_precision_mean": np.mean([m["macro_precision"] for m in metric_dicts]),
            "overall_fpr_mean": np.mean([m["overall_fpr"] for m in metric_dicts]),
            "bytes_per_round_mean": np.mean(bytes_per_round_vals) if bytes_per_round_vals else None,
            "rounds_to_converge_mean": np.mean(converge_round_vals) if converge_round_vals else None,
            "n_seeds": len(metric_dicts),
        }
        for c in class_names:
            row[f"recall_{c}"] = np.mean([m["per_class_recall"][c] for m in metric_dicts])
            row[f"fpr_{c}"] = np.mean([m["per_class_fpr"][c] for m in metric_dicts])
        rows.append(row)
    return pd.DataFrame(rows)


def table2() -> pd.DataFrame:
    """Non-IID robustness: macro-F1 across alpha, FedAvg vs your-method (ROADMAP.md §3 Table 2)."""
    rows = []
    for method in ["fedavg", "topk"]:
        for alpha in [0.1, 0.5, 1.0, "iid"]:
            f1_vals = []
            for seed in SEEDS:
                run_id = f"{method}_alpha{alpha}_seed{seed}"
                run = load_run(run_id)
                if run is None:
                    continue
                f1 = run["final_metrics"]["macro_f1"]
                if f1 is not None:
                    f1_vals.append(f1)
            if not f1_vals:
                continue
            rows.append({
                "method": METHOD_LABELS[method],
                "alpha": alpha,
                "macro_f1_mean": np.mean(f1_vals),
                "macro_f1_std": np.std(f1_vals),
                "n_seeds": len(f1_vals),
            })
    return pd.DataFrame(rows)


def table3_alignment(shap_run_id: str) -> pd.DataFrame:
    """Feature/attack alignment (ROADMAP.md §3 Table 3), from results/shap/<run_id>.json."""
    path = Path("results/shap") / f"{shap_run_id}.json"
    if not path.exists():
        return pd.DataFrame()
    with open(path) as f:
        d = json.load(f)
    return pd.DataFrame(d["alignment"])


def privacy_ablation() -> pd.DataFrame:
    """DP noise-multiplier sweep: accuracy cost of privacy (ROADMAP.md §3 ablation)."""
    rows = []
    no_privacy_f1 = []
    for seed in SEEDS:
        run = load_run(f"topk_alpha0.5_seed{seed}")
        if run and run["final_metrics"]["macro_f1"] is not None:
            no_privacy_f1.append(run["final_metrics"]["macro_f1"])
    if no_privacy_f1:
        rows.append({"dp_noise_multiplier": 0.0, "label": "no privacy", "macro_f1_mean": np.mean(no_privacy_f1),
                      "macro_f1_std": np.std(no_privacy_f1), "dp_epsilon_mean": None, "n_seeds": len(no_privacy_f1)})

    for nm in [0.5, 1.0, 2.0]:
        f1_vals, eps_vals = [], []
        for seed in SEEDS:
            run_id = f"privacy_ablation_nm{nm}_alpha0.5_seed{seed}" if nm != 1.0 else f"topk_dp_alpha0.5_seed{seed}"
            run = load_run(run_id)
            if run is None:
                continue
            if run["final_metrics"]["macro_f1"] is not None:
                f1_vals.append(run["final_metrics"]["macro_f1"])
            if run["final_metrics"].get("dp_epsilon_max") is not None:
                eps_vals.append(run["final_metrics"]["dp_epsilon_max"])
        if f1_vals:
            rows.append({
                "dp_noise_multiplier": nm, "label": f"DP (noise_mult={nm})",
                "macro_f1_mean": np.mean(f1_vals), "macro_f1_std": np.std(f1_vals),
                "dp_epsilon_mean": np.mean(eps_vals) if eps_vals else None, "n_seeds": len(f1_vals),
            })
    return pd.DataFrame(rows)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t1 = table1()
    t1.to_csv(OUT_DIR / "table1_tradeoff.csv", index=False)
    print("Table 1:\n", t1, "\n")

    t2 = table2()
    t2.to_csv(OUT_DIR / "table2_noniid_robustness.csv", index=False)
    print("Table 2:\n", t2, "\n")

    # Uses the non-DP top-k model: it captures more attack classes (4 vs 3) than the
    # DP-noised flagship, giving a stronger demonstration of the XAI architecture itself.
    # The flagship (topk_dp_alpha0.5_seed0) SHAP results are also available in
    # results/shap/ for the privacy-specific angle -- weaker alignment there is itself an
    # honest finding about DP's utility cost, not a pipeline issue.
    t3 = table3_alignment("topk_alpha0.5_seed0")
    if not t3.empty:
        t3.to_csv(OUT_DIR / "table3_alignment.csv", index=False)
        print("Table 3:\n", t3, "\n")

    t4 = privacy_ablation()
    t4.to_csv(OUT_DIR / "privacy_ablation.csv", index=False)
    print("Privacy ablation:\n", t4, "\n")


if __name__ == "__main__":
    main()
