"""Federated XAI entrypoint (ROADMAP.md Phase 4): loads a run's converged global model,
partitions the held-out test set the same way the run's clients were partitioned, has
each "client" explain its own flagged traffic locally with SHAP, and aggregates only
mean(|SHAP|) summaries server-side. Also runs the feature/attack alignment check.

    python -m experiments.run_xai --run-id topk_dp_alpha0.5_seed0
"""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from src.data.labels import ATTACK_SIGNATURE_FEATURES
from src.data.partition import partition_data
from src.fl.dataset import load_processed
from src.utils.model_io import load_trained_model
from src.xai.alignment import alignment_table
from src.xai.federated_shap import run_federated_shap

RESULTS_DIR = Path("results/shap")
LOCAL_DEBUG_DIR = Path("results/shap_local_debug")


def run(
    run_id: str, n_clients: int | None = None, alpha=None, seed: int | None = None,
    export_local_examples: bool = False,
):
    with open(Path("results/run_configs") / f"{run_id}.yaml") as f:
        config = yaml.safe_load(f)

    splits, metadata = load_processed()
    class_names = metadata["classes"]
    feature_names = metadata["feature_columns"]
    X_test, y_test = splits["test"]

    model = load_trained_model(run_id, config, metadata["n_features"], len(class_names))

    n_clients = n_clients or config.get("n_clients", 10)
    alpha = alpha if alpha is not None else config.get("alpha", 0.5)
    seed = config.get("seed", 0) if seed is None else seed
    client_indices = partition_data(y_test, n_clients, alpha, seed=seed)
    client_indices = [np.asarray(idx) for idx in client_indices]

    global_summary, local_examples = run_federated_shap(
        model, X_test, client_indices, feature_names, class_names, benign_idx=0, seed=seed,
        collect_local_examples=export_local_examples,
    )
    alignment = alignment_table(global_summary, feature_names, ATTACK_SIGNATURE_FEATURES)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "run_id": run_id,
        "global_summary": global_summary,
        "alignment": alignment,
        "feature_names": feature_names,
        "class_names": class_names,
    }
    with open(RESULTS_DIR / f"{run_id}.json", "w") as f:
        json.dump(out, f, indent=2)
    if export_local_examples:
        # This is intentionally outside the normal server-facing result directory.
        # In a real deployment the caller must execute this export inside the client
        # security boundary and must not transmit the output to the coordinator.
        LOCAL_DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        with open(LOCAL_DEBUG_DIR / f"{run_id}.json", "w") as f:
            json.dump({"run_id": run_id, "local_examples": local_examples[:30],
                       "feature_names": feature_names}, f, indent=2)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--n-clients", type=int, default=None)
    parser.add_argument("--alpha", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--export-local-examples", action="store_true",
                        help="Write sensitive per-alert examples to the local debug directory.")
    args = parser.parse_args()
    out = run(args.run_id, n_clients=args.n_clients, alpha=args.alpha, seed=args.seed,
              export_local_examples=args.export_local_examples)
    print(f"Explained {len(out['global_summary'])} attack classes; server output contains summaries only.")
    for row in out["alignment"]:
        print(row["attack_class"], "overlap:", row["overlap_count"], "/", len(row["known_signature_features"]))


if __name__ == "__main__":
    main()
