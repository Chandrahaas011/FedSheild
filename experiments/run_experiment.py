"""Single experiment entrypoint: one YAML config -> one trained model -> one result JSON.

    python -m experiments.run_experiment --config configs/topk_dp.yaml --seed 1 --alpha 0.5

Every table cell in the paper is regenerable by running this on the matching config
(ROADMAP.md §1, "Experiment tracking" row).
"""
import argparse
import copy
from pathlib import Path

import numpy as np
import torch
import yaml

from src.data.partition import partition_data
from src.fl.dataset import load_processed, subsample
from src.fl.simulation import run_federated
from src.metrics import compute_metrics
from src.train_centralized import train_centralized
from src.utils.checkpoint import load_run_result, save_run_result
from src.utils.mlflow_utils import log_round_metrics, mlflow_run
from src.utils.model_io import load_trained_model
from src.utils.seed import set_seed

RESULTS_DIR = Path("results/runs")
RUN_CONFIGS_DIR = Path("results/run_configs")


def make_run_id(config: dict) -> str:
    parts = [config["experiment_name"]]
    if "alpha" in config:
        parts.append(f"alpha{config['alpha']}")
    parts.append(f"seed{config['seed']}")
    return "_".join(str(p) for p in parts)


def run_single(config: dict, force: bool = False) -> dict:
    run_id = make_run_id(config)
    result_path = RESULTS_DIR / f"{run_id}.json"
    if not force:
        existing = load_run_result(result_path)
        if existing is not None:
            return existing

    RUN_CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RUN_CONFIGS_DIR / f"{run_id}.yaml", "w") as f:
        yaml.safe_dump(config, f)

    set_seed(config["seed"])
    splits, metadata = load_processed()
    class_names = metadata["classes"]
    n_features = metadata["n_features"]
    n_classes = len(class_names)

    X_train, y_train = subsample(*splits["train"], frac=config.get("data_frac", 1.0), seed=config["seed"])
    X_val, y_val = splits["val"]
    X_test, y_test = splits["test"]

    with mlflow_run("fl-nids", run_id, config):
        if config["method"] == "centralized":
            model, test_metrics = train_centralized(
                X_train, y_train, X_val, y_val, X_test, y_test,
                model_name=config["model"], n_features=n_features, n_classes=n_classes,
                class_names=class_names, epochs=config["epochs"], lr=config["lr"],
                batch_size=config["batch_size"], seed=config["seed"],
                checkpoint_path=Path("checkpoints") / run_id / "centralized.pt",
                patience=config.get("patience", 5),
                reset=force,
            )
            result = {"run_id": run_id, "config": config, "final_metrics": test_metrics, "history": None}
            log_round_metrics(test_metrics, step=test_metrics.get("epochs_run", 0))
        else:
            client_indices = partition_data(y_train, config["n_clients"], config["alpha"], seed=config["seed"])
            client_indices = [np.asarray(idx) for idx in client_indices]

            adaptive_topk = config["method"] in ("ra_topk", "ra_topk_dp")
            topk_ratio = config.get("topk_ratio", -1) if config["method"] in ("topk", "topk_dp") else -1
            dp = config.get("dp", False) or config["method"] in ("topk_dp", "ra_topk_dp")

            history = run_federated(
                run_id=run_id, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                client_indices=client_indices, model_name=config["model"], n_features=n_features,
                n_classes=n_classes, class_names=class_names, n_rounds=config["n_rounds"],
                fraction_fit=config["fraction_fit"], local_epochs=config["local_epochs"],
                lr=config["lr"], batch_size=config["batch_size"], topk_ratio=topk_ratio,
                dp=dp, dp_noise_multiplier=config.get("dp_noise_multiplier", 1.0),
                dp_max_grad_norm=config.get("dp_max_grad_norm", 1.0),
                adaptive_topk=adaptive_topk,
                adaptive_min_ratio=config.get("adaptive_min_ratio", 0.02),
                adaptive_max_ratio=config.get("adaptive_max_ratio", 0.20),
                adaptive_energy_target=config.get("adaptive_energy_target", 0.90),
                rounds_per_checkpoint=config.get("rounds_per_checkpoint", 5), seed=config["seed"],
                reset=force,
            )
            # history only tracks macro_f1/accuracy per round (cheap to compute every
            # round); the full per-class recall/FPR/precision breakdown Table 1 needs is
            # computed once here, from the final converged global model.
            final_model = load_trained_model(run_id, config, n_features, n_classes)
            with torch.no_grad():
                test_preds = final_model(torch.from_numpy(X_test)).argmax(dim=1).numpy()
            final_metrics = compute_metrics(y_test, test_preds, class_names)
            final_metrics["cumulative_bytes"] = history["cumulative_bytes"][-1] if history["cumulative_bytes"] else None
            final_metrics["cumulative_client_upload_bytes"] = sum(
                b for b in history.get("client_upload_bytes", []) if b is not None
            )
            final_metrics["cumulative_total_array_payload_bytes"] = sum(
                b for b in history.get("total_array_payload_bytes", []) if b is not None
            )
            final_metrics["dp_epsilon_max"] = next((e for e in reversed(history["dp_epsilon_max"]) if e is not None), None)
            selected_ratios = [r for r in history.get("selected_sparsity_ratio_mean", []) if r is not None]
            final_metrics["selected_sparsity_ratio_mean"] = (
                float(np.mean(selected_ratios)) if selected_ratios else None
            )
            result = {"run_id": run_id, "config": config, "final_metrics": final_metrics, "history": history}
            for i, rnd in enumerate(history["rounds"]):
                round_metrics = {k: v[i] for k, v in history.items() if k != "rounds" and v[i] is not None}
                log_round_metrics(round_metrics, step=rnd)

    save_run_result(result_path, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--alpha", default=None)
    parser.add_argument("--data-frac", type=float, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    config = copy.deepcopy(config)
    if args.seed is not None:
        config["seed"] = args.seed
    if args.alpha is not None and "alpha" in config:
        config["alpha"] = args.alpha
    if args.data_frac is not None:
        config["data_frac"] = args.data_frac

    result = run_single(config, force=args.force)
    print(f"run_id={result['run_id']}")
    print(result["final_metrics"])


if __name__ == "__main__":
    main()
