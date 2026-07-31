"""Flower simulation driver with checkpoint/resume.

Runs training in chunks of `rounds_per_checkpoint` rounds; after each chunk the global
model + strategy byte-accounting state + accumulated per-round history are written to
disk, so an interrupted run resumes from the last completed chunk instead of round 0.
Built in from day one per ROADMAP.md Phase 3 (free-tier GPU sessions die mid-run); also
just makes long local CPU runs safe to interrupt.
"""
import json
from pathlib import Path

import flwr as fl
import numpy as np
import torch

from src.fl.client import FlowerNIDSClient
from src.fl.dataset import client_loaders
from src.fl.dp import make_dp_compatible
from src.fl.param_utils import unflatten
from src.fl.strategies.comm_efficient import CommEfficientFedAvg
from src.metrics import compute_metrics
from src.models.build import build_model


def _build_model(model_name: str, n_features: int, n_classes: int, dp: bool):
    model = build_model(model_name, n_features, n_classes)
    return make_dp_compatible(model) if dp else model


def _checkpoint_paths(checkpoint_dir: Path):
    return {
        "params": checkpoint_dir / "global_params.npz",
        "state": checkpoint_dir / "state.json",
    }


def run_federated(
    run_id: str,
    X_train, y_train, X_test, y_test,
    client_indices: list[np.ndarray],
    model_name: str, n_features: int, n_classes: int, class_names: list[str],
    n_rounds: int, fraction_fit: float, local_epochs: int, lr: float, batch_size: int,
    topk_ratio: float, dp: bool, dp_noise_multiplier: float, dp_max_grad_norm: float,
    rounds_per_checkpoint: int = 5, checkpoint_dir: Path = None, seed: int = 42,
    num_cpus_per_client: float = 1.0,
):
    n_clients = len(client_indices)
    checkpoint_dir = checkpoint_dir or Path("checkpoints") / run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    paths = _checkpoint_paths(checkpoint_dir)

    template_model = _build_model(model_name, n_features, n_classes, dp)
    initial_ndarrays = [v.cpu().numpy() for v in template_model.state_dict().values()]

    start_round, history = 0, {"rounds": [], "server_test_macro_f1": [], "server_test_accuracy": [],
                                "distributed_macro_f1": [], "bytes_this_round": [], "cumulative_bytes": [],
                                "dp_epsilon_max": []}
    if paths["state"].exists():
        with open(paths["state"]) as f:
            saved = json.load(f)
        start_round = saved["completed_round"]
        history = saved["history"]
        loaded = np.load(paths["params"])
        initial_ndarrays = [loaded[f"arr_{i}"] for i in range(len(loaded.files))]

    if start_round >= n_rounds:
        return history

    def client_fn(cid: str):
        client_id = int(cid)
        train_loader, val_loader, n_train = client_loaders(
            X_train, y_train, client_indices, client_id, batch_size, seed=seed
        )
        return FlowerNIDSClient(
            client_id, model_name, n_features, n_classes, train_loader, val_loader, n_train, class_names, dp=dp
        ).to_client()

    def fit_config_fn(server_round: int):
        return {
            "lr": lr, "local_epochs": local_epochs, "topk_ratio": topk_ratio,
            "dp": dp, "dp_noise_multiplier": dp_noise_multiplier, "dp_max_grad_norm": dp_max_grad_norm,
            "run_id": run_id, "server_round": start_round + server_round,
        }

    def evaluate_fn(server_round, parameters, config):
        model = _build_model(model_name, n_features, n_classes, dp)
        state_dict = model.state_dict()
        for k, v in zip(state_dict.keys(), parameters):
            state_dict[k] = torch.tensor(v)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        with torch.no_grad():
            out = model(torch.from_numpy(X_test))
            preds = out.argmax(dim=1).numpy()
        m = compute_metrics(y_test, preds, class_names)
        return 1.0 - m["macro_f1"], {"macro_f1": m["macro_f1"], "accuracy": m["accuracy"]}

    while start_round < n_rounds:
        chunk_rounds = min(rounds_per_checkpoint, n_rounds - start_round)

        strategy = CommEfficientFedAvg(
            initial_ndarrays=initial_ndarrays,
            topk_ratio=topk_ratio,
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_fit,
            min_fit_clients=max(1, int(n_clients * fraction_fit)),
            min_evaluate_clients=max(1, int(n_clients * fraction_fit)),
            min_available_clients=n_clients,
            on_fit_config_fn=fit_config_fn,
            evaluate_fn=evaluate_fn,
        )

        fl_history = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=n_clients,
            config=fl.server.ServerConfig(num_rounds=chunk_rounds),
            strategy=strategy,
            client_resources={"num_cpus": num_cpus_per_client},
            ray_init_args={"include_dashboard": False, "logging_level": "ERROR"},
        )

        # Flower's per-chunk History always includes a round-0 entry for the initial
        # (pre-training) global model. Look everything up by round number (not list
        # position) since fit-metrics only exist for rounds >=1 while centralized-eval
        # metrics include round 0 -> positional zipping silently misaligns them.
        centralized_f1 = dict(fl_history.metrics_centralized.get("macro_f1", []))
        centralized_acc = dict(fl_history.metrics_centralized.get("accuracy", []))
        fit_metrics = fl_history.metrics_distributed_fit
        bytes_by_round = dict(fit_metrics.get("bytes_this_round", []))
        cum_by_round = dict(fit_metrics.get("cumulative_bytes", []))
        eps_by_round = dict(fit_metrics.get("dp_epsilon_max", []))
        dist_f1_by_round = dict(fl_history.metrics_distributed.get("macro_f1", []))

        # The strategy's cumulative_bytes counter is relative to this chunk (a fresh
        # Strategy is constructed per checkpoint chunk) -> offset by whatever was
        # already accumulated in prior chunks so the running total spans the whole run.
        prior_cumulative = next((v for v in reversed(history["cumulative_bytes"]) if v is not None), 0)

        # Round 0 is only meaningful once, at the very start of training; every later
        # chunk's round 0 just duplicates the previous chunk's final round.
        rounds_this_chunk = range(1, chunk_rounds + 1) if start_round > 0 else range(0, chunk_rounds + 1)
        for rnd in rounds_this_chunk:
            history["rounds"].append(start_round + rnd)
            history["server_test_macro_f1"].append(centralized_f1.get(rnd))
            history["server_test_accuracy"].append(centralized_acc.get(rnd))
            history["bytes_this_round"].append(bytes_by_round.get(rnd))
            cum_this_round = cum_by_round.get(rnd)
            history["cumulative_bytes"].append(prior_cumulative + cum_this_round if cum_this_round is not None else None)
            history["dp_epsilon_max"].append(eps_by_round.get(rnd))
            history["distributed_macro_f1"].append(dist_f1_by_round.get(rnd))

        initial_ndarrays = unflatten(strategy.global_flat, strategy.model_shapes, strategy.sizes)

        start_round += chunk_rounds
        np.savez(paths["params"], *initial_ndarrays)
        with open(paths["state"], "w") as f:
            json.dump({"completed_round": start_round, "history": history}, f, indent=2, default=str)

    return history
