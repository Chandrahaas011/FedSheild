from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from flwr.client import NumPyClient

from src.fl.dp import make_dp_compatible, train_with_dp
from src.fl.param_utils import dense_bytes, flatten, get_shapes, unflatten
from src.fl.sparsify import residual_adaptive_topk_sparsify, sparse_bytes, topk_sparsify
from src.metrics import compute_metrics
from src.models.build import build_model

RESIDUAL_DIR = Path("data/partitions/_topk_residuals")


def _residual_path(client_id: int, run_id: str) -> Path:
    return RESIDUAL_DIR / run_id / f"client_{client_id}.npy"


def _load_residual(client_id: int, run_id: str, length: int) -> np.ndarray:
    path = _residual_path(client_id, run_id)
    if path.exists():
        arr = np.load(path)
        if len(arr) == length:
            return arr
    return np.zeros(length, dtype=np.float32)


def _save_residual(client_id: int, run_id: str, residual: np.ndarray) -> None:
    path = _residual_path(client_id, run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, residual)


class FlowerNIDSClient(NumPyClient):
    def __init__(self, client_id: int, model_name: str, n_features: int, n_classes: int,
                 train_loader, val_loader, n_train: int, class_names: list[str], dp: bool = False):
        self.client_id = client_id
        self.model = build_model(model_name, n_features, n_classes)
        if dp:
            # BatchNorm mixes information across a batch, which breaks the per-sample
            # gradient guarantee Opacus's DP-SGD relies on -> use GroupNorm instead.
            # Applied once here (not per-fit-call) so parameter shapes stay identical
            # across every round of a DP run, matching the server's tracked shapes.
            self.model = make_dp_compatible(self.model)
        self.dp = dp
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.n_train = n_train
        self.class_names = class_names
        self.n_features = n_features
        self.n_classes = n_classes

    def get_parameters(self, config):
        return [v.cpu().numpy() for v in self.model.state_dict().values()]

    def _set_parameters(self, parameters):
        state_dict = self.model.state_dict()
        for k, v in zip(state_dict.keys(), parameters):
            state_dict[k] = torch.tensor(v)
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self._set_parameters(parameters)
        if self.train_loader is None or len(self.train_loader.dataset) == 0:
            # Possible under severe non-IID (low Dirichlet alpha) with many clients: a
            # client can draw zero samples for every class. Report a no-op update with
            # num_examples=0 so the server's weighted aggregation simply ignores it.
            topk_ratio = float(config.get("topk_ratio", -1))
            if topk_ratio > 0 or bool(config.get("adaptive_topk", False)):
                return [np.array([], dtype=np.int64), np.array([], dtype=np.float32)], 0, {"bytes_sent": 0, "is_sparse": True}
            return parameters, 0, {"bytes_sent": 0, "is_sparse": False}

        lr = float(config.get("lr", 0.01))
        local_epochs = int(config.get("local_epochs", 1))
        topk_ratio = float(config.get("topk_ratio", -1))
        adaptive_topk = bool(config.get("adaptive_topk", False))
        use_dp = bool(config.get("dp", False))
        run_id = str(config.get("run_id", "default"))
        server_round = int(config.get("server_round", 0))

        global_flat, shapes_sizes = flatten(parameters)
        shapes = get_shapes(parameters)
        sizes = shapes_sizes[1]

        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        metrics = {}

        if use_dp:
            noise_mult = float(config.get("dp_noise_multiplier", 1.0))
            max_grad_norm = float(config.get("dp_max_grad_norm", 1.0))
            trained_model, avg_loss, epsilon = train_with_dp(
                self.model, optimizer, self.train_loader, local_epochs,
                noise_mult, max_grad_norm, self.client_id, run_id,
            )
            new_state = {}
            for (k, _), v in zip(self.model.state_dict().items(), trained_model.state_dict().values()):
                new_state[k] = v
            self.model.load_state_dict(new_state, strict=True)
            metrics["dp_epsilon"] = float(epsilon)
            metrics["train_loss"] = avg_loss
        else:
            criterion = nn.CrossEntropyLoss()
            self.model.train()
            total_loss, n_batches = 0.0, 0
            for _ in range(local_epochs):
                for xb, yb in self.train_loader:
                    optimizer.zero_grad()
                    out = self.model(xb)
                    loss = criterion(out, yb)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    n_batches += 1
            metrics["train_loss"] = total_loss / max(1, n_batches)

        new_params = [v.detach().cpu().numpy() for v in self.model.state_dict().values()]
        new_flat, _ = flatten(new_params)
        delta = new_flat - global_flat

        if adaptive_topk or topk_ratio > 0:
            residual = _load_residual(self.client_id, run_id, len(delta))
            if adaptive_topk:
                idx, values, new_residual, selected_ratio = residual_adaptive_topk_sparsify(
                    delta, residual,
                    min_ratio=float(config.get("adaptive_min_ratio", 0.02)),
                    max_ratio=float(config.get("adaptive_max_ratio", 0.20)),
                    energy_target=float(config.get("adaptive_energy_target", 0.90)),
                )
                metrics["selected_sparsity_ratio"] = float(selected_ratio)
            else:
                idx, values, new_residual = topk_sparsify(delta, topk_ratio, residual)
                metrics["selected_sparsity_ratio"] = float(topk_ratio)
            _save_residual(self.client_id, run_id, new_residual)
            metrics["bytes_sent"] = sparse_bytes(idx, values)
            metrics["is_sparse"] = True
            return [idx.astype(np.int64), values], self.n_train, metrics
        else:
            metrics["bytes_sent"] = dense_bytes(new_params)
            metrics["is_sparse"] = False
            return new_params, self.n_train, metrics

    def evaluate(self, parameters, config):
        self._set_parameters(parameters)
        self.model.eval()
        if self.val_loader is None or len(self.val_loader.dataset) == 0:
            return 0.0, 0, {"macro_f1": 0.0}
        all_preds, all_true = [], []
        total_loss, n_batches = 0.0, 0
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            for xb, yb in self.val_loader:
                out = self.model(xb)
                loss = criterion(out, yb)
                total_loss += loss.item()
                n_batches += 1
                preds = out.argmax(dim=1).cpu().numpy()
                all_preds.append(preds)
                all_true.append(yb.cpu().numpy())
        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_true)
        m = compute_metrics(y_true, y_pred, self.class_names)
        return total_loss / max(1, n_batches), len(y_true), {"macro_f1": m["macro_f1"], "accuracy": m["accuracy"]}
