"""DP-SGD via Opacus, applied client-side during local FL training.

Flagship privacy config (ROADMAP.md §0/§1): FL alone is not privacy-preserving, so the
"privacy-preserving" claim in the project title is backed by DP-SGD with a reported
epsilon, not by federation alone. Fixed (noise_multiplier, max_grad_norm) per round is
used rather than a target-epsilon-per-round API, because the meaningful number is the
*cumulative* epsilon spent by a client across all rounds it participates in, which
requires composing across rounds via a persisted accountant, not resetting each round.
"""
import pickle
from pathlib import Path

import torch
from opacus import PrivacyEngine
from opacus.accountants import RDPAccountant
from opacus.validators import ModuleValidator

ACCOUNTANT_DIR = Path("data/partitions/_dp_accountants")


def make_dp_compatible(model: torch.nn.Module) -> torch.nn.Module:
    """Opacus per-sample gradients require BatchNorm -> GroupNorm (BatchNorm mixes
    information across the batch, which breaks the per-sample gradient guarantee DP relies on)."""
    if not ModuleValidator.is_valid(model):
        model = ModuleValidator.fix(model)
    return model


def _accountant_path(client_id: int, run_id: str) -> Path:
    return ACCOUNTANT_DIR / run_id / f"client_{client_id}.pkl"


def load_accountant_history(client_id: int, run_id: str) -> list:
    path = _accountant_path(client_id, run_id)
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return []


def save_accountant_history(client_id: int, run_id: str, history: list) -> None:
    path = _accountant_path(client_id, run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(history, f)


def train_with_dp(model, optimizer, train_loader, epochs: int, noise_multiplier: float,
                   max_grad_norm: float, client_id: int, run_id: str, delta: float = 1e-5):
    """Wraps model/optimizer/loader with Opacus for one round of local training, restores
    and persists the client's cumulative privacy accountant across rounds, and returns the
    cumulative epsilon spent so far by this client."""
    model = make_dp_compatible(model)
    privacy_engine = PrivacyEngine(accountant="rdp")
    privacy_engine.accountant.history = load_accountant_history(client_id, run_id)

    dp_model, dp_optimizer, dp_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )

    criterion = torch.nn.CrossEntropyLoss()
    dp_model.train()
    total_loss, n_batches = 0.0, 0
    for _ in range(epochs):
        for xb, yb in dp_loader:
            dp_optimizer.zero_grad()
            out = dp_model(xb)
            loss = criterion(out, yb)
            loss.backward()
            dp_optimizer.step()
            total_loss += loss.item()
            n_batches += 1

    save_accountant_history(client_id, run_id, privacy_engine.accountant.history)
    epsilon = privacy_engine.get_epsilon(delta=delta) if privacy_engine.accountant.history else 0.0

    avg_loss = total_loss / max(1, n_batches)
    return dp_model, avg_loss, epsilon
