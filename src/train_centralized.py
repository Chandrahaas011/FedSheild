"""Centralized training baseline — the upper-bound comparison point for every FL config
(ROADMAP.md Phase 3, step 1)."""
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.metrics import compute_metrics
from src.models.build import build_model
from src.utils.checkpoint import load_checkpoint, save_checkpoint
from src.utils.seed import set_seed


def train_centralized(
    X_train, y_train, X_val, y_val, X_test, y_test,
    model_name: str, n_features: int, n_classes: int, class_names: list[str],
    epochs: int, lr: float, batch_size: int, seed: int = 42,
    checkpoint_path=None, patience: int = 5,
):
    set_seed(seed)
    model = build_model(model_name, n_features, n_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=batch_size, shuffle=True,
    )

    start_epoch, best_val_f1, epochs_no_improve, best_state = 0, -1.0, 0, None
    if checkpoint_path is not None:
        ckpt = load_checkpoint(checkpoint_path)
        if ckpt is not None:
            model.load_state_dict(ckpt["model_state"])
            start_epoch = ckpt["step"] + 1
            best_val_f1 = ckpt["extra"].get("best_val_f1", -1.0)
            epochs_no_improve = ckpt["extra"].get("epochs_no_improve", 0)

    t0 = time.time()
    last_epoch = start_epoch - 1
    for epoch in range(start_epoch, epochs):
        last_epoch = epoch
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_preds = model(torch.from_numpy(X_val)).argmax(dim=1).numpy()
        val_metrics = compute_metrics(y_val, val_preds, class_names)

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            epochs_no_improve = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1

        if checkpoint_path is not None:
            save_checkpoint(
                checkpoint_path, epoch, model.state_dict(),
                extra={"best_val_f1": best_val_f1, "epochs_no_improve": epochs_no_improve},
            )
        if epochs_no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        test_preds = model(torch.from_numpy(X_test)).argmax(dim=1).numpy()
    test_metrics = compute_metrics(y_test, test_preds, class_names)
    test_metrics["train_seconds"] = time.time() - t0
    test_metrics["epochs_run"] = last_epoch + 1

    return model, test_metrics
