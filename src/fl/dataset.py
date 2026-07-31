import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

PROCESSED_DIR = Path("data/processed")


def load_processed(processed_dir: Path = PROCESSED_DIR):
    with open(processed_dir / "metadata.json") as f:
        metadata = json.load(f)
    splits = {}
    for name in ["train", "val", "test"]:
        df = pd.read_parquet(processed_dir / f"{name}.parquet")
        y = df["label"].to_numpy(dtype=np.int64)
        X = df.drop(columns=["label"]).to_numpy(dtype=np.float32)
        splits[name] = (X, y)
    return splits, metadata


def subsample(X: np.ndarray, y: np.ndarray, frac: float, seed: int = 42):
    if frac >= 1.0:
        return X, y
    rng = np.random.default_rng(seed)
    n = int(len(y) * frac)
    idx = rng.choice(len(y), size=n, replace=False)
    return X[idx], y[idx]


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    drop_last = shuffle and len(ds) > batch_size
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)


def client_loaders(X: np.ndarray, y: np.ndarray, client_indices: list[np.ndarray], client_id: int,
                    batch_size: int, val_frac: float = 0.15, seed: int = 42):
    idx = client_indices[client_id]
    if len(idx) == 0:
        return None, None, 0
    rng = np.random.default_rng(seed + client_id)
    idx = rng.permutation(idx)
    n_val = max(1, int(len(idx) * val_frac)) if len(idx) > 5 else 0
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    train_loader = make_loader(X[train_idx], y[train_idx], batch_size, shuffle=True)
    val_loader = make_loader(X[val_idx], y[val_idx], batch_size, shuffle=False) if n_val > 0 else None
    return train_loader, val_loader, len(train_idx)
