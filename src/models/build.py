import torch.nn as nn

from src.models.cnn1d import CNN1D
from src.models.mlp import MLP


def build_model(name: str, n_features: int, n_classes: int) -> nn.Module:
    if name == "mlp":
        return MLP(n_features, n_classes)
    if name == "cnn1d":
        return CNN1D(n_features, n_classes)
    raise ValueError(f"Unknown model: {name}")
