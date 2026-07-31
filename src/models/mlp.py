import torch
import torch.nn as nn


class MLP(nn.Module):
    """Lightweight MLP classifier for tabular NIDS flow features."""

    def __init__(self, n_features: int, n_classes: int, hidden_dims: list[int] = (128, 64)):
        super().__init__()
        layers = []
        prev = n_features
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.2)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
