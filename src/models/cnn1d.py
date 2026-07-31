import torch
import torch.nn as nn


class CNN1D(nn.Module):
    """Lightweight 1D-CNN classifier treating flow features as a 1D signal."""

    def __init__(self, n_features: int, n_classes: int, channels: list[int] = (32, 64)):
        super().__init__()
        convs = []
        prev = 1
        for ch in channels:
            convs += [
                nn.Conv1d(prev, ch, kernel_size=3, padding=1),
                nn.BatchNorm1d(ch),
                nn.ReLU(),
                nn.MaxPool1d(2),
            ]
            prev = ch
        self.conv = nn.Sequential(*convs)
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_features)
            flat_dim = self.conv(dummy).flatten(1).shape[1]
        self.head = nn.Sequential(
            nn.Linear(flat_dim, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.flatten(1)
        return self.head(x)
