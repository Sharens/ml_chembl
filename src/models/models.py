from __future__ import annotations

import torch.nn as nn

from src.models.gnn import GNNRegressor


class MLPBaseline(nn.Module):
    def __init__(
        self,
        input_size=2048,
        use_batch_norm: bool = False,
        hidden_sizes: list[int] | None = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [512, 128]

        layers = []
        prev = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h

        layers.append(nn.Linear(prev, 1))
        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.network(x)


__all__ = ["MLPBaseline", "GNNRegressor"]
