from __future__ import annotations

import torch.nn as nn

from src.models.gnn import GNNRegressor


class MLPBaseline(nn.Module):
    def __init__(self, input_size=2048, use_batch_norm: bool = False):
        super().__init__()
        layers = [
            nn.Linear(input_size, 512),
        ]
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(512))
        layers.extend([nn.ReLU(), nn.Dropout(0.2)])

        layers.append(nn.Linear(512, 128))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(128))
        layers.extend([nn.ReLU(), nn.Dropout(0.2)])

        layers.append(nn.Linear(128, 1))

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
