from __future__ import annotations

import torch.nn as nn

from src.models.gnn import GNNRegressor


class MLPBaseline(nn.Module):
    def __init__(self, input_size=2048):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.network(x)


__all__ = ["MLPBaseline", "GNNRegressor"]
