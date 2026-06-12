from __future__ import annotations

import torch.nn as nn

from src.models.gnn import GNNRegressor


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.3, use_batch_norm: bool = True):
        super().__init__()
        layers = []
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dim, dim))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dim, dim))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.block(x)


class MLPBaseline(nn.Module):
    def __init__(
        self,
        input_size=2048,
        use_batch_norm: bool = False,
        hidden_sizes: list[int] | None = None,
        dropout: float = 0.2,
        num_residual_blocks: int = 0,
        use_augmentation: bool = False,
        augmentation_noise_prob: float = 0.02,
    ):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [512, 128]

        from src.models.augmentation import FingerprintAugmentor

        self.augmentor = FingerprintAugmentor(
            noise_prob=augmentation_noise_prob,
            enabled=use_augmentation,
        )

        layers = []
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        self.residual_blocks = nn.ModuleList()
        if num_residual_blocks > 0:
            for _ in range(num_residual_blocks):
                self.residual_blocks.append(
                    ResidualBlock(hidden_sizes[0], dropout, use_batch_norm)
                )

        prev = hidden_sizes[0]
        for h in hidden_sizes[1:]:
            layers.append(nn.Linear(prev, h))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h

        layers.append(nn.Linear(prev, 1))
        self.post_residual = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.augmentor(x)
        out = self.post_residual[0](x)
        for block in self.residual_blocks:
            out = block(out)
        for layer in self.post_residual[1:]:
            out = layer(out)
        return out


__all__ = ["MLPBaseline", "GNNRegressor", "ResidualBlock"]
