from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    BatchNorm,
    GINEConv,
    global_add_pool,
    global_mean_pool,
)
from torch_geometric.nn.aggr import AttentionalAggregation


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class GNNRegressor(nn.Module):
    def __init__(
        self,
        node_features: int,
        edge_features: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.15,
        pooling: str = "mean",
        descriptor_dim: int = 0,
        use_jk: bool = True,
        drop_path_rate: float = 0.0,
        use_augmentation: bool = False,
        augmentation_drop_edge: float = 0.05,
    ):
        super().__init__()
        if pooling not in {"mean", "add", "attention"}:
            raise ValueError("pooling must be 'mean', 'add', or 'attention'")

        self.pooling = pooling
        self.dropout = dropout
        self.descriptor_dim = descriptor_dim
        self.use_jk = use_jk

        from src.models.augmentation import GraphAugmentor

        self.graph_augmentor = GraphAugmentor(
            drop_edge_prob=augmentation_drop_edge,
            enabled=use_augmentation,
        )

        self.node_proj = nn.Linear(node_features, hidden_dim)
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.drop_paths = nn.ModuleList()

        for i in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINEConv(mlp, edge_dim=hidden_dim))
            self.norms.append(BatchNorm(hidden_dim))
            layer_drop_rate = drop_path_rate * (i / max(num_layers - 1, 1))
            self.drop_paths.append(DropPath(layer_drop_rate))

        pool_dim = hidden_dim * num_layers if use_jk else hidden_dim
        if self.pooling == "attention":
            gate_nn = nn.Sequential(
                nn.Linear(pool_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
            self.attn_pool = AttentionalAggregation(gate_nn)

        head_in = pool_dim
        if descriptor_dim > 0:
            head_in += descriptor_dim

        h = max(head_in // 2, 64)
        h2 = max(h // 2, 16)
        self.head = nn.Sequential(
            nn.Linear(head_in, h),
            nn.BatchNorm1d(h),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h, h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, 1),
        )

    def _pool(self, x, batch):
        if self.pooling == "add":
            return global_add_pool(x, batch)
        elif self.pooling == "attention":
            return self.attn_pool(x, batch)
        return global_mean_pool(x, batch)

    def forward(self, data):
        data = self.graph_augmentor(data)

        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )
        x = self.node_proj(x)
        edge_attr = self.edge_encoder(edge_attr)

        if self.use_jk:
            layer_outputs = []
            for conv, norm, drop_path in zip(self.convs, self.norms, self.drop_paths):
                residual = x
                x = conv(x, edge_index, edge_attr)
                x = norm(x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = residual + drop_path(x)
                layer_outputs.append(x)
            x = torch.cat(layer_outputs, dim=-1)
        else:
            for conv, norm, drop_path in zip(self.convs, self.norms, self.drop_paths):
                residual = x
                x = conv(x, edge_index, edge_attr)
                x = norm(x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = residual + drop_path(x)

        x = self._pool(x, batch)

        if self.descriptor_dim > 0 and hasattr(data, "desc"):
            desc = data.desc
            x = torch.cat([x, desc], dim=-1)

        return self.head(x)


__all__ = ["GNNRegressor", "DropPath"]
