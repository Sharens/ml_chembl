from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    BatchNorm,
    GINEConv,
    global_add_pool,
    global_mean_pool,
)
from torch_geometric.nn.aggr import AttentionalAggregation


class GNNRegressor(nn.Module):
    """Graph Neural Network for pIC50 regression using GINEConv message passing.

    Edge attributes (bond type, stereo, conjugated, in-ring) are fully propagated
    through GINEConv layers via the edge_dim parameter, giving the model access
    to rich bond-level information during message passing.
    """

    def __init__(
        self,
        node_features: int,
        edge_features: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.15,
        pooling: str = "mean",
    ):
        super().__init__()
        if pooling not in {"mean", "add", "attention"}:
            raise ValueError("pooling must be 'mean', 'add', or 'attention'")

        self.pooling = pooling
        self.dropout = dropout
        self.node_proj = nn.Linear(node_features, hidden_dim)
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINEConv(mlp, edge_dim=hidden_dim))
            self.norms.append(BatchNorm(hidden_dim))

        if self.pooling == "attention":
            gate_nn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
            self.attn_pool = AttentionalAggregation(gate_nn)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _pool(self, x, batch):
        if self.pooling == "add":
            return global_add_pool(x, batch)
        elif self.pooling == "attention":
            return self.attn_pool(x, batch)
        return global_mean_pool(x, batch)

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )
        x = self.node_proj(x)
        edge_attr = self.edge_encoder(edge_attr)

        for conv, norm in zip(self.convs, self.norms):
            residual = x
            x = conv(x, edge_index, edge_attr)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + residual

        x = self._pool(x, batch)
        return self.head(x)
