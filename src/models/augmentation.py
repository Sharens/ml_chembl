from __future__ import annotations

import torch


class FingerprintAugmentor:
    def __init__(self, noise_prob: float = 0.02, enabled: bool = True):
        self.noise_prob = noise_prob
        self.enabled = enabled
        self.training = True

    def train(self, mode: bool = True):
        self.training = mode

    def eval(self):
        self.training = False

    def __call__(self, fp_tensor: torch.Tensor) -> torch.Tensor:
        if not self.enabled or not self.training:
            return fp_tensor
        mask = torch.rand_like(fp_tensor) < self.noise_prob
        return torch.where(mask, 1.0 - fp_tensor, fp_tensor)


class DescriptorAugmentor:
    def __init__(self, noise_std: float = 0.05, enabled: bool = True):
        self.noise_std = noise_std
        self.enabled = enabled
        self.training = True

    def train(self, mode: bool = True):
        self.training = mode

    def eval(self):
        self.training = False

    def __call__(self, desc_tensor: torch.Tensor) -> torch.Tensor:
        if not self.enabled or not self.training:
            return desc_tensor
        noise = torch.randn_like(desc_tensor) * self.noise_std
        return desc_tensor + noise


class GraphAugmentor:
    def __init__(self, drop_edge_prob: float = 0.05, enabled: bool = True):
        self.drop_edge_prob = drop_edge_prob
        self.enabled = enabled
        self.training = True

    def train(self, mode: bool = True):
        self.training = mode

    def eval(self):
        self.training = False

    def __call__(self, data):
        if not self.enabled or not self.training:
            return data
        num_edges = data.edge_index.size(1)
        mask = (
            torch.rand(num_edges, device=data.edge_index.device) > self.drop_edge_prob
        )
        data.edge_index = data.edge_index[:, mask]
        data.edge_attr = data.edge_attr[mask]
        return data


__all__ = ["FingerprintAugmentor", "DescriptorAugmentor", "GraphAugmentor"]
