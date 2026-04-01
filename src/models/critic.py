from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

from src.models.mlp_blocks import _activation


class Critic(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        activation: str = "gelu",
        use_spectral_norm: bool = True,
        minibatch_discrimination_cfg: dict | None = None,
    ) -> None:
        super().__init__()
        dims = [input_dim, *list(hidden_dims)]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            linear = nn.Linear(dims[i], dims[i + 1])
            if use_spectral_norm:
                linear = spectral_norm(linear)
            layers.append(linear)
            layers.append(_activation(activation))
        self.feature_net = nn.Sequential(*layers) if layers else nn.Identity()
        feat_dim = dims[-1]

        cfg = dict(minibatch_discrimination_cfg or {})
        self.use_minibatch_discrimination = bool(cfg.get("enabled", False))
        self.minibatch_stat = str(cfg.get("stat", "mean_abs_diff"))
        out_in_dim = feat_dim + (1 if self.use_minibatch_discrimination else 0)
        out_linear = nn.Linear(out_in_dim, 1)
        if use_spectral_norm:
            out_linear = spectral_norm(out_linear)
        self.out = out_linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.feature_net(x)
        if self.use_minibatch_discrimination:
            if self.minibatch_stat != "mean_abs_diff":
                raise ValueError(f"不支持的 minibatch 统计方式: {self.minibatch_stat}")
            if feat.size(0) > 1:
                pair = torch.mean(torch.abs(feat[:, None, :] - feat[None, :, :]), dim=2)
                mb = (torch.sum(pair, dim=1, keepdim=True) / float(feat.size(0) - 1)).to(feat.dtype)
            else:
                mb = torch.zeros((feat.size(0), 1), device=feat.device, dtype=feat.dtype)
            feat = torch.cat([feat, mb], dim=1)
        return self.out(feat).view(-1)
