from __future__ import annotations

from typing import Iterable

import torch.nn as nn
from torch.nn.utils import spectral_norm


def _activation(name: str) -> nn.Module:
    key = name.lower()
    if key == "relu":
        return nn.ReLU(inplace=True)
    if key == "gelu":
        return nn.GELU()
    if key == "leaky_relu":
        return nn.LeakyReLU(0.2, inplace=True)
    raise ValueError(f"未知激活函数: {name}")


def make_mlp(
    input_dim: int,
    hidden_dims: Iterable[int],
    output_dim: int,
    activation: str = "gelu",
    use_spectral_norm: bool = False,
    final_activation: nn.Module | None = None,
) -> nn.Sequential:
    dims = [input_dim, *list(hidden_dims), output_dim]
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        linear = nn.Linear(dims[i], dims[i + 1])
        if use_spectral_norm:
            linear = spectral_norm(linear)
        layers.append(linear)
        if i < len(dims) - 2:
            layers.append(_activation(activation))
        elif final_activation is not None:
            layers.append(final_activation)
    return nn.Sequential(*layers)
