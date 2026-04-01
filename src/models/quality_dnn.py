from __future__ import annotations

import torch
import torch.nn as nn

from src.models.mlp_blocks import make_mlp


class QualityDNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        mode: str = "classifier",
        activation: str = "gelu",
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.output_dim = output_dim
        self.net = make_mlp(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=output_dim, activation=activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        if self.output_dim == 1:
            return out.view(-1)
        return out
