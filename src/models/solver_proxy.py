from __future__ import annotations

import torch
import torch.nn as nn

from src.models.mlp_blocks import make_mlp


class SolverProxy(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int] | None = None, activation: str = "gelu") -> None:
        super().__init__()
        hidden_dims = hidden_dims or [256, 256, 128, 128]
        self.net_n = make_mlp(input_dim, hidden_dims, input_dim, activation=activation)
        self.net_o = make_mlp(input_dim, hidden_dims, input_dim, activation=activation)
        self._init_oracle_bias()

    def _init_oracle_bias(self) -> None:
        with torch.no_grad():
            for p in self.net_o.parameters():
                p.add_(0.01 * torch.randn_like(p))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.net_n(x), self.net_o(x)

    @torch.no_grad()
    def error_scalar(self, x: torch.Tensor) -> torch.Tensor:
        n, o = self.forward(x)
        return torch.mean(torch.abs(n - o), dim=1)
