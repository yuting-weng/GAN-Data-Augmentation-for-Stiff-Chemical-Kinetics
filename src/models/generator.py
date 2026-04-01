from __future__ import annotations

import torch
import torch.nn as nn

from src.models.mlp_blocks import make_mlp


class Generator(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        condition_dim: int,
        output_dim: int,
        hidden_dims: list[int],
        activation: str = "gelu",
        condition_encoder_cfg: dict | None = None,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        cfg = dict(condition_encoder_cfg or {})
        self.use_condition_encoder = bool(cfg.get("enabled", False)) and condition_dim > 0
        self.condition_encoder = None
        cond_out_dim = condition_dim
        if self.use_condition_encoder:
            enc_hidden = list(cfg.get("hidden_dims", [64, 64]))
            enc_act = str(cfg.get("activation", activation))
            self.condition_encoder = make_mlp(
                input_dim=condition_dim,
                hidden_dims=enc_hidden,
                output_dim=latent_dim,
                activation=enc_act,
                use_spectral_norm=False,
            )
            cond_out_dim = latent_dim
        in_dim = latent_dim + cond_out_dim
        self.net = make_mlp(in_dim, hidden_dims, output_dim, activation=activation, use_spectral_norm=False)

    def forward(self, z: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        if self.condition_dim > 0:
            if c is None:
                raise ValueError("condition_dim > 0 时必须传入条件 c")
            if self.use_condition_encoder and self.condition_encoder is not None:
                c = self.condition_encoder(c)
            x = torch.cat([z, c], dim=1)
        else:
            x = z
        return self.net(x)
