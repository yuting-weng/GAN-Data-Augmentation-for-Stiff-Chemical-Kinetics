from __future__ import annotations

import torch


def sample_latent(batch_size: int, latent_dim: int, device: torch.device) -> torch.Tensor:
    return torch.randn(batch_size, latent_dim, device=device)


def mix_real_fake_for_quality(real: torch.Tensor, fake: torch.Tensor, real_ratio: float = 0.5) -> tuple[torch.Tensor, torch.Tensor]:
    b = real.shape[0]
    n_real = int(b * real_ratio)
    n_fake = b - n_real
    real_sel = real[torch.randperm(real.shape[0], device=real.device)[:n_real]]
    fake_sel = fake[torch.randperm(fake.shape[0], device=fake.device)[:n_fake]]
    x = torch.cat([real_sel, fake_sel], dim=0)
    y = torch.cat(
        [
            torch.ones(n_real, device=real.device, dtype=torch.float32),
            torch.zeros(n_fake, device=real.device, dtype=torch.float32),
        ],
        dim=0,
    )
    perm = torch.randperm(x.shape[0], device=real.device)
    return x[perm], y[perm]


def select_mixed_samples(real: torch.Tensor, fake: torch.Tensor, real_ratio: float = 0.5) -> torch.Tensor:
    b = real.shape[0]
    n_real = int(b * real_ratio)
    n_fake = b - n_real
    real_sel = real[torch.randperm(real.shape[0], device=real.device)[:n_real]]
    fake_sel = fake[torch.randperm(fake.shape[0], device=fake.device)[:n_fake]]
    x = torch.cat([real_sel, fake_sel], dim=0)
    perm = torch.randperm(x.shape[0], device=real.device)
    return x[perm]
