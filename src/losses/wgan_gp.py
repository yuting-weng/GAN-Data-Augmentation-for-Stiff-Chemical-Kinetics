from __future__ import annotations

import torch


def gradient_penalty(critic, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    batch_size = real.size(0)
    alpha = torch.rand(batch_size, 1, device=real.device)
    alpha = alpha.expand_as(real)
    interpolates = alpha * real + (1.0 - alpha) * fake
    interpolates.requires_grad_(True)
    critic_scores = critic(interpolates)
    grads = torch.autograd.grad(
        outputs=critic_scores,
        inputs=interpolates,
        grad_outputs=torch.ones_like(critic_scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grads = grads.view(batch_size, -1)
    return ((grads.norm(2, dim=1) - 1.0) ** 2).mean()


def critic_loss_wgan_gp(
    critic,
    real: torch.Tensor,
    fake: torch.Tensor,
    gp_lambda: float = 10.0,
) -> tuple[torch.Tensor, dict]:
    real_score = critic(real).mean()
    fake_score = critic(fake).mean()
    gp = gradient_penalty(critic, real, fake)
    loss = fake_score - real_score + gp_lambda * gp
    metrics = {"real_score": real_score.item(), "fake_score": fake_score.item(), "gp": gp.item()}
    return loss, metrics


def generator_loss_wgan(critic, fake: torch.Tensor) -> torch.Tensor:
    return -critic(fake).mean()
