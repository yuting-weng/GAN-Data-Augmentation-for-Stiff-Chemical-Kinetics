from __future__ import annotations

import torch


def mass_conservation_loss(fake_raw: torch.Tensor, ref_raw: torch.Tensor) -> torch.Tensor:
    fake_mass = fake_raw.sum(dim=1)
    ref_mass = ref_raw.sum(dim=1)
    return torch.mean((fake_mass - ref_mass) ** 2)


def non_negative_loss(fake_raw: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.relu(-fake_raw) ** 2)


def species_bounds_hinge_loss(
    fake_species_raw: torch.Tensor,
    species_min_raw: torch.Tensor,
    species_max_raw: torch.Tensor,
    use_hinge: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    upper = torch.relu(fake_species_raw - species_max_raw)
    lower = torch.relu(species_min_raw - fake_species_raw)
    if use_hinge:
        loss = torch.mean(upper + lower)
    else:
        loss = torch.mean((upper + lower) ** 2)
    violate_mask = (upper > 0) | (lower > 0)
    violate_ratio = violate_mask.float().mean()
    return loss, violate_ratio


def physics_loss(
    fake_raw: torch.Tensor,
    ref_raw: torch.Tensor,
    w_mass: float = 1.0,
    w_nonneg: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    l_mass = mass_conservation_loss(fake_raw, ref_raw)
    l_nonneg = non_negative_loss(fake_raw)
    total = w_mass * l_mass + w_nonneg * l_nonneg
    return total, {"mass": l_mass.item(), "nonneg": l_nonneg.item()}
