from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Tuple

import numpy as np
import torch


@lru_cache(maxsize=4)
def _get_ct_and_species(mechanism_path: str):
    import cantera as ct

    gas = ct.Solution(mechanism_path)
    return ct, tuple(gas.species_names), float(gas.P)


def _normalize_y(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    y = np.clip(y, 0.0, None)
    s = float(y.sum())
    if not np.isfinite(s) or s <= 0.0:
        return np.full_like(y, 1.0 / max(1, y.shape[0]))
    return y / s


def _project_tensor(x: torch.Tensor, target_dim: int) -> torch.Tensor:
    if x.shape[1] >= target_dim:
        return x[:, :target_dim]
    pad = torch.zeros(x.shape[0], target_dim - x.shape[1], device=x.device, dtype=x.dtype)
    return torch.cat([x, pad], dim=1)


def get_true_prediction(
    x_batch: torch.Tensor,
    target_dim: int,
    mechanism_path: str = "mechanism/Burke2012_s9r23.yaml",
    time_step: float = 1e-7,
    reference_pressure: float | None = None,
) -> Tuple[torch.Tensor, str]:
    if x_batch.ndim != 2:
        raise ValueError(f"x_batch 维度错误: {x_batch.shape}")
    if target_dim <= 0:
        raise ValueError(f"target_dim 必须为正数，当前为 {target_dim}")
    if time_step <= 0:
        raise ValueError(f"time_step 必须为正数，当前为 {time_step}")
    mech = str(Path(mechanism_path))
    ct, species_names, default_pressure = _get_ct_and_species(mech)
    n_species = len(species_names)

    x_cpu = x_batch.detach().to("cpu")
    x_np = x_cpu.numpy().astype(np.float64, copy=False)
    n = x_np.shape[0]
    y_true = np.zeros((n, target_dim), dtype=np.float32)
    failed = 0
    p_ref = default_pressure if reference_pressure is None else float(reference_pressure)

    for i in range(n):
        row = x_np[i]
        try:
            if row.shape[0] == n_species + 2:
                t_old = float(row[0])
                p_old = float(row[1])
                y_old = row[2 : 2 + n_species]
            elif row.shape[0] == n_species + 1:
                t_old = float(row[0])
                p_old = p_ref
                y_old = row[1 : 1 + n_species]
            else:
                raise ValueError(f"输入维度不支持: {row.shape[0]}，期望 {n_species+1} 或 {n_species+2}")
            gas = ct.Solution(mech)
            gas.TPY = t_old, p_old, _normalize_y(y_old)
            r = ct.IdealGasConstPressureReactor(gas, name="R1", clone=False)
            sim = ct.ReactorNet([r])
            sim.advance(float(time_step))
            y_next = gas.Y.astype(np.float32)
            if target_dim <= n_species:
                y_out = y_next[:target_dim]
            else:
                y_out = np.concatenate([y_next, np.zeros((target_dim - n_species,), dtype=np.float32)], axis=0)
        except Exception:
            failed += 1
            y_fallback = _project_tensor(torch.from_numpy(row[None, :]).float(), target_dim=target_dim)[0].numpy().astype(np.float32)
            y_out = y_fallback
        y_true[i] = y_out

    y_tensor = torch.from_numpy(y_true).to(device=x_batch.device, dtype=x_batch.dtype)
    source = f"oracle_cantera_single_step_ok={n-failed}_fail={failed}_mech={Path(mech).name}"
    return y_tensor, source
