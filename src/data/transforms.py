from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch


def _boxcox_forward(x: np.ndarray, lam: np.ndarray, eps: float) -> np.ndarray:
    x = np.maximum(x, eps)
    out = np.empty_like(x, dtype=np.float64)
    near_zero = np.isclose(lam, 0.0)
    out[:, near_zero] = np.log(x[:, near_zero])
    out[:, ~near_zero] = (np.power(x[:, ~near_zero], lam[~near_zero]) - 1.0) / lam[~near_zero]
    return out


def _boxcox_inverse(y: np.ndarray, lam: np.ndarray) -> np.ndarray:
    out = np.empty_like(y, dtype=np.float64)
    near_zero = np.isclose(lam, 0.0)
    out[:, near_zero] = np.exp(y[:, near_zero])
    base = lam[~near_zero] * y[:, ~near_zero] + 1.0
    base = np.maximum(base, 1e-12)
    out[:, ~near_zero] = np.power(base, 1.0 / lam[~near_zero])
    return out


@dataclass
class BCTStandardizer:
    use_bct: bool = True
    bct_epsilon: float = 1e-6
    standardize: bool = True
    bct_feature_mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        self._is_fitted = False
        self.shift: np.ndarray | None = None
        self.lam: np.ndarray | None = None
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.mask: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "BCTStandardizer":
        x = np.asarray(x, dtype=np.float64)
        n_features = x.shape[1]
        if self.bct_feature_mask is None:
            self.mask = np.ones(n_features, dtype=bool)
        else:
            m = np.asarray(self.bct_feature_mask, dtype=bool).reshape(-1)
            if m.shape[0] != n_features:
                raise ValueError(f"bct_feature_mask 长度应为 {n_features}，实际 {m.shape[0]}")
            self.mask = m
        self.shift = np.maximum(0.0, -x.min(axis=0) + self.bct_epsilon)
        x_pos = x + self.shift
        if self.use_bct and bool(np.any(self.mask)):
            lam_candidates = np.linspace(-2.0, 2.0, 81, dtype=np.float64)
            best_lam = np.zeros(n_features, dtype=np.float64)
            x_clip = np.maximum(x_pos, self.bct_epsilon)
            for j in range(n_features):
                if not self.mask[j]:
                    best_lam[j] = 0.0
                    continue
                col = x_clip[:, j]
                best_score = np.inf
                best = 0.0
                for lam in lam_candidates:
                    if abs(lam) < 1e-12:
                        y = np.log(col)
                    else:
                        y = (np.power(col, lam) - 1.0) / lam
                    score = np.var(y)
                    if score < best_score:
                        best_score = score
                        best = lam
                best_lam[j] = best
            self.lam = best_lam
            transformed = x_pos.copy()
            transformed[:, self.mask] = _boxcox_forward(
                x_pos[:, self.mask],
                self.lam[self.mask],
                self.bct_epsilon,
            )
        else:
            self.lam = np.zeros(n_features, dtype=np.float64)
            transformed = x_pos

        if self.standardize:
            self.mean = transformed.mean(axis=0)
            self.std = transformed.std(axis=0)
            self.std[self.std < 1e-8] = 1.0
        else:
            self.mean = np.zeros(n_features, dtype=np.float64)
            self.std = np.ones(n_features, dtype=np.float64)

        self._is_fitted = True
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        self._check_fitted()
        x = np.asarray(x, dtype=np.float64)
        x_pos = x + self.shift
        if self.use_bct and bool(np.any(self.mask)):
            y = x_pos.copy()
            y[:, self.mask] = _boxcox_forward(
                x_pos[:, self.mask],
                self.lam[self.mask],
                self.bct_epsilon,
            )
        else:
            y = x_pos
        y = (y - self.mean) / self.std
        return y.astype(np.float32)

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        self._check_fitted()
        y = np.asarray(y, dtype=np.float64)
        x = y * self.std + self.mean
        if self.use_bct and bool(np.any(self.mask)):
            x_inv = x.copy()
            x_inv[:, self.mask] = _boxcox_inverse(x[:, self.mask], self.lam[self.mask])
            x = x_inv
        x = x - self.shift
        return x.astype(np.float32)

    def inverse_transform_torch(self, y: torch.Tensor) -> torch.Tensor:
        self._check_fitted()
        mean = torch.as_tensor(self.mean, dtype=y.dtype, device=y.device)
        std = torch.as_tensor(self.std, dtype=y.dtype, device=y.device)
        shift = torch.as_tensor(self.shift, dtype=y.dtype, device=y.device)
        lam = torch.as_tensor(self.lam, dtype=y.dtype, device=y.device)
        x = y * std + mean
        if self.use_bct and bool(np.any(self.mask)):
            mask = torch.as_tensor(self.mask, dtype=torch.bool, device=y.device)
            x_mask = x[:, mask]
            lam_mask = lam[mask]
            near_zero = torch.isclose(lam_mask, torch.zeros_like(lam_mask), atol=1e-7)
            out_mask = x_mask.clone()
            if near_zero.any():
                out_mask[:, near_zero] = torch.exp(x_mask[:, near_zero])
            if (~near_zero).any():
                l = lam_mask[~near_zero]
                base = l * x_mask[:, ~near_zero] + 1.0
                base = torch.clamp(base, min=1e-12)
                out_mask[:, ~near_zero] = torch.pow(base, 1.0 / l)
            x = x.clone()
            x[:, mask] = out_mask
        x = x - shift
        return x

    def state_dict(self) -> Dict[str, np.ndarray]:
        self._check_fitted()
        return {
            "shift": self.shift,
            "lam": self.lam,
            "mean": self.mean,
            "std": self.std,
            "bct_feature_mask": self.mask.astype(np.int8),
        }

    def save(self, file_path: str | Path) -> None:
        np.savez(Path(file_path), **self.state_dict())

    def load(self, file_path: str | Path) -> "BCTStandardizer":
        z = np.load(Path(file_path))
        self.shift = z["shift"]
        self.lam = z["lam"]
        self.mean = z["mean"]
        self.std = z["std"]
        if "bct_feature_mask" in z:
            self.mask = z["bct_feature_mask"].astype(bool)
        else:
            self.mask = np.ones_like(self.lam, dtype=bool)
        self._is_fitted = True
        return self

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("BCTStandardizer 尚未 fit。")
