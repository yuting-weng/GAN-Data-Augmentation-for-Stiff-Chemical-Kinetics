from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.transforms import BCTStandardizer


class NumpyStateDataset(Dataset):
    def __init__(self, data: np.ndarray):
        self.data = torch.from_numpy(data.astype(np.float32))

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


class NumpyPairDataset(Dataset):
    def __init__(self, x_data: np.ndarray, y_data: np.ndarray):
        self.x_data = torch.from_numpy(x_data.astype(np.float32))
        self.y_data = torch.from_numpy(y_data.astype(np.float32))

    def __len__(self) -> int:
        return self.x_data.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x_data[idx], self.y_data[idx]


@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    transform: BCTStandardizer
    feature_dim: int
    train_raw: np.ndarray
    val_raw: np.ndarray


@dataclass
class PairedDataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    input_transform: BCTStandardizer
    target_transform: BCTStandardizer
    input_dim: int
    target_dim: int
    train_x_raw: np.ndarray
    train_y_raw: np.ndarray
    val_x_raw: np.ndarray
    val_y_raw: np.ndarray


def create_data_bundle(
    npy_path: str,
    batch_size: int,
    val_ratio: float = 0.1,
    seed: int = 42,
    num_workers: int = 0,
    subset_size: Optional[int] = None,
    use_bct: bool = True,
    bct_epsilon: float = 1e-6,
    standardize: bool = True,
    disable_input_dim0_bct: bool = False,
) -> DataBundle:
    path = Path(npy_path)
    if not path.exists():
        raise FileNotFoundError(f"数据文件不存在: {path}")
    data = np.load(path).astype(np.float32)
    if data.ndim != 2:
        raise ValueError(f"期望二维数组，实际 shape={data.shape}")

    if subset_size is not None and subset_size > 0:
        data = data[: min(subset_size, len(data))]

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(data))
    val_size = max(1, int(len(data) * val_ratio))
    val_idx = idx[:val_size]
    train_idx = idx[val_size:]
    if len(train_idx) == 0:
        raise ValueError("训练集为空，请增大数据量或减小 val_ratio")

    train_raw = data[train_idx]
    val_raw = data[val_idx]

    bct_mask = None
    if disable_input_dim0_bct and train_raw.shape[1] >= 1:
        bct_mask = np.ones(train_raw.shape[1], dtype=bool)
        bct_mask[0] = False
    transform = BCTStandardizer(
        use_bct=use_bct,
        bct_epsilon=bct_epsilon,
        standardize=standardize,
        bct_feature_mask=bct_mask,
    ).fit(train_raw)
    train_t = transform.transform(train_raw)
    val_t = transform.transform(val_raw)

    train_ds = NumpyStateDataset(train_t)
    val_ds = NumpyStateDataset(val_t)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=len(train_ds) >= batch_size,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        transform=transform,
        feature_dim=data.shape[1],
        train_raw=train_raw,
        val_raw=val_raw,
    )


def create_paired_data_bundle(
    input_npy_path: str,
    target_npy_path: str,
    batch_size: int,
    val_ratio: float = 0.1,
    seed: int = 42,
    num_workers: int = 0,
    subset_size: Optional[int] = None,
    use_bct: bool = True,
    bct_epsilon: float = 1e-6,
    standardize: bool = True,
    disable_input_dim0_bct: bool = False,
) -> PairedDataBundle:
    in_path = Path(input_npy_path)
    tgt_path = Path(target_npy_path)
    if not in_path.exists():
        raise FileNotFoundError(f"输入数据文件不存在: {in_path}")
    if not tgt_path.exists():
        raise FileNotFoundError(f"输出数据文件不存在: {tgt_path}")

    x_data = np.load(in_path).astype(np.float32)
    y_data = np.load(tgt_path).astype(np.float32)
    if x_data.ndim != 2 or y_data.ndim != 2:
        raise ValueError(f"期望二维数组，实际 input={x_data.shape}, target={y_data.shape}")
    if x_data.shape[0] != y_data.shape[0]:
        raise ValueError(f"输入输出样本数不一致: {x_data.shape[0]} vs {y_data.shape[0]}")

    if subset_size is not None and subset_size > 0:
        keep = min(subset_size, len(x_data))
        x_data = x_data[:keep]
        y_data = y_data[:keep]

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x_data))
    val_size = max(1, int(len(x_data) * val_ratio))
    val_idx = idx[:val_size]
    train_idx = idx[val_size:]
    if len(train_idx) == 0:
        raise ValueError("训练集为空，请增大数据量或减小 val_ratio")

    train_x_raw = x_data[train_idx]
    train_y_raw = y_data[train_idx]
    val_x_raw = x_data[val_idx]
    val_y_raw = y_data[val_idx]

    input_bct_mask = None
    if disable_input_dim0_bct and train_x_raw.shape[1] >= 1:
        input_bct_mask = np.ones(train_x_raw.shape[1], dtype=bool)
        input_bct_mask[0] = False
    input_transform = BCTStandardizer(
        use_bct=use_bct,
        bct_epsilon=bct_epsilon,
        standardize=standardize,
        bct_feature_mask=input_bct_mask,
    ).fit(train_x_raw)
    target_transform = BCTStandardizer(use_bct=use_bct, bct_epsilon=bct_epsilon, standardize=standardize).fit(train_y_raw)
    train_x = input_transform.transform(train_x_raw)
    val_x = input_transform.transform(val_x_raw)
    train_y = target_transform.transform(train_y_raw)
    val_y = target_transform.transform(val_y_raw)

    train_ds = NumpyPairDataset(train_x, train_y)
    val_ds = NumpyPairDataset(val_x, val_y)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=len(train_ds) >= batch_size,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return PairedDataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        input_transform=input_transform,
        target_transform=target_transform,
        input_dim=x_data.shape[1],
        target_dim=y_data.shape[1],
        train_x_raw=train_x_raw,
        train_y_raw=train_y_raw,
        val_x_raw=val_x_raw,
        val_y_raw=val_y_raw,
    )
