from __future__ import annotations

from pathlib import Path
from typing import Dict

from src.data.dataset import create_data_bundle, create_paired_data_bundle
from src.trainers.gan_trainer import train_gan
from src.trainers.quality_trainer import train_and_score_quality
from src.utils import adapt_hidden_dims, save_json


def run_smoke(config: Dict, output_dir: str | Path, device):
    data_cfg = dict(config["data"])
    train_cfg = dict(config["train"])
    data_cfg["subset_size"] = data_cfg.get("subset_size") or 2048
    train_cfg["epochs_gan"] = min(2, int(train_cfg.get("epochs_gan", 2)))
    train_cfg["epochs_quality"] = min(2, int(train_cfg.get("epochs_quality", 2)))
    train_cfg["n_critic"] = min(2, int(train_cfg.get("n_critic", 2)))

    bundle = create_data_bundle(
        npy_path=data_cfg["npy_path"],
        batch_size=int(data_cfg["batch_size"]),
        val_ratio=float(data_cfg["val_ratio"]),
        seed=int(config["seed"]),
        num_workers=int(data_cfg.get("num_workers", 0)),
        subset_size=int(data_cfg["subset_size"]),
        use_bct=bool(config["transform"]["use_bct"]),
        bct_epsilon=float(config["transform"]["bct_epsilon"]),
        standardize=bool(config["transform"]["standardize"]),
    )
    model_cfg = adapt_hidden_dims(config["model"], bundle.feature_dim)
    output_dir = Path(output_dir)
    bundle.transform.save(output_dir / "transform_stats.npz")
    quality_cfg = dict(config.get("quality", {}))
    paired_bundle = create_paired_data_bundle(
        input_npy_path=str(quality_cfg.get("regression_input_path")),
        target_npy_path=str(quality_cfg.get("regression_target_path")),
        batch_size=int(data_cfg["batch_size"]),
        val_ratio=float(data_cfg["val_ratio"]),
        seed=int(config["seed"]),
        num_workers=int(data_cfg.get("num_workers", 0)),
        subset_size=int(data_cfg["subset_size"]),
        use_bct=bool(config["transform"]["use_bct"]),
        bct_epsilon=float(config["transform"]["bct_epsilon"]),
        standardize=bool(config["transform"]["standardize"]),
    )
    paired_bundle.input_transform.save(output_dir / "reg_input_transform_stats.npz")
    paired_bundle.target_transform.save(output_dir / "reg_target_transform_stats.npz")

    g, _, gan_metrics = train_gan(
        train_loader=bundle.train_loader,
        transform=bundle.transform,
        feature_dim=bundle.feature_dim,
        model_cfg=model_cfg,
        optim_cfg=config["optim"],
        train_cfg=train_cfg,
        output_dir=output_dir,
        device=device,
        condition_dim=int(data_cfg.get("condition_dim", 0)),
    )
    quality_metrics = train_and_score_quality(
        gan_loader=bundle.train_loader,
        paired_loader=paired_bundle.train_loader,
        generator=g,
        feature_dim=bundle.feature_dim,
        target_dim=paired_bundle.target_dim,
        model_cfg=model_cfg,
        optim_cfg=config["optim"],
        train_cfg=train_cfg,
        quality_cfg=quality_cfg,
        output_dir=output_dir,
        device=device,
        mode=str(quality_cfg.get("default_mode", "hybrid")),
        condition_dim=int(data_cfg.get("condition_dim", 0)),
        gan_transform=bundle.transform,
        target_transform=paired_bundle.target_transform,
    )
    summary = {"gan": gan_metrics, **quality_metrics}
    save_json(summary, output_dir / "smoke_summary.json")
    return summary
