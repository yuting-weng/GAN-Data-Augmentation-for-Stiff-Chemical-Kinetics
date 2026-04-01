from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.samplers import mix_real_fake_for_quality, sample_latent
from src.models.quality_dnn import QualityDNN
from src.oracle.true_predictor import get_true_prediction
from src.utils import append_jsonl


def _binary_acc_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = (torch.sigmoid(logits) > 0.5).float()
    return (preds == y).float().mean().item()


def _collect_gan_fake_batch(
    gan_loader: DataLoader,
    generator,
    latent_dim: int,
    condition_dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    real = next(iter(gan_loader)).to(device)
    cond = real[:, :condition_dim] if condition_dim > 0 else None
    with torch.no_grad():
        z = sample_latent(real.size(0), latent_dim, device)
        fake = generator(z, cond)
    return real, fake


def _collect_gan_fake_samples(
    gan_loader: DataLoader,
    generator,
    latent_dim: int,
    condition_dim: int,
    device: torch.device,
    max_batches: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    reals = []
    fakes = []
    generator.eval()
    with torch.no_grad():
        for bi, real in enumerate(gan_loader):
            if bi >= max_batches:
                break
            real = real.to(device)
            cond = real[:, :condition_dim] if condition_dim > 0 else None
            z = sample_latent(real.size(0), latent_dim, device)
            fake = generator(z, cond)
            reals.append(real)
            fakes.append(fake)
    if not reals:
        return _collect_gan_fake_batch(gan_loader, generator, latent_dim, condition_dim, device)
    return torch.cat(reals, dim=0), torch.cat(fakes, dim=0)


def _minmax_norm(x: torch.Tensor) -> torch.Tensor:
    x_min = torch.min(x)
    x_max = torch.max(x)
    gap = torch.clamp(x_max - x_min, min=1e-8)
    return (x - x_min) / gap


def _oracle_target_from_fake(
    fake_t: torch.Tensor,
    target_dim: int,
    oracle_cfg: Dict,
    gan_transform,
    target_transform,
) -> tuple[torch.Tensor, str]:
    fake_raw = fake_t
    if gan_transform is not None:
        fake_raw = gan_transform.inverse_transform_torch(torch.clamp(fake_t, -8.0, 8.0))
    true_raw, oracle_source = get_true_prediction(
        fake_raw,
        target_dim=target_dim,
        mechanism_path=str(oracle_cfg.get("mechanism_path", "mechanism/Burke2012_s9r23.yaml")),
        time_step=float(oracle_cfg.get("time_step", 1e-7)),
        reference_pressure=oracle_cfg.get("reference_pressure", None),
    )
    true_t = true_raw
    if target_transform is not None:
        true_np = target_transform.transform(true_raw.detach().cpu().numpy())
        true_t = torch.from_numpy(true_np).to(device=fake_t.device, dtype=fake_t.dtype)
    else:
        true_t = true_raw.to(device=fake_t.device, dtype=fake_t.dtype)
    return true_t, oracle_source


def train_quality_classifier(
    gan_loader: DataLoader,
    generator,
    feature_dim: int,
    model_cfg: Dict,
    optim_cfg: Dict,
    train_cfg: Dict,
    output_dir: str | Path,
    device: torch.device,
    condition_dim: int = 0,
    real_mix_ratio: float = 0.5,
) -> tuple[QualityDNN, Dict]:
    net = QualityDNN(
        input_dim=feature_dim,
        hidden_dims=model_cfg["quality_hidden_dims"],
        mode="classifier",
        activation=model_cfg.get("activation", "gelu"),
        output_dim=1,
    ).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=float(optim_cfg["lr_quality"]))
    epochs = int(train_cfg["epochs_quality"])
    output_dir = Path(output_dir)
    logs_file = output_dir / "quality_classifier.jsonl"
    last_metrics: Dict[str, float] = {}
    generator.eval()
    for epoch in range(epochs):
        for step, real in enumerate(gan_loader):
            real = real.to(device)
            bsz = real.size(0)
            cond = real[:, :condition_dim] if condition_dim > 0 else None
            with torch.no_grad():
                z = sample_latent(bsz, model_cfg["latent_dim"], device)
                fake = generator(z, cond)
            x, y = mix_real_fake_for_quality(real, fake, real_ratio=real_mix_ratio)
            logits = net(x)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            metric_main = _binary_acc_from_logits(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            last_metrics = {
                "epoch": epoch,
                "step": step,
                "loss": float(loss.item()),
                "acc": float(metric_main),
            }
            append_jsonl(last_metrics, logs_file)

        torch.save(net.state_dict(), output_dir / "quality_classifier.pt")

    return net, last_metrics


def train_quality_regression(
    paired_loader: DataLoader,
    gan_loader: DataLoader,
    generator,
    model_cfg: Dict,
    optim_cfg: Dict,
    train_cfg: Dict,
    output_dir: str | Path,
    device: torch.device,
    target_dim: int,
    condition_dim: int = 0,
    hard_sample_direction: str = "larger_error_better",
    oracle_cfg: Dict | None = None,
    max_eval_batches: int = 8,
    gan_transform=None,
    target_transform=None,
) -> tuple[QualityDNN, Dict]:
    net = QualityDNN(
        input_dim=int(model_cfg["reg_input_dim"]),
        hidden_dims=model_cfg["quality_hidden_dims"],
        mode="error_regression",
        activation=model_cfg.get("activation", "gelu"),
        output_dim=target_dim,
    ).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=float(optim_cfg["lr_quality"]))
    epochs = int(train_cfg["epochs_quality"])
    output_dir = Path(output_dir)
    logs_file = output_dir / "quality_error_regression.jsonl"
    reg_last: Dict[str, float] = {}
    for epoch in range(epochs):
        for step, batch in enumerate(paired_loader):
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            pred = net(x)
            loss = F.mse_loss(pred, y)
            mae = F.l1_loss(pred, y).item()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            reg_last = {"epoch": epoch, "step": step, "loss": float(loss.item()), "mae": float(mae)}
            append_jsonl(reg_last, logs_file)
        torch.save(net.state_dict(), output_dir / "quality_error_regression.pt")

    oracle_cfg = dict(oracle_cfg or {})
    _, fake = _collect_gan_fake_samples(
        gan_loader=gan_loader,
        generator=generator,
        latent_dim=model_cfg["latent_dim"],
        condition_dim=condition_dim,
        device=device,
        max_batches=max(1, int(max_eval_batches)),
    )
    with torch.no_grad():
        pred = net(fake)
        true_pred, oracle_source = _oracle_target_from_fake(
            fake,
            target_dim=target_dim,
            oracle_cfg=oracle_cfg,
            gan_transform=gan_transform,
            target_transform=target_transform,
        )
        true_pred = true_pred.to(device)
        err = torch.mean(torch.abs(pred - true_pred), dim=1)
        if hard_sample_direction == "larger_error_better":
            score = err
        else:
            score = -err
        finite_mask = torch.isfinite(score) & torch.isfinite(err)
        valid_ratio = float(finite_mask.float().mean().item())
        if finite_mask.any():
            score_mean = float(score[finite_mask].mean().item())
            score_std = float(score[finite_mask].std().item())
            err_mean = float(err[finite_mask].mean().item())
        else:
            score_mean = 0.0
            score_std = 0.0
            err_mean = 0.0
    np.save(
        output_dir / "quality_regression_sample_scores.npy",
        torch.stack([score.detach().cpu(), err.detach().cpu(), finite_mask.float().detach().cpu()], dim=1).numpy().astype(np.float32),
    )
    reg_last = {
        **reg_last,
        "oracle_source": oracle_source,
        "score_mean": score_mean,
        "score_std": score_std,
        "error_mean": err_mean,
        "valid_ratio": valid_ratio,
        "num_scored": int(score.shape[0]),
    }
    append_jsonl(reg_last, logs_file)
    return net, reg_last


def train_quality_hybrid(
    gan_loader: DataLoader,
    generator,
    classifier: QualityDNN,
    regressor: QualityDNN,
    model_cfg: Dict,
    output_dir: str | Path,
    device: torch.device,
    target_dim: int,
    condition_dim: int = 0,
    w_classifier: float = 0.4,
    w_regression: float = 0.6,
    hard_sample_direction: str = "larger_error_better",
    oracle_cfg: Dict | None = None,
    max_eval_batches: int = 8,
    gan_transform=None,
    target_transform=None,
) -> Dict:
    output_dir = Path(output_dir)
    logs_file = output_dir / "quality_hybrid.jsonl"
    classifier.eval()
    regressor.eval()
    oracle_cfg = dict(oracle_cfg or {})
    _, fake = _collect_gan_fake_samples(
        gan_loader=gan_loader,
        generator=generator,
        latent_dim=model_cfg["latent_dim"],
        condition_dim=condition_dim,
        device=device,
        max_batches=max(1, int(max_eval_batches)),
    )
    with torch.no_grad():
        cls_logits = classifier(fake)
        cls_realness = torch.sigmoid(cls_logits).squeeze(-1)
        pred = regressor(fake)
        true_pred, oracle_source = _oracle_target_from_fake(
            fake,
            target_dim=target_dim,
            oracle_cfg=oracle_cfg,
            gan_transform=gan_transform,
            target_transform=target_transform,
        )
        true_pred = true_pred.to(device)
        reg_err = torch.mean(torch.abs(pred - true_pred), dim=1)
        if hard_sample_direction == "larger_error_better":
            reg_score = reg_err
        else:
            reg_score = -reg_err
        cls_score = _minmax_norm(cls_realness)
        reg_score_n = _minmax_norm(reg_score)
        hybrid_score = w_classifier * cls_score + w_regression * reg_score_n
        finite_mask = torch.isfinite(hybrid_score) & torch.isfinite(reg_err) & torch.isfinite(cls_realness)
        valid_ratio = float(finite_mask.float().mean().item())
        if finite_mask.any():
            cls_mean = float(cls_realness[finite_mask].mean().item())
            reg_err_mean = float(reg_err[finite_mask].mean().item())
            hyb_mean = float(hybrid_score[finite_mask].mean().item())
        else:
            cls_mean = 0.0
            reg_err_mean = 0.0
            hyb_mean = 0.0
    np.save(
        output_dir / "quality_hybrid_sample_scores.npy",
        torch.stack(
            [
                cls_realness.detach().cpu().squeeze(-1),
                reg_err.detach().cpu(),
                hybrid_score.detach().cpu(),
                finite_mask.float().detach().cpu(),
            ],
            dim=1,
        ).numpy().astype(np.float32),
    )
    metrics = {
        "oracle_source": oracle_source,
        "classifier_realness_mean": cls_mean,
        "regression_error_mean": reg_err_mean,
        "hybrid_score_mean": hyb_mean,
        "valid_ratio": valid_ratio,
        "num_scored": int(hybrid_score.shape[0]),
    }
    append_jsonl(metrics, logs_file)
    return metrics


def train_and_score_quality(
    gan_loader: DataLoader,
    paired_loader: DataLoader,
    generator,
    feature_dim: int,
    target_dim: int,
    model_cfg: Dict,
    optim_cfg: Dict,
    train_cfg: Dict,
    quality_cfg: Dict,
    output_dir: str | Path,
    device: torch.device,
    mode: str,
    condition_dim: int = 0,
    gan_transform=None,
    target_transform=None,
) -> Dict:
    model_cfg_local = dict(model_cfg)
    model_cfg_local["reg_input_dim"] = feature_dim
    real_mix_ratio = float(quality_cfg.get("real_mix_ratio", 0.5))
    hard_dir = str(quality_cfg.get("hard_sample_direction", "larger_error_better"))
    oracle_cfg = dict(quality_cfg.get("oracle", {}))
    max_eval_batches = int(quality_cfg.get("eval_batches", 8))
    hybrid_cfg = dict(quality_cfg.get("hybrid", {}))
    w_classifier = float(hybrid_cfg.get("w_classifier", 0.4))
    w_regression = float(hybrid_cfg.get("w_regression", 0.6))

    results: Dict[str, Dict] = {}
    classifier_model = None
    regressor_model = None

    if mode in {"classifier", "hybrid"}:
        classifier_model, cls_metrics = train_quality_classifier(
            gan_loader=gan_loader,
            generator=generator,
            feature_dim=feature_dim,
            model_cfg=model_cfg_local,
            optim_cfg=optim_cfg,
            train_cfg=train_cfg,
            output_dir=output_dir,
            device=device,
            condition_dim=condition_dim,
            real_mix_ratio=real_mix_ratio,
        )
        results["quality_classifier"] = cls_metrics

    if mode in {"error_regression", "hybrid"}:
        regressor_model, reg_metrics = train_quality_regression(
            paired_loader=paired_loader,
            gan_loader=gan_loader,
            generator=generator,
            model_cfg=model_cfg_local,
            optim_cfg=optim_cfg,
            train_cfg=train_cfg,
            output_dir=output_dir,
            device=device,
            target_dim=target_dim,
            condition_dim=condition_dim,
            hard_sample_direction=hard_dir,
            oracle_cfg=oracle_cfg,
            max_eval_batches=max_eval_batches,
            gan_transform=gan_transform,
            target_transform=target_transform,
        )
        results["quality_regression"] = reg_metrics

    if mode == "hybrid":
        if classifier_model is None or regressor_model is None:
            raise RuntimeError("hybrid 模式需要分类器和回归器都已训练")
        hybrid_metrics = train_quality_hybrid(
            gan_loader=gan_loader,
            generator=generator,
            classifier=classifier_model,
            regressor=regressor_model,
            model_cfg=model_cfg_local,
            output_dir=output_dir,
            device=device,
            target_dim=target_dim,
            condition_dim=condition_dim,
            w_classifier=w_classifier,
            w_regression=w_regression,
            hard_sample_direction=hard_dir,
            oracle_cfg=oracle_cfg,
            max_eval_batches=max_eval_batches,
            gan_transform=gan_transform,
            target_transform=target_transform,
        )
        results["quality_hybrid"] = hybrid_metrics

    results["default_quality_mode"] = {"mode": mode}
    return results
