from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.samplers import sample_latent
from src.losses.physics import physics_loss, species_bounds_hinge_loss
from src.losses.wgan_gp import critic_loss_wgan_gp, generator_loss_wgan
from src.models.critic import Critic
from src.models.generator import Generator
from src.models.quality_dnn import QualityDNN
from src.oracle.true_predictor import get_true_prediction
from src.utils import append_jsonl


def _param_l1_sum(module: torch.nn.Module) -> float:
    s = 0.0
    with torch.no_grad():
        for p in module.parameters():
            s += float(torch.sum(torch.abs(p.detach())).item())
    return s


def _condition_from_real(real: torch.Tensor, condition_dim: int) -> torch.Tensor | None:
    if condition_dim <= 0:
        return None
    return real[:, :condition_dim]


def _minmax_norm(x: torch.Tensor) -> torch.Tensor:
    x_min = torch.min(x)
    x_max = torch.max(x)
    gap = torch.clamp(x_max - x_min, min=1e-8)
    return (x - x_min) / gap


def _to_transformed_values(raw_vals: np.ndarray, shift: np.ndarray, lam: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    x_pos = raw_vals + shift
    out = np.empty_like(x_pos, dtype=np.float64)
    near_zero = np.isclose(lam, 0.0)
    out[near_zero] = np.log(np.maximum(x_pos[near_zero], 1e-12))
    out[~near_zero] = (np.power(np.maximum(x_pos[~near_zero], 1e-12), lam[~near_zero]) - 1.0) / lam[~near_zero]
    out = (out - mean) / std
    return out


def _clamp_species_transformed(
    fake_t: torch.Tensor,
    species_min_t: torch.Tensor | None,
    species_max_t: torch.Tensor | None,
) -> tuple[torch.Tensor, float]:
    if species_min_t is None or species_max_t is None:
        return fake_t, 0.0
    species = fake_t[:, 1:]
    clipped = torch.clamp(species, min=species_min_t, max=species_max_t)
    clip_ratio = float((species.ne(clipped)).float().mean().item())
    fake_t = fake_t.clone()
    fake_t[:, 1:] = clipped
    return fake_t, clip_ratio


def _oracle_target_transformed(
    fake_t: torch.Tensor,
    gan_transform,
    target_transform,
    target_dim: int,
    oracle_cfg: Dict,
) -> tuple[torch.Tensor, str]:
    fake_raw = gan_transform.inverse_transform_torch(torch.clamp(fake_t, -8.0, 8.0))
    true_raw, source = get_true_prediction(
        fake_raw,
        target_dim=target_dim,
        mechanism_path=str(oracle_cfg.get("mechanism_path", "mechanism/Burke2012_s9r23.yaml")),
        time_step=float(oracle_cfg.get("time_step", 1e-7)),
        reference_pressure=oracle_cfg.get("reference_pressure", None),
    )
    true_np = target_transform.transform(true_raw.detach().cpu().numpy())
    true_t = torch.from_numpy(true_np).to(device=fake_t.device, dtype=fake_t.dtype)
    return true_t, source


def _train_regressor_pretrain(
    regressor: QualityDNN,
    paired_loader: DataLoader,
    lr: float,
    epochs: int,
    output_dir: Path,
) -> Dict:
    opt = torch.optim.AdamW(regressor.parameters(), lr=float(lr))
    log_path = output_dir / "quality_regressor_pretrain.jsonl"
    last = {}
    for ep in range(max(1, int(epochs))):
        for st, batch in enumerate(paired_loader):
            x, y = batch
            x = x.to(next(regressor.parameters()).device)
            y = y.to(next(regressor.parameters()).device)
            pred = regressor(x)
            loss = F.mse_loss(pred, y)
            mae = F.l1_loss(pred, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            last = {"epoch": ep, "step": st, "loss": float(loss.item()), "mae": float(mae.item())}
            append_jsonl(last, log_path)
    torch.save(regressor.state_dict(), output_dir / "quality_regressor_pretrain.pt")
    return last


def train_gan(
    train_loader: DataLoader,
    transform,
    feature_dim: int,
    model_cfg: Dict,
    optim_cfg: Dict,
    train_cfg: Dict,
    output_dir: str | Path,
    device: torch.device,
    condition_dim: int = 0,
    species_min_raw: np.ndarray | None = None,
    species_max_raw: np.ndarray | None = None,
) -> tuple[Generator, Critic, Dict]:
    gen_cfg = dict(model_cfg.get("generator", {}))
    cond_enc_cfg = dict(gen_cfg.get("condition_encoder", {}))
    critic_cfg = dict(model_cfg.get("critic", {}))
    mbdisc_cfg = dict(critic_cfg.get("minibatch_discrimination", {}))
    g = Generator(
        latent_dim=model_cfg["latent_dim"],
        condition_dim=condition_dim,
        output_dim=feature_dim,
        hidden_dims=model_cfg["generator_hidden_dims"],
        activation=model_cfg.get("activation", "gelu"),
        condition_encoder_cfg=cond_enc_cfg,
    ).to(device)
    c = Critic(
        input_dim=feature_dim,
        hidden_dims=model_cfg["critic_hidden_dims"],
        activation=model_cfg.get("activation", "gelu"),
        use_spectral_norm=bool(model_cfg.get("use_spectral_norm", True)),
        minibatch_discrimination_cfg=mbdisc_cfg,
    ).to(device)

    betas = tuple(optim_cfg.get("betas", [0.5, 0.9]))
    opt_g = torch.optim.AdamW(g.parameters(), lr=float(optim_cfg["lr_g"]), betas=betas)
    opt_c = torch.optim.AdamW(c.parameters(), lr=float(optim_cfg["lr_c"]), betas=betas)

    n_critic = int(train_cfg.get("n_critic", 5))
    gp_lambda = float(train_cfg.get("wgan_gp_lambda", 10.0))
    physics_weight = float(train_cfg.get("physics_weight", 1.0))
    w_mass = float(train_cfg["physics_loss_weights"].get("mass_conservation", 1.0))
    w_nonneg = float(train_cfg["physics_loss_weights"].get("non_negative", 1.0))
    species_bounds_cfg = dict(train_cfg.get("physics_species_bounds", {}))
    use_species_bounds = bool(species_bounds_cfg.get("enabled", False))
    species_bounds_weight = float(species_bounds_cfg.get("weight", 1.0))
    species_bounds_hinge = bool(species_bounds_cfg.get("use_hinge", True))
    species_min_raw_t = None
    species_max_raw_t = None
    if use_species_bounds and species_min_raw is not None and species_max_raw is not None:
        lo = np.minimum(np.asarray(species_min_raw, dtype=np.float32), np.asarray(species_max_raw, dtype=np.float32))
        hi = np.maximum(np.asarray(species_min_raw, dtype=np.float32), np.asarray(species_max_raw, dtype=np.float32))
        species_min_raw_t = torch.from_numpy(lo).to(device=device)
        species_max_raw_t = torch.from_numpy(hi).to(device=device)
    grad_clip_norm = float(train_cfg.get("grad_clip_norm", 5.0))
    epochs = int(train_cfg["epochs_gan"])
    log_interval = int(train_cfg.get("log_interval", 20))

    output_dir = Path(output_dir)
    logs_file = output_dir / "gan_train.jsonl"
    global_step = 0
    last_metrics: Dict[str, float] = {}

    for epoch in range(epochs):
        for batch_idx, real in enumerate(train_loader):
            real = real.to(device)
            bsz = real.size(0)
            cond = _condition_from_real(real, condition_dim)

            for _ in range(n_critic):
                z = sample_latent(bsz, model_cfg["latent_dim"], device)
                with torch.no_grad():
                    fake = g(z, cond)
                loss_c, c_metrics = critic_loss_wgan_gp(c, real, fake, gp_lambda=gp_lambda)
                opt_c.zero_grad(set_to_none=True)
                loss_c.backward()
                torch.nn.utils.clip_grad_norm_(c.parameters(), grad_clip_norm)
                opt_c.step()

            z = sample_latent(bsz, model_cfg["latent_dim"], device)
            fake = g(z, cond)
            g_wgan = generator_loss_wgan(c, fake)
            fake_raw = transform.inverse_transform_torch(torch.clamp(fake, -8.0, 8.0))
            real_raw = transform.inverse_transform_torch(real)
            g_phys, phys_metrics = physics_loss(fake_raw, real_raw, w_mass=w_mass, w_nonneg=w_nonneg)
            if use_species_bounds and species_min_raw_t is not None and species_max_raw_t is not None and fake_raw.shape[1] > 1:
                g_species_bound, species_bound_violate_ratio = species_bounds_hinge_loss(
                    fake_species_raw=fake_raw[:, 1:],
                    species_min_raw=species_min_raw_t,
                    species_max_raw=species_max_raw_t,
                    use_hinge=species_bounds_hinge,
                )
            else:
                g_species_bound = torch.zeros((), device=device, dtype=fake_raw.dtype)
                species_bound_violate_ratio = torch.zeros((), device=device, dtype=fake_raw.dtype)
            g_phys_total = g_phys + species_bounds_weight * g_species_bound
            g_total = g_wgan + physics_weight * g_phys_total

            opt_g.zero_grad(set_to_none=True)
            g_total.backward()
            torch.nn.utils.clip_grad_norm_(g.parameters(), grad_clip_norm)
            opt_g.step()

            last_metrics = {
                "epoch": epoch,
                "step": global_step,
                "loss_c": float(loss_c.item()),
                "loss_g_wgan": float(g_wgan.item()),
                "loss_g_phys": float(g_phys.item()),
                "loss_g_species_bound": float(g_species_bound.item()),
                "loss_g_phys_total": float(g_phys_total.item()),
                "loss_g_total": float(g_total.item()),
                "species_bound_violate_ratio": float(species_bound_violate_ratio.item()),
                **c_metrics,
                **phys_metrics,
            }
            append_jsonl(last_metrics, logs_file)
            if global_step % log_interval == 0:
                print(last_metrics)
            global_step += 1

        torch.save(g.state_dict(), output_dir / "generator.pt")
        torch.save(c.state_dict(), output_dir / "critic.pt")

    return g, c, last_metrics


def train_gan_three_stage(
    train_loader: DataLoader,
    paired_loader: DataLoader,
    transform,
    target_transform,
    feature_dim: int,
    target_dim: int,
    model_cfg: Dict,
    optim_cfg: Dict,
    train_cfg: Dict,
    quality_cfg: Dict,
    output_dir: str | Path,
    device: torch.device,
    condition_dim: int = 0,
    species_min_raw: np.ndarray | None = None,
    species_max_raw: np.ndarray | None = None,
) -> tuple[Generator, Critic, Dict]:
    output_dir = Path(output_dir)
    logs_file = output_dir / "gan_train_three_stage.jsonl"
    gen_cfg = dict(model_cfg.get("generator", {}))
    cond_enc_cfg = dict(gen_cfg.get("condition_encoder", {}))
    critic_cfg = dict(model_cfg.get("critic", {}))
    mbdisc_cfg = dict(critic_cfg.get("minibatch_discrimination", {}))
    g = Generator(
        latent_dim=model_cfg["latent_dim"],
        condition_dim=condition_dim,
        output_dim=feature_dim,
        hidden_dims=model_cfg["generator_hidden_dims"],
        activation=model_cfg.get("activation", "gelu"),
        condition_encoder_cfg=cond_enc_cfg,
    ).to(device)
    c = Critic(
        input_dim=feature_dim,
        hidden_dims=model_cfg["critic_hidden_dims"],
        activation=model_cfg.get("activation", "gelu"),
        use_spectral_norm=bool(model_cfg.get("use_spectral_norm", True)),
        minibatch_discrimination_cfg=mbdisc_cfg,
    ).to(device)
    q_cls = QualityDNN(
        input_dim=feature_dim,
        hidden_dims=model_cfg["quality_hidden_dims"],
        mode="classifier",
        activation=model_cfg.get("activation", "gelu"),
        output_dim=1,
    ).to(device)
    q_reg = QualityDNN(
        input_dim=feature_dim,
        hidden_dims=model_cfg["quality_hidden_dims"],
        mode="error_regression",
        activation=model_cfg.get("activation", "gelu"),
        output_dim=target_dim,
    ).to(device)

    three_stage_cfg = dict(train_cfg.get("three_stage", {}))
    reg_pretrain_epochs = int(three_stage_cfg.get("reg_pretrain_epochs", train_cfg.get("epochs_quality", 3)))
    generator_wgan_weight = float(three_stage_cfg.get("generator_wgan_weight", 0.0))
    cls_real_ratio = float(three_stage_cfg.get("classifier_real_mix_ratio", quality_cfg.get("real_mix_ratio", 0.5)))
    hard_direction = str(quality_cfg.get("hard_sample_direction", "larger_error_better"))
    oracle_cfg = dict(quality_cfg.get("oracle", {}))
    hybrid_cfg = dict(quality_cfg.get("hybrid", {}))
    w_cls = float(hybrid_cfg.get("w_classifier", 0.8))
    w_reg = float(hybrid_cfg.get("w_regression", 0.2))
    loss_balance_cfg = dict(three_stage_cfg.get("loss_balance", {}))
    use_loss_balance = bool(loss_balance_cfg.get("enabled", False))
    ema_beta = float(loss_balance_cfg.get("ema_beta", 0.98))
    ema_eps = float(loss_balance_cfg.get("eps", 1e-6))
    phys_clip_max = float(loss_balance_cfg.get("phys_clip_max", 5.0))
    lambda_quality = float(loss_balance_cfg.get("lambda_quality", 1.0))
    lambda_phys = float(loss_balance_cfg.get("lambda_phys", 0.1))
    lambda_wgan = float(loss_balance_cfg.get("lambda_wgan", 0.0))
    lambda_cond = float(loss_balance_cfg.get("lambda_cond", 0.0))

    _train_regressor_pretrain(
        regressor=q_reg,
        paired_loader=paired_loader,
        lr=float(optim_cfg["lr_quality"]),
        epochs=reg_pretrain_epochs,
        output_dir=output_dir,
    )
    for p in q_reg.parameters():
        p.requires_grad_(False)
    q_reg.eval()
    q_reg_l1_anchor = _param_l1_sum(q_reg)

    species_min_t = None
    species_max_t = None
    if species_min_raw is not None and species_max_raw is not None:
        shift = np.asarray(transform.shift[1:], dtype=np.float64)
        lam = np.asarray(transform.lam[1:], dtype=np.float64)
        mean = np.asarray(transform.mean[1:], dtype=np.float64)
        std = np.asarray(transform.std[1:], dtype=np.float64)
        min_t_np = _to_transformed_values(np.asarray(species_min_raw, dtype=np.float64), shift, lam, mean, std)
        max_t_np = _to_transformed_values(np.asarray(species_max_raw, dtype=np.float64), shift, lam, mean, std)
        min_t_np = np.minimum(min_t_np, max_t_np)
        max_t_np = np.maximum(min_t_np, max_t_np)
        species_min_t = torch.from_numpy(min_t_np.astype(np.float32)).to(device=device)
        species_max_t = torch.from_numpy(max_t_np.astype(np.float32)).to(device=device)

    betas = tuple(optim_cfg.get("betas", [0.5, 0.9]))
    opt_g = torch.optim.AdamW(g.parameters(), lr=float(optim_cfg["lr_g"]), betas=betas)
    opt_c = torch.optim.AdamW(c.parameters(), lr=float(optim_cfg["lr_c"]), betas=betas)
    opt_qcls = torch.optim.AdamW(q_cls.parameters(), lr=float(optim_cfg["lr_quality"]))

    n_critic = int(train_cfg.get("n_critic", 5))
    gp_lambda = float(train_cfg.get("wgan_gp_lambda", 10.0))
    physics_weight = float(train_cfg.get("physics_weight", 1.0))
    w_mass = float(train_cfg["physics_loss_weights"].get("mass_conservation", 1.0))
    w_nonneg = float(train_cfg["physics_loss_weights"].get("non_negative", 1.0))
    species_bounds_cfg = dict(train_cfg.get("physics_species_bounds", {}))
    use_species_bounds = bool(species_bounds_cfg.get("enabled", False))
    species_bounds_weight = float(species_bounds_cfg.get("weight", 1.0))
    species_bounds_hinge = bool(species_bounds_cfg.get("use_hinge", True))
    species_min_raw_t = None
    species_max_raw_t = None
    if use_species_bounds and species_min_raw is not None and species_max_raw is not None:
        lo = np.minimum(np.asarray(species_min_raw, dtype=np.float32), np.asarray(species_max_raw, dtype=np.float32))
        hi = np.maximum(np.asarray(species_min_raw, dtype=np.float32), np.asarray(species_max_raw, dtype=np.float32))
        species_min_raw_t = torch.from_numpy(lo).to(device=device)
        species_max_raw_t = torch.from_numpy(hi).to(device=device)
    grad_clip_norm = float(train_cfg.get("grad_clip_norm", 5.0))
    epochs = int(train_cfg["epochs_gan"])
    log_interval = int(train_cfg.get("log_interval", 20))

    global_step = 0
    ema_abs_gq = 1.0
    ema_abs_gp = 1.0
    ema_abs_gw = 1.0
    ema_abs_gc = 1.0
    last_metrics: Dict[str, float] = {}
    for epoch in range(epochs):
        for batch_idx, real in enumerate(train_loader):
            real = real.to(device)
            bsz = real.size(0)
            cond = _condition_from_real(real, condition_dim)

            for _ in range(n_critic):
                z = sample_latent(bsz, model_cfg["latent_dim"], device)
                with torch.no_grad():
                    fake = g(z, cond)
                loss_c, c_metrics = critic_loss_wgan_gp(c, real, fake, gp_lambda=gp_lambda)
                opt_c.zero_grad(set_to_none=True)
                loss_c.backward()
                torch.nn.utils.clip_grad_norm_(c.parameters(), grad_clip_norm)
                opt_c.step()

            z = sample_latent(bsz, model_cfg["latent_dim"], device)
            with torch.no_grad():
                fake_det = g(z, cond)
                fake_det, _ = _clamp_species_transformed(fake_det, species_min_t, species_max_t)
            n_real = max(1, int(bsz * cls_real_ratio))
            n_fake = bsz - n_real
            real_sel = real[torch.randperm(bsz, device=device)[:n_real]]
            fake_sel = fake_det[torch.randperm(bsz, device=device)[:n_fake]]
            x_cls = torch.cat([real_sel, fake_sel], dim=0)
            y_cls = torch.cat(
                [
                    torch.ones(n_real, dtype=torch.float32, device=device),
                    torch.zeros(n_fake, dtype=torch.float32, device=device),
                ],
                dim=0,
            )
            perm = torch.randperm(x_cls.size(0), device=device)
            x_cls = x_cls[perm]
            y_cls = y_cls[perm]
            cls_logits = q_cls(x_cls)
            loss_qcls = F.binary_cross_entropy_with_logits(cls_logits, y_cls)
            opt_qcls.zero_grad(set_to_none=True)
            loss_qcls.backward()
            torch.nn.utils.clip_grad_norm_(q_cls.parameters(), grad_clip_norm)
            opt_qcls.step()

            for p in q_cls.parameters():
                p.requires_grad_(False)
            z = sample_latent(bsz, model_cfg["latent_dim"], device)
            fake = g(z, cond)
            fake, species_clip_ratio = _clamp_species_transformed(fake, species_min_t, species_max_t)
            g_wgan = generator_loss_wgan(c, fake)
            if condition_dim > 0 and cond is not None:
                g_cond = F.smooth_l1_loss(fake[:, :condition_dim], cond)
            else:
                g_cond = torch.zeros((), device=device, dtype=fake.dtype)
            fake_raw = transform.inverse_transform_torch(torch.clamp(fake, -8.0, 8.0))
            real_raw = transform.inverse_transform_torch(real)
            g_phys, phys_metrics = physics_loss(fake_raw, real_raw, w_mass=w_mass, w_nonneg=w_nonneg)
            if use_species_bounds and species_min_raw_t is not None and species_max_raw_t is not None and fake_raw.shape[1] > 1:
                g_species_bound, species_bound_violate_ratio = species_bounds_hinge_loss(
                    fake_species_raw=fake_raw[:, 1:],
                    species_min_raw=species_min_raw_t,
                    species_max_raw=species_max_raw_t,
                    use_hinge=species_bounds_hinge,
                )
            else:
                g_species_bound = torch.zeros((), device=device, dtype=fake_raw.dtype)
                species_bound_violate_ratio = torch.zeros((), device=device, dtype=fake_raw.dtype)
            g_phys_total = g_phys + species_bounds_weight * g_species_bound
            cls_realness = torch.sigmoid(q_cls(fake))
            reg_pred = q_reg(fake)
            true_t, oracle_source = _oracle_target_transformed(
                fake_t=fake,
                gan_transform=transform,
                target_transform=target_transform,
                target_dim=target_dim,
                oracle_cfg=oracle_cfg,
            )
            reg_err = torch.mean(torch.abs(reg_pred - true_t), dim=1)
            reg_score = reg_err if hard_direction == "larger_error_better" else -reg_err
            cls_score = torch.clamp(cls_realness, min=0.0, max=1.0)
            reg_score_n = _minmax_norm(reg_score)
            critic_score = w_cls * cls_score + w_reg * reg_score_n
            g_quality = -torch.mean(critic_score)
            if use_loss_balance:
                ema_abs_gq = ema_beta * ema_abs_gq + (1.0 - ema_beta) * abs(float(g_quality.detach().item()))
                ema_abs_gp = ema_beta * ema_abs_gp + (1.0 - ema_beta) * abs(float(g_phys_total.detach().item()))
                ema_abs_gw = ema_beta * ema_abs_gw + (1.0 - ema_beta) * abs(float(g_wgan.detach().item()))
                ema_abs_gc = ema_beta * ema_abs_gc + (1.0 - ema_beta) * abs(float(g_cond.detach().item()))
                g_quality_n = g_quality / (ema_abs_gq + ema_eps)
                g_phys_n = g_phys_total / (ema_abs_gp + ema_eps)
                g_wgan_n = g_wgan / (ema_abs_gw + ema_eps)
                g_cond_n = g_cond / (ema_abs_gc + ema_eps)
                if phys_clip_max > 0:
                    g_phys_n = torch.clamp(g_phys_n, min=-phys_clip_max, max=phys_clip_max)
                g_total = (
                    lambda_quality * g_quality_n
                    + lambda_phys * g_phys_n
                    + lambda_wgan * g_wgan_n
                    + lambda_cond * g_cond_n
                )
                abs_q = abs(float((lambda_quality * g_quality_n).detach().item()))
                abs_p = abs(float((lambda_phys * g_phys_n).detach().item()))
                abs_w = abs(float((lambda_wgan * g_wgan_n).detach().item()))
                abs_c = abs(float((lambda_cond * g_cond_n).detach().item()))
                share_den = max(abs_q + abs_p + abs_w + abs_c, 1e-12)
                g_share_quality = abs_q / share_den
                g_share_phys = abs_p / share_den
                g_share_wgan = abs_w / share_den
                g_share_cond = abs_c / share_den
                loss_g_quality_norm = float(g_quality_n.detach().item())
                loss_g_phys_norm = float(g_phys_n.detach().item())
                loss_g_wgan_norm = float(g_wgan_n.detach().item())
                loss_g_cond_norm = float(g_cond_n.detach().item())
            else:
                g_total = g_quality + physics_weight * g_phys_total + generator_wgan_weight * g_wgan + lambda_cond * g_cond
                abs_q = abs(float(g_quality.detach().item()))
                abs_p = abs(float((physics_weight * g_phys_total).detach().item()))
                abs_w = abs(float((generator_wgan_weight * g_wgan).detach().item()))
                abs_c = abs(float((lambda_cond * g_cond).detach().item()))
                share_den = max(abs_q + abs_p + abs_w + abs_c, 1e-12)
                g_share_quality = abs_q / share_den
                g_share_phys = abs_p / share_den
                g_share_wgan = abs_w / share_den
                g_share_cond = abs_c / share_den
                loss_g_quality_norm = float(g_quality.detach().item())
                loss_g_phys_norm = float(g_phys.detach().item())
                loss_g_wgan_norm = float(g_wgan.detach().item())
                loss_g_cond_norm = float(g_cond.detach().item())
            opt_g.zero_grad(set_to_none=True)
            g_total.backward()
            torch.nn.utils.clip_grad_norm_(g.parameters(), grad_clip_norm)
            opt_g.step()
            for p in q_cls.parameters():
                p.requires_grad_(True)

            cls_acc = float(((cls_logits > 0).float() == y_cls).float().mean().item())
            q_reg_l1_now = _param_l1_sum(q_reg)
            last_metrics = {
                "epoch": epoch,
                "step": global_step,
                "loss_c": float(loss_c.item()),
                "loss_qcls": float(loss_qcls.item()),
                "loss_g_wgan": float(g_wgan.item()),
                "loss_g_quality": float(g_quality.item()),
                "loss_g_phys": float(g_phys.item()),
                "loss_g_species_bound": float(g_species_bound.item()),
                "loss_g_phys_total": float(g_phys_total.item()),
                "loss_g_cond": float(g_cond.item()),
                "loss_g_total": float(g_total.item()),
                "loss_g_quality_norm": loss_g_quality_norm,
                "loss_g_phys_norm": loss_g_phys_norm,
                "loss_g_wgan_norm": loss_g_wgan_norm,
                "loss_g_cond_norm": loss_g_cond_norm,
                "critic_score_mean": float(critic_score.mean().item()),
                "cls_score_mean": float(cls_score.mean().item()),
                "reg_score_mean": float(reg_score_n.mean().item()),
                "cls_acc": cls_acc,
                "g_share_quality": float(g_share_quality),
                "g_share_phys": float(g_share_phys),
                "g_share_wgan": float(g_share_wgan),
                "g_share_cond": float(g_share_cond),
                "species_clip_ratio": float(species_clip_ratio),
                "species_bound_violate_ratio": float(species_bound_violate_ratio.item()),
                "oracle_source": oracle_source,
                "q_reg_l1_anchor": q_reg_l1_anchor,
                "q_reg_l1_now": q_reg_l1_now,
                "q_reg_l1_drift": float(q_reg_l1_now - q_reg_l1_anchor),
                **c_metrics,
                **phys_metrics,
            }
            append_jsonl(last_metrics, logs_file)
            if global_step % log_interval == 0:
                print(last_metrics)
            global_step += 1

        torch.save(g.state_dict(), output_dir / "generator.pt")
        torch.save(c.state_dict(), output_dir / "critic.pt")
        torch.save(q_cls.state_dict(), output_dir / "quality_classifier_joint.pt")
        torch.save(q_reg.state_dict(), output_dir / "quality_regressor_pretrain.pt")

    return g, c, last_metrics
