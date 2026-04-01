from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

from src.data.dataset import create_data_bundle
from src.data.samplers import sample_latent
from src.data.transforms import BCTStandardizer
from src.eval.plot_distribution import plot_distribution_comparison
from src.models.generator import Generator
from src.utils import adapt_hidden_dims, save_json


def _normalize_species(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    y = np.clip(y, 0.0, None)
    s = float(y.sum())
    if not np.isfinite(s) or s <= 0.0:
        return np.full_like(y, 1.0 / max(1, y.shape[0]))
    return y / s


def _sanitize_state_10(state: np.ndarray, t_min: float = 250.0, t_max: float = 6000.0) -> np.ndarray:
    st = np.asarray(state, dtype=np.float64).copy()
    st[0] = float(np.clip(st[0], t_min, t_max))
    st[1:] = _normalize_species(st[1:])
    return st.astype(np.float32)


def _with_pressure(state10: np.ndarray, pressure: float) -> np.ndarray:
    out = np.empty((11,), dtype=np.float64)
    out[0] = float(state10[0])
    out[1] = float(pressure)
    out[2:] = np.asarray(state10[1:], dtype=np.float64)
    return out


def _double_step_result(state11: np.ndarray, mechanism_path: str, time_step: float) -> Tuple[np.ndarray, np.ndarray]:
    import cantera as ct

    gas = ct.Solution(mechanism_path)
    n_species = gas.n_species
    t_old = float(state11[0])
    p_old = float(state11[1])
    y_old = _normalize_species(state11[2 : 2 + n_species])
    gas.TPY = t_old, p_old, y_old
    res_1st = [t_old, p_old] + list(gas.Y) + list(gas.partial_molar_enthalpies / gas.molecular_weights)
    r = ct.IdealGasConstPressureReactor(gas, name="R1", clone=False)
    sim = ct.ReactorNet([r])
    sim.advance(float(time_step))
    new_tpy = [gas.T, gas.P] + list(gas.Y) + list(gas.partial_molar_enthalpies / gas.molecular_weights)
    res_1st += new_tpy

    t_old = gas.T
    p_old = gas.P
    res_2nd = [t_old, p_old]
    y_old = gas.Y
    res_2nd += list(y_old) + list(gas.partial_molar_enthalpies / gas.molecular_weights)
    sim.advance(2.0 * float(time_step))
    new_tpy_2 = [gas.T, gas.P] + list(gas.Y) + list(gas.partial_molar_enthalpies / gas.molecular_weights)
    res_2nd += new_tpy_2
    return np.asarray(res_1st, dtype=np.float64), np.asarray(res_2nd, dtype=np.float64)


def _formation_enthalpies(mechanism_path: str, pressure: float) -> np.ndarray:
    import cantera as ct

    gas = ct.Solution(mechanism_path)
    gas.TP = 298.15, float(pressure)
    h = gas.standard_enthalpies_RT * ct.gas_constant * 298.15 / gas.molecular_weights
    return np.asarray(h, dtype=np.float64)


def _calc_qdot(twores: np.ndarray, n_species: int, formation_h: np.ndarray, time_step: float) -> float:
    y_old = twores[2 : 2 + n_species]
    y_new = twores[4 + 2 * n_species : 4 + 3 * n_species]
    return float(-(formation_h * (y_new - y_old) / float(time_step)).sum())


def _build_baseline_qdot(
    reference_states10: np.ndarray,
    mechanism_path: str,
    time_step: float,
    pressure: float,
    max_reference: int,
) -> tuple[np.ndarray, np.ndarray]:
    import cantera as ct

    gas = ct.Solution(mechanism_path)
    n_species = gas.n_species
    fh = _formation_enthalpies(mechanism_path, pressure)
    states = reference_states10[: min(max_reference, reference_states10.shape[0])]
    q0 = np.zeros((states.shape[0],), dtype=np.float64)
    q1 = np.zeros((states.shape[0],), dtype=np.float64)
    valid = np.zeros((states.shape[0],), dtype=bool)
    for i in range(states.shape[0]):
        try:
            st11 = _with_pressure(_sanitize_state_10(states[i]), pressure)
            res0, res1 = _double_step_result(st11, mechanism_path, time_step)
            q0[i] = _calc_qdot(res0, n_species, fh, time_step)
            q1[i] = _calc_qdot(res1, n_species, fh, time_step)
            valid[i] = np.isfinite(q0[i]) and np.isfinite(q1[i])
        except Exception:
            valid[i] = False
    if valid.any():
        med0 = float(np.median(np.abs(q0[valid])))
        med1 = float(np.median(np.abs(q1[valid])))
    else:
        med0 = 1.0
        med1 = 1.0
    q0[~valid] = med0
    q1[~valid] = med1
    q0 = np.where(np.abs(q0) < 1e-12, np.sign(q0) * 1e-12 + 1e-12, q0)
    q1 = np.where(np.abs(q1) < 1e-12, np.sign(q1) * 1e-12 + 1e-12, q1)
    return q0, q1


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
    out = fake_t.clone()
    out[:, 1:] = clipped
    return out, clip_ratio


def export_generated_dataset(
    config: Dict,
    run_dir: str | Path,
    device: torch.device,
    gan_checkpoint: str | None = None,
    transform_stats_path: str | None = None,
) -> Dict:
    run_dir = Path(run_dir)
    output_dir = run_dir / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = dict(config["data"])
    model_cfg = dict(config["model"])
    gen_cfg = dict(config.get("generate", {}))
    oracle_cfg = dict(config.get("quality", {}).get("oracle", {}))
    filter_cfg = dict(gen_cfg.get("filter", {}))
    qdot_cfg = dict(filter_cfg.get("qdot_screen", {}))

    bundle = create_data_bundle(
        npy_path=data_cfg["npy_path"],
        batch_size=int(data_cfg["batch_size"]),
        val_ratio=float(data_cfg["val_ratio"]),
        seed=int(config["seed"]),
        num_workers=int(data_cfg.get("num_workers", 0)),
        subset_size=data_cfg.get("subset_size"),
        use_bct=bool(config["transform"]["use_bct"]),
        bct_epsilon=float(config["transform"]["bct_epsilon"]),
        standardize=bool(config["transform"]["standardize"]),
        disable_input_dim0_bct=bool(config.get("transform", {}).get("disable_input_dim0_bct", False)),
    )
    feature_dim = int(bundle.feature_dim)
    model_cfg = adapt_hidden_dims(model_cfg, feature_dim)
    condition_dim = int(data_cfg.get("condition_dim", 0))

    g = Generator(
        latent_dim=model_cfg["latent_dim"],
        condition_dim=condition_dim,
        output_dim=feature_dim,
        hidden_dims=model_cfg["generator_hidden_dims"],
        activation=model_cfg.get("activation", "gelu"),
        condition_encoder_cfg=dict(model_cfg.get("generator", {}).get("condition_encoder", {})),
    ).to(device)
    ckpt_path = Path(gan_checkpoint) if gan_checkpoint else (run_dir / "generator.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"未找到 Generator 权重: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    g.load_state_dict(state)
    g.eval()

    transform = bundle.transform
    tf_path = Path(transform_stats_path) if transform_stats_path else (run_dir / "transform_stats.npz")
    if tf_path.exists():
        transform = BCTStandardizer().load(tf_path)
    species_min_t = None
    species_max_t = None
    if bundle.train_raw.shape[1] > 1:
        species_min_raw = bundle.train_raw[:, 1:].min(axis=0).astype(np.float64)
        species_max_raw = bundle.train_raw[:, 1:].max(axis=0).astype(np.float64)
        shift = np.asarray(transform.shift[1:], dtype=np.float64)
        lam = np.asarray(transform.lam[1:], dtype=np.float64)
        mean = np.asarray(transform.mean[1:], dtype=np.float64)
        std = np.asarray(transform.std[1:], dtype=np.float64)
        min_t_np = _to_transformed_values(species_min_raw, shift, lam, mean, std)
        max_t_np = _to_transformed_values(species_max_raw, shift, lam, mean, std)
        lo = np.minimum(min_t_np, max_t_np).astype(np.float32)
        hi = np.maximum(min_t_np, max_t_np).astype(np.float32)
        species_min_t = torch.from_numpy(lo).to(device=device)
        species_max_t = torch.from_numpy(hi).to(device=device)

    target_size = int(gen_cfg.get("target_size", 300000))
    sample_batch_size = int(gen_cfg.get("sample_batch_size", int(data_cfg["batch_size"])))
    max_attempt_batches = int(gen_cfg.get("max_attempt_batches", 20000))
    output_name = str(gen_cfg.get("output_path", "generated_dataset_300000.npy"))
    out_path = Path(output_name)
    if not out_path.is_absolute():
        out_path = output_dir / output_name
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mechanism_path = str(oracle_cfg.get("mechanism_path", "mechanism/Burke2012_s9r23.yaml"))
    time_step = float(oracle_cfg.get("time_step", 1e-7))
    pressure_ref = float(oracle_cfg.get("reference_pressure", 101325.0))
    enable_double_step = bool(filter_cfg.get("enable_double_step", False))
    enable_qdot_screen = bool(filter_cfg.get("enable_qdot_screen", False))
    qdot_alpha = float(qdot_cfg.get("alpha", 0.1))
    qdot_cq = float(qdot_cfg.get("cq", 100.0 * qdot_alpha))
    temp_low = float(qdot_cfg.get("temp_low", 800.0))
    temp_high = float(qdot_cfg.get("temp_high", 2600.0))
    temp_step_cap = float(qdot_cfg.get("temp_step_cap", 2600.0))
    qdot_reference_size = int(qdot_cfg.get("reference_size", 5000))

    reference_q0 = None
    reference_q1 = None
    formation_h = None
    n_species = 9
    if enable_qdot_screen:
        import cantera as ct

        gas = ct.Solution(mechanism_path)
        n_species = gas.n_species
        formation_h = _formation_enthalpies(mechanism_path, pressure_ref)
        reference_q0, reference_q1 = _build_baseline_qdot(
            reference_states10=bundle.train_raw,
            mechanism_path=mechanism_path,
            time_step=time_step,
            pressure=pressure_ref,
            max_reference=qdot_reference_size,
        )

    kept = []
    stats = {
        "target_size": target_size,
        "attempt_batches": 0,
        "attempt_samples": 0,
        "accepted_samples": 0,
        "rejected_double_step_error": 0,
        "rejected_qdot_screen": 0,
        "enable_double_step": enable_double_step,
        "enable_qdot_screen": enable_qdot_screen,
        "gan_checkpoint": str(ckpt_path),
        "species_clip_ratio_sum": 0.0,
        "species_clip_ratio_count": 0,
        "sanitize_species_l1_sum": 0.0,
        "sanitize_species_l1_count": 0,
    }

    train_iter = iter(bundle.train_loader)
    with torch.no_grad():
        while len(kept) < target_size and stats["attempt_batches"] < max_attempt_batches:
            stats["attempt_batches"] += 1
            try:
                real_t = next(train_iter).to(device)
            except StopIteration:
                train_iter = iter(bundle.train_loader)
                real_t = next(train_iter).to(device)
            if condition_dim > 0:
                cond = real_t[:, :condition_dim]
                if sample_batch_size < cond.shape[0]:
                    cond = cond[:sample_batch_size]
                bsz = cond.shape[0]
            else:
                cond = None
                bsz = sample_batch_size
            z = sample_latent(bsz, model_cfg["latent_dim"], device)
            fake_t = g(z, cond)
            fake_t, clip_ratio = _clamp_species_transformed(fake_t, species_min_t, species_max_t)
            stats["species_clip_ratio_sum"] += float(clip_ratio)
            stats["species_clip_ratio_count"] += 1
            fake_raw = transform.inverse_transform(fake_t.detach().cpu().numpy())
            for j in range(fake_raw.shape[0]):
                if len(kept) >= target_size:
                    break
                stats["attempt_samples"] += 1
                candidate = _sanitize_state_10(fake_raw[j])
                pre_species = np.clip(np.asarray(fake_raw[j][1:], dtype=np.float64), 0.0, None)
                post_species = np.asarray(candidate[1:], dtype=np.float64)
                stats["sanitize_species_l1_sum"] += float(np.abs(post_species - pre_species).mean())
                stats["sanitize_species_l1_count"] += 1
                if enable_double_step or enable_qdot_screen:
                    try:
                        st11 = _with_pressure(candidate, pressure_ref)
                        res0, res1 = _double_step_result(st11, mechanism_path, time_step)
                    except Exception:
                        stats["rejected_double_step_error"] += 1
                        continue
                    if enable_qdot_screen:
                        idx_ref = (stats["attempt_samples"] - 1) % len(reference_q0)
                        q0 = _calc_qdot(res0, n_species, formation_h, time_step)
                        q1 = _calc_qdot(res1, n_species, formation_h, time_step)
                        cond0 = (res0[0] > temp_low) and (res0[0] < temp_high) and (res0[2 + 2 * n_species] < temp_step_cap)
                        cond1 = (res1[0] > temp_low) and (res1[0] < temp_high) and (res1[2 + 2 * n_species] < temp_step_cap)
                        q0_ok = (q0 > (1.0 / qdot_cq) * reference_q0[idx_ref]) and (q0 < qdot_cq * reference_q0[idx_ref])
                        q1_ok = (q1 > (1.0 / qdot_cq) * reference_q1[idx_ref]) and (q1 < qdot_cq * reference_q1[idx_ref])
                        if not ((cond0 and q0_ok) or (cond1 and q1_ok)):
                            stats["rejected_qdot_screen"] += 1
                            continue
                kept.append(candidate)

    if not kept:
        raise RuntimeError("未生成任何可用样本，请检查筛选参数。")
    gen_arr = np.asarray(kept, dtype=np.float32)
    if gen_arr.shape[0] > target_size:
        gen_arr = gen_arr[:target_size]
    np.save(out_path, gen_arr)
    stats["accepted_samples"] = int(gen_arr.shape[0])
    stats["accept_ratio"] = float(gen_arr.shape[0] / max(1, stats["attempt_samples"]))
    stats["output_path"] = str(out_path)
    stats["species_clip_ratio_mean"] = float(
        stats["species_clip_ratio_sum"] / max(1, stats["species_clip_ratio_count"])
    )
    stats["sanitize_species_l1_mean"] = float(
        stats["sanitize_species_l1_sum"] / max(1, stats["sanitize_species_l1_count"])
    )

    real_arr = np.load(data_cfg["npy_path"]).astype(np.float32)
    plots = plot_distribution_comparison(real_arr, gen_arr, output_dir / "plots")
    stats["plots"] = plots

    save_json(stats, output_dir / "generation_summary.json")
    return stats
