from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.data.transforms import BCTStandardizer
from src.models.quality_dnn import QualityDNN
from src.oracle.true_predictor import get_true_prediction
from src.utils import adapt_hidden_dims, load_config, save_json, set_seed


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="55k真实 vs 55k+60k生成 的DNN泛化对比")
    p.add_argument("--config", type=str, default="configs/exp_real55k_cond_0p2.yaml")
    p.add_argument("--generated_path", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--total_size", type=int, default=60000)
    p.add_argument("--test_size", type=int, default=5000)
    p.add_argument("--split_npz", type=str, default=None)
    p.add_argument("--report_dir", type=str, default=None)
    p.add_argument("--oracle_batch_size", type=int, default=256)
    return p


def _split_indices(n: int, test_size: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    test_idx = idx[:test_size]
    train_idx = idx[test_size:]
    return train_idx, test_idx


def _select_device(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("请求CUDA但当前不可用。")
        return torch.device("cuda")
    return torch.device("cpu")


def _fit_transforms(
    x_train_raw: np.ndarray,
    y_train_raw: np.ndarray,
    use_bct: bool,
    bct_epsilon: float,
    standardize: bool,
    disable_input_dim0_bct: bool,
) -> tuple[BCTStandardizer, BCTStandardizer]:
    input_bct_mask = None
    if disable_input_dim0_bct and x_train_raw.shape[1] >= 1:
        input_bct_mask = np.ones(x_train_raw.shape[1], dtype=bool)
        input_bct_mask[0] = False
    x_tf = BCTStandardizer(
        use_bct=use_bct,
        bct_epsilon=bct_epsilon,
        standardize=standardize,
        bct_feature_mask=input_bct_mask,
    ).fit(x_train_raw)
    y_tf = BCTStandardizer(
        use_bct=use_bct,
        bct_epsilon=bct_epsilon,
        standardize=standardize,
    ).fit(y_train_raw)
    return x_tf, y_tf


def _build_regressor(model_cfg: Dict, input_dim: int, output_dim: int, device: torch.device) -> QualityDNN:
    cfg = adapt_hidden_dims(dict(model_cfg), input_dim)
    net = QualityDNN(
        input_dim=input_dim,
        hidden_dims=cfg["quality_hidden_dims"],
        mode="error_regression",
        activation=cfg.get("activation", "gelu"),
        output_dim=output_dim,
    ).to(device)
    return net


def _train_regressor(
    x_train_t: np.ndarray,
    y_train_t: np.ndarray,
    model_cfg: Dict,
    optim_cfg: Dict,
    epochs: int,
    batch_size: int,
    device: torch.device,
) -> QualityDNN:
    net = _build_regressor(model_cfg=model_cfg, input_dim=x_train_t.shape[1], output_dim=y_train_t.shape[1], device=device)
    opt = torch.optim.AdamW(net.parameters(), lr=float(optim_cfg["lr_quality"]))
    ds = TensorDataset(
        torch.from_numpy(x_train_t.astype(np.float32)),
        torch.from_numpy(y_train_t.astype(np.float32)),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=len(ds) >= batch_size)
    net.train()
    for _ in range(max(1, int(epochs))):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = net(xb)
            loss = F.mse_loss(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return net


def _eval_regressor(
    net: QualityDNN,
    x_tf: BCTStandardizer,
    y_tf: BCTStandardizer,
    x_test_raw: np.ndarray,
    y_test_raw: np.ndarray,
    device: torch.device,
) -> Dict:
    x_test_t = x_tf.transform(x_test_raw).astype(np.float32)
    y_test_t = y_tf.transform(y_test_raw).astype(np.float32)
    with torch.no_grad():
        pred_t = net(torch.from_numpy(x_test_t).to(device)).detach().cpu().numpy()
    mae_t = float(np.mean(np.abs(pred_t - y_test_t)))
    mse_t = float(np.mean((pred_t - y_test_t) ** 2))
    pred_raw = y_tf.inverse_transform(pred_t)
    mae_raw = float(np.mean(np.abs(pred_raw - y_test_raw)))
    mse_raw = float(np.mean((pred_raw - y_test_raw) ** 2))
    per_dim_mae_raw = np.mean(np.abs(pred_raw - y_test_raw), axis=0)
    return {
        "mae_transformed": mae_t,
        "mse_transformed": mse_t,
        "mae_raw": mae_raw,
        "mse_raw": mse_raw,
        "mae_raw_dim0": float(per_dim_mae_raw[0]) if per_dim_mae_raw.size > 0 else 0.0,
        "mae_raw_species_mean": float(np.mean(per_dim_mae_raw[1:])) if per_dim_mae_raw.size > 1 else 0.0,
        "mae_raw_species_max": float(np.max(per_dim_mae_raw[1:])) if per_dim_mae_raw.size > 1 else 0.0,
        "num_test": int(x_test_raw.shape[0]),
    }


def _parse_oracle_source(src: str) -> tuple[int, int]:
    m_ok = re.search(r"ok=(\d+)", src)
    m_fail = re.search(r"fail=(\d+)", src)
    ok = int(m_ok.group(1)) if m_ok else 0
    fail = int(m_fail.group(1)) if m_fail else 0
    return ok, fail


def _build_oracle_targets(
    x_raw: np.ndarray,
    target_dim: int,
    oracle_cfg: Dict,
    batch_size: int,
    report_dir: Path,
    device: torch.device,
) -> tuple[np.ndarray, Dict]:
    y_list = []
    ok_total = 0
    fail_total = 0
    for i in range(0, x_raw.shape[0], batch_size):
        xb = torch.from_numpy(x_raw[i : i + batch_size].astype(np.float32)).to(device)
        yb, src = get_true_prediction(
            xb,
            target_dim=target_dim,
            mechanism_path=str(oracle_cfg.get("mechanism_path", "mechanism/Burke2012_s9r23.yaml")),
            time_step=float(oracle_cfg.get("time_step", 1e-7)),
            reference_pressure=oracle_cfg.get("reference_pressure", None),
        )
        ok, fail = _parse_oracle_source(src)
        ok_total += ok
        fail_total += fail
        y_list.append(yb.detach().cpu().numpy())
    y = np.concatenate(y_list, axis=0).astype(np.float32)
    np.save(report_dir / "generated_oracle_targets.npy", y)
    stats = {
        "oracle_ok": int(ok_total),
        "oracle_fail": int(fail_total),
        "oracle_fail_ratio": float(fail_total / max(1, ok_total + fail_total)),
        "num_generated": int(x_raw.shape[0]),
    }
    return y, stats


def _distribution_stats(train_x: np.ndarray, gen_x: np.ndarray) -> Dict:
    temp_train = train_x[:, 0]
    temp_gen = gen_x[:, 0]
    species_train = train_x[:, 1:]
    species_gen = gen_x[:, 1:]
    lo = species_train.min(axis=0)
    hi = species_train.max(axis=0)
    violate = (species_gen < lo[None, :]) | (species_gen > hi[None, :])
    return {
        "temp_mean_train": float(temp_train.mean()),
        "temp_mean_gen": float(temp_gen.mean()),
        "temp_std_train": float(temp_train.std()),
        "temp_std_gen": float(temp_gen.std()),
        "temp_mean_abs_diff": float(abs(temp_gen.mean() - temp_train.mean())),
        "temp_std_ratio_gen_over_train": float(temp_gen.std() / max(1e-12, temp_train.std())),
        "species_out_of_range_ratio": float(violate.mean()) if violate.size > 0 else 0.0,
    }


def _write_comparison_md(report_dir: Path, baseline: Dict, augmented: Dict | None, details: Dict) -> None:
    lines = ["# DNN有效性对比结果", ""]
    lines.append("## 数据切分")
    lines.append(f"- train_size: {details['train_size']}")
    lines.append(f"- test_size: {details['test_size']}")
    lines.append("")
    lines.append("## 基线DNN（55k真实）")
    lines.append(f"- mae_raw: {baseline['mae_raw']:.6f}")
    lines.append(f"- mse_raw: {baseline['mse_raw']:.6f}")
    lines.append(f"- mae_raw_dim0: {baseline['mae_raw_dim0']:.6f}")
    lines.append(f"- mae_raw_species_mean: {baseline['mae_raw_species_mean']:.6f}")
    lines.append("")
    if augmented is not None:
        lines.append("## 增强DNN（55k真实 + 60k生成）")
        lines.append(f"- mae_raw: {augmented['mae_raw']:.6f}")
        lines.append(f"- mse_raw: {augmented['mse_raw']:.6f}")
        lines.append(f"- mae_raw_dim0: {augmented['mae_raw_dim0']:.6f}")
        lines.append(f"- mae_raw_species_mean: {augmented['mae_raw_species_mean']:.6f}")
        lines.append("")
        delta_mae = augmented["mae_raw"] - baseline["mae_raw"]
        delta_mse = augmented["mse_raw"] - baseline["mse_raw"]
        lines.append("## 同口径差值（增强 - 基线）")
        lines.append(f"- delta_mae_raw: {delta_mae:.6f}")
        lines.append(f"- delta_mse_raw: {delta_mse:.6f}")
        lines.append(f"- relative_mae_change_pct: {100.0 * delta_mae / max(1e-12, baseline['mae_raw']):.4f}%")
        lines.append(f"- relative_mse_change_pct: {100.0 * delta_mse / max(1e-12, baseline['mse_raw']):.4f}%")
        lines.append("")
        lines.append(f"- conclusion: {'有效（增强优于基线）' if delta_mae < 0 else '退化（增强劣于基线）'}")
    (report_dir / "comparison.md").write_text("\n".join(lines), encoding="utf-8")


def _write_degradation_analysis(report_dir: Path, baseline: Dict, augmented: Dict, oracle_stats: Dict, dist_stats: Dict) -> None:
    delta_mae = augmented["mae_raw"] - baseline["mae_raw"]
    lines = ["# 增强退化分析与优化方案", ""]
    lines.append("## 现象")
    lines.append(f"- 基线 mae_raw={baseline['mae_raw']:.6f}，增强 mae_raw={augmented['mae_raw']:.6f}，差值={delta_mae:.6f}")
    lines.append("")
    lines.append("## 证据")
    lines.append(f"- 生成样本温度均值差: {dist_stats['temp_mean_abs_diff']:.6f}")
    lines.append(f"- 生成/真实温度标准差比: {dist_stats['temp_std_ratio_gen_over_train']:.6f}")
    lines.append(f"- 生成species越界比: {dist_stats['species_out_of_range_ratio']:.6f}")
    lines.append(f"- 伪标签oracle失败率: {oracle_stats['oracle_fail_ratio']:.6f}")
    lines.append("")
    lines.append("## 优化方案")
    lines.append("- 网络结构：保持主干不变，增加温度头与species头双头回归，并共享前两层。")
    lines.append("- 优化逻辑：对生成样本按oracle置信度加权，低置信样本降权到0.2以下。")
    lines.append("- 训练顺序：先55k真实预训练，再1:1混合微调，最后仅真实样本短回灌2-3个epoch。")
    lines.append("- 采样策略：按温度分桶重采样生成样本，拉齐与55k真实训练分布。")
    (report_dir / "degradation_analysis.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    args = _build_parser().parse_args()
    cfg = load_config(args.config)
    set_seed(int(args.seed))
    device = _select_device(args.device)

    report_dir = Path(args.report_dir) if args.report_dir else Path("outputs") / "reports" / f"dnn_effectiveness_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    report_dir.mkdir(parents=True, exist_ok=True)

    x_all = np.load(cfg["quality"]["regression_input_path"]).astype(np.float32)
    y_all = np.load(cfg["quality"]["regression_target_path"]).astype(np.float32)
    total_size = min(int(args.total_size), x_all.shape[0], y_all.shape[0])
    x_all = x_all[:total_size]
    y_all = y_all[:total_size]

    split_npz = Path(args.split_npz) if args.split_npz else (report_dir / "split_indices.npz")
    if split_npz.exists():
        data = np.load(split_npz)
        train_idx = data["train_idx"]
        test_idx = data["test_idx"]
    else:
        train_idx, test_idx = _split_indices(total_size, int(args.test_size), int(args.seed))
        split_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(split_npz, train_idx=train_idx, test_idx=test_idx)

    x_train_raw = x_all[train_idx]
    y_train_raw = y_all[train_idx]
    x_test_raw = x_all[test_idx]
    y_test_raw = y_all[test_idx]

    tf_cfg = cfg["transform"]
    x_tf_base, y_tf_base = _fit_transforms(
        x_train_raw=x_train_raw,
        y_train_raw=y_train_raw,
        use_bct=bool(tf_cfg["use_bct"]),
        bct_epsilon=float(tf_cfg["bct_epsilon"]),
        standardize=bool(tf_cfg["standardize"]),
        disable_input_dim0_bct=bool(tf_cfg.get("disable_input_dim0_bct", False)),
    )
    x_train_t_base = x_tf_base.transform(x_train_raw).astype(np.float32)
    y_train_t_base = y_tf_base.transform(y_train_raw).astype(np.float32)
    reg_epochs = int(cfg["train"]["three_stage"].get("reg_pretrain_epochs", cfg["train"].get("epochs_quality", 3)))
    batch_size = int(cfg["data"]["batch_size"])
    baseline_net = _train_regressor(
        x_train_t=x_train_t_base,
        y_train_t=y_train_t_base,
        model_cfg=cfg["model"],
        optim_cfg=cfg["optim"],
        epochs=reg_epochs,
        batch_size=batch_size,
        device=device,
    )
    baseline_metrics = _eval_regressor(
        net=baseline_net,
        x_tf=x_tf_base,
        y_tf=y_tf_base,
        x_test_raw=x_test_raw,
        y_test_raw=y_test_raw,
        device=device,
    )
    save_json(baseline_metrics, report_dir / "baseline_metrics.json")

    result = {
        "split_npz": str(split_npz),
        "train_size": int(x_train_raw.shape[0]),
        "test_size": int(x_test_raw.shape[0]),
        "baseline_metrics": baseline_metrics,
    }
    save_json(result, report_dir / "result.json")

    if args.generated_path:
        gen_x = np.load(args.generated_path).astype(np.float32)
        if gen_x.ndim != 2:
            raise ValueError(f"generated_path 数据维度错误: {gen_x.shape}")
        if gen_x.shape[1] != x_train_raw.shape[1]:
            raise ValueError(f"generated_path 特征维度不一致: {gen_x.shape[1]} vs {x_train_raw.shape[1]}")
        oracle_cfg = dict(cfg.get("quality", {}).get("oracle", {}))
        y_gen, oracle_stats = _build_oracle_targets(
            x_raw=gen_x,
            target_dim=y_train_raw.shape[1],
            oracle_cfg=oracle_cfg,
            batch_size=int(args.oracle_batch_size),
            report_dir=report_dir,
            device=device,
        )
        x_aug_raw = np.concatenate([x_train_raw, gen_x], axis=0)
        y_aug_raw = np.concatenate([y_train_raw, y_gen], axis=0)
        x_tf_aug, y_tf_aug = _fit_transforms(
            x_train_raw=x_aug_raw,
            y_train_raw=y_aug_raw,
            use_bct=bool(tf_cfg["use_bct"]),
            bct_epsilon=float(tf_cfg["bct_epsilon"]),
            standardize=bool(tf_cfg["standardize"]),
            disable_input_dim0_bct=bool(tf_cfg.get("disable_input_dim0_bct", False)),
        )
        x_train_t_aug = x_tf_aug.transform(x_aug_raw).astype(np.float32)
        y_train_t_aug = y_tf_aug.transform(y_aug_raw).astype(np.float32)
        aug_net = _train_regressor(
            x_train_t=x_train_t_aug,
            y_train_t=y_train_t_aug,
            model_cfg=cfg["model"],
            optim_cfg=cfg["optim"],
            epochs=reg_epochs,
            batch_size=batch_size,
            device=device,
        )
        augmented_metrics = _eval_regressor(
            net=aug_net,
            x_tf=x_tf_aug,
            y_tf=y_tf_aug,
            x_test_raw=x_test_raw,
            y_test_raw=y_test_raw,
            device=device,
        )
        dist_stats = _distribution_stats(train_x=x_train_raw, gen_x=gen_x)
        comparison = {
            "baseline": baseline_metrics,
            "augmented": augmented_metrics,
            "delta_mae_raw": float(augmented_metrics["mae_raw"] - baseline_metrics["mae_raw"]),
            "delta_mse_raw": float(augmented_metrics["mse_raw"] - baseline_metrics["mse_raw"]),
            "relative_mae_change_pct": float(
                100.0 * (augmented_metrics["mae_raw"] - baseline_metrics["mae_raw"]) / max(1e-12, baseline_metrics["mae_raw"])
            ),
            "relative_mse_change_pct": float(
                100.0 * (augmented_metrics["mse_raw"] - baseline_metrics["mse_raw"]) / max(1e-12, baseline_metrics["mse_raw"])
            ),
            "oracle_stats": oracle_stats,
            "distribution_stats": dist_stats,
            "conclusion": "improved" if augmented_metrics["mae_raw"] < baseline_metrics["mae_raw"] else "degraded",
        }
        save_json(augmented_metrics, report_dir / "augmented_metrics.json")
        save_json(comparison, report_dir / "comparison.json")
        _write_comparison_md(report_dir, baseline_metrics, augmented_metrics, {"train_size": int(x_train_raw.shape[0]), "test_size": int(x_test_raw.shape[0])})
        if comparison["conclusion"] == "degraded":
            _write_degradation_analysis(
                report_dir=report_dir,
                baseline=baseline_metrics,
                augmented=augmented_metrics,
                oracle_stats=oracle_stats,
                dist_stats=dist_stats,
            )
        result["augmented_metrics"] = augmented_metrics
        result["comparison"] = comparison
    else:
        _write_comparison_md(report_dir, baseline_metrics, None, {"train_size": int(x_train_raw.shape[0]), "test_size": int(x_test_raw.shape[0])})

    save_json(result, report_dir / "result.json")
    print(json.dumps({"report_dir": str(report_dir), **result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
