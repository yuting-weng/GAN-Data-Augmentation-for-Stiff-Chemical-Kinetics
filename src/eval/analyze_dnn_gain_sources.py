from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, TensorDataset

from src.data.transforms import BCTStandardizer
from src.models.quality_dnn import QualityDNN
from src.utils import adapt_hidden_dims, load_config, save_json, set_seed


@dataclass
class RegimeResult:
    name: str
    best_epoch: int
    stop_epoch: int
    best_val_mae_raw: float
    best_mae_raw: float
    best_mse_raw: float
    train_core_size: int
    val_size: int
    curve: List[Dict]
    per_sample_abs_err_raw: np.ndarray


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="固定早停+最佳checkpoint策略的60k/120k/200k容量复核")
    p.add_argument("--config", type=str, default="configs/exp_best_real55k_sweep.yaml")
    p.add_argument("--split_npz", type=str, default="outputs/reports/hparam_sweep_real55k_20260326/split_indices.npz")
    p.add_argument("--gen60_path", type=str, default="outputs/generate_dataset_20260326_022359/generated/generated_dataset_60000_nofilter.npy")
    p.add_argument("--gen60_target", type=str, default="outputs/reports/hparam_sweep_real55k_20260326/T1_low_cond/eval_60k/generated_oracle_targets.npy")
    p.add_argument("--gen120_path", type=str, default=None)
    p.add_argument("--gen120_target", type=str, default=None)
    p.add_argument("--gen200_path", type=str, default="outputs/generate_dataset_20260326_050057/generated/generated_dataset_200000_nofilter.npy")
    p.add_argument("--gen200_target", type=str, default="outputs/reports/best_combo_200k_eval_20260326/generated_oracle_targets.npy")
    p.add_argument("--report_dir", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--max_epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--min_epochs", type=int, default=8)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--min_delta", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=42)
    return p


def _select_device(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("请求CUDA但当前不可用。")
        return torch.device("cuda")
    return torch.device("cpu")


def _fit_transforms(x_train_raw: np.ndarray, y_train_raw: np.ndarray, tf_cfg: Dict) -> tuple[BCTStandardizer, BCTStandardizer]:
    mask = None
    if bool(tf_cfg.get("disable_input_dim0_bct", False)) and x_train_raw.shape[1] >= 1:
        mask = np.ones(x_train_raw.shape[1], dtype=bool)
        mask[0] = False
    x_tf = BCTStandardizer(
        use_bct=bool(tf_cfg["use_bct"]),
        bct_epsilon=float(tf_cfg["bct_epsilon"]),
        standardize=bool(tf_cfg["standardize"]),
        bct_feature_mask=mask,
    ).fit(x_train_raw)
    y_tf = BCTStandardizer(
        use_bct=bool(tf_cfg["use_bct"]),
        bct_epsilon=float(tf_cfg["bct_epsilon"]),
        standardize=bool(tf_cfg["standardize"]),
    ).fit(y_train_raw)
    return x_tf, y_tf


def _build_net(model_cfg: Dict, input_dim: int, output_dim: int, device: torch.device) -> QualityDNN:
    cfg = adapt_hidden_dims(dict(model_cfg), input_dim)
    net = QualityDNN(
        input_dim=input_dim,
        hidden_dims=cfg["quality_hidden_dims"],
        mode="error_regression",
        activation=cfg.get("activation", "gelu"),
        output_dim=output_dim,
    ).to(device)
    return net


def _eval_raw(
    net: QualityDNN,
    x_tf: BCTStandardizer,
    y_tf: BCTStandardizer,
    x_eval_raw: np.ndarray,
    y_eval_raw: np.ndarray,
    device: torch.device,
) -> tuple[float, float, np.ndarray]:
    x_t = x_tf.transform(x_eval_raw).astype(np.float32)
    with torch.no_grad():
        pred_t = net(torch.from_numpy(x_t).to(device)).detach().cpu().numpy()
    pred_raw = y_tf.inverse_transform(pred_t)
    abs_err = np.mean(np.abs(pred_raw - y_eval_raw), axis=1)
    mae = float(np.mean(abs_err))
    mse = float(np.mean((pred_raw - y_eval_raw) ** 2))
    return mae, mse, abs_err


def _split_train_val(n: int, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if n < 2:
        raise ValueError(f"训练样本过少，无法拆分train/val: n={n}")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    val_size = int(round(n * val_ratio))
    val_size = max(1, min(val_size, n - 1))
    val_idx = idx[:val_size]
    train_idx = idx[val_size:]
    return train_idx, val_idx


def _train_with_fixed_early_stop(
    name: str,
    x_train_raw: np.ndarray,
    y_train_raw: np.ndarray,
    x_test_raw: np.ndarray,
    y_test_raw: np.ndarray,
    cfg: Dict,
    max_epochs: int,
    batch_size: int,
    val_ratio: float,
    min_epochs: int,
    patience: int,
    min_delta: float,
    split_seed: int,
    device: torch.device,
) -> RegimeResult:
    idx_train, idx_val = _split_train_val(x_train_raw.shape[0], val_ratio=val_ratio, seed=split_seed)
    x_core = x_train_raw[idx_train]
    y_core = y_train_raw[idx_train]
    x_val = x_train_raw[idx_val]
    y_val = y_train_raw[idx_val]

    x_tf, y_tf = _fit_transforms(x_core, y_core, cfg["transform"])
    x_train_t = x_tf.transform(x_core).astype(np.float32)
    y_train_t = y_tf.transform(y_core).astype(np.float32)
    ds = TensorDataset(torch.from_numpy(x_train_t), torch.from_numpy(y_train_t))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=len(ds) >= batch_size)

    net = _build_net(cfg["model"], input_dim=x_train_t.shape[1], output_dim=y_train_t.shape[1], device=device)
    opt = torch.optim.AdamW(net.parameters(), lr=float(cfg["optim"]["lr_quality"]))
    curve = []

    best_val_mae = float("inf")
    best_epoch = -1
    best_state = None
    wait = 0
    stop_epoch = int(max_epochs)

    for ep in range(max_epochs):
        net.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = net(xb)
            loss = F.mse_loss(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        net.eval()
        val_mae, val_mse, _ = _eval_raw(net, x_tf, y_tf, x_val, y_val, device)
        test_mae, test_mse, _ = _eval_raw(net, x_tf, y_tf, x_test_raw, y_test_raw, device)
        curve.append(
            {
                "epoch": ep + 1,
                "val_mae_raw": val_mae,
                "val_mse_raw": val_mse,
                "test_mae_raw": test_mae,
                "test_mse_raw": test_mse,
            }
        )

        if val_mae < (best_val_mae - min_delta):
            best_val_mae = val_mae
            best_epoch = ep + 1
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if (ep + 1) >= min_epochs and wait >= patience:
            stop_epoch = ep + 1
            break

    if best_state is None:
        best_epoch = stop_epoch
        best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        best_val_mae = curve[-1]["val_mae_raw"]

    net.load_state_dict(best_state)
    mae_best, mse_best, per_sample = _eval_raw(net, x_tf, y_tf, x_test_raw, y_test_raw, device)
    return RegimeResult(
        name=name,
        best_epoch=best_epoch,
        stop_epoch=stop_epoch,
        best_val_mae_raw=best_val_mae,
        best_mae_raw=mae_best,
        best_mse_raw=mse_best,
        train_core_size=int(x_core.shape[0]),
        val_size=int(x_val.shape[0]),
        curve=curve,
        per_sample_abs_err_raw=per_sample,
    )


def _plot_curves(report_dir: Path, results: List[RegimeResult]) -> None:
    plt.figure(figsize=(9, 5))
    for r in results:
        xs = [p["epoch"] for p in r.curve]
        ys = [p["test_mae_raw"] for p in r.curve]
        plt.plot(xs, ys, marker="o", label=r.name)
        plt.scatter([r.best_epoch], [r.curve[r.best_epoch - 1]["test_mae_raw"]], s=70)
    plt.xlabel("Epoch")
    plt.ylabel("MAE (raw)")
    plt.title("测试集MAE曲线（固定早停 + 最佳checkpoint）")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(report_dir / "curve_test_mae_raw.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    for r in results:
        xs = [p["epoch"] for p in r.curve]
        ys = [p["val_mae_raw"] for p in r.curve]
        plt.plot(xs, ys, marker="o", label=r.name)
        plt.scatter([r.best_epoch], [r.best_val_mae_raw], s=70)
    plt.xlabel("Epoch")
    plt.ylabel("MAE (raw)")
    plt.title("验证集MAE曲线（用于checkpoint选择）")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(report_dir / "curve_val_mae_raw.png", dpi=180)
    plt.close()


def _plot_capacity_bar(report_dir: Path, results: List[RegimeResult]) -> None:
    labels = [r.name for r in results]
    maes = [r.best_mae_raw for r in results]
    plt.figure(figsize=(9, 5))
    bars = plt.bar(labels, maes, color=["#7f8c8d", "#3498db", "#1abc9c", "#2ecc71"])
    for b, v in zip(bars, maes):
        plt.text(b.get_x() + b.get_width() / 2.0, v, f"{v:.6f}", ha="center", va="bottom", fontsize=9)
    plt.ylabel("Best MAE (raw)")
    plt.title("容量复核：55k基线 vs 60k/120k/200k增强")
    plt.xticks(rotation=10)
    plt.tight_layout()
    plt.savefig(report_dir / "capacity_best_mae_bar.png", dpi=180)
    plt.close()


def _knn_distance(test_x: np.ndarray, train_x: np.ndarray) -> np.ndarray:
    nbrs = NearestNeighbors(n_neighbors=1, algorithm="auto")
    nbrs.fit(train_x)
    d, _ = nbrs.kneighbors(test_x, return_distance=True)
    return d[:, 0]


def _plot_distribution_coverage(report_dir: Path, d55: np.ndarray, d120: np.ndarray, d200: np.ndarray) -> None:
    qs = np.linspace(0.0, 1.0, 200)
    c55 = np.quantile(d55, qs)
    c120 = np.quantile(d120, qs)
    c200 = np.quantile(d200, qs)
    plt.figure(figsize=(8, 5))
    plt.plot(c55, qs, label="test -> 55k真实最近邻距离")
    plt.plot(c120, qs, label="test -> 55k+120k生成最近邻距离")
    plt.plot(c200, qs, label="test -> 55k+200k生成最近邻距离")
    plt.xlabel("最近邻距离")
    plt.ylabel("CDF")
    plt.title("测试集覆盖度对比（距离越小越好）")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(report_dir / "coverage_knn_cdf.png", dpi=180)
    plt.close()


def _plot_hard_sample_gain(report_dir: Path, baseline_err: np.ndarray, aug_err: np.ndarray, d55: np.ndarray, d200: np.ndarray) -> None:
    improv = baseline_err - aug_err
    gain_dist = d55 - d200
    bins = np.quantile(d55, [0.0, 0.25, 0.5, 0.75, 1.0])
    vals = []
    names = []
    for i in range(4):
        lo = bins[i]
        hi = bins[i + 1]
        if i < 3:
            m = (d55 >= lo) & (d55 < hi)
        else:
            m = (d55 >= lo) & (d55 <= hi)
        vals.append(float(np.mean(improv[m])))
        names.append(f"Q{i+1}")
    plt.figure(figsize=(8, 5))
    bars = plt.bar(names, vals, color="#9b59b6")
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2.0, v, f"{v:.5f}", ha="center", va="bottom", fontsize=9)
    plt.ylabel("MAE改善量（baseline-aug200）")
    plt.xlabel("按 baseline 覆盖难度分桶")
    plt.title("难样本区域增益")
    plt.tight_layout()
    plt.savefig(report_dir / "hard_region_gain.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.scatter(gain_dist, improv, s=8, alpha=0.35)
    corr = float(np.corrcoef(gain_dist, improv)[0, 1])
    plt.xlabel("覆盖改善量 (d55-d200)")
    plt.ylabel("单样本误差改善量")
    plt.title(f"覆盖改善与误差改善相关性 corr={corr:.4f}")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(report_dir / "coverage_vs_error_gain_scatter.png", dpi=180)
    plt.close()


def _write_md(report_dir: Path, summary: Dict) -> None:
    lines = ["# 容量三点复核（固定早停+最佳checkpoint）", ""]
    lines.append("## 策略设置")
    lines.append(f"- max_epochs: {summary['early_stop']['max_epochs']}")
    lines.append(f"- min_epochs: {summary['early_stop']['min_epochs']}")
    lines.append(f"- patience: {summary['early_stop']['patience']}")
    lines.append(f"- min_delta: {summary['early_stop']['min_delta']}")
    lines.append(f"- val_ratio: {summary['early_stop']['val_ratio']}")
    lines.append("")
    lines.append("## 最优测试集结果")
    for k, v in summary["best_mae_by_regime"].items():
        lines.append(f"- {k}: {v:.6f}")
    lines.append("")
    lines.append("## checkpoint与早停轨迹")
    for k, v in summary["checkpoint_policy_by_regime"].items():
        lines.append(f"- {k}: best_epoch={v['best_epoch']}, stop_epoch={v['stop_epoch']}, val_mae_best={v['best_val_mae_raw']:.6f}")
    lines.append("")
    lines.append("## 容量结论")
    lines.append(f"- 60k相对55k基线改善: {summary['gain_pct']['gan60_vs_base']:.2f}%")
    lines.append(f"- 120k相对55k基线改善: {summary['gain_pct']['gan120_vs_base']:.2f}%")
    lines.append(f"- 200k相对55k基线改善: {summary['gain_pct']['gan200_vs_base']:.2f}%")
    lines.append(f"- 120k相对60k增益: {summary['gain_pct']['gan120_vs_60']:.2f}%")
    lines.append(f"- 200k相对120k增益: {summary['gain_pct']['gan200_vs_120']:.2f}%")
    lines.append(f"- 覆盖改善与误差改善相关系数(55k->200k): {summary['coverage_error_corr']:.4f}")
    lines.append("")
    lines.append("## 收敛后判断")
    lines.append(summary["convergence_judgement"])
    (report_dir / "analysis.md").write_text("\n".join(lines), encoding="utf-8")


def _load_or_derive_120k(
    x200: np.ndarray,
    y200: np.ndarray,
    gen120_path: str | None,
    gen120_target: str | None,
) -> tuple[np.ndarray, np.ndarray, str]:
    if gen120_path is not None and gen120_target is not None:
        x120 = np.load(gen120_path).astype(np.float32)
        y120 = np.load(gen120_target).astype(np.float32)
        return x120, y120, "from_explicit_120k_files"
    n120 = min(120000, x200.shape[0], y200.shape[0])
    return x200[:n120], y200[:n120], "derived_from_200k_head"


def main() -> None:
    args = _build_parser().parse_args()
    cfg = load_config(args.config)
    set_seed(int(args.seed))
    device = _select_device(args.device)
    report_dir = Path(args.report_dir) if args.report_dir else Path("outputs") / "reports" / f"dnn_capacity_recheck_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    report_dir.mkdir(parents=True, exist_ok=True)

    x_all = np.load(cfg["quality"]["regression_input_path"]).astype(np.float32)
    y_all = np.load(cfg["quality"]["regression_target_path"]).astype(np.float32)
    split = np.load(args.split_npz)
    train_idx = split["train_idx"]
    test_idx = split["test_idx"]
    x_train55 = x_all[train_idx]
    y_train55 = y_all[train_idx]
    x_test = x_all[test_idx]
    y_test = y_all[test_idx]

    x_gen60 = np.load(args.gen60_path).astype(np.float32)
    y_gen60 = np.load(args.gen60_target).astype(np.float32)
    x_gen200 = np.load(args.gen200_path).astype(np.float32)
    y_gen200 = np.load(args.gen200_target).astype(np.float32)
    x_gen120, y_gen120, gen120_source = _load_or_derive_120k(
        x200=x_gen200,
        y200=y_gen200,
        gen120_path=args.gen120_path,
        gen120_target=args.gen120_target,
    )

    res_base = _train_with_fixed_early_stop(
        name="55k真实",
        x_train_raw=x_train55,
        y_train_raw=y_train55,
        x_test_raw=x_test,
        y_test_raw=y_test,
        cfg=cfg,
        max_epochs=int(args.max_epochs),
        batch_size=int(args.batch_size),
        val_ratio=float(args.val_ratio),
        min_epochs=int(args.min_epochs),
        patience=int(args.patience),
        min_delta=float(args.min_delta),
        split_seed=int(args.seed) + 11,
        device=device,
    )
    res_aug60 = _train_with_fixed_early_stop(
        name="55k+60k GAN",
        x_train_raw=np.concatenate([x_train55, x_gen60], axis=0),
        y_train_raw=np.concatenate([y_train55, y_gen60], axis=0),
        x_test_raw=x_test,
        y_test_raw=y_test,
        cfg=cfg,
        max_epochs=int(args.max_epochs),
        batch_size=int(args.batch_size),
        val_ratio=float(args.val_ratio),
        min_epochs=int(args.min_epochs),
        patience=int(args.patience),
        min_delta=float(args.min_delta),
        split_seed=int(args.seed) + 23,
        device=device,
    )
    res_aug120 = _train_with_fixed_early_stop(
        name="55k+120k GAN",
        x_train_raw=np.concatenate([x_train55, x_gen120], axis=0),
        y_train_raw=np.concatenate([y_train55, y_gen120], axis=0),
        x_test_raw=x_test,
        y_test_raw=y_test,
        cfg=cfg,
        max_epochs=int(args.max_epochs),
        batch_size=int(args.batch_size),
        val_ratio=float(args.val_ratio),
        min_epochs=int(args.min_epochs),
        patience=int(args.patience),
        min_delta=float(args.min_delta),
        split_seed=int(args.seed) + 37,
        device=device,
    )
    res_aug200 = _train_with_fixed_early_stop(
        name="55k+200k GAN",
        x_train_raw=np.concatenate([x_train55, x_gen200], axis=0),
        y_train_raw=np.concatenate([y_train55, y_gen200], axis=0),
        x_test_raw=x_test,
        y_test_raw=y_test,
        cfg=cfg,
        max_epochs=int(args.max_epochs),
        batch_size=int(args.batch_size),
        val_ratio=float(args.val_ratio),
        min_epochs=int(args.min_epochs),
        patience=int(args.patience),
        min_delta=float(args.min_delta),
        split_seed=int(args.seed) + 53,
        device=device,
    )

    results = [res_base, res_aug60, res_aug120, res_aug200]
    _plot_curves(report_dir, results)
    _plot_capacity_bar(report_dir, results)

    tf55, _ = _fit_transforms(x_train55, y_train55, cfg["transform"])
    x_test_t = tf55.transform(x_test).astype(np.float32)
    x_train55_t = tf55.transform(x_train55).astype(np.float32)
    x_train_55p120_t = tf55.transform(np.concatenate([x_train55, x_gen120], axis=0)).astype(np.float32)
    x_train_55p200_t = tf55.transform(np.concatenate([x_train55, x_gen200], axis=0)).astype(np.float32)
    d55 = _knn_distance(x_test_t, x_train55_t)
    d120 = _knn_distance(x_test_t, x_train_55p120_t)
    d200 = _knn_distance(x_test_t, x_train_55p200_t)
    _plot_distribution_coverage(report_dir, d55, d120, d200)
    _plot_hard_sample_gain(report_dir, res_base.per_sample_abs_err_raw, res_aug200.per_sample_abs_err_raw, d55, d200)

    best_mae = {r.name: r.best_mae_raw for r in results}
    gain_60 = 100.0 * (res_base.best_mae_raw - res_aug60.best_mae_raw) / max(1e-12, res_base.best_mae_raw)
    gain_120 = 100.0 * (res_base.best_mae_raw - res_aug120.best_mae_raw) / max(1e-12, res_base.best_mae_raw)
    gain_200 = 100.0 * (res_base.best_mae_raw - res_aug200.best_mae_raw) / max(1e-12, res_base.best_mae_raw)
    gain_120_vs_60 = 100.0 * (res_aug60.best_mae_raw - res_aug120.best_mae_raw) / max(1e-12, res_aug60.best_mae_raw)
    gain_200_vs_120 = 100.0 * (res_aug120.best_mae_raw - res_aug200.best_mae_raw) / max(1e-12, res_aug120.best_mae_raw)
    corr = float(np.corrcoef((d55 - d200), (res_base.per_sample_abs_err_raw - res_aug200.per_sample_abs_err_raw))[0, 1])
    best_regime = min(results, key=lambda r: r.best_mae_raw).name
    judgement = (
        f"在固定早停与统一checkpoint选择策略下，最优容量为{best_regime}；"
        "60k/120k/200k之间并非必然单调，最终以同口径best-checkpoint测试MAE作为容量结论。"
    )
    summary = {
        "best_mae_by_regime": best_mae,
        "best_epoch_by_regime": {r.name: r.best_epoch for r in results},
        "checkpoint_policy_by_regime": {
            r.name: {
                "best_epoch": r.best_epoch,
                "stop_epoch": r.stop_epoch,
                "best_val_mae_raw": r.best_val_mae_raw,
                "train_core_size": r.train_core_size,
                "val_size": r.val_size,
            }
            for r in results
        },
        "gain_pct": {
            "gan60_vs_base": gain_60,
            "gan120_vs_base": gain_120,
            "gan200_vs_base": gain_200,
            "gan120_vs_60": gain_120_vs_60,
            "gan200_vs_120": gain_200_vs_120,
        },
        "knn_median_55k": float(np.median(d55)),
        "knn_median_55k_plus_120k": float(np.median(d120)),
        "knn_median_55k_plus_200k": float(np.median(d200)),
        "coverage_error_corr": corr,
        "gen120_source": gen120_source,
        "convergence_judgement": judgement,
        "early_stop": {
            "max_epochs": int(args.max_epochs),
            "min_epochs": int(args.min_epochs),
            "patience": int(args.patience),
            "min_delta": float(args.min_delta),
            "val_ratio": float(args.val_ratio),
        },
    }
    curves = {r.name: r.curve for r in results}
    save_json(summary, report_dir / "summary.json")
    save_json(curves, report_dir / "curves.json")
    _write_md(report_dir, summary)
    print(json.dumps({"report_dir": str(report_dir), "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
