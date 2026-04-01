from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="55k真实+GAN生成 的多超参数搜索")
    p.add_argument("--base_config", type=str, default="configs/exp_real55k_cond_0p2.yaml")
    p.add_argument("--report_root", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--python_exe", type=str, default=sys.executable)
    p.add_argument("--small_target_size", type=int, default=60000)
    return p


def _load_yaml(path: Path) -> Dict[str, Any]:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _save_yaml(path: Path, data: Dict[str, Any]) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _deep_set(cfg: Dict[str, Any], dotted_key: str, value: Any) -> None:
    cur = cfg
    keys = dotted_key.split(".")
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def _run(cmd: List[str], cwd: Path) -> None:
    proc = subprocess.run(cmd, cwd=str(cwd), text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"命令失败: {' '.join(cmd)}")


def _new_dirs(root: Path, prefix: str, before: set[str]) -> List[Path]:
    all_now = {p.name for p in root.glob(f"{prefix}_*") if p.is_dir()}
    created = sorted(all_now - before)
    return [root / x for x in created]


def _trial_defs() -> List[Dict[str, Any]]:
    return [
        {
            "id": "T0_ref_cond02",
            "overrides": {
                "train.three_stage.loss_balance.lambda_cond": 0.2,
                "train.three_stage.loss_balance.lambda_phys": 0.05,
                "train.three_stage.loss_balance.lambda_wgan": 0.1,
                "quality.hybrid.w_classifier": 0.8,
                "quality.hybrid.w_regression": 0.2,
                "optim.lr_quality": 1.0e-3,
            },
        },
        {
            "id": "T1_low_cond",
            "overrides": {
                "train.three_stage.loss_balance.lambda_cond": 0.1,
                "train.three_stage.loss_balance.lambda_phys": 0.05,
                "train.three_stage.loss_balance.lambda_wgan": 0.1,
                "quality.hybrid.w_classifier": 0.8,
                "quality.hybrid.w_regression": 0.2,
                "optim.lr_quality": 1.0e-3,
            },
        },
        {
            "id": "T2_high_wgan_low_phys",
            "overrides": {
                "train.three_stage.loss_balance.lambda_cond": 0.2,
                "train.three_stage.loss_balance.lambda_phys": 0.03,
                "train.three_stage.loss_balance.lambda_wgan": 0.2,
                "quality.hybrid.w_classifier": 0.85,
                "quality.hybrid.w_regression": 0.15,
                "optim.lr_quality": 1.0e-3,
            },
        },
        {
            "id": "T3_high_phys_low_wgan",
            "overrides": {
                "train.three_stage.loss_balance.lambda_cond": 0.2,
                "train.three_stage.loss_balance.lambda_phys": 0.1,
                "train.three_stage.loss_balance.lambda_wgan": 0.05,
                "quality.hybrid.w_classifier": 0.75,
                "quality.hybrid.w_regression": 0.25,
                "optim.lr_quality": 1.0e-3,
            },
        },
        {
            "id": "T4_balanced_cond015",
            "overrides": {
                "train.three_stage.loss_balance.lambda_cond": 0.15,
                "train.three_stage.loss_balance.lambda_phys": 0.05,
                "train.three_stage.loss_balance.lambda_wgan": 0.15,
                "quality.hybrid.w_classifier": 0.8,
                "quality.hybrid.w_regression": 0.2,
                "optim.lr_quality": 5.0e-4,
            },
        },
    ]


def main() -> None:
    args = _build_parser().parse_args()
    repo = Path(__file__).resolve().parents[2]
    base_cfg_path = repo / args.base_config
    base_cfg = _load_yaml(base_cfg_path)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_root = Path(args.report_root) if args.report_root else (repo / "outputs" / "reports" / f"hparam_sweep_real55k_{stamp}")
    report_root.mkdir(parents=True, exist_ok=True)
    split_npz = report_root / "split_indices.npz"
    outputs_root = repo / str(base_cfg.get("output_root", "outputs"))
    results = []
    trials = _trial_defs()

    for trial in trials:
        tid = trial["id"]
        trial_dir = report_root / tid
        trial_dir.mkdir(parents=True, exist_ok=True)
        cfg = deepcopy(base_cfg)
        for k, v in trial["overrides"].items():
            _deep_set(cfg, k, v)
        cfg_path = trial_dir / "config.yaml"
        _save_yaml(cfg_path, cfg)

        before_train = {p.name for p in outputs_root.glob("train_gan_*") if p.is_dir()}
        _run([args.python_exe, "train.py", "--config", str(cfg_path), "--device", args.device, "train_gan"], cwd=repo)
        new_train = _new_dirs(outputs_root, "train_gan", before_train)
        if not new_train:
            raise RuntimeError(f"{tid} 未找到train_gan输出目录")
        train_dir = new_train[-1]

        before_gen = {p.name for p in outputs_root.glob("generate_dataset_*") if p.is_dir()}
        _run(
            [
                args.python_exe,
                "train.py",
                "--config",
                str(cfg_path),
                "--device",
                args.device,
                "generate_dataset",
                "--gan_checkpoint",
                str(train_dir / "generator.pt"),
                "--transform_stats",
                str(train_dir / "transform_stats.npz"),
                "--target_size",
                str(int(args.small_target_size)),
            ],
            cwd=repo,
        )
        new_gen = _new_dirs(outputs_root, "generate_dataset", before_gen)
        if not new_gen:
            raise RuntimeError(f"{tid} 未找到generate_dataset输出目录")
        gen_dir = new_gen[-1]
        gen_path = gen_dir / "generated" / "generated_dataset_60000_nofilter.npy"
        eval_dir = trial_dir / "eval_60k"

        cmd_eval = [
            args.python_exe,
            "-m",
            "src.eval.dnn_effectiveness_real_vs_gan",
            "--config",
            str(cfg_path),
            "--device",
            args.device,
            "--report_dir",
            str(eval_dir),
            "--generated_path",
            str(gen_path),
            "--split_npz",
            str(split_npz),
            "--total_size",
            "60000",
            "--test_size",
            "5000",
            "--oracle_batch_size",
            "256",
        ]
        _run(cmd_eval, cwd=repo)
        comp = json.loads((eval_dir / "comparison.json").read_text(encoding="utf-8"))
        rec = {
            "trial_id": tid,
            "train_dir": str(train_dir),
            "gen_dir": str(gen_dir),
            "mae_raw": float(comp["augmented"]["mae_raw"]),
            "mse_raw": float(comp["augmented"]["mse_raw"]),
            "delta_mae_raw": float(comp["delta_mae_raw"]),
            "relative_mae_change_pct": float(comp["relative_mae_change_pct"]),
            "config_path": str(cfg_path),
            **trial["overrides"],
        }
        results.append(rec)
        (trial_dir / "summary.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")

    results_sorted = sorted(results, key=lambda x: x["mae_raw"])
    best = results_sorted[0]
    (report_root / "sweep_results.json").write_text(json.dumps(results_sorted, ensure_ascii=False, indent=2), encoding="utf-8")
    with (report_root / "sweep_results.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results_sorted[0].keys()))
        w.writeheader()
        for r in results_sorted:
            w.writerow(r)
    (report_root / "best_trial.json").write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report_root": str(report_root), "best_trial": best}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
