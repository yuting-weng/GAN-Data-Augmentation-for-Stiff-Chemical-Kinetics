from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev


def _parse_runs(items: list[str]) -> list[tuple[str, Path]]:
    runs: list[tuple[str, Path]] = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"run 参数格式应为 label=path，实际: {item}")
        label, path = item.split("=", 1)
        runs.append((label.strip(), Path(path).expanduser()))
    return runs


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"日志为空: {path}")
    return rows


def _avg(rows: list[dict], key: str) -> float:
    vals = [float(r[key]) for r in rows if key in r]
    return float(mean(vals)) if vals else float("nan")


def _std(rows: list[dict], key: str) -> float:
    vals = [float(r[key]) for r in rows if key in r]
    return float(pstdev(vals)) if len(vals) > 1 else 0.0


def _max_abs(rows: list[dict], key: str) -> float:
    vals = [abs(float(r[key])) for r in rows if key in r]
    return float(max(vals)) if vals else float("nan")


def _collect_one(label: str, run_dir: Path, last_n: int) -> dict:
    log_path = run_dir / "gan_train_three_stage.jsonl"
    rows = _read_jsonl(log_path)
    tail = rows[-min(last_n, len(rows)) :]
    first = rows[0]
    last = rows[-1]
    out = {
        "label": label,
        "run_dir": str(run_dir),
        "steps": len(rows),
        "epoch_last": int(last.get("epoch", -1)),
        "g_share_quality_mean": _avg(tail, "g_share_quality"),
        "g_share_phys_mean": _avg(tail, "g_share_phys"),
        "g_share_wgan_mean": _avg(tail, "g_share_wgan"),
        "loss_qcls_mean": _avg(tail, "loss_qcls"),
        "cls_acc_mean": _avg(tail, "cls_acc"),
        "gp_mean": _avg(tail, "gp"),
        "gp_last": float(last.get("gp", float("nan"))),
        "gp_first": float(first.get("gp", float("nan"))),
        "mass_mean": _avg(tail, "mass"),
        "nonneg_mean": _avg(tail, "nonneg"),
        "loss_g_total_mean": _avg(tail, "loss_g_total"),
        "loss_g_total_std": _std(tail, "loss_g_total"),
        "real_minus_fake_mean": _avg(tail, "real_score") - _avg(tail, "fake_score"),
        "species_clip_ratio_mean": _avg(tail, "species_clip_ratio"),
        "q_reg_l1_drift_max_abs": _max_abs(rows, "q_reg_l1_drift"),
    }
    return out


def _write_csv(rows: list[dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(v) -> str:
    if isinstance(v, float):
        if v != v:
            return "nan"
        return f"{v:.6g}"
    return str(v)


def _write_md(rows: list[dict], out_md: Path) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "label",
        "g_share_quality_mean",
        "g_share_phys_mean",
        "g_share_wgan_mean",
        "cls_acc_mean",
        "loss_qcls_mean",
        "gp_mean",
        "real_minus_fake_mean",
        "species_clip_ratio_mean",
        "q_reg_l1_drift_max_abs",
        "loss_g_total_mean",
        "loss_g_total_std",
    ]
    lines = []
    lines.append("# 短实验矩阵对比")
    lines.append("")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        lines.append("| " + " | ".join(_fmt(r[h]) for h in headers) + " |")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description="汇总短实验矩阵对比")
    p.add_argument("--runs", nargs="+", required=True, help="格式: A1=outputs/train_gan_xxx")
    p.add_argument("--last_n", type=int, default=200)
    p.add_argument("--out_csv", type=str, default="outputs/reports/short_matrix_comparison.csv")
    p.add_argument("--out_md", type=str, default="outputs/reports/short_matrix_comparison.md")
    args = p.parse_args()

    runs = _parse_runs(args.runs)
    rows = [_collect_one(label, run_dir, args.last_n) for label, run_dir in runs]
    _write_csv(rows, Path(args.out_csv))
    _write_md(rows, Path(args.out_md))
    print({"out_csv": args.out_csv, "out_md": args.out_md, "rows": len(rows)})


if __name__ == "__main__":
    main()

