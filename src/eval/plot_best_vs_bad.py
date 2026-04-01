from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"日志为空: {path}")
    return rows


def _tail(rows: list[dict], n: int) -> list[dict]:
    return rows[-min(n, len(rows)) :]


def _avg(rows: list[dict], key: str) -> float:
    vals = [float(r[key]) for r in rows if key in r]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _std(rows: list[dict], key: str) -> float:
    vals = [float(r[key]) for r in rows if key in r]
    if len(vals) <= 1:
        return 0.0
    m = sum(vals) / len(vals)
    return float((sum((x - m) ** 2 for x in vals) / len(vals)) ** 0.5)


def _collect_metrics(rows: list[dict], last_n: int) -> dict:
    t = _tail(rows, last_n)
    return {
        "g_share_quality_mean": _avg(t, "g_share_quality"),
        "g_share_phys_mean": _avg(t, "g_share_phys"),
        "g_share_wgan_mean": _avg(t, "g_share_wgan"),
        "gp_mean": _avg(t, "gp"),
        "real_minus_fake_mean": _avg(t, "real_score") - _avg(t, "fake_score"),
        "loss_g_total_mean": _avg(t, "loss_g_total"),
        "loss_g_total_std": _std(t, "loss_g_total"),
        "species_clip_ratio_mean": _avg(t, "species_clip_ratio"),
        "q_reg_l1_drift_max_abs": max(abs(float(r.get("q_reg_l1_drift", 0.0))) for r in rows),
    }


def _plot_curve(best: list[dict], bad: list[dict], key: str, out_path: Path, title: str) -> None:
    x1 = [int(r["step"]) for r in best if key in r]
    y1 = [float(r[key]) for r in best if key in r]
    x2 = [int(r["step"]) for r in bad if key in r]
    y2 = [float(r[key]) for r in bad if key in r]
    plt.figure(figsize=(8, 4))
    plt.plot(x1, y1, label="best", linewidth=1.3)
    plt.plot(x2, y2, label="A2_bad", linewidth=1.3)
    plt.xlabel("step")
    plt.ylabel(key)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def _write_table(best_m: dict, bad_m: dict, out_csv: Path, out_md: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    keys = list(best_m.keys())
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "best", "A2_bad", "delta(best-bad)"])
        for k in keys:
            b = best_m[k]
            a = bad_m[k]
            writer.writerow([k, b, a, b - a])

    lines = []
    lines.append("# best vs A2 指标对比")
    lines.append("")
    lines.append("| metric | best | A2_bad | delta(best-bad) |")
    lines.append("| --- | --- | --- | --- |")
    for k in keys:
        b = best_m[k]
        a = bad_m[k]
        lines.append(f"| {k} | {b:.6g} | {a:.6g} | {b-a:.6g} |")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description="可视化对比 best 与 A2_bad")
    p.add_argument("--best_run", type=str, required=True)
    p.add_argument("--bad_run", type=str, required=True)
    p.add_argument("--last_n", type=int, default=200)
    p.add_argument("--out_dir", type=str, default="outputs/reports/best_vs_a2")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    best_rows = _read_jsonl(Path(args.best_run) / "gan_train_three_stage.jsonl")
    bad_rows = _read_jsonl(Path(args.bad_run) / "gan_train_three_stage.jsonl")

    best_m = _collect_metrics(best_rows, args.last_n)
    bad_m = _collect_metrics(bad_rows, args.last_n)

    _plot_curve(best_rows, bad_rows, "g_share_quality", out_dir / "curve_g_share_quality.png", "g_share_quality")
    _plot_curve(best_rows, bad_rows, "g_share_wgan", out_dir / "curve_g_share_wgan.png", "g_share_wgan")
    _plot_curve(best_rows, bad_rows, "g_share_phys", out_dir / "curve_g_share_phys.png", "g_share_phys")
    _plot_curve(best_rows, bad_rows, "gp", out_dir / "curve_gp.png", "gp")
    _plot_curve(best_rows, bad_rows, "loss_g_total", out_dir / "curve_loss_g_total.png", "loss_g_total")

    _write_table(
        best_m,
        bad_m,
        out_dir / "best_vs_a2_metrics.csv",
        out_dir / "best_vs_a2_metrics.md",
    )
    summary = {
        "best_run": args.best_run,
        "bad_run": args.bad_run,
        "last_n": args.last_n,
        "best_metrics": best_m,
        "bad_metrics": bad_m,
    }
    (out_dir / "best_vs_a2_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

