from __future__ import annotations

import argparse
import json
from pathlib import Path


def _parse_runs(items: list[str]) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"run 参数格式应为 label=path，实际: {item}")
        label, path = item.split("=", 1)
        out.append((label.strip(), Path(path).expanduser()))
    return out


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
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def _safe(v: float) -> float:
    if v != v:
        return 0.0
    return float(v)


def _collect_score(label: str, run_dir: Path, last_n: int, w_dom: float, w_stab: float, w_health: float) -> dict:
    rows = _read_jsonl(run_dir / "gan_train_three_stage.jsonl")
    tail = rows[-min(last_n, len(rows)) :]

    quality_share = _safe(_avg(tail, "g_share_quality"))
    wgan_share = _safe(_avg(tail, "g_share_wgan"))
    gp_mean = _safe(_avg(tail, "gp"))
    std_total = _safe(_avg([{ "x": abs(float(r.get("loss_g_total", 0.0)))} for r in tail], "x"))
    nonneg_mean = _safe(_avg(tail, "nonneg"))
    clip_mean = _safe(_avg(tail, "species_clip_ratio"))

    dominance = quality_share + wgan_share
    stability_penalty = gp_mean + 0.5 * std_total
    health_penalty = nonneg_mean + 0.5 * clip_mean
    score = w_dom * dominance - w_stab * stability_penalty - w_health * health_penalty

    cfg_path = run_dir / "config_snapshot.json"
    params = {}
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        lb = cfg.get("train", {}).get("three_stage", {}).get("loss_balance", {})
        params = {
            "lambda_quality": lb.get("lambda_quality"),
            "lambda_phys": lb.get("lambda_phys"),
            "lambda_wgan": lb.get("lambda_wgan"),
            "n_critic": cfg.get("train", {}).get("n_critic"),
        }

    return {
        "label": label,
        "run_dir": str(run_dir),
        "dominance": dominance,
        "quality_share": quality_share,
        "wgan_share": wgan_share,
        "gp_mean": gp_mean,
        "loss_total_abs_mean": std_total,
        "nonneg_mean": nonneg_mean,
        "species_clip_ratio_mean": clip_mean,
        "score": score,
        "params": params,
    }


def _write_md(rows: list[dict], out_md: Path) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# 最优参数评分结果")
    lines.append("")
    lines.append("| label | dominance | gp_mean | loss_total_abs_mean | species_clip_ratio_mean | score |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for r in rows:
        lines.append(
            f"| {r['label']} | {r['dominance']:.6f} | {r['gp_mean']:.6f} | "
            f"{r['loss_total_abs_mean']:.6f} | {r['species_clip_ratio_mean']:.6f} | {r['score']:.6f} |"
        )
    lines.append("")
    lines.append("## 参数")
    lines.append("")
    for r in rows:
        lines.append(f"- {r['label']}: {json.dumps(r['params'], ensure_ascii=False)}")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description="按对抗主导+稳定标准选择最优参数")
    p.add_argument("--runs", nargs="+", required=True, help="格式: label=outputs/train_gan_xxx")
    p.add_argument("--last_n", type=int, default=200)
    p.add_argument("--w_dom", type=float, default=1.0)
    p.add_argument("--w_stab", type=float, default=0.35)
    p.add_argument("--w_health", type=float, default=0.15)
    p.add_argument("--out_json", type=str, default="outputs/reports/best_param_selection.json")
    p.add_argument("--out_md", type=str, default="outputs/reports/best_param_selection.md")
    args = p.parse_args()

    runs = _parse_runs(args.runs)
    rows = [
        _collect_score(label, run_dir, args.last_n, args.w_dom, args.w_stab, args.w_health)
        for label, run_dir in runs
    ]
    rows_sorted = sorted(rows, key=lambda x: x["score"], reverse=True)
    result = {
        "weights": {"w_dom": args.w_dom, "w_stab": args.w_stab, "w_health": args.w_health},
        "best": rows_sorted[0],
        "ranking": rows_sorted,
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_md(rows_sorted, Path(args.out_md))
    print(json.dumps({"best_label": rows_sorted[0]["label"], "best_run_dir": rows_sorted[0]["run_dir"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

