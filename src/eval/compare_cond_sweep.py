from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data.dataset import create_data_bundle
from src.data.samplers import sample_latent
from src.data.transforms import BCTStandardizer
from src.models.generator import Generator


def _metrics(real: np.ndarray, arr: np.ndarray) -> dict:
    mn = real[:, 1:].min(axis=0)
    mx = real[:, 1:].max(axis=0)
    return {
        "temp_mean": float(arr[:, 0].mean()),
        "temp_std": float(arr[:, 0].std()),
        "temp_q01": float(np.quantile(arr[:, 0], 0.01)),
        "temp_q99": float(np.quantile(arr[:, 0], 0.99)),
        "temp_mean_abs_diff": float(abs(arr[:, 0].mean() - real[:, 0].mean())),
        "temp_std_abs_diff": float(abs(arr[:, 0].std() - real[:, 0].std())),
        "species_below_rate": float((arr[:, 1:] < mn).mean()),
        "species_above_rate": float((arr[:, 1:] > mx).mean()),
        "species_std_mean": float(arr[:, 1:].std(axis=0).mean()),
    }


def _cond_temp_corr(run_dir: Path, max_n: int = 30000) -> float:
    cfg = json.loads((run_dir / "config_snapshot.json").read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = cfg["model"]
    condition_dim = int(cfg["data"].get("condition_dim", 0))
    g = Generator(
        latent_dim=model_cfg["latent_dim"],
        condition_dim=condition_dim,
        output_dim=10,
        hidden_dims=model_cfg["generator_hidden_dims"],
        activation=model_cfg.get("activation", "gelu"),
        condition_encoder_cfg=dict(model_cfg.get("generator", {}).get("condition_encoder", {})),
    ).to(device)
    g.load_state_dict(torch.load(run_dir / "generator.pt", map_location=device))
    g.eval()
    tf = BCTStandardizer().load(run_dir / "transform_stats.npz")
    bundle = create_data_bundle(
        npy_path=cfg["data"]["npy_path"],
        batch_size=512,
        val_ratio=cfg["data"]["val_ratio"],
        seed=cfg["seed"],
        num_workers=0,
        subset_size=None,
        use_bct=cfg["transform"]["use_bct"],
        bct_epsilon=cfg["transform"]["bct_epsilon"],
        standardize=cfg["transform"]["standardize"],
        disable_input_dim0_bct=cfg["transform"].get("disable_input_dim0_bct", False),
    )
    cond_t = []
    gen_t = []
    n = 0
    with torch.no_grad():
        for real_t in bundle.train_loader:
            real_t = real_t.to(device)
            cond = real_t[:, :condition_dim] if condition_dim > 0 else None
            z = sample_latent(real_t.shape[0], model_cfg["latent_dim"], device)
            fake = g(z, cond)
            fake_raw = tf.inverse_transform(fake.detach().cpu().numpy())
            real_raw = tf.inverse_transform(real_t.detach().cpu().numpy())
            cond_t.append(real_raw[:, 0])
            gen_t.append(fake_raw[:, 0])
            n += fake_raw.shape[0]
            if n >= max_n:
                break
    ct = np.concatenate(cond_t)[:max_n]
    gt = np.concatenate(gen_t)[:max_n]
    return float(np.corrcoef(ct, gt)[0, 1])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--real_path", required=True)
    p.add_argument("--baseline_arr", required=True)
    p.add_argument("--cond02_arr", required=True)
    p.add_argument("--cond01_arr", required=True)
    p.add_argument("--baseline_run", required=True)
    p.add_argument("--cond02_run", required=True)
    p.add_argument("--cond01_run", required=True)
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    real = np.load(args.real_path).astype(np.float32)
    arrs = {
        "baseline": np.load(args.baseline_arr).astype(np.float32),
        "cond02": np.load(args.cond02_arr).astype(np.float32),
        "cond01": np.load(args.cond01_arr).astype(np.float32),
    }
    runs = {
        "baseline": Path(args.baseline_run),
        "cond02": Path(args.cond02_run),
        "cond01": Path(args.cond01_run),
    }

    res = {}
    for k, arr in arrs.items():
        res[k] = _metrics(real, arr)
        res[k]["cond_temp_corr"] = _cond_temp_corr(runs[k])

    summary = {
        "real_ref": args.real_path,
        "runs": {k: str(v) for k, v in runs.items()},
        "metrics": res,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    keys = [
        "cond_temp_corr",
        "temp_std",
        "temp_std_abs_diff",
        "temp_mean_abs_diff",
        "temp_q01",
        "temp_q99",
        "species_below_rate",
        "species_above_rate",
        "species_std_mean",
    ]
    with (out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "baseline", "cond02", "cond01", "delta_cond01_base"])
        for key in keys:
            w.writerow([key, res["baseline"][key], res["cond02"][key], res["cond01"][key], res["cond01"][key] - res["baseline"][key]])

    lines = ["|metric|baseline|cond02|cond01|delta(cond01-base)|", "|---|---:|---:|---:|---:|"]
    for key in keys:
        b = res["baseline"][key]
        c2 = res["cond02"][key]
        c1 = res["cond01"][key]
        d = c1 - b
        lines.append("|{}|{:.6f}|{:.6f}|{:.6f}|{:.6f}|".format(key, b, c2, c1, d))
    (out_dir / "metrics.md").write_text("\n".join(lines), encoding="utf-8")

    plt.figure(figsize=(10, 5))
    bins = 80
    plt.hist(real[:, 0], bins=bins, density=True, alpha=0.30, label="real")
    plt.hist(arrs["baseline"][:, 0], bins=bins, density=True, alpha=0.30, label="baseline")
    plt.hist(arrs["cond02"][:, 0], bins=bins, density=True, alpha=0.30, label="cond02")
    plt.hist(arrs["cond01"][:, 0], bins=bins, density=True, alpha=0.30, label="cond01")
    plt.legend()
    plt.xlabel("Temperature")
    plt.ylabel("Density")
    plt.title("Temperature Distributions: real vs baseline/cond02/cond01")
    plt.tight_layout()
    plt.savefig(out_dir / "temp_hist_real_baseline_cond02_cond01.png", dpi=160)
    plt.close()

    mnames = ["cond_temp_corr", "temp_std", "species_below_rate", "species_above_rate"]
    x = np.arange(len(mnames))
    width = 0.25
    plt.figure(figsize=(11, 5))
    plt.bar(x - width, [res["baseline"][k] for k in mnames], width, label="baseline")
    plt.bar(x, [res["cond02"][k] for k in mnames], width, label="cond02")
    plt.bar(x + width, [res["cond01"][k] for k in mnames], width, label="cond01")
    plt.xticks(x, mnames, rotation=15)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "key_metrics_bar_threeway.png", dpi=160)
    plt.close()


if __name__ == "__main__":
    main()
