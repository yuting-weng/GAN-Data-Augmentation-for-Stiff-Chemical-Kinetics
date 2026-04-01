from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from src.utils import save_json


def _pca2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.mean(x, axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    z = u[:, :2] * s[:2]
    return z.astype(np.float32)


def plot_distribution_comparison(
    real_data: np.ndarray,
    gen_data: np.ndarray,
    output_dir: str | Path,
    max_points_scatter: int = 20000,
) -> Dict:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    real = np.asarray(real_data, dtype=np.float32)
    gen = np.asarray(gen_data, dtype=np.float32)
    d = real.shape[1]

    stats = {
        "real_shape": list(real.shape),
        "gen_shape": list(gen.shape),
        "real_mean": real.mean(axis=0).astype(float).tolist(),
        "gen_mean": gen.mean(axis=0).astype(float).tolist(),
        "real_std": real.std(axis=0).astype(float).tolist(),
        "gen_std": gen.std(axis=0).astype(float).tolist(),
        "mean_abs_diff": np.abs(real.mean(axis=0) - gen.mean(axis=0)).astype(float).tolist(),
        "std_abs_diff": np.abs(real.std(axis=0) - gen.std(axis=0)).astype(float).tolist(),
    }
    save_json(stats, out_dir / "distribution_stats.json")

    ncols = 2
    nrows = (d + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, max(4, nrows * 3)))
    axes = np.array(axes).reshape(-1)
    for i in range(d):
        ax = axes[i]
        lo = float(min(real[:, i].min(), gen[:, i].min()))
        hi = float(max(real[:, i].max(), gen[:, i].max()))
        bins = np.linspace(lo, hi, 80)
        ax.hist(real[:, i], bins=bins, alpha=0.45, density=True, label="real")
        ax.hist(gen[:, i], bins=bins, alpha=0.45, density=True, label="generated")
        ax.set_title(f"feature_{i}")
    for j in range(d, len(axes)):
        axes[j].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "feature_hist_compare.png", dpi=180)
    plt.close(fig)

    n_real = min(max_points_scatter, real.shape[0])
    n_gen = min(max_points_scatter, gen.shape[0])
    rng = np.random.default_rng(42)
    real_sel = real[rng.choice(real.shape[0], size=n_real, replace=False)] if real.shape[0] > n_real else real
    gen_sel = gen[rng.choice(gen.shape[0], size=n_gen, replace=False)] if gen.shape[0] > n_gen else gen
    z_real = _pca2(real_sel)
    z_gen = _pca2(gen_sel)

    fig = plt.figure(figsize=(8, 6))
    plt.scatter(z_real[:, 0], z_real[:, 1], s=4, alpha=0.25, label="real")
    plt.scatter(z_gen[:, 0], z_gen[:, 1], s=4, alpha=0.25, label="generated")
    plt.legend()
    plt.title("PCA-2D Distribution Comparison")
    plt.tight_layout()
    fig.savefig(out_dir / "pca2_compare.png", dpi=180)
    plt.close(fig)

    return {
        "stats_file": str(out_dir / "distribution_stats.json"),
        "hist_plot": str(out_dir / "feature_hist_compare.png"),
        "pca_plot": str(out_dir / "pca2_compare.png"),
    }
