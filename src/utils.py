from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


@dataclass
class RuntimePaths:
    run_dir: Path
    checkpoints_dir: Path
    logs_dir: Path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_runtime_dirs(output_root: str, command: str) -> RuntimePaths:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_root) / f"{command}_{stamp}"
    checkpoints_dir = run_dir / "checkpoints"
    logs_dir = run_dir / "logs"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    return RuntimePaths(run_dir=run_dir, checkpoints_dir=checkpoints_dir, logs_dir=logs_dir)


def load_config(config_path: str) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except Exception as exc:
            raise RuntimeError("读取 YAML 需要安装 pyyaml: pip install pyyaml") from exc
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    raise ValueError(f"不支持的配置格式: {path.suffix}")


def save_json(data: Dict[str, Any], out_path: os.PathLike[str] | str) -> None:
    Path(out_path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(data: Dict[str, Any], out_path: os.PathLike[str] | str) -> None:
    with Path(out_path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def adapt_hidden_dims(model_cfg: Dict[str, Any], feature_dim: int) -> Dict[str, Any]:
    cfg = dict(model_cfg)
    g_base = list(cfg.get("generator_hidden_dims", [128, 256, 256, 128]))
    c_base = list(cfg.get("critic_hidden_dims", [128, 256, 128]))
    q_base = list(cfg.get("quality_hidden_dims", [256, 256, 128]))

    g_scale = [2, 4, 4, 2]
    c_scale = [2, 4, 2]
    q_scale = [4, 4, 2]

    cfg["generator_hidden_dims"] = [max(b, s * feature_dim) for b, s in zip(g_base, g_scale)]
    cfg["critic_hidden_dims"] = [max(b, s * feature_dim) for b, s in zip(c_base, c_scale)]
    cfg["quality_hidden_dims"] = [max(b, s * feature_dim) for b, s in zip(q_base, q_scale)]
    return cfg
