import json
import torch
from pathlib import Path
from typing import Optional


def get_fork_directory(root: Path, fork: Optional[str] = None) -> Optional[Path]:
    if fork:
        return root / "fork" / fork
    else:
        return None


def get_model_config_json(root: Path) -> dict:
    with open(root / "config.json", "r") as f:
        config_dict = json.load(f)
    return config_dict


def get_external_config_json(root: Path) -> Optional[dict]:
    external_config_path = root / "external_config.json"
    if external_config_path.exists():
        with open(external_config_path, "r") as f:
            config_dict = json.load(f)
        return config_dict
    else:
        return None


def get_trainer_config_json(root: Path) -> dict:
    with open(root / "trainer_config.json", "r") as f:
        config_dict = json.load(f)
    return config_dict


def get_checkpoint(root: Path, checkpoint: Optional[str] = None) -> Path:
    if checkpoint:
        # check if the checkpoint exists
        if not (root / f"{checkpoint}.pth").exists():
            raise FileNotFoundError(f"checkpoint {checkpoint} not found")
        return root / f"{checkpoint}.pth"
    try:
        # 寻找文件夹下的最新的 checkpoint 的 name
        checkpoint_path = max(root.glob("*.pth"), key=lambda x: x.stat().st_ctime)
        # 去掉后缀
        return checkpoint_path
    except ValueError:
        raise FileNotFoundError(f"checkpoint not found in {root}")
