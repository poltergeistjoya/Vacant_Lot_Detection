"""Config loading utilities and shared root resolution."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

from .data_config import DataConfig
from .model_config import DLTrainConfig, TreeTrainConfig


def _get_shared_root() -> Path:
    """Auto-detect shared root (parent of all worktrees).

    Package lives at: <shared_root>/<worktree>/vacant_lot/config/loader.py
    parents[0] = config/
    parents[1] = vacant_lot/
    parents[2] = <worktree>/
    parents[3] = <shared_root>
    Override with VACANT_LOT_SHARED_ROOT env var.
    """
    env_root = os.environ.get("VACANT_LOT_SHARED_ROOT")
    if env_root:
        return Path(env_root).resolve()
    return Path(__file__).resolve().parents[3]


def _config_dir() -> Path:
    """Default config directory: <worktree>/config/"""
    return Path(__file__).resolve().parents[2] / "config"


def load_data_config(config_file: str = "data.yaml") -> DataConfig:
    """Load DataConfig from a YAML file.

    Args:
        config_file: Filename in <worktree>/config/ (default: data.yaml).

    Returns:
        Validated DataConfig with all paths resolved to absolute paths.
    """
    path = _config_dir() / config_file
    if not path.exists():
        raise FileNotFoundError(f"Data config not found: {path}")
    raw = yaml.safe_load(path.read_text())
    cfg = DataConfig(**raw)
    cfg._shared_root = _get_shared_root()
    return cfg


def load_train_config(config_file: str) -> tuple[DataConfig, TreeTrainConfig | DLTrainConfig]:
    """Load a model training config, resolving its data: reference.

    The model YAML must contain a ``data:`` key pointing to a data config
    filename (e.g. ``data: data.yaml``).

    Dispatches to TreeTrainConfig (random_forest / lightgbm) or DLTrainConfig
    (unet / deeplabv3plus) based on the ``model.type`` field.

    Args:
        config_file: Model config filename in <worktree>/config/.

    Returns:
        Tuple of (DataConfig, TreeTrainConfig | DLTrainConfig).
    """
    path = _config_dir() / config_file
    if not path.exists():
        raise FileNotFoundError(f"Train config not found: {path}")
    raw = yaml.safe_load(path.read_text())

    data_file = raw.pop("data", "data.yaml")
    data_cfg = load_data_config(data_file)

    model_type = raw.get("model", {}).get("type", "")
    if model_type in ("random_forest", "lightgbm"):
        train_cfg = TreeTrainConfig(**raw)
    elif model_type in ("unet", "deeplabv3plus"):
        train_cfg = DLTrainConfig(**raw)
    else:
        raise ValueError(
            f"Unknown model type: {model_type!r}. "
            "Expected 'random_forest', 'lightgbm', 'unet', or 'deeplabv3plus'."
        )

    return data_cfg, train_cfg
