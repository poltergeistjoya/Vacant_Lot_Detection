"""Model-specific configuration for pixel classifiers (tree-based and deep learning)."""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, model_validator


class SamplingConfig(BaseModel):
    """Pixel reservoir sampling + true distribution for class weight correction."""
    n_vacant: int = 5_000_000
    n_nonvacant: int = 15_000_000
    # True counts from Queens training split — used to compute corrected class weights.
    true_vacant_count: int = 19_700_000
    true_nonvacant_count: int = 566_000_000
    random_state: int = 42

    @property
    def vacant_weight(self) -> float:
        """Weight for vacant class to correct for undersampling."""
        return self.true_vacant_count / self.n_vacant

    @property
    def nonvacant_weight(self) -> float:
        """Weight for non-vacant class to correct for undersampling."""
        return self.true_nonvacant_count / self.n_nonvacant


class RFModelConfig(BaseModel):
    type: Literal["random_forest"]
    n_estimators: int = 200
    max_depth: int = 20
    min_samples_leaf: int = 50
    n_jobs: int = 4
    random_state: int = 42


class LGBMModelConfig(BaseModel):
    type: Literal["lightgbm"]
    n_estimators: int = 500
    max_depth: int = 12
    learning_rate: float = 0.05
    num_leaves: int = 63
    min_child_samples: int = 50
    n_jobs: int = -1
    random_state: int = 42


# ---------------------------------------------------------------------------
# Deep learning model configs
# ---------------------------------------------------------------------------

class DLModelConfig(BaseModel):
    type: Literal["unet", "deeplabv3plus"]
    encoder_name: str = "resnet18"
    encoder_weights: str | None = None  # "imagenet" or None
    in_channels: int = 10
    classes: int = 1
    decoder_channels: list[int] = [256, 128, 64, 32, 16]  # smp UNet decoder width


class DLTrainingConfig(BaseModel):
    batch_size: int = 4
    learning_rate: float = 0.001
    max_epochs: int = 50
    weight_decay: float = 0.0001
    patience: int = 10
    accumulation_steps: int = 1
    num_workers: int = 4  # DataLoader workers for prefetching patches (macOS: spawn-safe)
    seed: int = 42  # Random seed for reproducibility


class DLLossConfig(BaseModel):
    pos_weight: float = 10.0
    bce_weight: float = 0.5
    dice_weight: float = 0.5
    soft_positive_weight: float = 0.0
    soft_positive_target: float = 0.4


# ---------------------------------------------------------------------------
# Top-level training configs
# ---------------------------------------------------------------------------

class TreeTrainConfig(BaseModel):
    """Top-level training config for RF / LightGBM."""
    model: dict  # Raw dict; discriminated by model.type
    sampling: SamplingConfig = SamplingConfig()
    output_dir: str = "outputs/models"
    note: str = ""  # Human-readable note; override with VACANT_LOT_RUN_NOTE env var

    _shared_root: Optional[Path] = None

    @model_validator(mode="after")
    def parse_model(self):
        """Replace raw model dict with typed config based on type field."""
        model_type = self.model.get("type")
        if model_type == "random_forest":
            self.model = RFModelConfig(**self.model)
        elif model_type == "lightgbm":
            self.model = LGBMModelConfig(**self.model)
        else:
            raise ValueError(f"Unknown tree model type: {model_type!r}. Expected 'random_forest' or 'lightgbm'.")
        return self

    def get_output_dir(self, shared_root: Path) -> Path:
        return shared_root / self.output_dir


class DLTrainConfig(BaseModel):
    """Top-level training config for UNet / DeepLabV3+."""
    model: dict  # Raw dict; discriminated by model.type
    training: DLTrainingConfig = DLTrainingConfig()
    loss: DLLossConfig = DLLossConfig()
    output_dir: str = "outputs/models"
    note: str = ""  # Human-readable note; override with VACANT_LOT_RUN_NOTE env var

    _shared_root: Optional[Path] = None

    @model_validator(mode="after")
    def parse_model(self):
        """Replace raw model dict with typed DLModelConfig."""
        model_type = self.model.get("type")
        if model_type not in ("unet", "deeplabv3plus"):
            raise ValueError(f"Unknown DL model type: {model_type!r}. Expected 'unet' or 'deeplabv3plus'.")
        self.model = DLModelConfig(**self.model)
        return self

    def get_output_dir(self, shared_root: Path) -> Path:
        return shared_root / self.output_dir


# Backward-compat alias — existing RF/LGBM scripts can still import TrainConfig
TrainConfig = TreeTrainConfig
