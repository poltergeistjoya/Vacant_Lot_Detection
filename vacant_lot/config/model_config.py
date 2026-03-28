"""Model-specific configuration for tree-based pixel classifiers."""
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


class TrainConfig(BaseModel):
    """Top-level training config (model + sampling + output dir)."""
    model: dict  # Raw dict; discriminated by model.type
    sampling: SamplingConfig = SamplingConfig()
    output_dir: str = "outputs/models"
    note: str = ""  # Human-readable note written to metrics.json; override with VACANT_LOT_RUN_NOTE env var

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
            raise ValueError(f"Unknown model type: {model_type!r}. Expected 'random_forest' or 'lightgbm'.")
        return self

    def get_output_dir(self, shared_root: Path) -> Path:
        return shared_root / self.output_dir
