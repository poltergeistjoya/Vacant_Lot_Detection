"""Configuration subpackage.

Training scripts use DataConfig + load_train_config.
EDA notebooks use CityConfig + load_config (legacy interface, unchanged).
"""
from .data_config import (
    DataConfig,
    GeometryConfig,
    ImageryConfig,
    LabelsConfig,
    ParcelConfig,
    PatchConfig,
    RasterConfig,
    SegmentationBboxConfig,
    SplitConfig,
)
from .model_config import (
    DLLossConfig,
    DLModelConfig,
    DLTrainConfig,
    DLTrainingConfig,
    LGBMModelConfig,
    RFModelConfig,
    SamplingConfig,
    TrainConfig,        # backward-compat alias for TreeTrainConfig
    TreeTrainConfig,
)
from .loader import _get_shared_root, load_data_config, load_train_config
from .legacy import CityConfig, GCPConfig, load_config, generate_run_readme

__all__ = [
    # New training interface
    "DataConfig",
    "TreeTrainConfig",
    "DLTrainConfig",
    "DLModelConfig",
    "DLTrainingConfig",
    "DLLossConfig",
    "RFModelConfig",
    "LGBMModelConfig",
    "SamplingConfig",
    "load_data_config",
    "load_train_config",
    # Backward-compat alias
    "TrainConfig",
    # Legacy EDA interface
    "CityConfig",
    "GCPConfig",
    "load_config",
    "generate_run_readme",
    # Shared utility
    "_get_shared_root",
]
