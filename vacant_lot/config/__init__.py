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
    LGBMModelConfig,
    RFModelConfig,
    SamplingConfig,
    TrainConfig,
)
from .loader import _get_shared_root, load_data_config, load_train_config
from .legacy import CityConfig, GCPConfig, load_config, generate_run_readme

__all__ = [
    # New training interface
    "DataConfig",
    "TrainConfig",
    "RFModelConfig",
    "LGBMModelConfig",
    "SamplingConfig",
    "load_data_config",
    "load_train_config",
    # Legacy EDA interface
    "CityConfig",
    "GCPConfig",
    "load_config",
    "generate_run_readme",
    # Shared utility
    "_get_shared_root",
]
