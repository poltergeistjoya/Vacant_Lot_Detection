"""Legacy CityConfig preserved for EDA notebooks.

Training scripts should use DataConfig + TrainConfig from data_config.py
and model_config.py instead.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, ValidationError, model_validator

from ..logger import get_logger
from .loader import _get_shared_root

log = get_logger()


class GCPConfig(BaseModel):
    project_id: str = "vacant-lot-detection"
    bucket: str = "thesis_parcels"


class GeometryConfig(BaseModel):
    source: str = "tiger_counties"
    counties: list[str] | None = None
    state_fips: str | None = None
    path: str | None = None

    @model_validator(mode="after")
    def validate_source_fields(self):
        if self.source == "tiger_counties":
            if not self.counties:
                raise ValueError("counties is required for tiger_counties source")
            if not self.state_fips:
                raise ValueError("state_fips is required for tiger_counties source")
        elif self.source == "geojson":
            if not self.path:
                raise ValueError("path is required for geojson source")
        return self


class ParcelConfig(BaseModel):
    data_path: str
    layer: str
    id_column: str
    landuse_column: str
    vacant_codes: list[str]
    source_crs: str


class RasterConfig(BaseModel):
    output_crs: str
    resolution: float = 1.0


class GEEConfig(BaseModel):
    parcel_asset: Optional[str] = None
    export_prefix: Optional[str] = None


class NAIPConfig(BaseModel):
    collection_id: str = "USDA/NAIP/DOQQ"
    year: int = 2022
    start_date: Optional[str] = None
    end_date: Optional[str] = None

    @model_validator(mode="after")
    def set_date_range(self):
        if self.start_date is None:
            self.start_date = f"{self.year}-01-01"
        if self.end_date is None:
            self.end_date = f"{self.year}-12-31"
        return self


class SamplingConfig(BaseModel):
    total_samples: int = 25000
    min_pixels: int = 50
    vacant_min_fraction: float = 0.08
    random_state: int = 42


class ClusteringConfig(BaseModel):
    n_clusters: int = 5
    random_state: int = 42
    features: list[str] = [
        "B_mean", "G_mean", "R_mean", "N_mean",
        "NDVI_mean", "SAVI_mean", "Brightness_mean", "BareSoilProxy_mean"
    ]


class SplitConfig(BaseModel):
    strategy: str = "borough"
    train_boroughs: list[int] = [4]
    val_boroughs: list[int] = [2]
    test_boroughs: list[int] = [3]
    exclude_boroughs: list[int] = [1, 5]


class SegmentationConfig(BaseModel):
    bbox: Optional[tuple[float, float, float, float]] = None


class CityConfig(BaseModel):
    """Complete configuration for EDA pipeline. Used by EDA notebooks."""
    city: str
    city_name: Optional[str] = None
    run_name: Optional[str] = None

    gcp: GCPConfig = GCPConfig()
    geometry: GeometryConfig
    parcel: ParcelConfig
    raster: RasterConfig
    gee: GEEConfig = GEEConfig()
    naip: NAIPConfig = NAIPConfig()
    sampling: SamplingConfig = SamplingConfig()
    clustering: ClusteringConfig = ClusteringConfig()
    split: SplitConfig = SplitConfig()
    segmentation: SegmentationConfig = SegmentationConfig()

    @model_validator(mode="after")
    def set_derived_defaults(self):
        if self.city_name is None:
            self.city_name = self.city.upper()
        if self.run_name is None:
            self.run_name = self.city
        if self.gee.parcel_asset is None:
            self.gee.parcel_asset = f"projects/{self.gcp.project_id}/assets/{self.city}_parcels_raster"
        if self.gee.export_prefix is None:
            self.gee.export_prefix = f"eda/{self._run_key()}"
        return self

    def _run_key(self) -> str:
        if self.run_name and self.run_name != self.city:
            return f"{self.city}_{self.run_name}"
        return self.city

    def get_parcel_path(self) -> Path:
        return _get_shared_root() / self.parcel.data_path

    def get_output_dir(self) -> Path:
        return _get_shared_root() / "outputs" / "eda"

    def get_figures_dir(self) -> Path:
        return self.get_output_dir() / "figures"

    def get_data_dir(self) -> Path:
        return self.get_output_dir() / "data"

    def get_intermediaries_dir(self) -> Path:
        return self.get_output_dir() / "intermediaries"

    def ensure_output_dirs(self) -> None:
        for d in [self.get_output_dir(), self.get_figures_dir(),
                  self.get_data_dir(), self.get_intermediaries_dir()]:
            d.mkdir(parents=True, exist_ok=True)

    def get_final_figures_dir(self) -> Path:
        return _get_shared_root() / "outputs" / "figures"

    def get_naip_dir(self) -> Path:
        return _get_shared_root() / "data" / "imagery" / "naip" / self.city / str(self.naip.year)

    def get_naip_tiles_dir(self) -> Path:
        return self.get_naip_dir()

    def get_naip_vrt_path(self) -> Path:
        return self.get_naip_dir() / f"naip_{self.city}_{self.naip.year}.vrt"

    def get_seg_masks_dir(self) -> Path:
        """Legacy name — now points to outputs/labels/."""
        return _get_shared_root() / "outputs" / "labels"

    def get_modeling_dir(self) -> Path:
        return _get_shared_root() / "outputs" / "models"

    def get_modeling_models_dir(self) -> Path:
        return _get_shared_root() / "outputs" / "eda" / "parcel_classifier" / "models"

    def get_modeling_figures_dir(self) -> Path:
        return _get_shared_root() / "outputs" / "eda" / "parcel_classifier" / "figures"

    def get_modeling_data_dir(self) -> Path:
        return _get_shared_root() / "outputs" / "eda" / "parcel_classifier" / "data"

    def ensure_seg_output_dirs(self) -> None:
        self.get_naip_dir().mkdir(parents=True, exist_ok=True)
        self.get_seg_masks_dir().mkdir(parents=True, exist_ok=True)

    def ensure_modeling_dirs(self) -> None:
        for d in [self.get_modeling_models_dir(), self.get_modeling_figures_dir()]:
            d.mkdir(parents=True, exist_ok=True)


def generate_run_readme(config: CityConfig, output_dir: Path, stats: dict) -> Path:
    readme_path = output_dir / "README.md"
    content = f"""# {config.city_name} - {config.run_name}

**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Run Statistics

{chr(10).join(f'- **{k}**: {v}' for k, v in stats.items())}
"""
    readme_path.write_text(content)
    log.info(f"Generated README: {readme_path}")
    return readme_path


def load_config(config_file: str, config_dir: str | Path | None = None) -> CityConfig:
    """Load and validate a CityConfig from a YAML file (legacy EDA interface)."""
    if config_dir is None:
        # __file__ is vacant_lot/config/legacy.py → parents[2] = <worktree>
        config_dir = Path(__file__).resolve().parents[2] / "config"
    config_dir = Path(config_dir)
    config_path = config_dir / config_file

    if not config_path.exists():
        raise FileNotFoundError(f"City config not found: {config_path}")

    try:
        yaml_data = yaml.safe_load(config_path.read_text())
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing config YAML: {e}") from e

    try:
        log.info(f"Loaded config for: {config_file}")
        return CityConfig(**yaml_data)
    except ValidationError as e:
        raise ValueError(f"Invalid config:\n{e}") from e
