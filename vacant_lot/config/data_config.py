"""Data configuration models for the training pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, model_validator


class ImageryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    vrt: str  # Relative to shared root
    exclude_dates: list[str] = []


class ParcelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    gdb_path: str
    layer: str
    id_column: str
    landuse_column: str
    vacant_codes: list[str]
    source_crs: str


class RasterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    output_crs: str
    resolution: float = 1.0


class WaterMaskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    source: str = "tiger"  # 'tiger' only (pygris.area_water) for now
    cache: str = "data/geographic/nyc_water.geojson"


class RoadsMaskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    source: str = "tiger"  # 'tiger' only (pygris.roads) for now
    cache: str = "data/geographic/nyc_roads.geojson"
    # TIGER MTFCC codes to burn: S1100 primary, S1200 secondary,
    # S1630 ramp, S1640 service drive.
    road_mtfcc: list[str] = ["S1100", "S1200", "S1630", "S1640"]


class RoadbedConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    path: str  # Path to planimetric roadbed polygons (GeoJSON or shapefile)


class LabelsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    vacancy_mask: str
    borough_mask: str
    patch_splits: str
    building_pred: str = "outputs/labels/building_pred.tif"
    erosion_pixels: int = 2
    omit_bbls: list[int] = []
    force_nonvacant_bbls: list[int] = []
    force_vacant_bbls: list[int] = []
    water_mask: Optional[WaterMaskConfig] = None
    roads_mask: Optional[RoadsMaskConfig] = None
    roadbed: Optional[RoadbedConfig] = None


class SplitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    strategy: str = "borough"
    train_boroughs: list[int] = [4]
    val_boroughs: list[int] = [2]
    test_boroughs: list[int] = [3]
    exclude_boroughs: list[int] = [1, 5]


class PatchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    size: int = 256
    stride: int = 256
    in_channels: int = 10
    min_valid_pixels: int = 50


class GeometryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: str = "tiger_counties"
    state_fips: Optional[str] = None
    counties: Optional[list[str]] = None
    path: Optional[str] = None

    @model_validator(mode="after")
    def validate_source_fields(self):
        if self.source == "tiger_counties":
            if not self.counties:
                raise ValueError("counties required for tiger_counties source")
            if not self.state_fips:
                raise ValueError("state_fips required for tiger_counties source")
        elif self.source == "geojson":
            if not self.path:
                raise ValueError("path required for geojson source")
        return self


class SegmentationBboxConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    bbox: Optional[tuple[float, float, float, float]] = None
    year: Optional[int] = None  # override STAC query year; falls back to VRT filename if None


class DataConfig(BaseModel):
    """Shared data configuration used by all training scripts."""
    model_config = ConfigDict(extra="forbid")
    imagery: ImageryConfig
    parcels: ParcelConfig
    raster: RasterConfig
    labels: LabelsConfig
    split: SplitConfig = SplitConfig()
    patch: PatchConfig = PatchConfig()
    geometry: GeometryConfig
    segmentation: SegmentationBboxConfig = SegmentationBboxConfig()

    # Set by loader after construction — not from YAML
    _shared_root: Path = None

    def get_vrt_path(self) -> Path:
        return self._shared_root / self.imagery.vrt

    def get_parcel_path(self) -> Path:
        return self._shared_root / self.parcels.gdb_path

    def get_vacancy_mask_path(self) -> Path:
        return self._shared_root / self.labels.vacancy_mask

    def get_borough_mask_path(self) -> Path:
        return self._shared_root / self.labels.borough_mask

    def get_patch_splits_path(self) -> Path:
        return self._shared_root / self.labels.patch_splits

    def get_building_pred_path(self) -> Path:
        return self._shared_root / self.labels.building_pred

    def get_labels_dir(self) -> Path:
        return self.get_vacancy_mask_path().parent

    def ensure_labels_dir(self) -> None:
        self.get_labels_dir().mkdir(parents=True, exist_ok=True)
