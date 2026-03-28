"""Data configuration models for the training pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, model_validator


class ImageryConfig(BaseModel):
    vrt: str  # Relative to shared root
    exclude_dates: list[str] = []


class ParcelConfig(BaseModel):
    gdb_path: str
    layer: str
    id_column: str
    landuse_column: str
    vacant_codes: list[str]
    source_crs: str


class RasterConfig(BaseModel):
    output_crs: str
    resolution: float = 1.0


class LabelsConfig(BaseModel):
    vacancy_mask: str
    borough_mask: str
    patch_splits: str
    erosion_pixels: int = 2
    omit_bbls: list[int] = []


class SplitConfig(BaseModel):
    strategy: str = "borough"
    train_boroughs: list[int] = [4]
    val_boroughs: list[int] = [2]
    test_boroughs: list[int] = [3]
    exclude_boroughs: list[int] = [1, 5]


class PatchConfig(BaseModel):
    size: int = 256
    stride: int = 256
    in_channels: int = 10
    min_valid_pixels: int = 50


class GeometryConfig(BaseModel):
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
    bbox: Optional[tuple[float, float, float, float]] = None


class DataConfig(BaseModel):
    """Shared data configuration used by all training scripts."""
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

    def get_labels_dir(self) -> Path:
        return self.get_vacancy_mask_path().parent

    def ensure_labels_dir(self) -> None:
        self.get_labels_dir().mkdir(parents=True, exist_ok=True)
