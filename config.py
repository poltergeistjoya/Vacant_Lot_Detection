from pydantic import BaseModel, ValidationError, model_validator
from typing import Optional
import yaml
from pathlib import Path
from logger import get_logger

log = get_logger()


# ============================================================================
# Config Models with Defaults
# ============================================================================

class GCPConfig(BaseModel):
    """Google Cloud Platform configuration."""
    project_id: str = "vacant-lot-detection"
    bucket: str = "thesis_parcels"


class OutputsConfig(BaseModel):
    """Output directory configuration."""
    base_dir: str = "outputs"
    eda_subdir: str = "eda"  # Will be {base_dir}/{eda_subdir}/{city}
    figures_subdir: str = "figures"
    data_subdir: str = "data"

    def get_eda_dir(self, city: str) -> Path:
        """Get EDA output directory for a city: outputs/eda/{city}"""
        return Path(self.base_dir) / self.eda_subdir / city

    def get_figures_dir(self, city: str) -> Path:
        """Get figures directory: outputs/eda/{city}/figures"""
        return self.get_eda_dir(city) / self.figures_subdir

    def get_data_dir(self, city: str) -> Path:
        """Get data output directory: outputs/eda/{city}/data"""
        return self.get_eda_dir(city) / self.data_subdir


class ParcelConfig(BaseModel):
    """Parcel data source configuration."""
    data_path: str
    layer: str
    id_column: str = "BBL"
    landuse_column: str = "LandUse"
    vacant_codes: list[str] = ["11"]
    source_crs: str = "EPSG:2263"


class RasterConfig(BaseModel):
    """Raster output configuration."""
    output_crs: str = "EPSG:32618"  # UTM 18N for US Northeast
    resolution: float = 1.0  # 1 meter to match NAIP
    output_path: Optional[str] = None  # Auto-generated if not set


class GEEConfig(BaseModel):
    """Google Earth Engine asset configuration."""
    parcel_asset: Optional[str] = None  # Auto-generated from city if not set
    export_bucket: Optional[str] = None  # Falls back to gcp.bucket
    export_prefix: Optional[str] = None  # Auto-generated from city if not set


class NAIPConfig(BaseModel):
    """NAIP imagery configuration."""
    collection_id: str = "USDA/NAIP/DOQQ"
    year: int = 2022
    start_date: Optional[str] = None  # Auto-generated from year if not set
    end_date: Optional[str] = None  # Auto-generated from year if not set
    bands: list[str] = ["R", "G", "B", "N"]
    scale_factor: float = 255.0

    @model_validator(mode="after")
    def set_date_range(self):
        """Auto-generate date range from year if not specified."""
        if self.start_date is None:
            self.start_date = f"{self.year}-01-01"
        if self.end_date is None:
            self.end_date = f"{self.year}-12-31"
        return self


class SamplingConfig(BaseModel):
    """Parcel sampling configuration."""
    total_samples: int = 25000
    min_area: float = 2000  # sq ft
    max_area: float = 16000  # sq ft
    vacant_min_fraction: float = 0.08
    random_state: int = 42


class ClusteringConfig(BaseModel):
    """Clustering analysis configuration."""
    n_clusters: int = 5
    random_state: int = 42
    features: list[str] = [
        "B_mean", "G_mean", "R_mean", "N_mean",
        "NDVI_mean", "SAVI_mean", "Brightness_mean", "BareSoilProxy_mean"
    ]


class CityConfig(BaseModel):
    """Complete configuration for a city's EDA pipeline."""
    city: str
    city_name: Optional[str] = None

    # GCP settings (with defaults)
    gcp: GCPConfig = GCPConfig()

    # Output directories
    outputs: OutputsConfig = OutputsConfig()

    # Data paths
    data_dir: str = "data"

    # Parcel configuration (required - no sensible defaults for paths)
    parcel: ParcelConfig

    # Raster output settings
    raster: RasterConfig = RasterConfig()

    # GEE asset settings
    gee: GEEConfig = GEEConfig()

    # NAIP imagery settings
    naip: NAIPConfig = NAIPConfig()

    # Sampling settings
    sampling: SamplingConfig = SamplingConfig()

    # Spectral indices to compute
    spectral_indices: list[str] = ["NDVI", "SAVI", "Brightness", "BareSoilProxy"]

    # Clustering settings
    clustering: ClusteringConfig = ClusteringConfig()

    @model_validator(mode="after")
    def set_derived_defaults(self):
        """Set defaults that depend on city name."""
        # Auto-generate city_name if not set
        if self.city_name is None:
            self.city_name = self.city.upper()

        # Auto-generate raster output path
        if self.raster.output_path is None:
            self.raster.output_path = f"EDA/{self.city}_parcels_utm18n.tif"

        # Auto-generate GEE asset path
        if self.gee.parcel_asset is None:
            self.gee.parcel_asset = f"projects/{self.gcp.project_id}/assets/{self.city}_parcels_raster"

        # GEE export bucket falls back to GCP bucket
        if self.gee.export_bucket is None:
            self.gee.export_bucket = self.gcp.bucket

        # Auto-generate GEE export prefix
        if self.gee.export_prefix is None:
            self.gee.export_prefix = self.city

        return self

    def get_parcel_path(self, repo_root: Path | str = ".") -> Path:
        """Get full path to parcel data file."""
        return Path(repo_root) / self.parcel.data_path

    def get_output_dir(self, repo_root: Path | str = ".") -> Path:
        """Get EDA output directory: outputs/eda/{city}"""
        return Path(repo_root) / self.outputs.get_eda_dir(self.city)

    def get_figures_dir(self, repo_root: Path | str = ".") -> Path:
        """Get figures output directory: outputs/eda/{city}/figures"""
        return Path(repo_root) / self.outputs.get_figures_dir(self.city)

    def ensure_output_dirs(self, repo_root: Path | str = ".") -> None:
        """Create output directories if they don't exist."""
        repo_root = Path(repo_root)
        dirs = [
            self.outputs.get_eda_dir(self.city),
            self.outputs.get_figures_dir(self.city),
            self.outputs.get_data_dir(self.city),
        ]
        for d in dirs:
            (repo_root / d).mkdir(parents=True, exist_ok=True)


# ============================================================================
# Config Loading Functions
# ============================================================================

def load_city_config(city: str, config_dir: str | Path = "config") -> CityConfig:
    """
    Load city-specific config from config/{city}.yaml.

    Args:
        city: City identifier (e.g., 'nyc', 'philadelphia').
        config_dir: Directory containing city config files.

    Returns:
        CityConfig object with validated settings.
    """
    config_dir = Path(config_dir)
    config_path = config_dir / f"{city}.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"City config not found: {config_path}")

    try:
        yaml_data = yaml.safe_load(config_path.read_text())
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing city config YAML: {e}") from e

    try:
        log.info(f"Loaded city config for: {city}")
        return CityConfig(**yaml_data)
    except ValidationError as e:
        raise ValueError(f"Invalid city config:\n{e}") from e


def get_available_cities(config_dir: str | Path = "config") -> list[str]:
    """
    List available city configurations.

    Args:
        config_dir: Directory containing city config files.

    Returns:
        List of city identifiers.
    """
    config_dir = Path(config_dir)
    if not config_dir.exists():
        return []

    return [
        p.stem for p in config_dir.glob("*.yaml")
        if p.stem not in ("template",)  # Exclude template configs
    ]
