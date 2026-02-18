from pydantic import BaseModel, ValidationError, model_validator
from typing import Optional
import yaml
from pathlib import Path
from datetime import datetime
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
    rasters_subdir: str = "rasters"
    intermediaries_subdir: str = "intermediaries"

    def get_eda_dir(self, city: str) -> Path:
        """Get EDA output directory for a city: outputs/eda/{city}"""
        return Path(self.base_dir) / self.eda_subdir / city

    def get_figures_dir(self, city: str) -> Path:
        """Get figures directory: outputs/eda/{city}/figures"""
        return self.get_eda_dir(city) / self.figures_subdir

    def get_data_dir(self, city: str) -> Path:
        """Get data output directory: outputs/eda/{city}/data"""
        return self.get_eda_dir(city) / self.data_subdir

    def get_rasters_dir(self, city: str) -> Path:
        """Get rasters directory: outputs/eda/{city}/rasters"""
        return self.get_eda_dir(city) / self.rasters_subdir

    def get_intermediaries_dir(self, city: str) -> Path:
        """Get intermediaries directory: outputs/eda/{city}/intermediaries"""
        return self.get_eda_dir(city) / self.intermediaries_subdir


class GeometryConfig(BaseModel):
    """City geometry configuration."""
    source: str = "tiger_counties"  # "tiger_counties" or "geojson"

    # For tiger_counties source
    counties: list[str] | None = None
    state_fips: str | None = None  # Required for tiger_counties

    # For geojson source
    path: str | None = None

    @model_validator(mode="after")
    def validate_source_fields(self):
        """Validate required fields based on source type."""
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
    """Parcel data source configuration."""
    data_path: str
    layer: str
    id_column: str
    landuse_column: str
    vacant_codes: list[str]
    source_crs: str


class RasterConfig(BaseModel):
    """Raster output configuration."""
    output_crs: str
    output_path: Optional[str] = None  # Auto-generated if not set
    resolution: float = 1.0  # 1 meter to match NAIP


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

    # Run identification (optional, defaults to city name)
    run_name: Optional[str] = None
    run_description: Optional[str] = None  # For README generation

    # GCP settings (with defaults)
    gcp: GCPConfig = GCPConfig()

    # Output directories
    outputs: OutputsConfig = OutputsConfig()

    # Data paths
    data_dir: str = "data"

    # City geometry (required - no sensible defaults)
    geometry: GeometryConfig

    # Parcel configuration (required - no sensible defaults for paths)
    parcel: ParcelConfig

    # Raster output settings
    raster: RasterConfig

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

        # Default run_name to city if not set
        if self.run_name is None:
            self.run_name = self.city

        # Auto-generate GEE asset path
        if self.gee.parcel_asset is None:
            self.gee.parcel_asset = f"projects/{self.gcp.project_id}/assets/{self.city}_parcels_raster"

        # GEE export bucket falls back to GCP bucket
        if self.gee.export_bucket is None:
            self.gee.export_bucket = self.gcp.bucket

        # Auto-generate GEE export prefix
        if self.gee.export_prefix is None:
            self.gee.export_prefix = self.city

        # Auto-generate raster output path: outputs/eda/{city}_{run_name}/rasters/parcels_{crs}.tif
        if self.raster.output_path is None:
            # Determine the output directory path
            if self.run_name and self.run_name != self.city:
                city_with_run = f"{self.city}_{self.run_name}"
            else:
                city_with_run = self.city

            crs_suffix = self.raster.output_crs.replace(":", "").lower()
            self.raster.output_path = str(
                Path(self.outputs.base_dir) / self.outputs.eda_subdir / city_with_run /
                self.outputs.rasters_subdir / f"parcels_{crs_suffix}.tif"
            )

        return self

    def get_parcel_path(self, repo_root: Path | str = ".") -> Path:
        """Get full path to parcel data file."""
        return Path(repo_root) / self.parcel.data_path

    def get_output_dir(self, repo_root: Path | str = ".") -> Path:
        """Get EDA output directory: outputs/eda/{city}_{run_name} or outputs/eda/{city}"""
        # If run_name is set and different from city, append it
        if self.run_name and self.run_name != self.city:
            city_with_run = f"{self.city}_{self.run_name}"
        else:
            city_with_run = self.city

        return Path(repo_root) / self.outputs.base_dir / self.outputs.eda_subdir / city_with_run

    def get_figures_dir(self, repo_root: Path | str = ".") -> Path:
        """Get figures output directory."""
        return self.get_output_dir(repo_root) / self.outputs.figures_subdir

    def get_data_dir(self, repo_root: Path | str = ".") -> Path:
        """Get data output directory."""
        return self.get_output_dir(repo_root) / self.outputs.data_subdir

    def get_rasters_dir(self, repo_root: Path | str = ".") -> Path:
        """Get rasters output directory."""
        return self.get_output_dir(repo_root) / self.outputs.rasters_subdir

    def get_intermediaries_dir(self, repo_root: Path | str = ".") -> Path:
        """Get intermediaries directory."""
        return self.get_output_dir(repo_root) / self.outputs.intermediaries_subdir

    def ensure_output_dirs(self, repo_root: Path | str = ".") -> None:
        """Create output directories if they don't exist."""
        dirs = [
            self.get_output_dir(repo_root),
            self.get_figures_dir(repo_root),
            self.get_data_dir(repo_root),
            self.get_rasters_dir(repo_root),
            self.get_intermediaries_dir(repo_root),
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Config Loading Functions
# ============================================================================

def generate_run_readme(
    config: CityConfig,
    output_dir: Path,
    stats: dict,
) -> Path:
    """
    Generate README for a run with description and statistics.

    Args:
        config: City configuration
        output_dir: Output directory for the run
        stats: Dictionary of run statistics (samples, parcels, clusters, etc.)

    Returns:
        Path to generated README
    """
    readme_path = output_dir / "README.md"

    description = config.run_description or f"{config.city_name} EDA Analysis"

    content = f"""# {config.city_name} - {config.run_name}

## Description
{description}

## Configuration

**City**: {config.city_name}
**Run**: {config.run_name}
**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

### Data Source
- **Parcel data**: {config.parcel.data_path}
- **Layer**: {config.parcel.layer}
- **CRS**: {config.parcel.source_crs} → {config.raster.output_crs}

### Land Use Codes
{', '.join(config.parcel.vacant_codes)}

### Sampling Parameters
- Total samples: {config.sampling.total_samples}
- Min area: {config.sampling.min_area} sq ft
- Max area: {config.sampling.max_area} sq ft
- Vacant min fraction: {config.sampling.vacant_min_fraction}
- Random state: {config.sampling.random_state}

### Clustering Parameters
- Number of clusters: {config.clustering.n_clusters}
- Features: {', '.join(config.clustering.features)}
- Random state: {config.clustering.random_state}

### NAIP Imagery
- Year: {config.naip.year}
- Bands: {', '.join(config.naip.bands)}

## Run Statistics

{chr(10).join(f'- **{k}**: {v}' for k, v in stats.items())}

## Output Structure

```
{config.run_name}/
├── README.md          # This file
├── rasters/           # Parcel rasters
├── data/              # Processed data CSVs
├── figures/           # Visualizations
└── intermediaries/    # Shapefiles, ID mappings
```

---
*Generated by vacant lot detection pipeline*
"""

    readme_path.write_text(content)
    log.info(f"Generated README: {readme_path}")
    return readme_path


def load_config(config_file: str, config_dir: str | Path = "config") -> CityConfig:
    """
    FIX DOCSTRING
    """
    config_dir = Path(config_dir)
    config_path = config_dir / f"{config_file}"

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
