# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Purpose
Design and build a remote sensing model to identify vacant lots in northeast urban areas as an open source tool for city management (for zoning, housing development, etc).

**DO NOT UPDATE `EDA/cluster_vacant_lots.ipynb`**

# currentDate
Today's date is 2026-02-17.

# Setup

Requires Google ADC credentials and Earth Engine access:

```bash
# Authenticate with Google
gcloud auth application-default login

# Authenticate with Earth Engine
uv run earthengine authenticate --force

# Install dependencies
uv sync
```

GCP project `vacant-lot-detection` must have Earth Engine API enabled and the service account configured as `Service Usage Consumer + Earth Engine Resource Admin`.

# Running Notebooks

```bash
uv run jupyter notebook
```

Active development happens in `EDA/clean.ipynb`. `EDA/cluster_vacant_lots.ipynb` is read-only (frozen reference).

# Architecture

## Pipeline Flow

```
City YAML config → CityConfig (Pydantic) → MapPluto GDB → Stratified Sample
    → Rasterize Parcels → GCS Upload → GEE Ingest
    → NAIP Spectral Feature Extraction (GEE) → Clustering (K-Means) → Outputs
```

## Module Responsibilities

| Module | Role |
|--------|------|
| `config.py` | Pydantic `CityConfig` validates YAML configs; generates output paths and GEE asset names |
| `mappluto.py` | Load MapPluto GDB, stratified-sample parcels (oversamples vacant land code `"11"`), heuristic vacant identification |
| `raster_utils.py` | Rasterize sampled parcel polygons to GeoTIFF with pixel→BBL ID mapping |
| `gee_utils.py` | GEE init, ImageCollection loading, GCS upload, asset ingest, spectral stats export |
| `naip.py` | Compute NDVI, SAVI, Brightness, BareSoilProxy from NAIP R/G/B/NIR bands |
| `analysis.py` | K-Means clustering pipeline with StandardScaler + optional PCA; `find_optimal_clusters()` via inertia/silhouette |
| `geometry.py` | Load city boundary from TIGER counties or local GeoJSON |
| `data_utils.py` | Load ESRI GDB with layer caching; summary statistics helpers |
| `plotting.py` | Distribution plots for EDA |
| `logger.py` | Shared logger safe for Jupyter use |

## Configuration System

City configs live in `config/` as YAML files. `template.yaml` documents every option. Key required fields: `city`, `geometry`, `parcel` (data_path, layer, id_column, landuse_column, vacant_codes, source_crs), `raster.output_crs`.

Output paths are auto-generated as `outputs/eda/{city}_{run_name}/` with subdirs: `rasters/`, `intermediaries/`, `data/`, `figures/`.

## Data

- `data/nyc_mappluto_22v3_arc_fgdb/MapPLUTO22v3.gdb` — primary parcel dataset (~800k NYC tax lots)
- `data/philadelphia/`, `data/boston/` — additional cities for future multi-city support
- Parcel source CRS: EPSG:2263 (NY State Plane); output CRS: EPSG:32618 (UTM Zone 18N)
- NAIP imagery: `USDA/NAIP/DOQQ` collection via GEE (1m resolution, R/G/B/NIR bands)

## Sampling Strategy

`mappluto.load_and_sample()` filters by area (default 2,000–16,000 sq ft), then stratified-samples with oversampling of `LandUse == "11"` (vacant land) to reach at least `vacant_min_fraction` (default 8%) of the sample. Default total sample: 25,000 parcels.
