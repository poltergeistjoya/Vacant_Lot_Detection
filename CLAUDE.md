# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Purpose
Design and build a remote sensing model to identify vacant lots in northeast urban areas as an open source tool for city management (for zoning, housing development, etc).

**DO NOT UPDATE `EDA/cluster_vacant_lots.ipynb`**

# currentDate
Today's date is 2026-03-06.

# Repo Layout (Bare Repo + Worktrees)

```
Vacant_Lot_Detection/      ← SHARED_ROOT (auto-detected by _get_shared_root())
├── .bare/                 ← bare git repo
├── main/                  ← worktree
├── worktree-xxx/          ← feature worktrees
├── data/                  ← shared, NOT inside any worktree
│   ├── nyc_mappluto_22v3_arc_fgdb/   ← parcel GDB
│   └── naip/                         ← NAIP tiles (city/year keyed)
│       └── nyc/2022/
│           ├── *.tif
│           ├── naip_nyc_2022.vrt
│           └── naip_nyc_2022.json
└── outputs/               ← shared, NOT inside any worktree
    ├── eda/{run_key}/
    ├── masks/{run_key}/              ← per-run label masks
    ├── modeling/{run_key}/           ← per-run models, figures, data
    └── final/{run_key}/
```

`data/` and `outputs/` live at the **shared root**, one level above each worktree. All path-generating code resolves paths relative to the shared root via `_get_shared_root()`.

# Shared Root Auto-Detection

`vacant_lot/config.py` exports `_get_shared_root() -> Path`:
- Default: `Path(__file__).resolve().parents[2]` (package → worktree → shared root)
- Override: set `VACANT_LOT_SHARED_ROOT=/path/to/root` env var

# Setup

Requires Google ADC credentials and Earth Engine access for EDA scripts:

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

Active notebooks:
- `EDA/clean.ipynb` — EDA + clustering pipeline
- `modeling/01_visualize_sample.ipynb` — visualize 25k sample as GeoTIFF
- `modeling/02_parcel_classifier.ipynb` — supervised classifier on spectral features
- `data_prep/01_prepare_segmentation_data.ipynb` — NAIP tile download + VRT


# Architecture


## Module Responsibilities

| Module | Role |
|--------|------|
| `config.py` | Pydantic `CityConfig` validates YAML configs; `_get_shared_root()` for path resolution; generates output paths and GEE asset names |
| `mappluto.py` | Load MapPluto GDB, stratified-sample parcels (oversamples vacant land code `"11"`), heuristic vacant identification |
| `raster_utils.py` | Rasterize sampled parcel polygons to GeoTIFF with pixel→BBL ID mapping |
| `gee_utils.py` | GEE init, ImageCollection loading, GCS upload, asset ingest, spectral stats export |
| `naip.py` | Compute NDVI, SAVI, Brightness, BareSoilProxy from NAIP R/G/B/NIR bands |
| `analysis.py` | K-Means clustering pipeline with StandardScaler + optional PCA; `find_optimal_clusters()` via inertia/silhouette |
| `geometry.py` | Load city boundary from TIGER counties or local GeoJSON |
| `data_utils.py` | Load ESRI GDB with layer caching; summary statistics helpers |
| `plotting.py` | Distribution plots for EDA |
| `logger.py` | Shared logger safe for Jupyter use |
| `tile_export.py` | NAIP tile download from AWS Open Data via STAC; GDAL VRT construction |
| `modeling.py` | Build labels, feature matrix, evaluate classifiers, save models |

## Configuration System

City configs live in `<worktree>/config/` as YAML files. `template.yaml` documents every option. Key required fields: `city`, `geometry`, `parcel` (data_path, layer, id_column, landuse_column, vacant_codes, source_crs), `raster.output_crs`.

`load_config(config_file)` — no `config_dir` needed; auto-resolves to `<worktree>/config/`.

`nyc_buildings.yaml` is the **source-of-truth config** for the segmentation pipeline. `nyc_vacant.yaml` is synced for consistency but not used by active notebooks.

All `CityConfig` path methods take no arguments (no `repo_root`):
- `get_parcel_path()` → `<shared_root>/data/<parcel.data_path>`
- `get_output_dir()` → `<shared_root>/outputs/eda/<run_key>/`
- `get_figures_dir()` / `get_data_dir()` / `get_intermediaries_dir()` → subdirs of `get_output_dir()`
- `get_naip_dir()` / `get_naip_tiles_dir()` → `<shared_root>/data/naip/{city}/{year}/`
- `get_naip_vrt_path()` → `<shared_root>/data/naip/{city}/{year}/naip_{city}_{year}.vrt`
- `get_seg_masks_dir()` → `<shared_root>/outputs/masks/<run_key>/`
- `get_modeling_dir()` → `<shared_root>/outputs/modeling/<run_key>/`
- `get_modeling_models_dir()` / `get_modeling_figures_dir()` / `get_modeling_data_dir()` → subdirs of `get_modeling_dir()`
- `ensure_output_dirs()` / `ensure_seg_output_dirs()` / `ensure_modeling_dirs()` → create the above dirs

## Data

- `<shared_root>/data/nyc_mappluto_22v3_arc_fgdb/MapPLUTO22v3.gdb` — primary parcel dataset (~800k NYC tax lots)
- `<shared_root>/data/philadelphia/`, `<shared_root>/data/boston/` — additional cities
- `<shared_root>/data/naip/nyc/2022/` — NAIP tiles + VRT (shared input, city/year keyed)
- `<shared_root>/outputs/` — all pipeline outputs (EDA, masks, modeling)
- Parcel source CRS: EPSG:2263 (NY State Plane); output CRS: EPSG:26918 (NAD83 UTM Zone 18N, matches NAIP native CRS)
- NAIP imagery: `USDA/NAIP/DOQQ` via GEE (1m resolution, R/G/B/NIR) or direct AWS S3 download

## Sampling Strategy

`mappluto.load_and_sample()` filters by area (min from `min_pixels × resolution²`), drops null landuse labels, then stratified-samples with oversampling of the target vacant codes to reach at least `vacant_min_fraction` (default 8%) of the sample. Default total sample: 25,000 parcels.
