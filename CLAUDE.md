# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Purpose
Design and build a remote sensing model to identify vacant lots in northeast urban areas as an open source tool for city management (zoning, housing development, etc).

# GitHub Issues Warning
GitHub issues may be outdated. Always verify splits, paths, and hyperparameters against `config/data.yaml` (data paths) and `config/rf.yaml` / `config/lgbm.yaml` (model params). Issues take precedence for *what* to build; config files take precedence for *parameters*.

# Running Scripts

```bash
just data-prep::run               # full pipeline: download → masks → patches
just data-prep::download          # download NAIP tiles + build VRT
just data-prep::prepare-labels    # burn MapPLUTO → vacancy + borough masks
just data-prep::extract-patches   # generate patch_splits.json
just train::rf                    # train RF baseline (config/rf.yaml)
just train::rf --run-id 003       # explicit run ID
just train::lgbm                  # train LightGBM baseline (config/lgbm.yaml)
just train::plot --run outputs/models/rf/003   # generate figures for a run
just upload-kaggle                # upload dataset to Kaggle
```

# Running Notebooks

```bash
uv sync
uv run jupyter notebook
```

Active notebook: `EDA/clean.ipynb` — EDA + clustering pipeline.

GEE/GCP setup required for EDA only: `gcloud auth application-default login` then `uv run earthengine authenticate --force`.

# Repo Layout (Bare Repo + Worktrees)

```
Vacant_Lot_Detection/              ← SHARED_ROOT (auto-detected by _get_shared_root())
├── .bare/                         ← bare git repo
├── main/, dataset-explore/, pytorch-training-loop/  ← worktrees
├── data/                          ← shared inputs, NOT inside any worktree
│   ├── imagery/naip/nyc/2022/    ← 85 tiles downloaded, 47 NY in VRT, 38 NJ excluded
│   └── parcels/nyc/mappluto_22v3/
└── outputs/                       ← shared outputs, NOT inside any worktree
    ├── labels/                    ← vacancy_mask.tif, borough_mask.tif, patch_splits.json
    ├── models/rf/, models/gbm/    ← pixel-level models, numbered 001/002/...
    ├── eda/parcel_classifier/     ← EDA-only parcel models (not production)
    └── figures/
```

`_get_shared_root()` in `vacant_lot/config/loader.py` walks up `parents[3]` from `loader.py` to reach the shared root. Override with `VACANT_LOT_SHARED_ROOT` env var.

# Config System

Two interfaces, both re-exported from `vacant_lot.config`:

**Training scripts** — `DataConfig` + `TrainConfig`:
```python
from vacant_lot.config import load_data_config, load_train_config
data_cfg = load_data_config()                    # reads config/data.yaml
data_cfg, train_cfg = load_train_config("rf.yaml")
```

**EDA notebooks** — legacy `CityConfig` (unchanged interface):
```python
from vacant_lot.config import load_config
cfg = load_config("nyc_buildings.yaml")
```

Model YAMLs (`rf.yaml`, `lgbm.yaml`) contain a `data: data.yaml` key that `load_train_config` resolves automatically.

# Borough Splits (source: `config/data.yaml`)

| Borough | BoroCode | Split | Patches |
|---------|----------|-------|---------|
| Queens | 4 | Train | 12,576 |
| Bronx | 2 | Val | 4,943 |
| Brooklyn | 3 | Test | 8,137 |
| Manhattan | 1 | Excluded | tall building occlusion |
| Staten Island | 5 | Excluded | overwhelmingly tree cover |

True pixel distribution in Queens: 19.7M vacant / 566M non-vacant = **3.4% vacant**.

# Pixel-Level Sampling & Class Weights

`sample_pixels_from_patches` uses reservoir sampling (5M vacant, 15M non-vacant from train patches). Class weights correct for undersampling: `weight = true_count / sampled_count`. **Do NOT use `class_weight="balanced"`** — that's only correct when using all data unsampled.

For LightGBM, `scale_pos_weight = nonvacant_weight / vacant_weight` is used instead.

# Module Responsibilities

| Module | Role |
|--------|------|
| `vacant_lot/config/` | Config subpackage: `DataConfig` (scripts), `CityConfig` (EDA), `TrainConfig`, `_get_shared_root()` |
| `dataset.py` | Patch grid generation, borough splits, `NAIPSegmentationDataset`, `compute_spectral_indices` |
| `label_utils.py` | `create_vacancy_mask` (0/1/255), `create_borough_mask` (1-5), boundary erosion |
| `modeling.py` | `sample_pixels_from_patches`, `evaluate_segmentation_streaming`, `save_model`, `build_labels` |
| `tile_export.py` | NAIP download from Planetary Computer STAC, GDAL VRT construction |
| `mappluto.py` | Load MapPLUTO GDB, identify vacant parcels |
| `geometry.py` | Load city boundary from TIGER counties or GeoJSON |
| `data_utils.py` | Load ESRI GDB with layer caching |
| `raster_utils.py` | Rasterize parcel polygons → GeoTIFF |
| `gee_utils.py` | GEE init, GCS upload (EDA only) |
| `naip.py` | Spectral indices via GEE (EDA only) |
| `analysis.py` | K-Means clustering pipeline (EDA only) |
| `plotting.py` | Distribution plots for EDA |
| `logger.py` | Shared logger safe for Jupyter |

# Data Details

- **NAIP**: 4 bands (R, G, B, NIR) at **0.6m resolution**, EPSG:26918. 85 tiles downloaded, 38 NJ border tiles excluded from VRT via `imagery.exclude_dates` in `data.yaml` (those tiles are still on disk — useful for figures, masked out by borough mask anyway). Note: `config/data.yaml` has `raster.resolution: 1.0` — verify/update if regenerating masks or patches.
- **Features**: 4 NAIP bands (R, G, B, NIR), normalized to [0, 1]. 10-channel mode (4 bands + 6 spectral indices: NDVI, SAVI, Brightness, BareSoilProxy, EVI, GNDVI) was tried but showed no convergence speedup over 4-band input — use `in_channels: 4`.
- **Parcels**: ~800k NYC tax lots, source CRS EPSG:2263, rasterized to EPSG:26918. Vacant codes: V0–V9, G7.
- **Masks**: `vacancy_mask.tif` values: 0=non-vacant, 1=vacant, 255=ignore. Boundary pixels within 2px of parcel edges are eroded to 255 to reduce label noise.
- **Patch grid**: 256×256 px, stride 256 (no overlap), kept if ≥50 labeled pixels. At 0.6m/px this is ~154m/patch. Comparison paper uses 256px at 1.6m (~410m/patch) — to match spatial coverage, use 512×512 (~307m, 75% of paper) or 1024×1024 (~614m, 150% of paper); 512 is closer.
- **Model outputs**: `outputs/models/{rf,gbm}/NNN/` (flat). Files per run:
  - `model.joblib` — fitted model
  - `metrics.json` — scalar metrics + config snapshot (human-readable)
  - `pr_curves.npz` — precision/recall/threshold arrays for val and test (used by `plot_results.py`)
  - `figures/` — written by `just train::plot`
- **Run IDs**: auto-increment; pass `--run-id` to override.
- **Run notes**: set `note:` in the YAML or override with `VACANT_LOT_RUN_NOTE` env var — written to `metrics.json`.

# Compute

School compute server with two GPUs (not always both free):

| GPU | VRAM | TDP |
|-----|------|-----|
| NVIDIA TITAN RTX | 24 GB | 280W |
| NVIDIA TITAN V | 12 GB | 250W |

Train on the Titan RTX (24GB) when possible. At 512×512 batch sizes of 8–16 are comfortable; at 1024×1024 expect batch 2–4. The Titan V (12GB) can handle 512×512 at batch ~4–8.

# Housing Context Figures

`scripts/plot_building_permits.py` — building permits per capita from the `data/housing/housing_data_comparisons.json` dataset (NYC, Philadelphia, Dallas–Fort Worth, Phoenix MSAs). Produces two figures saved to `outputs/figures/`:

| Figure | Description |
|--------|-------------|
| `building_permits_per_capita.png` | Line chart of total permitted units per capita by MSA from 2000–present |
| `building_permits_mix.png` | 2×2 stacked-area panels showing single- vs. multi-family permit mix per MSA |

```bash
uv run python scripts/plot_building_permits.py                          # both figures, default metric
uv run python scripts/plot_building_permits.py --metric total_bldgs_per_capita
uv run python scripts/plot_building_permits.py --out path/to/fig.png --out-mix path/to/mix.png
```

Style: STIX Two Text font, minimal spines, no bold — consistent across all figures in this repo.

# Post-Training Plots

Generated by `just train::plot <run_dir>` (reads `model.joblib`, `metrics.json`, `pr_curves.npz`):

| Figure | Description |
|--------|-------------|
| `pr_curve.png` | Precision-Recall curve for val and test |
| `threshold_sweep.png` | Precision / Recall / F1 / F2 vs threshold on val; marks optimal threshold for each |
| `feature_importance.png` | Normalized feature importances (RF and LightGBM both expose `feature_importances_`) |
| `confusion_matrix.png` | Confusion matrices for val and test with TN/FP/FN/TP counts and percentages |

Note: F2 score (beta=2) prioritizes recall over precision. Reported for comparability with vacant land detection literature, not as a primary metric.
