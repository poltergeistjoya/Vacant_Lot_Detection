# Vacant Lot Detection

Remote sensing pipeline to identify vacant lots in NYC using NAIP aerial imagery. Designed for city management use cases (zoning, housing development). Scope: NYC, generalizable to Northeast US.

## Repo Layout

This repo uses a **bare git + worktrees** layout. Shared data and outputs live *above* each worktree:

```
Vacant_Lot_Detection/          ← shared root (SHARED_ROOT)
├── .bare/                     ← bare git repo
├── main/                      ← main worktree
├── <feature-worktrees>/       ← one per branch
├── data/                      ← shared inputs, NOT inside any worktree
│   ├── nyc_mappluto_22v3_arc_fgdb/   ← MapPLUTO parcel GDB
│   └── naip/nyc/2022/                ← NAIP tiles + VRT
└── outputs/                   ← shared outputs, NOT inside any worktree
    ├── eda/<run_key>/
    ├── masks/<run_key>/        ← vacancy mask, borough mask, patch splits
    └── modeling/<run_key>/
```

All path-generating code resolves to `SHARED_ROOT` via `_get_shared_root()` in `vacant_lot/config.py`. Override with the `VACANT_LOT_SHARED_ROOT` env var if needed.

## Data Flow

```
MapPLUTO GDB                  NAIP tiles (Planetary Computer)
     │                              │
     ▼                              ▼
data_prep/01_prepare_segmentation_data.ipynb
     │  Burns parcels → vacancy_mask.tif
     │  Fetches TIGER → borough_mask.tif
     │  Downloads tiles → naip_nyc_2022.vrt
     ▼
data_prep/02_extract_patches.ipynb
     │  Slides 256×256 window over VRT
     │  Assigns patches to train/val/test by borough
     │  Saves → patch_splits.json
     ▼
scripts/export_patch_grid.py   →  patch_grid.gpkg  (QGIS)
scripts/upload_kaggle_dataset.py → Kaggle dataset  (training)
     │
     ▼
modeling/  (training notebooks — in progress)
```

**Split strategy (spatial, by borough):**

| Split | Borough | BoroCode |
|-------|---------|----------|
| Train | Queens | 4 |
| Val | Bronx | 2 |
| Test | Brooklyn | 3 |
| Excluded | Manhattan, Staten Island | 1, 5 |

## Setup

```bash
uv sync
uv run nbstripout --install --attributes .gitattributes
```

**Google / Earth Engine (required for EDA notebooks):**

```bash
gcloud auth application-default login
uv run earthengine authenticate --force
```

GCP project `vacant-lot-detection` needs Earth Engine API enabled and the service account configured as `Service Usage Consumer + Earth Engine Resource Admin`.

**Kaggle (required for dataset upload script):**

Create `~/.kaggle/kaggle.json`:
```json
{"username": "your-username", "key": "your-api-key"}
```
```bash
chmod 600 ~/.kaggle/kaggle.json
```

## Notebooks

Run notebooks with:
```bash
uv run jupyter notebook
```

### EDA
| Notebook | Description |
|----------|-------------|
| `EDA/01_cluster_lots.ipynb` | K-means clustering on NAIP spectral features (GEE) |
| `EDA/02_parcel_classifier.ipynb` | Supervised parcel-level classifier |
| `EDA/03_manual_inspection_of_lots.ipynb` | Visual inspection of sampled parcels |
| `EDA/04_borough_vacancy_eda.ipynb` | Borough-level vacancy statistics, split rationale |

### Data Preparation
| Notebook | Description |
|----------|-------------|
| `data_prep/01_prepare_segmentation_data.ipynb` | Generate vacancy mask, borough mask, NAIP VRT |
| `data_prep/02_extract_patches.ipynb` | Extract patches, spatial split, verify distribution |

## Running

All recipes are invoked via `just`. Run `just` with no arguments to list everything.

### Data Prep

```bash
# Full pipeline: download → masks → patches
just data-prep::run

# Download NAIP tiles and build VRT
just data-prep::download
just data-prep::download --config data.yaml   # explicit config (default)

# Burn MapPLUTO → vacancy_mask.tif + borough_mask.tif
just data-prep::prepare-labels
just data-prep::prepare-labels --config data.yaml
just data-prep::prepare-labels --erosion-pixels 2   # boundary erosion disk radius (overrides config)

# Slide patch window, assign boroughs, write patch_splits.json
just data-prep::extract-patches
just data-prep::extract-patches --config data.yaml

# Export patch_splits.json → patch_grid.gpkg for QGIS inspection
just data-prep::export-patch-grid
just data-prep::export-patch-grid --config data.yaml
```

Output of `export-patch-grid`: `outputs/labels/patch_grid.gpkg`. Load in QGIS, use categorized symbology on the `split` field.

### Training

```bash
# Random Forest baseline
just train::rf
just train::rf --config rf.yaml       # explicit config (default: config/rf.yaml)
just train::rf --run-id 003           # fixed run ID instead of auto-increment

# LightGBM baseline
just train::lgbm
just train::lgbm --config lgbm.yaml   # default
just train::lgbm --run-id 003

# Deep learning (UNet / DeepLabV3+)
just train::dl
just train::dl --config unet_32.yaml  # default; swap for any config in config/
just train::dl --run-id 003
just train::dl --resume               # resume from latest.pt in the run dir
just train::dl --eval-stride 256      # overlapping inference stride for post-train eval (default: patch_size)

# Post-training plots (PR curve, threshold sweep, feature importance, confusion matrix)
just train::plot --run outputs/models/rf/001
```

### Visualize Predictions

Runs inference on val/test patches and writes georeferenced TIFs to `<run_dir>/figures/`.

```bash
# Basic: prob map + error map for val and test
just train::visualize --run outputs/models/unet/001

# Overlapping inference (averages predictions at stride < patch_size)
just train::visualize --run outputs/models/unet/001 --stride 128

# Custom binarization threshold
just train::visualize --run outputs/models/unet/001 --threshold 0.3

# Regenerate error map from existing pred TIF (no model loaded, no inference)
just train::visualize --run outputs/models/unet/001 --error-only

# Specific splits only
just train::visualize --run outputs/models/unet/001 --splits val
just train::visualize --run outputs/models/unet/001 --splits val test

# Override patch config (useful when run was trained with different splits)
just train::visualize --run outputs/models/unet/001 \
    --patch-size 512 \
    --patch-splits outputs/labels/patch_splits_512_valbk.json
```

**Output TIFs** written to `<run_dir>/figures/`:

| File | Description |
|------|-------------|
| `{split}_pred.tif` | Predicted probabilities (float32) |
| `{split}_pred_s{stride}.tif` | Same, with overlap suffix when `--stride` is set |
| `{split}_error.tif` | 4-band RGBA error map: Green=TP, Red=FP, Blue=FN, Black=TN, White=ignore |

### Upload Kaggle Dataset

```bash
# Create a new dataset
just upload-kaggle \
    --files outputs/labels/vacancy_mask.tif outputs/labels/patch_splits.json \
    --dataset-id username/dataset-slug \
    --title "Dataset Title" \
    --new

# Push a new version to an existing dataset
just upload-kaggle \
    --files outputs/labels/vacancy_mask.tif outputs/labels/patch_splits.json \
    --dataset-id username/dataset-slug \
    --message "Description of what changed"

# Custom staging directory
just upload-kaggle \
    --files outputs/labels/vacancy_mask.tif \
    --dataset-id username/dataset-slug \
    --staging-dir /tmp/my-staging
```

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--files` | Yes | — | One or more file paths to include |
| `--dataset-id` | Yes | — | `username/slug` format |
| `--title` | With `--new` | — | Human-readable dataset title |
| `--message` | No | `"New version"` | Version notes |
| `--new` | No | false | Create new dataset instead of pushing a version |
| `--staging-dir` | No | auto temp dir | Custom staging directory |

## Notebook Output Stripping

This repo uses [nbstripout](https://github.com/kynan/nbstripout) — cell outputs are stripped automatically on commit.

To **preserve a specific cell's output** (e.g. a figure), add the `keep_output` tag:
- **VS Code**: cell `...` menu → Add Cell Tag → `keep_output`
- **Jupyter**: View → Cell Toolbar → Tags → type `keep_output` → Add tag

The git filter config is not tracked. After cloning or creating a new worktree, run `uv sync && uv run nbstripout --install --attributes .gitattributes` to re-register it.
