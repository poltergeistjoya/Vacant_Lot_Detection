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

## Scripts

### Export Patch Grid (QGIS)

Converts `patch_splits.json` to a GeoPackage for spatial visualization. Patches are colored by split.

```bash
uv run python scripts/export_patch_grid.py
```

Output: `outputs/masks/<run_key>/patch_grid.gpkg`

In QGIS, use categorized symbology on the `split` field.

### Upload Kaggle Dataset

Stages files and creates or updates a Kaggle dataset via the Kaggle API.

```bash
# Create a new dataset
uv run python scripts/upload_kaggle_dataset.py \
    --files outputs/masks/nyc_buildings/vacancy_mask.tif \
             outputs/masks/nyc_buildings/patch_splits.json \
    --dataset-id username/dataset-slug \
    --title "Dataset Title" \
    --new

# Push a new version to an existing dataset
uv run python scripts/upload_kaggle_dataset.py \
    --files outputs/masks/nyc_buildings/vacancy_mask.tif \
             outputs/masks/nyc_buildings/patch_splits.json \
    --dataset-id username/dataset-slug \
    --message "Description of what changed"
```

**Flags:**

| Flag | Required | Description |
|------|----------|-------------|
| `--files` | Yes | One or more file paths to include |
| `--dataset-id` | Yes | `username/slug` format |
| `--title` | With `--new` | Human-readable dataset title |
| `--message` | No | Version notes (default: `"New version"`) |
| `--new` | No | Create new dataset instead of pushing a version |
| `--staging-dir` | No | Custom staging directory (default: auto temp dir) |

## Notebook Output Stripping

This repo uses [nbstripout](https://github.com/kynan/nbstripout) — cell outputs are stripped automatically on commit.

To **preserve a specific cell's output** (e.g. a figure), add the `keep_output` tag:
- **VS Code**: cell `...` menu → Add Cell Tag → `keep_output`
- **Jupyter**: View → Cell Toolbar → Tags → type `keep_output` → Add tag

The git filter config is not tracked. After cloning or creating a new worktree, run `uv sync && uv run nbstripout --install --attributes .gitattributes` to re-register it.
