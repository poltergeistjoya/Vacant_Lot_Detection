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

By default, inference runs at **50% overlap** (`stride = patch_size // 2`) and blends
overlapping predictions with a 2D Hann window — this is the standard for tile merging
and eliminates edge artifacts. Pass `--no-overlap` to disable blending, or `--stride N`
for a custom stride.

```bash
# Basic: prob map + error map for val and test (50% overlap by default)
just train::visualize --run outputs/models/unet/001

# Disable overlap blending (stride = patch_size, no edge smoothing)
just train::visualize --run outputs/models/unet/001 --no-overlap

# Custom stride (e.g. 25% overlap at patch_size=512)
just train::visualize --run outputs/models/unet/001 --stride 384

# Adjust inference batch size (lower for 1024x1024 on small GPUs)
just train::visualize --run outputs/models/unet/001 --batch-size 2

# Custom binarization threshold
just train::visualize --run outputs/models/unet/001 --threshold 0.3

# Regenerate error map from existing pred TIF (no model loaded, no inference)
just train::visualize --run outputs/models/unet/001 --error-only

# Append an extra suffix to output filenames (e.g. val_pred_s128_v2.tif)
just train::visualize --run outputs/models/unet/001 --stride 128 --suffix _v2

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
| `{split}_pred_s{stride}.tif` | Predicted probabilities (float32), Hann-blended across overlapping patches |
| `{split}_pred.tif` | Same but written when `--no-overlap` is passed (no suffix) |
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

## Scripts

All scripts are in `scripts/` and run via `uv run python scripts/<name>.py`. Relative paths passed to `--run`, `--out`, and `--fig-dir` are always resolved from the shared root (`../` relative to the worktree), not from CWD.

| Script | Description |
|--------|-------------|
| `collect_runs.py` | Aggregate `metrics.json` from every model run into a single CSV for comparison. Default output: `scripts/sorted_runs.csv`. |
| `eval_parcel_level.py` | Parcel-level recall: reads the prob TIF for a run+split, computes per-parcel pixel coverage, reports recall at multiple coverage thresholds. Outputs CSV and two figures: recall-vs-coverage curve and pred-fraction histogram. |
| `eval_confusion_by_class.py` | Parcel-level confusion matrix broken down by `BldgClass` and `LandUse`. Reports TP/FP/FN/TN, FP breakdown by land use / building class, FN missed subtypes, and TP detection rate by subtype. Outputs CSV and two figures. |
| `plot_prediction_comparison.py` | Three figure types for visual inspection: single-BBL NAIP+error overlay, UNet vs DeepLabV3+ side-by-side comparison, and a TP/FP/FN/TN patch gallery. |
| `run_prediction_comparisons.py` | Driver for `plot_prediction_comparison.py` — runs all three figure types for `kahan_027` (DeepLabV3+) and `kahan_031` (UNet) on sampled Bronx test BBLs. |
| `run_visualizations.py` | Batch inference + visualization for multiple runs, ordered by val F2. Assigns GPU device. Runs full inference at threshold 0.5 then a cheap error-only pass at the F2-optimal threshold. |
| `plot_thesis_architectures.py` | Architecture diagrams (UNet, DeepLabV3+ variants, atrous convolution) as PDF + PNG for LaTeX/Docs. Outputs to `outputs/figures/`. |
| `plot_building_permits.py` | Building permits per capita figures from `data/housing/housing_data_comparisons.json` — NYC, Philadelphia, Dallas–Fort Worth, Phoenix MSAs from 2000 onward. Outputs to `outputs/figures/`. |
| `upload_kaggle_dataset.py` | Stage files and push a new version (or create) a Kaggle dataset. Prefer `just upload-kaggle` which wraps this. |

### collect_runs.py

```bash
uv run python scripts/collect_runs.py                          # writes scripts/sorted_runs.csv
uv run python scripts/collect_runs.py --output runs.csv        # custom output path
uv run python scripts/collect_runs.py --models-dir /path/to/outputs/models
```

### eval_parcel_level.py

Reads the probability TIF written by `just train::visualize` and evaluates recall at the parcel level. Figures saved to `{run}/figures/` by default.

```bash
uv run python scripts/eval_parcel_level.py \
    --run outputs/models/deeplabv3plus/kahan_027 --split val

# Explicit pixel threshold (default: F2-optimal from pr_curves.npz)
uv run python scripts/eval_parcel_level.py \
    --run outputs/models/deeplabv3plus/kahan_027 --split val \
    --pixel-threshold 0.298

# Save CSV + figures
uv run python scripts/eval_parcel_level.py \
    --run outputs/models/deeplabv3plus/kahan_027 --split val \
    --out outputs/models/deeplabv3plus/kahan_027/parcel_eval_brooklyn.csv

# Custom coverage thresholds; skip figures
uv run python scripts/eval_parcel_level.py \
    --run outputs/models/deeplabv3plus/kahan_027 --split val \
    --coverage 0.2 0.4 0.6 --no-figures
```

| Output figure | Description |
|---------------|-------------|
| `parcel_recall_vs_coverage_{split}.png` | Recall curve sweeping coverage threshold 0→1; marks the default thresholds |
| `parcel_pred_fraction_hist_{split}.png` | Histogram of per-parcel pixel prediction fraction |

### eval_confusion_by_class.py

Evaluates all parcels (vacant + non-vacant) and reports confusion broken down by `LandUse` and `BldgClass`. Borough filter is inferred automatically from the run's `patch_splits`. Figures saved to `{run}/figures/` by default.

```bash
uv run python scripts/eval_confusion_by_class.py \
    --run outputs/models/deeplabv3plus/kahan_027 --split val \
    --out outputs/models/deeplabv3plus/kahan_027/confusion_by_class_brooklyn.csv

# FP-only pass (skip vacant-side panels, ~2× faster)
uv run python scripts/eval_confusion_by_class.py \
    --run outputs/models/deeplabv3plus/kahan_027 --split val --nonvacant-only

# Choose coverage threshold for figures (default: 0.2)
uv run python scripts/eval_confusion_by_class.py \
    --run outputs/models/deeplabv3plus/kahan_027 --split val --plot-coverage 0.3
```

| Output figure | Description |
|---------------|-------------|
| `confusion_fp_by_landuse_{split}.png` | FP count and FP rate per LandUse category |
| `confusion_vacant_by_bldgclass_{split}.png` | TP detection rate and FN count per BldgClass (vacant parcels only) |

### plot_prediction_comparison.py

```bash
# Single-BBL inspection (NAIP + error overlay + parcel outlines)
uv run python scripts/plot_prediction_comparison.py bbl \
    --bbl 2034910001 --arch unet --run-id kahan_031 --split test \
    --stride 256 --threshold 0.425

# Side-by-side UNet vs DeepLabV3+ comparison for one BBL
uv run python scripts/plot_prediction_comparison.py compare \
    --bbl 2034910001 --unet-run kahan_031 --deeplab-run kahan_027 --split test \
    --unet-stride 256 --unet-threshold 0.425 \
    --deeplab-stride 512 --deeplab-threshold 0.298

# TP/FP/FN/TN patch gallery
uv run python scripts/plot_prediction_comparison.py gallery \
    --run-id kahan_027 --arch deeplabv3plus --split test
```

### plot_thesis_architectures.py

```bash
uv run python scripts/plot_thesis_architectures.py           # all three diagrams
uv run python scripts/plot_thesis_architectures.py --only unet
uv run python scripts/plot_thesis_architectures.py --only deeplab
uv run python scripts/plot_thesis_architectures.py --only atrous
```

### plot_building_permits.py

Produces a single combined figure (`outputs/figures/building_permits.png`) with two panels:

- **(a)** Line chart of total permitted units per capita by MSA, with a "Start of Great Recession" marker at 2008
- **(b)** 2×2 stacked-area panels showing single- vs. multi-family permit mix per MSA, with a 2008 recession line

```bash
uv run python scripts/plot_building_permits.py

# Different per-capita metric for panel (a)
uv run python scripts/plot_building_permits.py --metric total_bldgs_per_capita
uv run python scripts/plot_building_permits.py --metric 1_unit_units_per_capita
uv run python scripts/plot_building_permits.py --metric 5_plus_units_units_per_capita

# Custom output path or start year
uv run python scripts/plot_building_permits.py --out path/to/fig.png --start-year 1990
```

## Notebook Output Stripping

This repo uses [nbstripout](https://github.com/kynan/nbstripout) — cell outputs are stripped automatically on commit.

To **preserve a specific cell's output** (e.g. a figure), add the `keep_output` tag:
- **VS Code**: cell `...` menu → Add Cell Tag → `keep_output`
- **Jupyter**: View → Cell Toolbar → Tags → type `keep_output` → Add tag

The git filter config is not tracked. After cloning or creating a new worktree, run `uv sync && uv run nbstripout --install --attributes .gitattributes` to re-register it.
