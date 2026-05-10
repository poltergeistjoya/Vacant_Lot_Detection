"""
Export parcel-level confusion results to GeoPackage for QGIS visualization.

Loads parcels, computes per-parcel predictions at a specified coverage threshold,
and exports to GeoPackage (.gpkg) with confusion labels, building class, and land use.

Columns in output:
  - BBL: Parcel ID
  - BldgClass: Building class
  - LandUse: Land use code
  - is_vacant: Ground truth (0=non-vacant, 1=vacant)
  - valid_pixels: Number of valid pixels in parcel
  - pred_fraction: Fraction of pixels predicted vacant
  - label_{cov}: TP/FP/FN/TN at specified coverage threshold
  - geometry: Parcel polygon

Usage:
    uv run python scripts/export_confusion_to_gpkg.py \\
        --run outputs/models/deeplabv3plus/kahan_027 \\
        --split val \\
        --coverage 0.2 \\
        --out outputs/models/deeplabv3plus/kahan_027/confusion_parcels_val.gpkg

    # Use test split
    uv run python scripts/export_confusion_to_gpkg.py \\
        --run outputs/models/deeplabv3plus/kahan_027 \\
        --split test \\
        --coverage 0.2 \\
        --out confusion_test.gpkg
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.mask
import yaml
from shapely.geometry import box

SCRIPT_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Shared-root resolution (same pattern as eval_confusion_by_class.py)
# ---------------------------------------------------------------------------
try:
    from vacant_lot.config import _get_shared_root
    SHARED_ROOT = _get_shared_root()
except Exception:
    p = SCRIPT_DIR.parent
    while p != p.parent:
        if (p / "outputs" / "models").exists():
            SHARED_ROOT = p
            break
        p = p.parent
    else:
        SHARED_ROOT = SCRIPT_DIR.parent

DATA_YAML = SCRIPT_DIR.parent / "config" / "data" / "nyc.yaml"

# ---------------------------------------------------------------------------
# Label mappings
# ---------------------------------------------------------------------------
LANDUSE_LABELS: dict[str, str] = {
    "01": "1-2 Family Residential",
    "02": "Multi-Family Walk-Up",
    "03": "Multi-Family Elevator",
    "04": "Mixed Res./Commercial",
    "05": "Commercial & Office",
    "06": "Industrial & Manufacturing",
    "07": "Transportation & Utility",
    "08": "Public Facilities",
    "09": "Open Space & Recreation",
    "10": "Parking Facilities",
    "11": "Vacant Land",
}

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_data_config() -> dict:
    with open(DATA_YAML) as f:
        return yaml.safe_load(f)


def load_run_config(run_dir: Path) -> dict:
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        sys.exit(f"No config.yaml found in {run_dir}")
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def get_borough_assignments(run_cfg: dict) -> dict[str, list[int]]:
    """Return {train/val/test: [boroCodes]} from the patch_splits JSON the run used."""
    splits_rel = run_cfg["data_paths"]["patch_splits"]
    splits_path = SHARED_ROOT / splits_rel
    if not splits_path.exists():
        sys.exit(f"patch_splits not found: {splits_path}")
    d = json.loads(splits_path.read_text())
    return d.get("split", {})


def using_v2_mask(run_cfg: dict) -> bool:
    return "v2" in run_cfg["data_paths"].get("vacancy_mask", "")


# ---------------------------------------------------------------------------
# Reused from eval_parcel_level.py / eval_confusion_by_class.py
# ---------------------------------------------------------------------------

def get_pixel_threshold(run_dir: Path, split: str,
                         override: float | None) -> float:
    if override is not None:
        return override
    pr_path = run_dir / "pr_curves.npz"
    if not pr_path.exists():
        print("[warn] no pr_curves.npz found; defaulting to threshold=0.5",
              file=sys.stderr)
        return 0.5
    d = np.load(pr_path)
    key_p = f"{split}_pr_precision"
    if key_p not in d:
        print(f"[warn] {key_p} not in pr_curves.npz; defaulting to 0.5",
              file=sys.stderr)
        return 0.5
    prec   = d[f"{split}_pr_precision"][:-1]
    rec    = d[f"{split}_pr_recall"][:-1]
    thresh = d[f"{split}_pr_thresholds"]
    f2 = 5 * prec * rec / np.maximum(4 * prec + rec, 1e-8)
    thr = float(thresh[np.argmax(f2)])
    print(f"F2-optimal threshold from {split} PR curve: {thr:.4f}")
    return thr


def find_prob_tif(run_dir: Path, split: str, run_cfg: dict) -> Path:
    figs = run_dir / "figures"

    # Try to get stride from run config to find exact TIF
    stride = run_cfg.get("eval_stride", None)
    if stride:
        exact_path = figs / f"{split}_pred_s{stride}.tif"
        if exact_path.exists():
            return exact_path

    # Fall back to wildcard search
    candidates = sorted(figs.glob(f"{split}_pred_s*.tif"))
    if not candidates:
        raise FileNotFoundError(
            f"No prob TIF found for split '{split}' in {figs}\n"
            f"  Expected pattern: {split}_pred_s*.tif\n"
            f"  Run visualize_predictions.py first."
        )
    if len(candidates) > 1:
        print(f"[warn] multiple prob TIFs found; using {candidates[-1].name}",
              file=sys.stderr)
    return candidates[-1]


def parcel_pred_fraction(geom, src: rasterio.DatasetReader,
                          pixel_threshold: float,
                          nodata: float = -1.0) -> tuple[int, int]:
    try:
        arr, _ = rasterio.mask.mask(src, [geom], crop=True, nodata=nodata,
                                    all_touched=False)
        data = arr[0]
        valid_mask = data != nodata
        valid_count = int(valid_mask.sum())
        if valid_count == 0:
            return 0, 0
        pred_count = int((data[valid_mask] >= pixel_threshold).sum())
        return valid_count, pred_count
    except Exception:
        return 0, 0


# ---------------------------------------------------------------------------
# All-parcel loader
# ---------------------------------------------------------------------------

def load_all_parcels(data_cfg: dict, apply_v2_corrections: bool) -> gpd.GeoDataFrame:
    parcels_cfg = data_cfg["parcels"]
    labels_cfg  = data_cfg.get("labels", {})

    gdb_path = SHARED_ROOT / parcels_cfg["gdb_path"]
    layer    = parcels_cfg["layer"]
    id_col   = parcels_cfg.get("id_column", "BBL")
    lc_col   = parcels_cfg.get("landuse_column", "BldgClass")
    vacant_codes = set(parcels_cfg["vacant_codes"])

    omit_bbls       = set(labels_cfg.get("omit_bbls", []))
    force_nonvacant = set(labels_cfg.get("force_nonvacant_bbls", []))
    force_vacant    = set(labels_cfg.get("force_vacant_bbls", []))

    print(f"Loading MapPLUTO from {gdb_path} ...")
    gdf = gpd.read_file(gdb_path, layer=layer)
    gdf[id_col] = gdf[id_col].astype(int)

    # Drop omit BBLs
    gdf = gdf[~gdf[id_col].isin(omit_bbls)].copy()

    # Apply corrections only if run used v2 mask
    if apply_v2_corrections:
        gdf.loc[gdf[id_col].isin(force_vacant),    lc_col] = "V0"
        gdf.loc[gdf[id_col].isin(force_nonvacant), lc_col] = "XX"
        print(f"  V2 corrections applied ({len(force_vacant)} force-vacant, "
              f"{len(force_nonvacant)} force-nonvacant)")

    # Tag vacancy
    gdf["is_vacant"] = gdf[lc_col].str[:2].isin(vacant_codes)

    print(f"  Parcels loaded: {len(gdf):,}  "
          f"(vacant: {gdf['is_vacant'].sum():,}, "
          f"non-vacant: {(~gdf['is_vacant']).sum():,})")
    return gdf


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export parcel-level confusion results to GeoPackage for QGIS"
    )
    parser.add_argument("--run", required=True,
                        help="Run directory (relative to shared root or absolute)")
    parser.add_argument("--split", default="val",
                        help="Split to evaluate: val or test")
    parser.add_argument("--pixel-threshold", type=float, default=None,
                        help="Pixel threshold (default: F2-optimal from pr_curves.npz)")
    parser.add_argument("--coverage", type=float, default=0.2,
                        help="Parcel coverage threshold (default: 0.2)")
    parser.add_argument("--min-parcel-pixels", type=int, default=5,
                        help="Minimum valid pixels to include a parcel (default: 5)")
    parser.add_argument("--out", default=None,
                        help="Output GeoPackage path (default: {run}/confusion_parcels_{split}.gpkg)")
    args = parser.parse_args()

    # Resolve run directory
    run_path = Path(args.run)
    if not run_path.is_absolute():
        run_dir = SHARED_ROOT / run_path
        if not run_dir.exists():
            try:
                num = int(run_path.name)
                run_dir = SHARED_ROOT / run_path.parent / f"{num:03d}"
            except ValueError:
                pass
    else:
        run_dir = run_path

    if not run_dir.exists():
        sys.exit(f"Run directory not found: {run_dir}")

    run_id = run_dir.name

    # Default output path if not specified
    if args.out is None:
        args.out = f"confusion_parcels_{args.split}.gpkg"

    print(f"Shared root : {SHARED_ROOT}")
    print(f"Run dir     : {run_dir}")
    print(f"Run ID      : {run_id}")
    print(f"Split       : {args.split}")
    print(f"Coverage    : {args.coverage*100:.0f}%")
    print(f"Output      : {args.out}")

    # Load configs
    data_cfg = load_data_config()
    run_cfg  = load_run_config(run_dir)
    apply_v2 = using_v2_mask(run_cfg)
    print(f"V2 mask     : {apply_v2}")

    # Determine borough filter from run's patch_splits
    borough_assignments = get_borough_assignments(run_cfg)
    key = f"{args.split}_boroughs"
    split_boroughs: list[int] = borough_assignments.get(key, [])
    if not split_boroughs:
        print(f"[warn] No borough assignment found for key '{key}' in patch_splits; "
              f"no borough filter applied", file=sys.stderr)
    else:
        boro_names = {1: "Manhattan", 2: "Bronx", 3: "Brooklyn",
                      4: "Queens", 5: "Staten Island"}
        names = [boro_names.get(b, str(b)) for b in split_boroughs]
        print(f"Boroughs    : {split_boroughs} ({', '.join(names)})")

    # Pixel threshold and prob TIF
    pixel_thr = get_pixel_threshold(run_dir, args.split, args.pixel_threshold)
    prob_tif  = find_prob_tif(run_dir, args.split, run_cfg)
    print(f"Prob TIF    : {prob_tif.name}")
    print(f"Pixel thr   : {pixel_thr:.4f}")

    # Load parcels
    gdf = load_all_parcels(data_cfg, apply_v2_corrections=apply_v2)

    # Reproject
    with rasterio.open(prob_tif) as src:
        tif_crs    = src.crs
        tif_bounds = src.bounds

    print(f"Reprojecting parcels to {tif_crs} ...")
    gdf = gdf.to_crs(tif_crs)

    # Borough filter
    if split_boroughs:
        boro_col = next((c for c in gdf.columns
                         if c.lower() in ("borocode", "boro_code", "boro")), None)
        if boro_col:
            gdf = gdf[gdf[boro_col].astype(int).isin(split_boroughs)].copy()
            print(f"After borough filter: {len(gdf):,} parcels")
        else:
            print("[warn] BoroCode column not found; borough filter skipped",
                  file=sys.stderr)

    # Spatial filter to TIF extent
    tif_box = box(tif_bounds.left, tif_bounds.bottom,
                  tif_bounds.right, tif_bounds.top)
    gdf = gdf[gdf.geometry.intersects(tif_box)].copy()
    print(f"Parcels within TIF extent: {len(gdf):,}")

    # Per-parcel prediction fractions
    print(f"\nComputing per-parcel predictions (pixel_thr={pixel_thr:.4f}) ...")
    valid_pixels_list: list[int] = []
    pred_pixels_list:  list[int] = []

    with rasterio.open(prob_tif) as src:
        nodata_val = src.nodata if src.nodata is not None else -1.0
        total = len(gdf)
        for i, (_, row) in enumerate(gdf.iterrows()):
            if i % 1000 == 0:
                print(f"  {i}/{total} parcels ...", end="\r")
            vp, pp = parcel_pred_fraction(row.geometry, src, pixel_thr,
                                          nodata=nodata_val)
            valid_pixels_list.append(vp)
            pred_pixels_list.append(pp)

    print(f"\nDone.                    ")

    gdf = gdf.copy()
    gdf["valid_pixels"] = valid_pixels_list
    gdf["pred_pixels"]  = pred_pixels_list
    gdf["pred_fraction"] = np.where(
        np.array(valid_pixels_list) > 0,
        np.array(pred_pixels_list) / np.array(valid_pixels_list),
        np.nan,
    )

    # Apply min-pixel filter
    evaluable = gdf[gdf["valid_pixels"] >= args.min_parcel_pixels].copy()
    print(f"\nEvaluable parcels (>= {args.min_parcel_pixels} valid pixels): "
          f"{len(evaluable):,} / {len(gdf):,}")

    id_col = data_cfg["parcels"].get("id_column", "BBL")

    # Classify parcels
    cov = args.coverage
    cov_key = f"label_{int(cov*100)}"

    for _, row in evaluable.iterrows():
        pf = row["pred_fraction"]
        is_vac = bool(row["is_vacant"])

        if np.isnan(pf):
            evaluable.loc[_, cov_key] = "NA"
        elif is_vac:
            evaluable.loc[_, cov_key] = "TP" if pf >= cov else "FN"
        else:
            evaluable.loc[_, cov_key] = "FP" if pf >= cov else "TN"

    # Keep only relevant columns
    id_col_actual = next((c for c in evaluable.columns if c == id_col), "BBL")
    bldgclass_col = next((c for c in evaluable.columns if c.lower() == "bldgclass"), "BldgClass")
    landuse_col = next((c for c in evaluable.columns if c.lower() == "landuse"), "LandUse")

    output_cols = [
        id_col_actual, bldgclass_col, landuse_col,
        "is_vacant", "valid_pixels", "pred_fraction", cov_key, "geometry"
    ]
    output_cols = [c for c in output_cols if c in evaluable.columns]

    out_gdf = evaluable[output_cols].copy()
    out_gdf = out_gdf.rename(columns={
        id_col_actual: "BBL",
        bldgclass_col: "BldgClass",
        landuse_col: "LandUse",
    })

    # Write to GeoPackage
    out_path = Path(args.out)
    if not out_path.is_absolute():
        # If just a filename (no parent dirs), save to run dir
        if "/" not in args.out and "\\" not in args.out:
            out_path = run_dir / out_path
        else:
            out_path = SHARED_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nWriting {len(out_gdf):,} parcels to {out_path} ...")
    out_gdf.to_file(out_path, driver="GPKG")
    print(f"Saved → {out_path}")
    print(f"  Columns: {', '.join(out_gdf.columns)}")
    print(f"  CRS: {out_gdf.crs}")


if __name__ == "__main__":
    main()
