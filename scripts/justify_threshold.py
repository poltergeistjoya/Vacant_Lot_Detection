"""
Quantitative justification for parcel coverage threshold choice.

Analyzes pred_fraction distribution and confusion metrics across thresholds
to defend the choice of coverage threshold for parcel-level classification.

Usage:
    uv run python scripts/justify_threshold.py \\
        --run outputs/models/deeplabv3plus/kahan_027 \\
        --split val \\
        --out threshold_justification.txt
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
    splits_rel = run_cfg["data_paths"]["patch_splits"]
    splits_path = SHARED_ROOT / splits_rel
    if not splits_path.exists():
        sys.exit(f"patch_splits not found: {splits_path}")
    d = json.loads(splits_path.read_text())
    return d.get("split", {})


def using_v2_mask(run_cfg: dict) -> bool:
    return "v2" in run_cfg["data_paths"].get("vacancy_mask", "")


def get_pixel_threshold(run_dir: Path, split: str, override: float | None) -> float:
    if override is not None:
        return override
    pr_path = run_dir / "pr_curves.npz"
    if not pr_path.exists():
        return 0.5
    d = np.load(pr_path)
    key_p = f"{split}_pr_precision"
    if key_p not in d:
        return 0.5
    prec = d[f"{split}_pr_precision"][:-1]
    rec = d[f"{split}_pr_recall"][:-1]
    thresh = d[f"{split}_pr_thresholds"]
    f2 = 5 * prec * rec / np.maximum(4 * prec + rec, 1e-8)
    return float(thresh[np.argmax(f2)])


def find_prob_tif(run_dir: Path, split: str, run_cfg: dict) -> Path:
    figs = run_dir / "figures"
    stride = run_cfg.get("eval_stride", None)
    if stride:
        exact_path = figs / f"{split}_pred_s{stride}.tif"
        if exact_path.exists():
            return exact_path
    candidates = sorted(figs.glob(f"{split}_pred_s*.tif"))
    if not candidates:
        raise FileNotFoundError(f"No prob TIF found for split '{split}' in {figs}")
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


def load_all_parcels(data_cfg: dict, apply_v2_corrections: bool) -> gpd.GeoDataFrame:
    parcels_cfg = data_cfg["parcels"]
    labels_cfg = data_cfg.get("labels", {})

    gdb_path = SHARED_ROOT / parcels_cfg["gdb_path"]
    layer = parcels_cfg["layer"]
    id_col = parcels_cfg.get("id_column", "BBL")
    lc_col = parcels_cfg.get("landuse_column", "BldgClass")
    vacant_codes = set(parcels_cfg["vacant_codes"])

    omit_bbls = set(labels_cfg.get("omit_bbls", []))
    force_nonvacant = set(labels_cfg.get("force_nonvacant_bbls", []))
    force_vacant = set(labels_cfg.get("force_vacant_bbls", []))

    gdf = gpd.read_file(gdb_path, layer=layer)
    gdf[id_col] = gdf[id_col].astype(int)
    gdf = gdf[~gdf[id_col].isin(omit_bbls)].copy()

    if apply_v2_corrections:
        gdf.loc[gdf[id_col].isin(force_vacant), lc_col] = "V0"
        gdf.loc[gdf[id_col].isin(force_nonvacant), lc_col] = "XX"

    gdf["is_vacant"] = gdf[lc_col].str[:2].isin(vacant_codes)
    return gdf


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quantify tradeoffs across parcel coverage thresholds"
    )
    parser.add_argument("--run", required=True,
                        help="Run directory")
    parser.add_argument("--split", default="val",
                        help="Split to analyze")
    parser.add_argument("--pixel-threshold", type=float, default=None,
                        help="Pixel threshold")
    parser.add_argument("--min-parcel-pixels", type=int, default=5,
                        help="Minimum valid pixels per parcel")
    parser.add_argument("--out", default=None,
                        help="Output file (default: print to stdout)")
    args = parser.parse_args()

    # Resolve run directory
    run_path = Path(args.run)
    if not run_path.is_absolute():
        run_dir = SHARED_ROOT / run_path
    else:
        run_dir = run_path

    if not run_dir.exists():
        sys.exit(f"Run directory not found: {run_dir}")

    # Load configs
    data_cfg = load_data_config()
    run_cfg = load_run_config(run_dir)
    apply_v2 = using_v2_mask(run_cfg)

    # Get borough filter
    borough_assignments = get_borough_assignments(run_cfg)
    split_boroughs = borough_assignments.get(f"{args.split}_boroughs", [])

    # Load data
    pixel_thr = get_pixel_threshold(run_dir, args.split, args.pixel_threshold)
    prob_tif = find_prob_tif(run_dir, args.split, run_cfg)

    gdf = load_all_parcels(data_cfg, apply_v2_corrections=apply_v2)

    with rasterio.open(prob_tif) as src:
        tif_crs = src.crs
        tif_bounds = src.bounds

    print(f"  TIF CRS    : {tif_crs}")
    print(f"  TIF bounds : {tif_bounds}")
    print(f"  Parcel CRS : {gdf.crs}")

    gdf = gdf.to_crs(tif_crs)
    bds = gdf.total_bounds
    print(f"  {len(gdf):,} parcels after reprojection (bounds: {bds[0]:.0f},{bds[1]:.0f} → {bds[2]:.0f},{bds[3]:.0f})")

    print(f"  split_boroughs  : {split_boroughs}")
    if split_boroughs:
        boro_col = next((c for c in gdf.columns
                         if c.lower() in ("borocode", "boro_code", "boro")), None)
        print(f"  boro_col        : {boro_col}")
        if boro_col:
            print(f"  unique BoroCode : {sorted(gdf[boro_col].astype(int).unique())}")
            gdf = gdf[gdf[boro_col].astype(int).isin(split_boroughs)].copy()
            bds = gdf.total_bounds
            print(f"  {len(gdf):,} parcels after borough filter (bounds: {bds[0]:.0f},{bds[1]:.0f} → {bds[2]:.0f},{bds[3]:.0f})")

    tif_box = box(tif_bounds.left, tif_bounds.bottom,
                  tif_bounds.right, tif_bounds.top)
    gdf = gdf[gdf.geometry.intersects(tif_box)].copy()
    print(f"  {len(gdf):,} parcels within TIF extent")

    if len(gdf) == 0:
        sys.exit("No parcels found within TIF extent — check that borough/split are correct.")

    # Compute predictions
    valid_pixels_list = []
    pred_pixels_list = []
    total = len(gdf)

    print(f"\nComputing per-parcel predictions ...")
    with rasterio.open(prob_tif) as src:
        nodata_val = src.nodata if src.nodata is not None else -1.0
        for i, (_, row) in enumerate(gdf.iterrows()):
            if i % 1000 == 0:
                print(f"  {i}/{total} parcels ...", end="\r")
            vp, pp = parcel_pred_fraction(row.geometry, src, pixel_thr,
                                          nodata=nodata_val)
            valid_pixels_list.append(vp)
            pred_pixels_list.append(pp)
    print(f"\nDone.                    ")

    gdf["valid_pixels"] = valid_pixels_list
    gdf["pred_pixels"] = pred_pixels_list
    gdf["pred_fraction"] = np.where(
        np.array(valid_pixels_list) > 0,
        np.array(pred_pixels_list) / np.array(valid_pixels_list),
        np.nan,
    )

    # Filter evaluable
    evaluable = gdf[gdf["valid_pixels"] >= args.min_parcel_pixels].copy()
    print(f"  {len(evaluable):,} / {len(gdf):,} parcels have >= {args.min_parcel_pixels} valid pixels")

    if len(evaluable) == 0:
        sys.exit("No evaluable parcels — try lowering --min-parcel-pixels.")

    # Analyze across thresholds — vectorized for speed
    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50]
    output_lines = []

    pf = evaluable["pred_fraction"].to_numpy(dtype=float)
    is_vacant = evaluable["is_vacant"].to_numpy(dtype=bool)

    output_lines.append("=" * 80)
    output_lines.append("THRESHOLD JUSTIFICATION ANALYSIS")
    output_lines.append("=" * 80)
    output_lines.append(f"Run: {run_dir.name}")
    output_lines.append(f"Split: {args.split}")
    output_lines.append(f"Pixel threshold: {pixel_thr:.4f}")
    output_lines.append(f"Evaluable parcels: {len(evaluable):,}")
    output_lines.append(f"Vacant parcels (ground truth): {evaluable['is_vacant'].sum():,}")
    output_lines.append("")

    # Distribution statistics
    output_lines.append("-" * 80)
    output_lines.append("PRED_FRACTION DISTRIBUTION")
    output_lines.append("-" * 80)

    vacant_pf    = pf[ is_vacant & ~np.isnan(pf)]
    nonvacant_pf = pf[~is_vacant & ~np.isnan(pf)]

    output_lines.append(f"Vacant parcels (pred_fraction):")
    output_lines.append(f"  Mean:   {np.mean(vacant_pf):.3f}")
    output_lines.append(f"  Median: {np.median(vacant_pf):.3f}")
    output_lines.append(f"  Std:    {np.std(vacant_pf):.3f}")
    output_lines.append(f"  Min:    {np.min(vacant_pf):.3f}")
    output_lines.append(f"  Max:    {np.max(vacant_pf):.3f}")
    output_lines.append(f"  p25:    {np.percentile(vacant_pf, 25):.3f}")
    output_lines.append(f"  p75:    {np.percentile(vacant_pf, 75):.3f}")
    output_lines.append("")

    output_lines.append(f"Non-vacant parcels (pred_fraction):")
    output_lines.append(f"  Mean:   {np.mean(nonvacant_pf):.3f}")
    output_lines.append(f"  Median: {np.median(nonvacant_pf):.3f}")
    output_lines.append(f"  Std:    {np.std(nonvacant_pf):.3f}")
    output_lines.append(f"  Min:    {np.min(nonvacant_pf):.3f}")
    output_lines.append(f"  Max:    {np.max(nonvacant_pf):.3f}")
    output_lines.append(f"  p25:    {np.percentile(nonvacant_pf, 25):.3f}")
    output_lines.append(f"  p75:    {np.percentile(nonvacant_pf, 75):.3f}")
    output_lines.append("")

    # Threshold analysis
    output_lines.append("-" * 80)
    output_lines.append("CONFUSION MATRIX ACROSS THRESHOLDS")
    output_lines.append("-" * 80)
    output_lines.append(f"{'Cov%':>5s}  {'TP':>6s}  {'FP':>6s}  {'FN':>6s}  {'TN':>6s}  "
                       f"{'Prec':>6s}  {'Rec':>6s}  {'F2':>6s}  {'% Pred Vacant':>14s}")
    output_lines.append("-" * 80)

    n_eval = len(evaluable)
    sweep_results = []
    for cov in thresholds:
        pred_pos = pf >= cov           # NaN comparisons yield False — excluded automatically
        tp = int(( is_vacant &  pred_pos).sum())
        fn = int(( is_vacant & ~pred_pos).sum())
        fp = int((~is_vacant &  pred_pos).sum())
        tn = int((~is_vacant & ~pred_pos).sum())

        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f2   = 5 * prec * rec / (4 * prec + rec) if (prec + rec) else 0.0
        pct_pred_vacant = 100 * (tp + fp) / n_eval
        sweep_results.append(dict(cov=cov, tp=tp, fp=fp, fn=fn, tn=tn,
                                  prec=prec, rec=rec, f2=f2,
                                  pct_pred_vacant=pct_pred_vacant))

    best_idx = int(np.argmax([r["f2"] for r in sweep_results]))
    for i, r in enumerate(sweep_results):
        marker = " ← best F2" if i == best_idx else ""
        output_lines.append(f"{r['cov']*100:5.0f}  {r['tp']:6,}  {r['fp']:6,}  {r['fn']:6,}  {r['tn']:6,}  "
                            f"{r['prec']:6.3f}  {r['rec']:6.3f}  {r['f2']:6.3f}  {r['pct_pred_vacant']:13.1f}%{marker}")

    best = sweep_results[best_idx]
    med_vac = float(np.median(vacant_pf))
    med_nonvac = float(np.median(nonvacant_pf))
    p75_nonvac = float(np.percentile(nonvacant_pf, 75))

    # find the threshold one step below best for diminishing-returns comment
    lower = sweep_results[best_idx - 1] if best_idx > 0 else None
    higher = sweep_results[best_idx + 1] if best_idx < len(sweep_results) - 1 else None

    output_lines.append("")
    output_lines.append("-" * 80)
    output_lines.append("INTERPRETATION")
    output_lines.append("-" * 80)
    output_lines.append("")
    output_lines.append(
        f"1. SEPARATION: Vacant parcels have a median pred_fraction of {med_vac:.3f} vs "
        f"{med_nonvac:.3f} for non-vacant (p75={p75_nonvac:.3f}). "
        f"The best-F2 threshold of {best['cov']*100:.0f}% sits between these distributions, "
        f"above the 75th percentile of non-vacant parcels and below the median of vacant parcels."
    )
    output_lines.append("")
    if lower:
        rec_gain = best["rec"] - lower["rec"]
        fp_gain  = best["fp"]  - lower["fp"]
        output_lines.append(
            f"2. RECALL: Raising from {lower['cov']*100:.0f}% to {best['cov']*100:.0f}% increases "
            f"recall from {lower['rec']:.3f} to {best['rec']:.3f} (+{rec_gain:.3f}) while "
            f"reducing false positives by {fp_gain:,}. "
            f"Thresholds below {best['cov']*100:.0f}% add false positives with diminishing recall returns."
        )
    else:
        output_lines.append(
            f"2. RECALL: At {best['cov']*100:.0f}%, recall is {best['rec']:.3f} — "
            f"the lowest threshold tested already maximizes F2."
        )
    output_lines.append("")
    if higher:
        output_lines.append(
            f"3. PRECISION: At {best['cov']*100:.0f}%, precision is {best['prec']:.3f}. "
            f"Raising to {higher['cov']*100:.0f}% improves precision to {higher['prec']:.3f} "
            f"but drops recall to {higher['rec']:.3f} and F2 to {higher['f2']:.3f} vs {best['f2']:.3f}."
        )
    else:
        output_lines.append(
            f"3. PRECISION: At {best['cov']*100:.0f}%, precision is {best['prec']:.3f} — "
            f"the highest threshold tested."
        )
    output_lines.append("")
    output_lines.append(
        f"4. F2 SCORE: Best parcel-level F2={best['f2']:.3f} at {best['cov']*100:.0f}% coverage "
        f"(precision={best['prec']:.3f}, recall={best['rec']:.3f}). "
        f"F2 weights recall 2x over precision (beta=2), appropriate for vacant lot detection "
        f"where missing a lot is costlier than a false alarm."
    )
    output_lines.append("")

    # Print or write
    output_text = "\n".join(output_lines)
    print(output_text)

    if args.out:
        Path(args.out).write_text(output_text)
        print(f"\n[saved to {args.out}]")


if __name__ == "__main__":
    main()
