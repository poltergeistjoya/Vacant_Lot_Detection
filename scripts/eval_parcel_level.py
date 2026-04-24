"""
Parcel-level vacancy detection recall.

For each vacant parcel in MapPLUTO (with v2 mask corrections applied),
computes the fraction of parcel pixels predicted as vacant at the model's
pixel threshold. Reports recall at parcel coverage thresholds of 30%,
50%, and 70%.

A parcel is "detected" if >= coverage_threshold of its valid pixels are
predicted vacant at pixel_threshold.

Figures saved into --fig-dir (default: {run}/figures/):
  parcel_recall_vs_coverage_{split}.png  — recall curve sweeping coverage 0→1
  parcel_pred_fraction_hist_{split}.png  — histogram of per-parcel pred fractions

Usage:
    uv run python scripts/eval_parcel_level.py \\
        --run outputs/models/deeplabv3plus/kahan_027 \\
        --split val

    # Explicit pixel threshold (default: auto-compute F2-optimal from pr_curves.npz)
    uv run python scripts/eval_parcel_level.py \\
        --run outputs/models/deeplabv3plus/kahan_027 \\
        --split val \\
        --pixel-threshold 0.298

    # Save per-parcel CSV
    uv run python scripts/eval_parcel_level.py \\
        --run outputs/models/deeplabv3plus/kahan_027 \\
        --split val \\
        --out outputs/models/deeplabv3plus/kahan_027/parcel_eval_brooklyn.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import rasterio.features
import rasterio.mask
import yaml

mpl.rcParams.update({
    "font.family":      "STIX Two Text",
    "mathtext.fontset": "stix",
    "font.size":        8,
    "axes.titleweight": "normal",
    "axes.labelweight": "normal",
})

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

DATA_YAML = SCRIPT_DIR.parent / "config" / "data.yaml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_data_config() -> dict:
    with open(DATA_YAML) as f:
        return yaml.safe_load(f)


def f2_optimal_threshold(prec: np.ndarray, rec: np.ndarray,
                         thresh: np.ndarray) -> float:
    f2 = 5 * prec * rec / np.maximum(4 * prec + rec, 1e-8)
    return float(thresh[np.argmax(f2)])


def get_pixel_threshold(run_dir: Path, split: str,
                        override: float | None) -> float:
    if override is not None:
        return override
    pr_path = run_dir / "pr_curves.npz"
    if not pr_path.exists():
        print(f"[warn] no pr_curves.npz found; defaulting to threshold=0.5",
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
    thr = f2_optimal_threshold(prec, rec, thresh)
    print(f"F2-optimal threshold from {split} PR curve: {thr:.4f}")
    return thr


def find_prob_tif(run_dir: Path, split: str) -> Path:
    """Glob for the probability TIF for this split."""
    figs = run_dir / "figures"
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


def load_vacant_parcels(cfg: dict) -> gpd.GeoDataFrame:
    """Load MapPLUTO, filter to vacant parcels, apply v2 corrections."""
    parcels_cfg = cfg["parcels"]
    labels_cfg  = cfg.get("labels", {})

    gdb_path = SHARED_ROOT / parcels_cfg["gdb_path"]
    layer    = parcels_cfg["layer"]
    id_col   = parcels_cfg.get("id_column", "BBL")
    lc_col   = parcels_cfg.get("landuse_column", "BldgClass")
    vacant_codes = set(parcels_cfg["vacant_codes"])

    omit_bbls        = set(labels_cfg.get("omit_bbls", []))
    force_nonvacant  = set(labels_cfg.get("force_nonvacant_bbls", []))
    force_vacant     = set(labels_cfg.get("force_vacant_bbls", []))

    print(f"Loading MapPLUTO from {gdb_path} ...")
    gdf = gpd.read_file(gdb_path, layer=layer)
    gdf[id_col] = gdf[id_col].astype(int)

    # Apply corrections before filtering
    gdf.loc[gdf[id_col].isin(force_vacant),    lc_col] = "V0"   # mark as vacant
    gdf.loc[gdf[id_col].isin(force_nonvacant), lc_col] = "XX"   # mark as non-vacant

    # Filter to vacant, drop omit BBLs
    vacant = gdf[
        gdf[lc_col].str[:2].isin(vacant_codes) &
        ~gdf[id_col].isin(omit_bbls)
    ].copy()

    print(f"  Vacant parcels after corrections: {len(vacant):,}")
    return vacant


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

def _apply_style(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.4)
    ax.spines["bottom"].set_linewidth(0.4)
    ax.tick_params(labelsize=7, length=2, width=0.4)
    ax.grid(axis="y", linestyle=":", linewidth=0.3, color="#cccccc")
    ax.set_axisbelow(True)


def plot_recall_vs_coverage(pred_fractions: np.ndarray, split: str,
                             coverage_marks: list[float],
                             fig_dir: Path) -> None:
    """Recall curve: x = coverage threshold, y = fraction of vacant parcels detected."""
    sweep = np.linspace(0, 1, 201)
    recall = np.array([(pred_fractions >= t).mean() for t in sweep])

    fig, ax = plt.subplots(figsize=(5, 3.2), constrained_layout=True)
    ax.plot(sweep * 100, recall * 100, color="#1b7837", linewidth=1.2)

    for cov in coverage_marks:
        r = float((pred_fractions >= cov).mean())
        ax.axvline(cov * 100, color="#888888", linewidth=0.6, linestyle="--")
        ax.annotate(f"{r:.0%}", xy=(cov * 100, r * 100),
                    xytext=(4, -2), textcoords="offset points",
                    fontsize=6.5, color="#555555")

    ax.set_xlabel("Parcel coverage threshold (%)", fontsize=8)
    ax.set_ylabel("Recall (%)", fontsize=8)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 105)
    ax.set_title(f"Parcel-level recall vs. coverage threshold ({split})", fontsize=8)
    _apply_style(ax)

    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / f"parcel_recall_vs_coverage_{split}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


def plot_pred_fraction_hist(pred_fractions: np.ndarray, split: str,
                             coverage_marks: list[float],
                             fig_dir: Path) -> None:
    """Histogram of per-parcel prediction fraction for vacant parcels."""
    fig, ax = plt.subplots(figsize=(5, 3.2), constrained_layout=True)
    ax.hist(pred_fractions, bins=50, color="#4393c3", edgecolor="white",
            linewidth=0.3)

    for cov in coverage_marks:
        ax.axvline(cov * 100, color="#d6604d", linewidth=0.8, linestyle="--",
                   label=f"{cov:.0%}")

    ax.set_xlabel("% of parcel pixels predicted vacant", fontsize=8)
    ax.set_ylabel("Vacant parcel count", fontsize=8)
    ax.set_title(f"Pixel coverage distribution over vacant parcels ({split})", fontsize=8)
    ax.legend(title="Coverage marks", fontsize=6.5, title_fontsize=6.5,
              frameon=False)
    _apply_style(ax)
    ax.grid(axis="x", linestyle=":", linewidth=0.3, color="#cccccc")

    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / f"parcel_pred_fraction_hist_{split}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


def parcel_pred_fraction(geom, src: rasterio.DatasetReader,
                         pixel_threshold: float,
                         nodata: float = -1.0) -> tuple[int, int]:
    """
    Return (valid_pixels, predicted_vacant_pixels) for a single parcel geometry.
    valid_pixels excludes nodata areas in the prob TIF.
    """
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
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parcel-level vacant lot detection recall"
    )
    parser.add_argument("--run", required=True,
                        help="Run directory, relative to shared root or absolute "
                             "(e.g. outputs/models/deeplabv3plus/027)")
    parser.add_argument("--split", default="val",
                        help="Split whose prob TIF to use (val or test)")
    parser.add_argument("--pixel-threshold", type=float, default=None,
                        help="Model pixel threshold (default: F2-optimal from pr_curves.npz)")
    parser.add_argument("--coverage", nargs="+", type=float,
                        default=[0.3, 0.5, 0.7],
                        help="Parcel coverage thresholds to evaluate (default: 0.3 0.5 0.7)")
    parser.add_argument("--min-parcel-pixels", type=int, default=5,
                        help="Minimum valid pixels in TIF to include a parcel (default: 5)")
    parser.add_argument("--boroughs", nargs="+", type=int, default=None,
                        help="Filter parcels by BoroCode (1=Manhattan 2=Bronx 3=Brooklyn "
                             "4=Queens 5=SI). Default: all non-excluded boroughs.")
    parser.add_argument("--out", default=None,
                        help="Save per-parcel results to this CSV path "
                             "(relative paths resolved from shared root)")
    parser.add_argument("--fig-dir", default=None,
                        help="Directory for output figures "
                             "(default: {run}/figures/)")
    parser.add_argument("--no-figures", action="store_true",
                        help="Skip figure generation")
    args = parser.parse_args()

    # Resolve run directory
    run_path = Path(args.run)
    if not run_path.is_absolute():
        run_dir = SHARED_ROOT / run_path
        # also try zero-padded plain number
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

    print(f"Shared root : {SHARED_ROOT}")
    print(f"Run dir     : {run_dir}")
    print(f"Split       : {args.split}")

    # Load config and data
    cfg = load_data_config()
    pixel_thr = get_pixel_threshold(run_dir, args.split, args.pixel_threshold)
    prob_tif  = find_prob_tif(run_dir, args.split)
    print(f"Prob TIF    : {prob_tif.name}")
    print(f"Pixel thr   : {pixel_thr:.4f}")

    vacant = load_vacant_parcels(cfg)

    # Reproject to TIF CRS
    with rasterio.open(prob_tif) as src:
        tif_crs = src.crs
        tif_bounds = src.bounds

    print(f"Reprojecting parcels to {tif_crs} ...")
    source_crs = cfg["parcels"].get("source_crs", "EPSG:2263")
    vacant = vacant.to_crs(tif_crs)

    # Filter by borough if requested
    if args.boroughs:
        boro_col = next((c for c in vacant.columns
                         if c.lower() in ("borocode", "boro_code", "boro")), None)
        if boro_col:
            vacant = vacant[vacant[boro_col].astype(int).isin(args.boroughs)]
            print(f"Filtered to boroughs {args.boroughs}: {len(vacant):,} parcels")
        else:
            print("[warn] BoroCode column not found; borough filter skipped",
                  file=sys.stderr)

    # Spatial filter to TIF extent (fast pre-filter)
    from shapely.geometry import box
    tif_box = box(tif_bounds.left, tif_bounds.bottom,
                  tif_bounds.right, tif_bounds.top)
    vacant = vacant[vacant.geometry.intersects(tif_box)].copy()
    print(f"Parcels within TIF extent: {len(vacant):,}")

    # Per-parcel prediction fractions
    print(f"Computing per-parcel predictions (pixel_thr={pixel_thr:.4f}) ...")
    id_col = cfg["parcels"].get("id_column", "BBL")

    valid_pixels_list = []
    pred_pixels_list  = []

    with rasterio.open(prob_tif) as src:
        nodata_val = src.nodata if src.nodata is not None else -1.0
        for i, (_, row) in enumerate(vacant.iterrows()):
            if i % 500 == 0:
                print(f"  {i}/{len(vacant)} parcels ...", end="\r")
            vp, pp = parcel_pred_fraction(
                row.geometry, src, pixel_thr, nodata=nodata_val
            )
            valid_pixels_list.append(vp)
            pred_pixels_list.append(pp)

    print(f"\nDone.")

    vacant = vacant.copy()
    vacant["valid_pixels"] = valid_pixels_list
    vacant["pred_pixels"]  = pred_pixels_list
    vacant["pred_fraction"] = np.where(
        np.array(valid_pixels_list) > 0,
        np.array(pred_pixels_list) / np.array(valid_pixels_list),
        np.nan
    )

    # Apply minimum pixel filter
    evaluable = vacant[vacant["valid_pixels"] >= args.min_parcel_pixels].copy()
    print(f"\nEvaluable parcels (>= {args.min_parcel_pixels} valid pixels): "
          f"{len(evaluable):,} / {len(vacant):,}")

    # Summary
    print(f"\n{'Coverage threshold':>20s}  {'Detected':>8s}  {'Total':>7s}  {'Recall':>8s}")
    print("-" * 52)
    for cov in sorted(args.coverage):
        detected = int((evaluable["pred_fraction"] >= cov).sum())
        total    = len(evaluable)
        recall   = detected / total if total > 0 else 0.0
        print(f"  >= {cov*100:4.0f}% pixels vacant  "
              f"{detected:>8,}  {total:>7,}  {recall:>8.1%}")

    # Additional stats
    nz = evaluable["pred_fraction"].dropna()
    print(f"\nPrediction fraction across evaluable vacant parcels:")
    print(f"  mean  : {nz.mean():.3f}")
    print(f"  median: {nz.median():.3f}")
    print(f"  >=10% : {(nz >= 0.10).mean():.1%}")
    print(f"  >=20% : {(nz >= 0.20).mean():.1%}")

    # BoroCode breakdown if available
    boro_col = next((c for c in evaluable.columns
                     if c.lower() in ("borocode", "boro_code", "boro")), None)
    if boro_col:
        boro_names = {1: "Manhattan", 2: "Bronx", 3: "Brooklyn",
                      4: "Queens", 5: "Staten Island"}
        print(f"\nRecall by borough (coverage >= {args.coverage[0]*100:.0f}%):")
        cov0 = args.coverage[0]
        for bc, grp in evaluable.groupby(evaluable[boro_col].astype(int)):
            det = int((grp["pred_fraction"] >= cov0).sum())
            print(f"  {boro_names.get(bc, bc):<14s}: {det:>5,} / {len(grp):>5,}  "
                  f"({det/len(grp):.1%})")

    # Save CSV
    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = SHARED_ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cols = [id_col, boro_col, "valid_pixels", "pred_pixels", "pred_fraction"]
        cols = [c for c in cols if c and c in evaluable.columns]
        # add BldgClass if present
        if "BldgClass" in evaluable.columns:
            cols = [id_col, "BldgClass"] + [c for c in cols if c != id_col]
        evaluable[cols].to_csv(out_path, index=False)
        print(f"\nPer-parcel results saved → {out_path}")

    # Figures
    if not args.no_figures:
        if args.fig_dir:
            fig_dir = Path(args.fig_dir)
            if not fig_dir.is_absolute():
                fig_dir = SHARED_ROOT / fig_dir
        else:
            fig_dir = run_dir / "figures"
        fracs = evaluable["pred_fraction"].dropna().values * 100
        marks = sorted(args.coverage)
        print(f"\nGenerating figures in {fig_dir} ...")
        plot_recall_vs_coverage(fracs / 100, args.split, marks, fig_dir)
        plot_pred_fraction_hist(fracs, args.split, marks, fig_dir)


if __name__ == "__main__":
    main()
