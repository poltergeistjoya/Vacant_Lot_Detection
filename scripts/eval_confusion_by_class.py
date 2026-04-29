"""
BldgClass / LandUse confusion analysis for parcel-level predictions.

For each parcel in MapPLUTO (all land-use types), computes the fraction of
pixels predicted vacant and classifies as TP/FP/FN/TN at multiple coverage
thresholds. Reports:

  (a) Overall confusion matrix + precision / recall / F2
  (b) False-positive breakdown by LandUse
  (c) False-positive breakdown by BldgClass prefix (top 15)
  (d) False-negative breakdown by BldgClass (missed vacant subtypes)
  (e) True-positive rate by BldgClass (reliably detected vacant subtypes)

Also saves two figures into --fig-dir (default: {run}/figures/):
  confusion_fp_by_landuse_{split}.png   — FP count & FP rate by LandUse
  confusion_vacant_by_bldgclass_{split}.png — TP rate + FN count by BldgClass (vacant only)

Config is read from the run's config.yaml (not global data.yaml) to pick up
the correct patch_splits and vacancy_mask for that run.

Usage:
    uv run python scripts/eval_confusion_by_class.py \\
        --run outputs/models/deeplabv3plus/kahan_027 \\
        --split val \\
        --out outputs/models/deeplabv3plus/kahan_027/confusion_by_class_brooklyn.csv

    # Faster FP-only pass (figures skipped for vacant-side panels)
    uv run python scripts/eval_confusion_by_class.py \\
        --run outputs/models/deeplabv3plus/kahan_027 \\
        --split val \\
        --nonvacant-only

    # Choose which coverage threshold to use for figures (default: 0.2)
    uv run python scripts/eval_confusion_by_class.py \\
        --run outputs/models/deeplabv3plus/kahan_027 \\
        --split val \\
        --plot-coverage 0.3
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import rasterio.mask
import yaml
from shapely.geometry import box

mpl.rcParams.update({
    "font.family":      "STIX Two Text",
    "mathtext.fontset": "stix",
    "font.size":        8,
    "axes.titleweight": "normal",
    "axes.labelweight": "normal",
})

SCRIPT_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Shared-root resolution (same pattern as eval_parcel_level.py)
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

DATA_YAML = SCRIPT_DIR.parent / "config" / "data.yaml"

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

BLDGCLASS_PREFIX: dict[str, str] = {
    "A": "Townhouses",
    "B": "Brownstones",
    "C": "Walk-up Apts",
    "D": "Elevator Apts",
    "E": "Warehouses",
    "F": "Factories",
    "G": "Garages/Parking",
    "H": "Hotels",
    "I": "Hospitals",
    "J": "Churches",
    "K": "Stores",
    "L": "Lofts",
    "M": "Religious",
    "O": "Offices",
    "P": "Assembly",
    "Q": "Recreation/Parks",
    "R": "Condos",
    "S": "Mixed Res.",
    "T": "Transportation",
    "U": "Utilities",
    "V": "Vacant",
    "W": "Schools",
    "Y": "Government",
    "Z": "Miscellaneous",
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
    return d.get("split", {})   # train_boroughs, val_boroughs, test_boroughs


def using_v2_mask(run_cfg: dict) -> bool:
    return "v2" in run_cfg["data_paths"].get("vacancy_mask", "")


# ---------------------------------------------------------------------------
# Reused from eval_parcel_level.py
# ---------------------------------------------------------------------------

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
    thr = f2_optimal_threshold(prec, rec, thresh)
    print(f"F2-optimal threshold from {split} PR curve: {thr:.4f}")
    return thr


def find_prob_tif(run_dir: Path, split: str) -> Path:
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
# All-parcel loader (extends load_vacant_parcels to all land-use types)
# ---------------------------------------------------------------------------

def load_all_parcels(data_cfg: dict, apply_v2_corrections: bool,
                     vacant_only: bool = False,
                     nonvacant_only: bool = False) -> gpd.GeoDataFrame:
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

    # Optional subset
    if vacant_only:
        gdf = gdf[gdf["is_vacant"]].copy()
    elif nonvacant_only:
        gdf = gdf[~gdf["is_vacant"]].copy()

    print(f"  Parcels loaded: {len(gdf):,}  "
          f"(vacant: {gdf['is_vacant'].sum():,}, "
          f"non-vacant: {(~gdf['is_vacant']).sum():,})")
    return gdf


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def _pct(n: int, d: int) -> str:
    return f"{n/d:.1%}" if d else "N/A"


def print_confusion_matrix(tp: int, fp: int, fn: int, tn: int,
                            cov: float) -> None:
    total    = tp + fp + fn + tn
    prec     = tp / (tp + fp) if (tp + fp) else 0.0
    rec      = tp / (tp + fn) if (tp + fn) else 0.0
    f2       = 5 * prec * rec / (4 * prec + rec) if (prec + rec) else 0.0
    print(f"\n=== Coverage threshold: {cov*100:.0f}% ===")
    print(f"  Confusion matrix ({total:,} evaluable parcels):")
    print(f"    TP={tp:,}  FP={fp:,}  FN={fn:,}  TN={tn:,}")
    print(f"  Parcel precision : {prec:.3f}")
    print(f"  Parcel recall    : {rec:.3f}")
    print(f"  Parcel F2        : {f2:.3f}")


def print_fp_by_landuse(rows: list[dict], fp_total: int, cov_key: str) -> None:
    from collections import defaultdict
    counts: dict[str, dict] = defaultdict(lambda: {"fp": 0, "total": 0})
    for r in rows:
        lu = str(r.get("LandUse", "") or "").zfill(2)
        label = r[cov_key]
        counts[lu]["total"] += 1
        if label == "FP":
            counts[lu]["fp"] += 1

    rows_out = []
    for lu, c in counts.items():
        if c["fp"] == 0:
            continue
        rows_out.append((lu, c["fp"], c["total"]))
    rows_out.sort(key=lambda x: -x[1])

    print(f"\n  FP breakdown by LandUse:")
    print(f"  {'LandUse':>7s}  {'Label':<28s}  {'FP count':>8s}  {'% of FPs':>9s}  {'FP rate':>8s}")
    print(f"  {'-'*7}  {'-'*28}  {'-'*8}  {'-'*9}  {'-'*8}")
    for lu, fp_cnt, total in rows_out:
        label = LANDUSE_LABELS.get(lu, "Unknown")
        pct_of_fps = f"{fp_cnt/fp_total:.1%}" if fp_total else "N/A"
        fp_rate    = f"{fp_cnt/total:.1%}" if total else "N/A"
        print(f"  {lu:>7s}  {label:<28s}  {fp_cnt:>8,}  {pct_of_fps:>9s}  {fp_rate:>8s}")


def print_fp_by_bldgclass_prefix(rows: list[dict], fp_total: int, cov_key: str,
                                  top_n: int = 15) -> None:
    from collections import defaultdict
    counts: dict[str, int] = defaultdict(int)
    for r in rows:
        if r[cov_key] == "FP":
            prefix = str(r.get("BldgClass", "") or "")[:1].upper()
            counts[prefix] += 1

    top = sorted(counts.items(), key=lambda x: -x[1])[:top_n]
    print(f"\n  FP breakdown by BldgClass prefix (top {top_n}):")
    print(f"  {'Prefix':>6s}  {'Category':<24s}  {'FP count':>8s}  {'% of FPs':>9s}")
    print(f"  {'-'*6}  {'-'*24}  {'-'*8}  {'-'*9}")
    for prefix, cnt in top:
        cat = BLDGCLASS_PREFIX.get(prefix, "Unknown")
        pct = f"{cnt/fp_total:.1%}" if fp_total else "N/A"
        print(f"  {prefix:>6s}  {cat:<24s}  {cnt:>8,}  {pct:>9s}")


def print_fn_by_bldgclass(rows: list[dict], fn_total: int, cov_key: str) -> None:
    from collections import defaultdict
    counts: dict[str, int] = defaultdict(int)
    for r in rows:
        if r[cov_key] == "FN":
            bc = str(r.get("BldgClass", "") or "")[:2]
            counts[bc] += 1

    top = sorted(counts.items(), key=lambda x: -x[1])[:15]
    print(f"\n  FN breakdown by BldgClass (missed vacant subtypes, top 15):")
    print(f"  {'BldgClass':>9s}  {'FN count':>8s}  {'% of FNs':>9s}")
    print(f"  {'-'*9}  {'-'*8}  {'-'*9}")
    for bc, cnt in top:
        pct = f"{cnt/fn_total:.1%}" if fn_total else "N/A"
        print(f"  {bc:>9s}  {cnt:>8,}  {pct:>9s}")


def print_fn_by_landuse(rows: list[dict], fn_total: int, cov_key: str) -> None:
    from collections import defaultdict
    counts: dict[str, dict] = defaultdict(lambda: {"fn": 0, "total_vacant": 0})
    for r in rows:
        if r["is_vacant"]:
            lu = str(r.get("LandUse", "") or "").zfill(2)
            counts[lu]["total_vacant"] += 1
            if r[cov_key] == "FN":
                counts[lu]["fn"] += 1

    rows_out = []
    for lu, c in counts.items():
        if c["fn"] == 0:
            continue
        rows_out.append((lu, c["fn"], c["total_vacant"]))
    rows_out.sort(key=lambda x: -x[1])

    print(f"\n  FN breakdown by LandUse:")
    print(f"  {'LandUse':>7s}  {'Label':<28s}  {'FN count':>8s}  {'% of FNs':>9s}  {'FN rate':>8s}")
    print(f"  {'-'*7}  {'-'*28}  {'-'*8}  {'-'*9}  {'-'*8}")
    for lu, fn_cnt, total_vacant in rows_out:
        label = LANDUSE_LABELS.get(lu, "Unknown")
        pct_of_fns = f"{fn_cnt/fn_total:.1%}" if fn_total else "N/A"
        fn_rate    = f"{fn_cnt/total_vacant:.1%}" if total_vacant else "N/A"
        print(f"  {lu:>7s}  {label:<28s}  {fn_cnt:>8,}  {pct_of_fns:>9s}  {fn_rate:>8s}")


def print_tp_rate_by_bldgclass(rows: list[dict], cov_key: str) -> None:
    from collections import defaultdict
    counts: dict[str, dict] = defaultdict(lambda: {"tp": 0, "total": 0})
    for r in rows:
        if r["is_vacant"]:
            bc = str(r.get("BldgClass", "") or "")[:2]
            counts[bc]["total"] += 1
            if r[cov_key] == "TP":
                counts[bc]["tp"] += 1

    rows_out = [(bc, c["tp"], c["total"]) for bc, c in counts.items() if c["total"] >= 3]
    rows_out.sort(key=lambda x: -(x[1] / x[2]))

    print(f"\n  TP rate by BldgClass (vacant subtypes reliably detected):")
    print(f"  {'BldgClass':>9s}  {'TP':>6s}  {'Total':>7s}  {'TP rate':>8s}")
    print(f"  {'-'*9}  {'-'*6}  {'-'*7}  {'-'*8}")
    for bc, tp, total in rows_out[:15]:
        print(f"  {bc:>9s}  {tp:>6,}  {total:>7,}  {tp/total:>8.1%}")


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

def _apply_style(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.4)
    ax.spines["bottom"].set_linewidth(0.4)
    ax.tick_params(labelsize=7, length=2, width=0.4)
    ax.grid(axis="x", linestyle=":", linewidth=0.3, color="#cccccc")
    ax.set_axisbelow(True)


def plot_fp_by_landuse(result_rows: list[dict], cov_key: str, cov: float,
                        split: str, fig_dir: Path, run_id: str = "",
                        borough_label: str = "") -> None:
    """Horizontal bar chart: FP count and FP rate per LandUse category."""
    from collections import defaultdict
    counts: dict[str, dict] = defaultdict(lambda: {"fp": 0, "total": 0})
    for r in result_rows:
        lu = str(r.get("LandUse", "") or "").zfill(2)
        counts[lu]["total"] += 1
        if r[cov_key] == "FP":
            counts[lu]["fp"] += 1

    # Build sorted rows (by FP count desc), skip LU 11 (truly vacant)
    entries = [
        (lu, c["fp"], c["total"])
        for lu, c in counts.items()
        if c["fp"] > 0
    ]
    entries.sort(key=lambda x: -x[1])

    if not entries:
        print(f"[warn] No FP parcels at coverage {cov*100:.0f}%; skipping FP-by-LandUse figure",
              file=sys.stderr)
        return

    labels = [
        f"{LANDUSE_LABELS.get(lu, 'Unknown')} ({lu})  n={total:,}"
        for lu, _, total in entries
    ]
    fp_counts  = [e[1] for e in entries]
    fp_rates   = [e[1] / e[2] if e[2] else 0.0 for e in entries]

    fig, axes = plt.subplots(1, 2, figsize=(9, max(3, 0.35 * len(entries) + 1)),
                              constrained_layout=True)

    sup_parts = [p for p in [run_id, f"{split}: {borough_label}" if borough_label else split] if p]
    fig.suptitle("  ·  ".join(sup_parts), fontsize=7, color="#555555")

    # Left: FP count
    ax = axes[0]
    y = np.arange(len(entries))
    ax.barh(y, fp_counts, color="#d6604d", height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("False positive parcels", fontsize=8)
    ax.set_title(f"FP count by land use  (coverage ≥ {cov*100:.0f}%)", fontsize=8)
    _apply_style(ax)

    # Right: FP rate within class
    ax2 = axes[1]
    ax2.barh(y, [r * 100 for r in fp_rates], color="#4393c3", height=0.6)
    ax2.set_yticks(y)
    ax2.set_yticklabels(labels, fontsize=7)
    ax2.invert_yaxis()
    ax2.set_xlabel("FP rate within class (%)", fontsize=8)
    ax2.set_title("FP rate by land use", fontsize=8)
    _apply_style(ax2)

    fig_dir.mkdir(parents=True, exist_ok=True)
    run_suffix = f"_{run_id}" if run_id else ""
    out = fig_dir / f"confusion_fp_by_landuse{run_suffix}_{split}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


def plot_fn_by_landuse(result_rows: list[dict], cov_key: str, cov: float,
                        split: str, fig_dir: Path, run_id: str = "",
                        borough_label: str = "") -> None:
    """Horizontal bar chart: FN count and FN rate per LandUse category (vacant land only)."""
    from collections import defaultdict
    counts: dict[str, dict] = defaultdict(lambda: {"fn": 0, "total_vacant": 0})
    for r in result_rows:
        if not r["is_vacant"]:
            continue
        lu = str(r.get("LandUse", "") or "").zfill(2)
        counts[lu]["total_vacant"] += 1
        if r[cov_key] == "FN":
            counts[lu]["fn"] += 1

    entries = [
        (lu, c["fn"], c["total_vacant"])
        for lu, c in counts.items()
        if c["fn"] > 0
    ]
    entries.sort(key=lambda x: -x[1])

    if not entries:
        print(f"[warn] No FN parcels at coverage {cov*100:.0f}%; skipping FN-by-LandUse figure",
              file=sys.stderr)
        return

    labels = [
        f"{LANDUSE_LABELS.get(lu, 'Unknown')} ({lu})  n={total:,}"
        for lu, _, total in entries
    ]
    fn_counts  = [e[1] for e in entries]
    fn_rates   = [e[1] / e[2] if e[2] else 0.0 for e in entries]

    fig, axes = plt.subplots(1, 2, figsize=(9, max(3, 0.35 * len(entries) + 1)),
                              constrained_layout=True)

    sup_parts = [p for p in [run_id, f"{split}: {borough_label}" if borough_label else split] if p]
    fig.suptitle("  ·  ".join(sup_parts), fontsize=7, color="#555555")

    # Left: FN count
    ax = axes[0]
    y = np.arange(len(entries))
    ax.barh(y, fn_counts, color="#d6604d", height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("False negative parcels (missed)", fontsize=8)
    ax.set_title(f"FN count by land use  (coverage ≥ {cov*100:.0f}%)", fontsize=8)
    _apply_style(ax)

    # Right: FN rate within vacant class
    ax2 = axes[1]
    ax2.barh(y, [r * 100 for r in fn_rates], color="#4393c3", height=0.6)
    ax2.set_yticks(y)
    ax2.set_yticklabels(labels, fontsize=7)
    ax2.invert_yaxis()
    ax2.set_xlabel("FN rate within vacant class (%)", fontsize=8)
    ax2.set_title("FN rate by land use", fontsize=8)
    _apply_style(ax2)

    fig_dir.mkdir(parents=True, exist_ok=True)
    run_suffix = f"_{run_id}" if run_id else ""
    out = fig_dir / f"confusion_fn_by_landuse{run_suffix}_{split}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


def plot_vacant_by_bldgclass(result_rows: list[dict], cov_key: str, cov: float,
                              split: str, fig_dir: Path,
                              min_parcels: int = 3, run_id: str = "",
                              borough_label: str = "") -> None:
    """Two-panel figure for vacant parcels: TP rate and FN count by BldgClass."""
    from collections import defaultdict
    counts: dict[str, dict] = defaultdict(lambda: {"tp": 0, "fn": 0})
    for r in result_rows:
        if not r["is_vacant"]:
            continue
        bc = str(r.get("BldgClass", "") or "")[:2]
        if r[cov_key] == "TP":
            counts[bc]["tp"] += 1
        elif r[cov_key] == "FN":
            counts[bc]["fn"] += 1

    entries = [
        (bc, c["tp"], c["fn"])
        for bc, c in counts.items()
        if (c["tp"] + c["fn"]) >= min_parcels
    ]
    if not entries:
        print("[warn] Not enough vacant parcels to plot BldgClass figure",
              file=sys.stderr)
        return

    # Sort by TP rate descending
    entries.sort(key=lambda x: -(x[1] / (x[1] + x[2])))

    labels    = [f"{e[0]}  n={e[1]+e[2]:,}" for e in entries]
    tp_rates  = [e[1] / (e[1] + e[2]) * 100 for e in entries]
    fn_counts = [e[2] for e in entries]

    fig, axes = plt.subplots(1, 2, figsize=(9, max(3, 0.35 * len(entries) + 1)),
                              constrained_layout=True)

    sup_parts = [p for p in [run_id, f"{split}: {borough_label}" if borough_label else split] if p]
    fig.suptitle("  ·  ".join(sup_parts), fontsize=7, color="#555555")

    y = np.arange(len(entries))

    # Left: TP rate
    ax = axes[0]
    ax.barh(y, tp_rates, color="#1b7837", height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Detection rate (%)", fontsize=8)
    ax.set_title(f"TP rate by BldgClass  (coverage ≥ {cov*100:.0f}%)", fontsize=8)
    ax.set_xlim(0, 100)
    _apply_style(ax)

    # Right: FN count
    ax2 = axes[1]
    ax2.barh(y, fn_counts, color="#d6604d", height=0.6)
    ax2.set_yticks(y)
    ax2.set_yticklabels(labels, fontsize=7)
    ax2.invert_yaxis()
    ax2.set_xlabel("Missed parcels (FN)", fontsize=8)
    ax2.set_title("FN count by BldgClass", fontsize=8)
    _apply_style(ax2)

    fig_dir.mkdir(parents=True, exist_ok=True)
    run_suffix = f"_{run_id}" if run_id else ""
    out = fig_dir / f"confusion_vacant_by_bldgclass{run_suffix}_{split}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="BldgClass/LandUse confusion analysis for parcel-level predictions"
    )
    parser.add_argument("--run", required=True,
                        help="Run directory (relative to shared root or absolute)")
    parser.add_argument("--split", default="val",
                        help="Split to evaluate: val or test")
    parser.add_argument("--pixel-threshold", type=float, default=None,
                        help="Pixel threshold (default: F2-optimal from pr_curves.npz)")
    parser.add_argument("--coverage", nargs="+", type=float,
                        default=[0.1, 0.2, 0.3, 0.5],
                        help="Parcel coverage thresholds (default: 0.1 0.2 0.3 0.5)")
    parser.add_argument("--min-parcel-pixels", type=int, default=5,
                        help="Minimum valid pixels to include a parcel (default: 5)")
    parser.add_argument("--vacant-only", action="store_true",
                        help="Only process vacant parcels (faster FN/TP analysis)")
    parser.add_argument("--nonvacant-only", action="store_true",
                        help="Only process non-vacant parcels (faster FP analysis)")
    parser.add_argument("--out", default=None,
                        help="Save per-parcel CSV to this path")
    parser.add_argument("--fig-dir", default=None,
                        help="Directory for output figures "
                             "(default: {run}/figures/)")
    parser.add_argument("--plot-coverage", type=float, default=0.2,
                        help="Coverage threshold to use for figures (default: 0.2)")
    parser.add_argument("--no-figures", action="store_true",
                        help="Skip figure generation")
    args = parser.parse_args()

    if args.vacant_only and args.nonvacant_only:
        sys.exit("Cannot use --vacant-only and --nonvacant-only together")

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

    print(f"Shared root : {SHARED_ROOT}")
    print(f"Run dir     : {run_dir}")
    print(f"Run ID      : {run_id}")
    print(f"Split       : {args.split}")

    # Load configs
    data_cfg = load_data_config()
    run_cfg  = load_run_config(run_dir)
    apply_v2 = using_v2_mask(run_cfg)
    print(f"V2 mask     : {apply_v2}")

    # Determine borough filter from run's patch_splits
    borough_assignments = get_borough_assignments(run_cfg)
    key = f"{args.split}_boroughs"
    split_boroughs: list[int] = borough_assignments.get(key, [])
    boro_names = {1: "Manhattan", 2: "Bronx", 3: "Brooklyn",
                  4: "Queens", 5: "Staten Island"}
    if not split_boroughs:
        print(f"[warn] No borough assignment found for key '{key}' in patch_splits; "
              f"no borough filter applied", file=sys.stderr)
        borough_label = ""
    else:
        names = [boro_names.get(b, str(b)) for b in split_boroughs]
        borough_label = ", ".join(names)
        print(f"Boroughs    : {split_boroughs} ({borough_label})")

    # Pixel threshold and prob TIF
    pixel_thr = get_pixel_threshold(run_dir, args.split, args.pixel_threshold)
    prob_tif  = find_prob_tif(run_dir, args.split)
    print(f"Prob TIF    : {prob_tif.name}")
    print(f"Pixel thr   : {pixel_thr:.4f}")
    print(f"Coverage    : {[f'{c*100:.0f}%' for c in sorted(args.coverage)]}")

    # Load parcels
    gdf = load_all_parcels(data_cfg, apply_v2_corrections=apply_v2,
                           vacant_only=args.vacant_only,
                           nonvacant_only=args.nonvacant_only)

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

    # Build per-row result dicts for reporting
    thresholds = sorted(args.coverage)
    cov_keys   = [f"label_{int(c*100)}" for c in thresholds]

    result_rows: list[dict] = []
    for _, row in evaluable.iterrows():
        pf       = row["pred_fraction"]
        is_vac   = bool(row["is_vacant"])
        bc       = str(row.get("BldgClass", "") or "")
        lu       = str(row.get("LandUse", "") or "").zfill(2)
        bbl      = row.get(id_col, "")
        boro_col_name = next((c for c in evaluable.columns
                              if c.lower() in ("borocode", "boro_code", "boro")), None)
        boro = int(row[boro_col_name]) if boro_col_name else None

        rec: dict = {
            "BBL": bbl,
            "BldgClass": bc,
            "LandUse": lu,
            "BoroCode": boro,
            "is_vacant": is_vac,
            "valid_pixels": int(row["valid_pixels"]),
            "pred_fraction": float(pf) if not np.isnan(pf) else None,
        }
        for cov, key in zip(thresholds, cov_keys):
            if np.isnan(pf):
                rec[key] = "NA"
            elif is_vac:
                rec[key] = "TP" if pf >= cov else "FN"
            else:
                rec[key] = "FP" if pf >= cov else "TN"
        result_rows.append(rec)

    # -------------------------------------------------------------------------
    # Print analysis at each coverage threshold
    # -------------------------------------------------------------------------
    for cov, key in zip(thresholds, cov_keys):
        if args.vacant_only:
            tp = sum(1 for r in result_rows if r[key] == "TP")
            fn = sum(1 for r in result_rows if r[key] == "FN")
            fp = tn = 0
        elif args.nonvacant_only:
            fp = sum(1 for r in result_rows if r[key] == "FP")
            tn = sum(1 for r in result_rows if r[key] == "TN")
            tp = fn = 0
        else:
            tp = sum(1 for r in result_rows if r[key] == "TP")
            fp = sum(1 for r in result_rows if r[key] == "FP")
            fn = sum(1 for r in result_rows if r[key] == "FN")
            tn = sum(1 for r in result_rows if r[key] == "TN")

        print_confusion_matrix(tp, fp, fn, tn, cov)

        if not args.vacant_only:
            print_fp_by_landuse(result_rows, fp, key)
            print_fp_by_bldgclass_prefix(result_rows, fp, key)

        if not args.nonvacant_only:
            print_fn_by_bldgclass(result_rows, fn, key)
            print_fn_by_landuse(result_rows, fn, key)
            print_tp_rate_by_bldgclass(result_rows, key)

    # -------------------------------------------------------------------------
    # Save CSV
    # -------------------------------------------------------------------------
    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = SHARED_ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = (
            ["BBL", "BldgClass", "LandUse", "BoroCode",
             "is_vacant", "valid_pixels", "pred_fraction"]
            + cov_keys
        )
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames,
                                    extrasaction="ignore")
            writer.writeheader()
            writer.writerows(result_rows)
        print(f"\nPer-parcel results saved → {out_path}")
        print(f"  Rows: {len(result_rows):,}")

    # -------------------------------------------------------------------------
    # Figures
    # -------------------------------------------------------------------------
    if not args.no_figures:
        # Resolve coverage for figures — fall back to nearest available threshold
        plot_cov = args.plot_coverage
        if plot_cov not in thresholds:
            nearest = min(thresholds, key=lambda c: abs(c - plot_cov))
            print(f"[warn] --plot-coverage {plot_cov} not in evaluated thresholds; "
                  f"using {nearest}", file=sys.stderr)
            plot_cov = nearest
        plot_key = f"label_{int(plot_cov * 100)}"

        if args.fig_dir:
            fig_dir = Path(args.fig_dir)
            if not fig_dir.is_absolute():
                fig_dir = SHARED_ROOT / fig_dir
        else:
            fig_dir = run_dir / "figures"
        print(f"\nGenerating figures in {fig_dir} ...")

        if not args.vacant_only:
            plot_fp_by_landuse(result_rows, plot_key, plot_cov,
                               args.split, fig_dir, run_id,
                               borough_label=borough_label)
        if not args.nonvacant_only:
            plot_fn_by_landuse(result_rows, plot_key, plot_cov,
                               args.split, fig_dir, run_id,
                               borough_label=borough_label)
            plot_vacant_by_bldgclass(result_rows, plot_key, plot_cov,
                                     args.split, fig_dir, run_id=run_id,
                                     borough_label=borough_label)


if __name__ == "__main__":
    main()
