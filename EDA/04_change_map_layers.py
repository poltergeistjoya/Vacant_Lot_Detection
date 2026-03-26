"""
04_change_map_layers.py — Generate QGIS-ready layers for NYC vacant lot change (2019→2022).

Outputs:
    - NAIP 2019 VRT (downloaded if not present)
    - GeoPackage with layers: vacant_2019, vacant_2022, change, hotspots
    - stats.csv with per-borough change statistics
    - Quick matplotlib preview PNG

Usage:
    uv run python EDA/04_change_map_layers.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Ensure package is importable when run from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vacant_lot.config import load_config, _get_shared_root
from vacant_lot.change_detection import (
    load_vacant_parcels,
    compute_change,
    find_hotspots,
    compute_stats,
)
from vacant_lot.tile_export import query_naip_stac, download_naip_tiles, build_naip_vrt
from vacant_lot.logger import get_logger

log = get_logger()

# ── Config ──────────────────────────────────────────────────────────────────

CFG_2019 = load_config("nyc_change_2019.yaml")
CFG_2022 = load_config("nyc_buildings.yaml")

OUTPUT_DIR = _get_shared_root() / "outputs" / "final" / "nyc_change_2019_2022"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
GPKG_PATH = OUTPUT_DIR / "nyc_change_layers.gpkg"

CHANGE_COLORS = {
    "remained_vacant": "#FFC832",
    "became_vacant": "#DC3232",
    "developed": "#32B464",
}

# ── Step 1: Download NAIP 2019 ─────────────────────────────────────────────


def download_naip_2019():
    """Download NAIP 2019 tiles and build VRT (idempotent)."""
    tiles_dir = CFG_2019.get_naip_tiles_dir()
    vrt_path = CFG_2019.get_naip_vrt_path()

    if vrt_path.exists():
        log.info(f"NAIP 2019 VRT already exists: {vrt_path}")
        return vrt_path

    tiles_dir.mkdir(parents=True, exist_ok=True)

    bbox = CFG_2019.segmentation.bbox
    log.info(f"Querying NAIP STAC for bbox={bbox}, year=2019")
    items = query_naip_stac(bbox=bbox, year=2019)
    log.info(f"Found {len(items)} NAIP 2019 tiles")

    download_naip_tiles(items, local_dir=tiles_dir)

    vrt_path = build_naip_vrt(tiles_dir=tiles_dir, output_vrt=vrt_path)
    log.info(f"Built NAIP 2019 VRT: {vrt_path}")
    return vrt_path


# ── Step 2: Load parcels and compute change ────────────────────────────────


def load_and_compute():
    """Load both MapPLUTO vintages, compute change categories."""
    log.info("Loading 2019 parcels...")
    all_19, vacant_19 = load_vacant_parcels(CFG_2019)
    log.info("Loading 2022 parcels...")
    all_22, vacant_22 = load_vacant_parcels(CFG_2022)

    log.info("Computing change categories...")
    change_gdf = compute_change(all_19, all_22)

    return all_19, all_22, vacant_19, vacant_22, change_gdf


# ── Step 3: Export layers ──────────────────────────────────────────────────


def export_layers(vacant_19, vacant_22, change_gdf, hotspots):
    """Write all layers to a single GeoPackage."""
    log.info(f"Exporting layers to {GPKG_PATH}")

    # Keep useful columns for QGIS styling
    v19_cols = [c for c in ["BBL", "BldgClass", "LandUse", "Borough", "Address", "LotArea", "geometry"]
                if c in vacant_19.columns]
    v22_cols = [c for c in ["BBL", "BldgClass", "LandUse", "Borough", "Address", "LotArea", "geometry"]
                if c in vacant_22.columns]

    vacant_19[v19_cols].to_file(GPKG_PATH, layer="vacant_2019", driver="GPKG")
    vacant_22[v22_cols].to_file(GPKG_PATH, layer="vacant_2022", driver="GPKG")
    change_gdf.to_file(GPKG_PATH, layer="change", driver="GPKG")
    hotspots.to_file(GPKG_PATH, layer="hotspots", driver="GPKG")

    log.info(f"GeoPackage written: {GPKG_PATH}")


# ── Step 4: Statistics ──────────────────────────────────────────────────────


def print_and_save_stats(change_gdf):
    """Compute, print, and save change statistics."""
    stats = compute_stats(change_gdf)
    stats_path = OUTPUT_DIR / "stats.csv"
    stats.to_csv(stats_path, index=False)
    log.info(f"Stats saved to {stats_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("NYC VACANT LOT CHANGE STATISTICS (2019 → 2022)")
    print("=" * 70)

    city_wide = stats[stats["Borough"] == "ALL"]
    for _, row in city_wide.iterrows():
        area_acres = row["total_area_m2"] / 4046.86
        print(f"  {row['change_category']:20s}  {row['count']:6,d} lots  {area_acres:10,.1f} acres")

    print(f"\n{'Borough':<15} {'Remained':>10} {'Became':>10} {'Developed':>10}")
    print("-" * 50)
    boroughs = stats[stats["Borough"] != "ALL"]
    pivot = boroughs.pivot_table(
        index="Borough", columns="change_category", values="count", fill_value=0
    )
    for boro in sorted(pivot.index):
        row = pivot.loc[boro]
        print(f"{boro:<15} {int(row.get('remained_vacant', 0)):>10,d} "
              f"{int(row.get('became_vacant', 0)):>10,d} {int(row.get('developed', 0)):>10,d}")
    print("=" * 70)

    return stats


# ── Step 5: Quick preview ──────────────────────────────────────────────────


def preview_map(change_gdf, hotspots):
    """Save a low-res matplotlib preview of the change map."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 14), facecolor="#1A1A2E")
    ax.set_facecolor("#1A1A2E")

    for cat, color in CHANGE_COLORS.items():
        subset = change_gdf[change_gdf["change_category"] == cat]
        if not subset.empty:
            subset.plot(ax=ax, color=color, markersize=0.5, linewidth=0.1)

    # Hotspot rectangles
    if not hotspots.empty:
        hotspots.boundary.plot(ax=ax, edgecolor="white", linewidth=1.0, linestyle="--")

    # Legend
    patches = [mpatches.Patch(color=c, label=k.replace("_", " ").title())
               for k, c in CHANGE_COLORS.items()]
    patches.append(mpatches.Patch(edgecolor="white", facecolor="none",
                                  linestyle="--", label="Change hotspot"))
    ax.legend(handles=patches, loc="lower left", fontsize=9,
              facecolor="#2A2A4E", edgecolor="white", labelcolor="white")

    ax.set_title("NYC Vacant Lot Change 2019–2022", color="white", fontsize=14, pad=10)
    ax.set_axis_off()

    preview_path = OUTPUT_DIR / "preview.png"
    fig.savefig(preview_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info(f"Preview saved to {preview_path}")


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    # Step 1: NAIP 2019
    download_naip_2019()

    # Step 2: Load and compute change
    all_19, all_22, vacant_19, vacant_22, change_gdf = load_and_compute()

    # Step 3: Hotspots
    log.info("Finding change hotspots...")
    hotspots = find_hotspots(change_gdf, cell_size=500, top_n=10)

    # Step 4: Export
    export_layers(vacant_19, vacant_22, change_gdf, hotspots)

    # Step 5: Stats
    print_and_save_stats(change_gdf)

    # Step 6: Preview
    preview_map(change_gdf, hotspots)

    print(f"\nAll outputs in: {OUTPUT_DIR}")
    print(f"GeoPackage:     {GPKG_PATH}")
    print(f"NAIP 2019 VRT:  {CFG_2019.get_naip_vrt_path()}")
    print(f"NAIP 2022 VRT:  {CFG_2022.get_naip_vrt_path()}")


if __name__ == "__main__":
    main()
