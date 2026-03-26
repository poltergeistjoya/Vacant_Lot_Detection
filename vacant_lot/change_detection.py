"""
change_detection.py — Temporal comparison of vacant lots between MapPLUTO vintages.

Joins two MapPLUTO datasets by BBL to identify parcels that:
- Remained vacant (both years)
- Became vacant (new in later year)
- Were developed (vacant in earlier year, not in later)

Also provides hotspot detection and per-borough statistics.
"""

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

from .config import CityConfig
from .data_utils import load_gdb
from .logger import get_logger

log = get_logger()

VACANT_CODES = ["G7", "V0", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9"]

KEEP_COLS = ["BBL", "Borough", "BldgClass", "LandUse", "Address", "LotArea", "BoroCode"]


def load_vacant_parcels(config: CityConfig) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Load all parcels from a MapPLUTO config and split into vacant / full.

    Returns:
        (all_parcels, vacant_parcels) both in config.raster.output_crs.
        all_parcels has an `is_vacant` boolean column.
    """
    path = config.get_parcel_path()
    layer = config.parcel.layer
    log.info(f"Loading parcels from {path} layer={layer}")

    gdf = load_gdb(gdb_path=str(path), layer=layer)
    gdf = gdf.to_crs(config.raster.output_crs)

    # Tag vacancy using BldgClass codes
    codes = config.parcel.vacant_codes
    gdf["is_vacant"] = gdf[config.parcel.landuse_column].isin(codes)

    log.info(f"Loaded {len(gdf)} parcels, {gdf['is_vacant'].sum()} vacant")

    vacant = gdf[gdf["is_vacant"]].copy()
    return gdf, vacant


def compute_change(
    gdf19: gpd.GeoDataFrame,
    gdf22: gpd.GeoDataFrame,
    vacant_codes: list[str] | None = None,
) -> gpd.GeoDataFrame:
    """Join two MapPLUTO vintages by BBL and assign change categories.

    Args:
        gdf19: All parcels from 2019 (must have `is_vacant` column).
        gdf22: All parcels from 2022 (must have `is_vacant` column).
        vacant_codes: Not used (vacancy already tagged via is_vacant). Kept for API compat.

    Returns:
        GeoDataFrame of changed parcels with columns:
        BBL, change_category, BldgClass_2019, BldgClass_2022, Borough,
        LotArea, area_m2, geometry.
    """
    # Prepare join keys — BBL as integer string
    def _bbl_key(gdf):
        return gdf["BBL"].dropna().astype(np.int64).astype(str)

    df19 = gdf19[["BBL", "BldgClass", "LandUse", "Borough", "LotArea", "is_vacant", "geometry"]].copy()
    df22 = gdf22[["BBL", "BldgClass", "LandUse", "Borough", "LotArea", "is_vacant", "geometry"]].copy()

    # Rename geometry to plain columns so merge produces proper suffixes
    df19["geom_2019"] = df19.geometry
    df22["geom_2022"] = df22.geometry
    df19 = pd.DataFrame(df19.drop(columns=["geometry"]))
    df22 = pd.DataFrame(df22.drop(columns=["geometry"]))

    df19["bbl_key"] = _bbl_key(pd.DataFrame({"BBL": df19["BBL"]}))
    df22["bbl_key"] = _bbl_key(pd.DataFrame({"BBL": df22["BBL"]}))

    # Drop rows with missing BBL
    df19 = df19.dropna(subset=["bbl_key"])
    df22 = df22.dropna(subset=["bbl_key"])

    # Merge on BBL key
    merged = df19.merge(
        df22,
        on="bbl_key",
        how="outer",
        suffixes=("_2019", "_2022"),
        indicator=True,
    )

    # Assign change categories
    # For parcels in both years
    both = merged["_merge"] == "both"
    v19 = merged["is_vacant_2019"].astype(object).fillna(False).astype(bool)
    v22 = merged["is_vacant_2022"].astype(object).fillna(False).astype(bool)

    conditions = [
        both & v19 & v22,       # remained vacant
        both & ~v19 & v22,      # became vacant
        both & v19 & ~v22,      # developed
    ]
    # Parcels only in 2019 that were vacant → treat as developed (lot may have been merged/removed)
    only_19_vacant = (merged["_merge"] == "left_only") & v19
    # Parcels only in 2022 that are vacant → treat as became vacant (new subdivision)
    only_22_vacant = (merged["_merge"] == "right_only") & v22

    conditions.extend([only_19_vacant, only_22_vacant])
    choices = ["remained_vacant", "became_vacant", "developed", "developed", "became_vacant"]

    merged["change_category"] = np.select(
        conditions, choices, default="unchanged"
    )

    # Keep only changed parcels
    changed = merged[merged["change_category"] != "unchanged"].copy()
    log.info(
        f"Change categories: {changed['change_category'].value_counts().to_dict()}"
    )

    # Choose geometry: 2022 for remained/became, 2019 for developed
    # For outer-join rows, only one geometry exists
    changed["geometry"] = changed["geom_2019"]
    mask_use_22 = changed["change_category"].isin(["remained_vacant", "became_vacant"])
    changed.loc[mask_use_22, "geometry"] = changed.loc[mask_use_22, "geom_2022"]
    # Fill any remaining nulls (outer-join: only one side has geometry)
    changed["geometry"] = changed["geometry"].fillna(changed["geom_2022"])
    changed["geometry"] = changed["geometry"].fillna(changed["geom_2019"])

    # Clean up columns
    changed = changed.rename(columns={
        "BBL_2022": "BBL",
        "BldgClass_2019": "BldgClass_2019",
        "BldgClass_2022": "BldgClass_2022",
        "Borough_2022": "Borough",
        "LotArea_2022": "LotArea",
    })
    # Fill BBL from 2019 where 2022 is missing
    changed["BBL"] = changed["BBL"].fillna(changed.get("BBL_2019", np.nan))
    changed["Borough"] = changed["Borough"].fillna(changed.get("Borough_2019", ""))

    keep = ["BBL", "bbl_key", "change_category", "BldgClass_2019", "BldgClass_2022",
            "Borough", "LotArea", "geometry"]
    keep = [c for c in keep if c in changed.columns]
    changed = changed[keep].copy()

    result = gpd.GeoDataFrame(changed, geometry="geometry", crs=gdf22.crs)
    result["area_m2"] = result.geometry.area
    return result


def find_hotspots(
    change_gdf: gpd.GeoDataFrame,
    cell_size: float = 500.0,
    top_n: int = 10,
    categories: list[str] | None = None,
) -> gpd.GeoDataFrame:
    """Find rectangular hotspots where change is concentrated.

    Uses a grid-based density approach: overlay a grid, count change parcels
    per cell, take top-N cells, merge adjacent cells into bounding rectangles.

    Args:
        change_gdf: Output of compute_change() (only changed parcels).
        cell_size: Grid cell size in CRS units (meters). Default 500m.
        top_n: Number of top cells to seed hotspots from.
        categories: Which change categories to count. Default: became_vacant + developed.

    Returns:
        GeoDataFrame of hotspot bounding rectangles with count attributes.
    """
    if categories is None:
        categories = ["became_vacant", "developed"]

    subset = change_gdf[change_gdf["change_category"].isin(categories)].copy()
    if subset.empty:
        log.warning("No parcels matching hotspot categories")
        return gpd.GeoDataFrame(columns=["geometry", "hotspot_id"], crs=change_gdf.crs)

    # Get centroids for density calculation
    centroids = subset.geometry.centroid
    xs, ys = centroids.x.values, centroids.y.values

    # Build grid
    xmin, ymin, xmax, ymax = subset.total_bounds
    x_bins = np.arange(xmin, xmax + cell_size, cell_size)
    y_bins = np.arange(ymin, ymax + cell_size, cell_size)

    # 2D histogram
    counts, _, _ = np.histogram2d(xs, ys, bins=[x_bins, y_bins])

    # Find top-N cells
    flat_idx = np.argsort(counts.ravel())[::-1][:top_n]
    ix, iy = np.unravel_index(flat_idx, counts.shape)

    # Build bounding rectangles for each hotspot cell
    # Expand each cell by 1 cell in each direction for context
    hotspots = []
    for i, (xi, yi) in enumerate(zip(ix, iy)):
        if counts[xi, yi] == 0:
            continue
        x0 = x_bins[max(xi - 1, 0)]
        y0 = y_bins[max(yi - 1, 0)]
        x1 = x_bins[min(xi + 2, len(x_bins) - 1)]
        y1 = y_bins[min(yi + 2, len(y_bins) - 1)]
        rect = box(x0, y0, x1, y1)

        # Count parcels of each category within this rectangle
        in_rect = subset[subset.geometry.centroid.within(rect)]
        cats = in_rect["change_category"].value_counts().to_dict()

        hotspots.append({
            "hotspot_id": i + 1,
            "geometry": rect,
            "total_change": len(in_rect),
            "became_vacant_count": cats.get("became_vacant", 0),
            "developed_count": cats.get("developed", 0),
            "remained_vacant_count": 0,  # not counted by default
        })

    if not hotspots:
        return gpd.GeoDataFrame(columns=["geometry", "hotspot_id"], crs=change_gdf.crs)

    result = gpd.GeoDataFrame(hotspots, crs=change_gdf.crs)
    # Add remained_vacant counts
    remained = change_gdf[change_gdf["change_category"] == "remained_vacant"]
    for idx, row in result.iterrows():
        in_rect = remained[remained.geometry.centroid.within(row.geometry)]
        result.at[idx, "remained_vacant_count"] = len(in_rect)

    result = result.sort_values("total_change", ascending=False).reset_index(drop=True)
    result["hotspot_id"] = range(1, len(result) + 1)
    log.info(f"Found {len(result)} hotspot rectangles")
    return result


def compute_stats(change_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Compute per-borough statistics for each change category.

    Returns:
        DataFrame with columns: Borough, change_category, count, total_area_m2.
    """
    if change_gdf.empty:
        return pd.DataFrame(columns=["Borough", "change_category", "count", "total_area_m2"])

    stats = (
        change_gdf.groupby(["Borough", "change_category"])
        .agg(count=("BBL", "size"), total_area_m2=("area_m2", "sum"))
        .reset_index()
    )

    # Add city-wide totals
    totals = (
        change_gdf.groupby("change_category")
        .agg(count=("BBL", "size"), total_area_m2=("area_m2", "sum"))
        .reset_index()
    )
    totals["Borough"] = "ALL"
    stats = pd.concat([stats, totals], ignore_index=True)

    return stats.sort_values(["Borough", "change_category"]).reset_index(drop=True)
