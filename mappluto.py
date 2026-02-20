import geopandas as gpd
import os
from typing import Optional
from pathlib import Path
import pandas as pd
import numpy as np 

from data_utils import summarize_numerical_features, summarize_categorical_features
from plotting import plot_categorical_distributions, plot_numerical_distributions
from logger import get_logger
from data_utils import load_gdb

log = get_logger()

def identify_potential_vacant_lots(gdf: gpd.GeoDataFrame) -> dict:
    """
    Identify potential vacant lots based on key MapPLUTO attributes.
    
    Logic:
    - A lot is considered potentially vacant if:
        1. NumBldgs == 0, AND
        2. (BldgArea == 0 OR null), AND
        3. (BuiltFAR == 0 OR null)
      OR all three attributes are null.
    - Also returns "strange" lots where NumBldgs is NOT 0/null 
      but BldgArea and BuiltFAR have values (inconsistent cases).
    
    Args:
        gdf (GeoDataFrame): MapPLUTO dataset
    
    Returns:
        potential_vacant: GeoDataFrame,
        strange_lots: GeoDataFrame,
        landuse_counts: pd.Series
        }
    """
    required_cols = ['NumBldgs', 'BldgArea', 'BuiltFAR', 'LandUse']
    missing = [c for c in required_cols if c not in gdf.columns]
    if missing:
        log.warning(f"Missing columns in gdf: {missing}")
        return {}

    # Ensure numeric comparisons work
    gdf = gdf.copy()

    # Potential vacant logic
    cond_vacant = (
        ((gdf['NumBldgs'] == 0) &
         ((gdf['BldgArea'].isna()) | (gdf['BldgArea'] == 0)) &
         ((gdf['BuiltFAR'].isna()) | (gdf['BuiltFAR'] == 0)))
        |
        (gdf[['NumBldgs', 'BldgArea', 'BuiltFAR']].isna().all(axis=1))
    )

    potential_vacant = gdf[cond_vacant]

    # "Strange" lots: have no buildings (0/null) but nonzero areas
    cond_strange = (
        ((gdf['NumBldgs'].isna()) | (gdf['NumBldgs'] == 0)) &
        ((gdf['BldgArea'] > 0) | (gdf['BuiltFAR'] > 0))
    )

    lots_to_inspect = gdf[cond_strange]

    # Land use breakdown for potential vacant
    landuse_counts = (
        potential_vacant['LandUse'].value_counts(dropna=False)
        .sort_index()
    )

    log.info(f"Identified {len(potential_vacant)} potential vacant lots "
                f"({len(potential_vacant) / len(gdf) * 100:.2f}% of total).")
    log.info(f"Found {len(lots_to_inspect)} inconsistent lots, inspect manually")

    return potential_vacant,lots_to_inspect,landuse_counts

def perform_mappluto_eda(
    gdf: gpd.GeoDataFrame,
    output_dir: str = "EDA/outputs",
    numerical_features: Optional[list[str]] = None,
    categorical_features: Optional[list[str]] = None,
    n_bins:int = 50,
    top_n_categories: int = 10
) -> dict:
    """
    Perform full EDA workflow for NYC MapPLUTO dataset.

    Args:
        gdf: GeoDataFrame containing MapPLUTO dataset.
        output_dir: Directory where outputs (summaries, plots) will be saved.
        numerical_features: Optional list of numerical columns to summarize/plot.
        categorical_features: Optional list of categorical columns to summarize/plot.
        top_n_categories: Number of top categories to display in categorical plots.

    Returns:
        dict: Dictionary containing:
            - "numerical_summary": DataFrame of numerical stats
            - "categorical_summary": DataFrame of categorical stats
            - "vacant_lots": GeoDataFrame of identified vacant lots
            - "lots_to_inspect": GeoDataFrame of lots with inconsistencies
            - "vacant_lot_landuse_counts": Series of LandUse counts for vacant lots
    """
    os.makedirs(output_dir, exist_ok=True)
    log.info("🚀 Starting MapPLUTO EDA pipeline")

    # --- Step 1: Numerical summaries ---
    if numerical_features:
        log.info("📈 Summarizing numerical features...")
        num_summary = summarize_numerical_features(gdf, numerical_features)
        num_summary_path = os.path.join(output_dir, "numerical_summary.csv")
        num_summary.to_csv(num_summary_path, index=True)
        log.info(f"Saved numerical summary to {num_summary_path}")

        log.info("📊 Plotting numerical distributions...")
        plot_numerical_distributions(
            gdf=gdf,
            numerical_features=numerical_features,
            output_dir=output_dir,
            n_cols=3,
            bins=n_bins,
            save=True,
            clip_percentile=.95
        )
    else:
        log.warning("No numerical features provided for EDA.")
        num_summary = pd.DataFrame()

    # --- Step 2: Categorical summaries ---
    if categorical_features:
        log.info("📂 Summarizing categorical features...")
        cat_summary = summarize_categorical_features(gdf, categorical_features)

        for col, summary in cat_summary.items():
            cat_summary_path = Path(output_dir) / f"categorical_summary_{col}.csv"
            summary.to_csv(cat_summary_path)
            log.info(f"Saved categorical summary for {col} to {cat_summary_path}")


        log.info("🧭 Plotting categorical distributions...")
        plot_categorical_distributions(
            gdf=gdf,
            categorical_features=categorical_features,
            output_dir=output_dir,
            n_cols=3,
            top_n=top_n_categories,
            save=True
        )
    else:
        log.warning("No categorical features provided for EDA.")
        cat_summary = pd.DataFrame()

    # --- Step 3: Identify potential vacant lots ---
    log.info("🏗️ Identifying potential vacant lots...")
    vacant_lots, lots_to_inspect, landuse_counts = identify_potential_vacant_lots(gdf)

    vacant_lots_path = os.path.join(output_dir, "potential_vacant_lots.geojson")
    vacant_lots.to_file(vacant_lots_path, driver="GeoJSON")
    log.info(f"Saved potential vacant lots to {vacant_lots_path}")

    lots_to_inspect_path = os.path.join(output_dir, "lots_to_inspect.geojson")
    lots_to_inspect.to_file(lots_to_inspect_path, driver="GeoJSON")
    log.info(f"Saved lots to inspect to {lots_to_inspect_path}")

    # --- Step 4: Return results ---
    log.info("✅ MapPLUTO EDA pipeline completed successfully.")

    return {
        "numerical_summary": num_summary,
        "categorical_summary": cat_summary,
        "vacant_lots": vacant_lots,
        "lots_to_inspect": lots_to_inspect,
        "vacant_lot_landuse_counts": landuse_counts
    }

def load_and_sample(
    path: Path,
    layer: str,
    land_use_codes: list[str],
    col_to_sample: str,
    projected_crs: str,
    vacant_min_fraction: float,
    total_samples: int = 25000,
    resolution: float = 1.0,
    min_pixels: int = 50,
    random_state: int = 42,
):
    """
    Load MapPLUTO parcels and create a stratified sample.

    Args:
        path: Path to the MapPLUTO GDB file
        layer: Layer name in the GDB
        land_use_codes: Land use codes to oversample; first entry is used as the target class
        col_to_sample: Column name for stratified sampling (e.g. "LandUse")
        projected_crs: Projected metric CRS for area computation (e.g. "EPSG:32618")
        vacant_min_fraction: Minimum fraction for oversampled class
        total_samples: Total number of samples to draw
        resolution: Raster pixel size in meters (e.g. 1.0 for NAIP)
        min_pixels: Minimum parcel area as pixel count; min_area_m2 = min_pixels × resolution²
        random_state: Random seed for reproducibility

    Returns:
        tuple: (full_gdf, sampled_gdf) both in EPSG:4326
    """
    gdf = load_gdb(path, layer=layer)
    log.info(f"Loaded {path}")

    # Reproject to metric CRS for geometry-derived area computation
    gdf_m = gdf.to_crs(projected_crs)
    epsg_tag = projected_crs.replace(":", "").lower()
    gdf_m[f"area_m2_{epsg_tag}"] = gdf_m.geometry.area
    gdf_m[f"geom_perimeter_{epsg_tag}"] = gdf_m.geometry.length
    log.info(f"Computed 'area_m2_{epsg_tag}' and 'geom_perimeter_{epsg_tag}' in {projected_crs}")

    # Filter: only remove parcels too small to yield reliable spectral stats
    min_area_m2 = min_pixels * resolution ** 2
    filtered = gdf_m[gdf_m[f"area_m2_{epsg_tag}"] >= min_area_m2]
    log.info(f"min_area_m2: {min_area_m2:.1f} m² ({min_pixels} pixels × {resolution}² m) — {len(filtered):,} parcels after min filter")

    n_target = int(total_samples * vacant_min_fraction)
    n_rest = total_samples - n_target
    target = filtered[filtered[col_to_sample].isin(land_use_codes)]
    rest = filtered[~filtered[col_to_sample].isin(land_use_codes)]
    sampled_gdf = pd.concat([
        target.sample(n=min(n_target, len(target)), random_state=random_state),
        rest.sample(n=min(n_rest, len(rest)), random_state=random_state),
    ])
    log.info(f"Sampled {len(sampled_gdf):,} parcels ({n_target} target {land_use_codes}, {n_rest} non-target)")

    # Convert to EPSG:4326 for GEE output; float columns survive the reprojection
    log.info(f"Converting CRS: {projected_crs} --> EPSG:4326")
    gdf_out = gdf_m.to_crs(epsg=4326)
    sampled_out = sampled_gdf.to_crs(epsg=4326)

    return gdf_out, sampled_out
