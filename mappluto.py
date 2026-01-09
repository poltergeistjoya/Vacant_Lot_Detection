import geopandas as gpd
import os
from typing import Optional
from pathlib import Path
from config import Config
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


def stratified_sample(gdf, landuse_col:str='LandUse', total_samples:int=25000, class_resample: str = '11', vacant_min_frac:float=0.08, random_state:int=42):
    np.random.seed(random_state)

    # proportion of class
    landuse_pcts = gdf[landuse_col].value_counts(normalize=True)

    # Target sample size per class (proportional)
    target_per_class = (landuse_pcts * total_samples).astype(int)

    # Force minimum % for Vacant Land (LandUse == 11)
    if class_resample in target_per_class.index:
        log.info(f"Resampling class {class_resample}: {landuse_pcts[class_resample]} -> {vacant_min_frac}")
        vacant_target = max(int(total_samples * vacant_min_frac), target_per_class[class_resample])
        target_per_class[class_resample] = vacant_target

        # Rebalance other classes
        scale = (1.00-vacant_min_frac)/(1.00-landuse_pcts[class_resample])
        for lu_class in target_per_class.index:
            if lu_class != class_resample:
                target_per_class[lu_class] = int(target_per_class[lu_class] * scale)

    # Draw samples
    # TODO get rid of this warning
    """
    FutureWarning: DataFrameGroupBy.apply operated on the grouping columns. This behavior is deprecated, and in a future version of pandas the grouping columns will be excluded from the operation. Either pass `include_groups=False` to exclude the groupings or explicitly select the grouping columns after groupby to silence this warning.
  sampled_gdf = (gdf.groupby(landuse_col, group_keys=False).apply(lambda grp: grp.sample(n=min(target_per_class.get(grp.name,0), len(grp)), random_state=random_state)))
    """
    sampled_gdf = (gdf.groupby(landuse_col, group_keys=False).apply(lambda grp: grp.sample(n=min(target_per_class.get(grp.name,0), len(grp)), random_state=random_state)))

    log.info("Sampled sucessfully")
    # log.info(sampled_gdf[landuse_col].value_counts(normalize=True).round(3))

    return sampled_gdf

def load_and_sample(path: Path, layer: str, col_to_sample: str="LandUse", n_samples:int= 25000, vacant_label: str = "11", percent: float =0.08):
    """
    - Load MapPLUTO
    - Calculate perimeter of lot
    - Subset by area for EDA: 2000 < Shape_Area < 16000
    - Sample vacant (LandUse 11) up to 8%
    - Convert CRS: (NYC ft) EPSG:2263 --> (WGS84) EPSG:4326
    """
    gdf = load_gdb(path, layer=layer)
    log.info(f"Loaded {path}")
    gdf["geom_perimeter"] = gdf.geometry.length
    log.info(f"added geom_perimeter column")
    
    # subset for EDA
    filtered_by_shape_area = gdf[(gdf["Shape_Area"] >= 2000) & (gdf["Shape_Area"] <= 16000)]
    sampled_gdf = stratified_sample(filtered_by_shape_area, col_to_sample, n_samples, vacant_label, percent, 42)
    log.info(f"Sampling {col_to_sample} {vacant_label} to {percent}")
    # change from EPSG:2263 to EPSG:4326
    log.info(f"Converting CRS: {gdf.crs}, {sampled_gdf.crs} --> EPSG:4326")
    sampled_gdf = sampled_gdf.to_crs(epsg=4326)
    

    return gdf, sampled_gdf
