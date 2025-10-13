import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import os
from typing import List
from logger import get_logger

log = get_logger()

def plot_numerical_distributions(
    gdf: gpd.GeoDataFrame,
    numerical_features: List[str],
    output_dir: str,
    n_cols: int = 3,
    bins: int = 10,
    save: bool = True,
    clip_percentile=0.95
) -> None:
    """
    Plot and optionally save histograms (and KDE curves) for numerical features in the dataset.

    Args:
        gdf: GeoDataFrame containing MapPluto or similar parcel data.
        numerical_features: List of numerical column names to plot.
        output_dir: Directory to save the plot image.
        n_cols: Number of plots per row in the grid layout (default=3).
        bins: Number of histogram bins to use (default=10).
        save: Whether to save the resulting figure (default=True).
        clip_percentile: fraction of data to keep (e.g., 0.95 keeps 5th–95th percentile)

    Notes:
        - Each subplot shows a histogram of values with an optional smoothed
          Kernel Density Estimate (KDE) curve overlayed for interpretability.
        - If some numerical columns are missing, they’ll be skipped with a log warning.
    """

    n_num_features = len(numerical_features)
    if n_num_features == 0:
        log.warning("No numerical features found for plotting.")
        return

    n_rows = (n_num_features + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    log.info(f"Clipping data to {clip_percentile} percentile for plotting")
    for i, col in enumerate(numerical_features):
        if col not in gdf.columns:
            log.warning(f"Column '{col}' not found in GeoDataFrame — skipping.")
            continue
        low, high = gdf[col].quantile([(1 - clip_percentile)/2, 1 - (1 - clip_percentile)/2])
        subset = gdf[gdf[col].between(low, high)][col].dropna()
        sns.histplot(subset.dropna(), bins=bins, kde=True, ax=axes[i])
        axes[i].set_title(f"Distribution of {col}")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Frequency")

    # Hide unused subplots
    for i in range(n_num_features, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    if save:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, "mappluto_numerical_distributions.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        log.info(f"📊 Saved numerical distributions to {out_path}")

    plt.close()

def plot_categorical_distributions(
    gdf: gpd.GeoDataFrame,
    categorical_features: list[str],
    output_dir: str,
    n_cols: int = 3,
    top_n: int = 10,
    save: bool = True
) -> None:
    """
    Plot and optionally save bar charts of categorical feature distributions.

    Args:
        gdf: GeoDataFrame containing the dataset (e.g., MapPLUTO).
        categorical_features: List of categorical column names to plot.
        output_dir: Directory path to save the resulting figure.
        n_cols: Number of subplots per row (default=3).
        top_n: Number of top categories to display per feature (default=10).
        save: Whether to save the resulting figure to output_dir (default=True).

    Notes:
        - Each subplot shows the top-N most frequent categories in descending order.
        - Missing columns are skipped with a warning in logs.
        - Categories are plotted with consistent labeling for easy comparison.
    """
    n_cat_features = len(categorical_features)
    if n_cat_features == 0:
        log.warning("No categorical features provided for plotting.")
        return

    n_rows = (n_cat_features + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(categorical_features):
        if col not in gdf.columns:
            log.warning(f"Column '{col}' not found in GeoDataFrame — skipping.")
            continue

        value_counts = gdf[col].value_counts(dropna=False).head(top_n)
        sns.barplot(
            x=value_counts.values,
            y=value_counts.index.astype(str),
            ax=axes[i],
            orient="h"
        )
        axes[i].set_title(f"Top {top_n} Categories in {col}")
        axes[i].set_xlabel("Count")
        axes[i].set_ylabel(col)

    # Hide unused subplots
    for i in range(n_cat_features, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()

    if save:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, "mappluto_categorical_distributions.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        log.info(f"📊 Saved categorical distributions to {out_path}")

    plt.close()
