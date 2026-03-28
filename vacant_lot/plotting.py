import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import geopandas as gpd
import numpy as np
import os
from pathlib import Path
from typing import List, Optional
import pandas as pd
import rasterio
from rasterio.mask import mask as rio_mask
from shapely.geometry import box, mapping
from shapely.ops import unary_union
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from .logger import get_logger

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

    filename = "mappluto_numerical_distributions.png"
    plt.tight_layout()
    if save:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, filename)
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        log.info(f"📊 Saved numerical distributions to output dir / {filename}")

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
    filename = "mappluto_categorical_distributions.png"
    if save:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, filename)
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        log.info(f"📊 Saved categorical distributions to out dir / {filename}")

    plt.close()


def plot_pca(
    df: pd.DataFrame,
    output_dir: Path | str,
    color_col: Optional[str] = "cluster",
    pc1_col: str = "pc1",
    pc2_col: str = "pc2",
    highlight_col: Optional[str] = None,
    highlight_value: Optional[str | list] = None,
    title: str = "PCA — Parcel Spectral Features",
    filename: str = "pca_clusters.png",
    figsize: tuple = (10, 7),
    alpha: float = 0.4,
    point_size: int = 8,
) -> Path:
    """
    Scatter plot of PCA components, coloured by cluster or any categorical column.

    Optionally overlays a second pass of points (e.g. LandUse == "11") in a
    distinct colour so vacant lots are always visible regardless of cluster palette.

    Args:
        df: DataFrame with pc1/pc2 columns and at least one colour column.
        output_dir: Directory to write the PNG (created if absent).
        color_col: Column used to colour points (default: 'cluster').
        pc1_col: Name of the first PC column (default: 'pc1').
        pc2_col: Name of the second PC column (default: 'pc2').
        highlight_col: Optional column to filter a highlight layer (e.g. 'LandUse').
        highlight_value: Value in highlight_col to draw on top (e.g. '11').
        title: Figure title.
        filename: Output filename (default: 'pca_clusters.png').
        figsize: Figure size tuple.
        alpha: Point transparency for the base scatter.
        point_size: Marker size.

    Returns:
        Path to the saved figure.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)

    # Base scatter coloured by color_col
    categories = sorted(df[color_col].unique())
    palette = sns.color_palette("tab10", n_colors=len(categories))

    for cat, color in zip(categories, palette):
        mask = df[color_col] == cat
        ax.scatter(
            df.loc[mask, pc1_col],
            df.loc[mask, pc2_col],
            c=[color],
            label=str(cat),
            s=point_size,
            alpha=alpha,
            linewidths=0,
        )

    # Optional highlight layer drawn on top
    if highlight_col is not None and highlight_value is not None:
        values = highlight_value if isinstance(highlight_value, list) else [highlight_value]
        highlight_mask = df[highlight_col].isin(values)
        ax.scatter(
            df.loc[highlight_mask, pc1_col],
            df.loc[highlight_mask, pc2_col],
            c="#FFE026",
            edgecolors="#333333",
            linewidths=0.2,
            s=point_size * 2,
            alpha=min(alpha * 2, 1.0),
            label=f"{highlight_col} in {values}",
            zorder=5,
        )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.legend(
        title=color_col,
        markerscale=3,
        framealpha=0.7,
        loc="best",
    )
    sns.despine(ax=ax)
    plt.tight_layout()

    out_path = output_dir / filename
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    log.info(f"Saved PCA plot to output dir / {filename}")
    plt.show()

    return out_path


def plot_cluster_composition(
    df: pd.DataFrame,
    output_dir: Path | str,
    category_col: str,
    cluster_col: str = "cluster",
    vacant_codes: Optional[list] = None,
    title: str = "Land use composition per cluster",
    filename: str = "cluster_composition.png",
    figsize: tuple = (10, 6),
) -> tuple[Path, pd.DataFrame]:
    """
    Stacked bar chart showing the fraction of each category within each cluster.
    Vacant codes are merged into a single "vacant" category and highlighted in yellow.

    Args:
        df: DataFrame with cluster assignments and a categorical column.
        output_dir: Directory to write the PNG.
        category_col: Column with category labels (e.g. 'LandUse', 'BldgClass').
        cluster_col: Column with cluster labels (default: 'cluster').
        vacant_codes: List of values in category_col that represent vacant lots.
                      All are collapsed into one "vacant" bar segment in yellow.
        title: Figure title.
        filename: Output filename.
        figsize: Figure size tuple.

    Returns:
        Tuple of (path to saved figure, composition DataFrame).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_df = df.copy()

    # Collapse all vacant codes into a single label so they share one color
    VACANT_LABEL = "vacant"
    if vacant_codes:
        plot_df[category_col] = plot_df[category_col].apply(
            lambda x: VACANT_LABEL if x in vacant_codes else x
        )

    composition = (
        plot_df.groupby(cluster_col)[category_col]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
    )

    # Build color map: grey palette for non-vacant, yellow for vacant
    categories = composition.columns.tolist()
    palette = sns.color_palette("tab20", n_colors=len(categories))
    color_map = {cat: palette[i] for i, cat in enumerate(categories)}
    if VACANT_LABEL in color_map:
        color_map[VACANT_LABEL] = "#FFE026"
    bar_colors = [color_map[c] for c in categories]

    fig, ax = plt.subplots(figsize=figsize)
    composition.plot(kind="bar", stacked=True, color=bar_colors, ax=ax)
    ax.set_title(title)
    ax.set_ylabel("Fraction")
    ax.set_xlabel("Cluster")
    ax.legend(title=category_col, bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    plt.xticks(rotation=0)

    out_path = output_dir / filename
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    log.info(f"Saved cluster composition plot to output dir /{filename}")
    plt.show()

    return out_path, composition


def plot_feature_importance(
    models: dict,
    output_dir: Path | str,
    title: str = "Spectral feature importance across clusters",
    filename: str = "feature_importance.png",
    figsize: tuple = (9, 6),
) -> Path:
    """
    Show which spectral features most discriminate clusters.

    Uses the fitted KMeans centroids in scaled space. The importance of each
    feature is the standard deviation of its centroid values across all clusters:
    a high value means cluster centroids are spread far apart along that feature,
    so it drives cluster separation.  A heatmap of the raw centroid values is
    also shown so you can see *how* each cluster differs per feature.

    Args:
        models: Dict returned by cluster_dataframe() or cluster_spectral_data(),
                must contain 'kmeans', 'scaler', and 'feature_columns'.
        output_dir: Directory to write the PNG.
        title: Overall figure title.
        filename: Output filename.
        figsize: Figure size tuple.

    Returns:
        Path to the saved figure.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    kmeans = models["kmeans"]
    scaler = models["scaler"]
    feature_columns = models["feature_columns"]

    # Centroid coordinates are already in standardised (scaled) space.
    # Inverse-transform to get original-scale means per cluster.
    centroids_scaled = kmeans.cluster_centers_                    # (k, n_features)
    centroids_orig   = scaler.inverse_transform(centroids_scaled) # (k, n_features)

    centroid_df = pd.DataFrame(centroids_orig, columns=feature_columns)
    centroid_df.index.name = "cluster"

    # Feature importance = std of centroid values across clusters (scaled space)
    importance = pd.Series(
        centroids_scaled.std(axis=0), index=feature_columns
    ).sort_values(ascending=True)

    # Short labels (strip _mean suffix for readability)
    short_labels = [c.replace("_mean", "") for c in importance.index]

    fig, (ax_bar, ax_heat) = plt.subplots(
        1, 2, figsize=figsize,
        gridspec_kw={"width_ratios": [1, 1.6]},
    )

    # --- Left: importance bar chart ---
    bars = ax_bar.barh(short_labels, importance.values, color="steelblue")
    ax_bar.set_xlabel("Centroid std (scaled space)")
    ax_bar.set_title("Feature importance")
    ax_bar.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
    sns.despine(ax=ax_bar)

    # --- Right: centroid heatmap (original scale) ---
    heat_data = centroid_df[importance.index].T  # features × clusters
    heat_data.index = short_labels
    sns.heatmap(
        heat_data,
        ax=ax_heat,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        linewidths=0.5,
        cbar_kws={"label": "mean value (original scale)"},
    )
    ax_heat.set_title("Cluster centroids per feature")
    ax_heat.set_xlabel("Cluster")
    ax_heat.set_ylabel("")

    fig.suptitle(title, fontsize=13, y=1.01)
    plt.tight_layout()

    out_path = output_dir / filename
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    log.info(f"Saved feature importance plot to out dir / {filename}")
    plt.show()

    return out_path


def plot_pca_loadings(
    models: dict,
    output_dir: Path | str,
    title: str = "PCA loadings — which features drive each axis",
    filename: str = "pca_loadings.png",
    figsize: tuple = (10, 4),
) -> Path:
    """
    Bar charts of PCA component loadings for PC1 and PC2.

    A loading is the weight of each original feature on a principal component.
    High absolute loading = that feature strongly defines the axis.
    Sign tells direction: features with the same sign move together along that PC.

    Args:
        models: Dict from cluster_dataframe() — must contain 'pca' and 'feature_columns'.
        output_dir: Directory to write the PNG.
        title: Figure title.
        filename: Output filename.
        figsize: Figure size tuple.

    Returns:
        Path to the saved figure.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pca = models["pca"]
    feature_columns = models["feature_columns"]
    short_labels = [c.replace("_mean", "") for c in feature_columns]
    explained = pca.explained_variance_ratio_

    n_components = min(2, pca.n_components_)
    fig, axes = plt.subplots(1, n_components, figsize=figsize, sharey=False)
    if n_components == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        loadings = pca.components_[i]
        colors = ["steelblue" if v >= 0 else "tomato" for v in loadings]
        ax.barh(short_labels, loadings, color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(f"PC{i+1}  ({explained[i]*100:.1f}% variance)")
        ax.set_xlabel("Loading")
        sns.despine(ax=ax)

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()

    out_path = output_dir / filename
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    log.info(f"Saved PCA loadings plot output_dir / {filename}")
    plt.show()

    return out_path


# ── Geospatial figure helpers ─────────────────────────────────────────────────

def _add_scalebar(ax) -> None:
    """Add a 10 m scale bar with a dark background to *ax*."""
    scalebar = AnchoredSizeBar(
        ax.transData, 10, "10 m", loc="lower right",
        pad=0.5, color="white", frameon=True, size_vertical=1.5,
        fontproperties=fm.FontProperties(family="DejaVu Sans", size=9),
    )
    scalebar.patch.set_facecolor("black")
    scalebar.patch.set_edgecolor("none")
    scalebar.patch.set_alpha(0.6)
    ax.add_artist(scalebar)


def _add_callouts(ax, gdf: gpd.GeoDataFrame, color: str, start_n: int = 1) -> int:
    """Draw numbered circle callouts on *ax* for each row in *gdf*.

    Returns the next available callout number (start_n + len(gdf)).
    """
    for i, (_, row) in enumerate(gdf.iterrows(), start_n):
        minx, miny, maxx, maxy = row.geometry.bounds
        width = maxx - minx
        pt = row.geometry.centroid
        offset = max(width * 0.29, 9)
        ax.text(
            pt.x + offset, pt.y + offset, str(i),
            color="white", fontsize=10, fontweight="bold",
            fontfamily="DejaVu Sans",
            ha="center", va="center",
            bbox=dict(boxstyle="circle,pad=0.2", facecolor=color, edgecolor="none"),
        )
    return start_n + len(gdf)


def plot_vacancy_overview(
    naip_vrt: Path,
    export_gdf: gpd.GeoDataFrame,
    vacancy_mask_path: Path,
    cx: float,
    cy: float,
    radius: float,
    figsize: tuple = (14, 7),
) -> plt.Figure:
    """Two-panel vacancy overview figure.

    Panel (a): NAIP RGB with all parcel boundaries (white) and vacant
               parcel boundaries (red) overlaid.
    Panel (b): Greyscale NAIP with vacant pixels (mask==1) highlighted
               in red. Nodata pixels (mask==255, roads/water) show as
               plain greyscale — no overlay.

    Args:
        naip_vrt: Path to NAIP VRT.
        export_gdf: GeoDataFrame with parcel geometries and 'is_vacant' column.
        vacancy_mask_path: Path to vacancy mask GeoTIFF (0/1/255).
        cx, cy: AOI centre in the same CRS as export_gdf.
        radius: Half-side of the square AOI in CRS units (metres).

    Returns:
        Figure (does not save).
    """
    plt.rcParams["font.family"] = "DejaVu Sans"

    aoi     = box(cx - radius, cy - radius, cx + radius, cy + radius)
    aoi_gdf = gpd.GeoDataFrame(geometry=[aoi], crs=export_gdf.crs)

    with rasterio.open(naip_vrt) as src:
        img, tf = rio_mask(src, [mapping(aoi)], crop=True)

    rgb  = np.moveaxis(img[:3], 0, -1).astype(float)
    rgb  = np.clip(rgb / np.percentile(rgb, 98), 0, 1)
    grey = rgb.mean(axis=2)
    extent = [tf.c, tf.c + tf.a * img.shape[2], tf.f + tf.e * img.shape[1], tf.f]

    with rasterio.open(vacancy_mask_path) as src:
        vmask_img, _ = rio_mask(src, [mapping(aoi)], crop=True)
    vmask = vmask_img[0]

    all_clip    = gpd.clip(export_gdf, aoi_gdf)
    vacant_clip = gpd.clip(export_gdf[export_gdf["is_vacant"] == 1], aoi_gdf)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # (a) NAIP + parcel boundaries
    ax = axes[0]
    ax.imshow(rgb, extent=extent)
    all_clip.boundary.plot(ax=ax, edgecolor="white", linewidth=0.4, alpha=0.5)
    vacant_clip.boundary.plot(ax=ax, edgecolor="red", linewidth=1.0)
    _add_scalebar(ax)
    ax.set_axis_off()
    ax.text(0.5, -0.02, "(a)", transform=ax.transAxes,
            fontsize=12, fontweight="bold", fontfamily="Times New Roman",
            ha="center", va="top", color="black")

    # (b) NAIP color + binary mask overlay (1=white, 0=black, 255=transparent)
    H, W = vmask.shape
    overlay = np.zeros((H, W, 4), dtype=float)
    overlay[vmask == 1] = [1, 1, 1, 1]  # vacant → white
    overlay[vmask == 0] = [0, 0, 0, 1]  # non-vacant → black
    # vmask == 255 stays [0,0,0,0] → fully transparent

    ax = axes[1]
    ax.imshow(rgb, extent=extent)
    ax.imshow(overlay, extent=extent)
    _add_scalebar(ax)
    ax.set_axis_off()
    ax.text(0.5, -0.02, "(b)", transform=ax.transAxes,
            fontsize=12, fontweight="bold", fontfamily="Times New Roman",
            ha="center", va="top", color="black")

    plt.tight_layout()
    return fig


def plot_naip_parcels(
    naip_vrt: Path,
    parcels_gdf: gpd.GeoDataFrame,
    context_m: float = 150,
    edgecolor: str = "red",
    linewidth: float = 1.0,
    figsize: tuple = (7, 7),
) -> plt.Figure:
    """Show a NAIP patch with parcel boundaries overlaid.

    Reads a context window around *parcels_gdf*, displays NAIP RGB,
    and draws parcel boundaries in *edgecolor*. Includes a scale bar.

    Returns the Figure (does not save — call save_figure() separately).
    """
    plt.rcParams["font.family"] = "DejaVu Sans"

    union_geom = unary_union(parcels_gdf.geometry)
    minx, miny, maxx, maxy = union_geom.bounds
    context_box = box(
        minx - context_m, miny - context_m,
        maxx + context_m, maxy + context_m,
    )

    with rasterio.open(naip_vrt) as src:
        img, tf = rio_mask(src, [mapping(context_box)], crop=True)

    rgb = np.moveaxis(img[:3], 0, -1).astype(float)
    rgb = np.clip(rgb / np.percentile(rgb, 98), 0, 1)
    extent = [tf.c, tf.c + tf.a * img.shape[2], tf.f + tf.e * img.shape[1], tf.f]

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(rgb, extent=extent)
    parcels_gdf.boundary.plot(ax=ax, edgecolor=edgecolor, linewidth=linewidth)
    _add_scalebar(ax)
    ax.set_axis_off()
    plt.tight_layout()
    return fig


def plot_naip_aoi_figure(
    naip_vrt: Path,
    export_gdf: gpd.GeoDataFrame,
    id_column: str,
    # Panel (a)
    cx: float,
    cy: float,
    radius: float,
    vacant_bbls: list,
    non_vacant_labeled_bbls: list,
    non_vacant_unlabeled_bbls: list,
    # Panel (b)
    highlight_bbls: list,
    pad: float = 30,
    figsize: tuple = (14, 7),
) -> plt.Figure:
    """Create a two-panel NAIP AOI figure.

    Panel (a): square AOI centred on (cx, cy) with side 2*radius showing NAIP
    imagery with red/blue parcel overlays and numbered callouts.
    Panel (b): area around *highlight_bbls* with *pad* metres of context.

    Returns the Figure (does not save — call save_figure() separately).
    """
    plt.rcParams["font.family"] = "DejaVu Sans"

    # ── Panel (a) data ────────────────────────────────────────────────────────
    aoi = box(cx - radius, cy - radius, cx + radius, cy + radius)
    aoi_gdf = gpd.GeoDataFrame(geometry=[aoi], crs=export_gdf.crs)

    with rasterio.open(naip_vrt) as src:
        img_a, tf_a = rio_mask(src, [mapping(aoi)], crop=True)

    rgb_a = np.moveaxis(img_a[:3], 0, -1).astype(float)
    rgb_a = np.clip(rgb_a / np.percentile(rgb_a, 98), 0, 1)

    vacant_clip         = gpd.clip(export_gdf[export_gdf["is_vacant"] == 1], aoi_gdf)
    vacant_label_gdf    = gpd.clip(export_gdf[export_gdf[id_column].isin(vacant_bbls)], aoi_gdf)
    non_vacant_lbl_gdf  = gpd.clip(export_gdf[export_gdf[id_column].isin(non_vacant_labeled_bbls)], aoi_gdf)
    non_vacant_brd_gdf  = gpd.clip(export_gdf[export_gdf[id_column].isin(non_vacant_unlabeled_bbls)], aoi_gdf)

    # ── Panel (b) data ────────────────────────────────────────────────────────
    highlight_gdf = export_gdf[export_gdf[id_column].isin(highlight_bbls)].copy()
    union_geom = unary_union(highlight_gdf.geometry)
    bx, by, bmaxx, bmaxy = union_geom.bounds
    b_aoi = box(bx - pad, by - pad, bmaxx + pad, bmaxy + pad)

    with rasterio.open(naip_vrt) as src:
        img_b, tf_b = rio_mask(src, [mapping(b_aoi)], crop=True)

    rgb_b = np.moveaxis(img_b[:3], 0, -1).astype(float)
    rgb_b = np.clip(rgb_b / np.percentile(rgb_b, 98), 0, 1)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # (a)
    ax = axes[0]
    extent_a = [tf_a.c, tf_a.c + tf_a.a * img_a.shape[2],
                tf_a.f + tf_a.e * img_a.shape[1], tf_a.f]
    ax.imshow(rgb_a, extent=extent_a)
    vacant_clip.boundary.plot(ax=ax, edgecolor="red", linewidth=1.0)
    vacant_label_gdf.boundary.plot(ax=ax, edgecolor="red", linewidth=1.0)
    non_vacant_lbl_gdf.boundary.plot(ax=ax, edgecolor="blue", linewidth=1.0)
    non_vacant_brd_gdf.boundary.plot(ax=ax, edgecolor="blue", linewidth=1.0)

    next_n = _add_callouts(ax, vacant_label_gdf, color="red", start_n=1)
    _add_callouts(ax, non_vacant_lbl_gdf, color="blue", start_n=next_n)

    _add_scalebar(ax)
    ax.set_axis_off()
    ax.text(0.5, -0.02, "(a)", transform=ax.transAxes,
            fontsize=12, fontweight="bold", fontfamily="Times New Roman",
            ha="center", va="top", color="black")

    # (b)
    ax = axes[1]
    extent_b = [tf_b.c, tf_b.c + tf_b.a * img_b.shape[2],
                tf_b.f + tf_b.e * img_b.shape[1], tf_b.f]
    ax.imshow(rgb_b, extent=extent_b)
    highlight_gdf.boundary.plot(ax=ax, edgecolor="red", linewidth=1.0)

    _add_scalebar(ax)
    ax.set_axis_off()
    ax.text(0.5, -0.02, "(b)", transform=ax.transAxes,
            fontsize=12, fontweight="bold", fontfamily="Times New Roman",
            ha="center", va="top", color="black")

    plt.tight_layout()
    return fig


# ── Segmentation visualization helpers ────────────────────────────────────────


def plot_segmentation_predictions(
    patches: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    n_cols: int = 4,
    figsize_per_patch: tuple = (3.5, 3.5),
) -> plt.Figure:
    """
    Grid of (RGB | Ground Truth | Prediction | P(vacant)) rows.

    Args:
        patches: List of (rgb, true_mask, pred_mask, prob_map) tuples.
            rgb: (3, H, W) uint8.
            true_mask / pred_mask: (H, W) uint8 with 0/1/255.
            prob_map: (H, W) float32 P(vacant), NaN for ignore pixels.
        n_cols: Number of patches per row (default 4).
        figsize_per_patch: (width, height) per subplot panel.

    Returns:
        Figure (does not save — call ``save_figure()`` separately).
    """
    n_patches = len(patches)
    n_rows = 4  # RGB, truth, pred, prob per patch column
    fig_w = figsize_per_patch[0] * min(n_patches, n_cols)
    fig_h = figsize_per_patch[1] * n_rows

    # Arrange patches in rows of n_cols
    patch_rows = (n_patches + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows * patch_rows, min(n_patches, n_cols),
        figsize=(fig_w, fig_h * patch_rows),
        squeeze=False,
    )

    row_labels = ["RGB", "Ground Truth", "Prediction", "P(vacant)"]

    # Mask colormap: black=non-vacant, red=vacant, grey=ignore
    mask_cmap = plt.cm.colors.ListedColormap(["#222222", "#FF4444"])
    mask_norm = plt.cm.colors.BoundaryNorm([0, 0.5, 1], mask_cmap.N)

    for idx, (rgb, true_mask, pred_mask, prob_map) in enumerate(patches):
        pr = idx // n_cols  # patch row group
        pc = idx % n_cols   # column within group
        base_row = pr * n_rows

        # RGB
        ax = axes[base_row + 0, pc]
        rgb_disp = np.moveaxis(rgb, 0, -1)  # (H, W, 3)
        rgb_float = rgb_disp.astype(float)
        p98 = np.percentile(rgb_float, 98)
        if p98 > 0:
            rgb_float = np.clip(rgb_float / p98, 0, 1)
        ax.imshow(rgb_float)
        ax.set_axis_off()
        if pc == 0:
            ax.set_ylabel(row_labels[0], fontsize=10, fontweight="bold")

        # Ground truth
        ax = axes[base_row + 1, pc]
        display_mask = np.where(true_mask == 255, np.nan, true_mask.astype(float))
        ax.imshow(rgb_float, alpha=0.3)
        ax.imshow(display_mask, cmap=mask_cmap, norm=mask_norm, alpha=0.7, interpolation="nearest")
        ax.set_axis_off()
        if pc == 0:
            ax.set_ylabel(row_labels[1], fontsize=10, fontweight="bold")

        # Prediction
        ax = axes[base_row + 2, pc]
        display_pred = np.where(pred_mask == 255, np.nan, pred_mask.astype(float))
        ax.imshow(rgb_float, alpha=0.3)
        ax.imshow(display_pred, cmap=mask_cmap, norm=mask_norm, alpha=0.7, interpolation="nearest")
        ax.set_axis_off()
        if pc == 0:
            ax.set_ylabel(row_labels[2], fontsize=10, fontweight="bold")

        # P(vacant) heatmap
        ax = axes[base_row + 3, pc]
        ax.imshow(rgb_float, alpha=0.3)
        im = ax.imshow(prob_map, cmap="RdYlGn_r", vmin=0, vmax=1, alpha=0.7, interpolation="nearest")
        ax.set_axis_off()
        if pc == 0:
            ax.set_ylabel(row_labels[3], fontsize=10, fontweight="bold")

    # Hide unused axes
    for pr_idx in range(patch_rows):
        for pc_idx in range(min(n_patches, n_cols)):
            if pr_idx * n_cols + pc_idx >= n_patches:
                for r in range(n_rows):
                    axes[pr_idx * n_rows + r, pc_idx].set_visible(False)

    plt.tight_layout()
    return fig


def plot_rf_feature_importance(
    importances: np.ndarray,
    feature_names: list[str],
    output_path: Path | str,
    figsize: tuple = (8, 5),
) -> Path:
    """
    Horizontal bar chart of Random Forest Gini importances.

    Args:
        importances: Array of feature importances (from ``clf.feature_importances_``).
        feature_names: List of feature names matching importance order.
        output_path: Path to save the figure.
        figsize: Figure size.

    Returns:
        Path to saved figure.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    order = np.argsort(importances)
    sorted_imp = importances[order]
    sorted_names = [feature_names[i] for i in order]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(sorted_names, sorted_imp, color="steelblue")
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.set_xlabel("Gini Importance")
    ax.set_title("Pixel-Level RF Feature Importance")
    sns.despine(ax=ax)
    plt.tight_layout()

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    log.info(f"Saved feature importance plot → {output_path}")
    plt.close(fig)

    return output_path


def save_figure(fig: plt.Figure, path, dpi: int = 300) -> None:
    """Save *fig* to *path*, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    log.info(f"Saved figure → {path}")
