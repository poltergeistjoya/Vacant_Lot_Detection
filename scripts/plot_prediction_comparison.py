"""
Compare model predictions against NAIP imagery and parcel ground truth.

Three figure types:

1. **Single-BBL inspection** (``plot_bbl_inspection``):
   Two panels for one BBL and one model run:
     (a) NAIP + error overlay (TP/FP/FN/TN colours, semi-transparent)
     (b) NAIP + parcel outlines coloured by vacancy status, with BBL labels

2. **Model comparison** (``plot_model_comparison``):
   1×3 panels for one BBL, comparing UNet vs DeepLabV3+:
     (a) NAIP + parcel outlines
     (b) UNet error overlay
     (c) DeepLabV3+ error overlay

3. **Error gallery** (``plot_error_gallery``):
   4-column × N-row grid sampling TP, FP, FN, TN patches from a given run.

Usage:
  uv run python scripts/plot_prediction_comparison.py bbl \\
      --bbl 2034910001 --arch unet --run-id kahan_031 --split test \\
      --stride 256 --threshold 0.425

  uv run python scripts/plot_prediction_comparison.py compare \\
      --bbl 2034910001 --unet-run kahan_031 --deeplab-run kahan_027 --split test \\
      --unet-stride 256 --unet-threshold 0.425 \\
      --deeplab-stride 512 --deeplab-threshold 0.298

  uv run python scripts/plot_prediction_comparison.py gallery \\
      --arch unet --run-id kahan_031 --split test --rows 8 \\
      --stride 256 --threshold 0.425
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from rasterio.mask import mask as rio_mask
from shapely.geometry import box, mapping

from vacant_lot.config import load_data_config, _get_shared_root
from vacant_lot.data_utils import load_gdb

# ── Style ────────────────────────────────────────────────────────────────────

_FONT_FAMILY = "STIX Two Text"

mpl.rcParams.update({
    "font.family": _FONT_FAMILY,
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.titleweight": "normal",
    "axes.labelweight": "normal",
})

# Error-map channel interpretation (from visualize_predictions.py):
#   R  G  B  A
#   TP: (0, 255, 0, 255)   green
#   FP: (255, 0, 0, 255)   red
#   FN: (0, 0, 255, 255)   blue
#   TN: (0, 0, 0, 255)     black
#   Ignore: (255, 255, 255, 128) white semi-transparent

_VACANT_EDGE = "#3498db"       # blue for vacant parcels
_NONVACANT_EDGE = "#e74c3c"    # red for non-vacant parcels
_LABEL_FONT = dict(family=_FONT_FAMILY, size=8)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _add_scalebar(ax, length_m: float = 20) -> None:
    """Add a scale bar to *ax*."""
    scalebar = AnchoredSizeBar(
        ax.transData, length_m, f"{int(length_m)} m", loc="lower right",
        pad=0.5, color="white", frameon=True, size_vertical=1.5,
        fontproperties=fm.FontProperties(family=_FONT_FAMILY, size=8),
    )
    scalebar.patch.set_facecolor("black")
    scalebar.patch.set_edgecolor("none")
    scalebar.patch.set_alpha(0.6)
    ax.add_artist(scalebar)


def _subplot_label(ax, label: str) -> None:
    """Place a centred label below the axes, e.g. '(a)'."""
    ax.text(0.5, -0.02, label, transform=ax.transAxes,
            fontsize=11, fontweight="bold", fontfamily="Times New Roman",
            ha="center", va="top", color="black")


def _load_parcels(data_cfg) -> gpd.GeoDataFrame:
    """Load MapPLUTO parcels with is_vacant column."""
    gdf = load_gdb(data_cfg.get_parcel_path(), layer=data_cfg.parcels.layer)
    gdf = gdf.to_crs(data_cfg.raster.output_crs)
    vacant_mask = gdf[data_cfg.parcels.landuse_column].isin(data_cfg.parcels.vacant_codes)
    gdf["is_vacant"] = vacant_mask.astype(int)
    return gdf


def _read_naip_crop(vrt_path: Path, aoi_geom):
    """Read NAIP RGB for an AOI geometry, return (rgb_float, extent, transform)."""
    with rasterio.open(vrt_path) as src:
        img, tf = rio_mask(src, [mapping(aoi_geom)], crop=True)
    rgb = np.moveaxis(img[:3], 0, -1).astype(float)
    p98 = np.percentile(rgb, 98)
    if p98 > 0:
        rgb = np.clip(rgb / p98, 0, 1)
    extent = [tf.c, tf.c + tf.a * img.shape[2],
              tf.f + tf.e * img.shape[1], tf.f]
    return rgb, extent, tf


def _read_error_crop(error_tif: Path, aoi_geom):
    """Read RGBA error map for an AOI, return (rgba_float H×W×4, extent)."""
    with rasterio.open(error_tif) as src:
        img, tf = rio_mask(src, [mapping(aoi_geom)], crop=True)
    # img shape: (4, H, W) uint8
    rgba = np.moveaxis(img, 0, -1).astype(float) / 255.0
    extent = [tf.c, tf.c + tf.a * img.shape[2],
              tf.f + tf.e * img.shape[1], tf.f]
    return rgba, extent


def _bbl_aoi(gdf: gpd.GeoDataFrame, bbl: int, id_column: str, pad: float = 40):
    """Return (aoi_box, clipped_parcels_near_bbl)."""
    target = gdf[gdf[id_column] == bbl]
    if target.empty:
        raise ValueError(f"BBL {bbl} not found in parcel data")
    geom = target.geometry.values[0]
    minx, miny, maxx, maxy = geom.bounds
    # Make it square-ish and add padding
    cx = (minx + maxx) / 2
    cy = (miny + maxy) / 2
    half = max(maxx - minx, maxy - miny) / 2 + pad
    aoi = box(cx - half, cy - half, cx + half, cy + half)
    return aoi


def _resolve_error_tif(shared_root: Path, arch: str, run_id: str,
                       split: str, stride: int | None = None,
                       threshold: float | None = None) -> Path:
    """Find the error TIF for a given run.

    Naming convention from visualize_predictions.py:
      {split}_error.tif                       — no overlap, default threshold
      {split}_error_s{stride}.tif             — with overlap stride
      {split}_error_s{stride}_t{thresh}.tif   — overlap + custom threshold
      {split}_error_t{thresh}.tif             — no overlap + custom threshold

    Threshold is encoded as int(threshold * 1000), e.g. 0.298 → t298.
    """
    figures_dir = shared_root / "outputs" / "models" / arch / run_id / "figures"
    t_suffix = f"_t{int(threshold * 1000)}" if threshold is not None else ""
    s_suffix = f"_s{stride}" if stride is not None else ""

    # Most specific first, then fall back
    candidates = []
    if stride is not None and threshold is not None:
        candidates.append(figures_dir / f"{split}_error_s{stride}_t{int(threshold * 1000)}.tif")
    if stride is not None:
        candidates.append(figures_dir / f"{split}_error_s{stride}.tif")
    if threshold is not None:
        candidates.append(figures_dir / f"{split}_error_t{int(threshold * 1000)}.tif")
    candidates.append(figures_dir / f"{split}_error.tif")
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No error TIF found for {arch}/{run_id} split={split}. "
        f"Tried: {[str(c) for c in candidates]}")


def _add_callouts(ax, gdf: gpd.GeoDataFrame, color: str,
                  start_n: int = 1) -> int:
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
            fontfamily=_FONT_FAMILY,
            ha="center", va="center",
            bbox=dict(boxstyle="circle,pad=0.2", facecolor=color,
                      edgecolor="none"),
        )
    return start_n + len(gdf)


def _draw_naip_with_parcels(ax, rgb, extent, gdf, aoi, id_column: str):
    """Draw NAIP with all parcel outlines and numbered callouts.

    All parcels in the AOI are outlined (red = vacant, blue = non-vacant).
    Vacant parcels get red numbered callouts; non-vacant parcels with
    area > 50 m^2 get blue numbered callouts.
    """
    aoi_gdf = gpd.GeoDataFrame(geometry=[aoi], crs=gdf.crs)
    clipped = gpd.clip(gdf, aoi_gdf)
    vacant_clip = clipped[clipped["is_vacant"] == 1]
    nonvacant_clip = clipped[clipped["is_vacant"] == 0]

    ax.imshow(rgb, extent=extent)
    # Outline all parcels
    nonvacant_clip.boundary.plot(ax=ax, edgecolor=_NONVACANT_EDGE,
                                 linewidth=0.6, alpha=0.6)
    vacant_clip.boundary.plot(ax=ax, edgecolor=_VACANT_EDGE, linewidth=1.2)

    # Numbered callouts — vacant first (red), then non-vacant (blue)
    # Only label non-vacant parcels above a minimum size to avoid clutter
    nonvacant_notable = nonvacant_clip[nonvacant_clip.geometry.area > 50]
    next_n = _add_callouts(ax, vacant_clip, color=_VACANT_EDGE, start_n=1)
    _add_callouts(ax, nonvacant_notable, color=_NONVACANT_EDGE, start_n=next_n)


def _draw_naip_with_error(ax, rgb, extent, error_rgba, error_extent,
                          alpha: float = 0.55):
    """Draw NAIP with transparent error overlay."""
    ax.imshow(rgb, extent=extent)
    # Mask out the white/ignore regions for cleaner overlay
    overlay = error_rgba.copy()
    # Where alpha channel in error map is < 1 (ignore regions), make fully transparent
    is_ignore = (error_rgba[:, :, 0] > 0.9) & (error_rgba[:, :, 1] > 0.9) & (error_rgba[:, :, 2] > 0.9)
    overlay[is_ignore, 3] = 0
    # For non-ignore regions, set desired alpha
    overlay[~is_ignore, 3] = alpha
    # TN (black) — make mostly transparent so NAIP shows through
    is_tn = (error_rgba[:, :, 0] < 0.1) & (error_rgba[:, :, 1] < 0.1) & (error_rgba[:, :, 2] < 0.1) & (~is_ignore)
    overlay[is_tn, 3] = 0.1
    ax.imshow(overlay, extent=error_extent, interpolation="nearest")


def _error_legend(ax) -> None:
    """Add a small legend for TP/FP/FN colours."""
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=(0, 1, 0, 0.7), label="TP (correct vacant)"),
        Patch(facecolor=(1, 0, 0, 0.7), label="FP (false vacant)"),
        Patch(facecolor=(0, 0, 1, 0.7), label="FN (missed vacant)"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=6,
              framealpha=0.7, edgecolor="none")


# ── Figure 1: Single-BBL inspection ─────────────────────────────────────────

def plot_bbl_inspection(
    bbl: int,
    arch: str,
    run_id: str,
    split: str,
    pad: float = 40,
    stride: int | None = None,
    threshold: float | None = None,
    figsize: tuple = (14, 7),
) -> plt.Figure:
    """Two-panel figure: (a) error overlay, (b) parcel outlines + BBL labels."""
    shared_root = _get_shared_root()
    data_cfg = load_data_config()
    gdf = _load_parcels(data_cfg)
    vrt_path = data_cfg.get_vrt_path()
    id_column = data_cfg.parcels.id_column

    aoi = _bbl_aoi(gdf, bbl, id_column, pad=pad)
    rgb, extent, _ = _read_naip_crop(vrt_path, aoi)

    error_tif = _resolve_error_tif(shared_root, arch, run_id, split, stride, threshold)
    error_rgba, error_extent = _read_error_crop(error_tif, aoi)

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    title = f"{arch.upper()} {run_id} — BBL {bbl} ({split})"
    fig.suptitle(title, fontsize=12, y=1.01)

    # (a) Error overlay
    ax = axes[0]
    _draw_naip_with_error(ax, rgb, extent, error_rgba, error_extent)
    _error_legend(ax)
    _add_scalebar(ax)
    ax.set_axis_off()
    _subplot_label(ax, "(a)")

    # (b) Parcel outlines + BBL labels
    ax = axes[1]
    _draw_naip_with_parcels(ax, rgb, extent, gdf, aoi, id_column)
    _add_scalebar(ax)
    ax.set_axis_off()
    _subplot_label(ax, "(b)")

    plt.tight_layout()
    return fig


# ── Figure 2: UNet vs DeepLabV3+ comparison ─────────────────────────────────

def plot_model_comparison(
    bbl: int,
    unet_run: str,
    deeplab_run: str,
    split: str,
    pad: float = 40,
    unet_stride: int | None = None,
    unet_threshold: float | None = None,
    deeplab_stride: int | None = None,
    deeplab_threshold: float | None = None,
    figsize: tuple = (21, 7),
) -> plt.Figure:
    """1×3 figure: (a) NAIP + parcels, (b) UNet error, (c) DeepLabV3+ error."""
    shared_root = _get_shared_root()
    data_cfg = load_data_config()
    gdf = _load_parcels(data_cfg)
    vrt_path = data_cfg.get_vrt_path()
    id_column = data_cfg.parcels.id_column

    aoi = _bbl_aoi(gdf, bbl, id_column, pad=pad)
    rgb, extent, _ = _read_naip_crop(vrt_path, aoi)

    unet_tif = _resolve_error_tif(shared_root, "unet", unet_run, split,
                                   unet_stride, unet_threshold)
    unet_rgba, unet_ext = _read_error_crop(unet_tif, aoi)

    deeplab_tif = _resolve_error_tif(shared_root, "deeplabv3plus", deeplab_run, split,
                                      deeplab_stride, deeplab_threshold)
    deeplab_rgba, deeplab_ext = _read_error_crop(deeplab_tif, aoi)

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    title = f"BBL {bbl} ({split}) — UNet {unet_run} vs DeepLabV3+ {deeplab_run}"
    fig.suptitle(title, fontsize=12, y=1.01)

    # (a) NAIP + parcel outlines
    ax = axes[0]
    _draw_naip_with_parcels(ax, rgb, extent, gdf, aoi, id_column)
    _add_scalebar(ax)
    ax.set_axis_off()
    _subplot_label(ax, "(a) NAIP + parcels")

    # (b) UNet error
    ax = axes[1]
    _draw_naip_with_error(ax, rgb, extent, unet_rgba, unet_ext)
    _error_legend(ax)
    _add_scalebar(ax)
    ax.set_axis_off()
    _subplot_label(ax, f"(b) UNet {unet_run}")

    # (c) DeepLabV3+ error
    ax = axes[2]
    _draw_naip_with_error(ax, rgb, extent, deeplab_rgba, deeplab_ext)
    _error_legend(ax)
    _add_scalebar(ax)
    ax.set_axis_off()
    _subplot_label(ax, f"(c) DeepLab {deeplab_run}")

    plt.tight_layout()
    return fig


# ── Figure 3: Error gallery (TP / FP / FN / TN grid) ────────────────────────

def plot_error_gallery(
    arch: str,
    run_id: str,
    split: str,
    rows: int = 8,
    patch_radius: float = 80,
    stride: int | None = None,
    threshold: float | None = None,
    seed: int = 42,
    figsize_per_cell: tuple = (3.5, 3.5),
) -> plt.Figure:
    """4-column × N-row grid: each column is one error class (TP, FP, FN, TN).

    Samples random locations from the error map where each class dominates,
    then shows the NAIP crop with the error overlay for that patch.
    """
    shared_root = _get_shared_root()
    data_cfg = load_data_config()
    vrt_path = data_cfg.get_vrt_path()

    error_tif = _resolve_error_tif(shared_root, arch, run_id, split, stride, threshold)

    rng = np.random.default_rng(seed)

    # Read full error map
    with rasterio.open(error_tif) as src:
        error_full = src.read()  # (4, H, W) uint8
        error_tf = src.transform
        error_crs = src.crs

    r_chan, g_chan, b_chan, a_chan = error_full

    # Classify each pixel by error type (only where alpha == 255 = valid)
    valid = a_chan == 255
    tp = valid & (g_chan == 255) & (r_chan == 0) & (b_chan == 0)
    fp = valid & (r_chan == 255) & (g_chan == 0) & (b_chan == 0)
    fn = valid & (b_chan == 255) & (r_chan == 0) & (g_chan == 0)
    tn = valid & (r_chan == 0) & (g_chan == 0) & (b_chan == 0)

    categories = [
        ("TP", tp, (0, 1, 0)),
        ("FP", fp, (1, 0, 0)),
        ("FN", fn, (0, 0, 1)),
        ("TN", tn, (0.2, 0.2, 0.2)),
    ]

    # For each category, find dense clusters by downsampling
    block = int(patch_radius * 2 / abs(error_tf.a))  # block size in pixels
    half_block = block // 2

    def _sample_centres(mask, n, block_sz):
        """Find n random pixel locations where *mask* is dense."""
        h, w = mask.shape
        # Downsample to coarse grid, pick blocks with most hits
        bh = h // block_sz
        bw = w // block_sz
        if bh == 0 or bw == 0:
            return []
        trimmed = mask[:bh * block_sz, :bw * block_sz]
        blocks = trimmed.reshape(bh, block_sz, bw, block_sz).sum(axis=(1, 3))
        # Need at least some hits
        threshold = block_sz * block_sz * 0.02  # at least 2% of pixels
        good_ys, good_xs = np.where(blocks > threshold)
        if len(good_ys) == 0:
            return []
        # Sort by density, take top candidates, then sample
        densities = blocks[good_ys, good_xs]
        order = np.argsort(-densities)
        top_n = min(len(order), n * 5)
        indices = rng.choice(top_n, size=min(n, top_n), replace=False)
        centres = []
        for idx in indices:
            by = good_ys[order[idx]]
            bx = good_xs[order[idx]]
            # Centre of block in pixel coords
            py = by * block_sz + half_block
            px = bx * block_sz + half_block
            # Convert to map coords
            mx, my = rasterio.transform.xy(error_tf, py, px)
            centres.append((mx, my))
        return centres

    fig_w = figsize_per_cell[0] * 4
    fig_h = figsize_per_cell[1] * rows
    fig, axes = plt.subplots(rows, 4, figsize=(fig_w, fig_h))
    if rows == 1:
        axes = axes[np.newaxis, :]

    for col_idx, (cat_name, cat_mask, cat_color) in enumerate(categories):
        centres = _sample_centres(cat_mask, rows, block)
        # Column header
        axes[0, col_idx].set_title(cat_name, fontsize=14, fontweight="bold",
                                    color=cat_color, pad=8)
        for row_idx in range(rows):
            ax = axes[row_idx, col_idx]
            if row_idx >= len(centres):
                ax.set_visible(False)
                continue
            mx, my = centres[row_idx]
            aoi = box(mx - patch_radius, my - patch_radius,
                      mx + patch_radius, my + patch_radius)
            try:
                rgb, extent, _ = _read_naip_crop(vrt_path, aoi)
                error_rgba, error_ext = _read_error_crop(error_tif, aoi)
                _draw_naip_with_error(ax, rgb, extent, error_rgba, error_ext,
                                      alpha=0.5)
            except Exception:
                ax.set_visible(False)
                continue
            _add_scalebar(ax, length_m=20)
            ax.set_axis_off()

    fig.suptitle(f"{arch.upper()} {run_id} — {split} error gallery",
                 fontsize=14, y=1.0)
    plt.tight_layout()
    return fig


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare model predictions against NAIP imagery and parcels")
    sub = parser.add_subparsers(dest="command", required=True)

    # -- bbl subcommand --
    p_bbl = sub.add_parser("bbl", help="Single-BBL inspection (2 panels)")
    p_bbl.add_argument("--bbl", type=int, required=True)
    p_bbl.add_argument("--arch", required=True, choices=["unet", "deeplabv3plus"])
    p_bbl.add_argument("--run-id", required=True)
    p_bbl.add_argument("--split", default="test")
    p_bbl.add_argument("--pad", type=float, default=40)
    p_bbl.add_argument("--stride", type=int, default=None)
    p_bbl.add_argument("--threshold", type=float, default=None,
                       help="Threshold used when generating error TIF (e.g. 0.425)")
    p_bbl.add_argument("--out", default=None, help="Output path (default: figures dir)")

    # -- compare subcommand --
    p_cmp = sub.add_parser("compare", help="UNet vs DeepLabV3+ (3 panels)")
    p_cmp.add_argument("--bbl", type=int, required=True)
    p_cmp.add_argument("--unet-run", required=True)
    p_cmp.add_argument("--deeplab-run", required=True)
    p_cmp.add_argument("--split", default="test")
    p_cmp.add_argument("--pad", type=float, default=40)
    p_cmp.add_argument("--unet-stride", type=int, default=None)
    p_cmp.add_argument("--unet-threshold", type=float, default=None)
    p_cmp.add_argument("--deeplab-stride", type=int, default=None)
    p_cmp.add_argument("--deeplab-threshold", type=float, default=None)
    p_cmp.add_argument("--out", default=None)

    # -- gallery subcommand --
    p_gal = sub.add_parser("gallery", help="TP/FP/FN/TN error gallery")
    p_gal.add_argument("--arch", required=True, choices=["unet", "deeplabv3plus"])
    p_gal.add_argument("--run-id", required=True)
    p_gal.add_argument("--split", default="test")
    p_gal.add_argument("--rows", type=int, default=8)
    p_gal.add_argument("--stride", type=int, default=None)
    p_gal.add_argument("--threshold", type=float, default=None,
                       help="Threshold used when generating error TIF (e.g. 0.425)")
    p_gal.add_argument("--seed", type=int, default=42)
    p_gal.add_argument("--out", default=None)

    args = parser.parse_args()
    shared_root = _get_shared_root()

    if args.command == "bbl":
        fig = plot_bbl_inspection(
            bbl=args.bbl, arch=args.arch, run_id=args.run_id,
            split=args.split, pad=args.pad, stride=args.stride,
            threshold=args.threshold,
        )
        out = args.out or str(
            shared_root / "outputs" / "models" / args.arch / args.run_id
            / "figures" / f"bbl_{args.bbl}_{args.split}.png")

    elif args.command == "compare":
        fig = plot_model_comparison(
            bbl=args.bbl, unet_run=args.unet_run, deeplab_run=args.deeplab_run,
            split=args.split, pad=args.pad,
            unet_stride=args.unet_stride, unet_threshold=args.unet_threshold,
            deeplab_stride=args.deeplab_stride, deeplab_threshold=args.deeplab_threshold,
        )
        out = args.out or str(
            shared_root / "outputs" / "figures"
            / f"compare_bbl_{args.bbl}_{args.split}.png")

    elif args.command == "gallery":
        fig = plot_error_gallery(
            arch=args.arch, run_id=args.run_id, split=args.split,
            rows=args.rows, stride=args.stride, threshold=args.threshold,
            seed=args.seed,
        )
        out = args.out or str(
            shared_root / "outputs" / "models" / args.arch / args.run_id
            / "figures" / f"error_gallery_{args.split}.png")

    from vacant_lot.plotting import save_figure
    save_figure(fig, out)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
