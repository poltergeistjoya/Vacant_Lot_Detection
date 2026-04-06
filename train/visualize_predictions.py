"""
Visualize model predictions across all val/test patches as georeferenced rasters.

Produces for each split:
  - {split}_pred.tif       — predicted probabilities (float32)
  - {split}_error.tif      — color-coded error map:
                              Green = TP (correctly predicted vacant)
                              Red   = FP (predicted vacant, actually not)
                              Blue  = FN (missed vacant)
                              Black = TN (correctly predicted non-vacant)
                              White = ignore (255 in ground truth)

Supports overlapping inference (--stride) to reduce edge artifacts by averaging
predictions from multiple overlapping patches.

Usage:
  uv run python train/visualize_predictions.py --run outputs/models/unet/kahan_011
  uv run python train/visualize_predictions.py --run outputs/models/unet/kahan_011 --stride 128
  uv run python train/visualize_predictions.py --run outputs/models/unet/kahan_011 --threshold 0.3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
import torch
from tqdm import tqdm

from vacant_lot.config import load_data_config, _get_shared_root
from vacant_lot.dataset import NAIPSegmentationDataset, load_patch_splits, generate_overlap_splits


def main():
    parser = argparse.ArgumentParser(description="Visualize predictions on val/test splits")
    parser.add_argument("--run", required=True, help="Path to run directory (e.g. outputs/models/unet/001)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Binarization threshold (default: 0.5)")
    parser.add_argument("--stride", type=int, default=None,
                        help="Inference stride (default: patch_size = no overlap). Use e.g. 128 for 50%% overlap.")
    parser.add_argument("--splits", nargs="+", default=["val", "test"], help="Splits to visualize")
    parser.add_argument("--patch-size", type=int, default=None, help="Patch size (overrides data config)")
    parser.add_argument("--patch-splits", default=None, help="Path to patch_splits JSON (overrides data config)")
    parser.add_argument("--error-only", action="store_true", help="Skip writing prob TIF, only write error map")
    args = parser.parse_args()

    shared_root = _get_shared_root()
    run_dir = shared_root / args.run
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load data config (for mask paths)
    data_cfg = load_data_config()
    vacancy_mask_path = data_cfg.get_vacancy_mask_path()

    # Determine inference stride suffix
    inference_stride = args.stride
    use_overlap = inference_stride is not None

    if args.error_only:
        _error_only(args, shared_root, run_dir, figures_dir, data_cfg,
                    vacancy_mask_path, inference_stride, use_overlap)
        return

    vrt_path = data_cfg.get_vrt_path()

    # Load model config from metrics.json
    metrics = json.loads((run_dir / "metrics.json").read_text())
    model_cfg = metrics["model"]
    in_channels = model_cfg.get("in_channels", 10)

    # Load patch splits — prefer CLI override, fall back to data.yaml
    if args.patch_splits is not None:
        splits_path = shared_root / args.patch_splits
    else:
        splits_path = data_cfg.get_patch_splits_path()
    splits, splits_meta = load_patch_splits(splits_path)
    patch_size = args.patch_size if args.patch_size is not None else splits_meta["patch_size"]

    # Determine inference stride
    if inference_stride is None:
        inference_stride = patch_size
        use_overlap = False

    # Generate overlap grid if needed
    if use_overlap:
        print(f"Generating overlapping grid with stride={inference_stride} (patch_size={patch_size})")
        borough_mask_path = data_cfg.get_borough_mask_path()
        splits = generate_overlap_splits(
            vacancy_mask_path=vacancy_mask_path,
            borough_mask_path=borough_mask_path,
            split_meta=splits_meta,
            patch_size=patch_size,
            stride=inference_stride,
        )

    # Build model and load weights
    from vacant_lot.segmentation import build_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        arch=model_cfg["type"],
        in_channels=in_channels,
        encoder_name=model_cfg["encoder_name"],
        encoder_weights=None,  # loading from checkpoint
        classes=model_cfg["classes"],
        decoder_channels=model_cfg.get("decoder_channels"),
    ).to(device)

    best_pt = run_dir / "best.pt"
    ckpt = torch.load(best_pt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded {best_pt} (epoch {ckpt['epoch']})")

    # Get raster metadata from vacancy mask
    with rasterio.open(vacancy_mask_path) as src:
        raster_profile = src.profile.copy()
        raster_transform = src.transform

    for split_name in args.splits:
        if split_name not in splits:
            print(f"Skipping {split_name} — not in splits")
            continue

        patch_coords = splits[split_name]
        print(f"\n=== {split_name}: {len(patch_coords)} patches ===")

        # Compute bounding box of patches to avoid allocating full-raster arrays
        rows = [r for r, c in patch_coords]
        cols = [c for r, c in patch_coords]
        min_row, max_row = min(rows), max(rows) + patch_size
        min_col, max_col = min(cols), max(cols) + patch_size
        crop_h = max_row - min_row
        crop_w = max_col - min_col
        print(f"  Crop region: rows [{min_row}:{max_row}], cols [{min_col}:{max_col}] ({crop_h}x{crop_w})")

        dataset = NAIPSegmentationDataset(
            vrt_path=vrt_path,
            vacancy_mask_path=vacancy_mask_path,
            patch_coords=patch_coords,
            patch_size=patch_size,
            in_channels=in_channels,
        )

        # Initialize cropped output arrays
        prob_sum = np.zeros((crop_h, crop_w), dtype=np.float64)
        weight_sum = np.zeros((crop_h, crop_w), dtype=np.float64)
        gt_map = np.full((crop_h, crop_w), 255, dtype=np.uint8)

        # Build 2D Gaussian blending kernel (sigma = patch_size/4 → ~0 at edges)
        # Floor at small epsilon so edge pixels still contribute when no other patch covers them.
        coord = np.arange(patch_size) - (patch_size - 1) / 2.0
        sigma = patch_size / 4.0
        gauss_1d = np.exp(-(coord ** 2) / (2 * sigma ** 2))
        kernel = np.outer(gauss_1d, gauss_1d).astype(np.float64)
        kernel = np.maximum(kernel, 1e-3)

        # Run inference patch by patch
        with torch.no_grad():
            for i in tqdm(range(len(dataset)), desc=f"Predict {split_name}"):
                row_off, col_off = patch_coords[i]
                image, mask = dataset[i]

                logits = model(image.unsqueeze(0).to(device))
                probs = torch.sigmoid(logits).squeeze().cpu().numpy()

                r = row_off - min_row
                c = col_off - min_col
                prob_sum[r:r + patch_size, c:c + patch_size] += probs * kernel
                weight_sum[r:r + patch_size, c:c + patch_size] += kernel
                gt_map[r:r + patch_size, c:c + patch_size] = mask.numpy()

        # Gaussian-weighted average of overlapping predictions
        prob_map = np.full((crop_h, crop_w), np.nan, dtype=np.float32)
        has_pred = weight_sum > 0
        prob_map[has_pred] = (prob_sum[has_pred] / weight_sum[has_pred]).astype(np.float32)

        if use_overlap:
            print(f"  Overlap: max weight {weight_sum.max():.2f}, "
                  f"mean weight {weight_sum[has_pred].mean():.2f}")

        # Compute georeferenced transform for the crop
        crop_transform = rasterio.transform.from_origin(
            raster_transform.c + min_col * raster_transform.a,
            raster_transform.f + min_row * raster_transform.e,
            abs(raster_transform.a),
            abs(raster_transform.e),
        )

        # Save probability map
        suffix = f"_s{inference_stride}" if use_overlap else ""
        prob_profile = raster_profile.copy()
        prob_profile.update(
            dtype="float32", count=1, nodata=np.nan,
            height=crop_h, width=crop_w, transform=crop_transform,
        )
        prob_path = figures_dir / f"{split_name}_pred{suffix}.tif"
        with rasterio.open(prob_path, "w", **prob_profile) as dst:
            dst.write(prob_map, 1)
        print(f"Saved {prob_path}")

        # Build and save error map
        _write_error_map(prob_map, vacancy_mask_path, crop_transform,
                         crop_h, crop_w, raster_profile, args.threshold,
                         figures_dir, split_name, suffix)


def _error_only(args, shared_root, run_dir, figures_dir, data_cfg,
                vacancy_mask_path, inference_stride, use_overlap):
    """Regenerate error maps from existing prediction TIFs (no model needed)."""
    suffix = f"_s{inference_stride}" if use_overlap else ""

    for split_name in args.splits:
        pred_path = figures_dir / f"{split_name}_pred{suffix}.tif"
        if not pred_path.exists():
            print(f"Skipping {split_name} — {pred_path} not found")
            continue

        print(f"\n=== {split_name}: reading {pred_path} ===")
        with rasterio.open(pred_path) as src:
            prob_map = src.read(1)  # (H, W) float32
            crop_transform = src.transform
            raster_profile = src.profile.copy()
            crop_h, crop_w = src.height, src.width

        _write_error_map(prob_map, vacancy_mask_path, crop_transform,
                         crop_h, crop_w, raster_profile, args.threshold,
                         figures_dir, split_name, suffix)


def _write_error_map(prob_map, vacancy_mask_path, crop_transform,
                     crop_h, crop_w, raster_profile, threshold,
                     figures_dir, split_name, suffix):
    """Build RGBA error map from prob_map + ground truth and write to disk."""
    # Read ground truth for the same spatial extent
    with rasterio.open(vacancy_mask_path) as src:
        gt_win = rasterio.windows.from_bounds(
            *rasterio.transform.array_bounds(crop_h, crop_w, crop_transform),
            transform=src.transform,
        )
        gt_win = gt_win.round_offsets().round_lengths()
        gt_map = src.read(1, window=gt_win,
                          out_shape=(crop_h, crop_w),
                          resampling=rasterio.enums.Resampling.nearest)

    has_pred = ~np.isnan(prob_map)
    pred_bin = prob_map > threshold
    valid = gt_map != 255
    mask = valid & has_pred

    gt_pos = gt_map == 1
    tp = mask & pred_bin & gt_pos
    fp = mask & pred_bin & ~gt_pos
    fn = mask & ~pred_bin & gt_pos
    tn = mask & ~pred_bin & ~gt_pos
    ignore = ~mask

    # RGBA: TP=green, FP=red, FN=blue, TN=black, ignore=white
    error_rgba = np.zeros((4, crop_h, crop_w), dtype=np.uint8)
    error_rgba[0][fp] = 255
    error_rgba[0][ignore] = 255
    error_rgba[1][tp] = 255
    error_rgba[1][ignore] = 255
    error_rgba[2][fn] = 255
    error_rgba[2][ignore] = 255
    error_rgba[3][mask] = 255
    error_rgba[3][ignore] = 128

    error_profile = raster_profile.copy()
    error_profile.update(
        dtype="uint8", count=4, nodata=None,
        height=crop_h, width=crop_w, transform=crop_transform,
    )
    error_path = figures_dir / f"{split_name}_error{suffix}.tif"
    with rasterio.open(error_path, "w", **error_profile) as dst:
        dst.write(error_rgba)
    print(f"Saved {error_path}")

    # Print summary
    n_tp = tp.sum()
    n_fp = fp.sum()
    n_fn = fn.sum()
    n_tn = tn.sum()
    iou = n_tp / max(n_tp + n_fp + n_fn, 1)
    prec = n_tp / max(n_tp + n_fp, 1)
    rec = n_tp / max(n_tp + n_fn, 1)
    print(f"  Threshold: {threshold}")
    print(f"  TP={n_tp:,}  FP={n_fp:,}  FN={n_fn:,}  TN={n_tn:,}")
    print(f"  IoU={iou:.4f}  Precision={prec:.4f}  Recall={rec:.4f}")


if __name__ == "__main__":
    main()
