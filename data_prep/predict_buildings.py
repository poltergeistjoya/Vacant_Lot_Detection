"""
Run pretrained UAGLNet building segmentation over the NYC NAIP VRT and write
a georeferenced building probability raster for use as an input feature to
the downstream vacant-lot segmentation model.

This is a one-time data-prep step. The output raster (uint8, [0, 254] =
prob * 254, 255 = nodata) is written to the same pixel grid as the vacancy
mask and covers the same patch extent as the training set — all 5 NYC
boroughs including Manhattan and Staten Island.

Loads UAGLNet from the vendored copy in vacant_lot/uaglnet.py with the
HuggingFace checkpoint ldxxx/UAGLNet_Inria, iterates 256x256 NAIP windows
over the patch grid produced by data_prep/extract_patches.py, upsamples each
window 2x to 512x512 for inference, and writes predictions back to a uint8
GeoTIFF aligned to the NAIP grid.

Usage:
  just data-prep::predict-buildings
  just data-prep::predict-buildings --batch-size 8
  just data-prep::predict-buildings --limit 16  # smoke test
  uv run python data_prep/predict_buildings.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from vacant_lot.building_inference_dataset import NAIPBuildingInferenceDataset
from vacant_lot.config import _get_shared_root, load_data_config
from vacant_lot.dataset import generate_patch_grid


def load_or_build_grid(
    vacancy_mask_path: Path,
    cache_path: Path,
    patch_size: int,
    stride: int,
    min_valid_pixels: int,
) -> list[tuple[int, int]]:
    """
    Load cached inference grid if present, otherwise generate and cache.

    Uses the same ``generate_patch_grid`` logic as ``data_prep/extract_patches.py``
    so the inference extent matches the training patch extent exactly — covering
    all 5 NYC boroughs (including Manhattan and Staten Island) while excluding
    water, NJ, and VRT padding.
    """
    cache_key = {
        "patch_size": patch_size,
        "stride": stride,
        "min_valid_pixels": min_valid_pixels,
    }
    if cache_path.exists():
        data = json.loads(cache_path.read_text())
        if all(data.get(k) == v for k, v in cache_key.items()):
            print(f"Loaded cached grid from {cache_path}: {len(data['coords'])} patches")
            return [tuple(c) for c in data["coords"]]
        print(f"Cached grid has mismatched params, regenerating")

    coords = generate_patch_grid(
        vacancy_mask_path,
        patch_size=patch_size,
        stride=stride,
        min_valid_pixels=min_valid_pixels,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps({**cache_key, "coords": [list(c) for c in coords]})
    )
    print(f"Cached grid to {cache_path}: {len(coords)} patches")
    return coords


def main() -> None:
    parser = argparse.ArgumentParser(description="Run UAGLNet building inference on NAIP")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="ldxxx/UAGLNet_Inria",
        help="HuggingFace checkpoint repo for UAGLNet",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/labels/building_pred.tif",
        help="Output GeoTIFF path (relative to shared root)",
    )
    parser.add_argument(
        "--grid-cache",
        type=str,
        default="outputs/labels/building_inference_grid.json",
        help="Cached full-NYC patch grid (relative to shared root)",
    )
    parser.add_argument("--patch-size", type=int, default=256, help="NAIP patch size")
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Inference stride in NAIP pixels. Defaults to patch_size // 2 (50%% "
             "overlap), which softens patch-edge seams via Hann-window blending. "
             "Set equal to --patch-size to disable overlap (faster, harder edges).",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="If set, only process the first N patches (for smoke testing).",
    )
    parser.add_argument(
        "--debug-dir",
        type=str,
        default=None,
        help="If set, save side-by-side input RGB + prob heatmap PNGs for the first "
             "--debug-limit patches to this dir (relative to shared root).",
    )
    parser.add_argument(
        "--debug-limit",
        type=int,
        default=16,
        help="Max number of debug PNGs to write when --debug-dir is set.",
    )
    parser.add_argument(
        "--building-class",
        type=int,
        default=0,
        help="Class index for building (Inria CLASSES = ('Building', 'Background'), so 0).",
    )
    args = parser.parse_args()
    if args.stride is None:
        args.stride = args.patch_size // 2
    assert 1 <= args.stride <= args.patch_size, "stride must be in [1, patch_size]"

    shared_root = _get_shared_root()
    output_path = shared_root / args.output
    grid_cache_path = shared_root / args.grid_cache
    debug_dir = (shared_root / args.debug_dir) if args.debug_dir else None
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        print(f"Debug PNGs will be written to {debug_dir} (limit={args.debug_limit})")

    # ImageNet stats for denormalizing debug previews (kept in sync with
    # building_inference_dataset.IMAGENET_MEAN/STD).
    imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

    # 1. Load UAGLNet from the vendored copy in vacant_lot/uaglnet.py
    from vacant_lot.uaglnet import UAGLNet

    device = torch.device(args.device)
    print(f"Loading UAGLNet from {args.checkpoint} on {device}")
    model = UAGLNet.from_pretrained(args.checkpoint).to(device).eval()

    # 2. Build inference grid — mirror the training patch extent.
    # Uses vacancy_mask.tif (burned from MapPLUTO across all 5 boroughs) as the
    # spatial filter, so we get Manhattan + Staten Island but skip water/NJ.
    data_cfg = load_data_config()
    vrt_path = data_cfg.get_vrt_path()
    vacancy_mask_path = data_cfg.get_vacancy_mask_path()
    print(f"NAIP VRT:     {vrt_path}")
    print(f"Vacancy mask: {vacancy_mask_path}")

    # Use args.stride (not data_cfg.patch.stride) — inference wants overlap for
    # Hann-window blending, which is independent of the training patch stride.
    coords = load_or_build_grid(
        vacancy_mask_path,
        grid_cache_path,
        patch_size=args.patch_size,
        stride=args.stride,
        min_valid_pixels=data_cfg.patch.min_valid_pixels,
    )
    if args.limit is not None:
        # Uniform stride sample so smoke tests span the whole VRT instead of
        # clustering in the first contiguous strip the scan finds.
        n_total = len(coords)
        if args.limit < n_total:
            idxs = np.linspace(0, n_total - 1, args.limit).astype(int)
            coords = [coords[i] for i in idxs]
        print(f"Limiting to {len(coords)} patches evenly sampled across {n_total} (smoke test)")

    dataset = NAIPBuildingInferenceDataset(
        vrt_path=vrt_path,
        patch_coords=coords,
        naip_patch_size=args.patch_size,
        uagl_patch_size=512,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=device.type == args.device,
    )

    # 3. Set up accumulator + weight arrays for Hann-window blended inference.
    # With stride < patch_size, patches overlap and we average their predictions
    # weighted by a 2D Hann window so patch-edge seams smooth out. We size the
    # accumulators to the bounding box of the coord set (NYC land only), not the
    # full VRT — saves ~2x memory by skipping water/NJ/padding.
    ps = args.patch_size
    rows_all = np.array([r for r, _ in coords], dtype=np.int64)
    cols_all = np.array([c for _, c in coords], dtype=np.int64)
    bbox_row0 = int(rows_all.min())
    bbox_col0 = int(cols_all.min())
    bbox_row1 = int(rows_all.max()) + ps
    bbox_col1 = int(cols_all.max()) + ps
    bbox_h = bbox_row1 - bbox_row0
    bbox_w = bbox_col1 - bbox_col0
    accum_bytes = bbox_h * bbox_w * 4 * 2  # float32 accum + float32 weight
    print(
        f"Accumulator bbox: {bbox_h}x{bbox_w} px, "
        f"stride={args.stride}, ~{accum_bytes / 1e9:.1f} GB RAM"
    )
    accum = np.zeros((bbox_h, bbox_w), dtype=np.float32)
    weight = np.zeros((bbox_h, bbox_w), dtype=np.float32)

    # 2D Hann window with a 0.05 floor. Edges-of-raster pixels still get a
    # non-zero contribution so the final divide is numerically stable, and
    # interior pixels are dominated by the high-weight center of each tile.
    hann_1d = np.hanning(ps + 2)[1:-1].astype(np.float32)  # drop exact zeros
    window = (hann_1d[:, None] * hann_1d[None, :]).astype(np.float32)
    window = np.maximum(window, 0.05)

    with rasterio.open(vrt_path) as src:
        H, W = src.height, src.width
        vrt_profile = src.profile.copy()
        vrt_transform = src.transform
        vrt_crs = src.crs

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Store probs as uint8 scaled to [0, 254]; 255 = nodata.
    # 1/254 ≈ 0.004 precision is finer than the model's actual confidence,
    # and this makes the file 4x smaller + 4x faster to render in QGIS.
    NODATA = 255
    out_profile = vrt_profile.copy()
    out_profile.update(
        driver="GTiff",
        dtype="uint8",
        count=1,
        nodata=NODATA,
        height=H,
        width=W,
        transform=vrt_transform,
        crs=vrt_crs,
        compress="deflate",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )

    n_processed = 0
    prob_sum_processed = 0.0

    print(f"Running inference on {len(dataset)} patches (batch={args.batch_size})")
    with torch.no_grad():
        for imgs, coords_batch in tqdm(loader, mininterval=5.0):
            imgs = imgs.to(device, non_blocking=True)
            logits = model(imgs)  # (B, 2, 512, 512)
            probs_512 = torch.softmax(logits, dim=1)[:, args.building_class]  # (B, 512, 512)
            probs_256 = (
                F.interpolate(
                    probs_512.unsqueeze(1),
                    size=ps,
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze(1)
                .cpu()
                .numpy()
            )  # (B, ps, ps)

            rows, cols = coords_batch
            imgs_cpu = imgs.detach().cpu().numpy() if debug_dir is not None else None
            probs_512_cpu = (
                probs_512.detach().cpu().numpy() if debug_dir is not None else None
            )
            for i, (prob, r, c) in enumerate(zip(probs_256, rows.tolist(), cols.tolist())):
                assert prob.min() >= 0.0 and prob.max() <= 1.0, (
                    f"Softmax output out of range: min={prob.min():.4f} max={prob.max():.4f}"
                )
                # Accumulate into the bbox-local frame with the Hann window
                rr = r - bbox_row0
                cc = c - bbox_col0
                accum[rr : rr + ps, cc : cc + ps] += prob * window
                weight[rr : rr + ps, cc : cc + ps] += window

                if debug_dir is not None and n_processed < args.debug_limit:
                    rgb_norm = imgs_cpu[i]
                    rgb = (rgb_norm * imagenet_std + imagenet_mean).clip(0, 1)
                    rgb_u8 = (rgb.transpose(1, 2, 0) * 255).astype(np.uint8)
                    prob_512 = probs_512_cpu[i]

                    import matplotlib.pyplot as plt
                    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                    axes[0].imshow(rgb_u8)
                    axes[0].set_title(f"NAIP RGB (row={r}, col={c})")
                    axes[0].axis("off")
                    im = axes[1].imshow(prob_512, cmap="magma", vmin=0, vmax=1)
                    axes[1].set_title(
                        f"Building prob (class={args.building_class}, mean={prob_512.mean():.3f})"
                    )
                    axes[1].axis("off")
                    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
                    fig.tight_layout()
                    fig.savefig(debug_dir / f"patch_{n_processed:04d}_r{r}_c{c}.png", dpi=100)
                    plt.close(fig)

                n_processed += 1
                prob_sum_processed += float(prob.mean())

    # Normalize accumulator by weight. weight==0 pixels (outside any patch)
    # become NODATA; everywhere else gets the blended mean probability.
    print("Normalizing accumulator and quantizing to uint8...")
    valid = weight > 0
    blended = np.zeros_like(accum)
    blended[valid] = accum[valid] / weight[valid]
    blended = np.clip(blended, 0.0, 1.0)
    blended_u8 = np.full_like(blended, NODATA, dtype=np.uint8)
    blended_u8[valid] = (blended[valid] * 254.0 + 0.5).astype(np.uint8)

    print(f"Writing {output_path}")
    with rasterio.open(output_path, "w", **out_profile) as dst:
        # Pre-fill every tile with nodata so unwritten regions (outside the
        # coord bbox — water, NJ, VRT padding) read back as NODATA instead of
        # uninitialized garbage.
        bx = out_profile["blockxsize"]
        by = out_profile["blockysize"]
        fill_tile = np.full((by, bx), NODATA, dtype=np.uint8)
        print("  Pre-filling output raster with nodata tiles...")
        for _, win in tqdm(list(dst.block_windows(1)), mininterval=5.0):
            tile = fill_tile[: win.height, : win.width]
            dst.write(tile, 1, window=win)

        # Write the blended result into the bbox window.
        dst.write(
            blended_u8,
            1,
            window=Window(bbox_col0, bbox_row0, bbox_w, bbox_h),
        )

    mean_prob = prob_sum_processed / n_processed if n_processed > 0 else float("nan")
    valid_pct = 100.0 * valid.sum() / (H * W)
    print(
        f"Coverage: {valid_pct:.1f}% of VRT pixels predicted "
        f"({n_processed} patches, stride={args.stride})"
    )
    print(f"Mean per-patch building probability: {mean_prob:.3f}")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
