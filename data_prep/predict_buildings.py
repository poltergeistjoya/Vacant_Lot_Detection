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

    coords = load_or_build_grid(
        vacancy_mask_path,
        grid_cache_path,
        patch_size=args.patch_size,
        stride=data_cfg.patch.stride,
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

    # 3. Open output GeoTIFF for streaming writes (no large in-memory buffers).
    # Non-overlapping patches (stride == patch_size) means each pixel is visited
    # exactly once, so we can write predictions directly as we go.
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

    n_written = 0
    prob_sum_written = 0.0

    print(f"Running inference on {len(dataset)} patches (batch={args.batch_size})")
    print(f"Writing predictions directly to {output_path} (no large RAM buffers)")
    with rasterio.open(output_path, "w", **out_profile) as dst:
        # Pre-fill every tile with nodata so unwritten regions read back as
        # NODATA (-1) instead of uninitialized float32 garbage (±3.4e38).
        # Tiled GTiff doesn't eagerly materialize blocks, so we write one
        # nodata tile per block window up front. Deflate-compresses to tiny.
        bx = out_profile["blockxsize"]
        by = out_profile["blockysize"]
        fill_tile = np.full((by, bx), NODATA, dtype=np.uint8)
        print("Pre-filling output raster with nodata tiles...")
        for _, win in tqdm(list(dst.block_windows(1))):
            # Trim fill to block size at raster edges
            tile = fill_tile[: win.height, : win.width]
            dst.write(tile, 1, window=win)

        with torch.no_grad():
            for imgs, coords_batch in tqdm(loader):
                imgs = imgs.to(device, non_blocking=True)
                logits = model(imgs)  # (B, 2, 512, 512)
                probs_512 = torch.softmax(logits, dim=1)[:, args.building_class]  # (B, 512, 512)
                probs_256 = (
                    F.interpolate(
                        probs_512.unsqueeze(1),
                        size=args.patch_size,
                        mode="bilinear",
                        align_corners=False,
                    )
                    .squeeze(1)
                    .cpu()
                    .numpy()
                )  # (B, 256, 256)

                rows, cols = coords_batch
                # Keep a CPU copy of the normalized inputs for debug previews
                imgs_cpu = imgs.detach().cpu().numpy() if debug_dir is not None else None
                probs_512_cpu = (
                    probs_512.detach().cpu().numpy() if debug_dir is not None else None
                )
                for i, (prob, r, c) in enumerate(zip(probs_256, rows.tolist(), cols.tolist())):
                    ps = args.patch_size
                    assert prob.min() >= 0.0 and prob.max() <= 1.0, (
                        f"Softmax output out of range: min={prob.min():.4f} max={prob.max():.4f}"
                    )
                    prob = np.clip(prob, 0.0, 1.0)
                    # Scale [0, 1] -> uint8 [0, 254]; 255 reserved for nodata.
                    prob_u8 = (prob * 254.0 + 0.5).astype(np.uint8)
                    dst.write(prob_u8, 1, window=Window(c, r, ps, ps))

                    if debug_dir is not None and n_written < args.debug_limit:
                        # Denormalize input back to uint8 RGB for visualization
                        rgb_norm = imgs_cpu[i]  # (3, 512, 512)
                        rgb = (rgb_norm * imagenet_std + imagenet_mean).clip(0, 1)
                        rgb_u8 = (rgb.transpose(1, 2, 0) * 255).astype(np.uint8)
                        prob_512 = probs_512_cpu[i]  # (512, 512), the native output

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
                        fig.savefig(debug_dir / f"patch_{n_written:04d}_r{r}_c{c}.png", dpi=100)
                        plt.close(fig)

                    n_written += 1
                    prob_sum_written += float(prob.mean())

    coverage_pct = 100.0 * n_written * args.patch_size ** 2 / (H * W)
    mean_prob = prob_sum_written / n_written if n_written > 0 else float("nan")
    print(f"Coverage: {coverage_pct:.1f}% of VRT pixels predicted ({n_written} patches)")
    print(f"Mean building probability: {mean_prob:.3f}")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
