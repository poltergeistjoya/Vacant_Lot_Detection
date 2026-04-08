"""
Sanity-check the NAIP -> UAGLNet input transform before running full inference.

Dumps a few randomly-sampled NAIP patches as multi-panel PNGs covering:
  1. Raw NAIP RGB (256x256 uint8) — should look like a normal aerial photo
  2. Raw NAIP NIR (256x256 grayscale) — vegetation should be bright
  3. Upsampled RGB (512x512) — smoother version of panel 1
  4. ImageNet-normalized then un-normalized — should match panel 3
  5. Per-channel histograms after normalization — mean ~ 0, std ~ 1

Usage:
  just data-prep::visualize-building-inputs
  just data-prep::visualize-building-inputs --num-samples 6 --seed 42
  uv run python data_prep/visualize_building_inputs.py
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.windows import Window

from vacant_lot.building_inference_dataset import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    NAIPBuildingInferenceDataset,
)
from vacant_lot.config import _get_shared_root, load_data_config
from vacant_lot.dataset import generate_patch_grid


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity-check building inference inputs")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/figures/building_debug",
        help="Directory to write debug PNGs (relative to shared root)",
    )
    parser.add_argument("--num-samples", type=int, default=6)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--visualize-only",
        action="store_true",
        help="Skip grid generation and reuse the cached grid from predict_buildings.py. "
        "Fails if the cache is missing.",
    )
    parser.add_argument(
        "--grid-cache",
        type=str,
        default="outputs/labels/building_inference_grid.json",
        help="Cached grid path (relative to shared root), shared with predict_buildings.py",
    )
    args = parser.parse_args()

    shared_root = _get_shared_root()
    output_dir = shared_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = load_data_config()
    vrt_path = data_cfg.get_vrt_path()
    vacancy_mask_path = data_cfg.get_vacancy_mask_path()
    stride = data_cfg.patch.stride
    min_valid_pixels = data_cfg.patch.min_valid_pixels
    print(f"NAIP VRT:     {vrt_path}")
    print(f"Vacancy mask: {vacancy_mask_path}")
    print(f"Output dir:   {output_dir}")

    # Grid source: cached if --visualize-only, otherwise generate fresh from
    # the vacancy mask (same extent as training patches — all 5 boroughs).
    cache_key = {
        "patch_size": args.patch_size,
        "stride": stride,
        "min_valid_pixels": min_valid_pixels,
    }
    grid_cache_path = shared_root / args.grid_cache
    if args.visualize_only:
        if not grid_cache_path.exists():
            raise FileNotFoundError(
                f"--visualize-only set but no cached grid at {grid_cache_path}. "
                "Run predict_buildings.py (or this script without --visualize-only) first."
            )
        data = json.loads(grid_cache_path.read_text())
        for k, v in cache_key.items():
            if data.get(k) != v:
                raise ValueError(
                    f"Cached grid {k}={data.get(k)} != expected {v}"
                )
        coords = [tuple(c) for c in data["coords"]]
        print(f"Loaded cached grid from {grid_cache_path}: {len(coords)} patches")
    else:
        coords = generate_patch_grid(
            vacancy_mask_path,
            patch_size=args.patch_size,
            stride=stride,
            min_valid_pixels=min_valid_pixels,
        )
        grid_cache_path.parent.mkdir(parents=True, exist_ok=True)
        grid_cache_path.write_text(
            json.dumps({**cache_key, "coords": [list(c) for c in coords]})
        )
        print(f"Cached grid to {grid_cache_path}: {len(coords)} patches")

    rng = random.Random(args.seed)
    sample_coords = rng.sample(coords, min(args.num_samples, len(coords)))

    dataset = NAIPBuildingInferenceDataset(
        vrt_path=vrt_path,
        patch_coords=sample_coords,
        naip_patch_size=args.patch_size,
        uagl_patch_size=512,
    )

    for i, (row_off, col_off) in enumerate(sample_coords):
        # Re-read raw bands (incl. NIR) for the visualization
        with rasterio.open(vrt_path) as src:
            win = Window(col_off, row_off, args.patch_size, args.patch_size)
            rgbn = src.read(window=win)  # (4, H, W) uint8
        rgb_raw = rgbn[:3].transpose(1, 2, 0)  # (H, W, 3)
        nir = rgbn[3]  # (H, W)

        # Pull the dataset's normalized tensor for this patch
        norm_tensor, _ = dataset[i]
        rgb_norm = norm_tensor.numpy()  # (3, 512, 512)

        # Un-normalize round-trip — should match the upsampled raw RGB
        rgb_recovered = rgb_norm * IMAGENET_STD + IMAGENET_MEAN  # (3, 512, 512)
        rgb_recovered = np.clip(rgb_recovered, 0, 1).transpose(1, 2, 0)

        # Reconstruct the upsampled (pre-normalize) view from the dataset path
        # by re-running the same transform without normalization for panel 3.
        import torch
        import torch.nn.functional as F
        rgb_t = torch.from_numpy(rgb_raw.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        rgb_up = F.interpolate(rgb_t, size=(512, 512), mode="bilinear", align_corners=False)
        rgb_up = rgb_up.squeeze(0).permute(1, 2, 0).numpy()

        # Build the figure
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 4)

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(rgb_raw)
        ax1.set_title(f"1. Raw NAIP RGB\n{rgb_raw.shape} uint8")
        ax1.axis("off")

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(nir, cmap="gray")
        ax2.set_title(f"2. Raw NAIP NIR\n{nir.shape} uint8")
        ax2.axis("off")

        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(rgb_up)
        ax3.set_title(f"3. Upsampled RGB\n{rgb_up.shape}")
        ax3.axis("off")

        ax4 = fig.add_subplot(gs[0, 3])
        ax4.imshow(rgb_recovered)
        ax4.set_title("4. Normalized -> un-normalized\n(should match panel 3)")
        ax4.axis("off")

        # Panel 5: per-channel histograms after normalization
        ax5 = fig.add_subplot(gs[1, :])
        colors = ["red", "green", "blue"]
        names = ["R", "G", "B"]
        for c in range(3):
            ax5.hist(
                rgb_norm[c].ravel(),
                bins=80,
                alpha=0.5,
                color=colors[c],
                label=f"{names[c]}: mean={rgb_norm[c].mean():.3f}, std={rgb_norm[c].std():.3f}",
            )
        ax5.set_title("5. Per-channel histograms after ImageNet normalization (target: mean ~ 0, std ~ 1)")
        ax5.legend()
        ax5.set_xlabel("Normalized value")
        ax5.set_ylabel("Pixel count")

        fig.suptitle(f"Patch {i}: row={row_off}, col={col_off}", fontsize=14)
        fig.tight_layout()

        out_path = output_dir / f"patch_{i:02d}_r{row_off}_c{col_off}.png"
        fig.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {out_path}")

    print(f"\nDone. Inspect {output_dir} before running predict_buildings.py.")


if __name__ == "__main__":
    main()
