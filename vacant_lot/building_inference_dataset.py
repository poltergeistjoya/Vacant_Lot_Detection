"""
Dataset and grid generation for running pretrained UAGLNet building
segmentation inference over a NAIP VRT.

UAGLNet expects (3, 512, 512) RGB inputs at 0.3 m/pixel, normalized with
ImageNet mean/std. NAIP is (4, H, W) RGBN at 0.6 m/pixel. Each NAIP 256x256
window covers the same ground area as a 512x512 UAGL input, so a single
2x bilinear upsample maps 1 NAIP patch -> 1 UAGL inference.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from rasterio.windows import Window
from torch.utils.data import Dataset

from .logger import get_logger

log = get_logger()

# ImageNet stats — match albumentations.Normalize() defaults that UAGLNet
# was trained with. See ~/repos/UAGLNet/geoseg/datasets/inria_dataset.py
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


class NAIPBuildingInferenceDataset(Dataset):
    """
    Yields UAGLNet-ready RGB patches from a NAIP VRT.

    Each sample is a 2-tuple ``(image, coords)``:

        image:  (3, 512, 512) float32 — RGB only, ImageNet-normalized,
                2x bilinear upsample of a 256x256 NAIP window
        coords: (row_off, col_off) int — original NAIP pixel offsets,
                used to write predictions back into the NAIP grid

    Args:
        vrt_path: Path to NAIP VRT (4 bands: R, G, B, NIR).
        patch_coords: List of (row_off, col_off) source patch origins.
        naip_patch_size: Source patch side length at NAIP resolution.
        uagl_patch_size: Target patch side length for UAGLNet input.
    """

    def __init__(
        self,
        vrt_path: Path | str,
        patch_coords: list[tuple[int, int]],
        naip_patch_size: int = 256,
        uagl_patch_size: int = 512,
    ):
        self.vrt_path = Path(vrt_path)
        self.patch_coords = patch_coords
        self.naip_patch_size = naip_patch_size
        self.uagl_patch_size = uagl_patch_size

    def __len__(self) -> int:
        return len(self.patch_coords)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, tuple[int, int]]:
        row_off, col_off = self.patch_coords[idx]
        win = Window(col_off, row_off, self.naip_patch_size, self.naip_patch_size)

        with rasterio.open(self.vrt_path) as src:
            # NAIP VRT band order is [R, G, B, NIR] (1-indexed in rasterio).
            # Documented in vacant_lot/dataset.py:215 and :363.
            # Pass an explicit band list — NIR (band 4) is never loaded.
            rgb = src.read([1, 2, 3], window=win)  # (3, 256, 256) uint8

        # uint8 [0, 255] -> float32 [0, 1]
        rgb = rgb.astype(np.float32) / 255.0

        # 2x bilinear upsample: (3, 256, 256) -> (3, 512, 512)
        rgb_t = torch.from_numpy(rgb).unsqueeze(0)  # (1, 3, 256, 256)
        rgb_up = (
            F.interpolate(
                rgb_t,
                size=(self.uagl_patch_size, self.uagl_patch_size),
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(0)
            .numpy()
        )  # (3, 512, 512)

        # ImageNet normalize (matches albu.Normalize defaults UAGLNet trained with)
        rgb_norm = (rgb_up - IMAGENET_MEAN) / IMAGENET_STD

        return torch.from_numpy(rgb_norm), (row_off, col_off)
