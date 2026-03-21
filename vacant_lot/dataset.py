"""
Patch extraction, spatial borough split, and PyTorch Dataset for NAIP
segmentation training.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

from .config import SplitConfig
from .logger import get_logger

log = get_logger()


# ---------------------------------------------------------------------------
# Patch grid generation
# ---------------------------------------------------------------------------

def generate_patch_grid(
    vacancy_mask_path: Path | str,
    patch_size: int = 256,
    stride: int = 256,
    min_valid_fraction: float = 0.5,
) -> list[tuple[int, int]]:
    """
    Enumerate (row, col) patch origins on a regular grid, keeping only
    patches where the fraction of non-255 (labelled) pixels meets
    ``min_valid_fraction``.

    Args:
        vacancy_mask_path: Path to vacancy mask GeoTIFF (values 0, 1, 255).
        patch_size: Side length of square patches in pixels.
        stride: Step size between patch origins (== patch_size for no overlap).
        min_valid_fraction: Minimum fraction of non-ignore pixels to keep a patch.

    Returns:
        List of (row_offset, col_offset) tuples.
    """
    vacancy_mask_path = Path(vacancy_mask_path)
    coords: list[tuple[int, int]] = []

    with rasterio.open(vacancy_mask_path) as src:
        height = src.height
        width = src.width

        for row_off in range(0, height - patch_size + 1, stride):
            for col_off in range(0, width - patch_size + 1, stride):
                win = Window(col_off, row_off, patch_size, patch_size)
                block = src.read(1, window=win)
                valid_frac = (block != 255).sum() / block.size
                if valid_frac >= min_valid_fraction:
                    coords.append((row_off, col_off))

    log.info(
        f"Patch grid: {len(coords)} patches kept "
        f"(patch_size={patch_size}, stride={stride}, "
        f"min_valid_fraction={min_valid_fraction})"
    )
    return coords


# ---------------------------------------------------------------------------
# Spatial borough split
# ---------------------------------------------------------------------------

def spatial_split_patches(
    patch_coords: list[tuple[int, int]],
    borough_mask_path: Path | str,
    split_cfg: SplitConfig,
    patch_size: int = 256,
) -> dict[str, list[tuple[int, int]]]:
    """
    Assign patches to train/val/test splits by majority borough vote.

    For each patch, the borough mask is read and the most-common borough
    code determines the assignment.  Patches whose majority borough is in
    ``split_cfg.exclude_boroughs`` (or outside any borough, code 0) are
    dropped.

    Args:
        patch_coords: List of (row_offset, col_offset) from ``generate_patch_grid``.
        borough_mask_path: Borough mask GeoTIFF (values 1-5, 0 = outside).
        split_cfg: ``SplitConfig`` with borough lists per split.
        patch_size: Side length matching the grid used to generate ``patch_coords``.

    Returns:
        ``{"train": [...], "val": [...], "test": [...]}`` of patch coordinate lists.
    """
    borough_mask_path = Path(borough_mask_path)

    boro_to_split: dict[int, str] = {}
    for code in split_cfg.train_boroughs:
        boro_to_split[code] = "train"
    for code in split_cfg.val_boroughs:
        boro_to_split[code] = "val"
    for code in split_cfg.test_boroughs:
        boro_to_split[code] = "test"

    splits: dict[str, list[tuple[int, int]]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    n_excluded = 0

    with rasterio.open(borough_mask_path) as src:
        for row_off, col_off in patch_coords:
            win = Window(col_off, row_off, patch_size, patch_size)
            block = src.read(1, window=win)

            # Majority vote (exclude 0 = outside NYC)
            values, counts = np.unique(block[block > 0], return_counts=True)
            if len(values) == 0:
                n_excluded += 1
                continue
            majority_code = int(values[counts.argmax()])

            split_name = boro_to_split.get(majority_code)
            if split_name is None:
                n_excluded += 1
                continue
            splits[split_name].append((row_off, col_off))

    log.info(
        f"Spatial split: train={len(splits['train'])}, "
        f"val={len(splits['val'])}, test={len(splits['test'])}, "
        f"excluded={n_excluded}"
    )
    return splits


# ---------------------------------------------------------------------------
# Spectral indices (numpy, matching gee_utils formulas)
# ---------------------------------------------------------------------------

def compute_spectral_indices(rgbn: np.ndarray) -> np.ndarray:
    """
    Compute NDVI, SAVI, Brightness, and BareSoilProxy from NAIP bands.

    Replicates the GEE spectral index formulas in ``gee_utils.py`` but
    operates on numpy arrays.

    Input band order follows NAIP convention: R, G, B, NIR (uint8 0-255).

    Args:
        rgbn: (4, H, W) uint8 array — R, G, B, NIR bands.

    Returns:
        (4, H, W) float32 array — NDVI, SAVI, Brightness, BareSoilProxy.
        NDVI and SAVI are unit-normalized to [0, 1] to match GEE pipeline.
    """
    r, g, b, nir = rgbn.astype(np.float32) / 255.0

    # NDVI: (NIR - R) / (NIR + R), then unit-normalize to [0, 1]
    ndvi = (nir - r) / (nir + r + 1e-8)
    ndvi = (ndvi + 1.0) / 2.0  # scale [-1,1] → [0,1]

    # SAVI: (1 + L)(NIR - R) / (NIR + R + L), L = 0.5, clamp then unit-normalize
    L = 0.5
    savi = (1.0 + L) * (nir - r) / (nir + r + L)
    savi = np.clip(savi, -1.0, 1.0)
    savi = (savi + 1.0) / 2.0  # scale [-1,1] → [0,1]

    # Brightness: mean of visible bands (already in [0, 1])
    brightness = (r + g + b) / 3.0

    # BareSoilProxy: (1 - NDVI) * Brightness  (using unit-normalized NDVI)
    bare_soil = (1.0 - ndvi) * brightness

    return np.stack([ndvi, savi, brightness, bare_soil], axis=0)


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

try:
    import torch
    from torch.utils.data import Dataset as _TorchDataset

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    _TorchDataset = object  # type: ignore[assignment,misc]


class NAIPSegmentationDataset(_TorchDataset):
    """
    PyTorch Dataset that yields (image, mask) pairs from a NAIP VRT and
    a vacancy mask, using pre-computed patch coordinates.

    Each sample is:
        image: (8, patch_size, patch_size) float32
               — 4 NAIP bands (R,G,B,NIR scaled to [0,1]) + 4 spectral indices
        mask:  (patch_size, patch_size) int64
               — 0 = non-vacant, 1 = vacant, 255 = ignore

    Args:
        vrt_path: Path to NAIP VRT file.
        vacancy_mask_path: Path to vacancy mask GeoTIFF.
        patch_coords: List of (row_offset, col_offset) tuples.
        patch_size: Side length of square patches.
        transform: Optional albumentations transform (should accept
            ``image`` as (H, W, C) and ``mask`` as (H, W)).
    """

    def __init__(
        self,
        vrt_path: Path | str,
        vacancy_mask_path: Path | str,
        patch_coords: list[tuple[int, int]],
        patch_size: int = 256,
        transform=None,
    ):
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for NAIPSegmentationDataset. "
                "Install with: uv add torch"
            )
        self.vrt_path = Path(vrt_path)
        self.vacancy_mask_path = Path(vacancy_mask_path)
        self.patch_coords = patch_coords
        self.patch_size = patch_size
        self.transform = transform

    def __len__(self) -> int:
        return len(self.patch_coords)

    def __getitem__(self, idx: int) -> tuple:
        row_off, col_off = self.patch_coords[idx]
        win = Window(col_off, row_off, self.patch_size, self.patch_size)

        # Read NAIP bands (R, G, B, NIR) — shape (4, H, W) uint8
        with rasterio.open(self.vrt_path) as src:
            rgbn = src.read(window=win)  # (4, H, W)

        # Read vacancy mask — shape (H, W) uint8
        with rasterio.open(self.vacancy_mask_path) as src:
            mask = src.read(1, window=win)  # (H, W)

        # Compute spectral indices → (4, H, W) float32
        indices = compute_spectral_indices(rgbn)

        # Stack NAIP bands (scaled) + indices → (8, H, W)
        naip_scaled = rgbn.astype(np.float32) / 255.0
        image = np.concatenate([naip_scaled, indices], axis=0)  # (8, H, W)

        # Apply albumentations transform if provided
        if self.transform is not None:
            # albumentations expects (H, W, C) for image
            transformed = self.transform(
                image=image.transpose(1, 2, 0),
                mask=mask,
            )
            image = transformed["image"].transpose(2, 0, 1)
            mask = transformed["mask"]

        image_tensor = torch.from_numpy(image.copy())
        mask_tensor = torch.from_numpy(mask.astype(np.int64).copy())

        return image_tensor, mask_tensor
