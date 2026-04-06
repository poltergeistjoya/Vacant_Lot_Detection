"""
Extract 256×256 patch grid from the NAIP VRT + vacancy mask, assign
patches to train/val/test splits by borough, and save to JSON.

Output (relative to shared root):
  outputs/labels/patch_splits.json

Usage:
  uv run python scripts/extract_patches.py
  uv run python scripts/extract_patches.py --config config/data.yaml
"""
from __future__ import annotations

import argparse

from vacant_lot.config import load_data_config
from vacant_lot.dataset import generate_patch_grid, save_patch_splits, spatial_split_patches
from vacant_lot.logger import get_logger

log = get_logger()


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract patch grid and split by borough")
    parser.add_argument(
        "--config",
        default="data.yaml",
        help="Path to data config YAML (default: config/data.yaml)",
    )
    args = parser.parse_args()

    cfg = load_data_config(args.config)
    vacancy_mask_path = cfg.get_vacancy_mask_path()
    borough_mask_path = cfg.get_borough_mask_path()
    splits_path = cfg.get_patch_splits_path()

    patch_size = cfg.patch.size
    stride = cfg.patch.stride
    min_valid_pixels = cfg.patch.min_valid_pixels

    log.info(f"Vacancy mask:  {vacancy_mask_path}")
    log.info(f"Borough mask:  {borough_mask_path}")
    log.info(f"Output:        {splits_path}")
    log.info(f"Patch size:    {patch_size}, stride: {stride}, min_valid_pixels: {min_valid_pixels}")

    log.info("Generating patch grid...")
    patch_coords = generate_patch_grid(
        vacancy_mask_path,
        patch_size=patch_size,
        stride=stride,
        min_valid_pixels=min_valid_pixels,
    )
    log.info(f"Kept {len(patch_coords):,} patches")

    log.info("Assigning patches to splits...")
    splits = spatial_split_patches(
        patch_coords,
        borough_mask_path,
        cfg.split,
        patch_size=patch_size,
    )

    n_total = sum(len(v) for v in splits.values())
    n_excluded = len(patch_coords) - n_total
    for name, coords in splits.items():
        pct = len(coords) / len(patch_coords) * 100
        log.info(f"  {name:5s}: {len(coords):,} patches ({pct:.1f}%)")
    log.info(f"  excl.: {n_excluded:,} patches ({n_excluded / len(patch_coords) * 100:.1f}%)")

    train_set = set(splits["train"])
    val_set = set(splits["val"])
    test_set = set(splits["test"])
    assert not (train_set & val_set), "train/val overlap"
    assert not (train_set & test_set), "train/test overlap"
    assert not (val_set & test_set), "val/test overlap"

    cfg.ensure_labels_dir()
    save_patch_splits(
        splits,
        splits_path,
        patch_size=patch_size,
        stride=stride,
        min_valid_pixels=min_valid_pixels,
        split_cfg=cfg.split,
    )
    log.info(f"Saved to: {splits_path}")


if __name__ == "__main__":
    main()
