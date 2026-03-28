"""
Export patch grid as a GeoPackage for QGIS visualization.

Reads patch_splits.json and writes patch outlines as rectangles to
patch_grid.gpkg, with a 'split' column (train/val/test) for color-coding.

Output: outputs/labels/patch_grid.gpkg

Usage:
  uv run python data_prep/export_patch_grid.py
"""
from __future__ import annotations

import argparse
import json

import geopandas as gpd
import rasterio
from shapely.geometry import box

from vacant_lot.config import load_data_config
from vacant_lot.dataset import load_patch_splits
from vacant_lot.logger import get_logger

log = get_logger()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export patch grid to GeoPackage")
    parser.add_argument("--config", default="data.yaml")
    args = parser.parse_args()

    cfg = load_data_config(args.config)
    splits_path = cfg.get_patch_splits_path()
    vacancy_mask_path = cfg.get_vacancy_mask_path()
    gpkg_path = cfg.get_labels_dir() / "patch_grid.gpkg"

    splits = load_patch_splits(splits_path)
    patch_size = json.loads(splits_path.read_text())["patch_size"]

    with rasterio.open(vacancy_mask_path) as src:
        transform = src.transform
        crs = src.crs

    rows = []
    for split_name, coords in splits.items():
        for row_off, col_off in coords:
            x_min, y_max = transform * (col_off, row_off)
            x_max, y_min = transform * (col_off + patch_size, row_off + patch_size)
            rows.append({
                "split": split_name,
                "row_off": row_off,
                "col_off": col_off,
                "geometry": box(x_min, y_min, x_max, y_max),
            })

    gdf = gpd.GeoDataFrame(rows, crs=crs)
    gdf.to_file(gpkg_path, driver="GPKG", layer="patch_grid")

    log.info(f"Written: {gpkg_path}")
    for split_name, count in gdf.groupby("split").size().items():
        log.info(f"  {split_name}: {count:,} patches")


if __name__ == "__main__":
    main()
