"""
Export patch grid as a GeoPackage for QGIS visualization.

Reads patch_splits.json (written by 02_extract_patches.ipynb) and
writes patch outlines as rectangles to patch_grid.gpkg, color-coded
by split (train/val/test).

Usage:
    uv run python scripts/export_patch_grid.py
"""
import geopandas as gpd
import rasterio
from shapely.geometry import box

from vacant_lot.config import load_config
from vacant_lot.dataset import load_patch_splits
from vacant_lot.logger import get_logger

log = get_logger()


def main() -> None:
    cfg = load_config("nyc_buildings.yaml")
    masks_dir = cfg.get_seg_masks_dir()

    splits_path = masks_dir / "patch_splits.json"
    vacancy_mask_path = masks_dir / "vacancy_mask.tif"
    gpkg_path = masks_dir / "patch_grid.gpkg"

    splits = load_patch_splits(splits_path)
    patch_size = 256  # read from JSON if needed

    # Read patch_size from the JSON directly
    import json
    meta = json.loads(splits_path.read_text())
    patch_size = meta["patch_size"]

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
