"""
Prepare pixel-level label masks from MapPLUTO parcel data.

Outputs (relative to shared root):
  outputs/labels/vacancy_mask.tif   — 0=non-vacant, 1=vacant, 255=ignore
  outputs/labels/borough_mask.tif   — 1-5 = BoroCode, 0=nodata

Usage:
  uv run python data_prep/prepare_labels.py
  uv run python data_prep/prepare_labels.py --erosion-pixels 0
"""
from __future__ import annotations

import argparse
from pathlib import Path

from vacant_lot.config import load_data_config
from vacant_lot.data_utils import load_gdb
from vacant_lot.label_utils import create_borough_mask, create_vacancy_mask
from vacant_lot.logger import get_logger

log = get_logger()


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare vacancy and borough masks")
    parser.add_argument(
        "--config",
        default="data.yaml",
        help="Path to data config YAML (default: config/data.yaml)",
    )
    parser.add_argument(
        "--erosion-pixels",
        type=int,
        default=None,
        help="Disk radius for boundary erosion (overrides labels.erosion_pixels in config)",
    )
    args = parser.parse_args()

    cfg = load_data_config(args.config)
    vrt_path = cfg.get_vrt_path()
    vacancy_mask_path = cfg.get_vacancy_mask_path()
    borough_mask_path = cfg.get_borough_mask_path()
    erosion_pixels = args.erosion_pixels if args.erosion_pixels is not None else cfg.labels.erosion_pixels

    cfg.ensure_labels_dir()

    log.info(f"VRT:          {vrt_path}")
    log.info(f"Vacancy mask: {vacancy_mask_path}")
    log.info(f"Borough mask: {borough_mask_path}")
    log.info(f"Erosion:      {erosion_pixels}px")

    gdb_path = cfg.get_parcel_path()
    log.info(f"Loading parcels from {gdb_path}")
    parcel_gdf = load_gdb(str(gdb_path), layer=cfg.parcels.layer)
    log.info(f"Loaded {len(parcel_gdf):,} parcels")

    omit_bbls = cfg.labels.omit_bbls
    parcel_gdf = parcel_gdf[~parcel_gdf[cfg.parcels.id_column].isin(omit_bbls)].copy()
    log.info(f"Parcels after omission: {len(parcel_gdf):,} ({len(omit_bbls)} omitted)")

    log.info("Creating vacancy mask...")
    create_vacancy_mask(
        parcel_gdf=parcel_gdf,
        cfg=cfg,
        reference_raster_path=vrt_path,
        output_path=vacancy_mask_path,
        erosion_pixels=erosion_pixels,
    )

    log.info("Creating borough mask...")
    create_borough_mask(
        reference_raster_path=vrt_path,
        output_path=borough_mask_path,
    )

    log.info("Done.")


if __name__ == "__main__":
    main()
