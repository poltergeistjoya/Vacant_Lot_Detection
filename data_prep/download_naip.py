"""
Download NAIP tiles from Microsoft Planetary Computer and build a GDAL VRT.

Outputs (relative to shared root):
  data/imagery/naip/nyc/2022/*.tif    — individual COG tiles
  data/imagery/naip/nyc/2022/naip_nyc_2022.vrt
  data/imagery/naip/nyc/2022/naip_nyc_2022.json

Idempotent — tiles that already exist on disk are skipped.

Usage:
  uv run python data_prep/download_naip.py
  uv run python data_prep/download_naip.py --config config/data.yaml
"""
from __future__ import annotations

import argparse

from vacant_lot.config import load_data_config
from vacant_lot.logger import get_logger
from vacant_lot.tile_export import build_naip_vrt, download_naip_tiles, query_naip_stac

log = get_logger()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download NAIP tiles and build VRT")
    parser.add_argument(
        "--config",
        default="data.yaml",
        help="Path to data config YAML (default: config/data.yaml)",
    )
    args = parser.parse_args()

    cfg = load_data_config(args.config)
    vrt_path = cfg.get_vrt_path()
    tiles_dir = vrt_path.parent

    bbox = cfg.segmentation.bbox
    if bbox is None:
        raise ValueError("segmentation.bbox must be set in data.yaml to query NAIP STAC")

    # Year: explicit config value takes priority; fall back to VRT filename (naip_nyc_2022.vrt → 2022)
    if cfg.segmentation.year is not None:
        year = cfg.segmentation.year
    else:
        year = int(vrt_path.stem.split("_")[-1])

    log.info(f"Tiles dir: {tiles_dir}")
    log.info(f"VRT:       {vrt_path}")
    log.info(f"bbox:      {bbox}, year: {year}")

    items = query_naip_stac(bbox=bbox, year=year)
    log.info(f"Found {len(items)} NAIP tiles")

    download_naip_tiles(items=items, local_dir=tiles_dir)

    build_naip_vrt(tiles_dir, vrt_path, exclude_dates=set(cfg.imagery.exclude_dates))
    log.info("Done.")


if __name__ == "__main__":
    main()
