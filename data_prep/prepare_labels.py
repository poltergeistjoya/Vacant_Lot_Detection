"""
Prepare pixel-level label masks from MapPLUTO parcel data.

Outputs (relative to shared root):
  outputs/labels/vacancy_mask.tif   — 0=non-vacant, 1=vacant, 255=ignore
  outputs/labels/borough_mask.tif   — 1-5 = BoroCode, 0=nodata

Usage:
  uv run python data_prep/prepare_labels.py
  uv run python data_prep/prepare_labels.py --erosion-pixels 0
  uv run python data_prep/prepare_labels.py \
      --output-vacancy-mask outputs/labels/vacancy_mask_v2.tif
"""
from __future__ import annotations

import argparse
from pathlib import Path

from vacant_lot.config import _get_shared_root, load_data_config
from vacant_lot.data_utils import load_gdb
from vacant_lot.label_utils import (
    analyze_borough_vacancy,
    create_borough_mask,
    create_vacancy_mask,
    load_nyc_roadbed_geometry,
    load_nyc_roads_geometry,
    load_nyc_water_geometry,
)
from vacant_lot.logger import get_logger

log = get_logger()


def stats(argv: list[str] | None = None) -> None:
    """Print per-borough vacancy stats for one or more masks."""
    parser = argparse.ArgumentParser(description="Per-borough vacancy stats for a mask")
    parser.add_argument(
        "masks",
        nargs="+",
        help="Path(s) to vacancy mask GeoTIFF(s) (relative to shared root or absolute)",
    )
    parser.add_argument(
        "--config",
        default="data.yaml",
        help="Path to data config YAML (default: config/data.yaml)",
    )
    args = parser.parse_args(argv)

    cfg = load_data_config(args.config)
    shared_root = _get_shared_root()
    borough_mask_path = cfg.get_borough_mask_path()

    for mask_arg in args.masks:
        p = Path(mask_arg)
        mask_path = p if p.is_absolute() else shared_root / p
        log.info(f"=== {mask_path.name} ===")
        analyze_borough_vacancy(mask_path, borough_mask_path)


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
    parser.add_argument(
        "--output-vacancy-mask",
        default=None,
        help=(
            "Override output path for the vacancy mask (relative to shared root or absolute). "
            "Use this to write a v2 mask alongside the existing one for diff/verification."
        ),
    )
    parser.add_argument(
        "--skip-borough-mask",
        action="store_true",
        help="Skip regenerating borough_mask.tif (it rarely changes — saves the pygris fetch)",
    )
    args = parser.parse_args()

    cfg = load_data_config(args.config)
    shared_root = _get_shared_root()
    vrt_path = cfg.get_vrt_path()
    borough_mask_path = cfg.get_borough_mask_path()
    erosion_pixels = args.erosion_pixels if args.erosion_pixels is not None else cfg.labels.erosion_pixels

    # Resolve output path: override or config default
    if args.output_vacancy_mask:
        override = Path(args.output_vacancy_mask)
        vacancy_mask_path = override if override.is_absolute() else shared_root / override
    else:
        vacancy_mask_path = cfg.get_vacancy_mask_path()

    old_mask_path = cfg.get_vacancy_mask_path()  # the one in config, for diff
    vacancy_mask_path.parent.mkdir(parents=True, exist_ok=True)

    log.info(f"VRT:              {vrt_path}")
    log.info(f"Vacancy mask OUT: {vacancy_mask_path}")
    log.info(f"Borough mask:     {borough_mask_path}")
    log.info(f"Erosion:          {erosion_pixels}px")

    gdb_path = cfg.get_parcel_path()
    log.info(f"Loading parcels from {gdb_path}")
    parcel_gdf = load_gdb(str(gdb_path), layer=cfg.parcels.layer)
    log.info(f"Loaded {len(parcel_gdf):,} parcels")

    omit_bbls = cfg.labels.omit_bbls
    parcel_gdf = parcel_gdf[~parcel_gdf[cfg.parcels.id_column].isin(omit_bbls)].copy()
    log.info(f"Parcels after omission: {len(parcel_gdf):,} ({len(omit_bbls)} omitted)")

    # Water / roads geometry (optional)
    water_gdf = None
    if cfg.labels.water_mask and cfg.labels.water_mask.enabled:
        cache_path = shared_root / cfg.labels.water_mask.cache
        water_gdf = load_nyc_water_geometry(cache_path)

    roads_gdf = None
    if cfg.labels.roads_mask and cfg.labels.roads_mask.enabled:
        cache_path = shared_root / cfg.labels.roads_mask.cache
        roads_gdf = load_nyc_roads_geometry(
            cache_path,
            mtfcc_classes=cfg.labels.roads_mask.road_mtfcc,
        )

    roadbed_gdf = None
    if cfg.labels.roadbed and cfg.labels.roadbed.enabled:
        roadbed_path = shared_root / cfg.labels.roadbed.path
        roadbed_gdf = load_nyc_roadbed_geometry(roadbed_path)

    log.info("Creating vacancy mask...")
    create_vacancy_mask(
        parcel_gdf=parcel_gdf,
        cfg=cfg,
        reference_raster_path=vrt_path,
        output_path=vacancy_mask_path,
        erosion_pixels=erosion_pixels,
        force_nonvacant_bbls=cfg.labels.force_nonvacant_bbls,
        force_vacant_bbls=cfg.labels.force_vacant_bbls,
        water_gdf=water_gdf,
        roads_gdf=roads_gdf,
        roadbed_gdf=roadbed_gdf,
    )

    if not args.skip_borough_mask:
        log.info("Creating borough mask...")
        create_borough_mask(
            reference_raster_path=vrt_path,
            output_path=borough_mask_path,
        )
    else:
        log.info("Skipping borough mask (--skip-borough-mask)")

    # Auto-diff: if we wrote to an override path and the config path exists, compare.
    if args.output_vacancy_mask and old_mask_path.exists() and old_mask_path != vacancy_mask_path:
        log.info("")
        log.info("=" * 72)
        log.info(f"Diff: old={old_mask_path.name}  vs  new={vacancy_mask_path.name}")
        log.info("=" * 72)
        log.info("Old mask per-borough counts:")
        old_df = analyze_borough_vacancy(old_mask_path, borough_mask_path)
        log.info("New mask per-borough counts:")
        new_df = analyze_borough_vacancy(vacancy_mask_path, borough_mask_path)

        diff = new_df.copy()
        for col in ["Vacant Pixels", "Non-Vacant Pixels", "Ignore (255)"]:
            diff[f"{col} Δ"] = new_df[col] - old_df[col]
        diff_cols = ["Borough", "Vacant Pixels Δ", "Non-Vacant Pixels Δ", "Ignore (255) Δ"]
        log.info("")
        log.info("Per-borough delta (new - old):")
        log.info("\n" + diff[diff_cols].to_string(index=False))

    log.info("Done.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "stats":
        sys.argv.pop(1)
        stats()
    else:
        main()
