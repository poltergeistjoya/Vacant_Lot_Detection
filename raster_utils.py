"""
Utilities for rasterizing vector parcels and creating parcel ID mappings.
"""
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds

from logger import get_logger

log = get_logger()


def rasterize_parcels(
    gdf: gpd.GeoDataFrame,
    id_column: str,
    raster_output_path: Path | str,
    mapping_output_path: Path | str,
    crs: str,
    resolution: float = 1.0,
    nodata: int = 0,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """
    Rasterize parcels and save ID mapping in one operation.

    Args:
        gdf: GeoDataFrame with parcel geometries.
        id_column: Column containing original parcel identifiers (e.g., BBL).
        raster_output_path: Path for output parcel raster GeoTIFF.
        mapping_output_path: Path for output ID mapping CSV.
        crs: Target CRS for the raster.
        resolution: Pixel size in CRS units (default: 1.0 meter for UTM).
        nodata: Value for pixels outside parcels (default: 0).
        overwrite: Whether to recreate raster if it already exists (default: False).

    Returns:
        Tuple of (raster_path, mapping_path).
    """
    raster_output_path = Path(raster_output_path)
    mapping_output_path = Path(mapping_output_path)

    if overwrite or not raster_output_path.exists() or not mapping_output_path.exists():
        raster_path, id_mapping = create_parcel_raster(
            gdf=gdf,
            id_column=id_column,
            output_path=raster_output_path,
            crs=crs,
            resolution=resolution,
            nodata=nodata,
        )

        # Save ID mapping
        mapping_output_path.parent.mkdir(parents=True, exist_ok=True)
        id_mapping.to_csv(mapping_output_path, index=False)
        log.info(f"Saved ID mapping to {mapping_output_path}")

    else:
        log.info(f"Outputs already exist (overwrite=False), skipping rasterization")
        raster_path=raster_output_path

    stats = get_parcel_raster_stats(raster_path)
    print(f"Parcel raster stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    return raster_path, mapping_output_path


def create_parcel_raster(
    gdf: gpd.GeoDataFrame,
    id_column: str,
    output_path: Path | str,
    crs: str,
    resolution: float = 1.0,
    nodata: int = 0,
) -> tuple[Path, pd.DataFrame]:
    """
    Rasterize parcels with pixel value = numeric parcel ID.

    Creates a GeoTIFF where each parcel polygon is filled with a unique
    integer ID. Also returns a mapping from numeric ID to original ID column.

    Args:
        gdf: GeoDataFrame with parcel geometries.
        id_column: Column containing original parcel identifiers (e.g., BBL).
        output_path: Path for output GeoTIFF.
        resolution: Pixel size in CRS units (default: 1.0 meter for UTM).
        crs: Target CRS for the raster (default: UTM 18N).
        nodata: Value for pixels outside parcels (default: 0).

    Returns:
        Tuple of (output_path, id_mapping_df) where id_mapping_df has columns
        ['parcel_id', id_column] mapping numeric IDs to original values.
    """
    output_path = Path(output_path)
    log.info(f"Creating parcel raster: {output_path}")
    log.info(f"Resolution: {resolution}m, CRS: {crs}")

    # Reproject if needed
    if gdf.crs is None:
        raise ValueError("GeoDataFrame must have a CRS defined")

    if str(gdf.crs) != crs:
        log.info(f"Reprojecting from {gdf.crs} to {crs}")
        gdf = gdf.to_crs(crs)

    # Create numeric parcel IDs starting from 1 (0 is nodata)
    gdf = gdf.copy()
    gdf["_parcel_id"] = range(1, len(gdf) + 1)

    # Build ID mapping
    id_mapping = gdf[["_parcel_id", id_column]].copy()
    id_mapping = id_mapping.rename(columns={"_parcel_id": "parcel_id"})

    # Calculate raster dimensions from bounds
    bounds = gdf.total_bounds  # minx, miny, maxx, maxy
    minx, miny, maxx, maxy = bounds

    width = int(np.ceil((maxx - minx) / resolution))
    height = int(np.ceil((maxy - miny) / resolution))

    log.info(f"Raster dimensions: {width} x {height} pixels")
    log.info(f"Bounds: {bounds}")

    # Create affine transform
    transform = from_bounds(minx, miny, maxx, maxy, width, height)

    # Generate (geometry, value) pairs for rasterization
    shapes = ((geom, val) for geom, val in zip(gdf.geometry, gdf["_parcel_id"]))

    # Rasterize
    log.info("Rasterizing parcels...")
    parcel_raster = rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=nodata,
        dtype=rasterio.uint32,  # Supports up to ~4 billion parcels
    )

    # Save as GeoTIFF with compression
    log.info(f"Saving to {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=rasterio.uint32,
        crs=crs,
        transform=transform,
        compress="lzw",
        nodata=nodata,
    ) as dst:
        dst.write(parcel_raster, 1)

    log.info(f"Parcel raster created with {len(gdf)} parcels")
    return output_path, id_mapping


def load_parcel_raster(raster_path: Path | str) -> tuple[np.ndarray, dict]:
    """
    Load a parcel raster and return the array with metadata.

    Args:
        raster_path: Path to the GeoTIFF.

    Returns:
        Tuple of (raster_array, metadata_dict).
    """
    raster_path = Path(raster_path)
    log.info(f"Loading parcel raster: {raster_path}")

    with rasterio.open(raster_path) as src:
        raster = src.read(1)
        meta = {
            "crs": src.crs,
            "transform": src.transform,
            "bounds": src.bounds,
            "width": src.width,
            "height": src.height,
            "nodata": src.nodata,
        }

    log.info(f"Loaded raster: {meta['width']}x{meta['height']}, CRS: {meta['crs']}")
    return raster, meta


def get_parcel_raster_stats(raster_path: Path | str) -> dict:
    """
    Get summary statistics about a parcel raster.

    Args:
        raster_path: Path to the parcel raster GeoTIFF.

    Returns:
        Dict with stats like unique_parcels, nodata_pixels, etc.
    """
    raster, meta = load_parcel_raster(raster_path)

    nodata = meta["nodata"] or 0
    valid_mask = raster != nodata

    unique_values = np.unique(raster[valid_mask])

    stats = {
        "total_pixels": raster.size,
        "valid_pixels": valid_mask.sum(),
        "nodata_pixels": (~valid_mask).sum(),
        "unique_parcels": len(unique_values),
        "min_parcel_id": int(unique_values.min()) if len(unique_values) > 0 else None,
        "max_parcel_id": int(unique_values.max()) if len(unique_values) > 0 else None,
        "crs": str(meta["crs"]),
        "bounds": meta["bounds"],
    }

    log.info(f"Parcel raster stats: {stats['unique_parcels']} unique parcels")
    return stats
