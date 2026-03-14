"""
Pixel-level label mask generation for segmentation training.

Converts MapPLUTO parcel-level vacancy labels into pixel masks
grid-aligned to a NAIP VRT reference raster.
"""
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize

from .config import CityConfig
from .logger import get_logger
from .modeling import build_labels

log = get_logger()


def create_vacancy_mask(
    parcel_gdf: gpd.GeoDataFrame,
    cfg: CityConfig,
    reference_raster_path: Path | str,
    output_path: Path | str,
    erosion_pixels: int = 2,
    min_parcel_pixels: int = 25,
) -> Path:
    """
    Rasterize parcel vacancy labels onto the VRT reference grid.

    Pixel values: 1 = vacant, 0 = non-vacant, 255 = nodata/ignore.
    Boundary pixels (within erosion_pixels of any parcel edge) are set to 255.
    Parcels smaller than min_parcel_pixels are excluded (set to 255).

    Args:
        parcel_gdf: Full MapPLUTO GeoDataFrame (all parcels, no sampling).
        cfg: CityConfig with parcel.landuse_column and parcel.vacant_codes.
        reference_raster_path: Path to NAIP VRT — defines the output grid.
        output_path: Where to write the output uint8 GeoTIFF.
        erosion_pixels: Disk radius for boundary erosion (0 = skip).
        min_parcel_pixels: Parcels with area < this many pixels are excluded
            from training (set to 255). Default: 25 (~5x5 px at 1m resolution).

    Returns:
        Path to the written GeoTIFF.
    """
    reference_raster_path = Path(reference_raster_path)
    output_path = Path(output_path)

    with rasterio.open(reference_raster_path) as src:
        vrt_crs = src.crs
        vrt_transform = src.transform
        vrt_width = src.width
        vrt_height = src.height

    log.info(f"Reference grid: {vrt_width}x{vrt_height}, CRS: {vrt_crs}")

    gdf = parcel_gdf.to_crs(vrt_crs)

    pixel_area = abs(vrt_transform.a * vrt_transform.e)  # m² per pixel
    min_area_m2 = min_parcel_pixels * pixel_area
    small_mask = gdf.geometry.area < min_area_m2
    n_small = small_mask.sum()
    if n_small > 0:
        log.info(
            f"Excluding {n_small} parcels smaller than {min_parcel_pixels} pixels "
            f"({min_area_m2:.1f} m²)"
        )
    gdf = gdf[~small_mask].copy()

    labels = build_labels(gdf, cfg)

    # Pass 1: burn all parcels = 0 (background stays 255 = nodata)
    all_shapes = ((geom, 0) for geom in gdf.geometry if geom is not None)
    mask = rasterize(
        all_shapes,
        out_shape=(vrt_height, vrt_width),
        transform=vrt_transform,
        fill=255,
        dtype=np.uint8,
    )

    # Pass 2: burn vacant parcels = 1 (overwrites their 0s)
    vacant_idx = labels[labels == 1].index
    vacant_gdf = gdf.loc[vacant_idx]
    if len(vacant_gdf) > 0:
        vacant_shapes = (
            (geom, 1) for geom in vacant_gdf.geometry if geom is not None
        )
        rasterize(
            vacant_shapes,
            out_shape=(vrt_height, vrt_width),
            transform=vrt_transform,
            fill=255,
            dtype=np.uint8,
            out=mask,
        )

    if erosion_pixels > 0:
        mask = erode_label_mask(mask, erosion_pixels)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=vrt_height,
        width=vrt_width,
        count=1,
        dtype=np.uint8,
        crs=vrt_crs,
        transform=vrt_transform,
        compress="lzw",
        nodata=255,
    ) as dst:
        dst.write(mask, 1)

    log.info(f"Vacancy mask written: {output_path}")
    return output_path


def erode_label_mask(mask: np.ndarray, erosion_pixels: int = 2) -> np.ndarray:
    """
    Set pixels near class-transition boundaries to 255 (ignore).

    Only erodes at 0↔1 transitions (vacant/non-vacant borders). Shared borders
    between two vacant parcels or two non-vacant parcels are left untouched.

    Args:
        mask: uint8 array with values 0, 1, or 255 (nodata).
        erosion_pixels: Disk radius for dilation footprint.

    Returns:
        Modified mask array with class-boundary pixels set to 255.
    """
    from skimage.morphology import binary_dilation, disk

    footprint = disk(erosion_pixels)
    vacant = mask == 1
    nonvacant = mask == 0

    # Pixels within erosion_pixels of both a vacant and a non-vacant parcel
    class_boundary = binary_dilation(vacant, footprint=footprint) & binary_dilation(
        nonvacant, footprint=footprint
    )

    result = mask.copy()
    result[class_boundary] = 255
    return result


def create_borough_mask(
    parcel_gdf: gpd.GeoDataFrame,
    reference_raster_path: Path | str,
    output_path: Path | str,
) -> Path:
    """
    Rasterize BoroCode values onto the VRT reference grid.

    Pixel values: 1-5 = borough code, 0 = nodata (outside all parcels).

    Args:
        parcel_gdf: Full MapPLUTO GeoDataFrame (must have 'BoroCode' column).
        reference_raster_path: Path to NAIP VRT — defines the output grid.
        output_path: Where to write the output uint8 GeoTIFF.

    Returns:
        Path to the written GeoTIFF.
    """
    reference_raster_path = Path(reference_raster_path)
    output_path = Path(output_path)

    with rasterio.open(reference_raster_path) as src:
        vrt_crs = src.crs
        vrt_transform = src.transform
        vrt_width = src.width
        vrt_height = src.height

    gdf = parcel_gdf.to_crs(vrt_crs)
    shapes = (
        (geom, int(boro_code))
        for geom, boro_code in zip(gdf.geometry, gdf["BoroCode"])
        if geom is not None
    )
    boro_mask = rasterize(
        shapes,
        out_shape=(vrt_height, vrt_width),
        transform=vrt_transform,
        fill=0,
        dtype=np.uint8,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=vrt_height,
        width=vrt_width,
        count=1,
        dtype=np.uint8,
        crs=vrt_crs,
        transform=vrt_transform,
        compress="lzw",
        nodata=0,
    ) as dst:
        dst.write(boro_mask, 1)

    log.info(f"Borough mask written: {output_path}")
    return output_path
