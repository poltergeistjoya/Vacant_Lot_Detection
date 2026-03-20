"""
Pixel-level label mask generation for segmentation training.

Converts MapPLUTO parcel-level vacancy labels into pixel masks
grid-aligned to a NAIP VRT reference raster.
"""
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window

from .config import CityConfig
from .logger import get_logger
from .modeling import build_labels

log = get_logger()

# Single source of truth for NYC borough configuration
BOROUGH_CONFIG = {
    1: {"name": "Manhattan", "county_fips": "061"},
    2: {"name": "Bronx", "county_fips": "005"},
    3: {"name": "Brooklyn", "county_fips": "047"},
    4: {"name": "Queens", "county_fips": "081"},
    5: {"name": "Staten Island", "county_fips": "085"},
}

# Derived mappings (auto-generated from BOROUGH_CONFIG)
BOROUGH_NAMES: dict[int, str] = {
    code: cfg["name"] for code, cfg in BOROUGH_CONFIG.items()
}
COUNTY_TO_BORO: dict[str, int] = {
    cfg["county_fips"]: code for code, cfg in BOROUGH_CONFIG.items()
}


def create_vacancy_mask(
    parcel_gdf: gpd.GeoDataFrame,
    cfg: CityConfig,
    reference_raster_path: Path | str,
    output_path: Path | str,
    erosion_pixels: int = 2,
    min_parcel_pixels: int = 50,
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

    n_vacant = int((mask == 1).sum())
    n_nonvacant = int((mask == 0).sum())
    n_ignore = int((mask == 255).sum())
    n_labelled = n_vacant + n_nonvacant
    vac_pct = n_vacant / n_labelled * 100 if n_labelled > 0 else 0.0
    log.info(
        f"Pixel counts — vacant: {n_vacant:,} ({vac_pct:.2f}%), "
        f"non-vacant: {n_nonvacant:,}, ignore: {n_ignore:,}"
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
    from skimage.morphology import dilation, disk

    footprint = disk(erosion_pixels)
    vacant = mask == 1
    nonvacant = mask == 0

    # Pixels within erosion_pixels of both a vacant and a non-vacant parcel
    class_boundary = dilation(vacant, footprint=footprint) & dilation(
        nonvacant, footprint=footprint
    )

    result = mask.copy()
    result[class_boundary & (mask == 1)] = 255
    return result


def create_borough_mask(
    reference_raster_path: Path | str,
    output_path: Path | str,
    state_fips: str = "36",
) -> Path:
    """
    Rasterize NYC borough boundaries from TIGER county data onto the VRT grid.

    Uses TIGER county boundaries so every pixel (roads, water, parks) gets a
    borough assignment — not just parcel pixels.

    Pixel values: 1=Manhattan, 2=Bronx, 3=Brooklyn, 4=Queens, 5=Staten Island.

    Args:
        reference_raster_path: Path to NAIP VRT — defines the output grid.
        output_path: Where to write the output uint8 GeoTIFF.
        state_fips: State FIPS code (default "36" = New York).

    Returns:
        Path to the written GeoTIFF.
    """
    import pygris

    reference_raster_path = Path(reference_raster_path)
    output_path = Path(output_path)

    with rasterio.open(reference_raster_path) as src:
        vrt_crs = src.crs
        vrt_transform = src.transform
        vrt_width = src.width
        vrt_height = src.height

    log.info("Fetching NYC county boundaries from TIGER")
    counties = pygris.counties(state=state_fips, cb=True, cache=True)
    nyc = counties[counties["COUNTYFP"].isin(COUNTY_TO_BORO)].copy()
    nyc["boro_code"] = nyc["COUNTYFP"].map(COUNTY_TO_BORO)
    nyc = nyc.to_crs(vrt_crs)

    shapes = (
        (geom, int(code))
        for geom, code in zip(nyc.geometry, nyc["boro_code"])
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


def analyze_borough_vacancy(
    vacancy_mask_path: Path,
    borough_mask_path: Path,
    block_size: int = 4096,
) -> pd.DataFrame:
    """
    Count vacant/non-vacant/ignore pixels per borough using block-wise reads.

    Both masks must share the same grid (CRS, transform, dimensions) — as
    produced by ``create_vacancy_mask`` and ``create_borough_mask`` against
    the same NAIP VRT reference raster.

    Args:
        vacancy_mask_path: GeoTIFF with values 0 (non-vacant), 1 (vacant), 255 (ignore).
        borough_mask_path: GeoTIFF with values 1-5 (borough codes), 0 (outside NYC).
        block_size: Pixel side length for block-wise reads.

    Returns:
        DataFrame with columns: Borough, Vacant Pixels, Non-Vacant Pixels,
        Ignore (255), Vacant Fraction.
    """
    vacancy_mask_path = Path(vacancy_mask_path)
    borough_mask_path = Path(borough_mask_path)

    counts: dict[int, dict[str, int]] = {
        code: {"vacant": 0, "nonvacant": 0, "ignore": 0}
        for code in BOROUGH_NAMES
    }

    with rasterio.open(vacancy_mask_path) as vac_src, rasterio.open(
        borough_mask_path
    ) as boro_src:
        height = vac_src.height
        width = vac_src.width

        for row_off in range(0, height, block_size):
            row_size = min(block_size, height - row_off)
            for col_off in range(0, width, block_size):
                col_size = min(block_size, width - col_off)
                win = Window(col_off, row_off, col_size, row_size)

                vac_block = vac_src.read(1, window=win)
                boro_block = boro_src.read(1, window=win)

                for code in BOROUGH_NAMES:
                    boro_mask = boro_block == code
                    if not boro_mask.any():
                        continue
                    vac_vals = vac_block[boro_mask]
                    counts[code]["vacant"] += int((vac_vals == 1).sum())
                    counts[code]["nonvacant"] += int((vac_vals == 0).sum())
                    counts[code]["ignore"] += int((vac_vals == 255).sum())

    rows = []
    for code, name in BOROUGH_NAMES.items():
        c = counts[code]
        labelled = c["vacant"] + c["nonvacant"]
        frac = c["vacant"] / labelled if labelled > 0 else 0.0
        rows.append(
            {
                "Borough": name,
                "Vacant Pixels": c["vacant"],
                "Non-Vacant Pixels": c["nonvacant"],
                "Ignore (255)": c["ignore"],
                "Vacant Fraction": frac,
            }
        )

    df = pd.DataFrame(rows)
    log.info("\n" + df.to_string(index=False))
    return df


def characterize_vacant_land_cover(
    vacancy_mask_path: Path | str,
    borough_mask_path: Path | str,
    land_cover_path: Path | str,
    land_cover_classes: dict[int, str],
    block_size: int = 4096,
) -> pd.DataFrame:
    """
    Cross-tabulate land cover classes for vacant pixels, per borough.

    For each block of the vacancy mask, the land cover raster is
    reprojected/resampled (nearest neighbour) to the vacancy mask grid.
    Only pixels where ``vacancy_mask == 1`` are counted.

    Args:
        vacancy_mask_path: GeoTIFF with values 0/1/255.
        borough_mask_path: GeoTIFF with borough codes 1-5.
        land_cover_path: Land cover GeoTIFF (any CRS/resolution — reprojected
            on the fly via nearest-neighbour resampling).
        land_cover_classes: ``{pixel_value: "class_name", ...}`` mapping.
        block_size: Pixel side length for block-wise reads.

    Returns:
        DataFrame with columns: Borough, one column per land cover class name,
        plus a Total column.  Values are pixel counts.
    """
    from rasterio.warp import reproject, Resampling

    vacancy_mask_path = Path(vacancy_mask_path)
    borough_mask_path = Path(borough_mask_path)
    land_cover_path = Path(land_cover_path)

    class_names = list(land_cover_classes.values())
    class_values = list(land_cover_classes.keys())

    # {boro_code: {class_name: count}}
    counts: dict[int, dict[str, int]] = {
        code: {cn: 0 for cn in class_names} for code in BOROUGH_NAMES
    }

    with (
        rasterio.open(vacancy_mask_path) as vac_src,
        rasterio.open(borough_mask_path) as boro_src,
        rasterio.open(land_cover_path) as lc_src,
    ):
        height = vac_src.height
        width = vac_src.width

        for row_off in range(0, height, block_size):
            row_size = min(block_size, height - row_off)
            for col_off in range(0, width, block_size):
                col_size = min(block_size, width - col_off)
                win = Window(col_off, row_off, col_size, row_size)

                vac_block = vac_src.read(1, window=win)

                # Skip blocks with no vacant pixels
                if not (vac_block == 1).any():
                    continue

                boro_block = boro_src.read(1, window=win)

                # Reproject land cover to this block's grid
                dst_transform = vac_src.window_transform(win)
                lc_block = np.empty((row_size, col_size), dtype=lc_src.dtypes[0])
                reproject(
                    source=rasterio.band(lc_src, 1),
                    destination=lc_block,
                    dst_transform=dst_transform,
                    dst_crs=vac_src.crs,
                    dst_nodata=255,
                    resampling=Resampling.nearest,
                )

                vacant_mask = vac_block == 1
                for code in BOROUGH_NAMES:
                    sel = vacant_mask & (boro_block == code)
                    if not sel.any():
                        continue
                    lc_vals = lc_block[sel]
                    for cv, cn in zip(class_values, class_names):
                        counts[code][cn] += int((lc_vals == cv).sum())

    rows = []
    for code, name in BOROUGH_NAMES.items():
        row: dict[str, object] = {"Borough": name}
        row.update(counts[code])
        row["Total"] = sum(counts[code].values())
        rows.append(row)

    df = pd.DataFrame(rows)
    log.info("\n" + df.to_string(index=False))
    return df
