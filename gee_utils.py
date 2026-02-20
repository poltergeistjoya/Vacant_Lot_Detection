import ee
import re
import zipfile
from typing import Optional, List
from pathlib import Path


from logger import get_logger
from config import GCPConfig
from data_utils import upload_to_gcs


log = get_logger()


def upload_raster_to_gcs(
    raster_path: Path | str,
    gcs_bucket: str,
    gcs_prefix: str,
    gcs_filename: str = "parcels_raster.tif",
) -> str:
    """
    Upload a local raster to Google Cloud Storage.

    Args:
        raster_path: Local path to GeoTIFF.
        gcs_bucket: GCS bucket name.
        gcs_prefix: Path prefix in bucket (e.g., 'nyc').
        gcs_filename: Filename in GCS (default: parcels_raster.tif).

    Returns:
        Full GCS URI (gs://bucket/prefix/filename).
    """
    from google.cloud import storage

    raster_path = Path(raster_path)
    if not raster_path.exists():
        raise FileNotFoundError(f"Raster not found: {raster_path}")

    client = storage.Client()
    bucket = client.bucket(gcs_bucket)
    gcs_path = f"{gcs_prefix}/{gcs_filename}"
    blob = bucket.blob(gcs_path)

    log.info(f"Uploading {raster_path} to gs://{gcs_bucket}/{gcs_path}")
    blob.upload_from_filename(raster_path)
    log.info("Raster upload to GCS complete")

    return f"gs://{gcs_bucket}/{gcs_path}"


def ingest_raster_to_gee(
    gcs_uri: str,
    asset_id: str,
    description: str = "parcel_raster_ingestion",
) -> dict:
    """
    Ingest a raster from GCS to GEE as an Image asset.

    Args:
        gcs_uri: Full GCS URI (gs://bucket/path/file.tif).
        asset_id: Destination GEE asset ID (e.g., projects/project-id/assets/name).
        description: Task description.

    Returns:
        Dict with asset_id, gcs_uri, and task info.
    """
    # https://developers.google.com/earth-engine/guides/image_manifest 
    log.info(f"Ingesting raster from {gcs_uri} to GEE asset: {asset_id}")

    manifest = {
    "name": asset_id,
    "tilesets": [
        {
        "sources": [
            {
            "uris": [
                gcs_uri
            ]
            }
        ]
        }
    ]
    }

    task = ee.data.startIngestion(None, manifest)

    log.info(f"Ingestion task started: {task['id']}")

    return {
        "asset_id": asset_id,
        "gcs_uri": gcs_uri,
        "task_id": task["id"],
    }

def init_gee(config: GCPConfig):
    """
    Initialize GEE w project-scoped credentials.
    Assumes ADC (Application Default Credentials) are set via `gcloud auth application-default login`
    """

    project_id = config.project_id
    log.info(f"Initializing GEE with ADC credentials and project:{project_id}")
    try:
        ee.Initialize(project=project_id)
        log.info("GEE sucessfully initialized")
    except Exception as e:
        log.error(f"Failed to intialize GEE: {e}")
        raise

def export_image_to_gee(
        image: ee.Image,
        description: str, 
        asset_id: str, 
        project_id: str, 
        region: Optional[ee.Geometry] = None, 
        scale: int =1,
        max_pixels: int = 1e13,
):
    """
    Export ee.Image to Earth Engine Asset

    Args:
        image (ee.Image): Image to export.
        description (str): task description.
        asset_id (str): Destination asset path (e.g. 'tmp/naip_2022_nyc').
        region (ee.Geometry): Region to clip/export (use small region for testing).
        scale (int): Pixel resolution (default: 1 m for NAIP).
        max_pixels (int): Max pixels allowed in export.

    Returns: 
        ee.batch.Task: The Earth Engine export task object.
    """

    # --- validate asset_id ---
    expected_prefix = f"projects/{project_id}/assets/"
    pattern = rf"^{re.escape(expected_prefix)}"
    assert re.match(pattern, asset_id), (
        f"asset_id must start with '{expected_prefix}', got '{asset_id}'"
    )

    log.info(f" Starting export to GEE Asset: {asset_id}")

    params = dict(
        image=image,
        description=description,
        assetId=asset_id,
        scale=scale,
        maxPixels=max_pixels,
    )

    if region is not None:
        params["region"] = region

    task = ee.batch.Export.image.toAsset(**params)

    task.start()
    log.info("Export task started. Monitor progress in Earth Engine Code Editor Tasks tab")

    return task

def load_image_collection(
        collection_id: str,
        start_date: str, 
        end_date:str,
        region: Optional[ee.Geometry] = None, 
        mosaic: bool = True
):
    """
    Load and optionally mosaic an Earth Engine ImageCollection
    filtered by date range and region.

    Args:
        collection_id: Earth Engine ImageCollection ID 
                       (e.g., 'USDA/NAIP/DOQQ', 'COPERNICUS/S2_SR')
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        region: Optional ee.Geometry to spatially filter
        mosaic: Whether to mosaic the filtered collection into a single ee.Image

    Returns:
        ee.Image or ee.ImageCollection:
            - ee.Image if mosaic=True
            - ee.ImageCollection otherwise
    """
    log.info(f"Loading ImageCollection: {collection_id}")
    log.info(f"Date range: {start_date} → {end_date}")

    if region:
        log.info("FilterBounds by region")
        collection = ee.ImageCollection(collection_id).filterDate(start_date, end_date).filterBounds(region)
    else: 
        collection = ee.ImageCollection(collection_id).filterDate(start_date, end_date)

    size = collection.size().getInfo()
    log.info(f"Filtered collection contains {size} images")

    if mosaic:
        log.info("Creating mosaic from filtered collection")
        image = collection.mosaic()
        log.info("Mosaic created successfully")
        if region:
            # Only clip mosaiced images for memory reasons
            image = image.clip(region)
            log.info("Clipping mosaiced image")
        return image
    
    log.info("Returning filtered ImageCollection (not mosaiced)")
    return collection

def scale_bands_to_unit(image: ee.Image, band_names: list, scale: float = 255.0):
    """
    Returns an image where the specified bands are scaled to [0,1].
    Leaves other bands untouched.
    """
    log.info(f"scaling {band_names} by {scale}")
    scaled = image.select(band_names).divide(scale)
    scaled = scaled.rename(band_names)
    
    # Replace existing bands with scaled ones
    return image.addBands(scaled, overwrite=True)

def calculate_ndvi(image:ee.Image, near_infrared: str, red: str, unit_normalization=True):
    """
    Calculate Normalized Difference Vegetation Index
    NDVI: (NIR - Red) / (NIR + Red) -> [-1, 1]
        low reflectance in both bands (water)    -> -1
        NIR >(slightly) Red (bare soil/dry land) -> 0
        Red close to 0 (dense vegetation)        -> 1
    
    Args: 
        image: ee.Image 
        near_infrared: str near_infrared band name 
        red: str red band name 

    Returns:
        ndvi: ee.Image with 1 band 'NDVI'
    """
    log.info("calculating NDVI")
    ndvi = image.normalizedDifference([near_infrared, red]).rename('NDVI')

    if unit_normalization:
        log.info("scaling NDVI to [0,1]")
        ndvi = ndvi.add(1).divide(2)
    
    return ndvi

def calculate_savi(image:ee.Image, near_infrared: str, red: str, L:float= 0.5, clamp=True, unit_normalization=True):
    """
    Calculate Soil-Adjusted Vegetation Index
    SAVI: (1+ L)(NIR - Red) / (NIR + Red + L)
    Minimize soil brightness influences from spectral vegetation indices

    Args: 
        image: ee.Image 
        near_infrared: str near_infrared band name 
        red: str red band name 
        L: float canopy background adjustment factor

    Returns:
        savi: ee.Image with 1 band 'SAVI'
    """

    log.info("calculating SAVI")
    savi = image.expression(
        '((NIR - RED) / (NIR + RED + 0.5)) * (1 + 0.5)',
        {
            'NIR': image.select(near_infrared),
            'RED': image.select(red),
            'L': L
        }
    ).rename('SAVI')
    # should not unit normalize without clamping first ...?
    if clamp:
        # clamp SAVI values since theoretical range with [0,1] is +- 3
        log.info("clamping SAVI values [-1, 1]")
        savi = savi.clamp(-1,1)
    if unit_normalization:
        log.info("scaling SAVI values to [0,1]")
        savi = savi.add(1).divide(2)
    
    return savi

def calculate_brightness(image: ee.Image, red: str, green: str, blue: str, normalize: Optional[float] = None) -> ee.Image:
    """
    Calculate a simple brightness index based on visible bands.

    This metric represents the mean reflectance of visible bands, 
    often used as a proxy for surface brightness or albedo.

    Formula:
        Brightness = (Red + Green + Blue) / 3

    Args:
        image (ee.Image): The Earth Engine image containing visible bands.
        red (str): Name of the Red band.
        green (str): Name of the Green band.
        blue (str): Name of the Blue band.
        normalize (float, optional): Value to divide by to scale brightness to [0, 1].

    Returns:
        ee.Image: A single-band image representing average visible brightness.
    """
    log.info("calculating brightness")
    brightness = image.expression(
        '(R + G + B) / 3',
        {
            'R': image.select(red),
            'G': image.select(green),
            'B': image.select(blue)
        }
    ).rename('Brightness')

    if normalize:
        log.info("normalizing brightness")
        brightness = brightness.divide(normalize).rename("Brightness")
    return brightness

def calculate_bare_soil_proxy(
    image: ee.Image,
    ndvi_band: str = 'NDVI',
    brightness_band: str = 'Brightness',
) -> ee.Image:
    """
    Estimate a simple bare soil proxy using NDVI and brightness.

    Since NAIP lacks SWIR bands, this index approximates bare soil areas
    as those with low vegetation (low NDVI) and high visible brightness.

    Formula:
        BareSoilProxy = (1 - NDVI) * Brightness

    Args:
        image (ee.Image): The Earth Engine image containing NDVI and brightness bands.
        ndvi_band (str, optional): Name of the NDVI band. Default 'NDVI'.
        brightness_band (str, optional): Name of the brightness band. Default 'Brightness'.

    Returns:
        ee.Image: A single-band image representing bare soil likelihood.
    """
    brightness = image.select(brightness_band)

    bare_soil_proxy = image.expression(
        '(1 - NDVI) * B',
        {
            'NDVI': image.select(ndvi_band),
            'B': brightness
        }
    ).rename('BareSoilProxy')

    return bare_soil_proxy

def reduce_by_parcel_raster(
    imagery: ee.Image,
    parcel_raster: ee.Image,
    region: ee.Geometry,
    bands: list[str],
    scale: int = 1,
    max_pixels: int = 1e13,
    debug: bool = False,
    parcel_ids: list[int] | None = None,
) -> ee.Dictionary:
    """
    Reduce imagery stats grouped by parcel ID using raster approach.

    Uses ee.Reducer.mean().group() instead of reduceRegions(), which is
    more efficient for large numbers of parcels.

    Args:
        imagery: ee.Image with bands to reduce.
        parcel_raster: ee.Image where pixel values are parcel IDs (int).
        region: ee.Geometry defining the area to process.
        bands: List of band names to compute stats for.
        scale: Pixel resolution in meters (default: 1 for NAIP).
        max_pixels: Maximum pixels to process.
        debug: If True, prints band info via getInfo() (slow, for debugging only).
        parcel_ids: Optional list of parcel IDs to include. If provided, only these
                   parcels will be processed (useful for sampled subsets).

    Returns:
        ee.Dictionary with 'groups' key containing list of dicts,
        each with 'parcel_id' and aggregated stats per band.
    """
    log.info(f"Reducing imagery by parcel raster, bands: {bands}, scale: {scale}")

    # Select ONLY the bands we want to reduce (ensures consistent ordering)
    selected_imagery = imagery.select(bands)

    # Explicitly select first band from parcel raster and rename it
    # (in case parcel_raster has multiple bands)
    parcel_band = parcel_raster.select(0).rename('parcel_id')

    # If parcel_ids provided, mask to only include those parcels
    if parcel_ids is not None:
        log.info(f"Filtering parcel raster to {len(parcel_ids)} sampled parcels")
        # Create mask: pixels with IDs in parcel_ids list get value 1, others get 0
        parcel_ids_ee = ee.List(parcel_ids)
        ones = ee.List.repeat(1, parcel_ids_ee.size())
        mask = parcel_band.remap(parcel_ids_ee, ones, 0)
        parcel_band = parcel_band.updateMask(mask)

    # Stack: spectral bands first, then parcel_id last
    stacked = selected_imagery.addBands(parcel_band)

    # parcel_id is at index len(bands)
    parcel_id_index = len(bands)
    log.info(f"Parcel ID band index: {parcel_id_index}")

    # Debug logging (only use for development - getInfo() forces client-side execution)
    if debug:
        log.info(f"Parcel raster bands: {parcel_raster.bandNames().getInfo()}")
        log.info(f"Selected imagery bands: {selected_imagery.bandNames().getInfo()}")
        log.info(f"Stacked bands: {stacked.bandNames().getInfo()}")
        log.info(f"Number of bands in stacked: {stacked.bandNames().size().getInfo()}")

    # Build grouped reducer: mean + stdDev + count for each band
    # .repeat(n) tells the reducer to process n input bands
    # .group(groupField=n) says the (n+1)th band contains group IDs
    num_bands = len(bands)
    reducer = (
        ee.Reducer.mean()
        .combine(ee.Reducer.stdDev(), sharedInputs=True)
        .combine(ee.Reducer.count(), sharedInputs=True)
        .repeat(num_bands)  # Process 8 spectral bands
        .group(groupField=num_bands, groupName='parcel_id')  # Group by band at index 8
    )

    result = stacked.reduceRegion(
        reducer=reducer,
        geometry=region,
        scale=scale,
        maxPixels=max_pixels,
        tileScale=16,  # Helps with memory for large regions
    )

    log.info("Grouped reduction complete")
    return result


def load_parcel_raster_asset(asset_id: str) -> ee.Image:
    """
    Load a parcel raster from a GEE Image asset.

    Args:
        asset_id: Full asset path (e.g., projects/project-id/assets/name).

    Returns:
        ee.Image with parcel IDs.
    """
    log.info(f"Loading parcel raster asset: {asset_id}")
    return ee.Image(asset_id)


def export_grouped_stats_to_gcs(
    stats_dict: ee.Dictionary,
    description: str,
    bucket: str,
    file_prefix: str,
) -> ee.batch.Task:
    """
    Export grouped parcel stats to Cloud Storage as a CSV-like format.

    Note: For large results, consider using ee.batch.Export.table.toCloudStorage
    with a FeatureCollection instead.

    Args:
        stats_dict: Result from reduce_by_parcel_raster().
        description: Export task description.
        bucket: GCS bucket name.
        file_prefix: File path prefix in bucket.

    Returns:
        The export task.
    """
    log.info(f"Exporting grouped stats to gs://{bucket}/{file_prefix}")

    # Convert grouped stats to FeatureCollection for export
    groups = ee.List(stats_dict.get('groups'))

    def dict_to_feature(d):
        d = ee.Dictionary(d)
        return ee.Feature(None, d)

    fc = ee.FeatureCollection(groups.map(dict_to_feature))

    task = ee.batch.Export.table.toCloudStorage(
        collection=fc,
        description=description,
        bucket=bucket,
        fileNamePrefix=file_prefix,
        fileFormat='CSV',
    )
    task.start()
    log.info("Export task started")
    return task


def calculate_glcm_texture(image: ee.Image, band:str = 'N', window_size: int =3) -> ee.Image:
    """
    Calculate GLCM texture features from a specific band.

    Args:
        image: ee.Image
        band: Band name to calculate texture from (default: NIR)
        window_size: Window size for GLCM calculation

    Returns:
        ee.Image: Image with GLCM texture bands
    """
    # Select the band and calculate GLCM
    glcm = image.select(band).glcmTexture(size=window_size)

    # Select specific texture features
    # Available: asm, contrast, corr, var, idm, savg, svar, sent, ent, dvar, dent, imcorr1, imcorr2, maxcorr, diss, inertia, shade, prom
    texture_features = glcm.select([
        f'{band}_asm',      # Angular Second Moment (homogeneity)
        f'{band}_contrast', # Contrast
        f'{band}_diss',     # Dissimilarity
        f'{band}_ent',      # Entropy
        f'{band}_idm'       # Inverse Difference Moment (homogeneity)
    ]).rename([
        f'GLCM_{band}_ASM',
        f'GLCM_{band}_Contrast',
        f'GLCM_{band}_Dissimilarity',
        f'GLCM_{band}_Entropy',
        f'GLCM_{band}_Homogeneity'
    ])

    return texture_features


def find_naip_years(geometry: ee.Geometry) -> List[int]:
    """
    Find available NAIP years for a region.

    NAIP doesn't image every state every year, so use this to discover
    which years have coverage for a given geometry.

    Args:
        geometry: ee.Geometry to check for NAIP coverage.

    Returns:
        Sorted list of years with NAIP imagery available.
    """
    log.info("Finding available NAIP years for region")

    naip = ee.ImageCollection("USDA/NAIP/DOQQ")
    filtered = naip.filterBounds(geometry)

    # Get unique years from the collection
    years = filtered.aggregate_array("system:time_start").map(
        lambda t: ee.Date(t).get("year")
    ).distinct().getInfo()

    sorted_years = sorted(years)
    log.info(f"Available NAIP years: {sorted_years}")
    return sorted_years


# ============================================================================
# Vector-based Parcel Processing (for reduceRegions workflow)
# ============================================================================

def create_parcel_shapefile(
    gdf,
    id_column: str,
    output_dir: Path | str,
    filename_prefix: str = "parcels",
) -> tuple[Path, Path]:
    """
    Create a shapefile from a GeoDataFrame for GEE ingestion.

    GEE requires shapefiles for table ingestion. This function:
    1. Extracts only the ID column and geometry (minimal for GEE)
    2. Converts ID to string (GEE/shapefiles don't handle large ints well)
    3. Writes shapefile and creates a zip for upload

    Args:
        gdf: GeoDataFrame with parcels.
        id_column: Column name for parcel identifier (e.g., 'BBL').
        output_dir: Directory to write shapefile and zip.
        filename_prefix: Prefix for output files (default: 'parcels').

    Returns:
        Tuple of (shapefile_path, zip_path).

    Example:
        shp_path, zip_path = create_parcel_shapefile(
            gdf=sampled_gdf,
            id_column='BBL',
            output_dir=CONFIG.get_intermediaries_dir(REPO_ROOT),
            filename_prefix='nyc_parcels'
        )
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create minimal GeoDataFrame with just ID and geometry
    minimal_gdf = gdf[[id_column, "geometry"]].copy()

    # Convert ID to string (shapefiles don't handle large ints, GEE needs strings)
    minimal_gdf[id_column] = minimal_gdf[id_column].astype(str)
    log.info(f"Created minimal GDF with {len(minimal_gdf)} parcels, ID column '{id_column}' as string")

    # Write shapefile
    shp_path = output_dir / f"{filename_prefix}.shp"
    minimal_gdf.to_file(shp_path)
    log.info(f"Wrote shapefile: {shp_path}")

    # Zip all shapefile components (GEE needs .shp, .shx, .dbf, .prj together)
    zip_path = output_dir / f"{filename_prefix}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for file in output_dir.glob(f"{filename_prefix}.*"):
            if file.suffix != ".zip":
                z.write(file, arcname=file.name)

    log.info(f"Created zip: {zip_path}")
    return shp_path, zip_path


def upload_parcels_to_gcs(
    gdf,
    id_column: str,
    output_dir: Path | str,
    bucket_name: str,
    gcs_prefix: str,
    filename_prefix: str = "parcels",
) -> str:
    """
    Prepare a parcel GeoDataFrame as a zipped shapefile and upload to GCS.

    Wraps create_parcel_shapefile + upload_to_gcs.

    Args:
        gdf: GeoDataFrame with parcels.
        id_column: Parcel identifier column (e.g., 'BBL').
        output_dir: Local directory for intermediate shapefile/zip.
        bucket_name: GCS bucket name.
        gcs_prefix: Path prefix in bucket (e.g., 'eda/new_york_new_york').
        filename_prefix: Prefix for output files (default: 'parcels').

    Returns:
        Full GCS URI (gs://bucket/prefix/filename.zip).
    """
    _, zip_path = create_parcel_shapefile(
        gdf=gdf,
        id_column=id_column,
        output_dir=output_dir,
        filename_prefix=filename_prefix,
    )
    return upload_to_gcs(
        local_path=zip_path,
        bucket_name=bucket_name,
        gcs_prefix=gcs_prefix,
        gcs_filename=f"{filename_prefix}.zip",
    )


def ingest_parcels_to_gee(
    asset_id: str,
    gcs_uri: str | None = None,
    bucket_name: str | None = None,
    gcs_prefix: str | None = None,
    gcs_filename: str | None = None,
) -> dict | None:
    """
    Ingest a zipped shapefile from GCS into GEE as a FeatureCollection asset.

    Checks whether the asset already exists before attempting ingestion and
    warns the user with deletion instructions if it does.

    Args:
        asset_id: Destination GEE asset ID (e.g., 'projects/project/assets/name').
        gcs_uri: Full GCS URI. If None, built from bucket_name/gcs_prefix/gcs_filename.
        bucket_name: GCS bucket (required if gcs_uri is None).
        gcs_prefix: GCS path prefix (required if gcs_uri is None).
        gcs_filename: Filename in GCS (required if gcs_uri is None).

    Returns:
        Dict with asset_id, gcs_uri, and task_id, or None if asset already exists.
    """
    if gcs_uri is None:
        if not all([bucket_name, gcs_prefix, gcs_filename]):
            raise ValueError("Provide gcs_uri, or all of bucket_name, gcs_prefix, gcs_filename.")
        gcs_uri = f"gs://{bucket_name}/{gcs_prefix}/{gcs_filename}"

    try:
        ee.data.getAsset(asset_id)
        log.warning(
            f"GEE asset already exists: {asset_id}\n"
            f"To overwrite:\n"
            f"  1. Delete the asset at https://code.earthengine.google.com/ → Assets\n"
            f"     or run: ee.data.deleteAsset('{asset_id}')\n"
            f"  2. Re-run ingest_parcels_to_gee()"
        )
        return None
    except ee.EEException:
        pass  # asset does not exist, proceed with ingestion

    log.info(f"Ingesting {gcs_uri} → {asset_id}")
    task = ee.data.startTableIngestion(None, {"name": asset_id, "sources": [{"uris": [gcs_uri]}]})
    log.info(f"Ingestion task started: {task['id']}")
    return {"asset_id": asset_id, "gcs_uri": gcs_uri, "task_id": task["id"]}


def ingest_table_to_gee(
    gcs_uri: str,
    asset_id: str,
) -> dict:
    """
    Ingest a table (shapefile/zip) from GCS to GEE as a FeatureCollection asset.

    Args:
        gcs_uri: Full GCS URI (gs://bucket/path/file.zip).
        asset_id: Destination GEE asset ID (e.g., projects/project-id/assets/name).

    Returns:
        Dict with asset_id, gcs_uri, and task_id.

    Example:
        task_info = ingest_table_to_gee(
            gcs_uri='gs://thesis_parcels/eda/new_york_new_york/parcels.zip',
            asset_id='projects/vacant-lot-detection/assets/nyc_parcels'
        )
    """
    log.info(f"Ingesting table from {gcs_uri} to GEE asset: {asset_id}")

    manifest = {
        "name": asset_id,
        "sources": [{"uris": [gcs_uri]}],
    }

    task = ee.data.startTableIngestion(None, manifest)

    log.info(f"Table ingestion task started: {task['id']}")
    return {
        "asset_id": asset_id,
        "gcs_uri": gcs_uri,
        "task_id": task["id"],
    }


def reduce_regions_to_gcs(
    imagery: ee.Image,
    parcels_asset_id: str,
    bucket_name: str,
    gcs_prefix: str,
    filename: str = "parcel_spectral_stats",
    scale: int = 1,
    tile_scale: int = 4,
    include_median: bool = True,
    include_count: bool = False,
) -> ee.batch.Task:
    """
    Run reduceRegions on imagery and export results to GCS.

    This is the vector-based approach for computing per-parcel statistics.
    Faster than raster-based grouping for subsets of parcels.

    Args:
        imagery: ee.Image with bands to reduce (e.g., NAIP with spectral indices).
        parcels_asset_id: GEE asset ID for parcel FeatureCollection.
        bucket_name: GCS bucket for output.
        gcs_prefix: Path prefix in bucket (e.g., 'eda/new_york_new_york').
        filename: Output filename without extension (default: 'parcel_spectral_stats').
        scale: Pixel resolution in meters (default: 1 for NAIP).
        tile_scale: Tile scale for memory management (default: 4).
        include_median: Whether to include median reducer (default: True).
        include_count: Whether to include count reducer (default: False).

    Returns:
        The export task (already started).

    Example:
        task = reduce_regions_to_gcs(
            imagery=naip,
            parcels_asset_id='projects/vacant-lot-detection/assets/nyc_parcels',
            bucket_name='thesis_parcels',
            gcs_prefix='eda/new_york_new_york',
        )
    """
    log.info(f"Loading parcels from asset: {parcels_asset_id}")
    parcels = ee.FeatureCollection(parcels_asset_id)

    # Build reducer: mean + stdDev (+ optional median, count)
    reducer = ee.Reducer.mean().combine(ee.Reducer.stdDev(), "", True)

    if include_median:
        reducer = reducer.combine(ee.Reducer.median(), "", True)

    if include_count:
        reducer = reducer.combine(ee.Reducer.count(), "", True)

    log.info(f"Running reduceRegions with scale={scale}, tileScale={tile_scale}")
    stats = imagery.reduceRegions(
        collection=parcels,
        reducer=reducer,
        scale=scale,
        tileScale=tile_scale,
    )

    # Export to GCS
    file_prefix = f"{gcs_prefix}/{filename}"
    log.info(f"Exporting to gs://{bucket_name}/{file_prefix}.csv")

    task = ee.batch.Export.table.toCloudStorage(
        collection=stats,
        description=filename,
        bucket=bucket_name,
        fileNamePrefix=file_prefix,
        fileFormat="CSV",
    )
    task.start()

    log.info(f"Export task started: {task.id}")
    return task


def vector_reduce_pipeline(
    gdf,
    imagery: ee.Image,
    id_column: str,
    output_dir: Path | str,
    bucket_name: str,
    gcs_prefix: str,
    asset_id: str,
    filename_prefix: str = "parcels",
    scale: int = 1,
    skip_upload: bool = False,
    skip_ingestion: bool = False,
) -> dict:
    """
    Full pipeline: shapefile -> GCS -> GEE asset -> reduceRegions -> GCS export.

    Convenience function that chains all vector workflow steps.

    Args:
        gdf: GeoDataFrame with parcels to process.
        imagery: ee.Image with bands to reduce.
        id_column: Parcel identifier column (e.g., 'BBL').
        output_dir: Local directory for intermediate files.
        bucket_name: GCS bucket name.
        gcs_prefix: Path prefix in GCS (e.g., 'eda/new_york_new_york').
        asset_id: GEE asset ID for the parcels table.
        filename_prefix: Prefix for shapefile/zip (default: 'parcels').
        scale: Pixel resolution for reduceRegions (default: 1).
        skip_upload: Skip shapefile upload (if already in GCS).
        skip_ingestion: Skip table ingestion (if asset already exists).

    Returns:
        Dict with paths, URIs, and task info.
    """
    result = {}

    # Step 1: Create shapefile
    shp_path, zip_path = create_parcel_shapefile(
        gdf=gdf,
        id_column=id_column,
        output_dir=output_dir,
        filename_prefix=filename_prefix,
    )
    result["shp_path"] = shp_path
    result["zip_path"] = zip_path

    # Step 2: Upload to GCS
    if not skip_upload:
        gcs_uri = upload_to_gcs(
            local_path=zip_path,
            bucket_name=bucket_name,
            gcs_prefix=gcs_prefix,
            gcs_filename=f"{filename_prefix}.zip",
        )
        result["gcs_uri"] = gcs_uri
    else:
        gcs_uri = f"gs://{bucket_name}/{gcs_prefix}/{filename_prefix}.zip"
        result["gcs_uri"] = gcs_uri
        log.info(f"Skipping upload, using existing: {gcs_uri}")

    # Step 3: Ingest to GEE
    if not skip_ingestion:
        ingest_info = ingest_table_to_gee(gcs_uri=gcs_uri, asset_id=asset_id)
        result["ingest_task_id"] = ingest_info["task_id"]
        log.info("Waiting for ingestion to complete before reduceRegions...")
        log.info("Check progress at: https://code.earthengine.google.com/tasks")
    else:
        log.info(f"Skipping ingestion, using existing asset: {asset_id}")

    result["asset_id"] = asset_id

    # Step 4: reduceRegions (only if not skipping ingestion, otherwise asset might not exist yet)
    if skip_ingestion:
        task = reduce_regions_to_gcs(
            imagery=imagery,
            parcels_asset_id=asset_id,
            bucket_name=bucket_name,
            gcs_prefix=gcs_prefix,
            filename="parcel_spectral_stats",
            scale=scale,
        )
        result["export_task"] = task

    return result
