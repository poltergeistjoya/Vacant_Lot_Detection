import ee
import re
from typing import Optional, List


from logger import get_logger
from config import EarthEngineConfig, Config


log = get_logger()

# import os
# import json
# import google.auth
# from google.auth.transport.requests import Request

# def log_adc_identity():
#     adc_path = os.path.expanduser("~/.config/gcloud/application_default_credentials.json")
#     email = None
#     client_id = None
#     project = None
#     auth_type = None

#     # Try standard ADC load first
#     try:
#         credentials, project = google.auth.default()
#         credentials.refresh(Request())
#         auth_type = type(credentials).__name__
#         if hasattr(credentials, "service_account_email"):
#             email = credentials.service_account_email
#         elif hasattr(credentials, "_subject"):
#             email = credentials._subject
#     except Exception:
#         pass

#     # Fallback: read the JSON manually
#     if not email and os.path.exists(adc_path):
#         try:
#             with open(adc_path, "r") as f:
#                 adc_json = json.load(f)
#                 email = adc_json.get("client_email")
#                 client_id = adc_json.get("client_id")
#         except Exception as e:
#             log.error(f"⚠️ Failed to read ADC file: {e}")

#     log.info("🔍 Google ADC info:")
#     log.info(f"   Auth type: {auth_type or 'unknown'}")
#     log.info(f"   Project: {project or 'unknown'}")
#     log.info(f"   Client email: {email or 'unknown'}")
#     log.info(f"   Client ID: {client_id or 'unknown'}")

#     return {
#         "auth_type": auth_type,
#         "project": project,
#         "client_email": email,
#         "client_id": client_id
#     }

def init_gee(config:EarthEngineConfig):
    """
    Initialize GEE w project-scoped credentials.
    Assumes ADC (Application Default Credentials) are set via `gcloud auth application-default login`
    """

    project_id = config.PROJECT_ID
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
        region: Optional[ee.Geometry] = None, 
        scale: int =1,
        max_pixels: int = 1e13,
        # config:EarthEngineConfig = None
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

    # # --- validate asset_id ---
    # expected_prefix = f"projects/{config.EARTH_ENGINE.PROJECT_ID}/assets/"
    # pattern = rf"^{re.escape(expected_prefix)}"
    # assert re.match(pattern, asset_id), (
    #     f"asset_id must start with '{expected_prefix}', got '{asset_id}'"
    # )

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
