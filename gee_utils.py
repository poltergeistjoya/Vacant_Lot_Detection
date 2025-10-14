import ee
from typing import Optional


from logger import get_logger
from config import EarthEngineConfig


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
        collection = ee.ImageCollection(collection_id).filterDate(start_date, end_date).filterBounds(region)
        log.info("Clipping images to region")
        collection = collection.map(lambda img: img.clip(region))
    else: 
        collection = ee.ImageCollection(collection_id).filterDate(start_date, end_date)

    size = collection.size().getInfo()
    log.info(f"Filtered collection contains {size} images")

    if mosaic:
        log.info("Creating mosaic from filtered collection")
        image = collection.mosaic()
        log.info("Mosaic created successfully")
        return image
    
    log.info("Returning filtered ImageCollection (not mosaiced)")
    return collection
