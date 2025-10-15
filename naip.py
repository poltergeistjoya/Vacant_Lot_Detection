import ee

from config import Config
from gee_utils import calculate_ndvi, calculate_savi, calculate_brightness, calculate_bare_soil_proxy
from logger import get_logger

log = get_logger()
def calculate_spectral_indices(image: ee.Image, config:Config) -> ee.Image:
    """
    Calculate spectral indices from NAIP imagery.
    
    Args:
        image: NAIP ee.Image with bands R, G, B, N
        
    Returns:
        ee.Image: Image with added spectral index bands
    """
    
    ndvi = calculate_ndvi(image, 'N', 'R')
    
    # SAVI: ((NIR - Red) / (NIR + Red + L)) * (1 + L), L=0.5
    savi = calculate_savi(image, 'N', 'R', L=0.5)

    # Simple brightness index (mean of visible bands)
    brightness = calculate_brightness(image, 'R', 'G', 'B', config.SENSOR_NORMALIZATION.NAIP)
    
    # ndvi and brightness Needed for bare soil proxy.
    image = image.addBands([ndvi, savi, brightness])
    # Bare soil proxy: Low NDVI + High brightness in visible bands
    # This is a simple proxy since NAIP lacks SWIR
    bare_soil_proxy = calculate_bare_soil_proxy(image, 'NDVI', 'Brightness', normalize_brightness="True")
    image = image.addBands([bare_soil_proxy])
    
    return image
