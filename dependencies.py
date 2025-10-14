import ee
from joblib import Memory
from logger import get_logger

log = get_logger()
memory = Memory("cache", verbose=0)

# TODO maybe a lru cache here
@memory.cache
def get_nyc_geometry() -> ee.Geometry:
    """
    Returns a unified ee.Geometry for the five NYC counties
    (New York, Kings, Queens, Bronx, Richmond), cached in memory
    to avoid redundant Earth Engine requests.
    """
    counties = ee.FeatureCollection("TIGER/2018/Counties")
    
    nyc_counties = counties.filter(
        ee.Filter.Or(
            ee.Filter.eq("NAME", "New York"),   # Manhattan
            ee.Filter.eq("NAME", "Kings"),      # Brooklyn
            ee.Filter.eq("NAME", "Queens"),     # Queens
            ee.Filter.eq("NAME", "Bronx"),      # Bronx
            ee.Filter.eq("NAME", "Richmond")    # Staten Island
        )
    )
    
    nyc_geom = nyc_counties.union().geometry()
    return nyc_geom

