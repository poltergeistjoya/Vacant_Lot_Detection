import ee
from joblib import Memory

from config import CityConfig, GeometryConfig
from logger import get_logger

log = get_logger()
memory = Memory("cache", verbose=0)


@memory.cache
def get_city_geometry(config: CityConfig) -> ee.Geometry:
    """
    Get city geometry based on config.

    Args:
        config: CityConfig with geometry settings.

    Returns:
        ee.Geometry for the city.
    """
    geo_config = config.geometry

    if geo_config.source == "tiger_counties":
        return _get_geometry_from_tiger(geo_config)
    elif geo_config.source == "geojson":
        return _get_geometry_from_geojson(geo_config)
    else:
        raise ValueError(f"Unknown geometry source: {geo_config.source}")


def _get_geometry_from_tiger(geo_config: GeometryConfig) -> ee.Geometry:
    """
    Build geometry from TIGER county names.

    Args:
        geo_config: GeometryConfig with counties list and state_fips.

    Returns:
        ee.Geometry for the combined counties.
    """
    log.info(f"Loading geometry from TIGER counties: {geo_config.counties} (state FIPS: {geo_config.state_fips})")

    counties = ee.FeatureCollection("TIGER/2018/Counties")

    # Build filter for county names within state
    name_filters = [ee.Filter.eq("NAME", name) for name in geo_config.counties]
    county_filter = ee.Filter.And(
        ee.Filter.Or(*name_filters),
        ee.Filter.eq("STATEFP", geo_config.state_fips)
    )

    filtered = counties.filter(county_filter)
    geometry = filtered.geometry().dissolve(maxError=100)

    log.info("Geometry loaded successfully")
    return geometry


def _get_geometry_from_geojson(geo_config: GeometryConfig) -> ee.Geometry:
    """
    Load geometry from local GeoJSON file.

    Args:
        geo_config: GeometryConfig with path to GeoJSON.

    Returns:
        ee.Geometry from the GeoJSON.
    """
    if not geo_config.path:
        raise ValueError("path required for geojson source")

    log.info(f"Loading geometry from GeoJSON: {geo_config.path}")

    import json
    with open(geo_config.path) as f:
        geojson = json.load(f)

    geometry = ee.Geometry(geojson)
    log.info("Geometry loaded successfully")
    return geometry
