"""
tile_export.py — NAIP tile download (STAC) and VRT creation.

Downloads raw NAIP COG tiles directly from Microsoft Planetary Computer
STAC catalog — no GEE, no GCS, no egress cost. Stitches into a GDAL VRT
for downstream pixel-level segmentation.
"""

import json
import subprocess
from pathlib import Path

from .config import CityConfig
from .logger import get_logger

log = get_logger()

_NAIP_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
_NAIP_COLLECTION = "naip"


def query_naip_stac(
    bbox: tuple[float, float, float, float],
    year: int = 2022,
    stac_url: str = _NAIP_STAC_URL,
) -> list:
    """
    Query the Microsoft Planetary Computer STAC catalog for NAIP items.

    Planetary Computer mirrors NAIP and serves it free via signed Azure SAS tokens.
    The planetary_computer.sign_inplace modifier automatically attaches a token to
    every item href — no account or payment needed.

    Args:
        bbox: (west, south, east, north) in EPSG:4326.
        year: NAIP acquisition year to search.
        stac_url: STAC API endpoint (default: Planetary Computer).

    Returns:
        List of signed pystac Item objects covering the bbox for the given year.
    """
    import planetary_computer
    import pystac_client

    log.info(f"Querying Planetary Computer NAIP STAC for year={year}, bbox={bbox}")
    catalog = pystac_client.Client.open(
        stac_url,
        modifier=planetary_computer.sign_inplace,
    )
    search = catalog.search(
        collections=[_NAIP_COLLECTION],
        bbox=bbox,
        datetime=f"{year}-01-01/{year}-12-31",
        max_items=None,
    )
    items = list(search.items())
    log.info(f"Found {len(items)} NAIP tiles")
    return items


def download_naip_tiles(
    items: list,
    local_dir: Path | str,
    asset_key: str = "image",
) -> list[Path]:
    """
    Download signed NAIP COG GeoTIFFs from Planetary Computer to local_dir.

    Items must come from query_naip_stac() — the SAS token in each href is what
    grants free access. Skips files that already exist locally (idempotent).

    **Try to make async for speed**
    
    Args:
        items: Signed pystac Item objects from query_naip_stac().
        local_dir: Local directory to download tiles into.
        asset_key: Asset key for the COG image (default: "image").

    Returns:
        List of local Paths for downloaded tiles, skipping items with missing assets.
    """
    import requests

    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    local_paths = []
    for i, item in enumerate(items):
        asset = item.assets.get(asset_key)
        if asset is None:
            log.warning(f"Item {item.id}: no asset '{asset_key}', skipping")
            continue

        local_path = local_dir / f"{item.id}.tif"
        tmp_path = local_path.with_suffix(".tmp")

        if local_path.exists():
            log.info(f"[{i+1}/{len(items)}] Skipping (exists): {local_path.name}")
        else:
            log.info(f"[{i+1}/{len(items)}] Downloading {local_path.name}")
            for attempt in range(1, 4):
                try:
                    # timeout=(connect_s, read_s): if the socket stalls for 90s
                    # (e.g. after system sleep drops the TCP connection), raises
                    # so the retry loop fires instead of hanging forever.
                    with requests.get(asset.href, stream=True, timeout=(10, 90)) as r:
                        r.raise_for_status()
                        with open(tmp_path, "wb") as f:
                            for chunk in r.iter_content(chunk_size=1 << 20):
                                f.write(chunk)
                    tmp_path.rename(local_path)
                    break
                except Exception as e:
                    tmp_path.unlink(missing_ok=True)
                    if attempt == 3:
                        raise RuntimeError(f"Failed after 3 attempts: {local_path.name}") from e
                    log.warning(f"Attempt {attempt} failed ({e}), retrying...")

        local_paths.append(local_path)

    log.info(f"Done: {len(local_paths)} tiles in {local_dir}")
    return local_paths


def merge_tiles_to_vrt(
    tile_paths: list[Path],
    output_vrt: Path | str,
) -> Path:
    """
    Stitch GeoTIFF tiles into a single GDAL VRT using gdalbuildvrt.

    Args:
        tile_paths: List of local GeoTIFF paths.
        output_vrt: Output VRT file path.

    Returns:
        Path to the created VRT file.

    Also writes a sidecar JSON next to the VRT:
        {"tile_count": N, "band_order": ["R","G","B","NIR"], "crs": "...", "bounds": [...]}
    """
    import rasterio

    output_vrt = Path(output_vrt)

    log.info(f"Building VRT from {len(tile_paths)} tiles → {output_vrt}")
    result = subprocess.run(
        ["gdalbuildvrt", str(output_vrt)] + [str(p) for p in tile_paths],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"gdalbuildvrt failed:\n{result.stderr}")
    log.info("VRT created")

    with rasterio.open(output_vrt) as ds:
        crs_str = ds.crs.to_string() if ds.crs else "unknown"
        bounds = list(ds.bounds)

    sidecar = {
        "tile_count": len(tile_paths),
        "band_order": ["R", "G", "B", "NIR"],
        "crs": crs_str,
        "bounds": bounds,
    }
    sidecar_path = output_vrt.with_suffix(".json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2))
    log.info(f"Sidecar: {sidecar_path}")

    return output_vrt
