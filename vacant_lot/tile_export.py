"""
tile_export.py — NAIP tile export, GCS download, and VRT creation.

Exports raw NAIP pixels as a grid of GeoTIFFs to GCS, downloads locally,
and stitches them into a GDAL VRT for downstream pixel-level segmentation.
"""

import json
import subprocess
import time
from pathlib import Path

import ee

from .config import CityConfig
from .logger import get_logger

log = get_logger()


def build_tile_grid(
    region: ee.Geometry,
    tile_size_deg: float = 0.09,
) -> list[dict]:
    """
    Divide the bounding box of a region into a regular lat/lon grid of tiles.

    Args:
        region: ee.Geometry defining the area to tile.
        tile_size_deg: Grid cell size in degrees (~10 km at mid-latitudes).

    Returns:
        List of {"row": int, "col": int, "bbox": [west, south, east, north]} dicts
        covering the bounding box of the region.
    """
    bounds = region.bounds().getInfo()["coordinates"][0]
    lons = [pt[0] for pt in bounds]
    lats = [pt[1] for pt in bounds]
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)

    tiles = []
    row = 0
    lat = min_lat
    while lat < max_lat:
        col = 0
        lon = min_lon
        while lon < max_lon:
            west = lon
            south = lat
            east = min(lon + tile_size_deg, max_lon)
            north = min(lat + tile_size_deg, max_lat)
            tiles.append({"row": row, "col": col, "bbox": [west, south, east, north]})
            lon += tile_size_deg
            col += 1
        lat += tile_size_deg
        row += 1

    log.info(f"Built tile grid: {len(tiles)} tiles ({row} rows × {col} cols) at {tile_size_deg}°")
    return tiles


def export_naip_tiles(
    config: CityConfig,
    region: ee.Geometry,
    imagery: ee.Image,
    scale: int = 1,
    crs: str = "EPSG:32618",
) -> list[dict]:
    """
    Submit one GEE toCloudStorage export task per tile in the grid.

    Args:
        config: CityConfig with GCP and segmentation settings.
        region: ee.Geometry defining the export area.
        imagery: ee.Image to export (should have R, G, B, NIR bands).
        scale: Pixel resolution in meters (default: 1 for NAIP).
        crs: Output CRS (default: EPSG:32618 — UTM Zone 18N).

    Returns:
        List of {"row", "col", "task_id", "gcs_path"} dicts.
        Also writes tasks.json to the naip_tiles dir for resumability.
    """
    run_key = config._run_key()
    bucket = config.gcp.bucket
    tiles = build_tile_grid(region, config.segmentation.tile_size_deg)

    task_records = []
    for tile in tiles:
        row, col = tile["row"], tile["col"]
        west, south, east, north = tile["bbox"]

        description = f"naip_tile_{row:03d}_{col:03d}"
        file_prefix = f"segmentation/{run_key}/naip_tiles/tile_{row:03d}_{col:03d}"

        task = ee.batch.Export.image.toCloudStorage(
            image=imagery,
            description=description,
            bucket=bucket,
            fileNamePrefix=file_prefix,
            region=ee.Geometry.BBox(west, south, east, north),
            scale=scale,
            crs=crs,
            maxPixels=int(1e10),
            fileFormat="GeoTIFF",
        )
        task.start()

        gcs_path = f"gs://{bucket}/{file_prefix}.tif"
        task_records.append({
            "row": row,
            "col": col,
            "task_id": task.id,
            "gcs_path": gcs_path,
        })
        log.info(f"Submitted tile ({row:03d}, {col:03d}): task {task.id}")

    # Persist for resumability
    tiles_dir = config.get_naip_tiles_dir()
    tasks_json = tiles_dir / "tasks.json"
    tasks_json.write_text(json.dumps(task_records, indent=2))
    log.info(f"Saved {len(task_records)} task records to {tasks_json}")

    return task_records


def wait_for_exports(
    task_ids: list[str],
    poll_interval: int = 60,
    timeout_hours: float = 6.0,
) -> None:
    """
    Poll GEE task status until all tasks complete or fail.

    Args:
        task_ids: List of GEE task IDs to monitor.
        poll_interval: Seconds between status checks (default: 60).
        timeout_hours: Max hours to wait before raising TimeoutError (default: 6).

    Raises:
        RuntimeError: If any task fails or is cancelled.
        TimeoutError: If tasks don't complete within timeout_hours.
    """
    deadline = time.time() + timeout_hours * 3600
    pending = set(task_ids)
    log.info(f"Waiting for {len(pending)} GEE export tasks (timeout: {timeout_hours}h)")

    while pending:
        if time.time() > deadline:
            raise TimeoutError(f"Timed out waiting for tasks: {pending}")

        done = set()
        failed = []
        for task_id in list(pending):
            status = ee.data.getTaskStatus([task_id])[0]
            state = status["state"]
            if state == "COMPLETED":
                done.add(task_id)
            elif state in ("FAILED", "CANCELLED"):
                failed.append((task_id, state, status.get("error_message", "")))

        if failed:
            msgs = "; ".join(f"{tid} [{st}]: {err}" for tid, st, err in failed)
            raise RuntimeError(f"GEE export task(s) failed: {msgs}")

        pending -= done
        if pending:
            log.info(f"Tasks remaining: {len(pending)} / {len(task_ids)} — sleeping {poll_interval}s")
            time.sleep(poll_interval)

    log.info("All GEE export tasks completed successfully")


def download_tiles_from_gcs(
    config: CityConfig,
    gcs_paths: list[str],
    local_dir: Path | str | None = None,
) -> list[Path]:
    """
    Download tiles from GCS to a local directory.

    Args:
        config: CityConfig (used to resolve default local_dir).
        gcs_paths: List of GCS URIs (gs://bucket/path/tile.tif).
        local_dir: Local directory to download into (defaults to get_naip_tiles_dir()).

    Returns:
        List of local Paths for downloaded tiles (in same order as gcs_paths).
    """
    from google.cloud import storage

    if local_dir is None:
        local_dir = config.get_naip_tiles_dir()
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    client = storage.Client(project=config.gcp.project_id)
    local_paths = []

    for gcs_uri in gcs_paths:
        # Parse gs://bucket/blob_path
        without_scheme = gcs_uri[len("gs://"):]
        bucket_name, blob_path = without_scheme.split("/", 1)
        filename = Path(blob_path).name
        local_path = local_dir / filename

        if local_path.exists():
            log.info(f"Skipping (already exists): {local_path.name}")
        else:
            log.info(f"Downloading {gcs_uri} → {local_path}")
            blob = client.bucket(bucket_name).blob(blob_path)
            blob.download_to_filename(str(local_path))

        local_paths.append(local_path)

    log.info(f"Downloaded {len(local_paths)} tiles to {local_dir}")
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

    Also writes a sidecar JSON next to the VRT with metadata:
        {"tile_count": N, "band_order": ["R","G","B","NIR"], "crs": "...", "bounds": [...]}
    """
    import rasterio

    output_vrt = Path(output_vrt)
    tile_strs = [str(p) for p in tile_paths]

    log.info(f"Building VRT from {len(tile_paths)} tiles → {output_vrt}")
    result = subprocess.run(
        ["gdalbuildvrt", str(output_vrt)] + tile_strs,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"gdalbuildvrt failed:\n{result.stderr}")
    log.info("VRT created successfully")

    # Read metadata from VRT for sidecar
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
    log.info(f"Wrote sidecar metadata: {sidecar_path}")

    return output_vrt
