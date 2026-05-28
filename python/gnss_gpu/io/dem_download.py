"""Automatic DEM download utilities (public elevation tiles)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import requests


TILE_URL_TEMPLATE = "https://s3.amazonaws.com/elevation-tiles-prod/geotiff/{z}/{x}/{y}.tif"


def _lonlat_to_tile(lon_deg: float, lat_deg: float, z: int) -> tuple[int, int]:
    lat_rad = math.radians(max(-85.05112878, min(85.05112878, lat_deg)))
    n = 2**z
    x = int((lon_deg + 180.0) / 360.0 * n)
    y = int((1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def _meters_to_deg_lat(meters: float) -> float:
    return float(meters) / 111_320.0


def _meters_to_deg_lon(meters: float, lat_deg: float) -> float:
    return float(meters) / (111_320.0 * max(1e-6, math.cos(math.radians(lat_deg))))


def download_dem_for_bbox(
    *,
    south: float,
    west: float,
    north: float,
    east: float,
    output_tif: str | Path,
    zoom: int = 12,
    timeout_s: int = 60,
    user_agent: str = "gnss_gpu_dem_downloader/1.0",
) -> tuple[Path, int]:
    """Download and merge DEM geotiff tiles covering a lat/lon bbox."""
    try:
        import rasterio
        from rasterio.merge import merge
    except ImportError as exc:  # pragma: no cover
        raise ImportError("DEM download requires rasterio (pip install rasterio).") from exc

    z = int(zoom)
    x0, y1 = _lonlat_to_tile(west, south, z)
    x1, y0 = _lonlat_to_tile(east, north, z)
    xmin, xmax = sorted((x0, x1))
    ymin, ymax = sorted((y0, y1))

    out = Path(output_tif)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = out.parent / f".dem_tiles_z{z}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    headers = {"User-Agent": user_agent}
    tile_paths: list[Path] = []
    for x in range(xmin, xmax + 1):
        for y in range(ymin, ymax + 1):
            url = TILE_URL_TEMPLATE.format(z=z, x=x, y=y)
            resp = session.get(url, headers=headers, timeout=timeout_s)
            resp.raise_for_status()
            p = tmp_dir / f"{z}_{x}_{y}.tif"
            p.write_bytes(resp.content)
            tile_paths.append(p)

    if not tile_paths:
        raise RuntimeError("No DEM tiles downloaded for requested bbox.")

    srcs = [rasterio.open(str(p)) for p in tile_paths]
    try:
        mosaic, transform = merge(srcs)
        profile = srcs[0].profile.copy()
        profile.update(
            driver="GTiff",
            height=mosaic.shape[1],
            width=mosaic.shape[2],
            transform=transform,
            count=mosaic.shape[0],
            compress="deflate",
        )
    finally:
        for s in srcs:
            s.close()

    with rasterio.open(str(out), "w", **profile) as dst:
        dst.write(mosaic)
        if np.isnan(mosaic).any():
            pass
    return out, len(tile_paths)


def trajectory_bbox_with_buffer(
    latlon_deg: np.ndarray,
    *,
    buffer_m: float = 3000.0,
) -> tuple[float, float, float, float]:
    """Return (south, west, north, east) from trajectory points + metric buffer."""
    if latlon_deg.ndim != 2 or latlon_deg.shape[1] != 2 or latlon_deg.shape[0] == 0:
        raise ValueError("latlon_deg must have shape [N,2] with N>0")
    lats = latlon_deg[:, 0]
    lons = latlon_deg[:, 1]
    lat_c = float(np.mean(lats))
    dlat = _meters_to_deg_lat(buffer_m)
    dlon = _meters_to_deg_lon(buffer_m, lat_c)
    south = float(np.min(lats) - dlat)
    north = float(np.max(lats) + dlat)
    west = float(np.min(lons) - dlon)
    east = float(np.max(lons) + dlon)
    return south, west, north, east
