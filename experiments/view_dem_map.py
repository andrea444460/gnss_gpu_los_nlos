#!/usr/bin/env python3
"""Render a DEM GeoTIFF overlay on an interactive Folium map."""

from __future__ import annotations

import argparse
from pathlib import Path

import folium
import matplotlib.cm as cm
import numpy as np
import rasterio
from rasterio.warp import Resampling, calculate_default_transform, reproject, transform_bounds


def _reproject_to_wgs84(ds: rasterio.io.DatasetReader, band1: np.ndarray) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    dst_crs = "EPSG:4326"
    transform, width, height = calculate_default_transform(
        ds.crs, dst_crs, ds.width, ds.height, *ds.bounds
    )
    dst = np.full((height, width), np.nan, dtype=np.float32)
    reproject(
        source=band1,
        destination=dst,
        src_transform=ds.transform,
        src_crs=ds.crs,
        dst_transform=transform,
        dst_crs=dst_crs,
        src_nodata=ds.nodata,
        dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )
    west, south, east, north = transform_bounds(ds.crs, dst_crs, *ds.bounds, densify_pts=21)
    return dst, (south, west, north, east)


def _rgba_from_dem(arr: np.ndarray) -> np.ndarray:
    valid = np.isfinite(arr)
    if not np.any(valid):
        raise ValueError("DEM has no finite values after reprojection.")
    vals = arr[valid]
    vmin = float(np.percentile(vals, 2))
    vmax = float(np.percentile(vals, 98))
    if vmax <= vmin:
        vmax = vmin + 1.0
    norm = np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0)
    rgba = cm.get_cmap("terrain")(norm)  # [H,W,4] float 0..1
    rgba[..., 3] = np.where(valid, 0.72, 0.0)
    return (rgba * 255).astype(np.uint8)


def main() -> None:
    p = argparse.ArgumentParser(description="Create Folium map with DEM overlay")
    p.add_argument("--dem-path", type=Path, required=True, help="Input DEM GeoTIFF")
    p.add_argument("--out-html", type=Path, required=True, help="Output HTML map path")
    p.add_argument("--zoom-start", type=int, default=13, help="Initial map zoom")
    args = p.parse_args()

    with rasterio.open(args.dem_path) as ds:
        band1 = ds.read(1).astype(np.float32)
        if ds.nodata is not None:
            band1[band1 == ds.nodata] = np.nan
        dem_wgs84, (south, west, north, east) = _reproject_to_wgs84(ds, band1)

    rgba = _rgba_from_dem(dem_wgs84)
    center_lat = (south + north) * 0.5
    center_lon = (west + east) * 0.5

    m = folium.Map(location=[center_lat, center_lon], zoom_start=int(args.zoom_start), tiles="OpenStreetMap")
    folium.raster_layers.ImageOverlay(
        image=rgba,
        bounds=[[south, west], [north, east]],
        opacity=0.9,
        name="DEM",
        interactive=True,
        cross_origin=False,
    ).add_to(m)
    folium.LayerControl().add_to(m)

    args.out_html.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(args.out_html))
    print(f"saved map html: {args.out_html}")
    print(f"bounds lat/lon: south={south:.6f}, west={west:.6f}, north={north:.6f}, east={east:.6f}")


if __name__ == "__main__":
    main()
