#!/usr/bin/env python3
"""Fetch OSM buildings in a bbox and export cache, triangles and optional GLB.

Example:
    python experiments/fetch_osm_buildings_bbox.py \
      --south 44.403 --west 8.922 --north 44.413 --east 8.938 \
      --cache-json experiments/results/genova_bbox_osm_cache.json \
      --out-triangles experiments/results/genova_bbox_osm_triangles.npy \
      --export-glb --out-glb experiments/results/genova_bbox_osm.glb
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import requests

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
_PYTHON_PKG = _REPO_ROOT / "python"
if _PYTHON_PKG.is_dir() and str(_PYTHON_PKG) not in sys.path:
    sys.path.insert(0, str(_PYTHON_PKG))

from gnss_gpu.io.osm_buildings import buildings_to_triangles_ecef  # noqa: E402
from gnss_gpu.viz.plateau_glb import export_plateau_roi_glb  # noqa: E402

OVERPASS_ENDPOINTS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
)


def _build_dem_alt_sampler(dem_path: Path):
    import rasterio
    from pyproj import Transformer

    ds = rasterio.open(str(dem_path))
    to_dem = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)

    def sample_alt_m(lat_deg: float, lon_deg: float) -> float:
        x, y = to_dem.transform(float(lon_deg), float(lat_deg))
        v = float(next(ds.sample([(x, y)]))[0])
        return v

    return sample_alt_m


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch OSM buildings for full BBOX.")
    p.add_argument("--south", type=float, required=True)
    p.add_argument("--west", type=float, required=True)
    p.add_argument("--north", type=float, required=True)
    p.add_argument("--east", type=float, required=True)
    p.add_argument("--cache-json", type=Path, required=True, help="Output path for raw Overpass merged JSON")
    p.add_argument("--out-triangles", type=Path, required=True, help="Output path for triangles .npy [N,3,3]")
    p.add_argument("--levels-height-m", type=float, default=3.0, help="Height per OSM level when height is missing")
    p.add_argument("--default-height-m", type=float, default=10.0, help="Default building height [m]")
    p.add_argument("--base-alt-m", type=float, default=0.0, help="Base altitude for footprints [m ellipsoidal]")
    p.add_argument(
        "--dem-path",
        type=Path,
        default=None,
        help="Optional DEM GeoTIFF; when provided, building bases start at local DEM altitude + base-alt-m.",
    )
    p.add_argument("--export-glb", action="store_true", help="Also export GLB sidecar from output triangles")
    p.add_argument("--out-glb", type=Path, default=None, help="Explicit GLB path (default: triangles path with .glb)")
    p.add_argument("--overpass-timeout-s", type=int, default=240)
    return p.parse_args()


def _build_query(south: float, west: float, north: float, east: float) -> str:
    return (
        f"[out:json][timeout:120];"
        f"(way[\"building\"]({south},{west},{north},{east});"
        f"relation[\"building\"]({south},{west},{north},{east}););"
        f"out body geom;"
    )


def main() -> None:
    args = _parse_args()
    query = _build_query(args.south, args.west, args.north, args.east)
    payload = None
    last_error: Exception | None = None
    for endpoint in OVERPASS_ENDPOINTS:
        for attempt in range(5):
            try:
                resp = requests.post(
                    endpoint,
                    data=query,
                    headers={
                        "Content-Type": "application/x-www-form-urlencoded",
                        "User-Agent": "gnss_gpu_bbox_fetch/1.0",
                        "Accept": "application/json,text/plain,*/*",
                    },
                    timeout=int(args.overpass_timeout_s),
                )
                if resp.status_code == 429:
                    wait_s = min(120.0, 2.0**attempt + random.uniform(0.0, 1.0))
                    print(f"[429] {endpoint} retry in {wait_s:.1f}s")
                    time.sleep(wait_s)
                    continue
                resp.raise_for_status()
                payload = resp.json()
                break
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                wait_s = min(60.0, 2.0**attempt + random.uniform(0.0, 1.0))
                time.sleep(wait_s)
        if payload is not None:
            break
    if payload is None:
        if last_error is not None:
            raise last_error
        raise RuntimeError("Overpass building fetch failed without a specific exception.")

    elements = payload.get("elements", [])
    print("OSM elements:", len(elements))

    args.cache_json.parent.mkdir(parents=True, exist_ok=True)
    args.cache_json.write_text(json.dumps({"elements": elements}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved cache:", args.cache_json)

    terrain_alt_fn = _build_dem_alt_sampler(args.dem_path) if args.dem_path is not None else None
    if terrain_alt_fn is not None:
        print(f"using DEM-grounded building bases: {args.dem_path}")

    tri = buildings_to_triangles_ecef(
        elements,
        levels_height_m=float(args.levels_height_m),
        default_height_m=float(args.default_height_m),
        base_alt_m=float(args.base_alt_m),
        terrain_alt_fn=terrain_alt_fn,
    )
    args.out_triangles.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out_triangles, tri)
    print("triangles shape:", tri.shape, "saved:", args.out_triangles)

    if args.export_glb:
        if tri.size == 0:
            raise RuntimeError("Empty triangle mesh: no valid building geometry in bbox.")
        glb_path = args.out_glb or args.out_triangles.with_suffix(".glb")
        pivot = tri.reshape(-1, 3).mean(axis=0)
        kept, total = export_plateau_roi_glb(
            tri,
            pivot,
            glb_path,
            full_mesh=True,
            max_triangles=max(1, int(tri.shape[0])),
        )
        print(f"saved glb: {glb_path} ({kept}/{total} triangles)")


if __name__ == "__main__":
    main()
