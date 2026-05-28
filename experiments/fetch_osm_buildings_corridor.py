#!/usr/bin/env python3
"""Fetch OSM buildings around trajectory corridor and export mesh artifacts.

Outputs:
- raw Overpass elements JSON
- ECEF triangles NPY with shape [N,3,3]
- optional GLB sidecar for visualization
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
_PYTHON_PKG = _REPO_ROOT / "python"
if _PYTHON_PKG.is_dir() and str(_PYTHON_PKG) not in sys.path:
    sys.path.insert(0, str(_PYTHON_PKG))

from gnss_gpu.io.osm_buildings import (  # noqa: E402
    build_corridor_tiles,
    buildings_to_triangles_ecef,
    fetch_buildings_overpass,
    load_trajectory_latlon,
)
from gnss_gpu.viz.plateau_glb import export_plateau_roi_glb  # noqa: E402


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
    p = argparse.ArgumentParser(description="Fetch OSM building mesh along trajectory corridor")
    p.add_argument("--reference-csv", type=Path, required=True, help="Trajectory CSV (UrbanNav or lat/lon format)")
    p.add_argument("--buffer-m", type=float, default=100.0, help="Corridor buffer radius [m]")
    p.add_argument("--tile-step-m", type=float, default=500.0, help="Tile spacing [m] used for Overpass chunking")
    p.add_argument("--levels-height-m", type=float, default=3.0, help="Height per OSM level when height is missing")
    p.add_argument("--default-height-m", type=float, default=10.0, help="Default building height [m]")
    p.add_argument("--base-alt-m", type=float, default=0.0, help="Base altitude for footprints [m ellipsoidal]")
    p.add_argument(
        "--dem-path",
        type=Path,
        default=None,
        help="Optional DEM GeoTIFF; when provided, building bases start at local DEM altitude + base-alt-m.",
    )
    p.add_argument("--cache-json", type=Path, required=True, help="Output path for raw Overpass merged JSON")
    p.add_argument("--out-triangles", type=Path, required=True, help="Output path for triangles .npy [N,3,3]")
    p.add_argument("--export-glb", action="store_true", help="Also export GLB sidecar from output triangles")
    p.add_argument("--out-glb", type=Path, default=None, help="Explicit GLB path (default: triangles path with .glb)")
    p.add_argument("--overpass-timeout-s", type=int, default=120, help="Timeout per Overpass request")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    latlon = load_trajectory_latlon(args.reference_csv)
    print(f"trajectory samples: {len(latlon)}")

    tiles = build_corridor_tiles(
        latlon,
        buffer_m=float(args.buffer_m),
        tile_step_m=float(args.tile_step_m),
    )
    print(f"overpass tiles: {len(tiles)}")
    if not tiles:
        raise RuntimeError("No tiles produced from trajectory corridor.")

    elements = fetch_buildings_overpass(tiles, timeout_s=int(args.overpass_timeout_s))
    print(f"unique OSM elements: {len(elements)}")
    args.cache_json.parent.mkdir(parents=True, exist_ok=True)
    args.cache_json.write_text(json.dumps({"elements": elements}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved cache json: {args.cache_json}")

    height_stats: dict[str, int] = {}
    terrain_alt_fn = _build_dem_alt_sampler(args.dem_path) if args.dem_path is not None else None
    if terrain_alt_fn is not None:
        print(f"using DEM-grounded building bases: {args.dem_path}")
    tri = buildings_to_triangles_ecef(
        elements,
        levels_height_m=float(args.levels_height_m),
        default_height_m=float(args.default_height_m),
        base_alt_m=float(args.base_alt_m),
        terrain_alt_fn=terrain_alt_fn,
        stats=height_stats,
    )
    if tri.ndim != 3 or tri.shape[1:] != (3, 3):
        raise RuntimeError(f"Invalid triangle mesh shape: {tri.shape}")
    args.out_triangles.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out_triangles, tri)
    print(f"saved triangles: {args.out_triangles} ({tri.shape[0]} triangles)")
    print(
        "height stats: "
        f"defined={height_stats.get('height_defined_count', 0)}, "
        f"generated={height_stats.get('height_generated_count', 0)} "
        f"(from_levels={height_stats.get('generated_from_levels_count', 0)}, "
        f"default={height_stats.get('generated_default_count', 0)}), "
        f"meshed={height_stats.get('meshed_buildings', 0)}, "
        f"skipped_no_valid_geometry={height_stats.get('skipped_no_valid_geometry_count', 0)}"
    )

    if args.export_glb:
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
