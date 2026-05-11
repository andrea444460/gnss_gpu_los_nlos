#!/usr/bin/env python3
"""Build raytrace-ready triangle mesh from extracted HK glTF tiles.

This script scans a directory of extracted HK tiles (containing .gltf/.glb),
loads geometry, converts coordinates to ECEF when needed, and writes one
triangle tensor compatible with gnss_gpu ray tracing:

    triangles: float64 array with shape [N, 3, 3] in ECEF meters.

It also prints ready-to-run commands for:
  - experiments/export_los_labels_from_urbannav.py
  - experiments/build_3d_visualization.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable

import numpy as np


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert extracted HK glTF tiles to ECEF triangle mesh (.npy)")
    p.add_argument("--extracted-dir", type=Path, required=True, help="Root directory containing extracted .gltf/.glb files")
    p.add_argument(
        "--output-triangles-npy",
        type=Path,
        required=True,
        help="Output .npy path (triangles [N,3,3] ECEF meters)",
    )
    p.add_argument(
        "--input-crs",
        type=str,
        default="epsg:4978",
        help="Input CRS of glTF vertex coordinates. Examples: epsg:4978, epsg:2326, epsg:4979",
    )
    p.add_argument(
        "--hk-local-axis",
        action="store_true",
        help="HK non-textured glTF axis fix: interpret transformed vertices as E=X, N=-Z, H=Y before CRS conversion",
    )
    p.add_argument(
        "--latlon-order",
        action="store_true",
        help="For geographic CRS inputs (e.g. epsg:4979), treat x=lat and y=lon instead of x=lon,y=lat.",
    )
    p.add_argument(
        "--geoid-correction",
        type=str,
        default="none",
        help=(
            "Vertical correction before ECEF conversion: "
            "'none' (default), 'egm96', 'hkapi', or a numeric constant in meters."
        ),
    )
    p.add_argument(
        "--hkapi-url",
        type=str,
        default="https://www.geodetic.gov.hk/transform/v2/",
        help="LandsD transformation API endpoint used when --geoid-correction hkapi",
    )
    p.add_argument(
        "--hkapi-grid-step-m",
        type=float,
        default=100.0,
        help="Grid spacing [m] for HKAPI geoid sampling (default 100)",
    )
    p.add_argument(
        "--hkapi-cache-json",
        type=Path,
        default=Path("experiments/data/hkapi_geoid_cache.json"),
        help="Persistent JSON cache for HKAPI sampled nodes",
    )
    p.add_argument(
        "--hkapi-timeout-s",
        type=float,
        default=20.0,
        help="HTTP timeout in seconds for HKAPI requests (default 20)",
    )
    p.add_argument(
        "--hkapi-retries",
        type=int,
        default=2,
        help="Retry count for failed HKAPI requests (default 2)",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Print detailed progress logs (useful to diagnose apparent stalls)",
    )
    p.add_argument(
        "--min-triangle-area-m2",
        type=float,
        default=1e-8,
        help="Drop near-degenerate triangles below this area (default 1e-8)",
    )
    p.add_argument(
        "--summary-json",
        type=Path,
        default=Path("experiments/results/hk_mesh_summary.json"),
        help="Write a summary JSON report",
    )
    p.add_argument(
        "--include-categories",
        type=str,
        default="",
        help=(
            "Comma-separated top-level folder filters under extracted dir, "
            "e.g. BUILDING or BUILDING,INFRASTRUCTURE. "
            "Empty means include all."
        ),
    )
    p.add_argument("--reference-csv", type=Path, default=Path(""), help="Optional reference.csv for printed commands")
    p.add_argument("--obs-path", type=Path, default=Path(""), help="Optional rover obs path for printed commands")
    p.add_argument("--nav-path", type=Path, default=Path(""), help="Optional nav path for printed commands")
    return p.parse_args()


def _load_one_mesh(mesh_path: Path):
    try:
        import trimesh
    except Exception as exc:
        raise RuntimeError("Missing dependency: trimesh. Install with `pip install trimesh`") from exc
    scene_or_mesh = trimesh.load(str(mesh_path), force="scene")
    if isinstance(scene_or_mesh, trimesh.Scene):
        mesh = scene_or_mesh.dump(concatenate=True)
    else:
        mesh = scene_or_mesh
    if mesh is None or len(mesh.faces) == 0:
        return None
    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.faces, dtype=np.int64)
    if v.size == 0 or f.size == 0:
        return None
    tri = v[f]
    return tri


def _resolve_geoid_correction(spec: str) -> tuple[Callable[[np.ndarray, np.ndarray], np.ndarray] | None, str]:
    s = str(spec).strip().lower()
    if s in ("", "none", "off", "false", "0"):
        return None, "none"
    try:
        c = float(s)
        return (
            lambda lat_deg, lon_deg: np.full_like(np.asarray(lat_deg, dtype=np.float64), c, dtype=np.float64),
            f"constant({c:g}m)",
        )
    except ValueError:
        pass
    if s == "hkapi":
        return None, "hkapi"
    if s != "egm96":
        raise ValueError(f"Unsupported --geoid-correction={spec!r}; use none, egm96, hkapi, or numeric meters.")
    try:
        from pyproj import Transformer
    except Exception as exc:
        raise RuntimeError("geoid correction 'egm96' requires pyproj (`pip install pyproj`).") from exc

    src = "+proj=longlat +ellps=WGS84 +geoidgrids=egm96_15.gtx +vunits=m +no_defs"
    dst = "+proj=longlat +ellps=WGS84 +vunits=m +no_defs"
    tr = Transformer.from_pipeline(f"{src} +step {dst}")

    def _n(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
        lon = np.asarray(lon_deg, dtype=np.float64)
        lat = np.asarray(lat_deg, dtype=np.float64)
        z0 = np.zeros_like(lat, dtype=np.float64)
        _x, _y, z = tr.transform(lon, lat, z0)
        return np.asarray(z, dtype=np.float64)

    return _n, "egm96"


def _hkapi_undulation_on_hkgrid(
    e: np.ndarray,
    n: np.ndarray,
    *,
    api_url: str,
    step_m: float,
    cache_json: Path,
    timeout_s: float,
    retries: int,
    debug: bool = False,
) -> np.ndarray:
    """Return geoid undulation N(e,n) [m] sampled from LandsD API then bilinearly interpolated."""
    e = np.asarray(e, dtype=np.float64).reshape(-1)
    n = np.asarray(n, dtype=np.float64).reshape(-1)
    if e.size == 0:
        return np.zeros(0, dtype=np.float64)
    step = max(float(step_m), 1.0)
    e0 = math.floor(float(np.min(e)) / step) * step
    e1 = math.ceil(float(np.max(e)) / step) * step
    n0 = math.floor(float(np.min(n)) / step) * step
    n1 = math.ceil(float(np.max(n)) / step) * step
    e_nodes = np.arange(e0, e1 + step * 0.5, step, dtype=np.float64)
    n_nodes = np.arange(n0, n1 + step * 0.5, step, dtype=np.float64)
    if e_nodes.size < 2:
        e_nodes = np.array([e0, e0 + step], dtype=np.float64)
    if n_nodes.size < 2:
        n_nodes = np.array([n0, n0 + step], dtype=np.float64)

    cache: dict[str, float] = {}
    if cache_json.exists():
        try:
            cache = {k: float(v) for k, v in json.loads(cache_json.read_text(encoding="utf-8")).items()}
        except Exception:
            cache = {}
    if debug:
        print(f"HKAPI cache file: {cache_json} (entries={len(cache)})", flush=True)

    def _key(ee: float, nn: float) -> str:
        return f"{ee:.3f},{nn:.3f}"

    def _fetch_node(ee: float, nn: float) -> float:
        k = _key(ee, nn)
        if k in cache:
            return cache[k]
        params = urllib.parse.urlencode(
            {
                "inSys": "hkgrid",
                "e": f"{ee:.3f}",
                "n": f"{nn:.3f}",
                # Interpret input height as HKPD=0; returned wgsEllHgt is undulation N.
                "h": "0",
            }
        )
        url = api_url.rstrip("?") + ("&" if "?" in api_url else "?") + params
        last_err: Exception | None = None
        for attempt in range(max(1, int(retries) + 1)):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "gnss_gpu/hkapi-geoid"})
                with urllib.request.urlopen(req, timeout=float(timeout_s)) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                if "wgsEllHgt" not in data:
                    raise RuntimeError(f"HKAPI response missing wgsEllHgt for ({ee},{nn}): {data}")
                v = float(data["wgsEllHgt"])
                if not np.isfinite(v):
                    raise RuntimeError(f"HKAPI response non-finite wgsEllHgt for ({ee},{nn}): {data}")
                cache[k] = v
                return v
            except Exception as exc:
                last_err = exc
                if debug:
                    print(
                        f"  HKAPI node retry {attempt + 1}/{max(1, int(retries) + 1)} for e={ee:.3f}, n={nn:.3f}: {exc}",
                        flush=True,
                    )
        raise RuntimeError(f"HKAPI failed for ({ee},{nn}) after retries: {last_err}")

    total = int(e_nodes.size * n_nodes.size)
    print(f"HKAPI geoid sampling grid: {n_nodes.size}x{e_nodes.size} ({total} nodes, step={step:g}m)", flush=True)
    grid = np.zeros((n_nodes.size, e_nodes.size), dtype=np.float64)
    done = 0
    cache_hits = 0
    cache_miss = 0
    for i, nn in enumerate(n_nodes):
        for j, ee in enumerate(e_nodes):
            k = _key(float(ee), float(nn))
            if k in cache:
                cache_hits += 1
                grid[i, j] = cache[k]
            else:
                cache_miss += 1
                grid[i, j] = _fetch_node(float(ee), float(nn))
            done += 1
        if i % 10 == 0 or i == n_nodes.size - 1:
            print(
                f"  HKAPI rows {i + 1}/{n_nodes.size} (nodes {done}/{total}, cache_hit={cache_hits}, cache_miss={cache_miss})",
                flush=True,
            )

    cache_json.parent.mkdir(parents=True, exist_ok=True)
    cache_json.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    if debug:
        print(f"HKAPI cache saved: {cache_json} (entries={len(cache)})", flush=True)

    ie = np.searchsorted(e_nodes, e, side="right") - 1
    in_ = np.searchsorted(n_nodes, n, side="right") - 1
    ie = np.clip(ie, 0, e_nodes.size - 2)
    in_ = np.clip(in_, 0, n_nodes.size - 2)
    e_lo = e_nodes[ie]
    e_hi = e_nodes[ie + 1]
    n_lo = n_nodes[in_]
    n_hi = n_nodes[in_ + 1]
    tx = np.where(e_hi > e_lo, (e - e_lo) / (e_hi - e_lo), 0.0)
    ty = np.where(n_hi > n_lo, (n - n_lo) / (n_hi - n_lo), 0.0)
    z00 = grid[in_, ie]
    z10 = grid[in_, ie + 1]
    z01 = grid[in_ + 1, ie]
    z11 = grid[in_ + 1, ie + 1]
    out = (1.0 - tx) * (1.0 - ty) * z00 + tx * (1.0 - ty) * z10 + (1.0 - tx) * ty * z01 + tx * ty * z11
    return np.asarray(out, dtype=np.float64)


def _to_ecef(
    tri: np.ndarray,
    input_crs: str,
    latlon_order: bool,
    hk_local_axis: bool = False,
    geoid_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    geoid_label: str = "none",
    hkapi_url: str = "",
    hkapi_grid_step_m: float = 100.0,
    hkapi_cache_json: Path | None = None,
    hkapi_timeout_s: float = 20.0,
    hkapi_retries: int = 2,
    debug: bool = False,
) -> np.ndarray:
    crs = input_crs.strip().lower()
    if hk_local_axis:
        # HK non-textured glTFs often encode transformed vertices where:
        #   X -> Easting, Y -> Height, Z -> -Northing
        # Convert to (E, N, H) before any CRS transform.
        tri = np.stack([tri[..., 0], -tri[..., 2], tri[..., 1]], axis=-1)
    if crs in ("ecef", "epsg:4978", "4978"):
        return tri
    try:
        from pyproj import Transformer
    except Exception as exc:
        raise RuntimeError(
            "CRS conversion requested but pyproj is missing. Install with `pip install pyproj`."
        ) from exc

    x = tri[..., 0].reshape(-1)
    y = tri[..., 1].reshape(-1)
    z = tri[..., 2].reshape(-1)
    if debug:
        print(
            f"CRS transform start: input_crs={crs}, vertices={x.size}, geoid={geoid_label}, hk_local_axis={hk_local_axis}",
            flush=True,
        )
    if crs in ("epsg:4979", "4979", "epsg:4326", "4326"):
        if latlon_order:
            lat = x
            lon = y
        else:
            lon = x
            lat = y
        h = z
        if geoid_fn is not None:
            h = h + geoid_fn(lat, lon)
        elif geoid_label == "hkapi":
            t_hk = Transformer.from_crs("epsg:4979", "epsg:2326", always_xy=True)
            e_hk, n_hk, _ = t_hk.transform(lon, lat, np.zeros_like(h))
            h = h + _hkapi_undulation_on_hkgrid(
                e_hk,
                n_hk,
                api_url=hkapi_url,
                step_m=hkapi_grid_step_m,
                cache_json=hkapi_cache_json or Path("experiments/data/hkapi_geoid_cache.json"),
                timeout_s=hkapi_timeout_s,
                retries=hkapi_retries,
                debug=debug,
            )
        t = Transformer.from_crs("epsg:4979", "epsg:4978", always_xy=True)
        xx, yy, zz = t.transform(lon, lat, h)
    elif crs in ("ecef", "epsg:4978", "4978"):
        if geoid_fn is not None:
            print("Warning: --geoid-correction ignored for ECEF input CRS (already Cartesian).")
        xx, yy, zz = x, y, z
    else:
        if geoid_fn is None:
            if geoid_label != "hkapi":
                t = Transformer.from_crs(crs, "epsg:4978", always_xy=True)
                xx, yy, zz = t.transform(x, y, z)
            else:
                # hkapi needs HK grid coordinates: apply N(e,n) on HKPD heights first.
                if crs not in ("epsg:2326", "2326"):
                    raise ValueError("--geoid-correction hkapi currently supports input CRS epsg:2326/4979/4326 only.")
                und = _hkapi_undulation_on_hkgrid(
                    x,
                    y,
                    api_url=hkapi_url,
                    step_m=hkapi_grid_step_m,
                    cache_json=hkapi_cache_json or Path("experiments/data/hkapi_geoid_cache.json"),
                    timeout_s=hkapi_timeout_s,
                    retries=hkapi_retries,
                    debug=debug,
                )
                t = Transformer.from_crs(crs, "epsg:4978", always_xy=True)
                xx, yy, zz = t.transform(x, y, z + und)
        else:
            t_to_geo = Transformer.from_crs(crs, "epsg:4979", always_xy=True)
            lon, lat, h = t_to_geo.transform(x, y, z)
            h = np.asarray(h, dtype=np.float64) + geoid_fn(np.asarray(lat, dtype=np.float64), np.asarray(lon, dtype=np.float64))
            t_to_ecef = Transformer.from_crs("epsg:4979", "epsg:4978", always_xy=True)
            xx, yy, zz = t_to_ecef.transform(lon, lat, h)
    out = np.stack([xx, yy, zz], axis=1).reshape(tri.shape)
    if debug:
        print("CRS transform done.", flush=True)
    return np.asarray(out, dtype=np.float64)


def _triangle_area(tri: np.ndarray) -> np.ndarray:
    a = tri[:, 1, :] - tri[:, 0, :]
    b = tri[:, 2, :] - tri[:, 0, :]
    return 0.5 * np.linalg.norm(np.cross(a, b), axis=1)


def main() -> None:
    args = _parse_args()
    geoid_fn, geoid_label = _resolve_geoid_correction(args.geoid_correction)
    root = args.extracted_dir.resolve()
    if not root.exists():
        raise FileNotFoundError(f"Extracted dir not found: {root}")
    gltfs = sorted(list(root.rglob("*.gltf")) + list(root.rglob("*.glb")))
    include_categories: set[str] = set()
    if args.include_categories.strip():
        include_categories = {
            x.strip().upper() for x in args.include_categories.split(",") if x.strip()
        }
        gltfs = [
            p
            for p in gltfs
            if any(part.upper() in include_categories for part in p.parts)
        ]
    if not gltfs:
        raise RuntimeError(f"No .gltf/.glb files found under {root}")

    print(f"Scanning meshes: {root}", flush=True)
    print(f"Found glTF/GLB files: {len(gltfs)}", flush=True)

    all_tri = []
    loaded = 0
    skipped = 0
    for i, p in enumerate(gltfs, start=1):
        try:
            if args.debug:
                print(f"  loading [{i}/{len(gltfs)}] {p.name}", flush=True)
            tri = _load_one_mesh(p)
            if tri is None:
                skipped += 1
                continue
            all_tri.append(tri)
            loaded += 1
            if i % 25 == 0 or i == len(gltfs):
                print(f"  loaded {i}/{len(gltfs)} files (usable={loaded}, skipped={skipped})", flush=True)
        except Exception as exc:
            skipped += 1
            print(f"  skip {p.name}: {exc}", flush=True)

    if not all_tri:
        raise RuntimeError("No usable triangles loaded from glTF/GLB files")

    print("Concatenating triangle arrays...", flush=True)
    tri = np.concatenate(all_tri, axis=0).astype(np.float64, copy=False)
    print(f"Triangle count before CRS conversion: {tri.shape[0]}", flush=True)
    print("Starting CRS/ECEF conversion...", flush=True)
    tri = _to_ecef(
        tri,
        args.input_crs,
        bool(args.latlon_order),
        hk_local_axis=bool(args.hk_local_axis),
        geoid_fn=geoid_fn,
        geoid_label=geoid_label,
        hkapi_url=str(args.hkapi_url),
        hkapi_grid_step_m=float(args.hkapi_grid_step_m),
        hkapi_cache_json=args.hkapi_cache_json.resolve(),
        hkapi_timeout_s=float(args.hkapi_timeout_s),
        hkapi_retries=int(args.hkapi_retries),
        debug=bool(args.debug),
    )

    print("Filtering invalid/degenerate triangles...", flush=True)
    finite_mask = np.isfinite(tri).all(axis=(1, 2))
    tri = tri[finite_mask]
    area = _triangle_area(tri)
    tri = tri[area >= float(args.min_triangle_area_m2)]
    if tri.shape[0] == 0:
        raise RuntimeError("All triangles were filtered out; check CRS and mesh content.")

    out_npy = args.output_triangles_npy.resolve()
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving triangles to {out_npy} ...", flush=True)
    np.save(out_npy, tri)

    mins = tri.reshape(-1, 3).min(axis=0).tolist()
    maxs = tri.reshape(-1, 3).max(axis=0).tolist()
    summary = {
        "extracted_dir": str(root),
        "include_categories": sorted(include_categories) if include_categories else "ALL",
        "gltf_files_found": len(gltfs),
        "gltf_files_loaded": loaded,
        "gltf_files_skipped": skipped,
        "input_crs": args.input_crs,
        "hk_local_axis": bool(args.hk_local_axis),
        "geoid_correction": geoid_label,
        "triangle_count": int(tri.shape[0]),
        "bbox_ecef_min_m": mins,
        "bbox_ecef_max_m": maxs,
        "output_triangles_npy": str(out_npy),
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Triangles saved: {out_npy} shape={tri.shape}", flush=True)
    print(f"Summary JSON: {args.summary_json.resolve()}", flush=True)

    tri_arg = str(out_npy)
    if str(args.obs_path):
        nav_arg = f'--nav-path "{args.nav_path}" ' if str(args.nav_path) else ""
        ref_arg = f'--reference-csv "{args.reference_csv}" ' if str(args.reference_csv) else ""
        obs_filter_arg = f'--obs "{args.obs_path}" --filter-by-obs ' if str(args.obs_path) else ""
        print("\nSuggested LOS-label command:")
        print(
            "PYTHONPATH=python python experiments/export_los_labels_from_urbannav.py "
            f"--obs-path \"{args.obs_path}\" "
            f"{nav_arg}"
            f"{ref_arg}"
            f"--triangles-npy \"{tri_arg}\" "
            "--output-csv \"experiments/results/hk_los_labels.csv\" --systems G,R,E,C,J,I,S --batch-size 512"
        )
        print("\nSuggested 3D visualization command:")
        print(
            "PYTHONPATH=python python experiments/build_3d_visualization.py "
            "--area-name HK_KLT "
            f"{ref_arg}"
            f"--triangles-npy \"{tri_arg}\" "
            f"{obs_filter_arg}"
            "--n-epochs 500 --epoch-min-interval-s 1 --out-html \"experiments/results/hk_klt_viz.html\""
        )


if __name__ == "__main__":
    main()

