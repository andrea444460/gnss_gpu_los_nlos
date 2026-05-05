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
from pathlib import Path

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
        "--latlon-order",
        action="store_true",
        help="For geographic CRS inputs (e.g. epsg:4979), treat x=lat and y=lon instead of x=lon,y=lat.",
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


def _to_ecef(tri: np.ndarray, input_crs: str, latlon_order: bool) -> np.ndarray:
    crs = input_crs.strip().lower()
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
    if crs in ("epsg:4979", "4979", "epsg:4326", "4326"):
        if latlon_order:
            lat = x
            lon = y
        else:
            lon = x
            lat = y
        h = z
        t = Transformer.from_crs("epsg:4979", "epsg:4978", always_xy=True)
        xx, yy, zz = t.transform(lon, lat, h)
    else:
        t = Transformer.from_crs(crs, "epsg:4978", always_xy=True)
        xx, yy, zz = t.transform(x, y, z)
    out = np.stack([xx, yy, zz], axis=1).reshape(tri.shape)
    return np.asarray(out, dtype=np.float64)


def _triangle_area(tri: np.ndarray) -> np.ndarray:
    a = tri[:, 1, :] - tri[:, 0, :]
    b = tri[:, 2, :] - tri[:, 0, :]
    return 0.5 * np.linalg.norm(np.cross(a, b), axis=1)


def main() -> None:
    args = _parse_args()
    root = args.extracted_dir.resolve()
    if not root.exists():
        raise FileNotFoundError(f"Extracted dir not found: {root}")
    gltfs = sorted(list(root.rglob("*.gltf")) + list(root.rglob("*.glb")))
    if not gltfs:
        raise RuntimeError(f"No .gltf/.glb files found under {root}")

    print(f"Scanning meshes: {root}")
    print(f"Found glTF/GLB files: {len(gltfs)}")

    all_tri = []
    loaded = 0
    skipped = 0
    for i, p in enumerate(gltfs, start=1):
        try:
            tri = _load_one_mesh(p)
            if tri is None:
                skipped += 1
                continue
            all_tri.append(tri)
            loaded += 1
            if i % 25 == 0 or i == len(gltfs):
                print(f"  loaded {i}/{len(gltfs)} files (usable={loaded}, skipped={skipped})")
        except Exception as exc:
            skipped += 1
            print(f"  skip {p.name}: {exc}")

    if not all_tri:
        raise RuntimeError("No usable triangles loaded from glTF/GLB files")

    tri = np.concatenate(all_tri, axis=0).astype(np.float64, copy=False)
    tri = _to_ecef(tri, args.input_crs, bool(args.latlon_order))

    finite_mask = np.isfinite(tri).all(axis=(1, 2))
    tri = tri[finite_mask]
    area = _triangle_area(tri)
    tri = tri[area >= float(args.min_triangle_area_m2)]
    if tri.shape[0] == 0:
        raise RuntimeError("All triangles were filtered out; check CRS and mesh content.")

    out_npy = args.output_triangles_npy.resolve()
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_npy, tri)

    mins = tri.reshape(-1, 3).min(axis=0).tolist()
    maxs = tri.reshape(-1, 3).max(axis=0).tolist()
    summary = {
        "extracted_dir": str(root),
        "gltf_files_found": len(gltfs),
        "gltf_files_loaded": loaded,
        "gltf_files_skipped": skipped,
        "input_crs": args.input_crs,
        "triangle_count": int(tri.shape[0]),
        "bbox_ecef_min_m": mins,
        "bbox_ecef_max_m": maxs,
        "output_triangles_npy": str(out_npy),
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Triangles saved: {out_npy} shape={tri.shape}")
    print(f"Summary JSON: {args.summary_json.resolve()}")

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

