"""Export PLATEAU building triangles to minimal binary GLB for CesiumJS.

Vertices are **East / North / Up** offsets (metres) relative to the pivot, derived by
projecting ECEF deltas onto the local tangent basis at ``pivot_ecef`` (same convention as
``Transforms.eastNorthUpToFixedFrame``). Load in CesiumJS with:

``modelMatrix = Transforms.eastNorthUpToFixedFrame(pivotCart)``,
``upAxis: Axis.Z``, ``forwardAxis: Axis.X`` so glTF ``POSITION`` is interpreted as
(x=east, y=north, z=up). Then ``world ≈ pivot + delta_ecef`` matches the ray-traced mesh.
"""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path
from typing import Tuple

import numpy as np


def _ecef_to_lla_rad(x: float, y: float, z: float) -> tuple[float, float, float]:
    """WGS-84 ECEF → (lat, lon, alt) radians / metres (compact duplicate of urban helpers)."""
    a = 6378137.0
    f = 1.0 / 298.257223563
    e2 = 2.0 * f - f * f
    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    lat = math.atan2(z, p * (1.0 - e2))
    for _ in range(10):
        sin_lat = math.sin(lat)
        n = a / math.sqrt(1.0 - e2 * sin_lat * sin_lat)
        lat = math.atan2(z + e2 * n * sin_lat, p)
    sin_lat = math.sin(lat)
    n = a / math.sqrt(1.0 - e2 * sin_lat * sin_lat)
    alt = p / math.cos(lat) - n if abs(math.cos(lat)) > 1e-10 else abs(z) - n * (1.0 - e2)
    return lat, lon, alt


def _ecef_offsets_to_enu(offset_xyz: np.ndarray, pivot_ecef: np.ndarray) -> np.ndarray:
    """Map ECEF Cartesian deltas (corners - pivot) to local East/North/Up [m] at pivot."""
    pivot = np.asarray(pivot_ecef, dtype=np.float64).reshape(3)
    lat, lon, _ = _ecef_to_lla_rad(float(pivot[0]), float(pivot[1]), float(pivot[2]))
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    sin_lon, cos_lon = math.sin(lon), math.cos(lon)
    east = np.array([-sin_lon, cos_lon, 0.0], dtype=np.float64)
    north = np.array([-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat], dtype=np.float64)
    up = np.array([cos_lat * cos_lon, cos_lat * sin_lon, sin_lat], dtype=np.float64)
    r = np.column_stack([east, north, up])
    d = np.asarray(offset_xyz, dtype=np.float64).reshape(-1, 3)
    return d @ r


def export_plateau_roi_glb(
    triangles_ecef: np.ndarray,
    pivot_ecef: np.ndarray,
    out_path: str | Path,
    *,
    radius_m: float = 650.0,
    max_triangles: int = 150_000,
    full_mesh: bool = False,
) -> Tuple[int, int]:
    """Write triangles within ``radius_m`` of ``pivot_ecef`` to GLB.

    Parameters
    ----------
    triangles_ecef
        Array ``(n_tri, 3, 3)`` — corners in ECEF [m].
    pivot_ecef
        Reference point (trajectory centroid) ECEF [m].
    radius_m
        Keep triangles whose centroid lies within this 3-D distance [m] of pivot.
        Ignored when ``full_mesh`` is True.
    max_triangles
        If more triangles pass the ROI filter, evenly subsample to this cap.
    full_mesh
        If True, skip the distance filter and export all triangles (subject only to
        ``max_triangles`` subsampling).

    Returns
    -------
    (n_kept, n_total)
    """
    tri = np.asarray(triangles_ecef, dtype=np.float64)
    if tri.ndim != 3 or tri.shape[1:] != (3, 3):
        raise ValueError("triangles_ecef must have shape (n_tri, 3, 3)")
    pivot = np.asarray(pivot_ecef, dtype=np.float64).reshape(3)
    n_tot = tri.shape[0]
    if n_tot == 0:
        raise ValueError("empty triangle mesh")

    if full_mesh:
        sel = np.arange(n_tot, dtype=np.int64)
    else:
        centers = tri.mean(axis=1)
        diff = centers - pivot
        dist = np.linalg.norm(diff, axis=1)
        mask = dist <= float(radius_m)
        sel = np.nonzero(mask)[0]
        if sel.size == 0:
            raise ValueError(
                f"no triangles within radius_m={radius_m:g} m of pivot - "
                "try a larger --plateau-glb-radius-m or --plateau-glb-full-mesh"
            )

    if sel.size > int(max_triangles):
        idx = np.linspace(0, sel.size - 1, int(max_triangles))
        idx = np.unique(idx.astype(np.int64))
        sel = sel[idx]

    corners = tri[sel].reshape(-1, 3)
    n_corner = corners.shape[0]

    pos = _ecef_offsets_to_enu(corners - pivot, pivot).astype(np.float32).reshape(-1)

    xyz_min = pos.reshape(-1, 3).min(axis=0).tolist()
    xyz_max = pos.reshape(-1, 3).max(axis=0).tolist()

    byte_length = int(pos.nbytes)
    bin_chunk = pos.tobytes()
    pad_bin = (4 - (byte_length % 4)) % 4
    bin_chunk += b"\x00" * pad_bin
    padded_bin_len = len(bin_chunk)

    gltf = {
        "asset": {"version": "2.0", "generator": "gnss_gpu.plateau_glb"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {"POSITION": 0},
                        "material": 0,
                        "mode": 4,
                    }
                ]
            }
        ],
        "materials": [
            {
                "pbrMetallicRoughness": {
                    "baseColorFactor": [0.58, 0.66, 0.74, 0.82],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.88,
                },
                "doubleSided": True,
                "alphaMode": "BLEND",
            }
        ],
        "buffers": [{"byteLength": padded_bin_len}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": byte_length}],
        "accessors": [
            {
                "bufferView": 0,
                "byteOffset": 0,
                "componentType": 5126,
                "count": int(n_corner),
                "type": "VEC3",
                "min": xyz_min,
                "max": xyz_max,
            }
        ],
    }

    json_str = json.dumps(gltf, separators=(",", ":"))
    json_bytes = json_str.encode("utf-8")
    pad_json = (4 - (len(json_bytes) % 4)) % 4
    json_bytes += b" " * pad_json

    chunk0_len = len(json_bytes)
    chunk1_len = len(bin_chunk)
    total_len = 12 + 8 + chunk0_len + 8 + chunk1_len

    out = bytearray()
    out += struct.pack("<I", 0x46546C67)
    out += struct.pack("<I", 2)
    out += struct.pack("<I", total_len)
    out += struct.pack("<I", chunk0_len)
    out += struct.pack("<I", 0x4E4F534A)
    out += json_bytes
    out += struct.pack("<I", chunk1_len)
    out += struct.pack("<I", 0x004E4942)
    out += bin_chunk

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_bytes(out)
    return int(sel.size), int(n_tot)
