"""Export PLATEAU building triangles to minimal binary GLB for CesiumJS.

Vertices are **ECEF Cartesian offsets** from a pivot (metres), parallel to ECEF axes — same
vectors as ``corner_ecef - pivot_ecef`` used for ray tracing.

In CesiumJS load with ``modelMatrix = Matrix4.fromTranslation(pivotCart)`` and
``upAxis: Axis.Z``, ``forwardAxis: Axis.X`` so glTF axis correction does not rotate those
offsets; world corners match ``pivotCart + offset`` in the WGS84 ellipsoid frame.
"""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path
from typing import Literal, Tuple

import numpy as np

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = 2.0 * WGS84_F - WGS84_F * WGS84_F


def _ecef_to_lla_rad(x: float, y: float, z: float) -> tuple[float, float, float]:
    lon = math.atan2(y, x)
    p = math.sqrt(x * x + y * y)
    lat = math.atan2(z, p * (1.0 - WGS84_E2))
    for _ in range(10):
        sin_lat = math.sin(lat)
        n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
        lat = math.atan2(z + WGS84_E2 * n * sin_lat, p)
    sin_lat = math.sin(lat)
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    alt = p / max(1e-12, math.cos(lat)) - n
    return lat, lon, alt


def _ecef_delta_to_enu(delta: np.ndarray, lat_rad: float, lon_rad: float) -> np.ndarray:
    d = np.asarray(delta, dtype=np.float64).reshape(-1, 3)
    sl, cl = math.sin(lat_rad), math.cos(lat_rad)
    so, co = math.sin(lon_rad), math.cos(lon_rad)
    dx, dy, dz = d[:, 0], d[:, 1], d[:, 2]
    east = -so * dx + co * dy
    north = -sl * co * dx - sl * so * dy + cl * dz
    up = cl * co * dx + cl * so * dy + sl * dz
    return np.stack([east, north, up], axis=1)


def export_plateau_roi_glb(
    triangles_ecef: np.ndarray,
    pivot_ecef: np.ndarray,
    out_path: str | Path,
    *,
    radius_m: float = 650.0,
    max_triangles: int = 150_000,
    full_mesh: bool = False,
    base_color_factor: tuple[float, float, float, float] | None = None,
    coordinate_frame: Literal["ecef_offset", "enu_yup"] = "ecef_offset",
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
        Warning: uniform triangle subsampling breaks building meshes in GLB viewers
        (scattered triangle shards). Use a high cap for area-wide visualization.
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

    delta = corners - pivot
    if coordinate_frame == "enu_yup":
        lat_rad, lon_rad, _alt = _ecef_to_lla_rad(float(pivot[0]), float(pivot[1]), float(pivot[2]))
        enu = _ecef_delta_to_enu(delta, lat_rad, lon_rad)
        # glTF Y-up: X=east, Y=up, Z=-north (right-handed)
        pos = np.stack([enu[:, 0], enu[:, 2], -enu[:, 1]], axis=1).astype(np.float32).reshape(-1)
    elif coordinate_frame == "ecef_offset":
        pos = delta.astype(np.float32).reshape(-1)
    else:
        raise ValueError(f"unsupported coordinate_frame: {coordinate_frame}")

    xyz_min = pos.reshape(-1, 3).min(axis=0).tolist()
    xyz_max = pos.reshape(-1, 3).max(axis=0).tolist()

    byte_length = int(pos.nbytes)
    bin_chunk = pos.tobytes()
    pad_bin = (4 - (byte_length % 4)) % 4
    bin_chunk += b"\x00" * pad_bin
    padded_bin_len = len(bin_chunk)

    color = list(base_color_factor) if base_color_factor is not None else [0.58, 0.66, 0.74, 0.82]
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
                    "baseColorFactor": color,
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
