#!/usr/bin/env python3
"""Convert a Cesium 3D Tiles folder (b3dm + optional cmpt + tileset.json) to ``triangles.npy``.

Output: ``float64`` array shape ``[N, 3, 3]`` in **ECEF metres** (WGS84), suitable for
``experiments/build_3d_visualization.py --triangles-npy``.

The root ``transform`` from ``tileset.json`` (column-major 4×4, same as Cesium) is applied to
all vertices after decoding embedded glTF binaries inside each ``b3dm``.

Examples::

    python experiments/b3dm_tileset_to_triangles_npy.py \\
        --tileset-dir \"C:/Users/Me/Desktop/luigiFerrarisGeoLoc\" \\
        --output-npy \"C:/Users/Me/Desktop/luigi_ferraris_triangles.npy\"

    # Include tiles unpacked from *.cmpt (may overlap standalone *.b3dm — use if mesh looks incomplete)
    python experiments/b3dm_tileset_to_triangles_npy.py \\
        --tileset-dir ... --output-npy ... --include-cmpt

    # Reduce triangle count (RAM / BVH): see experiments/decimate_triangles_npy.py

Requires: ``trimesh``, ``numpy``.
"""

from __future__ import annotations

import argparse
import io
import json
import struct
import sys
from pathlib import Path

import numpy as np


def _align8(offset: int) -> int:
    return (int(offset) + 7) // 8 * 8


def glb_slice_from_b3dm(blob: bytes) -> bytes:
    """Return embedded GLB bytes from a full ``b3dm`` tile blob."""
    if len(blob) < 28:
        raise ValueError("b3dm too short")
    if blob[:4] != b"b3dm":
        raise ValueError(f"expected b3dm magic, got {blob[:4]!r}")
    ft_json_len = struct.unpack_from("<I", blob, 12)[0]
    ft_bin_len = struct.unpack_from("<I", blob, 16)[0]
    bt_json_len = struct.unpack_from("<I", blob, 20)[0]
    bt_bin_len = struct.unpack_from("<I", blob, 24)[0]
    offset = 28
    offset += int(ft_json_len)
    offset = _align8(offset)
    offset += int(ft_bin_len)
    offset = _align8(offset)
    offset += int(bt_json_len)
    offset = _align8(offset)
    offset += int(bt_bin_len)
    offset = _align8(offset)
    glb = blob[offset:]
    if len(glb) < 12 or glb[:4] != b"glTF":
        raise ValueError("no GLB payload found after b3dm tables")
    return glb


def iter_inner_tiles_cmpt(blob: bytes) -> list[bytes]:
    """Split a ``cmpt`` blob into inner tile byte strings (each starts with its own magic)."""
    if len(blob) < 16:
        raise ValueError("cmpt too short")
    if blob[:4] != b"cmpt":
        raise ValueError(f"expected cmpt magic, got {blob[:4]!r}")
    out: list[bytes] = []
    offset = 16
    while offset < len(blob):
        if offset + 12 > len(blob):
            break
        inner_len = struct.unpack_from("<I", blob, offset + 8)[0]
        if inner_len <= 0 or offset + inner_len > len(blob):
            raise ValueError(f"invalid inner tile length {inner_len} at offset {offset}")
        out.append(blob[offset : offset + inner_len])
        offset += inner_len
    return out


def root_transform_4x4(tileset_path: Path) -> np.ndarray:
    """Column-major 4×4 from tileset root ``transform`` → NumPy 4×4 for ``M @ P_hom``."""
    data = json.loads(tileset_path.read_text(encoding="utf-8"))
    root = data.get("root") or {}
    tflat = root.get("transform")
    if not tflat or len(tflat) != 16:
        raise RuntimeError(f"No root.transform[16] in {tileset_path}")
    return np.array(tflat, dtype=np.float64).reshape((4, 4), order="F")


def apply_transform_xyz(v_xyz: np.ndarray, m_4x4: np.ndarray) -> np.ndarray:
    """Apply ``m_4x4`` to N×3 points (row vectors → homogeneous columns)."""
    v = np.asarray(v_xyz, dtype=np.float64)
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError(f"expected (N,3) vertices, got {v.shape}")
    n = v.shape[0]
    ones = np.ones((n, 1), dtype=np.float64)
    hom = np.hstack([v, ones])  # N×4
    # Column-vector convention: P_out = M @ P.T  → (4,N), then transpose
    out = (m_4x4 @ hom.T).T[:, :3]
    return out


def _decode_draco_triangles_from_glb(glb_bytes: bytes) -> np.ndarray | None:
    """Decode Draco-compressed primitives from GLB and return triangles [T,3,3] in local coords."""
    try:
        import DracoPy
    except Exception:
        return None

    if len(glb_bytes) < 20 or glb_bytes[:4] != b"glTF":
        return None

    # Parse GLB chunks
    offset = 12
    json_chunk = None
    bin_chunk = None
    while offset + 8 <= len(glb_bytes):
        chunk_len = struct.unpack_from("<I", glb_bytes, offset)[0]
        chunk_type = glb_bytes[offset + 4 : offset + 8]
        body = glb_bytes[offset + 8 : offset + 8 + chunk_len]
        if chunk_type == b"JSON":
            try:
                json_chunk = json.loads(body.decode("utf-8"))
            except Exception:
                return None
        elif chunk_type == b"BIN\x00":
            bin_chunk = body
        offset += 8 + chunk_len

    if not isinstance(json_chunk, dict) or bin_chunk is None:
        return None

    mesh_defs = json_chunk.get("meshes") or []
    view_defs = json_chunk.get("bufferViews") or []
    parts: list[np.ndarray] = []
    for mdef in mesh_defs:
        for prim in (mdef.get("primitives") or []):
            ext = (prim.get("extensions") or {}).get("KHR_draco_mesh_compression")
            if not ext:
                continue
            bv_idx = int(ext.get("bufferView", -1))
            if bv_idx < 0 or bv_idx >= len(view_defs):
                continue
            bv = view_defs[bv_idx]
            start = int(bv.get("byteOffset", 0))
            nbyte = int(bv.get("byteLength", 0))
            if nbyte <= 0:
                continue
            raw = bin_chunk[start : start + nbyte]
            if not raw:
                continue
            try:
                dmesh = DracoPy.decode(raw)
            except Exception:
                continue
            pts = np.asarray(dmesh.points, dtype=np.float64)
            faces = np.asarray(dmesh.faces, dtype=np.int64)
            if pts.size == 0 or faces.size == 0:
                continue
            tri = pts[faces]
            if tri.ndim == 3 and tri.shape[1:] == (3, 3) and tri.shape[0] > 0:
                parts.append(np.asarray(tri, dtype=np.float64))

    if not parts:
        return None
    return np.concatenate(parts, axis=0)


def _gltf_yup_to_tile_local_zup(v_xyz: np.ndarray) -> np.ndarray:
    """Map Draco/glTF **Y-up** vertex coords into **Z-up** local metres before ``tile.transform``.

    Cesium 3D Tiles often concatenate rotations assuming glTF-style upright meshes; decoded Draco
    POSITION accessors stay in glTF space (+Y up). Applying the tile root 4×4 directly otherwise
    shears / twists footprints (~90°) vs imagery."""
    v = np.asarray(v_xyz, dtype=np.float64).reshape(-1, 3)
    out = np.empty_like(v)
    # Right-handed +90° about +X: (x,y,z) → (x, -z, y); maps glTF +Y into tile-local +Z.
    out[:, 0] = v[:, 0]
    out[:, 1] = -v[:, 2]
    out[:, 2] = v[:, 1]
    return out


def mesh_triangles_from_glb(glb_bytes: bytes, m_ecef: np.ndarray) -> np.ndarray | None:
    """Return triangles [T,3,3] ECEF or None if empty."""
    try:
        import trimesh
    except Exception as exc:
        raise RuntimeError("trimesh is required (`pip install trimesh`)") from exc

    loaded = trimesh.load(io.BytesIO(glb_bytes), file_type="glb", force="scene")
    mesh = None
    if isinstance(loaded, trimesh.Scene):
        # ``Scene.to_geometry()`` can mis-merge mixed glTF content from Cesium tiles and emit a
        # degenerate triangle soup (duplicate corners). ``to_mesh()`` only concatenates Trimesh
        # instances after baking scene-graph transforms — correct path for b3dm geometry.
        mesh = loaded.to_mesh()
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        return None

    if mesh is None or getattr(mesh, "faces", None) is None or len(mesh.faces) == 0:
        return None
    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.faces, dtype=np.int64)
    if v.size == 0 or f.size == 0:
        return None
    # Some Cesium b3dm tiles use KHR_draco_mesh_compression: generic loaders may yield
    # zeroed POSITION arrays. Detect this and decode Draco payload directly.
    if (v.shape[0] > 0) and (np.max(np.ptp(v, axis=0)) < 1e-12):
        tri_local = _decode_draco_triangles_from_glb(glb_bytes)
        if tri_local is not None:
            tri_flat = _gltf_yup_to_tile_local_zup(tri_local.reshape(-1, 3)).reshape(-1, 3, 3)
            v_ecef = apply_transform_xyz(tri_flat.reshape(-1, 3), m_ecef).reshape(-1, 3, 3)
            tri = np.asarray(v_ecef, dtype=np.float64)
        else:
            v_ecef = apply_transform_xyz(v, m_ecef)
            tri = np.asarray(v_ecef[f], dtype=np.float64)
    else:
        v_ecef = apply_transform_xyz(v, m_ecef)
        tri = np.asarray(v_ecef[f], dtype=np.float64)

    # Drop degenerate faces (parser quirks / duplicate vertex triplets).
    e01 = np.linalg.norm(tri[:, 0] - tri[:, 1], axis=1)
    e12 = np.linalg.norm(tri[:, 1] - tri[:, 2], axis=1)
    e20 = np.linalg.norm(tri[:, 2] - tri[:, 0], axis=1)
    tri = tri[(e01 + e12 + e20) > 1e-9]
    if tri.shape[0] == 0:
        return None
    return tri


def collect_b3dm_blobs(root: Path, *, include_cmpt: bool) -> list[bytes]:
    blobs: list[bytes] = []
    for p in sorted(root.glob("*.b3dm")):
        blobs.append(p.read_bytes())
    if include_cmpt:
        for p in sorted(root.glob("*.cmpt")):
            cmpt = p.read_bytes()
            for inner in iter_inner_tiles_cmpt(cmpt):
                if inner[:4] == b"b3dm":
                    blobs.append(inner)
                elif inner[:4] == b"cmpt":
                    # Nested composite (rare): recurse one level
                    for inner2 in iter_inner_tiles_cmpt(inner):
                        if inner2[:4] == b"b3dm":
                            blobs.append(inner2)
    return blobs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tileset-dir", type=Path, required=True, help="Folder with tileset.json + b3dm/cmpt")
    ap.add_argument("--tileset-json", type=Path, default=None, help="Explicit tileset path (default: DIR/tileset.json)")
    ap.add_argument("--output-npy", type=Path, required=True)
    ap.add_argument(
        "--include-cmpt",
        action="store_true",
        help="Also decode every *.cmpt inner b3dm (can duplicate geometry vs loose *.b3dm)",
    )
    ap.add_argument("--summary-json", type=Path, default=None)
    args = ap.parse_args()

    ts_dir = args.tileset_dir.resolve()
    ts_path = args.tileset_json.resolve() if args.tileset_json else ts_dir / "tileset.json"
    if not ts_path.is_file():
        sys.exit(f"tileset not found: {ts_path}")

    m_ecef = root_transform_4x4(ts_path)
    blobs = collect_b3dm_blobs(ts_dir, include_cmpt=bool(args.include_cmpt))

    if not blobs:
        sys.exit(f"No *.b3dm files under {ts_dir} (use --include-cmpt if only composite tiles exist).")

    all_parts: list[np.ndarray] = []
    ok = 0
    for bi, blob in enumerate(blobs):
        try:
            glb = glb_slice_from_b3dm(blob)
            tri = mesh_triangles_from_glb(glb, m_ecef)
            if tri is None:
                continue
            all_parts.append(tri)
            ok += 1
        except Exception as exc:
            print(f"[warn] tile {bi}: skipped ({exc})", flush=True)

    if not all_parts:
        sys.exit("No triangles extracted from any b3dm.")

    merged = np.concatenate(all_parts, axis=0)
    verts_flat = merged.reshape(-1, 3)
    extent_m = float(np.linalg.norm(verts_flat.max(axis=0) - verts_flat.min(axis=0)))
    nuniq_mm = int(np.unique(np.round(verts_flat, decimals=3), axis=0).shape[0])

    out = args.output_npy.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, merged)

    summary = {
        "tileset_dir": str(ts_dir),
        "tileset_json": str(ts_path),
        "include_cmpt": bool(args.include_cmpt),
        "b3dm_tiles_read": len(blobs),
        "tiles_decoded_ok": ok,
        "triangle_count": int(merged.shape[0]),
        "mesh_extent_m": extent_m,
        "unique_vertices_rounded_mm": nuniq_mm,
        "output_triangles_npy": str(out),
    }
    if args.summary_json:
        sj = args.summary_json.resolve()
        sj.parent.mkdir(parents=True, exist_ok=True)
        sj.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        f"Saved {out} shape={merged.shape} (triangles={merged.shape[0]}, tiles_ok={ok}/{len(blobs)}, "
        f"extent_m={extent_m:.3f}, unique_verts~={nuniq_mm})",
        flush=True,
    )
    if extent_m < 1.0 or nuniq_mm < 32:
        print(
            "[warn] Mesh extent or vertex diversity looks degenerate. "
            "Viewer mesh GLB will appear as a point/speck until triangles.npy is regenerated.",
            flush=True,
        )


if __name__ == "__main__":
    main()
