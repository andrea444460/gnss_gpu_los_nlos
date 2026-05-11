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


def mesh_triangles_from_glb(glb_bytes: bytes, m_ecef: np.ndarray) -> np.ndarray | None:
    """Return triangles [T,3,3] ECEF or None if empty."""
    try:
        import trimesh
    except Exception as exc:
        raise RuntimeError("trimesh is required (`pip install trimesh`)") from exc

    scene_or_mesh = trimesh.load(io.BytesIO(glb_bytes), file_type="glb", force="scene")
    if isinstance(scene_or_mesh, trimesh.Scene):
        dump = getattr(scene_or_mesh, "to_geometry", None)
        mesh = dump() if callable(dump) else scene_or_mesh.dump(concatenate=True)
    else:
        mesh = scene_or_mesh
    if mesh is None or mesh.faces is None or len(mesh.faces) == 0:
        return None
    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.faces, dtype=np.int64)
    if v.size == 0 or f.size == 0:
        return None
    v_ecef = apply_transform_xyz(v, m_ecef)
    tri = v_ecef[f]  # [T,3,3]
    return np.asarray(tri, dtype=np.float64)


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
        "output_triangles_npy": str(out),
    }
    if args.summary_json:
        sj = args.summary_json.resolve()
        sj.parent.mkdir(parents=True, exist_ok=True)
        sj.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved {out} shape={merged.shape} (triangles={merged.shape[0]}, tiles_ok={ok}/{len(blobs)})")


if __name__ == "__main__":
    main()
