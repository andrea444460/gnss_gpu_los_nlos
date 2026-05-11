#!/usr/bin/env python3
"""Reduce triangle count in ``triangles.npy`` (shape ``[N, 3, 3]``, ECEF metres).

Useful before ``build_3d_visualization.py`` / BVH builds when RAM is tight (e.g. Colab).

**quadric** (default when ``fast-simplification`` is installed): merges duplicate vertices
then quadric mesh decimation - better shape preservation than blind subsampling.

**stride**: uniformly subsamples triangles by index (no extra dependency; coarser / may miss
thin features).

Examples::

    pip install fast-simplification

    python experiments/decimate_triangles_npy.py \\
        --input-npy experiments/data/LuigiFerraris/luigi_ferraris_triangles.npy \\
        --output-npy experiments/data/LuigiFerraris/luigi_ferraris_triangles_8k.npy \\
        --target-tris 8000

    python experiments/decimate_triangles_npy.py ... --ratio 0.25 --method stride

Requires: ``numpy``, ``trimesh``. Optional: ``fast-simplification`` for ``quadric`` /
``auto``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _triangle_soup_to_mesh(tri: np.ndarray):
    import trimesh

    tri = np.asarray(tri, dtype=np.float64, order="C")
    if tri.ndim != 3 or tri.shape[1:] != (3, 3):
        raise ValueError(f"expected shape [N, 3, 3], got {tri.shape}")
    v = tri.reshape(-1, 3)
    f = np.arange(len(v), dtype=np.int64).reshape(-1, 3)
    return trimesh.Trimesh(vertices=v, faces=f, process=True)


def decimate_quadric(
    tri: np.ndarray,
    target_faces: int,
    *,
    aggression: int | None = None,
) -> np.ndarray:
    mesh = _triangle_soup_to_mesh(tri)
    n = len(mesh.faces)
    tgt = max(4, min(int(target_faces), n))
    if tgt >= n:
        return np.asarray(tri, dtype=np.float64, order="C")
    kwargs: dict = {"face_count": tgt}
    if aggression is not None:
        kwargs["aggression"] = int(aggression)
    simple = mesh.simplify_quadric_decimation(**kwargs)
    return np.asarray(simple.triangles, dtype=np.float64, order="C")


def decimate_stride(tri: np.ndarray, target_faces: int) -> np.ndarray:
    tri = np.asarray(tri, dtype=np.float64, order="C")
    n = tri.shape[0]
    tgt = max(1, min(int(target_faces), n))
    if tgt >= n:
        return tri
    idx = np.linspace(0, n - 1, num=tgt, dtype=np.float64).astype(np.int64)
    idx = np.unique(idx)
    return tri[idx]


def decimate(
    tri: np.ndarray,
    target_faces: int,
    method: str,
    *,
    aggression: int | None = None,
) -> tuple[np.ndarray, str]:
    m = method.lower().strip()
    if m == "stride":
        return decimate_stride(tri, target_faces), "stride"
    if m == "quadric":
        try:
            return decimate_quadric(tri, target_faces, aggression=aggression), "quadric"
        except ModuleNotFoundError as exc:
            raise SystemExit(
                "quadric requires: pip install fast-simplification\n"
                f"({exc})"
            ) from exc
    if m != "auto":
        raise ValueError(f"unknown method {method!r} (use auto|quadric|stride)")

    try:
        return decimate_quadric(tri, target_faces, aggression=aggression), "quadric"
    except ModuleNotFoundError:
        print(
            "[warn] fast-simplification not installed; using stride subsampling "
            "(pip install fast-simplification for quadric decimation)",
            flush=True,
        )
        return decimate_stride(tri, target_faces), "stride"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-npy", type=Path, required=True)
    ap.add_argument("--output-npy", type=Path, required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--target-tris", type=int, help="Desired triangle count (approx for stride)")
    g.add_argument("--ratio", type=float, help="Fraction of triangles to keep (0 < ratio <= 1)")
    ap.add_argument(
        "--method",
        choices=("auto", "quadric", "stride"),
        default="auto",
        help="Decimation strategy (default: auto = quadric if fast-simplification exists)",
    )
    ap.add_argument(
        "--aggression",
        type=int,
        default=None,
        help="Quadric aggressiveness 0-10 (trimesh; lower = slower, nicer). Omit for library default.",
    )
    ap.add_argument("--summary-json", type=Path, default=None)
    args = ap.parse_args()

    inp = args.input_npy.resolve()
    if not inp.is_file():
        sys.exit(f"input not found: {inp}")

    tri = np.load(inp)
    n0 = int(tri.shape[0])

    if args.ratio is not None:
        r = float(args.ratio)
        if not (0.0 < r <= 1.0):
            sys.exit("--ratio must satisfy 0 < ratio <= 1")
        target = max(1, int(round(n0 * r)))
    else:
        target = int(args.target_tris)
        if target < 1:
            sys.exit("--target-tris must be >= 1")

    out_tri, used = decimate(
        tri,
        target,
        args.method,
        aggression=args.aggression,
    )

    out = args.output_npy.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, out_tri)

    summary = {
        "input_triangles_npy": str(inp),
        "output_triangles_npy": str(out),
        "triangles_in": n0,
        "triangles_out": int(out_tri.shape[0]),
        "method_requested": args.method,
        "method_used": used,
        "target_faces": target,
    }
    if args.summary_json:
        sj = args.summary_json.resolve()
        sj.parent.mkdir(parents=True, exist_ok=True)
        sj.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        f"Saved {out} shape={out_tri.shape} "
        f"(triangles {n0} -> {out_tri.shape[0]}, method={used})",
        flush=True,
    )


if __name__ == "__main__":
    main()
