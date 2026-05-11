#!/usr/bin/env python3
"""Remove large bottom-closure triangles from an existing ``triangles.npy`` (ECEF ``[N,3,3]``).

See ``mesh_triangle_filters.remove_bottom_closure_slab`` and ``b3dm_tileset_to_triangles_npy.py
--strip-bottom-slab`` for the same heuristic.

Example::

    python experiments/strip_mesh_bottom_slab.py \\
        --input-npy experiments/data/LuigiFerraris/luigiFerrarisGeoLoc/luigi_ferraris_triangles.npy \\
        --output-npy experiments/results/luigi_ferraris_triangles_no_base.npy
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from mesh_triangle_filters import remove_bottom_closure_slab


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-npy", type=Path, required=True)
    ap.add_argument("--output-npy", type=Path, required=True)
    ap.add_argument("--bottom-alt-quantile", type=float, default=0.03)
    ap.add_argument("--bottom-min-area-m2", type=float, default=120.0)
    ap.add_argument("--summary-json", type=Path, default=None)
    args = ap.parse_args()

    inp = args.input_npy.resolve()
    if not inp.is_file():
        sys.exit(f"not found: {inp}")

    tri = np.load(inp)
    n_before = int(tri.shape[0])
    tri_kept, n_drop = remove_bottom_closure_slab(
        tri,
        alt_quantile=float(args.bottom_alt_quantile),
        min_area_m2=float(args.bottom_min_area_m2),
    )

    out = args.output_npy.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, tri_kept)

    summary = {
        "input_npy": str(inp),
        "output_npy": str(out),
        "triangle_count_before": n_before,
        "triangle_count_after": int(tri_kept.shape[0]),
        "removed": int(n_drop),
        "bottom_alt_quantile": float(args.bottom_alt_quantile),
        "bottom_min_area_m2": float(args.bottom_min_area_m2),
    }
    if args.summary_json:
        sj = args.summary_json.resolve()
        sj.parent.mkdir(parents=True, exist_ok=True)
        sj.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        f"Saved {out} shape={tri_kept.shape} (removed {n_drop}/{n_before} triangles)",
        flush=True,
    )


if __name__ == "__main__":
    main()
