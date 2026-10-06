#!/usr/bin/env python3
"""Export PLATEAU CityGML directory to ECEF triangles.npy for ray tracing.

Example (Colab / local)::

    PYTHONPATH=python python experiments/plateau_dir_to_triangles_npy.py \\
      --plateau-dir experiments/data/plateau_odaiba \\
      --plateau-zone 9 \\
      --output-npy experiments/results/odaiba_triangles.npy
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_ROOT / "python"))

from gnss_gpu.io.plateau import PlateauLoader


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plateau-dir", type=Path, required=True)
    p.add_argument("--plateau-zone", type=int, default=9)
    p.add_argument(
        "--geoid-correction",
        default="egm96",
        help="PLATEAU orthometric→ellipsoidal correction (egm96 default for Tokyo).",
    )
    p.add_argument("--output-npy", type=Path, required=True)
    p.add_argument("--manifest-json", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    plateau_dir = args.plateau_dir.resolve()
    if not plateau_dir.is_dir():
        raise FileNotFoundError(f"plateau dir not found: {plateau_dir}")

    t0 = time.perf_counter()
    loader = PlateauLoader(zone=int(args.plateau_zone))
    building = loader.load_directory(
        plateau_dir,
        geoid_correction=args.geoid_correction,
    )
    tri = np.asarray(building.triangles, dtype=np.float64)
    if tri.ndim != 3 or tri.shape[1:] != (3, 3):
        raise RuntimeError(f"unexpected triangle shape: {tri.shape}")

    args.output_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_npy, tri)
    elapsed = time.perf_counter() - t0
    print(f"saved {args.output_npy} | triangles={tri.shape[0]} | {elapsed:.1f}s")

    if args.manifest_json is not None:
        manifest = {
            "plateau_dir": str(plateau_dir),
            "plateau_zone": int(args.plateau_zone),
            "geoid_correction": str(args.geoid_correction),
            "output_npy": str(args.output_npy.resolve()),
            "n_triangles": int(tri.shape[0]),
            "elapsed_s": float(elapsed),
        }
        args.manifest_json.parent.mkdir(parents=True, exist_ok=True)
        args.manifest_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"manifest: {args.manifest_json}")


if __name__ == "__main__":
    main()
