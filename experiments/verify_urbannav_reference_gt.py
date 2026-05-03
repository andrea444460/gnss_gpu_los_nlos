#!/usr/bin/env python3
"""Sanity-check UrbanNav-style ``reference.csv`` ground truth.

UrbanNav Tokyo GT is Applanix POS LV620 output as ECEF (m); altitude derived via
``ecef_to_lla`` is **ellipsoidal** (WGS84-style geodetic h). Maps that drape on
terrain / orthometric elevation often show the trajectory visually \"floating\"
if you extrude markers using ellipsoid height — this script confirms XYZ ↔ LLA
consistency and expected Tokyo bbox.

Example::

    python experiments/verify_urbannav_reference_gt.py \\
      experiments/data/urbannav/Odaiba/reference.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gnss_gpu.io.urbannav import UrbanNavLoader  # noqa: E402
from gnss_gpu.urban_signal_sim import ecef_to_lla  # noqa: E402

# Rough bbox checks (Tokyo runs — shrink margin if you want stricter pass)
_BBOX = {
    "odaiba": {"lat": (35.60, 35.68), "lon": (139.72, 139.82)},
    "shinjuku": {"lat": (35.66, 35.74), "lon": (139.66, 139.76)},
    "tokyo": {"lat": (35.55, 35.78), "lon": (139.65, 139.85)},
}


def _normalize_header(row: dict) -> dict:
    return {(k or "").strip(): v for k, v in row.items()}


def _read_ellipsoid_height_col(path: Path) -> np.ndarray | None:
    with open(path, newline="", encoding="utf-8") as fh:
        r = csv.DictReader(fh, skipinitialspace=True)
        rows = [_normalize_header(x) for x in r]
    if not rows:
        return None
    for key in ("Ellipsoid Height (m)", "Ellipsoid Height"):
        if key in rows[0]:
            return np.array([float(rows[i][key]) for i in range(len(rows))])
    return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("reference_csv", type=Path, help="UrbanNav reference.csv path")
    p.add_argument(
        "--bbox",
        choices=sorted(_BBOX.keys()),
        default="tokyo",
        help="Latitude/longitude acceptance box",
    )
    args = p.parse_args()
    path = args.reference_csv.resolve()
    if not path.is_file():
        print(f"ERROR: file not found: {path}")
        return 1

    loader = UrbanNavLoader(path.parent)
    times, ecef = loader.load_ground_truth(filepath=path)
    n = ecef.shape[0]
    if n == 0:
        print("ERROR: no rows loaded")
        return 1

    # Distance from Earth center ~6378 km at surface
    r = np.linalg.norm(ecef, axis=1)
    r_med = float(np.median(r))

    lats = np.empty(n)
    lons = np.empty(n)
    alts = np.empty(n)
    for i in range(n):
        lat, lon, alt = ecef_to_lla(float(ecef[i, 0]), float(ecef[i, 1]), float(ecef[i, 2]))
        lats[i] = math.degrees(lat)
        lons[i] = math.degrees(lon)
        alts[i] = alt

    col_h = _read_ellipsoid_height_col(path)

    bb = _BBOX[args.bbox]
    inside = (
        np.all(lats >= bb["lat"][0])
        & np.all(lats <= bb["lat"][1])
        & np.all(lons >= bb["lon"][0])
        & np.all(lons <= bb["lon"][1])
    )

    print(f"File: {path}")
    print(f"Rows: {n}  GPS time span: {float(times[-1] - times[0]):.1f} s")
    print(f"|r| from Earth center: median {r_med:.3f} m (expect ~6.37e6 for surface)")
    print(
        f"Geodetic from ECEF (ellipsoid h): lat [{lats.min():.6f}, {lats.max():.6f}], "
        f"lon [{lons.min():.6f}, {lons.max():.6f}], "
        f"h [{alts.min():.2f}, {alts.max():.2f}] m"
    )
    print(f"Bbox '{args.bbox}': {'PASS' if inside else 'FAIL — coordinates outside expected area'}")
    if col_h is not None:
        if col_h.shape[0] != n:
            print("WARNING: Ellipsoid Height column length mismatch vs ECEF rows")
        else:
            dh = np.abs(col_h - alts)
            print(
                f"CSV 'Ellipsoid Height (m)' vs alt from ECEF: max abs diff = {float(np.max(dh)):.6f} m "
                f"(expect ~0 if consistent)"
            )
            if float(np.max(dh)) > 0.05:
                print(
                    "  WARNING: large mismatch — check column definitions or mixed CRS in CSV."
                )

    print(
        "\nNote for maps (Flutter / Mapbox / Google): "
        "GT height is ellipsoidal. Basemap terrain is usually orthometric + DEM smoothing; "
        "drawing polyline with LatLng only follows the 2D road plane; "
        "using raw altitude with extrusion without geoid correction often looks tens of meters 'floating' in Japan."
    )
    return 0 if inside else 2


if __name__ == "__main__":
    raise SystemExit(main())
