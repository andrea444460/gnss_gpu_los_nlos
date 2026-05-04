#!/usr/bin/env python3
"""Sanity-check PLATEAU CityGML vertical geometry (GPU / BVH uses this mesh).

Reads ``*.gml`` under ``--plateau-dir`` and, per building:

  * **z_span_raw**: range of the third ``posList`` coordinate (metres above the
    ellipsoid for Japanese plane-rect PLATEAU data — see :mod:`gnss_gpu.io.plateau`).
  * **h_span_ecef**: same vertices converted with :class:`PlateauLoader` to ECEF,
    then ellipsoidal height via ``ecef_to_lla`` — should track ``z_span_raw`` if
    axes/units are consistent.
  * **measuredHeight**: optional ``bldg:measuredHeight`` from CityGML — compare to
    geometry when present.

This does **not** validate against Cesium Ion OSM (different source). Low spans on
LOD2 models can reflect roof detail vs block height; LOD1 boxes should match
``measuredHeight`` closely when the attribute exists.

Examples::

  python experiments/verify_plateau_building_heights.py \\
      --plateau-dir experiments/data/plateau_odaiba --plateau-zone 9 --max-files 30

  python experiments/verify_plateau_building_heights.py \\
      --plateau-dir data --glob \"sample_plateau.gml\" --plateau-zone 9
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import sys
from typing import Optional

import numpy as np

# Repo root on path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "python"))

from gnss_gpu.io.citygml import parse_citygml  # noqa: E402
from gnss_gpu.io.plateau import PlateauLoader  # noqa: E402
from gnss_gpu.urban_signal_sim import ecef_to_lla  # noqa: E402


def _find_measured_height_m(bldg_elem) -> Optional[float]:
    """First bldg:measuredHeight text in metres (any namespace suffix)."""
    for el in bldg_elem.iter():
        if el.tag.endswith("}measuredHeight") and el.text:
            try:
                return float(el.text.strip().split()[0])
            except ValueError:
                return None
    return None


def _measured_heights_in_order(filepath: str) -> list[Optional[float]]:
    import xml.etree.ElementTree as ET

    tree = ET.parse(filepath)
    root = tree.getroot()
    out: list[Optional[float]] = []
    for bldg_elem in root.iter():
        if bldg_elem.tag.endswith("}Building"):
            out.append(_find_measured_height_m(bldg_elem))
    return out


def _collect_vertices(building) -> np.ndarray:
    """All polygon vertices stacked (N, 3) raw CityGML coords."""
    parts = []
    for poly in building.polygons:
        if poly.size:
            parts.append(np.asarray(poly, dtype=np.float64))
    if not parts:
        return np.empty((0, 3), dtype=np.float64)
    return np.vstack(parts)


def _ecef_heights(loader: PlateauLoader, xyz: np.ndarray) -> np.ndarray:
    """Ellipsoidal heights [m] after PlateauLoader plane/geodetic conversion."""
    ecef = loader._polygon_to_ecef(xyz)
    h = np.empty(ecef.shape[0], dtype=np.float64)
    for i in range(ecef.shape[0]):
        _lat, _lon, hi = ecef_to_lla(float(ecef[i, 0]), float(ecef[i, 1]), float(ecef[i, 2]))
        h[i] = hi
    return h


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Check PLATEAU building height consistency")
    p.add_argument("--plateau-dir", type=str, required=True, help="Directory with CityGML *.gml")
    p.add_argument("--plateau-zone", type=int, default=9, help="Japanese plane rect zone (default 9)")
    p.add_argument("--glob", type=str, default="*.gml", dest="glob_pat", help="GML glob under plateau-dir")
    p.add_argument("--max-files", type=int, default=0, help="Max GML files (0 = all)")
    p.add_argument(
        "--max-buildings-per-file",
        type=int,
        default=0,
        help="Stop after N buildings per file (0 = all)",
    )
    p.add_argument("--csv", type=str, default="", help="Optional CSV path for per-building rows")
    args = p.parse_args(argv)

    plateau_dir = os.path.abspath(args.plateau_dir)
    pattern = os.path.join(plateau_dir, "**", args.glob_pat)
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        pattern2 = os.path.join(plateau_dir, args.glob_pat)
        files = sorted(glob.glob(pattern2))
    if not files:
        print(f"No GML files under {plateau_dir!r} (glob {args.glob_pat!r})", file=sys.stderr)
        return 1
    if args.max_files > 0:
        files = files[: args.max_files]

    loader = PlateauLoader(zone=int(args.plateau_zone))

    rows = []
    lod_counts: dict[int, int] = {}
    z_spans: list[float] = []
    mh_geoms: list[tuple[float, float]] = []  # (measured, z_span)
    z_vs_h_diff: list[float] = []

    for fp in files:
        try:
            buildings = parse_citygml(fp)
        except Exception as e:
            print(f"SKIP parse {fp}: {e}", file=sys.stderr)
            continue
        mh_order = _measured_heights_in_order(fp)
        if len(mh_order) != len(buildings):
            mh_order = [None] * len(buildings)

        for bi, b in enumerate(buildings):
            if args.max_buildings_per_file > 0 and bi >= args.max_buildings_per_file:
                break
            verts = _collect_vertices(b)
            if verts.shape[0] < 3:
                continue
            lod_counts[b.lod] = lod_counts.get(b.lod, 0) + 1

            z_col = verts[:, 2]
            z_span = float(np.ptp(z_col))
            z_spans.append(z_span)

            h_ecef = _ecef_heights(loader, verts)
            h_span = float(np.ptp(h_ecef))
            z_vs_h_diff.append(abs(z_span - h_span))

            mh = mh_order[bi] if bi < len(mh_order) else None
            if mh is not None and math.isfinite(mh) and mh > 0:
                mh_geoms.append((mh, z_span))

            rows.append(
                {
                    "file": os.path.basename(fp),
                    "id": b.id or "",
                    "lod": b.lod,
                    "z_span_m": z_span,
                    "h_span_m": h_span,
                    "abs_z_minus_h_m": abs(z_span - h_span),
                    "measured_height_m": mh if mh is not None else float("nan"),
                }
            )

    if not rows:
        print("No buildings with geometry found.", file=sys.stderr)
        return 1

    zs = np.array(z_spans, dtype=np.float64)
    zh = np.array(z_vs_h_diff, dtype=np.float64)

    print(f"Files scanned: {len(files)}  Buildings: {len(rows)}")
    print(f"LOD counts: {dict(sorted(lod_counts.items()))}")
    print()
    print("Vertical extent from raw Z (3rd coordinate), metres - distribution:")
    for label, q in [
        ("min", 0),
        ("p25", 25),
        ("p50", 50),
        ("p75", 75),
        ("p90", 90),
        ("p95", 95),
        ("max", 100),
    ]:
        print(f"  {label:>4}: {np.percentile(zs, q):.2f} m")
    print()
    print("|z_span_raw - h_span_ecef| - should be ~0 if projection/Z axis OK:")
    print(f"  median: {np.median(zh):.4f} m   p95: {np.percentile(zh, 95):.4f} m   max: {np.max(zh):.4f} m")

    if mh_geoms:
        arr = np.array(mh_geoms, dtype=np.float64)
        mh_m = arr[:, 0]
        g_m = arr[:, 1]
        rel = np.abs(mh_m - g_m) / np.maximum(mh_m, 1.0)
        print()
        print(f"buildings with measuredHeight: {len(mh_geoms)}")
        print("  |measuredHeight - z_span| / measuredHeight : median {:.1f}%  p95 {:.1f}%".format(
            float(np.median(rel)) * 100.0,
            float(np.percentile(rel, 95)) * 100.0,
        ))
        worst = np.argsort(-np.abs(mh_m - g_m))[:5]
        print("  largest |meas - geom| (m):")
        for i in worst:
            print(f"    meas={mh_m[i]:.2f}  geom_z_span={g_m[i]:.2f}  diff={mh_m[i]-g_m[i]:+.2f}")

    csv_path = args.csv.strip()
    if csv_path:
        import csv

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {csv_path}")

    print()
    print(
        "Interpretation: GPU LOS uses this PLATEAU mesh. "
        "Ion/OSM heights are unrelated. Small LOD2 mismatches vs measuredHeight are normal."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
