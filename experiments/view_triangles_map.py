#!/usr/bin/env python3
"""Visualize an ECEF triangles mesh on an interactive Folium map."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import folium
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_ROOT / "python"))

from gnss_gpu.urban_signal_sim import ecef_to_lla  # noqa: E402

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = 2.0 * WGS84_F - WGS84_F * WGS84_F


def ecef_to_lla_deg_batch(xyz: np.ndarray) -> np.ndarray:
    out = np.empty((xyz.shape[0], 3), dtype=np.float64)
    for i in range(xyz.shape[0]):
        lat, lon, alt = ecef_to_lla(float(xyz[i, 0]), float(xyz[i, 1]), float(xyz[i, 2]))
        out[i, 0] = math.degrees(lat)
        out[i, 1] = math.degrees(lon)
        out[i, 2] = alt
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--triangles-npy", type=Path, required=True)
    p.add_argument("--out-html", type=Path, required=True)
    p.add_argument("--stride", type=int, default=40, help="Plot every N-th triangle footprint")
    p.add_argument("--max-triangles", type=int, default=4000, help="Cap rendered triangle footprints")
    p.add_argument("--zoom-start", type=int, default=13)
    p.add_argument(
        "--expected-bbox",
        type=str,
        default="",
        help="Optional south,west,north,east to overlay expected coverage",
    )
    return p.parse_args()


def _alt_color(alt_m: float, vmin: float, vmax: float) -> str:
    if vmax <= vmin:
        t = 0.5
    else:
        t = (alt_m - vmin) / (vmax - vmin)
        t = max(0.0, min(1.0, t))
    # low=blue, mid=yellow, high=red
    if t < 0.5:
        r = int(255 * (t * 2.0))
        g = int(255 * (t * 2.0))
        b = 255
    else:
        u = (t - 0.5) * 2.0
        r = 255
        g = int(255 * (1.0 - u))
        b = int(255 * (1.0 - u))
    return f"#{r:02x}{g:02x}{b:02x}"


def main() -> None:
    args = _parse_args()
    tri = np.asarray(np.load(args.triangles_npy), dtype=np.float64)
    if tri.ndim != 3 or tri.shape[1:] != (3, 3):
        raise ValueError(f"Expected shape [N,3,3], got {tri.shape}")

    corners = tri.reshape(-1, 3)
    lla = ecef_to_lla_deg_batch(corners)
    lats = lla[:, 0]
    lons = lla[:, 1]
    alts = lla[:, 2]

    south, west, north, east = float(lats.min()), float(lons.min()), float(lats.max()), float(lons.max())
    center_lat = (south + north) * 0.5
    center_lon = (west + east) * 0.5

    stride = max(1, int(args.stride))
    idx = np.arange(0, tri.shape[0], stride, dtype=np.int64)
    if int(args.max_triangles) > 0 and idx.size > int(args.max_triangles):
        idx = idx[: int(args.max_triangles)]

    tri_sel = tri[idx]
    lla_sel = ecef_to_lla_deg_batch(tri_sel.reshape(-1, 3)).reshape(-1, 3, 3)
    alt_sel = lla_sel[:, :, 2].mean(axis=1)
    vmin = float(np.percentile(alt_sel, 5))
    vmax = float(np.percentile(alt_sel, 95))
    if vmax <= vmin:
        vmax = vmin + 1.0

    m = folium.Map(location=[center_lat, center_lon], zoom_start=int(args.zoom_start), tiles="OpenStreetMap")
    folium.TileLayer("CartoDB positron", name="CartoDB").add_to(m)

    folium.Rectangle(
        bounds=[[south, west], [north, east]],
        color="#111111",
        weight=2,
        fill=False,
        popup="Mesh bbox",
    ).add_to(m)

    if args.expected_bbox.strip():
        parts = [float(x.strip()) for x in args.expected_bbox.split(",")]
        if len(parts) == 4:
            es, ew, en, ee = parts
            folium.Rectangle(
                bounds=[[es, ew], [en, ee]],
                color="#0066cc",
                weight=2,
                dash_array="6",
                fill=False,
                popup="Expected bbox",
            ).add_to(m)

    for ti in range(tri_sel.shape[0]):
        poly = [(float(lla_sel[ti, j, 0]), float(lla_sel[ti, j, 1])) for j in range(3)]
        folium.Polygon(
            locations=poly,
            color=_alt_color(float(alt_sel[ti]), vmin, vmax),
            weight=0,
            fill=True,
            fill_opacity=0.55,
            popup=f"mean_alt_m={alt_sel[ti]:.2f}",
        ).add_to(m)

    stats = {
        "n_triangles": int(tri.shape[0]),
        "bbox": {"south": south, "west": west, "north": north, "east": east},
        "alt_m": {
            "min": float(alts.min()),
            "p05": float(np.percentile(alts, 5)),
            "median": float(np.median(alts)),
            "p95": float(np.percentile(alts, 95)),
            "max": float(alts.max()),
        },
        "rendered_footprints": int(tri_sel.shape[0]),
        "stride": stride,
    }
    note = (
        f"<b>Triangles mesh preview</b><br>"
        f"triangles: {stats['n_triangles']:,}<br>"
        f"rendered: {stats['rendered_footprints']:,} (stride={stride})<br>"
        f"bbox lat [{south:.5f}, {north:.5f}]<br>"
        f"bbox lon [{west:.5f}, {east:.5f}]<br>"
        f"alt min/med/max: {stats['alt_m']['min']:.1f} / "
        f"{stats['alt_m']['median']:.1f} / {stats['alt_m']['max']:.1f} m<br>"
        f"<small>color = mean triangle altitude (m ellipsoid)</small>"
    )
    folium.Marker(
        [center_lat, center_lon],
        popup=folium.Popup(note, max_width=320),
        icon=folium.Icon(color="green", icon="info-sign"),
    ).add_to(m)
    folium.LayerControl().add_to(m)

    args.out_html.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(args.out_html))

    meta_path = args.out_html.with_suffix(".json")
    meta_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"[view_triangles_map] html={args.out_html}")
    print(f"[view_triangles_map] stats={meta_path}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
