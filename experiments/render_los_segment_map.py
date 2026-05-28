#!/usr/bin/env python3
"""Render LOS/NLOS CSV as colored line segments between nearby points."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import branca.colormap as bcm
import folium
import numpy as np


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp * 0.5) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl * 0.5) ** 2
    return 2.0 * r * math.asin(min(1.0, math.sqrt(a)))


def _load_points(csv_path: Path, metric: str) -> list[dict]:
    out: list[dict] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                lat = float(row["lat_deg"])
                lon = float(row["lon_deg"])
                val = float(row["mean_n_los"] if metric == "los" else row["mean_n_nlos"])
                out.append(
                    {
                        "lat": lat,
                        "lon": lon,
                        "val": val,
                        "way_id": int(row.get("way_id", -1)),
                        "highway": row.get("highway", ""),
                        "name": row.get("name", ""),
                    }
                )
            except Exception:
                continue
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Render colored segments between nearby contiguous points")
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--out-html", type=Path, required=True)
    p.add_argument("--metric", choices=("los", "nlos"), default="los")
    p.add_argument("--max-link-m", type=float, default=60.0, help="Max distance to connect contiguous points")
    p.add_argument("--line-weight", type=int, default=4)
    p.add_argument("--show-points", action="store_true")
    args = p.parse_args()

    pts = _load_points(args.csv, args.metric)
    if not pts:
        raise RuntimeError("No valid points loaded from CSV.")

    lat0 = float(np.mean([x["lat"] for x in pts]))
    lon0 = float(np.mean([x["lon"] for x in pts]))
    m = folium.Map(location=[lat0, lon0], zoom_start=14, tiles="CartoDB positron")

    vals = np.asarray([x["val"] for x in pts], dtype=np.float64)
    vmin = float(np.nanpercentile(vals, 5))
    vmax = float(np.nanpercentile(vals, 95))
    if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or vmax <= vmin:
        vmin = float(np.nanmin(vals))
        vmax = float(np.nanmax(vals) + 1.0)

    cmap = bcm.LinearColormap(["#d73027", "#fee08b", "#1a9850"], vmin=vmin, vmax=vmax)
    cmap.caption = f"mean_n_{args.metric}"
    cmap.add_to(m)

    # Group by road id for contiguity.
    by_way: dict[int, list[dict]] = {}
    for pt in pts:
        by_way.setdefault(pt["way_id"], []).append(pt)

    seg_count = 0
    for way_id, arr in by_way.items():
        if len(arr) < 2:
            continue
        # Sort by latitude+longitude as a robust fallback when row order is unknown.
        arr = sorted(arr, key=lambda x: (x["lat"], x["lon"]))
        for i in range(len(arr) - 1):
            a = arr[i]
            b = arr[i + 1]
            d = _haversine_m(a["lat"], a["lon"], b["lat"], b["lon"])
            if d > float(args.max_link_m):
                continue
            v_mid = 0.5 * (a["val"] + b["val"])
            popup = (
                f"way_id={way_id}<br>"
                f"{args.metric.upper()} midpoint={v_mid:.2f}<br>"
                f"d={d:.1f} m<br>"
                f"highway={a.get('highway','') or b.get('highway','')}"
            )
            folium.PolyLine(
                locations=[[a["lat"], a["lon"]], [b["lat"], b["lon"]]],
                color=cmap(v_mid),
                weight=int(args.line_weight),
                opacity=0.9,
                popup=popup,
            ).add_to(m)
            seg_count += 1

    if args.show_points:
        for pt in pts:
            folium.CircleMarker(
                location=[pt["lat"], pt["lon"]],
                radius=1,
                weight=0,
                fill=True,
                fill_opacity=0.6,
                fill_color="#111111",
            ).add_to(m)

    args.out_html.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(args.out_html))
    print(f"points={len(pts)} segments={seg_count} saved={args.out_html}")


if __name__ == "__main__":
    main()
