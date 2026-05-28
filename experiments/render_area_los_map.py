#!/usr/bin/env python3
"""Render LOS/NLOS CSV results to interactive HTML map."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import branca.colormap as bcm
import folium
import numpy as np


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, required=True, help="Input CSV from build_area_los_nlos_map.py")
    p.add_argument("--out-html", type=Path, required=True, help="Output HTML map path")
    p.add_argument(
        "--metric",
        type=str,
        default="mean_n_los",
        choices=[
            "mean_n_los",
            "mean_n_nlos",
            "mean_n_visible",
            "mean_n_terrain_blocked",
            "mean_n_terrain_blocked_visible",
        ],
        help="Metric used for marker color",
    )
    p.add_argument("--max-points", type=int, default=0, help="Optional cap of rendered points (0 = all)")
    p.add_argument("--stride", type=int, default=1, help="Render every N-th row to reduce HTML size")
    p.add_argument("--zoom-start", type=int, default=14)
    return p.parse_args()


def _read_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for rr in r:
            try:
                rows.append(
                    {
                        "lat_deg": float(rr["lat_deg"]),
                        "lon_deg": float(rr["lon_deg"]),
                        "way_id": int(rr.get("way_id", -1)),
                        "highway": str(rr.get("highway", "")),
                        "name": str(rr.get("name", "")),
                        "mean_n_los": float(rr["mean_n_los"]),
                        "mean_n_nlos": float(rr["mean_n_nlos"]),
                        "mean_n_visible": float(rr["mean_n_visible"]),
                        "mean_n_terrain_blocked": float(rr["mean_n_terrain_blocked"]),
                        "mean_n_terrain_blocked_visible": float(rr.get("mean_n_terrain_blocked_visible", 0.0)),
                    }
                )
            except Exception:
                continue
    return rows


def _build_html_map(rows: list[dict], out_html: Path, *, metric_key: str, zoom_start: int) -> None:
    if not rows:
        raise ValueError("No rows available for HTML map.")
    lat0 = float(np.mean([r["lat_deg"] for r in rows]))
    lon0 = float(np.mean([r["lon_deg"] for r in rows]))
    m = folium.Map(location=[lat0, lon0], zoom_start=int(zoom_start), tiles="OpenStreetMap")
    vals = np.asarray([float(r[metric_key]) for r in rows], dtype=np.float64)
    vmin = float(np.nanpercentile(vals, 5))
    vmax = float(np.nanpercentile(vals, 95))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals) + 1.0)
    cmap = bcm.LinearColormap(["#d73027", "#fee08b", "#1a9850"], vmin=vmin, vmax=vmax)
    cmap.caption = metric_key
    cmap.add_to(m)
    for r in rows:
        metric = float(r[metric_key])
        popup = (
            f"LOS mean: {r['mean_n_los']:.2f}<br>"
            f"NLOS mean: {r['mean_n_nlos']:.2f}<br>"
            f"Visible mean: {r['mean_n_visible']:.2f}<br>"
            f"Terrain blocked mean: {r['mean_n_terrain_blocked']:.2f}<br>"
            f"Terrain blocked visible mean: {r.get('mean_n_terrain_blocked_visible', 0.0):.2f}<br>"
            f"Highway: {r.get('highway','')}"
        )
        folium.CircleMarker(
            location=[float(r["lat_deg"]), float(r["lon_deg"])],
            radius=3,
            weight=0,
            fill=True,
            fill_color=cmap(metric),
            fill_opacity=0.8,
            popup=popup,
        ).add_to(m)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_html))


def main() -> None:
    args = _parse_args()
    rows = _read_rows(args.csv)
    if not rows:
        raise RuntimeError(f"No valid rows read from CSV: {args.csv}")

    stride = max(1, int(args.stride))
    rows_view = rows[::stride]
    if int(args.max_points) > 0:
        rows_view = rows_view[: int(args.max_points)]

    print(f"[render] input_rows={len(rows)} render_rows={len(rows_view)} metric={args.metric}", flush=True)
    _build_html_map(rows_view, args.out_html, metric_key=str(args.metric), zoom_start=int(args.zoom_start))
    print(f"[render] done: html={args.out_html}", flush=True)


if __name__ == "__main__":
    main()
