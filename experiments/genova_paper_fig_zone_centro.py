#!/usr/bin/env python3
"""Square historic-centre LOS map with bottom colorbar.

Example::

    python experiments/genova_paper_fig_zone_centro.py \\
      --csv \"C:/Users/Andrea/Desktop/Gneova Map/genova_area_points_osm (3).csv\" \\
      --out experiments/results/genova_paper_fig_zone_centro.png
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_EXPERIMENTS = Path(__file__).resolve().parent
if str(_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from genova_paper_figures import (
    _add_osm_basemap,
    _lonlat_to_webmerc,
    _square_extent_from_lonlat,
    _style_map_axes,
)
from genova_zones import (
    DEFAULT_CENTRO_BBOX,
    DEFAULT_COASTAL_BUFFER_M,
    DEFAULT_MAP_CLIP_NORTH,
    classify_zones,
    filter_map_rows,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--csv",
        type=Path,
        default=Path(r"C:/Users/Andrea/Desktop/Gneova Map/genova_area_points_osm (3).csv"),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("experiments/results/genova_paper_fig_zone_centro.png"),
    )
    p.add_argument("--coastal-buffer-m", type=float, default=DEFAULT_COASTAL_BUFFER_M)
    p.add_argument("--map-clip-north", type=float, default=DEFAULT_MAP_CLIP_NORTH)
    p.add_argument("--point-size", type=float, default=2.2)
    p.add_argument("--zoom", type=int, default=16)
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def _load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, newline="", encoding="utf-8") as f:
        for rr in csv.DictReader(f):
            try:
                rows.append(
                    {
                        "lat_deg": float(rr["lat_deg"]),
                        "lon_deg": float(rr["lon_deg"]),
                        "mean_n_los": float(rr["mean_n_los"]),
                    }
                )
            except (KeyError, ValueError):
                continue
    if not rows:
        raise ValueError(f"no valid rows in {path}")
    return rows


def main() -> None:
    args = _parse_args()
    north_clip = float(args.map_clip_north) if float(args.map_clip_north) < 90.0 else None
    rows = filter_map_rows(_load_rows(args.csv.resolve()), north_max=north_clip)

    lats = np.array([r["lat_deg"] for r in rows], dtype=np.float64)
    lons = np.array([r["lon_deg"] for r in rows], dtype=np.float64)
    zones, _ = classify_zones(
        lats, lons,
        centro_bbox=DEFAULT_CENTRO_BBOX,
        coastal_buffer_m=float(args.coastal_buffer_m),
    )
    subset = [r for r, z in zip(rows, zones) if z == "centro_storico"]
    if not subset:
        raise RuntimeError("no historic-centre points")

    lons = np.array([r["lon_deg"] for r in subset], dtype=np.float64)
    lats = np.array([r["lat_deg"] for r in subset], dtype=np.float64)
    vals = np.array([r["mean_n_los"] for r in subset], dtype=np.float64)
    vmin = float(np.percentile(vals, 5))
    vmax = float(np.percentile(vals, 95))
    cmap = plt.get_cmap("RdYlGn")

    extent = _square_extent_from_lonlat(lons, lats, pad_frac=0.14)
    xmin, xmax, ymin, ymax = extent
    geo_w = max(xmax - xmin, 1.0)
    geo_h = max(ymax - ymin, 1.0)
    data_aspect = geo_h / geo_w

    fig_w = 4.5
    margin_x = 0.28
    margin_top = 0.06
    cbar_h = 0.22
    gap = 0.06
    margin_bottom = 0.10
    map_h = fig_w * data_aspect
    fig_h = map_h + margin_top + cbar_h + gap + margin_bottom

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=int(args.dpi))
    map_bottom = (margin_bottom + cbar_h + gap) / fig_h
    map_height = map_h / fig_h
    ax = fig.add_axes(
        [margin_x / fig_w, map_bottom, 1.0 - 2.0 * margin_x / fig_w, map_height]
    )
    cbar_ax = fig.add_axes([0.18, margin_bottom / fig_h, 0.64, cbar_h / fig_h])

    xs, ys = _lonlat_to_webmerc(lons, lats)
    print(f"[centro] basemap n={len(subset)} zoom={int(args.zoom)} ...", flush=True)
    _add_osm_basemap(ax, extent, zoom=int(args.zoom))
    ax.set_box_aspect(1)
    sc = ax.scatter(
        xs, ys, c=vals, s=float(args.point_size), cmap=cmap, vmin=vmin, vmax=vmax,
        linewidths=0, alpha=0.9, zorder=3, rasterized=True,
    )
    _style_map_axes(ax)

    cb = fig.colorbar(sc, cax=cbar_ax, orientation="horizontal")
    cb.set_label("LOS satellites", fontsize=8, labelpad=1)
    cb.ax.tick_params(labelsize=7, pad=1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=int(args.dpi), bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    print(f"wrote {args.out.resolve()}  n={len(subset):,}  LOS p5–p95: {vmin:.1f}–{vmax:.1f}")


if __name__ == "__main__":
    main()
