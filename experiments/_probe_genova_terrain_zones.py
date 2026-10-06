#!/usr/bin/env python3
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from genova_zones import DEFAULT_CENTRO_BBOX, assign_zones

CSV = Path(r"C:/Users/Andrea/Desktop/Gneova Map/genova_area_points_osm (3).csv")


def main() -> None:
    rows: list[dict] = []
    with CSV.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(
                {
                    "lat_deg": float(r["lat_deg"]),
                    "lon_deg": float(r["lon_deg"]),
                    "mean_n_los": float(r["mean_n_los"]),
                    "mean_n_nlos": float(r["mean_n_nlos"]),
                    "mean_n_visible": float(r["mean_n_visible"]),
                    "mean_n_terrain_blocked": float(r["mean_n_terrain_blocked"]),
                    "mean_n_terrain_blocked_visible": float(r["mean_n_terrain_blocked_visible"]),
                }
            )

    n = len(rows)
    tbv = np.array([r["mean_n_terrain_blocked_visible"] for r in rows], dtype=np.float64)
    tb = np.array([r["mean_n_terrain_blocked"] for r in rows], dtype=np.float64)
    vis = np.array([r["mean_n_visible"] for r in rows], dtype=np.float64)
    los = np.array([r["mean_n_los"] for r in rows], dtype=np.float64)
    nlos = np.array([r["mean_n_nlos"] for r in rows], dtype=np.float64)
    lats = np.array([r["lat_deg"] for r in rows], dtype=np.float64)
    lons = np.array([r["lon_deg"] for r in rows], dtype=np.float64)
    zones = assign_zones(lats, lons, centro_bbox=DEFAULT_CENTRO_BBOX)

    print(f"n={n}")
    print(
        "citywide tbv: "
        f"mean={tbv.mean():.3f} med={np.median(tbv):.3f} "
        f"p90={np.percentile(tbv, 90):.3f} p95={np.percentile(tbv, 95):.3f} "
        f"p99={np.percentile(tbv, 99):.3f} max={tbv.max():.3f}"
    )
    print(f"citywide tb={tb.mean():.3f} visible={vis.mean():.3f}")
    print(f"citywide tbv/visible mean={(tbv / np.maximum(vis, 1e-9)).mean() * 100:.2f}%")
    print(f"citywide pct points tbv>0={(tbv > 0).mean() * 100:.1f}% tbv>1={(tbv > 1).mean() * 100:.1f}%")

    for z in ("centro_storico", "costiera", "other"):
        m = zones == z
        if not np.any(m):
            continue
        print(
            f"{z}: n={int(m.sum())} "
            f"tbv_mean={tbv[m].mean():.3f} med={np.median(tbv[m]):.3f} "
            f"p90={np.percentile(tbv[m], 90):.3f} max={tbv[m].max():.3f} "
            f"pct>0={(tbv[m] > 0).mean() * 100:.1f}% pct>1={(tbv[m] > 1).mean() * 100:.1f}% "
            f"vis={vis[m].mean():.2f} los={los[m].mean():.2f} nlos={nlos[m].mean():.2f}"
        )

    bands = [
        ("lat<44.40", lats < 44.40),
        ("44.40-44.42", (lats >= 44.40) & (lats < 44.42)),
        ("44.42-44.44", (lats >= 44.42) & (lats < 44.44)),
        ("44.44-44.46", (lats >= 44.44) & (lats <= 44.46)),
        ("lat>44.46", lats > 44.46),
    ]
    for name, m in bands:
        if not np.any(m):
            continue
        print(
            f"band {name}: n={int(m.sum())} "
            f"tbv={tbv[m].mean():.3f} p90={np.percentile(tbv[m], 90):.3f} max={tbv[m].max():.3f}"
        )

    lon_bands = [
        ("west lon<8.85", lons < 8.85),
        ("mid 8.85-8.95", (lons >= 8.85) & (lons < 8.95)),
        ("east lon>=8.95", lons >= 8.95),
    ]
    for name, m in lon_bands:
        print(
            f"lon {name}: n={int(m.sum())} "
            f"tbv={tbv[m].mean():.3f} p90={np.percentile(tbv[m], 90):.3f}"
        )

    # Relative to city mean
    city = float(tbv.mean())
    for z in ("centro_storico", "costiera", "other"):
        m = zones == z
        if not np.any(m):
            continue
        ratio = tbv[m].mean() / max(city, 1e-12)
        print(f"vs city mean: {z} = {ratio:.2f}x")

    idx = np.argsort(-tbv)[:20]
    print("TOP20 tbv points:")
    for i in idx:
        r = rows[int(i)]
        print(
            f"  lat={r['lat_deg']:.5f} lon={r['lon_deg']:.5f} "
            f"tbv={r['mean_n_terrain_blocked_visible']:.3f} "
            f"vis={r['mean_n_visible']:.2f} los={r['mean_n_los']:.2f} "
            f"zone={zones[int(i)]}"
        )


if __name__ == "__main__":
    main()
