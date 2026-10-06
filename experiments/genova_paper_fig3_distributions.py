#!/usr/bin/env python3
"""Figure 3: box plot of LOS satellites per road point (historic centre vs coastal zone).

Uses ``mean_n_los`` from the area map CSV: satellites with unobstructed line-of-sight
after building ray-tracing (subset of ``mean_n_visible``, which also includes NLOS).

Example::

    python experiments/genova_paper_fig3_distributions.py \\
      --csv \"C:/Users/Andrea/Desktop/Gneova Map/genova_area_points_osm (3).csv\" \\
      --out experiments/results/genova_paper_fig3_distributions.png
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

from genova_zones import (
    DEFAULT_CENTRO_BBOX,
    DEFAULT_COASTAL_BUFFER_M,
    DEFAULT_MAP_CLIP_NORTH,
    classify_zones,
    filter_map_rows,
)

ZONE_COLORS = {
    "costiera": "#2471a3",
    "centro_storico": "#c0392b",
}


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
        default=Path("experiments/results/genova_paper_fig3_distributions.png"),
    )
    p.add_argument("--coastal-buffer-m", type=float, default=DEFAULT_COASTAL_BUFFER_M)
    p.add_argument("--map-clip-north", type=float, default=DEFAULT_MAP_CLIP_NORTH)
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

    centro = np.array(
        [r["mean_n_los"] for r, z in zip(rows, zones) if z == "centro_storico"],
        dtype=np.float64,
    )
    cost = np.array(
        [r["mean_n_los"] for r, z in zip(rows, zones) if z == "costiera"],
        dtype=np.float64,
    )
    if len(centro) == 0 or len(cost) == 0:
        raise RuntimeError("empty zone after classification")

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    datasets = [centro, cost]
    labels = ["Historic centre", "Coastal zone"]
    colors = [ZONE_COLORS["centro_storico"], ZONE_COLORS["costiera"]]

    bp = ax.boxplot(
        datasets,
        tick_labels=labels,
        patch_artist=True,
        widths=0.5,
        showfliers=True,
        flierprops={"marker": ".", "markersize": 3, "alpha": 0.35},
        medianprops={"color": "#222222", "linewidth": 1.2},
    )
    for box, color in zip(bp["boxes"], colors):
        box.set_facecolor(color)
        box.set_alpha(0.65)

    ymax = float(max(centro.max(), cost.max()))
    ymin = float(min(centro.min(), cost.min()))
    ypad = max(0.8, 0.08 * (ymax - ymin))
    for i, data in enumerate(datasets, start=1):
        mu = float(np.mean(data))
        sd = float(np.std(data, ddof=1))
        ax.text(
            i, ymax + 0.12 * ypad,
            f"n={len(data):,}\n{mu:.2f} ± {sd:.2f}",
            ha="center", va="bottom", fontsize=8.5,
        )

    ax.set_ylabel("LOS satellites")
    ax.set_ylim(max(0.0, ymin - ypad), ymax + 2.2 * ypad)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
    ax.tick_params(labelsize=9)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=int(args.dpi), bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {args.out.resolve()}")
    for name, data in zip(labels, datasets):
        print(f"  {name}: n={len(data):,}  mean={np.mean(data):.3f}  std={np.std(data, ddof=1):.3f}")


if __name__ == "__main__":
    main()
