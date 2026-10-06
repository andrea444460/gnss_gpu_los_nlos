#!/usr/bin/env python3
"""Compare GNSS visibility statistics between Genoa historic center and coastal zone.

Reads a ``build_area_los_nlos_map`` CSV and splits road points into:
  - **centro_storico**: compact bbox around Porto Antico / caruggi
  - **costiera**: road points up to ``--coastal-buffer-m`` north of the southernmost
    road envelope (waterfront proxy), excluding the historic center

Outputs JSON summary, text report, and comparison bar chart.

Example::

    python experiments/genova_zone_statistics.py \\
      --csv \"C:/Users/Andrea/Desktop/Gneova Map/genova_area_points_osm (3).csv\" \\
      --out-prefix experiments/results/genova_centro_vs_costiera
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
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
    classify_zones,
)

METRIC_KEYS = (
    "mean_n_los",
    "mean_n_nlos",
    "mean_n_visible",
    "mean_n_terrain_blocked",
    "mean_n_terrain_blocked_visible",
)


@dataclass(frozen=True)
class ZoneStats:
    name: str
    n_points: int
    metrics: dict[str, dict[str, float]]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--out-prefix", type=Path, required=True)
    p.add_argument("--coastal-buffer-m", type=float, default=DEFAULT_COASTAL_BUFFER_M)
    p.add_argument("--centro-south", type=float, default=DEFAULT_CENTRO_BBOX["south"])
    p.add_argument("--centro-west", type=float, default=DEFAULT_CENTRO_BBOX["west"])
    p.add_argument("--centro-north", type=float, default=DEFAULT_CENTRO_BBOX["north"])
    p.add_argument("--centro-east", type=float, default=DEFAULT_CENTRO_BBOX["east"])
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
                        "highway": str(rr.get("highway", "")),
                        "name": str(rr.get("name", "")),
                        **{k: float(rr[k]) for k in METRIC_KEYS if k in rr},
                    }
                )
            except (KeyError, ValueError):
                continue
    if not rows:
        raise ValueError(f"no valid rows in {path}")
    return rows


def _summarize(name: str, rows: list[dict]) -> ZoneStats:
    metrics: dict[str, dict[str, float]] = {}
    for key in METRIC_KEYS:
        if key not in rows[0]:
            continue
        arr = np.array([float(r[key]) for r in rows], dtype=np.float64)
        metrics[key] = {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr)),
            "p25": float(np.percentile(arr, 25)),
            "p75": float(np.percentile(arr, 75)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }
    if "mean_n_nlos" in metrics and "mean_n_visible" in metrics:
        frac = np.array(
            [
                float(r["mean_n_nlos"]) / float(r["mean_n_visible"])
                if float(r["mean_n_visible"]) > 0
                else 0.0
                for r in rows
            ],
            dtype=np.float64,
        )
        metrics["nlos_fraction"] = {
            "mean": float(np.mean(frac)),
            "median": float(np.median(frac)),
            "std": float(np.std(frac)),
            "p25": float(np.percentile(frac, 25)),
            "p75": float(np.percentile(frac, 75)),
            "min": float(np.min(frac)),
            "max": float(np.max(frac)),
        }
    return ZoneStats(name=name, n_points=len(rows), metrics=metrics)


def _plot_comparison(centro: ZoneStats, costiera: ZoneStats, out_png: Path) -> None:
    keys = ["mean_n_los", "mean_n_nlos", "mean_n_visible", "nlos_fraction"]
    labels = ["LOS", "NLOS", "Visible", "NLOS / visible"]
    x = np.arange(len(keys))
    w = 0.35
    centro_vals = [centro.metrics.get(k, {}).get("mean", float("nan")) for k in keys]
    cost_vals = [costiera.metrics.get(k, {}).get("mean", float("nan")) for k in keys]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w / 2, centro_vals, w, label=f"Centro storico (n={centro.n_points:,})", color="#c0392b")
    ax.bar(x + w / 2, cost_vals, w, label=f"Costiera (n={costiera.n_points:,})", color="#2980b9")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean count / fraction")
    ax.set_title("Genoa — historic centre vs coastal zone (road-point means)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _write_report(
    path: Path,
    centro: ZoneStats,
    costiera: ZoneStats,
    *,
    csv_path: Path,
    coastal_buffer_m: float,
    centro_bbox: dict[str, float],
) -> None:
    lines = [
        "Genoa — historic centre vs coastal zone statistics",
        f"CSV: {csv_path}",
        f"Centro storico bbox: S={centro_bbox['south']:.4f} W={centro_bbox['west']:.4f} "
        f"N={centro_bbox['north']:.4f} E={centro_bbox['east']:.4f}",
        f"Costiera: entro {coastal_buffer_m:.0f} m a nord dell'inviluppo strade più meridionali (escluso centro)",
        "",
    ]
    for zs in (centro, costiera):
        lines.append(f"## {zs.name} — {zs.n_points:,} punti strada")
        for key, st in zs.metrics.items():
            lines.append(
                f"  {key}: mean={st['mean']:.2f} med={st['median']:.2f} "
                f"p25={st['p25']:.2f} p75={st['p75']:.2f}"
            )
        lines.append("")
    c = centro.metrics
    k = costiera.metrics
    if "mean_n_los" in c and "mean_n_los" in k:
        lines.extend(
            [
                "## Confronto sintetico",
                f"  LOS medio: centro {c['mean_n_los']['mean']:.2f} vs costiera {k['mean_n_los']['mean']:.2f} "
                f"(delta {c['mean_n_los']['mean'] - k['mean_n_los']['mean']:+.2f})",
                f"  NLOS medio: centro {c['mean_n_nlos']['mean']:.2f} vs costiera {k['mean_n_nlos']['mean']:.2f} "
                f"(delta {c['mean_n_nlos']['mean'] - k['mean_n_nlos']['mean']:+.2f})",
                f"  Frazione NLOS/visible: centro {c['nlos_fraction']['mean']:.0%} vs costiera {k['nlos_fraction']['mean']:.0%}",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    rows = _load_rows(args.csv.resolve())
    centro_bbox = {
        "south": float(args.centro_south),
        "west": float(args.centro_west),
        "north": float(args.centro_north),
        "east": float(args.centro_east),
    }

    def in_centro(r: dict) -> bool:
        lat, lon = float(r["lat_deg"]), float(r["lon_deg"])
        return (
            centro_bbox["south"] <= lat <= centro_bbox["north"]
            and centro_bbox["west"] <= lon <= centro_bbox["east"]
        )

    centro_rows = [r for r in rows if in_centro(r)]
    lats = np.array([float(r["lat_deg"]) for r in rows], dtype=np.float64)
    lons = np.array([float(r["lon_deg"]) for r in rows], dtype=np.float64)
    zones, envelope = classify_zones(
        lats,
        lons,
        centro_bbox=centro_bbox,
        coastal_buffer_m=float(args.coastal_buffer_m),
    )
    costiera_rows = [r for r, z in zip(rows, zones) if z == "costiera"]

    if not centro_rows:
        raise RuntimeError("no points in centro storico bbox — adjust --centro-*")
    if not costiera_rows:
        raise RuntimeError("no coastal points — adjust --coastal-buffer-m")

    centro = _summarize("Centro storico", centro_rows)
    costiera = _summarize("Coastal zone", costiera_rows)

    env_lon, env_lat = envelope

    prefix = args.out_prefix.resolve()
    summary_json = Path(str(prefix) + "_summary.json")
    report_txt = Path(str(prefix) + "_report.txt")
    chart_png = Path(str(prefix) + "_comparison.png")

    payload = {
        "csv": str(args.csv.resolve()),
        "centro_bbox": centro_bbox,
        "coastal_buffer_m": float(args.coastal_buffer_m),
        "southern_envelope": {
            "n_bins": int(len(env_lon)),
            "lon_deg": env_lon.tolist(),
            "lat_deg": env_lat.tolist(),
        },
        "zones": {
            "centro_storico": {"n_points": centro.n_points, "metrics": centro.metrics},
            "costiera": {"n_points": costiera.n_points, "metrics": costiera.metrics},
        },
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_report(
        report_txt,
        centro,
        costiera,
        csv_path=args.csv.resolve(),
        coastal_buffer_m=float(args.coastal_buffer_m),
        centro_bbox=centro_bbox,
    )
    _plot_comparison(centro, costiera, chart_png)

    print(f"centro storico: {centro.n_points:,} punti")
    print(f"costiera:       {costiera.n_points:,} punti")
    print(f"wrote:\n  {summary_json}\n  {report_txt}\n  {chart_png}")


if __name__ == "__main__":
    main()
