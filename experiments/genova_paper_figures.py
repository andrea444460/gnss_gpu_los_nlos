#!/usr/bin/env python3
"""Publication figures: Genoa historic centre vs coastal GNSS visibility zones.

Outputs (300 dpi PNG):
  *_fig_map_complete       — full city map, all road points coloured by LOS
  *_fig_map_complete_nlos  — full city map, all road points coloured by NLOS
  *_fig_zone_centro         — historic centre, square, coloured points (no title/legend)
  *_fig_zone_costiera       — coastal zone, square, coloured points (no title/legend)
  *_fig1_zone_overview      — study areas on city road network
  *_fig2_los_nlos_maps      — mean LOS / NLOS along roads (two panels)
  *_fig3_metrics_bars       — bar chart comparison
  *_fig4_boxplots           — distribution boxplots

Example::

    python experiments/genova_paper_figures.py \\
      --csv \"C:/Users/Andrea/Desktop/Gneova Map/genova_area_points_osm (3).csv\" \\
      --out-prefix experiments/results/genova_paper
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
from matplotlib.lines import Line2D

from genova_zones import (
    DEFAULT_CENTRO_BBOX,
    DEFAULT_COASTAL_BUFFER_M,
    DEFAULT_MAP_CLIP_NORTH,
    classify_zones,
    filter_coast_levante,
    filter_map_rows,
)

ZONE_COLORS = {
    "centro_storico": "#c0392b",
    "costiera": "#2471a3",
    "other": "#d5d8dc",
}

METRIC_KEYS = ("mean_n_los", "mean_n_nlos", "mean_n_visible")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--out-prefix", type=Path, required=True)
    p.add_argument("--coastal-buffer-m", type=float, default=DEFAULT_COASTAL_BUFFER_M)
    p.add_argument("--centro-south", type=float, default=DEFAULT_CENTRO_BBOX["south"])
    p.add_argument("--centro-west", type=float, default=DEFAULT_CENTRO_BBOX["west"])
    p.add_argument("--centro-north", type=float, default=DEFAULT_CENTRO_BBOX["north"])
    p.add_argument("--centro-east", type=float, default=DEFAULT_CENTRO_BBOX["east"])
    p.add_argument("--bg-stride", type=int, default=25, help="Subsample 'other' zone points for overview")
    p.add_argument(
        "--map-clip-north",
        type=float,
        default=DEFAULT_MAP_CLIP_NORTH,
        help="Drop road points north of this latitude (use a large value to disable)",
    )
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
                        "mean_n_nlos": float(rr["mean_n_nlos"]),
                        "mean_n_visible": float(rr["mean_n_visible"]),
                    }
                )
            except (KeyError, ValueError):
                continue
    if not rows:
        raise ValueError(f"no valid rows in {path}")
    return rows


def _lonlat_to_webmerc(
    lon: np.ndarray | float, lat: np.ndarray | float
) -> tuple[np.ndarray, np.ndarray]:
    lon_arr = np.asarray(lon, dtype=np.float64)
    lat_arr = np.asarray(lat, dtype=np.float64)
    x = lon_arr * 20037508.34 / 180.0
    lat_clip = np.clip(lat_arr, -85.0, 85.0)
    y = np.log(np.tan((90.0 + lat_clip) * np.pi / 360.0)) * 20037508.34 / np.pi
    return x, y


def _pad_extent(
    xmin: float, xmax: float, ymin: float, ymax: float, *, frac: float = 0.10
) -> tuple[float, float, float, float]:
    dx = max(xmax - xmin, 1.0)
    dy = max(ymax - ymin, 1.0)
    return xmin - dx * frac, xmax + dx * frac, ymin - dy * frac, ymax + dy * frac


def _extent_from_lonlat(
    lons: np.ndarray, lats: np.ndarray, *, pad_frac: float = 0.10
) -> tuple[float, float, float, float]:
    x, y = _lonlat_to_webmerc(lons, lats)
    return _pad_extent(float(x.min()), float(x.max()), float(y.min()), float(y.max()), frac=pad_frac)


def _add_osm_basemap(ax: plt.Axes, extent: tuple[float, float, float, float], *, zoom: str | int = "auto") -> None:
    import contextily as cx

    xmin, xmax, ymin, ymax = extent
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    cx.add_basemap(
        ax,
        source=cx.providers.OpenStreetMap.Mapnik,
        zoom=zoom,
        attribution_size=6,
        zorder=0,
    )


def _style_map_axes(ax: plt.Axes) -> None:
    ax.set_axis_off()


def _save(fig: plt.Figure, prefix: Path, stem: str, dpi: int) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(prefix.parent / f"{prefix.name}_{stem}.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _full_city_thematic_map(
    rows: list[dict],
    *,
    key: str,
    title: str,
    cbar_label: str,
    cmap_name: str,
    out_prefix: Path,
    stem: str,
    dpi: int,
) -> None:
    """All road points on OSM basemap; compact colorbar directly under the map."""
    lons = np.array([r["lon_deg"] for r in rows], dtype=np.float64)
    lats = np.array([r["lat_deg"] for r in rows], dtype=np.float64)
    vals = np.array([r[key] for r in rows], dtype=np.float64)
    vmin = float(np.percentile(vals, 5))
    vmax = float(np.percentile(vals, 95))
    cmap = plt.get_cmap(cmap_name)

    extent = _extent_from_lonlat(lons, lats, pad_frac=0.02)
    xmin, xmax, ymin, ymax = extent
    geo_w = max(xmax - xmin, 1.0)
    geo_h = max(ymax - ymin, 1.0)
    data_aspect = geo_h / geo_w

    fig_w = 10.0
    margin_x = 0.35
    margin_top = 0.30
    cbar_h = 0.32
    gap = 0.08
    margin_bottom = 0.12
    map_h = fig_w * data_aspect
    fig_h = map_h + margin_top + cbar_h + gap + margin_bottom

    fig = plt.figure(figsize=(fig_w, fig_h))
    map_bottom = (margin_bottom + cbar_h + gap) / fig_h
    map_height = map_h / fig_h
    ax = fig.add_axes(
        [margin_x / fig_w, map_bottom, 1.0 - 2.0 * margin_x / fig_w, map_height]
    )
    cbar_bottom = margin_bottom / fig_h
    cbar_height = cbar_h / fig_h
    cbar_ax = fig.add_axes(
        [0.14, cbar_bottom, 0.72, cbar_height]
    )

    xs, ys = _lonlat_to_webmerc(lons, lats)
    _add_osm_basemap(ax, extent)
    sc = ax.scatter(
        xs, ys, c=vals, s=2.8, cmap=cmap, vmin=vmin, vmax=vmax,
        linewidths=0, alpha=0.92, zorder=3, rasterized=True,
    )
    ax.set_title(title, fontsize=12, pad=6, fontweight="bold")
    _style_map_axes(ax)

    cb = fig.colorbar(sc, cax=cbar_ax, orientation="horizontal")
    cb.set_label(cbar_label, fontsize=10, labelpad=2)
    cb.ax.tick_params(labelsize=8, pad=1)
    _save(fig, out_prefix, stem, dpi)


def _square_extent_from_lonlat(
    lons: np.ndarray, lats: np.ndarray, *, pad_frac: float = 0.10
) -> tuple[float, float, float, float]:
    x, y = _lonlat_to_webmerc(lons, lats)
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    half = 0.5 * max(xmax - xmin, ymax - ymin) * (1.0 + pad_frac)
    return cx - half, cx + half, cy - half, cy + half


def _save_tight(fig: plt.Figure, prefix: Path, stem: str, dpi: int) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        prefix.parent / f"{prefix.name}_{stem}.png",
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(fig)


def _zone_square_map(
    subset: list[dict],
    *,
    key: str,
    cmap,
    vmin: float,
    vmax: float,
    pad_frac: float,
    zoom: str | int,
    out_prefix: Path,
    stem: str,
    dpi: int,
    point_size: float = 8.0,
) -> None:
    lons = np.array([r["lon_deg"] for r in subset], dtype=np.float64)
    lats = np.array([r["lat_deg"] for r in subset], dtype=np.float64)
    vals = np.array([r[key] for r in subset], dtype=np.float64)

    fig = plt.figure(figsize=(4.5, 4.5), dpi=dpi)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    xs, ys = _lonlat_to_webmerc(lons, lats)
    extent = _square_extent_from_lonlat(lons, lats, pad_frac=pad_frac)
    print(f"[paper] basemap {stem} zoom={zoom} n={len(subset)} ...", flush=True)
    _add_osm_basemap(ax, extent, zoom=zoom)
    ax.set_box_aspect(1)
    ax.scatter(
        xs, ys, c=vals, s=point_size, cmap=cmap, vmin=vmin, vmax=vmax,
        linewidths=0, alpha=0.92, zorder=3, rasterized=True,
    )
    _style_map_axes(ax)
    out = out_prefix.parent / f"{out_prefix.name}_{stem}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, pad_inches=0)
    plt.close(fig)
    print(f"[paper] wrote {out}", flush=True)


def fig_zone_squares(
    centro: list[dict],
    cost: list[dict],
    out_prefix: Path,
    dpi: int,
    *,
    key: str = "mean_n_los",
    cmap_name: str = "RdYlGn",
) -> None:
    vals = [r[key] for r in centro + cost]
    vmin = float(np.percentile(vals, 5))
    vmax = float(np.percentile(vals, 95))
    cmap = plt.get_cmap(cmap_name)

    _zone_square_map(
        centro, key=key, cmap=cmap, vmin=vmin, vmax=vmax,
        pad_frac=0.14, zoom=16, out_prefix=out_prefix, stem="fig_zone_centro", dpi=dpi,
        point_size=9.0,
    )
    cost_levante = filter_coast_levante(cost)
    if not cost_levante:
        raise RuntimeError("no coastal Levante points — lower DEFAULT_COAST_LEVANTE_LON")
    _zone_square_map(
        cost_levante, key=key, cmap=cmap, vmin=vmin, vmax=vmax,
        pad_frac=0.10, zoom=16, out_prefix=out_prefix, stem="fig_zone_costiera", dpi=dpi,
        point_size=8.0,
    )


def fig_map_complete(rows: list[dict], out_prefix: Path, dpi: int) -> None:
    _full_city_thematic_map(
        rows,
        key="mean_n_los",
        title="Genoa — geometry-based LOS availability along sampled roads",
        cbar_label="LOS satellites",
        cmap_name="RdYlGn",
        out_prefix=out_prefix,
        stem="fig_map_complete",
        dpi=dpi,
    )
    _full_city_thematic_map(
        rows,
        key="mean_n_nlos",
        title="Genoa — building-blocked (NLOS) satellites along sampled roads",
        cbar_label="NLOS satellites",
        cmap_name="YlOrRd",
        out_prefix=out_prefix,
        stem="fig_map_complete_nlos",
        dpi=dpi,
    )


def fig_zone_overview(
    rows: list[dict],
    zones: np.ndarray,
    *,
    centro_bbox: dict[str, float],
    envelope: tuple[np.ndarray, np.ndarray],
    coastal_buffer_m: float,
    bg_stride: int,
    out_prefix: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    by_zone: dict[str, list[dict]] = {"centro_storico": [], "costiera": [], "other": []}
    for i, r in enumerate(rows):
        z = str(zones[i])
        if z == "other" and i % max(1, bg_stride) != 0:
            continue
        by_zone[z].append(r)

    plot_rows = by_zone["centro_storico"] + by_zone["costiera"] + by_zone["other"]
    plot_lons = np.array([r["lon_deg"] for r in plot_rows], dtype=np.float64)
    plot_lats = np.array([r["lat_deg"] for r in plot_rows], dtype=np.float64)
    _add_osm_basemap(ax, _extent_from_lonlat(plot_lons, plot_lats, pad_frac=0.06))

    for zone, subset in by_zone.items():
        if not subset:
            continue
        lons = np.array([r["lon_deg"] for r in subset], dtype=np.float64)
        lats = np.array([r["lat_deg"] for r in subset], dtype=np.float64)
        xs, ys = _lonlat_to_webmerc(lons, lats)
        size = 3.0 if zone == "other" else 5.5
        alpha = 0.45 if zone == "other" else 0.82
        ax.scatter(xs, ys, s=size, c=ZONE_COLORS[zone], alpha=alpha, linewidths=0, zorder=3, rasterized=True)

    env_lon, env_lat = envelope
    coast_x, coast_y = _lonlat_to_webmerc(env_lon, env_lat)
    ax.plot(coast_x, coast_y, color="#1a5276", linewidth=2.0, linestyle="--", zorder=4)

    bb = centro_bbox
    bb_x, bb_y = _lonlat_to_webmerc(
        [bb["west"], bb["east"], bb["east"], bb["west"], bb["west"]],
        [bb["south"], bb["south"], bb["north"], bb["north"], bb["south"]],
    )
    ax.plot(bb_x, bb_y, color=ZONE_COLORS["centro_storico"], linewidth=2.2, zorder=5)

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=ZONE_COLORS["centro_storico"],
               markersize=8, label=f"Historic centre ({len(by_zone['centro_storico']):,} road points)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=ZONE_COLORS["costiera"],
               markersize=8, label=f"Coastal zone ({len(by_zone['costiera']):,} road points)"),
        Line2D([0], [0], color="#1a5276", linestyle="--", linewidth=1.8,
               label=f"Southern road envelope (+{coastal_buffer_m:.0f} m inland)"),
        Line2D([0], [0], color=ZONE_COLORS["centro_storico"], linewidth=2.2,
               label="Historic-centre bounding box"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=8, framealpha=0.92)
    ax.set_title("Genoa study areas for GNSS visibility comparison", fontsize=11, pad=8)
    _style_map_axes(ax)
    fig.tight_layout()
    _save(fig, out_prefix, "fig1_zone_overview", dpi)


def _thematic_map_panel(
    ax: plt.Axes,
    subset: list[dict],
    *,
    title: str,
    key: str,
    cmap,
    vmin: float,
    vmax: float,
    pad_frac: float = 0.12,
    zoom: str | int = "auto",
) -> object:
    lons = np.array([r["lon_deg"] for r in subset], dtype=np.float64)
    lats = np.array([r["lat_deg"] for r in subset], dtype=np.float64)
    xs, ys = _lonlat_to_webmerc(lons, lats)
    _add_osm_basemap(ax, _extent_from_lonlat(lons, lats, pad_frac=pad_frac), zoom=zoom)
    sc = ax.scatter(
        xs,
        ys,
        c=[r[key] for r in subset],
        s=7,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidths=0.15,
        edgecolors="#33333333",
        alpha=0.88,
        zorder=3,
        rasterized=True,
    )
    ax.set_title(title, fontsize=10, pad=6)
    _style_map_axes(ax)
    return sc


def fig_los_nlos_panels(
    rows: list[dict],
    zones: np.ndarray,
    *,
    out_prefix: Path,
    dpi: int,
) -> None:
    centro = [r for r, z in zip(rows, zones) if z == "centro_storico"]
    cost = [r for r, z in zip(rows, zones) if z == "costiera"]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))

    vals = [r["mean_n_los"] for r in centro + cost] + [r["mean_n_nlos"] for r in centro + cost]
    vmin = float(np.percentile(vals, 5))
    vmax = float(np.percentile(vals, 95))
    cmap = plt.get_cmap("RdYlGn")

    sc = _thematic_map_panel(
        axes[0], centro, title="Historic centre", key="mean_n_los",
        cmap=cmap, vmin=vmin, vmax=vmax, pad_frac=0.18, zoom=16,
    )
    _thematic_map_panel(
        axes[1], cost, title="Coastal zone", key="mean_n_los",
        cmap=cmap, vmin=vmin, vmax=vmax, pad_frac=0.08, zoom="auto",
    )

    cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), fraction=0.035, pad=0.02)
    cbar.set_label("LOS satellites")
    fig.suptitle("Geometry-based LOS availability along sampled roads", fontsize=11, y=0.98)
    fig.tight_layout()
    _save(fig, out_prefix, "fig2_los_maps", dpi)

    fig2, axes2 = plt.subplots(1, 2, figsize=(10.5, 4.8))
    vals_n = [r["mean_n_nlos"] for r in centro + cost]
    vmin_n = float(np.percentile(vals_n, 5))
    vmax_n = float(np.percentile(vals_n, 95))
    cmap_n = plt.get_cmap("YlOrRd")

    sc2 = _thematic_map_panel(
        axes2[0], centro, title="Historic centre", key="mean_n_nlos",
        cmap=cmap_n, vmin=vmin_n, vmax=vmax_n, pad_frac=0.18, zoom=16,
    )
    _thematic_map_panel(
        axes2[1], cost, title="Coastal zone", key="mean_n_nlos",
        cmap=cmap_n, vmin=vmin_n, vmax=vmax_n, pad_frac=0.08, zoom="auto",
    )

    cbar2 = fig2.colorbar(sc2, ax=axes2.ravel().tolist(), fraction=0.035, pad=0.02)
    cbar2.set_label("NLOS satellites")
    fig2.suptitle("Building-blocked (NLOS) satellites along sampled roads", fontsize=11, y=0.98)
    fig2.tight_layout()
    _save(fig2, out_prefix, "fig2b_nlos_maps", dpi)


def fig_metrics_bars(centro: list[dict], cost: list[dict], out_prefix: Path, dpi: int) -> None:
    def mean_key(rows: list[dict], key: str) -> float:
        return float(np.mean([r[key] for r in rows]))

    def frac_nlos(rows: list[dict]) -> float:
        f = [r["mean_n_nlos"] / r["mean_n_visible"] if r["mean_n_visible"] > 0 else 0.0 for r in rows]
        return float(np.mean(f))

    keys = ["mean_n_los", "mean_n_nlos", "mean_n_visible", "nlos_fraction"]
    labels = ["LOS", "NLOS", "Visible", "NLOS / visible"]
    centro_vals = [mean_key(centro, "mean_n_los"), mean_key(centro, "mean_n_nlos"),
                   mean_key(centro, "mean_n_visible"), frac_nlos(centro)]
    cost_vals = [mean_key(cost, "mean_n_los"), mean_key(cost, "mean_n_nlos"),
                 mean_key(cost, "mean_n_visible"), frac_nlos(cost)]

    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.bar(x - w / 2, centro_vals, w, label="Historic centre", color=ZONE_COLORS["centro_storico"])
    ax.bar(x + w / 2, cost_vals, w, label="Coastal zone", color=ZONE_COLORS["costiera"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean count")
    ax.set_title("GNSS visibility metrics by study area")
    ax.legend(frameon=True)
    ax.grid(True, axis="y", alpha=0.3)
    for i, (cv, kv) in enumerate(zip(centro_vals, cost_vals)):
        ax.text(i - w / 2, cv + 0.3, f"{cv:.1f}", ha="center", va="bottom", fontsize=7)
        ax.text(i + w / 2, kv + 0.3, f"{kv:.1f}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    _save(fig, out_prefix, "fig3_metrics_bars", dpi)


def fig_boxplots(centro: list[dict], cost: list[dict], out_prefix: Path, dpi: int) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.8))
    metrics = [
        ("mean_n_visible", "Visible satellites"),
        ("mean_n_los", "LOS satellites"),
        ("mean_n_nlos", "NLOS satellites"),
    ]
    data_map = {
        "mean_n_visible": ([r["mean_n_visible"] for r in centro], [r["mean_n_visible"] for r in cost]),
        "mean_n_los": ([r["mean_n_los"] for r in centro], [r["mean_n_los"] for r in cost]),
        "mean_n_nlos": ([r["mean_n_nlos"] for r in centro], [r["mean_n_nlos"] for r in cost]),
    }

    for ax, (key, ylabel) in zip(axes, metrics):
        cvals, kvals = data_map[key]
        bp = ax.boxplot(
            [cvals, kvals],
            tick_labels=["Historic centre", "Coastal zone"],
            patch_artist=True,
            widths=0.55,
            showfliers=False,
        )
        bp["boxes"][0].set_facecolor(ZONE_COLORS["centro_storico"])
        bp["boxes"][1].set_facecolor(ZONE_COLORS["costiera"])
        for b in bp["boxes"]:
            b.set_alpha(0.65)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Road-point visibility distributions", fontsize=11)
    fig.tight_layout()
    _save(fig, out_prefix, "fig4_boxplots", dpi)


def main() -> None:
    args = _parse_args()
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": args.dpi,
        "savefig.dpi": args.dpi,
    })

    rows = _load_rows(args.csv.resolve())
    n_raw = len(rows)
    north_clip = float(args.map_clip_north) if float(args.map_clip_north) < 90.0 else None
    rows = filter_map_rows(rows, north_max=north_clip)
    if not rows:
        raise RuntimeError("no points left after map clip")
    if len(rows) < n_raw:
        print(f"[paper] clipped {n_raw - len(rows):,} points north of {north_clip:.4f} ({len(rows):,} kept)")
    lats = np.array([r["lat_deg"] for r in rows], dtype=np.float64)
    lons = np.array([r["lon_deg"] for r in rows], dtype=np.float64)
    centro_bbox = {
        "south": float(args.centro_south),
        "west": float(args.centro_west),
        "north": float(args.centro_north),
        "east": float(args.centro_east),
    }
    zones, envelope = classify_zones(
        lats, lons,
        centro_bbox=centro_bbox,
        coastal_buffer_m=float(args.coastal_buffer_m),
    )
    centro = [r for r, z in zip(rows, zones) if z == "centro_storico"]
    cost = [r for r, z in zip(rows, zones) if z == "costiera"]
    if not centro or not cost:
        raise RuntimeError("empty zone after classification")

    prefix = args.out_prefix.resolve()
    print(f"[paper] loaded {len(rows):,} points | centro={len(centro):,} costiera={len(cost):,}")

    fig_map_complete(rows, out_prefix=prefix, dpi=int(args.dpi))
    fig_zone_squares(centro, cost, prefix, int(args.dpi))
    fig_zone_overview(
        rows, zones, centro_bbox=centro_bbox, envelope=envelope,
        coastal_buffer_m=float(args.coastal_buffer_m), bg_stride=int(args.bg_stride),
        out_prefix=prefix, dpi=int(args.dpi),
    )
    fig_los_nlos_panels(rows, zones, out_prefix=prefix, dpi=int(args.dpi))
    fig_metrics_bars(centro, cost, prefix, int(args.dpi))
    fig_boxplots(centro, cost, prefix, int(args.dpi))

    stems = [
        "fig_map_complete", "fig_map_complete_nlos", "fig_zone_centro", "fig_zone_costiera",
        "fig1_zone_overview",
        "fig2_los_maps", "fig2b_nlos_maps", "fig3_metrics_bars", "fig4_boxplots",
    ]
    print("[paper] wrote:")
    for s in stems:
        print(f"  {prefix.parent / (prefix.name + '_' + s)}.png")


if __name__ == "__main__":
    main()
