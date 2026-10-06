#!/usr/bin/env python3
"""Compare area LOS/NLOS map satellite counts with UrbanNav RINEX along the drive.

For each RINEX epoch along the UrbanNav reference trajectory:
  - find the nearest road-grid point from ``build_area_los_nlos_map`` CSV
  - compare map ``mean_n_*`` vs RINEX satellite counts (tracked / good C/N0 / weak C/N0)

Outputs: epoch CSV, summary JSON, scatter/timeline PNG, residual Folium HTML.

Example (after area map is built)::

    PYTHONPATH=python python experiments/validate_area_map_vs_urbannav_rinex.py \\
      --area-csv experiments/results/odaiba_area_points.csv \\
      --run-dir experiments/data/urbannav/Odaiba \\
      --rover-source trimble \\
      --nav-systems G,E,J,C \\
      --map-dt-s 300 \\
      --out-prefix experiments/results/odaiba_map_vs_rinex
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_ROOT / "python"))

from gnss_gpu.bvh import BVHAccelerator
from gnss_gpu.city_model_validator import (
    CN0_LOS_THRESHOLD,
    CN0_NLOS_THRESHOLD,
    ELEVATION_MASK_DEG,
)
from gnss_gpu.ephemeris import Ephemeris
from gnss_gpu.io.nav_rinex import _datetime_to_gps_seconds_of_week, read_nav_rinex_multi
from gnss_gpu.io.rinex import read_rinex_obs
from gnss_gpu.io.urbannav import UrbanNavLoader, _pick_observation_value, _SYSTEM_PR_FALLBACKS, _SYSTEM_SNR_FALLBACKS
from gnss_gpu.raytrace import BuildingModel
from gnss_gpu.urban_signal_sim import _sat_elevation_azimuth, ecef_to_lla

WGS84_A = 6_378_137.0


@dataclass(frozen=True)
class MapPoint:
    lat_deg: float
    lon_deg: float
    mean_n_los: float
    mean_n_nlos: float
    mean_n_visible: float


@dataclass(frozen=True)
class PairMetrics:
    n: int
    pearson_r: float
    mae: float
    rmse: float
    bias: float


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--area-csv", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--rover-source", type=str, default="trimble")
    p.add_argument("--nav-systems", type=str, default="G,E,J,C")
    p.add_argument("--max-neighbor-m", type=float, default=30.0)
    p.add_argument("--epoch-step", type=int, default=1)
    p.add_argument("--max-epochs", type=int, default=0, help="0 = all epochs")
    p.add_argument("--map-dt-s", type=float, default=300.0, help="Area-map ephemeris step; RINEX rolling window default.")
    p.add_argument("--rinex-window-s", type=float, default=-1.0, help="-1 uses --map-dt-s")
    p.add_argument("--out-prefix", type=Path, required=True)
    p.add_argument("--triangles-npy", type=Path, default=None, help="Required with --instantaneous-geometry")
    p.add_argument("--instantaneous-geometry", action="store_true")
    p.add_argument("--elevation-mask-deg", type=float, default=ELEVATION_MASK_DEG)
    return p.parse_args()


def _haversine_m(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    lat1r = math.radians(lat1)
    lon1r = math.radians(lon1)
    lat2r = np.radians(lat2)
    lon2r = np.radians(lon2)
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    return 2.0 * WGS84_A * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _load_area_map(path: Path) -> list[MapPoint]:
    rows: list[MapPoint] = []
    with open(path, newline="", encoding="utf-8") as f:
        for rr in csv.DictReader(f):
            try:
                rows.append(
                    MapPoint(
                        lat_deg=float(rr["lat_deg"]),
                        lon_deg=float(rr["lon_deg"]),
                        mean_n_los=float(rr["mean_n_los"]),
                        mean_n_nlos=float(rr["mean_n_nlos"]),
                        mean_n_visible=float(rr["mean_n_visible"]),
                    )
                )
            except (KeyError, ValueError):
                continue
    if not rows:
        raise ValueError(f"no valid rows in area csv: {path}")
    return rows


def _interp_ref_ecef(gt_times: np.ndarray, gt_ecef: np.ndarray, tow: float) -> np.ndarray | None:
    if gt_times.size == 0:
        return None
    diffs = np.abs(gt_times - tow)
    idx = int(np.argmin(diffs))
    if float(diffs[idx]) > 10.0:
        return None
    if idx == 0 or idx == len(gt_times) - 1:
        return gt_ecef[idx]
    if tow > gt_times[idx] and idx < len(gt_times) - 1:
        i0, i1 = idx, idx + 1
    else:
        i0, i1 = max(0, idx - 1), idx
    span = float(gt_times[i1] - gt_times[i0])
    if span <= 0.0:
        return gt_ecef[idx]
    alpha = float(np.clip((tow - gt_times[i0]) / span, 0.0, 1.0))
    return (1.0 - alpha) * gt_ecef[i0] + alpha * gt_ecef[i1]


def _nearest_map_point(
    lat_deg: float,
    lon_deg: float,
    map_pts: list[MapPoint],
    map_lats: np.ndarray,
    map_lons: np.ndarray,
    max_neighbor_m: float,
) -> tuple[MapPoint | None, float]:
    dist = _haversine_m(lat_deg, lon_deg, map_lats, map_lons)
    idx = int(np.argmin(dist))
    dmin = float(dist[idx])
    if dmin > float(max_neighbor_m):
        return None, dmin
    return map_pts[idx], dmin


def _count_rinex_epoch(
    epoch,
    systems: tuple[str, ...],
) -> tuple[int, int, int]:
    n_tracked = 0
    n_good = 0
    n_weak = 0
    for sat_id in epoch.satellites:
        if not sat_id or sat_id[0] not in systems:
            continue
        obs = epoch.observations.get(sat_id, {})
        _code, pr = _pick_observation_value(sat_id[0], obs, "C1C", _SYSTEM_PR_FALLBACKS)
        if not pr or pr < 1e6:
            continue
        n_tracked += 1
        _s_code, cn0 = _pick_observation_value(sat_id[0], obs, "S1C", _SYSTEM_SNR_FALLBACKS)
        if not np.isfinite(cn0) or cn0 <= 0.0:
            continue
        if float(cn0) >= CN0_LOS_THRESHOLD:
            n_good += 1
        if float(cn0) < CN0_NLOS_THRESHOLD:
            n_weak += 1
    return n_tracked, n_good, n_weak


def _rolling_mean(values: np.ndarray, center_idx: int, tows: np.ndarray, window_s: float) -> float:
    if values.size == 0:
        return float("nan")
    if window_s <= 0.0:
        return float(values[center_idx])
    t0 = float(tows[center_idx])
    half = float(window_s) * 0.5
    mask = np.abs(tows - t0) <= half
    if not np.any(mask):
        return float(values[center_idx])
    return float(np.mean(values[mask]))


def _pair_metrics(x: np.ndarray, y: np.ndarray) -> PairMetrics:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = int(x.size)
    if n < 2:
        return PairMetrics(n=n, pearson_r=float("nan"), mae=float("nan"), rmse=float("nan"), bias=float("nan"))
    if np.std(x) < 1e-9 or np.std(y) < 1e-9:
        r = float("nan")
    else:
        r = float(np.corrcoef(x, y)[0, 1])
    diff = y - x
    return PairMetrics(
        n=n,
        pearson_r=r,
        mae=float(np.mean(np.abs(diff))),
        rmse=float(np.sqrt(np.mean(diff ** 2))),
        bias=float(np.mean(diff)),
    )


def _instantaneous_counts(
    rx_ecef: np.ndarray,
    tow: float,
    eph: Ephemeris,
    bvh: BVHAccelerator,
    prn_catalog: list[str],
    mask_rad: float,
) -> tuple[int, int, int]:
    sat_ecef, _clk, used = eph.compute(float(tow), prn_list=prn_catalog)
    if len(used) == 0:
        return 0, 0, 0
    sats = np.asarray(sat_ecef, dtype=np.float64).reshape(-1, 3)
    el, _az = _sat_elevation_azimuth(rx_ecef, sats)
    visible = el >= mask_rad
    n_visible = int(np.sum(visible))
    if n_visible == 0:
        return 0, 0, 0
    los = np.asarray(bvh.check_los(rx_ecef, sats[visible]), dtype=bool)
    n_los = int(np.sum(los))
    n_nlos = int(np.sum(~los))
    return n_visible, n_los, n_nlos


def _write_epoch_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "gps_tow",
        "lat_deg",
        "lon_deg",
        "neighbor_dist_m",
        "mean_n_visible",
        "mean_n_los",
        "mean_n_nlos",
        "n_tracked",
        "n_good_cn0",
        "n_weak_cn0",
        "n_tracked_roll",
        "n_good_cn0_roll",
        "n_weak_cn0_roll",
        "delta_visible_tracked",
        "delta_los_good",
        "delta_nlos_weak",
        "inst_n_visible",
        "inst_n_los",
        "inst_n_nlos",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _plot_scatter(rows: list[dict], out_png: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    pairs = [
        ("mean_n_visible", "n_tracked_roll", "Visible (map) vs tracked (RINEX roll)"),
        ("mean_n_los", "n_good_cn0_roll", "LOS (map) vs good C/N0 (RINEX roll)"),
        ("mean_n_nlos", "n_weak_cn0_roll", "NLOS (map) vs weak C/N0 (RINEX roll)"),
    ]
    for ax, (xk, yk, title) in zip(axes, pairs):
        x = np.array([float(r[xk]) for r in rows], dtype=np.float64)
        y = np.array([float(r[yk]) for r in rows], dtype=np.float64)
        ax.scatter(x, y, s=8, alpha=0.35, c="#1f77b4")
        if x.size:
            lo = float(min(np.min(x), np.min(y)))
            hi = float(max(np.max(x), np.max(y)))
            ax.plot([lo, hi], [lo, hi], "--", color="#888", lw=1)
        ax.set_xlabel(xk)
        ax.set_ylabel(yk)
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Area map vs UrbanNav RINEX satellite counts", fontsize=12)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_timeline(rows: list[dict], out_png: Path) -> None:
    tow = np.array([float(r["gps_tow"]) for r in rows], dtype=np.float64)
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    series = [
        ("mean_n_visible", "n_tracked_roll", "Visible / tracked"),
        ("mean_n_los", "n_good_cn0_roll", "LOS / good C/N0"),
        ("mean_n_nlos", "n_weak_cn0_roll", "NLOS / weak C/N0"),
    ]
    for ax, (map_k, rin_k, title) in zip(axes, series):
        ax.plot(tow, [float(r[map_k]) for r in rows], label=f"map {map_k}", lw=1.2)
        ax.plot(tow, [float(r[rin_k]) for r in rows], label=f"rinex {rin_k}", lw=1.0, alpha=0.85)
        ax.set_ylabel("count")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("GPS TOW [s]")
    fig.suptitle("Satellite counts along UrbanNav drive", fontsize=12)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_residual_map(rows: list[dict], out_html: Path) -> None:
    import branca.colormap as bcm
    import folium

    lats = [float(r["lat_deg"]) for r in rows]
    lons = [float(r["lon_deg"]) for r in rows]
    res = np.array([float(r["delta_visible_tracked"]) for r in rows], dtype=np.float64)
    lat0 = float(np.mean(lats))
    lon0 = float(np.mean(lons))
    m = folium.Map(location=[lat0, lon0], zoom_start=15, tiles="OpenStreetMap")
    vmin = float(np.nanpercentile(res, 5))
    vmax = float(np.nanpercentile(res, 95))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = float(np.nanmin(res)), float(np.nanmax(res) + 1.0)
    cmap = bcm.LinearColormap(["#d73027", "#ffffbf", "#1a9850"], vmin=vmin, vmax=vmax)
    cmap.caption = "n_tracked_roll - mean_n_visible (positive = RINEX sees more)"
    cmap.add_to(m)
    for lat, lon, delta in zip(lats, lons, res):
        folium.CircleMarker(
            location=[lat, lon],
            radius=3,
            color=cmap(delta),
            fill=True,
            fill_color=cmap(delta),
            fill_opacity=0.85,
            weight=1,
            popup=f"delta={delta:.2f}",
        ).add_to(m)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_html))


def main() -> None:
    args = _parse_args()
    systems = tuple(s.strip() for s in str(args.nav_systems).split(",") if s.strip())
    rinex_window_s = float(args.map_dt_s) if float(args.rinex_window_s) < 0 else float(args.rinex_window_s)

    map_pts = _load_area_map(args.area_csv.resolve())
    map_lats = np.array([p.lat_deg for p in map_pts], dtype=np.float64)
    map_lons = np.array([p.lon_deg for p in map_pts], dtype=np.float64)
    print(f"[validate] area map points: {len(map_pts)} from {args.area_csv}")

    loader = UrbanNavLoader(args.run_dir.resolve())
    gt_times, gt_ecef = loader.load_ground_truth()
    rover_path = loader._find_rover_obs(rover_source=str(args.rover_source))
    if rover_path is None:
        raise FileNotFoundError(f"rover obs not found in {args.run_dir}")
    rinex = read_rinex_obs(rover_path)
    print(f"[validate] RINEX epochs: {len(rinex.epochs)} from {rover_path}")

    bvh = None
    eph = None
    prn_catalog: list[str] = []
    mask_rad = math.radians(float(args.elevation_mask_deg))
    if args.instantaneous_geometry:
        if args.triangles_npy is None:
            raise ValueError("--triangles-npy is required with --instantaneous-geometry")
        tri = np.asarray(np.load(args.triangles_npy), dtype=np.float64)
        building = BuildingModel(tri)
        bvh = BVHAccelerator.from_building_model(building)
        nav_path = loader._find_base_nav()
        nav = read_nav_rinex_multi(str(nav_path), systems=systems)
        eph = Ephemeris(nav)
        prn_catalog = list(eph.available_prns)
        print(f"[validate] instantaneous geometry: {tri.shape[0]} triangles, {len(prn_catalog)} PRNs")

    # Pass 1: per-epoch RINEX counts + map join.
    raw_epochs: list[dict] = []
    skipped_neighbor = 0
    skipped_ref = 0
    epoch_indices = range(0, len(rinex.epochs), max(1, int(args.epoch_step)))
    if int(args.max_epochs) > 0:
        epoch_indices = list(epoch_indices)[: int(args.max_epochs)]

    for ei in epoch_indices:
        ep = rinex.epochs[ei]
        tow = float(_datetime_to_gps_seconds_of_week(ep.time))
        rx = _interp_ref_ecef(gt_times, gt_ecef, tow)
        if rx is None:
            skipped_ref += 1
            continue
        lat, lon, _alt = ecef_to_lla(float(rx[0]), float(rx[1]), float(rx[2]))
        lat_deg = math.degrees(lat)
        lon_deg = math.degrees(lon)

        mp, dist_m = _nearest_map_point(lat_deg, lon_deg, map_pts, map_lats, map_lons, float(args.max_neighbor_m))
        if mp is None:
            skipped_neighbor += 1
            continue

        n_tracked, n_good, n_weak = _count_rinex_epoch(ep, systems)
        inst_vis = inst_los = inst_nlos = float("nan")
        if bvh is not None and eph is not None:
            iv, il, inl = _instantaneous_counts(rx, tow, eph, bvh, prn_catalog, mask_rad)
            inst_vis, inst_los, inst_nlos = float(iv), float(il), float(inl)

        raw_epochs.append(
            {
                "gps_tow": tow,
                "lat_deg": lat_deg,
                "lon_deg": lon_deg,
                "neighbor_dist_m": float(dist_m),
                "mean_n_visible": float(mp.mean_n_visible),
                "mean_n_los": float(mp.mean_n_los),
                "mean_n_nlos": float(mp.mean_n_nlos),
                "n_tracked": int(n_tracked),
                "n_good_cn0": int(n_good),
                "n_weak_cn0": int(n_weak),
                "inst_n_visible": inst_vis,
                "inst_n_los": inst_los,
                "inst_n_nlos": inst_nlos,
            }
        )

    if not raw_epochs:
        raise RuntimeError(
            f"no epochs matched (skipped_ref={skipped_ref}, skipped_neighbor={skipped_neighbor})"
        )

    tows = np.array([float(r["gps_tow"]) for r in raw_epochs], dtype=np.float64)
    tracked = np.array([float(r["n_tracked"]) for r in raw_epochs], dtype=np.float64)
    good = np.array([float(r["n_good_cn0"]) for r in raw_epochs], dtype=np.float64)
    weak = np.array([float(r["n_weak_cn0"]) for r in raw_epochs], dtype=np.float64)

    rows: list[dict] = []
    for i, r in enumerate(raw_epochs):
        n_tr_roll = _rolling_mean(tracked, i, tows, rinex_window_s)
        n_good_roll = _rolling_mean(good, i, tows, rinex_window_s)
        n_weak_roll = _rolling_mean(weak, i, tows, rinex_window_s)
        row = dict(r)
        row.update(
            {
                "n_tracked_roll": n_tr_roll,
                "n_good_cn0_roll": n_good_roll,
                "n_weak_cn0_roll": n_weak_roll,
                "delta_visible_tracked": n_tr_roll - float(r["mean_n_visible"]),
                "delta_los_good": n_good_roll - float(r["mean_n_los"]),
                "delta_nlos_weak": n_weak_roll - float(r["mean_n_nlos"]),
            }
        )
        rows.append(row)

    prefix = args.out_prefix.resolve()
    epoch_csv = Path(str(prefix) + "_epoch_comparison.csv")
    summary_json = Path(str(prefix) + "_summary.json")
    scatter_png = Path(str(prefix) + "_scatter.png")
    timeline_png = Path(str(prefix) + "_timeline.png")
    residual_html = Path(str(prefix) + "_residual_map.html")

    _write_epoch_csv(epoch_csv, rows)

    map_visible = np.array([float(r["mean_n_visible"]) for r in rows])
    map_los = np.array([float(r["mean_n_los"]) for r in rows])
    map_nlos = np.array([float(r["mean_n_nlos"]) for r in rows])
    rin_tr = np.array([float(r["n_tracked_roll"]) for r in rows])
    rin_good = np.array([float(r["n_good_cn0_roll"]) for r in rows])
    rin_weak = np.array([float(r["n_weak_cn0_roll"]) for r in rows])

    summary = {
        "area_csv": str(args.area_csv.resolve()),
        "run_dir": str(args.run_dir.resolve()),
        "nav_systems": list(systems),
        "rinex_window_s": rinex_window_s,
        "max_neighbor_m": float(args.max_neighbor_m),
        "n_epochs_used": int(len(rows)),
        "skipped_ref": int(skipped_ref),
        "skipped_neighbor": int(skipped_neighbor),
        "pairs": {
            "visible_vs_tracked": _pair_metrics(map_visible, rin_tr).__dict__,
            "los_vs_good_cn0": _pair_metrics(map_los, rin_good).__dict__,
            "nlos_vs_weak_cn0": _pair_metrics(map_nlos, rin_weak).__dict__,
        },
        "thresholds": {
            "cn0_los_dbhz": CN0_LOS_THRESHOLD,
            "cn0_nlos_dbhz": CN0_NLOS_THRESHOLD,
            "elevation_mask_deg": float(args.elevation_mask_deg),
        },
        "outputs": {
            "epoch_csv": str(epoch_csv),
            "scatter_png": str(scatter_png),
            "timeline_png": str(timeline_png),
            "residual_html": str(residual_html),
        },
    }

    if args.instantaneous_geometry:
        inst_vis = np.array([float(r["inst_n_visible"]) for r in rows])
        inst_los = np.array([float(r["inst_n_los"]) for r in rows])
        inst_nlos = np.array([float(r["inst_n_nlos"]) for r in rows])
        summary["instantaneous_pairs"] = {
            "visible_vs_tracked": _pair_metrics(inst_vis, rin_tr).__dict__,
            "los_vs_good_cn0": _pair_metrics(inst_los, rin_good).__dict__,
            "nlos_vs_weak_cn0": _pair_metrics(inst_nlos, rin_weak).__dict__,
            "visible_map_vs_inst": _pair_metrics(map_visible, inst_vis).__dict__,
            "los_map_vs_inst": _pair_metrics(map_los, inst_los).__dict__,
        }

    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    _plot_scatter(rows, scatter_png)
    _plot_timeline(rows, timeline_png)
    _plot_residual_map(rows, residual_html)

    print(f"[validate] epochs used: {len(rows)} (skipped ref={skipped_ref}, neighbor={skipped_neighbor})")
    print(f"[validate] visible vs tracked: r={summary['pairs']['visible_vs_tracked']['pearson_r']:.3f} "
          f"MAE={summary['pairs']['visible_vs_tracked']['mae']:.3f}")
    print(f"[validate] los vs good_cn0: r={summary['pairs']['los_vs_good_cn0']['pearson_r']:.3f} "
          f"MAE={summary['pairs']['los_vs_good_cn0']['mae']:.3f}")
    print(f"[validate] wrote:\n  {epoch_csv}\n  {summary_json}\n  {scatter_png}\n  {timeline_png}\n  {residual_html}")


if __name__ == "__main__":
    main()
