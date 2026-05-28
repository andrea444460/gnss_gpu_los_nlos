#!/usr/bin/env python3
"""Build LOS/NLOS coverage map on OSM road points inside a bbox.

Pipeline:
- fetch roads from OSM (Overpass)
- sample road points every N meters
- evaluate LOS/NLOS counts over time using NAV ephemeris + building mesh
- optional DEM horizon prefilter for terrain blocking
- output CSV and HTML map
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_ROOT / "python"))

import branca.colormap as bcm
import folium
import numpy as np

from gnss_gpu.bvh import BVHAccelerator
from gnss_gpu.ephemeris import Ephemeris
from gnss_gpu.io.dem_download import download_dem_for_bbox
from gnss_gpu.io.nav_rinex import read_nav_rinex_multi
from gnss_gpu.io.osm_roads import BBox, fetch_roads_overpass, sample_road_points, split_bbox_into_tiles
from gnss_gpu.raytrace import BuildingModel
from gnss_gpu.terrain_horizon import HorizonConfig, TerrainHorizonMask
from gnss_gpu.urban_signal_sim import _sat_elevation_azimuth


WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = 2.0 * WGS84_F - WGS84_F * WGS84_F


def _gps_week_tow_from_utc(dt: datetime) -> tuple[int, float]:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    gps_epoch = datetime(1980, 1, 6, tzinfo=timezone.utc)
    leap_seconds = 18.0
    sec = (dt.astimezone(timezone.utc) - gps_epoch).total_seconds() + leap_seconds
    week = int(sec // 604800.0)
    tow = float(sec - week * 604800.0)
    return week, tow


def _lla_deg_to_ecef(lat_deg: float, lon_deg: float, alt_m: float) -> np.ndarray:
    lat = math.radians(float(lat_deg))
    lon = math.radians(float(lon_deg))
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    return np.array(
        [
            (n + alt_m) * cos_lat * cos_lon,
            (n + alt_m) * cos_lat * sin_lon,
            (n * (1.0 - WGS84_E2) + alt_m) * sin_lat,
        ],
        dtype=np.float64,
    )


def _time_samples(start_tow_s: float, duration_s: float, dt_s: float) -> np.ndarray:
    n = max(1, int(math.floor(float(duration_s) / float(dt_s))) + 1)
    tows = float(start_tow_s) + np.arange(n, dtype=np.float64) * float(dt_s)
    return np.mod(tows, 604800.0)


def _build_html_map(rows: list[dict], out_html: Path, *, metric_key: str = "mean_n_los") -> None:
    if not rows:
        raise ValueError("No rows for HTML map.")
    lat0 = float(np.mean([r["lat_deg"] for r in rows]))
    lon0 = float(np.mean([r["lon_deg"] for r in rows]))
    m = folium.Map(location=[lat0, lon0], zoom_start=14, tiles="OpenStreetMap")
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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--south", type=float, required=True)
    p.add_argument("--west", type=float, required=True)
    p.add_argument("--north", type=float, required=True)
    p.add_argument("--east", type=float, required=True)
    p.add_argument("--triangles-npy", type=Path, required=True)
    p.add_argument("--nav", type=Path, required=True)
    p.add_argument("--out-csv", type=Path, required=True)
    p.add_argument("--out-html", type=Path, required=True)
    p.add_argument("--step-m", type=float, default=25.0, help="Road sampling step in meters.")
    p.add_argument("--rx-alt-m", type=float, default=1.5, help="Receiver altitude above ellipsoid [m].")
    p.add_argument("--include-pedestrian", action="store_true")
    p.add_argument("--tile-size-m", type=float, default=0.0, help="Spatial tile size for scaling hooks.")
    p.add_argument("--dt-s", type=float, default=300.0, help="Time sampling step [s].")
    p.add_argument("--duration-s", type=float, default=7200.0, help="Total simulated duration [s].")
    p.add_argument("--start-utc", type=str, default="", help="UTC ISO start, e.g. 2026-05-28T08:00:00Z")
    p.add_argument("--tow-start-s", type=float, default=0.0, help="Fallback GPS TOW start if --start-utc is empty.")
    p.add_argument("--elevation-mask-deg", type=float, default=10.0)
    p.add_argument("--nav-systems", type=str, default="G,E,J,C", help="Comma-separated RINEX systems.")
    p.add_argument("--eph-batch-chunk", type=int, default=16)
    p.add_argument("--point-batch-chunk", type=int, default=128)
    p.add_argument("--dem-path", type=Path, default=None)
    p.add_argument("--dem-auto-download", action="store_true")
    p.add_argument("--dem-auto-out", type=Path, default=Path("experiments/results/area_auto_dem.tif"))
    p.add_argument("--dem-auto-zoom", type=int, default=12)
    p.add_argument("--terrain-max-distance-m", type=float, default=20000.0)
    p.add_argument("--terrain-step-m", type=float, default=60.0)
    p.add_argument("--terrain-azimuth-step-deg", type=float, default=2.0)
    p.add_argument("--terrain-cache-resolution-m", type=float, default=50.0)
    p.add_argument("--terrain-margin-deg", type=float, default=0.0)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    t0 = time.perf_counter()
    bbox = BBox(args.south, args.west, args.north, args.east)

    # Roads sampling with tiling.
    tile_boxes = split_bbox_into_tiles(bbox, tile_size_m=float(args.tile_size_m))
    all_points: list[dict] = []
    for tb in tile_boxes:
        roads = fetch_roads_overpass(tb, include_pedestrian=bool(args.include_pedestrian))
        pts = sample_road_points(roads, step_m=float(args.step_m))
        all_points.extend(pts)
    # Dedup across tiles.
    uniq: dict[tuple[int, int], dict] = {}
    for p in all_points:
        k = (int(round(float(p["lat_deg"]) * 1e6)), int(round(float(p["lon_deg"]) * 1e6)))
        uniq[k] = p
    points = list(uniq.values())
    if not points:
        raise RuntimeError("No road points sampled in the selected area.")
    print(f"[area] sampled road points: {len(points)} from {len(tile_boxes)} tile(s)")

    # Mesh + BVH.
    tri = np.asarray(np.load(args.triangles_npy), dtype=np.float64)
    if tri.ndim != 3 or tri.shape[1:] != (3, 3):
        raise ValueError(f"--triangles-npy must have shape [N,3,3], got {tri.shape}")
    building = BuildingModel(tri)
    bvh = BVHAccelerator.from_building_model(building)
    if not hasattr(bvh, "check_los_batch"):
        raise RuntimeError("BVH check_los_batch unavailable; rebuild gnss_gpu with CUDA BVH support.")
    print(f"[area] mesh triangles={len(tri)}, bvh_nodes={bvh.n_nodes}")

    # Ephemeris.
    systems = tuple(s.strip() for s in str(args.nav_systems).split(",") if s.strip())
    nav_messages = read_nav_rinex_multi(str(args.nav.resolve()), systems=systems)
    eph = Ephemeris(nav_messages)
    prn_catalog = eph.available_prns
    print(f"[area] NAV PRNs available: {len(prn_catalog)} ({','.join(systems)})")

    if args.start_utc.strip():
        dt = datetime.fromisoformat(args.start_utc.replace("Z", "+00:00"))
        _week, tow_start = _gps_week_tow_from_utc(dt)
    else:
        tow_start = float(args.tow_start_s)
    tow_samples = _time_samples(tow_start, float(args.duration_s), float(args.dt_s))
    print(f"[area] time samples: {tow_samples.size} (dt={args.dt_s:g}s, duration={args.duration_s:g}s)")

    # Optional terrain prefilter.
    dem_path = args.dem_path
    if dem_path is None and args.dem_auto_download:
        dem_path, ntiles = download_dem_for_bbox(
            south=float(bbox.south),
            west=float(bbox.west),
            north=float(bbox.north),
            east=float(bbox.east),
            output_tif=args.dem_auto_out,
            zoom=int(args.dem_auto_zoom),
        )
        print(f"[area] DEM auto-downloaded: {dem_path} ({ntiles} tiles)")

    terrain_mask: TerrainHorizonMask | None = None
    if dem_path is not None:
        terrain_mask = TerrainHorizonMask(
            dem_path,
            HorizonConfig(
                max_distance_m=float(args.terrain_max_distance_m),
                sample_step_m=float(args.terrain_step_m),
                azimuth_step_deg=float(args.terrain_azimuth_step_deg),
                cache_resolution_m=float(args.terrain_cache_resolution_m),
                margin_deg=float(args.terrain_margin_deg),
            ),
        )
        print(f"[area] terrain prefilter: enabled ({dem_path})")
    else:
        print("[area] terrain prefilter: disabled")

    n_points = len(points)
    los_sum = np.zeros(n_points, dtype=np.float64)
    nlos_sum = np.zeros(n_points, dtype=np.float64)
    vis_sum = np.zeros(n_points, dtype=np.float64)
    tblk_sum = np.zeros(n_points, dtype=np.float64)
    n_epochs_total = 0

    point_ecef = np.asarray(
        [_lla_deg_to_ecef(p["lat_deg"], p["lon_deg"], float(args.rx_alt_m)) for p in points],
        dtype=np.float64,
    )
    p_chunk = max(1, int(args.point_batch_chunk))
    e_chunk = max(1, int(args.eph_batch_chunk))
    mask_rad = math.radians(float(args.elevation_mask_deg))

    for es in range(0, tow_samples.size, e_chunk):
        ee = min(es + e_chunk, tow_samples.size)
        tow_blk = np.asarray(tow_samples[es:ee], dtype=np.float64)
        sat_b, _clk_b, _used = eph.compute_batch(tow_blk, prn_list=prn_catalog)
        if sat_b.shape[1] == 0:
            continue
        n_t, n_sat = sat_b.shape[0], sat_b.shape[1]
        n_epochs_total += n_t

        for ps in range(0, n_points, p_chunk):
            pe = min(ps + p_chunk, n_points)
            rx_chunk = point_ecef[ps:pe]
            n_p = rx_chunk.shape[0]

            rx_flat = np.repeat(rx_chunk, n_t, axis=0)  # point-major
            sat_flat = np.tile(sat_b, (n_p, 1, 1))  # point-major [n_p*n_t, n_sat, 3]
            sat_work = np.array(sat_flat, copy=True)
            visible = np.zeros((n_p * n_t, n_sat), dtype=bool)
            terrain_blocked = np.zeros((n_p * n_t, n_sat), dtype=bool)

            for pi in range(n_p):
                rx = rx_chunk[pi]
                for ti in range(n_t):
                    idx = pi * n_t + ti
                    sats = sat_b[ti]
                    el, _az = _sat_elevation_azimuth(rx, sats)
                    vis = el >= mask_rad
                    if terrain_mask is not None:
                        terr_vis = terrain_mask.terrain_visible_mask(rx, sats)
                        terrain_blocked[idx] = ~terr_vis
                        vis = np.logical_and(vis, terr_vis)
                    visible[idx] = vis
                    sat_work[idx][~vis] = np.nan

            los = np.asarray(bvh.check_los_batch(rx_flat, sat_work), dtype=bool)
            los_vis = np.logical_and(los, visible)
            nlos_vis = np.logical_and(~los, visible)

            los_3d = los_vis.reshape(n_p, n_t, n_sat)
            nlos_3d = nlos_vis.reshape(n_p, n_t, n_sat)
            vis_3d = visible.reshape(n_p, n_t, n_sat)
            tblk_3d = terrain_blocked.reshape(n_p, n_t, n_sat)
            los_sum[ps:pe] += np.sum(los_3d, axis=(1, 2))
            nlos_sum[ps:pe] += np.sum(nlos_3d, axis=(1, 2))
            vis_sum[ps:pe] += np.sum(vis_3d, axis=(1, 2))
            tblk_sum[ps:pe] += np.sum(tblk_3d, axis=(1, 2))

    if n_epochs_total <= 0:
        raise RuntimeError("No valid epochs processed.")

    # Mean satellites per epoch (averaged over processed epochs).
    los_mean = los_sum / float(n_epochs_total)
    nlos_mean = nlos_sum / float(n_epochs_total)
    vis_mean = vis_sum / float(n_epochs_total)
    tblk_mean = tblk_sum / float(n_epochs_total)

    rows: list[dict] = []
    for i, p in enumerate(points):
        rows.append(
            {
                "lat_deg": float(p["lat_deg"]),
                "lon_deg": float(p["lon_deg"]),
                "way_id": int(p.get("way_id", -1)),
                "highway": str(p.get("highway", "")),
                "name": str(p.get("name", "")),
                "mean_n_los": float(los_mean[i]),
                "mean_n_nlos": float(nlos_mean[i]),
                "mean_n_visible": float(vis_mean[i]),
                "mean_n_terrain_blocked": float(tblk_mean[i]),
            }
        )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "lat_deg",
                "lon_deg",
                "way_id",
                "highway",
                "name",
                "mean_n_los",
                "mean_n_nlos",
                "mean_n_visible",
                "mean_n_terrain_blocked",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "lat_deg": f"{r['lat_deg']:.8f}",
                    "lon_deg": f"{r['lon_deg']:.8f}",
                    "way_id": r["way_id"],
                    "highway": r["highway"],
                    "name": r["name"],
                    "mean_n_los": f"{r['mean_n_los']:.4f}",
                    "mean_n_nlos": f"{r['mean_n_nlos']:.4f}",
                    "mean_n_visible": f"{r['mean_n_visible']:.4f}",
                    "mean_n_terrain_blocked": f"{r['mean_n_terrain_blocked']:.4f}",
                }
            )

    _build_html_map(rows, args.out_html, metric_key="mean_n_los")
    dt_total = time.perf_counter() - t0
    print(
        f"[area] done: points={n_points}, epochs={tow_samples.size}, "
        f"csv={args.out_csv}, html={args.out_html}, runtime_s={dt_total:.1f}"
    )


if __name__ == "__main__":
    main()
