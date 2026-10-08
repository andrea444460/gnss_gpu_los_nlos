#!/usr/bin/env python3
"""Build a compact per-way GNSS quality pack (``.rqz``) via real BVH LOS raytracing.

Same inputs as ``build_area_los_nlos_map.py`` (OSM roads, building triangles,
NAV RINEX, optional DEM), but instead of a mean-over-time point CSV this keeps
the **time series** of (HDOP, n_LOS) per OSM way and writes the quantized
change-point pack used by GNSS-aware routing / ``route_gui``.

Example (Genova-style Colab run)::

    PYTHONPATH=python python3 experiments/build_road_quality_rqz.py \\
      --south $SOUTH --west $WEST --north $NORTH --east $EAST \\
      --triangles-npy "$OUT_DIR/genova_osm_triangles.npy" \\
      --nav "$NAV_RINEX" \\
      --out "$OUT_DIR/genova_quality_24h.rqz" \\
      --step-m 30 \\
      --dt-s 300 \\
      --duration-s 86400 \\
      --rx-alt-mode dem \\
      --tile-size-m 12000 \\
      --eph-batch-chunk 32 \\
      --point-batch-chunk 256 \\
      --dem-auto-download \\
      --dem-auto-out "$OUT_DIR/genova_area_dem.tif" \\
      --arterial-only
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_ROOT / "python"))

import numpy as np
import rasterio
from rasterio.warp import Resampling, calculate_default_transform, reproject

from gnss_gpu.bvh import BVHAccelerator
from gnss_gpu.dop import n_los_and_hdop_chunk
from gnss_gpu.ephemeris import Ephemeris
from gnss_gpu.io.dem_download import download_dem_for_bbox
from gnss_gpu.io.nav_rinex import read_nav_rinex_multi
from gnss_gpu.io.osm_roads import BBox, fetch_roads_overpass, sample_road_points, split_bbox_into_tiles
from gnss_gpu.io.road_quality_pack import (
    naive_dense_bytes,
    pack_from_samples,
    read_road_quality_pack,
    write_road_quality_pack,
)
from gnss_gpu.raytrace import BuildingModel
from gnss_gpu.terrain_horizon import HorizonConfig, TerrainHorizonMask
from gnss_gpu.urban_signal_sim import _sat_elevation_azimuth


WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = 2.0 * WGS84_F - WGS84_F * WGS84_F

ARTERIAL = frozenset(
    {
        "motorway",
        "trunk",
        "primary",
        "secondary",
        "tertiary",
        "motorway_link",
        "trunk_link",
        "primary_link",
        "secondary_link",
        "tertiary_link",
    }
)


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


def _load_dem_wgs84_grid(dem_path: Path) -> tuple[np.ndarray, float, float, float, float]:
    with rasterio.open(str(dem_path)) as src:
        dst_crs = "EPSG:4326"
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        dem = np.full((height, width), np.nan, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=dem,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
            dst_nodata=np.nan,
        )
    lat0 = float(transform.f + transform.e * 0.5)
    lon0 = float(transform.c + transform.a * 0.5)
    lat_step = float(transform.e)
    lon_step = float(transform.a)
    return dem, lat0, lon0, lat_step, lon_step


def _sample_dem_wgs84_bilinear(
    dem_h: np.ndarray,
    lat0_deg: float,
    lon0_deg: float,
    lat_step_deg: float,
    lon_step_deg: float,
    lats_deg: np.ndarray,
    lons_deg: np.ndarray,
) -> np.ndarray:
    h = np.asarray(dem_h, dtype=np.float64)
    lats = np.asarray(lats_deg, dtype=np.float64).ravel()
    lons = np.asarray(lons_deg, dtype=np.float64).ravel()
    out = np.full(lats.shape, np.nan, dtype=np.float64)
    if h.ndim != 2 or h.shape[0] < 2 or h.shape[1] < 2:
        return out
    rows, cols = h.shape
    rr = (lats - float(lat0_deg)) / float(lat_step_deg)
    cc = (lons - float(lon0_deg)) / float(lon_step_deg)
    valid = (
        np.isfinite(rr)
        & np.isfinite(cc)
        & (rr >= 0.0)
        & (cc >= 0.0)
        & (rr < (rows - 1))
        & (cc < (cols - 1))
    )
    if not np.any(valid):
        return out
    rv = rr[valid]
    cv = cc[valid]
    r0 = np.floor(rv).astype(np.int64)
    c0 = np.floor(cv).astype(np.int64)
    r1 = r0 + 1
    c1 = c0 + 1
    fr = rv - r0
    fc = cv - c0
    z00 = h[r0, c0]
    z01 = h[r0, c1]
    z10 = h[r1, c0]
    z11 = h[r1, c1]
    finite = np.isfinite(z00) & np.isfinite(z01) & np.isfinite(z10) & np.isfinite(z11)
    if np.any(finite):
        z0 = (1.0 - fc[finite]) * z00[finite] + fc[finite] * z01[finite]
        z1 = (1.0 - fc[finite]) * z10[finite] + fc[finite] * z11[finite]
        zv = (1.0 - fr[finite]) * z0 + fr[finite] * z1
        idx_valid = np.flatnonzero(valid)
        out[idx_valid[finite]] = zv
    return out


def _cpu_preprocess_chunk(
    rx_chunk: np.ndarray,
    sat_b: np.ndarray,
    n_t: int,
    n_sat: int,
    mask_rad: float,
    terrain_mask: TerrainHorizonMask | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (visible, sat_work) with point-major layout [n_p*n_t, n_sat]."""
    n_p = rx_chunk.shape[0]
    sat_flat = np.tile(sat_b, (n_p, 1, 1))
    sat_work = np.array(sat_flat, copy=True)
    visible = np.zeros((n_p * n_t, n_sat), dtype=bool)
    for pi in range(n_p):
        rx = rx_chunk[pi]
        for ti in range(n_t):
            idx = pi * n_t + ti
            sats = sat_b[ti]
            el, _az = _sat_elevation_azimuth(rx, sats)
            vis = el >= mask_rad
            if terrain_mask is not None:
                terr_vis = terrain_mask.terrain_visible_mask(rx, sats)
                vis = np.logical_and(vis, terr_vis)
            visible[idx] = vis
            sat_work[idx][~vis] = np.nan
    return visible, sat_work


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--south", type=float, required=True)
    p.add_argument("--west", type=float, required=True)
    p.add_argument("--north", type=float, required=True)
    p.add_argument("--east", type=float, required=True)
    p.add_argument("--triangles-npy", type=Path, required=True)
    p.add_argument("--nav", type=Path, required=True)
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output .rqz pack path.",
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Optional dense per-way timeline CSV (way_id,t_s,hdop,n_los).",
    )
    p.add_argument("--step-m", type=float, default=30.0, help="Road sampling step [m].")
    p.add_argument("--rx-alt-m", type=float, default=1.5)
    p.add_argument("--rx-alt-mode", type=str, choices=["fixed", "dem"], default="fixed")
    p.add_argument("--rx-ant-height-m", type=float, default=1.5)
    p.add_argument("--include-pedestrian", action="store_true")
    p.add_argument(
        "--arterial-only",
        action="store_true",
        help="Keep only arterial highway classes (recommended for routing packs).",
    )
    p.add_argument("--tile-size-m", type=float, default=0.0)
    p.add_argument("--dt-s", type=float, default=300.0)
    p.add_argument("--duration-s", type=float, default=86400.0)
    p.add_argument("--start-utc", type=str, default="")
    p.add_argument("--tow-start-s", type=float, default=0.0)
    p.add_argument("--elevation-mask-deg", type=float, default=10.0)
    p.add_argument("--nav-systems", type=str, default="G,E,J,C")
    p.add_argument("--eph-batch-chunk", type=int, default=16)
    p.add_argument("--point-batch-chunk", type=int, default=128)
    p.add_argument("--dem-path", type=Path, default=None)
    p.add_argument("--dem-auto-download", action="store_true")
    p.add_argument("--dem-auto-out", type=Path, default=Path("experiments/results/rqz_auto_dem.tif"))
    p.add_argument("--dem-auto-zoom", type=int, default=12)
    p.add_argument("--terrain-max-distance-m", type=float, default=20000.0)
    p.add_argument("--terrain-step-m", type=float, default=60.0)
    p.add_argument("--terrain-azimuth-step-deg", type=float, default=2.0)
    p.add_argument("--terrain-cache-resolution-m", type=float, default=50.0)
    p.add_argument("--terrain-margin-deg", type=float, default=0.0)
    p.add_argument("--hdop-step", type=float, default=0.5)
    p.add_argument("--n-los-step", type=float, default=1.0)
    p.add_argument(
        "--metrics-csv",
        type=Path,
        default=None,
        help="Optional one-row scalability metrics CSV.",
    )
    return p.parse_args()


def _write_metrics_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)


def _write_dense_csv(path: Path, samples_by_way: dict[int, list[tuple[float, float, float]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["way_id", "t_s", "hdop", "n_los"])
        w.writeheader()
        for wid in sorted(samples_by_way):
            for t_s, hdop, n_los in samples_by_way[wid]:
                w.writerow(
                    {
                        "way_id": int(wid),
                        "t_s": f"{float(t_s):.3f}",
                        "hdop": f"{float(hdop):.4f}" if math.isfinite(hdop) else "",
                        "n_los": f"{float(n_los):.4f}" if math.isfinite(n_los) else "",
                    }
                )


def main() -> None:
    args = _parse_args()
    t0 = time.perf_counter()
    bbox = BBox(args.south, args.west, args.north, args.east)

    # --- roads ----------------------------------------------------------------
    tile_boxes = split_bbox_into_tiles(bbox, tile_size_m=float(args.tile_size_m))
    all_points: list[dict] = []
    for ti, tb in enumerate(tile_boxes, start=1):
        roads = fetch_roads_overpass(tb, include_pedestrian=bool(args.include_pedestrian))
        if args.arterial_only:
            roads = [
                w
                for w in roads
                if str((w.get("tags") or {}).get("highway", "")).lower() in ARTERIAL
            ]
        pts = sample_road_points(roads, step_m=float(args.step_m))
        all_points.extend(pts)
        print(
            f"[rqz][roads] tile {ti}/{len(tile_boxes)}: roads={len(roads)} "
            f"points={len(pts)} cumulative={len(all_points)}",
            flush=True,
        )
    uniq: dict[tuple[int, int], dict] = {}
    for p in all_points:
        k = (int(round(float(p["lat_deg"]) * 1e6)), int(round(float(p["lon_deg"]) * 1e6)))
        uniq[k] = p
    points = list(uniq.values())
    if not points:
        raise RuntimeError("No road points sampled in the selected area.")
    print(f"[rqz] sampled road points: {len(points)}", flush=True)

    way_to_idxs: dict[int, list[int]] = defaultdict(list)
    for i, p in enumerate(points):
        way_to_idxs[int(p.get("way_id", -1))].append(i)
    way_ids = sorted(wid for wid in way_to_idxs if wid >= 0)
    if not way_ids:
        raise RuntimeError("Sampled points have no valid way_id.")
    print(f"[rqz] ways with samples: {len(way_ids)}", flush=True)

    # --- mesh / BVH -----------------------------------------------------------
    tri = np.asarray(np.load(args.triangles_npy), dtype=np.float64)
    if tri.ndim != 3 or tri.shape[1:] != (3, 3):
        raise ValueError(f"--triangles-npy must have shape [N,3,3], got {tri.shape}")
    building = BuildingModel(tri)
    bvh = BVHAccelerator.from_building_model(building)
    if not hasattr(bvh, "check_los_batch"):
        raise RuntimeError("BVH check_los_batch unavailable; rebuild gnss_gpu with CUDA BVH support.")
    print(f"[rqz] mesh triangles={len(tri)}, bvh_nodes={bvh.n_nodes}", flush=True)

    # --- ephemeris / time grid ------------------------------------------------
    systems = tuple(s.strip() for s in str(args.nav_systems).split(",") if s.strip())
    nav_messages = read_nav_rinex_multi(str(args.nav.resolve()), systems=systems)
    eph = Ephemeris(nav_messages)
    prn_catalog = eph.available_prns
    print(f"[rqz] NAV PRNs available: {len(prn_catalog)} ({','.join(systems)})", flush=True)

    if args.start_utc.strip():
        dt = datetime.fromisoformat(args.start_utc.replace("Z", "+00:00"))
        _week, tow_start = _gps_week_tow_from_utc(dt)
    else:
        tow_start = float(args.tow_start_s)
    tow_samples = _time_samples(tow_start, float(args.duration_s), float(args.dt_s))
    # Relative times for the pack (t=0 at simulation start).
    t_rel = np.arange(tow_samples.size, dtype=np.float64) * float(args.dt_s)
    print(
        f"[rqz] time samples: {tow_samples.size} (dt={args.dt_s:g}s, duration={args.duration_s:g}s)",
        flush=True,
    )

    # --- DEM / terrain --------------------------------------------------------
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
        print(f"[rqz] DEM auto-downloaded: {dem_path} ({ntiles} tiles)", flush=True)

    terrain_mask: TerrainHorizonMask | None = None
    dem_grid: np.ndarray | None = None
    dem_meta: tuple[float, float, float, float] | None = None
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
        dem_grid, lat0, lon0, lat_step, lon_step = _load_dem_wgs84_grid(dem_path)
        dem_meta = (lat0, lon0, lat_step, lon_step)
        print(f"[rqz] terrain prefilter: enabled ({dem_path})", flush=True)
    else:
        print("[rqz] terrain prefilter: disabled", flush=True)

    point_lats = np.asarray([float(p["lat_deg"]) for p in points], dtype=np.float64)
    point_lons = np.asarray([float(p["lon_deg"]) for p in points], dtype=np.float64)
    if args.rx_alt_mode == "dem":
        if dem_grid is None or dem_meta is None:
            raise RuntimeError("--rx-alt-mode dem requires DEM (--dem-path or --dem-auto-download).")
        lat0, lon0, lat_step, lon_step = dem_meta
        dem_z = _sample_dem_wgs84_bilinear(
            dem_grid, lat0, lon0, lat_step, lon_step, point_lats, point_lons
        )
        n_valid = int(np.sum(np.isfinite(dem_z)))
        if n_valid <= 0:
            raise RuntimeError("DEM altitude sampling failed for all road points.")
        if n_valid < dem_z.size:
            fallback = float(np.nanmedian(dem_z[np.isfinite(dem_z)]))
            dem_z = np.where(np.isfinite(dem_z), dem_z, fallback)
        rx_alts = dem_z + float(args.rx_ant_height_m)
        print(
            f"[rqz] rx-alt mode=dem ant={args.rx_ant_height_m:.2f}m "
            f"range=[{float(np.nanmin(rx_alts)):.1f},{float(np.nanmax(rx_alts)):.1f}]",
            flush=True,
        )
    else:
        rx_alts = np.full((len(points),), float(args.rx_alt_m), dtype=np.float64)
        print(f"[rqz] rx-alt mode=fixed alt={float(args.rx_alt_m):.2f}m", flush=True)

    point_ecef = np.asarray(
        [_lla_deg_to_ecef(point_lats[i], point_lons[i], float(rx_alts[i])) for i in range(len(points))],
        dtype=np.float64,
    )

    n_points = len(points)
    n_epochs = int(tow_samples.size)
    # Accumulators for way-mean quality per epoch.
    way_hdop_sum = {wid: np.zeros(n_epochs, dtype=np.float64) for wid in way_ids}
    way_nlos_sum = {wid: np.zeros(n_epochs, dtype=np.float64) for wid in way_ids}
    way_hdop_cnt = {wid: np.zeros(n_epochs, dtype=np.int32) for wid in way_ids}
    way_nlos_cnt = {wid: np.zeros(n_epochs, dtype=np.int32) for wid in way_ids}

    p_chunk = max(1, int(args.point_batch_chunk))
    e_chunk = max(1, int(args.eph_batch_chunk))
    mask_rad = math.radians(float(args.elevation_mask_deg))
    n_epochs_processed = 0
    m_rays = 0
    t_ray = 0.0
    t_last_log = t0

    for es in range(0, n_epochs, e_chunk):
        ee = min(es + e_chunk, n_epochs)
        tow_blk = np.asarray(tow_samples[es:ee], dtype=np.float64)
        sat_b, _clk_b, _used = eph.compute_batch(tow_blk, prn_list=prn_catalog)
        if sat_b.shape[1] == 0:
            continue
        n_t, n_sat = sat_b.shape[0], sat_b.shape[1]
        n_epochs_processed += n_t
        epoch_chunk_idx = es // e_chunk + 1
        epoch_chunk_total = int(math.ceil(n_epochs / e_chunk))
        print(
            f"[rqz][epochs] chunk {epoch_chunk_idx}/{epoch_chunk_total}: epochs={n_t} sats={n_sat}",
            flush=True,
        )

        for ps in range(0, n_points, p_chunk):
            pe = min(ps + p_chunk, n_points)
            rx_chunk = point_ecef[ps:pe]
            n_p = rx_chunk.shape[0]
            rx_flat = np.repeat(rx_chunk, n_t, axis=0)

            visible, sat_work = _cpu_preprocess_chunk(
                rx_chunk, sat_b, n_t, n_sat, mask_rad, terrain_mask
            )
            t_cr = time.perf_counter()
            los = np.asarray(bvh.check_los_batch(rx_flat, sat_work), dtype=bool)
            t_ray += time.perf_counter() - t_cr
            m_rays += int(n_p) * int(n_t) * int(n_sat)

            los_vis = np.logical_and(los, visible).reshape(n_p, n_t, n_sat)
            n_los_pt, hdop_pt = n_los_and_hdop_chunk(rx_chunk, sat_b, los_vis)

            # Scatter point stats into way accumulators for epochs [es:ee].
            for local_i, global_i in enumerate(range(ps, pe)):
                wid = int(points[global_i].get("way_id", -1))
                if wid not in way_hdop_sum:
                    continue
                for ti in range(n_t):
                    ei = es + ti
                    n_val = float(n_los_pt[local_i, ti])
                    h_val = float(hdop_pt[local_i, ti])
                    way_nlos_sum[wid][ei] += n_val
                    way_nlos_cnt[wid][ei] += 1
                    if math.isfinite(h_val):
                        way_hdop_sum[wid][ei] += h_val
                        way_hdop_cnt[wid][ei] += 1

            now = time.perf_counter()
            if (now - t_last_log) >= 5.0 or pe == n_points:
                frac_points = float(pe) / max(1.0, float(n_points))
                progress = min(
                    1.0,
                    (epoch_chunk_idx - 1 + frac_points) / max(1.0, float(epoch_chunk_total)),
                )
                elapsed = max(1e-9, now - t0)
                eta_s = elapsed * (1.0 - progress) / max(1e-9, progress)
                print(
                    f"[rqz][progress] {progress*100:5.1f}% | "
                    f"epoch_chunk={epoch_chunk_idx}/{epoch_chunk_total} "
                    f"points={pe}/{n_points} | elapsed={elapsed:7.1f}s eta={eta_s:7.1f}s",
                    flush=True,
                )
                t_last_log = now

    if n_epochs_processed <= 0:
        raise RuntimeError("No valid epochs processed.")

    samples_by_way: dict[int, list[tuple[float, float, float]]] = {}
    for wid in way_ids:
        seq: list[tuple[float, float, float]] = []
        for ei in range(n_epochs):
            if way_nlos_cnt[wid][ei] <= 0:
                continue
            n_mean = float(way_nlos_sum[wid][ei] / way_nlos_cnt[wid][ei])
            if way_hdop_cnt[wid][ei] > 0:
                h_mean = float(way_hdop_sum[wid][ei] / way_hdop_cnt[wid][ei])
            else:
                h_mean = float("nan")
            seq.append((float(t_rel[ei]), h_mean, n_mean))
        if seq:
            samples_by_way[wid] = seq

    pack = pack_from_samples(
        samples_by_way,
        bbox=[bbox.south, bbox.west, bbox.north, bbox.east],
        t0_s=0.0,
        horizon_s=float(args.duration_s),
        dt_s=float(args.dt_s),
        hdop_step=float(args.hdop_step),
        n_los_step=float(args.n_los_step),
        source="bvh_los_raycast_v1",
        meta={
            "triangles_npy": str(args.triangles_npy),
            "nav": str(args.nav),
            "step_m": float(args.step_m),
            "arterial_only": bool(args.arterial_only),
            "include_pedestrian": bool(args.include_pedestrian),
            "elevation_mask_deg": float(args.elevation_mask_deg),
            "rx_alt_mode": str(args.rx_alt_mode),
            "n_points": int(n_points),
            "n_epochs": int(n_epochs),
            "n_epochs_processed": int(n_epochs_processed),
            "start_utc": str(args.start_utc),
            "tow_start_s": float(tow_start),
        },
    )
    out = write_road_quality_pack(args.out, pack)
    rt = read_road_quality_pack(out)
    size = out.stat().st_size
    dense = naive_dense_bytes(len(samples_by_way), n_epochs)

    if args.out_csv is not None:
        _write_dense_csv(args.out_csv, samples_by_way)
        print(f"[rqz] dense csv: {args.out_csv}", flush=True)

    dt_total = time.perf_counter() - t0
    print(
        f"[rqz] wrote {out} ({size/1024:.1f} KB) ways={pack.n_ways} runs={pack.n_runs} "
        f"avg_runs/way={pack.n_runs/max(1,pack.n_ways):.2f} "
        f"vs_dense≈{dense/1024:.0f} KB  roundtrip_ways={rt.n_ways}",
        flush=True,
    )
    print(
        f"[rqz] done runtime_s={dt_total:.1f} raytrace_s={t_ray:.1f} "
        f"rays={m_rays} rays/s={m_rays/max(t_ray,1e-12):,.0f}",
        flush=True,
    )

    if args.metrics_csv is not None:
        _write_metrics_csv(
            args.metrics_csv,
            {
                "run_id": out.stem,
                "n_points": n_points,
                "n_ways": pack.n_ways,
                "n_runs": pack.n_runs,
                "n_epochs": n_epochs,
                "n_epochs_processed": n_epochs_processed,
                "n_triangles": int(len(tri)),
                "n_bvh_nodes": int(bvh.n_nodes),
                "n_rays": m_rays,
                "time_wall_s": f"{dt_total:.6f}",
                "time_raytrace_los_s": f"{t_ray:.6f}",
                "out_rqz_bytes": size,
                "dense_equiv_bytes": dense,
                "triangles_npy": str(args.triangles_npy),
                "nav": str(args.nav),
            },
        )
        print(f"[rqz] metrics: {args.metrics_csv}", flush=True)


if __name__ == "__main__":
    main()
