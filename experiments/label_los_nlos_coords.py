#!/usr/bin/env python3
"""Label LOS/NLOS satellites for arbitrary receiver coordinates.

Given a CSV/JSON list of lat/lon/(alt) points, a RINEX NAV file, and a
building mesh (``triangles.npy`` ECEF ``[N,3,3]``), places a ground
receiver at each point and classifies every NAV satellite above the
elevation mask as LOS or NLOS via BVH ray tracing.

Optional DEM (recommended for Genova / hilly sites)::

  - ``--rx-alt-mode dem``: antenna = DEM elevation + ``--rx-ant-height-m``
  - ``--dem-path`` / ``--dem-auto-download``: terrain horizon prefilter
  - final ``is_los`` = buildings AND elev mask AND not terrain-blocked
  - ``is_los_buildings`` ignores terrain (ablation)

Reuses the same stack as ``export_los_labels_from_urbannav.py`` /
``build_area_los_nlos_map.py``:

  - ``Ephemeris.compute`` for satellite ECEF
  - ``_sat_elevation_azimuth`` + elevation mask
  - ``BVHAccelerator.check_los`` for building occlusion
  - ``TerrainHorizonMask`` + ``NativeDemSampler`` for DEM

Genova notes (from area-map / rqz pipeline)::

  - Prefer ``--rx-alt-mode dem --dem-auto-download`` (fixed alt=1.5 m
    ellipsoidal is wrong on hills).
  - ``dem_download`` AWS elevation tiles are **orthometric (EGM96-like)**.
    For Genova / that auto-DEM use ``--geoid-correction egm96`` on **both**
    labeling and OSM mesh fetch — otherwise RX+buildings sit ~45–50 m low
    in ECEF and LOS/terrain collisions are wrong (not only the Cesium view).
  - When building OSM mesh, pass the same DEM + same geoid flag to
    ``fetch_osm_buildings_bbox.py --dem-path --geoid-correction egm96``.
  - Terrain horizon reaches ``--terrain-max-distance-m`` (default 20 km);
    DEM auto bbox buffer should cover that.
  - Typical city bbox used historically:
    south=44.385 west=8.880 north=44.445 east=8.980

CSV input columns (case-insensitive aliases)::

  id (optional), lat|latitude, lon|longitude, alt|height (optional)
  optional per-row: gps_week, gps_tow  OR  utc|time|timestamp (ISO-8601)

CLI example (Genova-style)::

  python experiments/label_los_nlos_coords.py \\
    --coords points.csv \\
    --nav BRDC....rnx \\
    --triangles genova_osm_triangles.npy \\
    --gps-week 2320 --gps-tow 360000 \\
    --rx-alt-mode dem --rx-ant-height-m 1.5 \\
    --dem-auto-download --dem-auto-out genova_dem.tif \\
    --output-csv los_labels.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "python"))

from gnss_gpu.bvh import BVHAccelerator
from gnss_gpu.ephemeris import Ephemeris
from gnss_gpu.io.dem_download import download_dem_for_bbox
from gnss_gpu.io.dem_native import NativeDemSampler, parse_geoid_correction_arg
from gnss_gpu.io.nav_rinex import read_nav_rinex_multi
from gnss_gpu.raytrace import BuildingModel
from gnss_gpu.terrain_horizon import HorizonConfig, TerrainHorizonMask
from gnss_gpu.urban_signal_sim import _sat_elevation_azimuth

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = 2.0 * WGS84_F - WGS84_F * WGS84_F
GPS_EPOCH = datetime(1980, 1, 6, tzinfo=timezone.utc)
GPS_WEEK_SECONDS = 604800.0
LEAP_SECONDS = 18.0  # modern datasets


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


def _utc_to_gps_week_tow(dt: datetime) -> tuple[int, float]:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    gps_seconds = (dt - GPS_EPOCH).total_seconds() + LEAP_SECONDS
    week = int(gps_seconds // GPS_WEEK_SECONDS)
    tow = float(gps_seconds - week * GPS_WEEK_SECONDS)
    return week, tow


def _parse_utc(value: str) -> datetime:
    s = str(value).strip().replace("Z", "+00:00")
    return datetime.fromisoformat(s)


def _norm_key(k: str) -> str:
    return "".join(ch for ch in str(k).strip().lower() if ch.isalnum() or ch == "_")


_LAT_KEYS = ("lat", "latitude", "lat_deg", "latdeg")
_LON_KEYS = ("lon", "lng", "long", "longitude", "lon_deg", "londeg")
_ALT_KEYS = ("alt", "altitude", "height", "h", "ellipsoidal_height_m", "alt_m", "height_m")
_ID_KEYS = ("id", "name", "point_id", "label", "pid")
_WEEK_KEYS = ("gps_week", "week", "gpsweek")
_TOW_KEYS = ("gps_tow", "tow", "gpstow", "sow")
_UTC_KEYS = ("utc", "time", "timestamp", "datetime", "epoch")


def _pick(row: dict, keys: tuple[str, ...], default=None):
    for k in keys:
        if k in row and row[k] not in (None, ""):
            return row[k]
    return default


def _normalize_row(raw: dict) -> dict:
    return {_norm_key(k): v for k, v in raw.items()}


def load_coords(path: Path, default_alt_m: float) -> list[dict]:
    """Load receiver points from CSV or JSON."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    points: list[dict] = []
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            rows = payload.get("points") or payload.get("coords") or payload.get("coordinates")
            if rows is None:
                raise ValueError("JSON object must contain 'points' (or 'coords') array")
            global_week = payload.get("gps_week")
            global_tow = payload.get("gps_tow")
            global_utc = payload.get("utc") or payload.get("time")
        elif isinstance(payload, list):
            rows = payload
            global_week = global_tow = global_utc = None
        else:
            raise ValueError("JSON must be a list of points or an object with 'points'")
        for i, raw in enumerate(rows):
            row = _normalize_row(raw)
            lat = _pick(row, _LAT_KEYS)
            lon = _pick(row, _LON_KEYS)
            if lat is None or lon is None:
                raise ValueError(f"Point {i}: missing lat/lon")
            alt = _pick(row, _ALT_KEYS, default_alt_m)
            pid = _pick(row, _ID_KEYS, f"p{i}")
            week = _pick(row, _WEEK_KEYS, global_week)
            tow = _pick(row, _TOW_KEYS, global_tow)
            utc = _pick(row, _UTC_KEYS, global_utc)
            points.append(
                {
                    "id": str(pid),
                    "lat": float(lat),
                    "lon": float(lon),
                    "alt": float(alt),
                    "gps_week": None if week is None else int(week),
                    "gps_tow": None if tow is None else float(tow),
                    "utc": None if utc is None else str(utc),
                }
            )
        return points

    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header")
        for i, raw in enumerate(reader):
            row = _normalize_row(raw)
            lat = _pick(row, _LAT_KEYS)
            lon = _pick(row, _LON_KEYS)
            if lat is None or lon is None:
                raise ValueError(f"CSV row {i+2}: missing lat/lon")
            alt = _pick(row, _ALT_KEYS, default_alt_m)
            pid = _pick(row, _ID_KEYS, f"p{i}")
            points.append(
                {
                    "id": str(pid),
                    "lat": float(lat),
                    "lon": float(lon),
                    "alt": float(alt),
                    "gps_week": None if _pick(row, _WEEK_KEYS) is None else int(_pick(row, _WEEK_KEYS)),
                    "gps_tow": None if _pick(row, _TOW_KEYS) is None else float(_pick(row, _TOW_KEYS)),
                    "utc": None if _pick(row, _UTC_KEYS) is None else str(_pick(row, _UTC_KEYS)),
                }
            )
    if not points:
        raise ValueError(f"No points loaded from {path}")
    return points


def _resolve_time(
    point: dict,
    cli_week: int | None,
    cli_tow: float | None,
    cli_utc: str | None,
) -> tuple[int, float]:
    if point.get("gps_week") is not None and point.get("gps_tow") is not None:
        return int(point["gps_week"]), float(point["gps_tow"])
    if point.get("utc"):
        return _utc_to_gps_week_tow(_parse_utc(point["utc"]))
    if cli_week is not None and cli_tow is not None:
        return int(cli_week), float(cli_tow)
    if cli_utc:
        return _utc_to_gps_week_tow(_parse_utc(cli_utc))
    raise ValueError(
        f"Point {point['id']}: no time. Pass --gps-week/--gps-tow or --utc, "
        "or put gps_week/gps_tow/utc on each row."
    )


def _points_bbox_deg(points: list[dict], buffer_m: float) -> tuple[float, float, float, float]:
    lats = np.asarray([p["lat"] for p in points], dtype=np.float64)
    lons = np.asarray([p["lon"] for p in points], dtype=np.float64)
    lat0 = float(np.mean(lats))
    dlat = float(buffer_m) / 111_320.0
    dlon = float(buffer_m) / (111_320.0 * max(1e-6, math.cos(math.radians(lat0))))
    return (
        float(np.min(lats) - dlat),
        float(np.min(lons) - dlon),
        float(np.max(lats) + dlat),
        float(np.max(lons) + dlon),
    )


def _sat_row(sid_raw, el, az, is_visible: bool, is_los_buildings: bool, terrain_blocked: bool) -> dict:
    sid = str(sid_raw).strip().upper()
    system = sid[0] if sid and sid[0].isalpha() else "G"
    prn_num = sid[1:] if len(sid) > 1 and sid[0].isalpha() else sid
    above = bool(is_visible)
    terr = bool(terrain_blocked)
    los_b = bool(is_los_buildings) and above
    # Same semantics as export_los_labels_from_urbannav.py
    is_los = int(los_b and not terr)
    return {
        "sat_id": sid,
        "system": system,
        "prn": prn_num,
        "is_visible": int(above and not terr),
        "is_los": is_los,
        "is_los_buildings": int(los_b),
        "terrain_blocked": int(terr),
        "terrain_blocked_visible": int(above and terr),
        "elevation_deg": float(math.degrees(el)),
        "azimuth_deg": float(math.degrees(az)),
    }


def classify_point(
    rx_ecef: np.ndarray,
    tow: float,
    eph: Ephemeris,
    bvh: BVHAccelerator,
    prn_catalog: list[str],
    mask_rad: float,
    terrain_mask: TerrainHorizonMask | None,
    include_below_mask: bool,
) -> list[dict]:
    """Per-satellite rows for one receiver / epoch."""
    sat_ecef, _clk, used = eph.compute(float(tow), prn_list=prn_catalog)
    if len(used) == 0:
        return []
    sats = np.asarray(sat_ecef, dtype=np.float64).reshape(-1, 3)
    el, az = _sat_elevation_azimuth(rx_ecef, sats)
    above_mask = el >= mask_rad

    terrain_blocked = np.zeros(len(used), dtype=bool)
    if terrain_mask is not None:
        terrain_visible = terrain_mask.terrain_visible_mask(rx_ecef, sats)
        terrain_blocked = ~np.asarray(terrain_visible, dtype=bool)

    los_buildings = np.zeros(len(used), dtype=bool)
    vis_idx = np.where(above_mask)[0]
    if len(vis_idx):
        los_buildings[vis_idx] = np.asarray(bvh.check_los(rx_ecef, sats[vis_idx]), dtype=bool)

    rows: list[dict] = []
    for i, sid_raw in enumerate(used):
        if not include_below_mask and not bool(above_mask[i]):
            continue
        rows.append(
            _sat_row(
                sid_raw,
                el[i],
                az[i],
                is_visible=bool(above_mask[i]),
                is_los_buildings=bool(los_buildings[i]),
                terrain_blocked=bool(terrain_blocked[i]),
            )
        )
    return rows


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LOS/NLOS labels for CSV/JSON receiver coordinates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--coords", type=Path, required=True, help="CSV or JSON with lat/lon/(alt)")
    p.add_argument("--nav", type=Path, required=True, help="RINEX NAV (mixed OK)")
    p.add_argument("--triangles", type=Path, required=True, help="Mesh triangles.npy [N,3,3] ECEF")
    p.add_argument("--output-csv", type=Path, required=True, help="Per-sat labels CSV")
    p.add_argument("--summary-csv", type=Path, default=None, help="Optional per-point summary CSV")
    p.add_argument("--gps-week", type=int, default=None, help="Default GPS week for all points")
    p.add_argument("--gps-tow", type=float, default=None, help="Default GPS TOW [s] for all points")
    p.add_argument("--utc", type=str, default=None, help="Default UTC ISO-8601 (alternative to week/tow)")
    p.add_argument(
        "--rx-alt-m",
        type=float,
        default=1.5,
        help="Ellipsoidal altitude [m] when --rx-alt-mode fixed (also CSV default alt).",
    )
    p.add_argument(
        "--rx-alt-mode",
        choices=("fixed", "dem"),
        default="fixed",
        help="fixed=CSV/cli alt; dem=DEM sample + --rx-ant-height-m (Genova: use dem).",
    )
    p.add_argument(
        "--rx-ant-height-m",
        type=float,
        default=1.5,
        help="Antenna height above DEM terrain [m] when --rx-alt-mode dem.",
    )
    p.add_argument("--elevation-mask-deg", type=float, default=10.0)
    p.add_argument(
        "--systems",
        type=str,
        default="G,E,J,C",
        help="Comma-separated GNSS systems to load from NAV (default G,E,J,C)",
    )
    p.add_argument(
        "--include-below-mask",
        action="store_true",
        help="Also write rows for sats below elevation mask",
    )
    p.add_argument("--dem-path", type=Path, default=None, help="DEM GeoTIFF for rx-alt and/or terrain")
    p.add_argument(
        "--geoid-correction",
        default="none",
        help="Orthometric DEM only: egm96 or constant N [m]. Satellite tiles: none (default).",
    )
    p.add_argument("--dem-auto-download", action="store_true")
    p.add_argument(
        "--dem-auto-out",
        type=Path,
        default=Path("experiments/results/coords_auto_dem.tif"),
    )
    p.add_argument("--dem-auto-zoom", type=int, default=12)
    p.add_argument(
        "--dem-auto-buffer-m",
        type=float,
        default=25_000.0,
        help="BBox padding around points for DEM download (cover terrain horizon).",
    )
    p.add_argument("--terrain-max-distance-m", type=float, default=20_000.0)
    p.add_argument("--terrain-step-m", type=float, default=60.0)
    p.add_argument("--terrain-azimuth-step-deg", type=float, default=2.0)
    p.add_argument("--terrain-cache-resolution-m", type=float, default=50.0)
    p.add_argument("--terrain-margin-deg", type=float, default=0.0)
    p.add_argument(
        "--no-terrain-horizon",
        action="store_true",
        help="Use DEM only for rx altitude; skip terrain horizon prefilter.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    systems = tuple(s.strip().upper() for s in args.systems.split(",") if s.strip())

    points = load_coords(args.coords, default_alt_m=float(args.rx_alt_m))
    print(f"Loaded {len(points)} point(s) from {args.coords}")

    dem_path = args.dem_path
    if dem_path is None and args.dem_auto_download:
        south, west, north, east = _points_bbox_deg(points, float(args.dem_auto_buffer_m))
        dem_path, n_tiles = download_dem_for_bbox(
            south=south,
            west=west,
            north=north,
            east=east,
            output_tif=args.dem_auto_out,
            zoom=int(args.dem_auto_zoom),
        )
        print(f"DEM auto-download: {dem_path} ({n_tiles} tiles) bbox=[{south:.4f},{west:.4f},{north:.4f},{east:.4f}]")

    dem_sampler: NativeDemSampler | None = None
    terrain_mask: TerrainHorizonMask | None = None
    if dem_path is not None:
        geoid = parse_geoid_correction_arg(str(args.geoid_correction))
        dem_sampler = NativeDemSampler(dem_path, geoid_correction=geoid)
        print(
            f"DEM sampler: {dem_path} geoid={dem_sampler.geoid_label} "
            f"elev={dem_sampler.min_m:.1f}..{dem_sampler.max_m:.1f} m"
        )
        if not args.no_terrain_horizon:
            cfg = HorizonConfig(
                max_distance_m=float(args.terrain_max_distance_m),
                sample_step_m=float(args.terrain_step_m),
                azimuth_step_deg=float(args.terrain_azimuth_step_deg),
                cache_resolution_m=float(args.terrain_cache_resolution_m),
                margin_deg=float(args.terrain_margin_deg),
            )
            terrain_mask = TerrainHorizonMask(dem_path, cfg)
            print(
                f"Terrain horizon: on (max_dist={cfg.max_distance_m:g}m, "
                f"step={cfg.sample_step_m:g}m, az_step={cfg.azimuth_step_deg:g}deg)"
            )
        else:
            print("Terrain horizon: off (--no-terrain-horizon)")
    else:
        print("DEM: none (buildings only). For Genova use --dem-auto-download + --rx-alt-mode dem.")

    if args.rx_alt_mode == "dem" and dem_sampler is None:
        raise RuntimeError("--rx-alt-mode dem requires --dem-path or --dem-auto-download")

    tri = np.load(args.triangles)
    if tri.ndim != 3 or tri.shape[1:] != (3, 3):
        raise ValueError(f"Expected triangles shape [N,3,3], got {tri.shape}")
    bvh = BVHAccelerator(BuildingModel(tri).triangles)
    print(f"Mesh: {tri.shape[0]} triangles → BVH OK")

    nav_messages = read_nav_rinex_multi(args.nav, systems=systems)
    eph = Ephemeris(nav_messages)
    prn_catalog = [str(p).strip().upper() for p in eph.available_prns]
    if not prn_catalog:
        raise RuntimeError(f"No PRNs in NAV {args.nav} for systems {systems}")
    print(f"NAV: {len(prn_catalog)} PRNs from {args.nav.name}")

    mask_rad = math.radians(float(args.elevation_mask_deg))
    label_rows: list[dict] = []
    summary_rows: list[dict] = []

    for pt in points:
        week, tow = _resolve_time(pt, args.gps_week, args.gps_tow, args.utc)
        if args.rx_alt_mode == "dem":
            assert dem_sampler is not None
            dem_z = float(dem_sampler.sample(pt["lat"], pt["lon"]))
            alt = dem_z + float(args.rx_ant_height_m)
        else:
            alt = float(pt["alt"])
        rx = _lla_deg_to_ecef(pt["lat"], pt["lon"], alt)

        sat_rows = classify_point(
            rx,
            tow,
            eph,
            bvh,
            prn_catalog,
            mask_rad,
            terrain_mask,
            include_below_mask=bool(args.include_below_mask),
        )

        n_vis = sum(r["is_visible"] for r in sat_rows)
        n_los = sum(r["is_los"] for r in sat_rows)
        n_nlos = n_vis - n_los
        n_tbv = sum(r["terrain_blocked_visible"] for r in sat_rows)
        print(
            f"  {pt['id']}: lat={pt['lat']:.6f} lon={pt['lon']:.6f} alt={alt:.2f} "
            f"week={week} tow={tow:.1f} → vis={n_vis} LOS={n_los} NLOS={n_nlos} terrain_vis_block={n_tbv}"
        )

        for r in sat_rows:
            label_rows.append(
                {
                    "point_id": pt["id"],
                    "lat_deg": pt["lat"],
                    "lon_deg": pt["lon"],
                    "alt_m": alt,
                    "gps_week": week,
                    "gps_tow": tow,
                    "rx_x_m": float(rx[0]),
                    "rx_y_m": float(rx[1]),
                    "rx_z_m": float(rx[2]),
                    **r,
                }
            )
        summary_rows.append(
            {
                "point_id": pt["id"],
                "lat_deg": pt["lat"],
                "lon_deg": pt["lon"],
                "alt_m": alt,
                "gps_week": week,
                "gps_tow": tow,
                "n_visible": n_vis,
                "n_los": n_los,
                "n_nlos": n_nlos,
                "n_terrain_blocked_visible": n_tbv,
            }
        )

    fields = [
        "point_id",
        "lat_deg",
        "lon_deg",
        "alt_m",
        "gps_week",
        "gps_tow",
        "sat_id",
        "system",
        "prn",
        "is_visible",
        "is_los",
        "is_los_buildings",
        "terrain_blocked",
        "terrain_blocked_visible",
        "elevation_deg",
        "azimuth_deg",
        "rx_x_m",
        "rx_y_m",
        "rx_z_m",
    ]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for row in label_rows:
            w.writerow(row)
    print(f"Wrote {len(label_rows)} sat rows → {args.output_csv}")

    summary_path = args.summary_csv
    if summary_path is None:
        summary_path = args.output_csv.with_name(args.output_csv.stem + "_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "point_id",
                "lat_deg",
                "lon_deg",
                "alt_m",
                "gps_week",
                "gps_tow",
                "n_visible",
                "n_los",
                "n_nlos",
                "n_terrain_blocked_visible",
            ],
        )
        w.writeheader()
        for row in summary_rows:
            w.writerow(row)
    print(f"Wrote summary → {summary_path}")

    if dem_sampler is not None:
        dem_sampler.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
