#!/usr/bin/env python3
"""Daily-mean LOS/NLOS counts and LOS-only DOP on a pitch grid (stadium use case).

Places an equidistant ``rows x cols`` grid on the football pitch, then for each point
averages over a full GPS day (broadcast ephemeris):

  - mean number of LOS satellites (building mesh + elevation mask)
  - mean number of NLOS satellites (visible but blocked)
  - mean PDOP/HDOP/VDOP/GDOP using **LOS-only** geometry

Example (Luigi Ferraris)::

    python experiments/stadium_pitch_grid_daily_metrics.py \\
        --trajectory-csv experiments/data/LuigiFerraris/luigiFerrarisTrajectory.csv \\
        --triangles-npy experiments/data/LuigiFerraris/luigiFerrarisGeoLoc/luigi_ferraris_triangles.npy \\
        --nav path/to/BRDC00IGS_R_20240890000_01D_MN.rnx \\
        --start-utc 2024-03-28T12:00:00Z \\
        --grid-rows 3 --grid-cols 5 \\
        --dt-s 60 \\
        --out-csv experiments/results/luigi_pitch_grid_daily.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "python") not in sys.path:
    sys.path.insert(0, str(_REPO / "python"))
if str(_REPO / "experiments") not in sys.path:
    sys.path.insert(0, str(_REPO / "experiments"))

import numpy as np

from gnss_gpu.bvh import BVHAccelerator
from gnss_gpu.ephemeris import Ephemeris
from gnss_gpu.io.nav_rinex import read_nav_rinex_multi
from gnss_gpu.raytrace import BuildingModel
from gnss_gpu.urban_signal_sim import _sat_elevation_azimuth, ecef_to_lla

from build_stadium_visibility_viz import _gps_week_tow_from_utc, read_lat_lon_alt_csv


WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = 2.0 * WGS84_F - WGS84_F ** 2


def _lla_to_ecef(lat_deg: float, lon_deg: float, alt_m: float) -> np.ndarray:
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


def _enu_basis_at_ecef(rx_ecef: np.ndarray) -> np.ndarray:
    """Return 3x3 matrix whose columns are E, N, U unit vectors in ECEF."""
    lat, lon, _ = ecef_to_lla(float(rx_ecef[0]), float(rx_ecef[1]), float(rx_ecef[2]))
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    sin_lon, cos_lon = math.sin(lon), math.cos(lon)
    e = np.array([-sin_lon, cos_lon, 0.0], dtype=np.float64)
    n = np.array([-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat], dtype=np.float64)
    u = np.array([cos_lat * cos_lon, cos_lat * sin_lon, sin_lat], dtype=np.float64)
    return np.column_stack([e, n, u])


def _pitch_axes_from_trajectory(lat_lon_alt: list[tuple[float, float, float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (center_ecef, long_axis_ecef, short_axis_ecef) from perimeter samples."""
    lats = [p[0] for p in lat_lon_alt]
    lons = [p[1] for p in lat_lon_alt]
    hs = [p[2] for p in lat_lon_alt]
    lat0 = float(np.mean(lats))
    lon0 = float(np.mean(lons))
    h0 = float(np.mean(hs))
    center = _lla_to_ecef(lat0, lon0, h0)
    basis = _enu_basis_at_ecef(center)
    en = []
    for lat, lon, h in lat_lon_alt:
        d = _lla_to_ecef(lat, lon, h) - center
        en.append([float(basis[:, 0] @ d), float(basis[:, 1] @ d)])
    en = np.asarray(en, dtype=np.float64)
    cov = np.cov(en.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    long_en = eigvecs[:, int(np.argmax(eigvals))]
    short_en = eigvecs[:, int(np.argmin(eigvals))]
    long_axis = basis[:, 0] * long_en[0] + basis[:, 1] * long_en[1]
    short_axis = basis[:, 0] * short_en[0] + basis[:, 1] * short_en[1]
    long_axis /= np.linalg.norm(long_axis)
    short_axis /= np.linalg.norm(short_axis)
    return center, long_axis, short_axis


def build_pitch_grid_ecef(
    lat_lon_alt: list[tuple[float, float, float]],
    *,
    n_rows: int,
    n_cols: int,
    pitch_length_m: float,
    pitch_width_m: float,
    antenna_offset_m: float = 1.5,
) -> list[dict]:
    """Equidistant grid on the pitch plane; rows × cols points."""
    center, long_axis, short_axis = _pitch_axes_from_trajectory(lat_lon_alt)
    basis = _enu_basis_at_ecef(center)
    up = basis[:, 2]
    rx_height = center + up * float(antenna_offset_m)

    row_offs = np.linspace(-0.5 * pitch_width_m, 0.5 * pitch_width_m, int(n_rows))
    col_offs = np.linspace(-0.5 * pitch_length_m, 0.5 * pitch_length_m, int(n_cols))

    points: list[dict] = []
    for ri, off_short in enumerate(row_offs):
        for ci, off_long in enumerate(col_offs):
            pos = rx_height + long_axis * float(off_long) + short_axis * float(off_short)
            lat, lon, alt = ecef_to_lla(float(pos[0]), float(pos[1]), float(pos[2]))
            points.append(
                {
                    "row": ri,
                    "col": ci,
                    "lat_deg": math.degrees(lat),
                    "lon_deg": math.degrees(lon),
                    "alt_m": float(alt),
                    "ecef": pos,
                }
            )
    return points


def compute_dop_from_azel(az: np.ndarray, el: np.ndarray) -> tuple[float, float, float, float, int]:
    """PDOP, HDOP, VDOP, GDOP from az/el arrays (skyplot.cu convention)."""
    az = np.asarray(az, dtype=np.float64).ravel()
    el = np.asarray(el, dtype=np.float64).ravel()
    n = int(az.size)
    if n < 4:
        return float("nan"), float("nan"), float("nan"), float("nan"), n
    g = np.zeros((4, 4), dtype=np.float64)
    for i in range(n):
        ce, se = math.cos(el[i]), math.sin(el[i])
        sa, ca = math.sin(az[i]), math.cos(az[i])
        h = np.array([-ce * sa, -ce * ca, -se, 1.0], dtype=np.float64)
        g += np.outer(h, h)
    try:
        q = np.linalg.inv(g)
    except np.linalg.LinAlgError:
        return float("nan"), float("nan"), float("nan"), float("nan"), n
    tr_xyz = q[0, 0] + q[1, 1] + q[2, 2]
    tr_all = tr_xyz + q[3, 3]
    pdop = math.sqrt(tr_xyz) if tr_xyz > 0 else float("nan")
    hdop = math.sqrt(q[0, 0] + q[1, 1]) if (q[0, 0] + q[1, 1]) > 0 else float("nan")
    vdop = math.sqrt(q[2, 2]) if q[2, 2] > 0 else float("nan")
    gdop = math.sqrt(tr_all) if tr_all > 0 else float("nan")
    return pdop, hdop, vdop, gdop, n


def _daily_tow_samples(tow_start: float, duration_s: float, dt_s: float) -> np.ndarray:
    n = max(1, int(math.floor(float(duration_s) / float(dt_s))) + 1)
    tows = tow_start + np.arange(n, dtype=np.float64) * float(dt_s)
    return np.mod(tows, 604800.0)


def evaluate_grid_point_daily_means(
    rx_ecef: np.ndarray,
    bvh: BVHAccelerator,
    eph: Ephemeris,
    prn_catalog: list,
    tow_samples: np.ndarray,
    *,
    elevation_mask_deg: float,
    eph_batch_chunk: int,
) -> dict:
    mask_rad = math.radians(float(elevation_mask_deg))
    n_epochs = int(tow_samples.size)
    n_los = np.zeros(n_epochs, dtype=np.float64)
    n_nlos = np.zeros(n_epochs, dtype=np.float64)
    pdop = np.full(n_epochs, np.nan, dtype=np.float64)
    hdop = np.full(n_epochs, np.nan, dtype=np.float64)
    vdop = np.full(n_epochs, np.nan, dtype=np.float64)
    gdop = np.full(n_epochs, np.nan, dtype=np.float64)

    chunk = max(1, int(eph_batch_chunk))
    rx = np.ascontiguousarray(rx_ecef.reshape(1, 3), dtype=np.float64)

    for start in range(0, n_epochs, chunk):
        end = min(start + chunk, n_epochs)
        tow_chunk = np.asarray(tow_samples[start:end], dtype=np.float64)
        sat_b, _clk_b, used_prns = eph.compute_batch(tow_chunk, prn_list=prn_catalog)
        if sat_b.shape[1] == 0:
            continue

        n_b = sat_b.shape[0]
        n_sat = sat_b.shape[1]
        rx_blk = np.repeat(rx, n_b, axis=0)
        el_batch = np.zeros((n_b, n_sat), dtype=np.float64)
        az_batch = np.zeros((n_b, n_sat), dtype=np.float64)
        for i in range(n_b):
            el_batch[i], az_batch[i] = _sat_elevation_azimuth(rx_blk[i], sat_b[i])

        visible = el_batch >= mask_rad
        sat_work = np.array(sat_b, dtype=np.float64, copy=True)
        sat_work[~visible] = np.nan
        los_batch = np.asarray(bvh.check_los_batch(rx_blk, sat_work), dtype=bool)

        los_vis = los_batch & visible
        nlos_vis = (~los_batch) & visible
        n_los[start:end] = los_vis.sum(axis=1)
        n_nlos[start:end] = nlos_vis.sum(axis=1)

        for i in range(n_b):
            idx = np.nonzero(los_vis[i])[0]
            if idx.size < 4:
                continue
            p, h, v, g, _ = compute_dop_from_azel(az_batch[i, idx], el_batch[i, idx])
            j = start + i
            pdop[j], hdop[j], vdop[j], gdop[j] = p, h, v, g

    return {
        "mean_n_los": float(np.mean(n_los)),
        "mean_n_nlos": float(np.mean(n_nlos)),
        "mean_pdop_los": float(np.nanmean(pdop)),
        "mean_hdop_los": float(np.nanmean(hdop)),
        "mean_vdop_los": float(np.nanmean(vdop)),
        "mean_gdop_los": float(np.nanmean(gdop)),
        "n_epochs": n_epochs,
        "n_epochs_pdop_ok": int(np.sum(np.isfinite(pdop))),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--trajectory-csv",
        type=Path,
        required=True,
        help="Lat/lon CSV used to infer pitch center and orientation (perimeter loop is fine).",
    )
    ap.add_argument("--triangles-npy", type=Path, required=True)
    ap.add_argument("--nav", type=Path, required=True)
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--alt-m", type=float, default=62.0, help="Default ellipsoidal height if CSV has no alt column.")
    ap.add_argument("--grid-rows", type=int, default=3)
    ap.add_argument("--grid-cols", type=int, default=5)
    ap.add_argument("--pitch-length-m", type=float, default=105.0, help="Grid span along long axis [m].")
    ap.add_argument("--pitch-width-m", type=float, default=68.0, help="Grid span along short axis [m].")
    ap.add_argument("--antenna-offset-m", type=float, default=1.5, help="Antenna height above pitch plane [m].")
    ap.add_argument("--start-utc", type=str, default="", help="Day start UTC (ISO).")
    ap.add_argument("--gps-week", type=int, default=0)
    ap.add_argument("--tow-start-s", type=float, default=0.0)
    ap.add_argument("--day-duration-s", type=float, default=86400.0)
    ap.add_argument("--dt-s", type=float, default=60.0, help="Ephemeris sampling step along the day [s].")
    ap.add_argument("--elevation-mask-deg", type=float, default=10.0)
    ap.add_argument("--eph-batch-chunk", type=int, default=64)
    ap.add_argument(
        "--nav-systems",
        type=str,
        default="G,E,J",
        help="Constellations to include from RINEX NAV (comma-separated).",
    )
    args = ap.parse_args()

    lat_lon_alt, _alt_from_csv = read_lat_lon_alt_csv(args.trajectory_csv.resolve(), default_alt_m=float(args.alt_m))
    if args.start_utc.strip():
        dt = datetime.fromisoformat(args.start_utc.replace("Z", "+00:00"))
        _week, tow_start = _gps_week_tow_from_utc(dt)
        print(f"[grid] start UTC {args.start_utc} -> TOW {tow_start:.1f} s", flush=True)
    else:
        if int(args.gps_week) <= 0:
            ap.error("Provide --start-utc or --gps-week + --tow-start-s.")
        tow_start = float(args.tow_start_s)

    tow_samples = _daily_tow_samples(tow_start, float(args.day_duration_s), float(args.dt_s))
    print(
        f"[grid] day samples: n={tow_samples.size}, dt={float(args.dt_s):g}s, "
        f"span={float(args.day_duration_s):g}s",
        flush=True,
    )

    grid = build_pitch_grid_ecef(
        lat_lon_alt,
        n_rows=int(args.grid_rows),
        n_cols=int(args.grid_cols),
        pitch_length_m=float(args.pitch_length_m),
        pitch_width_m=float(args.pitch_width_m),
        antenna_offset_m=float(args.antenna_offset_m),
    )
    print(
        f"[grid] pitch grid {args.grid_rows}x{args.grid_cols} = {len(grid)} points "
        f"({float(args.pitch_length_m):g} x {float(args.pitch_width_m):g} m)",
        flush=True,
    )

    tri = np.load(args.triangles_npy.resolve())
    if tri.ndim != 3 or tri.shape[1:] != (3, 3):
        ap.error(f"--triangles-npy must be [N,3,3], got {tri.shape}")
    building = BuildingModel(np.asarray(tri, dtype=np.float64))
    bvh = BVHAccelerator.from_building_model(building)
    if not hasattr(bvh, "check_los_batch"):
        ap.error("BVH check_los_batch unavailable — rebuild gnss_gpu with CUDA BVH support.")

    systems = tuple(s.strip() for s in str(args.nav_systems).split(",") if s.strip())
    nav_messages = read_nav_rinex_multi(str(args.nav.resolve()), systems=systems)
    eph = Ephemeris(nav_messages)
    prn_catalog = eph.available_prns
    print(f"[grid] NAV PRNs available: {len(prn_catalog)} ({','.join(systems)})", flush=True)

    rows_out: list[dict] = []
    for gi, pt in enumerate(grid):
        stats = evaluate_grid_point_daily_means(
            pt["ecef"],
            bvh,
            eph,
            prn_catalog,
            tow_samples,
            elevation_mask_deg=float(args.elevation_mask_deg),
            eph_batch_chunk=int(args.eph_batch_chunk),
        )
        row = {**pt, **stats}
        rows_out.append(row)
        print(
            f"  [{gi + 1}/{len(grid)}] row={pt['row']} col={pt['col']} "
            f"mean LOS={stats['mean_n_los']:.2f} NLOS={stats['mean_n_nlos']:.2f} "
            f"PDOP={stats['mean_pdop_los']:.2f} (pdop epochs {stats['n_epochs_pdop_ok']}/{stats['n_epochs']})",
            flush=True,
        )

    out = args.out_csv.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "row",
        "col",
        "lat_deg",
        "lon_deg",
        "alt_m",
        "ecef_x",
        "ecef_y",
        "ecef_z",
        "mean_n_los",
        "mean_n_nlos",
        "mean_pdop_los",
        "mean_hdop_los",
        "mean_vdop_los",
        "mean_gdop_los",
        "n_epochs",
        "n_epochs_pdop_ok",
    ]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows_out:
            ecef = r["ecef"]
            w.writerow(
                {
                    "row": r["row"],
                    "col": r["col"],
                    "lat_deg": f"{r['lat_deg']:.8f}",
                    "lon_deg": f"{r['lon_deg']:.8f}",
                    "alt_m": f"{r['alt_m']:.3f}",
                    "ecef_x": f"{float(ecef[0]):.3f}",
                    "ecef_y": f"{float(ecef[1]):.3f}",
                    "ecef_z": f"{float(ecef[2]):.3f}",
                    "mean_n_los": f"{r['mean_n_los']:.4f}",
                    "mean_n_nlos": f"{r['mean_n_nlos']:.4f}",
                    "mean_pdop_los": f"{r['mean_pdop_los']:.4f}" if math.isfinite(r["mean_pdop_los"]) else "",
                    "mean_hdop_los": f"{r['mean_hdop_los']:.4f}" if math.isfinite(r["mean_hdop_los"]) else "",
                    "mean_vdop_los": f"{r['mean_vdop_los']:.4f}" if math.isfinite(r["mean_vdop_los"]) else "",
                    "mean_gdop_los": f"{r['mean_gdop_los']:.4f}" if math.isfinite(r["mean_gdop_los"]) else "",
                    "n_epochs": r["n_epochs"],
                    "n_epochs_pdop_ok": r["n_epochs_pdop_ok"],
                }
            )
    print(f"[grid] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
