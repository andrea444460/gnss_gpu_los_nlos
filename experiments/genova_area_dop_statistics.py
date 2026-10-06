#!/usr/bin/env python3
"""Compute mean PDOP/HDOP/VDOP/GDOP for area-map road points.

The area CSV from ``build_area_los_nlos_map`` stores LOS/NLOS counts but not DOP.
This script re-evaluates ephemeris geometry and optionally urban LOS masking (BVH)
to attach time-averaged DOP columns and zone summaries.

Example (Genova, sky-only DOP — no building mesh needed)::

    PYTHONPATH=python python experiments/genova_area_dop_statistics.py \\
      --csv \"C:/Users/Andrea/Desktop/Gneova Map/genova_area_points_osm_full.csv\" \\
      --nav \"C:/Users/Andrea/Desktop/Dataset_sample_2/base/BRDC00IGS_R_20252600000_01D_MN.rnx\" \\
      --start-utc 2025-09-17T10:00:00Z \\
      --duration-s 7200 --dt-s 300 \\
      --dop-mode visible \\
      --stride 20 \\
      --out-prefix experiments/results/genova_dop_visible

With OSM building mesh (urban LOS DOP) add ``--triangles-npy path/to/triangles.npy --dop-mode los``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_ROOT / "python"))

from gnss_gpu.bvh import BVHAccelerator
from gnss_gpu.ephemeris import Ephemeris
from gnss_gpu.io.nav_rinex import read_nav_rinex_multi
from gnss_gpu.raytrace import BuildingModel
from gnss_gpu.urban_signal_sim import _sat_elevation_azimuth, ecef_to_lla

WGS84_A = 6_378_137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = 2.0 * WGS84_F - WGS84_F * WGS84_F

DEFAULT_CENTRO_BBOX = dict(south=44.4038, west=8.9290, north=44.4108, east=8.9415)
DEFAULT_COAST_POLYLINE = [
    (44.4185, 8.8050), (44.4150, 8.8400), (44.4100, 8.8750), (44.4050, 8.9050),
    (44.4020, 8.9250), (44.4010, 8.9350), (44.4005, 8.9450), (44.3990, 8.9650),
    (44.3975, 8.9850), (44.3960, 9.0050), (44.3945, 9.0250),
]


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
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    sin_lon, cos_lon = math.sin(lon), math.cos(lon)
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
    return float(start_tow_s) + np.arange(n, dtype=np.float64) * float(dt_s)


def compute_dop_from_azel(az: np.ndarray, el: np.ndarray) -> tuple[float, float, float, float, int]:
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


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    return 2.0 * WGS84_A * math.asin(min(1.0, math.sqrt(a)))


def _dist_to_polyline_m(lat: float, lon: float, polyline: list[tuple[float, float]]) -> float:
    dmin = float("inf")
    for i in range(len(polyline) - 1):
        la0, lo0 = polyline[i]
        la1, lo1 = polyline[i + 1]
        for t in np.linspace(0.0, 1.0, 21):
            d = _haversine_m(lat, lon, la0 + t * (la1 - la0), lo0 + t * (lo1 - lo0))
            if d < dmin:
                dmin = d
    return dmin


@dataclass
class PointDop:
    lat_deg: float
    lon_deg: float
    zone: str
    mean_pdop: float
    mean_hdop: float
    mean_vdop: float
    mean_gdop: float
    n_epochs_used: int


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--nav", type=Path, required=True)
    p.add_argument("--triangles-npy", type=Path, default=None)
    p.add_argument("--dop-mode", choices=("visible", "los"), default="visible")
    p.add_argument("--nav-systems", type=str, default="G,E,J,C")
    p.add_argument("--start-utc", type=str, required=True)
    p.add_argument("--duration-s", type=float, default=7200.0)
    p.add_argument("--dt-s", type=float, default=300.0)
    p.add_argument("--elevation-mask-deg", type=float, default=10.0)
    p.add_argument("--rx-alt-m", type=float, default=1.5)
    p.add_argument("--stride", type=int, default=1, help="Process every N-th CSV row")
    p.add_argument("--max-points", type=int, default=0, help="0 = no cap")
    p.add_argument("--eph-batch-chunk", type=int, default=16)
    p.add_argument("--coastal-buffer-m", type=float, default=350.0)
    p.add_argument("--out-prefix", type=Path, required=True)
    return p.parse_args()


def _classify_zone(lat: float, lon: float, coastal_buffer_m: float) -> str:
    bb = DEFAULT_CENTRO_BBOX
    if bb["south"] <= lat <= bb["north"] and bb["west"] <= lon <= bb["east"]:
        return "centro_storico"
    if _dist_to_polyline_m(lat, lon, DEFAULT_COAST_POLYLINE) <= coastal_buffer_m:
        return "costiera"
    return "other"


def _evaluate_point(
    rx_ecef: np.ndarray,
    tow_samples: np.ndarray,
    eph: Ephemeris,
    prn_catalog: list[str],
    *,
    bvh: BVHAccelerator | None,
    dop_mode: str,
    mask_rad: float,
    eph_chunk: int,
) -> tuple[float, float, float, float, int]:
    pdops: list[float] = []
    hdops: list[float] = []
    vdops: list[float] = []
    gdops: list[float] = []
    rx = np.ascontiguousarray(rx_ecef.reshape(1, 3), dtype=np.float64)

    for start in range(0, tow_samples.size, eph_chunk):
        end = min(start + eph_chunk, tow_samples.size)
        tow_chunk = np.asarray(tow_samples[start:end], dtype=np.float64)
        sat_b, _clk, _used = eph.compute_batch(tow_chunk, prn_list=prn_catalog)
        if sat_b.shape[1] == 0:
            continue
        n_b, n_sat = sat_b.shape[0], sat_b.shape[1]
        rx_blk = np.repeat(rx, n_b, axis=0)
        el = np.zeros((n_b, n_sat), dtype=np.float64)
        az = np.zeros((n_b, n_sat), dtype=np.float64)
        for i in range(n_b):
            el[i], az[i] = _sat_elevation_azimuth(rx_blk[i], sat_b[i])
        visible = el >= mask_rad

        if dop_mode == "los":
            if bvh is None:
                raise ValueError("--dop-mode los requires --triangles-npy")
            sat_work = np.array(sat_b, dtype=np.float64, copy=True)
            sat_work[~visible] = np.nan
            los = np.asarray(bvh.check_los_batch(rx_blk, sat_work), dtype=bool)
            use = los & visible
        else:
            use = visible

        for i in range(n_b):
            idx = np.nonzero(use[i])[0]
            p, h, v, g, _n = compute_dop_from_azel(az[i, idx], el[i, idx])
            if np.isfinite(p):
                pdops.append(p)
                hdops.append(h)
                vdops.append(v)
                gdops.append(g)

    if not pdops:
        return float("nan"), float("nan"), float("nan"), float("nan"), 0
    return (
        float(np.mean(pdops)),
        float(np.mean(hdops)),
        float(np.mean(vdops)),
        float(np.mean(gdops)),
        len(pdops),
    )


def _zone_summary(rows: list[PointDop], zone: str) -> dict:
    sub = [r for r in rows if r.zone == zone]
    if not sub:
        return {"n": 0}
    out: dict = {"n": len(sub)}
    for key in ("mean_pdop", "mean_hdop", "mean_vdop", "mean_gdop"):
        arr = np.array([getattr(r, key) for r in sub], dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            out[key] = {
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
                "p25": float(np.percentile(arr, 25)),
                "p75": float(np.percentile(arr, 75)),
            }
    return out


def main() -> None:
    args = _parse_args()
    systems = tuple(s.strip() for s in str(args.nav_systems).split(",") if s.strip())
    dt = datetime.fromisoformat(args.start_utc.replace("Z", "+00:00"))
    _week, tow_start = _gps_week_tow_from_utc(dt)
    tow_samples = _time_samples(tow_start, float(args.duration_s), float(args.dt_s))
    mask_rad = math.radians(float(args.elevation_mask_deg))

    nav = read_nav_rinex_multi(str(args.nav.resolve()), systems=systems)
    eph = Ephemeris(nav)
    prn_catalog = list(eph.available_prns)
    print(f"[dop] NAV PRNs: {len(prn_catalog)} systems={','.join(systems)}")
    print(f"[dop] time samples: {tow_samples.size} dt={args.dt_s}s from TOW {tow_start:.1f}")

    bvh = None
    if args.dop_mode == "los" or args.triangles_npy is not None:
        if args.triangles_npy is None:
            raise ValueError("--triangles-npy required for --dop-mode los")
        tri = np.asarray(np.load(args.triangles_npy), dtype=np.float64)
        bvh = BVHAccelerator.from_building_model(BuildingModel(tri))
        print(f"[dop] BVH triangles={tri.shape[0]} mode={args.dop_mode}")

    # Load CSV rows (strided).
    csv_rows: list[dict] = []
    with open(args.csv, newline="", encoding="utf-8") as f:
        for i, rr in enumerate(csv.DictReader(f)):
            if i % max(1, int(args.stride)) != 0:
                continue
            try:
                csv_rows.append({"lat_deg": float(rr["lat_deg"]), "lon_deg": float(rr["lon_deg"])})
            except (KeyError, ValueError):
                continue
            if int(args.max_points) > 0 and len(csv_rows) >= int(args.max_points):
                break
    print(f"[dop] processing {len(csv_rows)} road points (stride={args.stride})")

    results: list[PointDop] = []
    for i, row in enumerate(csv_rows):
        lat, lon = float(row["lat_deg"]), float(row["lon_deg"])
        rx = _lla_deg_to_ecef(lat, lon, float(args.rx_alt_m))
        pd, hd, vd, gd, n_ep = _evaluate_point(
            rx,
            tow_samples,
            eph,
            prn_catalog,
            bvh=bvh,
            dop_mode=str(args.dop_mode),
            mask_rad=mask_rad,
            eph_chunk=int(args.eph_batch_chunk),
        )
        results.append(
            PointDop(
                lat_deg=lat,
                lon_deg=lon,
                zone=_classify_zone(lat, lon, float(args.coastal_buffer_m)),
                mean_pdop=pd,
                mean_hdop=hd,
                mean_vdop=vd,
                mean_gdop=gd,
                n_epochs_used=n_ep,
            )
        )
        if (i + 1) % 100 == 0 or i + 1 == len(csv_rows):
            print(f"  [{i+1}/{len(csv_rows)}] last PDOP={pd:.2f} zone={results[-1].zone}")

    prefix = args.out_prefix.resolve()
    out_csv = Path(str(prefix) + "_points.csv")
    out_json = Path(str(prefix) + "_zone_summary.json")
    out_png = Path(str(prefix) + "_zone_comparison.png")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["lat_deg", "lon_deg", "zone", "mean_pdop", "mean_hdop", "mean_vdop", "mean_gdop", "n_epochs_used"])
        for r in results:
            w.writerow([r.lat_deg, r.lon_deg, r.zone, r.mean_pdop, r.mean_hdop, r.mean_vdop, r.mean_gdop, r.n_epochs_used])

    summary = {
        "csv": str(args.csv.resolve()),
        "nav": str(args.nav.resolve()),
        "dop_mode": args.dop_mode,
        "start_utc": args.start_utc,
        "duration_s": float(args.duration_s),
        "dt_s": float(args.dt_s),
        "n_points": len(results),
        "zones": {
            "centro_storico": _zone_summary(results, "centro_storico"),
            "costiera": _zone_summary(results, "costiera"),
            "other": _zone_summary(results, "other"),
        },
    }
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Bar chart centro vs costiera (PDOP/HDOP/VDOP).
    fig, ax = plt.subplots(figsize=(8, 4.5))
    zones = ["centro_storico", "costiera"]
    labels = ["PDOP", "HDOP", "VDOP"]
    x = np.arange(len(labels))
    w = 0.35
    for zi, zone in enumerate(zones):
        zs = summary["zones"][zone]
        vals = [zs.get(f"mean_{k.lower()}", {}).get("mean", float("nan")) for k in labels]
        offset = -w / 2 if zi == 0 else w / 2
        ax.bar(x + offset, vals, w, label=f"{zone} (n={zs.get('n', 0)})")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean DOP")
    ax.set_title(f"Genova DOP by zone ({args.dop_mode} satellites)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\n[dop] wrote:\n  {out_csv}\n  {out_json}\n  {out_png}")
    for zone in zones:
        zs = summary["zones"][zone]
        if zs.get("n", 0):
            print(
                f"  {zone}: PDOP mean={zs['mean_pdop']['mean']:.2f} "
                f"HDOP={zs['mean_hdop']['mean']:.2f} VDOP={zs['mean_vdop']['mean']:.2f} (n={zs['n']})"
            )


if __name__ == "__main__":
    main()
