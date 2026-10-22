#!/usr/bin/env python3
"""Skyline FOV + obstruction skyplot (+ optional NAV satellite tracks).

For each RX, sweeps azimuth 0..360° (0 = North) and finds the minimum
elevation where a ray clears the mesh (BVH binary search). Optionally
takes max with the DEM terrain horizon.

Polar panels: gray obstruction mask + (if ``--nav``) per-PRN arcs from
ephemeris — same track style as ``build_skyplot_from_rinex.py``, but NAV-only
(no OBS). Linear FOV panels underneath use the same skyline.

Example (Genova Colab / local)::

  python experiments/plot_rx_skyline.py \\
    --coords /content/los_work/data/points.csv \\
    --triangles /content/los_work/results/genova_osm_triangles.npy \\
    --rx-alt-mode dem --dem-path /content/los_work/results/genova_area_dem.tif \\
    --geoid-correction egm96 \\
    --nav /content/los_work/data/BRDC....rnx \\
    --utc 2026-10-09T12:00:00Z --duration-s 7200 --dt-s 60 \\
    --out-png /content/los_work/results/rx_skyline.png \\
    --out-skyplot-png /content/los_work/results/rx_obstruction_skyplot.png \\
    --out-mask-json /content/los_work/results/rx_skyline_mask.json \\
    --out-mask-pkl /content/los_work/results/rx_skyline_mask.pkl
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "python"))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from gnss_gpu.bvh import BVHAccelerator
from gnss_gpu.ephemeris import Ephemeris
from gnss_gpu.io.dem_native import NativeDemSampler, parse_geoid_correction_arg
from gnss_gpu.io.nav_rinex import read_nav_rinex_multi
from gnss_gpu.raytrace import BuildingModel
from gnss_gpu.terrain_horizon import HorizonConfig, TerrainHorizonMask
from gnss_gpu.urban_signal_sim import _sat_elevation_azimuth
from label_los_nlos_coords import _lla_deg_to_ecef, _parse_utc, _utc_to_gps_week_tow, load_coords

# Same palette as build_skyplot_from_rinex.py
SYS_COLORS = {
    "G": "#1f77b4",
    "R": "#d62728",
    "E": "#2ca02c",
    "C": "#ff7f0e",
    "J": "#9467bd",
    "I": "#8c564b",
}
GPS_WEEK_SECONDS = 604800.0

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = 2.0 * WGS84_F - WGS84_F * WGS84_F


def _ecef_to_enu_basis(ecef: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, y, z = (float(ecef[0]), float(ecef[1]), float(ecef[2]))
    p = math.hypot(x, y)
    if p < 1e-3:
        return (
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        )
    lon = math.atan2(y, x)
    lat = math.atan2(z, p * (1.0 - WGS84_E2))
    for _ in range(5):
        sinlat = math.sin(lat)
        n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sinlat * sinlat)
        lat = math.atan2(z + WGS84_E2 * n * sinlat, p)
    sinlat, coslat = math.sin(lat), math.cos(lat)
    sinlon, coslon = math.sin(lon), math.cos(lon)
    east = np.array([-sinlon, coslon, 0.0])
    north = np.array([-sinlat * coslon, -sinlat * sinlon, coslat])
    up = np.array([coslat * coslon, coslat * sinlon, sinlat])
    return east, north, up


def building_skyline_deg(
    bvh: BVHAccelerator,
    rx: np.ndarray,
    az_deg: np.ndarray,
    *,
    probe_m: float = 8000.0,
    n_bisect: int = 8,
) -> np.ndarray:
    """Min elevation [deg] clear of buildings, per azimuth (0=N, clockwise from N via E)."""
    east, north, up = _ecef_to_enu_basis(rx)
    az = np.radians(np.asarray(az_deg, dtype=np.float64))
    n = az.size
    lo = np.zeros(n, dtype=np.float64)
    hi = np.full(n, 90.0, dtype=np.float64)
    # Batch binary search: shared elevation level across azimuths each iteration.
    for _ in range(int(n_bisect)):
        mid = 0.5 * (lo + hi)
        el = np.radians(mid)
        # az: 0=N, 90=E  →  ENU: n=cos(az), e=sin(az)
        cos_el, sin_el = np.cos(el), np.sin(el)
        e = np.sin(az) * cos_el
        n_comp = np.cos(az) * cos_el
        u = sin_el
        dirs = e[:, None] * east + n_comp[:, None] * north + u[:, None] * up
        sats = rx.reshape(1, 3) + float(probe_m) * dirs
        los = np.asarray(bvh.check_los(rx, sats), dtype=bool).ravel()
        hi = np.where(los, mid, hi)
        lo = np.where(los, lo, mid)
    return hi


def terrain_skyline_deg(
    terrain: TerrainHorizonMask,
    rx: np.ndarray,
    az_deg: np.ndarray,
) -> np.ndarray:
    """Interpolate DEM horizon profile onto requested azimuths [deg]."""
    prof = np.asarray(terrain.horizon_profile_rad(rx), dtype=np.float64)
    bins = np.asarray(terrain.azimuth_bins_rad, dtype=np.float64)
    # Wrap-friendly: append first bin + 2π
    bins_ext = np.concatenate([bins, bins[:1] + 2.0 * math.pi])
    prof_ext = np.concatenate([prof, prof[:1]])
    az = np.mod(np.radians(np.asarray(az_deg, dtype=np.float64)), 2.0 * math.pi)
    el = np.interp(az, bins_ext, prof_ext)
    return np.degrees(el)


def collect_nav_tracks(
    eph: Ephemeris,
    rx: np.ndarray,
    *,
    tow0: float,
    duration_s: float,
    dt_s: float,
    elev_mask_deg: float,
    systems: set[str],
) -> dict[str, list[tuple[float, float]]]:
    """Per-PRN list of (az_rad, r=90-el_deg) over a NAV time window (no OBS)."""
    catalog = [
        str(p).strip().upper()
        for p in eph.available_prns
        if str(p).strip() and str(p).strip().upper()[0] in systems
    ]
    tracks: dict[str, list[tuple[float, float]]] = {}
    if not catalog or duration_s <= 0 or dt_s <= 0:
        return tracks
    n_steps = int(math.floor(float(duration_s) / float(dt_s))) + 1
    mask = float(elev_mask_deg)
    for k in range(n_steps):
        tow = float(tow0) + k * float(dt_s)
        # Week wrap for compute(); Ephemeris usually wants TOW in [0, 604800)
        tow_mod = tow % GPS_WEEK_SECONDS
        try:
            sat_ecef, _clk, used = eph.compute(tow_mod, prn_list=catalog)
        except Exception:
            continue
        if len(used) == 0:
            continue
        el, az = _sat_elevation_azimuth(rx, sat_ecef)
        el_deg = np.degrees(el)
        az = np.mod(az, 2.0 * np.pi)
        for j, prn in enumerate(used):
            if float(el_deg[j]) < mask:
                continue
            sid = str(prn).strip().upper()
            tracks.setdefault(sid, []).append((float(az[j]), 90.0 - float(el_deg[j])))
    return tracks


def draw_sat_tracks(
    ax,
    tracks: dict[str, list[tuple[float, float]]],
    *,
    mono_color: str | None = None,
    label: bool = True,
) -> None:
    """Overlay PRN arcs (build_skyplot_from_rinex style)."""
    for sat_id in sorted(tracks.keys()):
        pts = tracks[sat_id]
        if len(pts) < 2:
            continue
        sys_id = sat_id[0]
        color = mono_color if mono_color else SYS_COLORS.get(sys_id, "#7f7f7f")
        az_vals = np.asarray([p[0] for p in pts], dtype=np.float64)
        r_vals = np.asarray([p[1] for p in pts], dtype=np.float64)
        # Break line on large azimuth wraps so the chord does not cross the plot.
        d = np.abs(np.diff(az_vals))
        breaks = np.where(d > math.radians(25.0))[0]
        starts = np.concatenate([[0], breaks + 1])
        ends = np.concatenate([breaks + 1, [len(az_vals)]])
        for s, e in zip(starts, ends):
            if e - s < 2:
                continue
            ax.plot(az_vals[s:e], r_vals[s:e], "-", color=color, alpha=0.9, lw=1.2, zorder=4)
        ax.plot(az_vals[-1], r_vals[-1], "o", color=color, ms=2.5, zorder=5)
        if label:
            ax.text(
                float(az_vals[-1] + math.radians(1.2)),
                float(min(max(r_vals[-1] + 1.0, 0.0), 90.0)),
                sat_id,
                fontsize=6,
                ha="left",
                va="center",
                color="#333333",
                zorder=6,
            )


def draw_obstruction_skyplot(
    ax,
    az_deg: np.ndarray,
    el_skyline_deg: np.ndarray,
    title: str,
    *,
    tracks: dict[str, list[tuple[float, float]]] | None = None,
    mono_tracks: bool = False,
) -> None:
    """Polar skyplot: zenith center, N up, CW; gray = blocked; optional sat tracks."""
    az = np.asarray(az_deg, dtype=np.float64)
    el = np.clip(np.asarray(el_skyline_deg, dtype=np.float64), 0.0, 90.0)
    # Close the loop for fill
    az_c = np.concatenate([az, az[:1]])
    el_c = np.concatenate([el, el[:1]])
    theta = np.radians(az_c)
    r_sky = 90.0 - el_c
    r_hor = np.full_like(r_sky, 90.0)
    # Polygon: along skyline, then back along horizon
    theta_poly = np.concatenate([theta, theta[::-1]])
    r_poly = np.concatenate([r_sky, r_hor[::-1]])
    ax.fill(theta_poly, r_poly, facecolor="0.78", edgecolor="none", zorder=1)
    ax.plot(theta, r_sky, color="0.35", lw=0.7, zorder=2)
    ax.plot(0.0, 0.0, "o", color="red", ms=3.5, zorder=3)  # zenith

    if tracks:
        draw_sat_tracks(ax, tracks, mono_color="#f0a000" if mono_tracks else None)

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)  # clockwise
    ax.set_ylim(0.0, 90.0)
    ticks = list(range(0, 91, 10))
    ax.set_yticks(ticks)
    ax.set_yticklabels([str(90 - t) for t in ticks], fontsize=7)
    ax.set_xticks(np.radians(np.arange(0.0, 360.0, 30.0)))
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(True, alpha=0.35)
    ax.set_title(title, fontsize=9, pad=12)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--coords", type=Path, default=None, help="CSV/JSON receiver points")
    p.add_argument("--triangles", type=Path, default=None, help="ECEF triangles.npy [N,3,3]")
    p.add_argument("--dem-path", type=Path, default=None, help="DEM GeoTIFF (orthometric → use geoid)")
    p.add_argument(
        "--source",
        choices=("buildings", "terrain", "both"),
        default="both",
        help="Skyline source (default both; falls back if mesh/DEM missing)",
    )
    p.add_argument("--rx-alt-mode", choices=("fixed", "dem"), default="fixed")
    p.add_argument("--rx-ant-height-m", type=float, default=1.5)
    p.add_argument("--default-alt-m", type=float, default=1.5)
    p.add_argument("--geoid-correction", default="none", help="egm96 | constant N [m] | none")
    p.add_argument("--az-step-deg", type=float, default=1.0)
    p.add_argument("--probe-m", type=float, default=8000.0, help="Synthetic ray length for building BVH")
    p.add_argument("--n-bisect", type=int, default=8, help="Binary-search iterations (~90/2^n deg)")
    p.add_argument("--terrain-max-distance-m", type=float, default=20000.0)
    p.add_argument("--terrain-step-m", type=float, default=60.0)
    p.add_argument("--out-png", type=Path, default=None, help="Combined PNG (skyplot top + FOV bottom)")
    p.add_argument(
        "--out-skyplot-png",
        type=Path,
        default=None,
        help="Optional separate PNG with only polar obstruction skyplots",
    )
    p.add_argument("--out-csv", type=Path, default=None, help="Long CSV: id,az_deg,el_deg")
    p.add_argument(
        "--out-mask-json",
        type=Path,
        default=None,
        help="Readable skyline mask JSON (per-RX masking_angle_deg[360], coords, ECEF)",
    )
    p.add_argument(
        "--out-mask-pkl",
        type=Path,
        default=None,
        help="Legacy-compatible pickle like receiverPoints_LOS_ele_00deg.pkl",
    )
    p.add_argument("--ylim-deg", type=float, default=90.0)
    # Satellite arcs (NAV-only; same idea as build_skyplot_from_rinex prn-tracks)
    p.add_argument("--nav", type=Path, default=None, help="RINEX NAV for satellite track overlay")
    p.add_argument("--utc", type=str, default=None, help="Track window start UTC (ISO-8601)")
    p.add_argument("--gps-week", type=int, default=None)
    p.add_argument("--gps-tow", type=float, default=None, help="Track window start GPS TOW [s]")
    p.add_argument("--duration-s", type=float, default=7200.0, help="Track window length [s]")
    p.add_argument("--dt-s", type=float, default=60.0, help="Ephemeris sample step [s]")
    p.add_argument("--systems", type=str, default="G,R,E,C,J", help="Comma-separated GNSS systems")
    p.add_argument("--track-elev-mask-deg", type=float, default=0.0)
    p.add_argument("--mono-tracks", action="store_true", help="Single color for all PRN arcs")
    p.add_argument(
        "--self-check",
        action="store_true",
        help="Run mock BVH skyline check (no mesh/coords) and exit",
    )
    return p.parse_args()


def _resample_mask_360(az_deg: np.ndarray, el_deg: np.ndarray) -> np.ndarray:
    """Length-360 mask: index i = azimuth i° from North."""
    az = np.asarray(az_deg, dtype=np.float64).ravel()
    el = np.asarray(el_deg, dtype=np.float64).ravel()
    if az.size == 360 and float(np.max(np.abs(az - np.arange(360.0)))) < 1e-9:
        return el.copy()
    az_ext = np.concatenate([az - 360.0, az, az + 360.0])
    el_ext = np.concatenate([el, el, el])
    return np.interp(np.arange(360.0), az_ext, el_ext).astype(np.float64)


def build_mask_payload(
    profiles: list[tuple[dict, np.ndarray, np.ndarray, float]],
    az_deg: np.ndarray,
    *,
    source: str,
) -> tuple[dict, dict]:
    """Return (readable_json_dict, legacy_pkl_dict) matching Desktop receiverPoints_*.pkl."""
    names: list[str] = []
    full_names: list[str] = []
    types: list[str] = []
    masks: list[np.ndarray] = []
    llh: list[list[float]] = []
    ecef: list[list[float]] = []
    env: list[float] = []
    receivers_json: list[dict] = []

    for pt, el, rx, alt in profiles:
        mask360 = _resample_mask_360(az_deg, el)
        rid = str(pt["id"])
        names.append(rid)
        full_names.append(f"{rid}_FullName")
        types.append("receiverType")
        masks.append(mask360)
        llh.append([float(pt["lat"]), float(pt["lon"]), float(alt)])
        ecef.append([float(rx[0]), float(rx[1]), float(rx[2])])
        env.append(0.3)
        receivers_json.append(
            {
                "id": rid,
                "lat_deg": float(pt["lat"]),
                "lon_deg": float(pt["lon"]),
                "alt_m": float(alt),
                "ecef_m": [float(rx[0]), float(rx[1]), float(rx[2])],
                "masking_angle_deg": [float(v) for v in mask360],
                "masking_angle_max_deg": float(np.max(mask360)),
                "masking_angle_mean_deg": float(np.mean(mask360)),
            }
        )

    lats = [r["lat_deg"] for r in receivers_json]
    lons = [r["lon_deg"] for r in receivers_json]
    readable = {
        "format": "gnss_gpu.rx_skyline_mask/v1",
        "azimuth_zero": "North",
        "azimuth_direction": "clockwise",
        "azimuth_step_deg": 1.0,
        "azimuth_bins": 360,
        "masking_angle_unit": "deg",
        "masking_angle_meaning": "min elevation free of buildings/terrain at that azimuth",
        "source": source,
        "bbox": {
            "min_lat": float(min(lats)),
            "max_lat": float(max(lats)),
            "min_lon": float(min(lons)),
            "max_lon": float(max(lons)),
        },
        "receivers": receivers_json,
    }
    legacy = {
        "receiverName": names,
        "receiverFullName": full_names,
        "receiverType": types,
        "maskingAngle": masks,
        "maxLon": float(max(lons)),
        "minLon": float(min(lons)),
        "maxLat": float(max(lats)),
        "minLat": float(min(lats)),
        "gridResolution": [0, 0],
        "coordEcef": ecef,
        "coordLlh": llh,
        "environment": env,
    }
    return readable, legacy


class _MockWallBvh:
    """Blocks rays with elevation below threshold_deg (ponytail: unit check only)."""

    def __init__(self, threshold_deg: float = 30.0):
        self.threshold_deg = float(threshold_deg)

    def check_los(self, rx_ecef, sat_ecef):
        rx = np.asarray(rx_ecef, dtype=np.float64).reshape(3)
        sat = np.asarray(sat_ecef, dtype=np.float64).reshape(-1, 3)
        east, north, up = _ecef_to_enu_basis(rx)
        rel = sat - rx.reshape(1, 3)
        e = rel @ east
        n = rel @ north
        u = rel @ up
        el_deg = np.degrees(np.arctan2(u, np.hypot(e, n)))
        return el_deg >= self.threshold_deg


def _self_check() -> None:
    # Genova-ish ECEF (approx) so ENU basis is well-conditioned
    rx = _lla_deg_to_ecef(44.42, 8.93, 50.0)
    az = np.arange(0.0, 360.0, 5.0)
    thr = 30.0
    el = building_skyline_deg(_MockWallBvh(thr), rx, az, probe_m=1000.0, n_bisect=10)
    err = float(np.max(np.abs(el - thr)))
    assert err < 0.2, f"skyline self-check failed: max|el-30|={err}"
    # Polar radius mapping: el=30 → r=60
    r = 90.0 - el
    assert float(np.max(np.abs(r - 60.0))) < 0.2
    m360 = _resample_mask_360(az, el)
    assert m360.shape == (360,)
    readable, legacy = build_mask_payload(
        [({"id": "0", "lat": 44.42, "lon": 8.93}, el, rx, 50.0)],
        az,
        source="buildings",
    )
    assert len(readable["receivers"][0]["masking_angle_deg"]) == 360
    assert len(legacy["maskingAngle"][0]) == 360
    assert legacy["coordLlh"][0][0] == 44.42
    print(f"self-check OK (max|el-30°|={err:.3f}°)")


def main() -> int:
    args = _parse_args()
    if args.self_check:
        _self_check()
        return 0
    want_plot = args.out_png is not None or args.out_skyplot_png is not None
    want_data = (
        args.out_csv is not None
        or args.out_mask_json is not None
        or args.out_mask_pkl is not None
    )
    if args.coords is None or (not want_plot and not want_data):
        raise SystemExit(
            "--coords plus at least one of --out-png/--out-skyplot-png/"
            "--out-csv/--out-mask-json/--out-mask-pkl (or --self-check)"
        )
    points = load_coords(args.coords, float(args.default_alt_m))
    print(f"Loaded {len(points)} point(s) from {args.coords}")

    want_b = args.source in ("buildings", "both")
    want_t = args.source in ("terrain", "both")

    bvh: BVHAccelerator | None = None
    if want_b:
        if args.triangles is None:
            raise SystemExit("--triangles required for buildings/both")
        tri = np.load(args.triangles)
        if tri.ndim != 3 or tri.shape[1:] != (3, 3):
            raise SystemExit(f"Expected triangles [N,3,3], got {tri.shape}")
        bvh = BVHAccelerator(BuildingModel(tri).triangles)
        print(f"Mesh: {tri.shape[0]} triangles → BVH OK")

    dem_sampler: NativeDemSampler | None = None
    terrain: TerrainHorizonMask | None = None
    geoid = parse_geoid_correction_arg(str(args.geoid_correction))
    if args.rx_alt_mode == "dem" or want_t:
        if args.dem_path is None:
            raise SystemExit("--dem-path required for --rx-alt-mode dem and/or terrain skyline")
        dem_sampler = NativeDemSampler(args.dem_path, geoid_correction=geoid)
        print(
            f"DEM: {args.dem_path} geoid={dem_sampler.geoid_label} "
            f"elev={dem_sampler.min_m:.1f}..{dem_sampler.max_m:.1f} m"
        )
    if want_t:
        assert args.dem_path is not None
        terrain = TerrainHorizonMask(
            args.dem_path,
            HorizonConfig(
                max_distance_m=float(args.terrain_max_distance_m),
                sample_step_m=float(args.terrain_step_m),
            ),
            geoid_correction=str(args.geoid_correction),
        )
        print(f"Terrain horizon: on (geoid={terrain._geoid_label})")

    eph: Ephemeris | None = None
    tow0: float | None = None
    systems: set[str] = set()
    if args.nav is not None:
        systems = {s.strip().upper() for s in str(args.systems).split(",") if s.strip()}
        if args.utc:
            _week, tow0 = _utc_to_gps_week_tow(_parse_utc(args.utc))
        elif args.gps_week is not None and args.gps_tow is not None:
            tow0 = float(args.gps_tow)
        else:
            raise SystemExit("--nav requires --utc or --gps-week/--gps-tow")
        nav_msgs = read_nav_rinex_multi(args.nav, systems=sorted(systems))
        eph = Ephemeris(nav_msgs)
        print(
            f"NAV tracks: {args.nav.name} systems={sorted(systems)} "
            f"tow0={tow0:.0f}s duration={float(args.duration_s):.0f}s dt={float(args.dt_s):.0f}s"
        )

    az_deg = np.arange(0.0, 360.0, float(args.az_step_deg), dtype=np.float64)
    rows_csv: list[dict] = []
    # (point, el_skyline, rx_ecef, alt_m)
    profiles: list[tuple[dict, np.ndarray, np.ndarray, float]] = []
    tracks_by_rx: list[dict[str, list[tuple[float, float]]]] = []

    for pt in points:
        if args.rx_alt_mode == "dem":
            assert dem_sampler is not None
            alt = float(dem_sampler.sample(pt["lat"], pt["lon"])) + float(args.rx_ant_height_m)
        else:
            alt = float(pt["alt"])
        rx = _lla_deg_to_ecef(pt["lat"], pt["lon"], alt)

        el = np.zeros_like(az_deg)
        if bvh is not None:
            el = np.maximum(
                el,
                building_skyline_deg(
                    bvh, rx, az_deg, probe_m=float(args.probe_m), n_bisect=int(args.n_bisect)
                ),
            )
        if terrain is not None:
            el = np.maximum(el, terrain_skyline_deg(terrain, rx, az_deg))
        profiles.append((pt, el, rx, alt))

        tracks: dict[str, list[tuple[float, float]]] = {}
        if eph is not None and tow0 is not None:
            tracks = collect_nav_tracks(
                eph,
                rx,
                tow0=float(tow0),
                duration_s=float(args.duration_s),
                dt_s=float(args.dt_s),
                elev_mask_deg=float(args.track_elev_mask_deg),
                systems=systems,
            )
        tracks_by_rx.append(tracks)
        print(
            f"  {pt['id']}: lat={pt['lat']:.5f} lon={pt['lon']:.5f} alt={alt:.1f} m  "
            f"skyline max={float(el.max()):.1f}° mean={float(el.mean()):.1f}°"
            + (f"  tracks={len(tracks)} PRN" if tracks else "")
        )
        if args.out_csv is not None:
            for a, e in zip(az_deg, el):
                rows_csv.append(
                    {
                        "id": pt["id"],
                        "lat": pt["lat"],
                        "lon": pt["lon"],
                        "az_deg": float(a),
                        "el_deg": float(e),
                    }
                )

    n = len(profiles)
    if args.out_png is not None:
        fig = plt.figure(figsize=(max(3.0 * n, 5.0), 6.4))
        for i, ((pt, el, _rx, _alt), tracks) in enumerate(zip(profiles, tracks_by_rx)):
            ax_p = fig.add_subplot(2, n, i + 1, projection="polar")
            draw_obstruction_skyplot(
                ax_p,
                az_deg,
                el,
                title=f"Skyplot [{pt['lat']:.2f}, {pt['lon']:.2f}]",
                tracks=tracks or None,
                mono_tracks=bool(args.mono_tracks),
            )
            ax_f = fig.add_subplot(2, n, n + i + 1)
            ax_f.plot(az_deg, el, color="C0", lw=1.0)
            ax_f.set_xlim(0, 360)
            ax_f.set_ylim(0, float(args.ylim_deg))
            ax_f.set_xlabel("360° FOV (0°=N)")
            ax_f.set_title(f"Receiver [{pt['lat']:.2f}, {pt['lon']:.2f}]")
            ax_f.grid(True, alpha=0.3)
            if i == 0:
                ax_f.set_ylabel("LOS elevation angle [deg]")
        fig.suptitle(f"Obstruction skyplot + skyline ({args.source})", fontsize=11)
        fig.tight_layout()
        args.out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {args.out_png}")

    if args.out_skyplot_png is not None:
        fig_s, axes_s = plt.subplots(
            1,
            n,
            figsize=(max(3.2 * n, 4.0), 3.8),
            subplot_kw={"projection": "polar"},
            squeeze=False,
        )
        for ax, (pt, el, _rx, _alt), tracks in zip(axes_s[0], profiles, tracks_by_rx):
            draw_obstruction_skyplot(
                ax,
                az_deg,
                el,
                title=f"Skyplot - Elevation angle [°]\n[{pt['lat']:.2f}, {pt['lon']:.2f}]",
                tracks=tracks or None,
                mono_tracks=bool(args.mono_tracks),
            )
        fig_s.tight_layout()
        args.out_skyplot_png.parent.mkdir(parents=True, exist_ok=True)
        fig_s.savefig(args.out_skyplot_png, dpi=150, bbox_inches="tight")
        plt.close(fig_s)
        print(f"Wrote {args.out_skyplot_png}")

    if args.out_csv is not None:
        import csv

        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["id", "lat", "lon", "az_deg", "el_deg"])
            w.writeheader()
            w.writerows(rows_csv)
        print(f"Wrote {args.out_csv}")

    if args.out_mask_json is not None or args.out_mask_pkl is not None:
        import json
        import pickle

        readable, legacy = build_mask_payload(profiles, az_deg, source=str(args.source))
        if args.out_mask_json is not None:
            args.out_mask_json.parent.mkdir(parents=True, exist_ok=True)
            with open(args.out_mask_json, "w", encoding="utf-8") as fh:
                json.dump(readable, fh, indent=2)
            print(f"Wrote {args.out_mask_json}")
        if args.out_mask_pkl is not None:
            args.out_mask_pkl.parent.mkdir(parents=True, exist_ok=True)
            with open(args.out_mask_pkl, "wb") as fh:
                pickle.dump(legacy, fh, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Wrote {args.out_mask_pkl}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
