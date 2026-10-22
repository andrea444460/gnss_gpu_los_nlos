#!/usr/bin/env python3
"""Skyline FOV + obstruction skyplot from receiver points (buildings ± DEM).

For each RX, sweeps azimuth 0..360° (0 = North) and finds the minimum
elevation where a ray clears the mesh (BVH binary search). Optionally
takes max with the DEM terrain horizon.

Writes one figure with polar skyplots (top) and linear FOV masks (bottom).
Same skyline data feeds both — polar is just ``r = 90 − el`` with gray fill
from skyline down to the horizon.

Related: ``build_skyplot_from_rinex.py`` / ``viz.plot_skyplot`` plot *satellites*;
this script plots the *obstruction mask* (no NAV needed).

Example (Genova Colab / local)::

  python experiments/plot_rx_skyline.py \\
    --coords /content/los_work/data/points.csv \\
    --triangles /content/los_work/results/genova_osm_triangles.npy \\
    --rx-alt-mode dem --dem-path /content/los_work/results/genova_area_dem.tif \\
    --geoid-correction egm96 \\
    --out-png /content/los_work/results/rx_skyline.png
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
from gnss_gpu.io.dem_native import NativeDemSampler, parse_geoid_correction_arg
from gnss_gpu.raytrace import BuildingModel
from gnss_gpu.terrain_horizon import HorizonConfig, TerrainHorizonMask
from label_los_nlos_coords import _lla_deg_to_ecef, load_coords

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


def draw_obstruction_skyplot(ax, az_deg: np.ndarray, el_skyline_deg: np.ndarray, title: str) -> None:
    """Polar skyplot: zenith center, N up, CW; gray = blocked below skyline."""
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
    p.add_argument("--out-csv", type=Path, default=None, help="Optional long CSV: id,az_deg,el_deg")
    p.add_argument("--ylim-deg", type=float, default=90.0)
    p.add_argument(
        "--self-check",
        action="store_true",
        help="Run mock BVH skyline check (no mesh/coords) and exit",
    )
    return p.parse_args()


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
    print(f"self-check OK (max|el-30°|={err:.3f}°)")


def main() -> int:
    args = _parse_args()
    if args.self_check:
        _self_check()
        return 0
    if args.coords is None or args.out_png is None:
        raise SystemExit("--coords and --out-png are required (or pass --self-check)")
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

    az_deg = np.arange(0.0, 360.0, float(args.az_step_deg), dtype=np.float64)
    rows_csv: list[dict] = []
    profiles: list[tuple[dict, np.ndarray]] = []

    for pt in points:
        if args.rx_alt_mode == "dem":
            assert dem_sampler is not None
            alt = float(dem_sampler.sample(pt["lat"], pt["lon"])) + float(args.rx_ant_height_m)
        else:
            alt = float(pt["alt"])
        rx = _lla_deg_to_ecef(pt["lat"], pt["lon"], alt)

        el = np.zeros_like(az_deg)
        if bvh is not None:
            el = np.maximum(el, building_skyline_deg(bvh, rx, az_deg, probe_m=float(args.probe_m), n_bisect=int(args.n_bisect)))
        if terrain is not None:
            el = np.maximum(el, terrain_skyline_deg(terrain, rx, az_deg))
        profiles.append((pt, el))
        print(
            f"  {pt['id']}: lat={pt['lat']:.5f} lon={pt['lon']:.5f} alt={alt:.1f} m  "
            f"skyline max={float(el.max()):.1f}° mean={float(el.mean()):.1f}°"
        )
        if args.out_csv is not None:
            for a, e in zip(az_deg, el):
                rows_csv.append({"id": pt["id"], "lat": pt["lat"], "lon": pt["lon"], "az_deg": float(a), "el_deg": float(e)})

    n = len(profiles)
    # Combined: polar skyplots on top, linear FOV below (same skyline data).
    fig = plt.figure(figsize=(max(3.0 * n, 5.0), 6.4))
    for i, (pt, el) in enumerate(profiles):
        ax_p = fig.add_subplot(2, n, i + 1, projection="polar")
        draw_obstruction_skyplot(
            ax_p, az_deg, el, title=f"Skyplot [{pt['lat']:.2f}, {pt['lon']:.2f}]"
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
            1, n, figsize=(max(3.2 * n, 4.0), 3.8), subplot_kw={"projection": "polar"}, squeeze=False
        )
        for ax, (pt, el) in zip(axes_s[0], profiles):
            draw_obstruction_skyplot(
                ax, az_deg, el, title=f"Skyplot - Elevation angle [°]\n[{pt['lat']:.2f}, {pt['lon']:.2f}]"
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
