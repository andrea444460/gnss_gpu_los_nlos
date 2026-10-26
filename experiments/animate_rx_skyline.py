#!/usr/bin/env python3
"""Presentation MP4: RX → sweeping skyline mask → obstruction skyplot.

Colab deps (install once)::

  !pip install -q imageio imageio-ffmpeg contextily pillow

Example (mask già calcolata)::

  python experiments/animate_rx_skyline.py \\
    --mask-json /content/los_work/results/rx_skyline_mask.json \\
    --triangles /content/los_work/results/genova_osm_triangles.npy \\
    --point-index 0 \\
    --dem-path /content/los_work/results/genova_area_dem.tif \\
    --geoid-correction egm96 \\
    --nav /content/los_work/data/BRDC00IGS_R_20262820000_01D_MN.rnx \\
    --utc 2026-10-09T12:00:00Z \\
    --mesh-radius-m 250 --max-tris 25000 \\
    --terrain-grid-n 96 --imagery-zoom 18 \\
    --out-mp4 /content/los_work/results/rx_mask_anim.mp4

Top-left 3D panel: DEM ground + Esri WorldImagery texture + OSM buildings (ENU).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.patches import FancyBboxPatch

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
from label_los_nlos_coords import _lla_deg_to_ecef, _parse_utc, _utc_to_gps_week_tow, load_coords
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from plot_rx_skyline import (
    _ecef_to_enu_basis,
    building_skyline_deg,
    collect_nav_tracks,
    draw_sat_tracks,
    terrain_skyline_deg,
)

BG = "#0b1220"
PANEL = "#121a2b"
ACCENT = "#3dd6c6"
ACCENT2 = "#f0a000"
TEXT = "#e8eef7"
MUTED = "#8b9bb4"
MASK = "#6b7c93"


def _setup_ffmpeg() -> None:
    """Prefer imageio-ffmpeg binary (works on Colab without apt)."""
    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        plt.rcParams["animation.ffmpeg_path"] = exe
        print("ffmpeg:", exe)
    except Exception as exc:
        print("imageio-ffmpeg non trovato (pip install imageio-ffmpeg):", exc)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mask-json", type=Path, default=None)
    p.add_argument("--point-id", type=str, default=None)
    p.add_argument("--point-index", type=int, default=0)
    p.add_argument("--coords", type=Path, default=None)
    p.add_argument("--triangles", type=Path, default=None)
    p.add_argument("--dem-path", type=Path, default=None)
    p.add_argument("--geoid-correction", default="egm96")
    p.add_argument("--source", choices=("buildings", "terrain", "both"), default="both")
    p.add_argument("--rx-alt-mode", choices=("fixed", "dem"), default="dem")
    p.add_argument("--rx-ant-height-m", type=float, default=1.5)
    p.add_argument("--default-alt-m", type=float, default=1.5)
    p.add_argument("--az-step-deg", type=float, default=1.0)
    p.add_argument("--fov-zero-deg", type=float, default=45.0)
    p.add_argument("--probe-m", type=float, default=8000.0)
    p.add_argument("--n-bisect", type=int, default=8)
    p.add_argument("--terrain-max-distance-m", type=float, default=20000.0)
    p.add_argument("--terrain-step-m", type=float, default=60.0)
    p.add_argument("--nav", type=Path, default=None)
    p.add_argument("--utc", type=str, default=None)
    p.add_argument("--duration-s", type=float, default=7200.0)
    p.add_argument("--dt-s", type=float, default=120.0)
    p.add_argument("--systems", type=str, default="G,R,E,C,J")
    p.add_argument("--map-half-m", type=float, default=500.0)
    p.add_argument("--mesh-radius-m", type=float, default=250.0, help="Local 3D mesh clip radius around RX")
    p.add_argument("--max-tris", type=int, default=25000, help="Max triangles in 3D panel")
    p.add_argument("--terrain-grid-n", type=int, default=96, help="DEM+imagery grid resolution for 3D ground")
    p.add_argument("--imagery-zoom", type=int, default=18, help="Esri WorldImagery zoom for 3D ground texture")
    p.add_argument("--no-terrain-texture", action="store_true", help="Skip satellite-textured DEM under buildings")
    p.add_argument("--basemap", action="store_true", help="2D fallback: Esri tiles via contextily (if no mesh)")
    p.add_argument("--no-mesh-3d", action="store_true", help="Force 2D map even if --triangles is set")
    p.add_argument("--out-mp4", type=Path, default=None)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--dpi", type=int, default=120)
    p.add_argument("--title", type=str, default="GNSS obstruction mask")
    p.add_argument("--self-check", action="store_true")
    return p.parse_args()


def _local_tris_enu(
    triangles_ecef: np.ndarray,
    rx_ecef: np.ndarray,
    *,
    radius_m: float,
    max_tris: int,
) -> np.ndarray | None:
    """Return [M,3,3] ENU triangles around RX, or None."""
    tri = np.asarray(triangles_ecef, dtype=np.float64)
    if tri.ndim != 3 or tri.shape[1:] != (3, 3) or tri.shape[0] == 0:
        return None
    rx = np.asarray(rx_ecef, dtype=np.float64).reshape(3)
    cents = tri.mean(axis=1)
    d = np.linalg.norm(cents - rx.reshape(1, 3), axis=1)
    sel = np.nonzero(d <= float(radius_m))[0]
    if sel.size == 0:
        # fallback: nearest max_tris
        sel = np.argsort(d)[: min(int(max_tris), d.size)]
    elif sel.size > int(max_tris):
        # keep closest
        sel = sel[np.argsort(d[sel])[: int(max_tris)]]
    east, north, up = _ecef_to_enu_basis(rx)
    R = np.column_stack([east, north, up])  # ECEF -> ENU columns
    local = tri[sel] - rx.reshape(1, 1, 3)
    # [M,3,3] @ R -> ENU
    enu = local @ R
    print(f"3D mesh panel: {enu.shape[0]} tris within {radius_m:.0f} m")
    return enu


def _dem_grid(sampler: NativeDemSampler, lat0: float, lon0: float, half_m: float, n: int = 100):
    dlat = half_m / 111_320.0
    dlon = half_m / (111_320.0 * max(1e-6, math.cos(math.radians(lat0))))
    lats = np.linspace(lat0 - dlat, lat0 + dlat, n)
    lons = np.linspace(lon0 - dlon, lon0 + dlon, n)
    Z = np.empty((n, n), dtype=np.float64)
    for i, la in enumerate(lats):
        for j, lo in enumerate(lons):
            try:
                Z[i, j] = float(sampler.sample(float(la), float(lo)))
            except Exception:
                Z[i, j] = np.nan
    return lons, lats, Z


def _terrain_enu_textured(
    sampler: NativeDemSampler,
    *,
    lat0: float,
    lon0: float,
    rx_ecef: np.ndarray,
    half_m: float,
    n: int = 96,
    zoom: int = 18,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """DEM patch in ENU + RGB facecolors from Esri World Imagery (Cesium-like ground).

    Returns (E, N, U, facecolors) with E/N/U shape (n,n) and facecolors (n-1,n-1,4).
    """
    try:
        import contextily as cx
        from PIL import Image
    except ImportError as exc:
        print("terrain texture needs contextily+pillow:", exc)
        return None

    lons, lats, H = _dem_grid(sampler, lat0, lon0, half_m, n=n)
    if not np.isfinite(H).any():
        return None

    rx = np.asarray(rx_ecef, dtype=np.float64).reshape(3)
    east, north, up = _ecef_to_enu_basis(rx)
    R = np.column_stack([east, north, up])
    E = np.zeros((n, n), dtype=np.float64)
    N_ = np.zeros((n, n), dtype=np.float64)
    U = np.zeros((n, n), dtype=np.float64)
    for i, la in enumerate(lats):
        for j, lo in enumerate(lons):
            h = float(H[i, j]) if np.isfinite(H[i, j]) else float(np.nanmedian(H))
            ecef = _lla_deg_to_ecef(float(la), float(lo), h)
            enu = (ecef - rx) @ R
            E[i, j], N_[i, j], U[i, j] = float(enu[0]), float(enu[1]), float(enu[2])

    west, east_ll = float(lons.min()), float(lons.max())
    south, north_ll = float(lats.min()), float(lats.max())
    try:
        img, _ext = cx.bounds2img(
            west,
            south,
            east_ll,
            north_ll,
            ll=True,
            source=cx.providers.Esri.WorldImagery,
            zoom=int(zoom),
        )
    except Exception as exc:
        print("satellite tile download failed:", exc)
        return None

    rgb = np.asarray(img)
    if rgb.ndim == 3 and rgb.shape[2] >= 3:
        rgb = rgb[:, :, :3]
    else:
        return None
    # plot_surface facecolors want (rows-1, cols-1, 3|4)
    tex = np.asarray(
        Image.fromarray(rgb.astype(np.uint8)).resize((n - 1, n - 1), Image.BILINEAR),
        dtype=np.float64,
    ) / 255.0
    # Flip north-south: bounds2img rows are typically north-at-top; our lats increase with i
    tex = np.flipud(tex)
    if tex.shape[2] == 3:
        alpha = np.full(tex.shape[:2] + (1,), 0.95, dtype=np.float64)
        facecolors = np.concatenate([tex, alpha], axis=2)
    else:
        facecolors = tex
    print(
        f"3D terrain+imagery: {n}x{n} DEM, tex={facecolors.shape}, "
        f"U=[{float(np.nanmin(U)):.1f},{float(np.nanmax(U)):.1f}] m"
    )
    return E, N_, U, facecolors


def _maybe_tracks(args, lat: float, lon: float, alt: float):
    if args.nav is None or not args.utc:
        return None
    systems = {s.strip().upper() for s in str(args.systems).split(",") if s.strip()}
    _w, tow0 = _utc_to_gps_week_tow(_parse_utc(args.utc))
    eph = Ephemeris(read_nav_rinex_multi(args.nav, systems=sorted(systems)))
    rx = _lla_deg_to_ecef(lat, lon, alt)
    return collect_nav_tracks(
        eph,
        rx,
        tow0=float(tow0),
        duration_s=float(args.duration_s),
        dt_s=float(args.dt_s),
        elev_mask_deg=0.0,
        systems=systems,
    )


def _load_or_compute(args: argparse.Namespace):
    if args.mask_json is not None:
        payload = json.loads(Path(args.mask_json).read_text(encoding="utf-8"))
        recs = payload["receivers"]
        if args.point_id is not None:
            rec = next(r for r in recs if str(r["id"]) == str(args.point_id))
        else:
            rec = recs[int(args.point_index) % len(recs)]
        az = np.arange(0.0, 360.0, dtype=np.float64)
        el = np.asarray(rec["masking_angle_deg"], dtype=np.float64)
        if el.size != 360:
            # resample if finer mask stored differently
            raise SystemExit(f"Expected masking_angle_deg len 360, got {el.size}")
        meta = {
            "id": str(rec["id"]),
            "lat": float(rec["lat_deg"]),
            "lon": float(rec["lon_deg"]),
            "alt": float(rec["alt_m"]),
        }
        return meta, az, el, _maybe_tracks(args, meta["lat"], meta["lon"], meta["alt"])

    if args.coords is None:
        raise SystemExit("Serve --mask-json oppure --coords (+ triangles/DEM)")
    points = load_coords(args.coords, float(args.default_alt_m))
    if args.point_id is not None:
        pt = next(p for p in points if str(p["id"]) == str(args.point_id))
    else:
        pt = points[int(args.point_index) % len(points)]

    geoid = parse_geoid_correction_arg(str(args.geoid_correction))
    dem_sampler = None
    if args.rx_alt_mode == "dem" or args.source in ("terrain", "both"):
        if args.dem_path is None:
            raise SystemExit("--dem-path required")
        dem_sampler = NativeDemSampler(args.dem_path, geoid_correction=geoid)
    if args.rx_alt_mode == "dem":
        assert dem_sampler is not None
        alt = float(dem_sampler.sample(pt["lat"], pt["lon"])) + float(args.rx_ant_height_m)
    else:
        alt = float(pt["alt"])
    rx = _lla_deg_to_ecef(pt["lat"], pt["lon"], alt)
    az = np.arange(0.0, 360.0, float(args.az_step_deg), dtype=np.float64)
    el = np.zeros_like(az)
    if args.source in ("buildings", "both"):
        if args.triangles is None:
            raise SystemExit("--triangles required")
        tri = np.load(args.triangles)
        bvh = BVHAccelerator(BuildingModel(tri).triangles)
        el = np.maximum(
            el,
            building_skyline_deg(bvh, rx, az, probe_m=float(args.probe_m), n_bisect=int(args.n_bisect)),
        )
    if args.source in ("terrain", "both"):
        terrain = TerrainHorizonMask(
            args.dem_path,
            HorizonConfig(
                max_distance_m=float(args.terrain_max_distance_m),
                sample_step_m=float(args.terrain_step_m),
            ),
            geoid_correction=str(args.geoid_correction),
        )
        el = np.maximum(el, terrain_skyline_deg(terrain, rx, az))
    meta = {"id": str(pt["id"]), "lat": float(pt["lat"]), "lon": float(pt["lon"]), "alt": float(alt)}
    return meta, az, el, _maybe_tracks(args, meta["lat"], meta["lon"], meta["alt"])


def build_animation(
    meta: dict,
    az_deg: np.ndarray,
    el_deg: np.ndarray,
    *,
    fov_zero_deg: float,
    dem_path: Path | None,
    geoid: str,
    map_half_m: float,
    basemap: bool,
    tracks: dict | None,
    title: str,
    fps: int,
    mesh_enu: np.ndarray | None = None,
    terrain_tex: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
):
    az = np.asarray(az_deg, dtype=np.float64)
    el = np.clip(np.asarray(el_deg, dtype=np.float64), 0.0, 90.0)
    fov0 = float(fov_zero_deg) % 360.0
    az_fov = np.mod(az - fov0, 360.0)
    order = np.argsort(az_fov)
    az_s, el_s, az_fov_s = az[order], el[order], az_fov[order]
    n = len(az_s)
    use_3d = (mesh_enu is not None and mesh_enu.size > 0) or (terrain_tex is not None)

    t_intro, t_map, t_sweep, t_tracks, t_hold = 1.6, 3.5, 9.5, 2.8, 2.2
    if not tracks:
        t_hold += t_tracks
        t_tracks = 0.0
    phase_frames = [max(1, int(round(d * fps))) for d in (t_intro, t_map, t_sweep, t_tracks, t_hold)]
    edges = np.cumsum([0] + phase_frames)
    n_frames = int(edges[-1])

    dem_img = extent = None
    if (not use_3d) and dem_path is not None and Path(dem_path).is_file():
        try:
            sampler = NativeDemSampler(dem_path, geoid_correction=parse_geoid_correction_arg(geoid))
            lons, lats, Z = _dem_grid(sampler, meta["lat"], meta["lon"], float(map_half_m), n=90)
            dem_img, extent = Z, [float(lons.min()), float(lons.max()), float(lats.min()), float(lats.max())]
        except Exception as exc:
            print("DEM backdrop skip:", exc)

    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": TEXT})
    fig = plt.figure(figsize=(12.8, 7.2), facecolor=BG)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1.0], hspace=0.30, wspace=0.22)
    if use_3d:
        ax_ctx = fig.add_subplot(gs[0, 0], projection="3d")
    else:
        ax_ctx = fig.add_subplot(gs[0, 0])
    ax_pol = fig.add_subplot(gs[0, 1], projection="polar")
    ax_fov = fig.add_subplot(gs[1, :])
    ax_fov.set_facecolor(PANEL)
    ax_pol.set_facecolor(PANEL)
    fig.suptitle(title, color=TEXT, fontsize=16, fontweight="bold", y=0.98)
    fig.text(
        0.5,
        0.935,
        f"RX {meta['id']}  ·  {meta['lat']:.5f}°, {meta['lon']:.5f}°  ·  h={meta['alt']:.1f} m",
        ha="center",
        color=MUTED,
        fontsize=11,
    )

    # --- Top-left: 3D mesh OR 2D map ---
    bear_line = rx_ring = None
    ray3d = None
    if use_3d:
        ax_ctx.set_facecolor("#0a1520")
        ax_ctx.set_title("3D scene (DEM + satellite + buildings)", color=TEXT, fontsize=11, pad=8)
        lim = float(map_half_m)
        zmin, zmax = -5.0, 40.0

        # Ground first (Cesium-like textured DEM)
        if terrain_tex is not None:
            Ee, Nn, Uu, fcols = terrain_tex
            ax_ctx.plot_surface(
                Ee,
                Nn,
                Uu,
                facecolors=fcols,
                linewidth=0,
                antialiased=False,
                shade=False,
                rstride=1,
                cstride=1,
                zorder=0,
            )
            lim = max(lim, float(np.nanmax(np.abs(Ee))), float(np.nanmax(np.abs(Nn))))
            zmin = min(zmin, float(np.nanmin(Uu)) - 2.0)
            zmax = max(zmax, float(np.nanmax(Uu)) + 5.0)

        if mesh_enu is not None and mesh_enu.size > 0:
            faces = [mesh_enu[i] for i in range(mesh_enu.shape[0])]
            coll = Poly3DCollection(
                faces,
                facecolors=(0.82, 0.86, 0.90, 0.72),
                edgecolors=(0.15, 0.18, 0.22, 0.35),
                linewidths=0.12,
            )
            ax_ctx.add_collection3d(coll)
            lim = max(lim, float(np.nanmax(np.abs(mesh_enu[:, :, :2]))) * 1.02)
            zmax = max(zmax, float(np.nanmax(mesh_enu[:, :, 2])) * 1.08)
            zmin = min(zmin, float(np.nanmin(mesh_enu[:, :, 2])) - 1.0)

        lim = max(lim, 40.0)
        ax_ctx.set_xlim(-lim, lim)
        ax_ctx.set_ylim(-lim, lim)
        ax_ctx.set_zlim(zmin, max(zmax, zmin + 25.0))
        ax_ctx.set_xlabel("East [m]", color=MUTED, labelpad=-2)
        ax_ctx.set_ylabel("North [m]", color=MUTED, labelpad=-2)
        ax_ctx.set_zlabel("Up [m]", color=MUTED, labelpad=-2)
        ax_ctx.tick_params(colors=MUTED, labelsize=7)
        ax_ctx.xaxis.pane.fill = False
        ax_ctx.yaxis.pane.fill = False
        ax_ctx.zaxis.pane.fill = False
        ax_ctx.xaxis.pane.set_edgecolor("#1a2740")
        ax_ctx.yaxis.pane.set_edgecolor("#1a2740")
        ax_ctx.zaxis.pane.set_edgecolor("#1a2740")
        ax_ctx.plot([0], [0], [1.5], "o", color=ACCENT, ms=7, zorder=10)
        (ray3d,) = ax_ctx.plot([0, 0], [0, 0], [1.5, 1.5], color=ACCENT2, lw=2.4, zorder=11)
        ax_ctx.view_init(elev=32, azim=-55)
        try:
            ax_ctx.set_box_aspect((1, 1, 0.45))
        except Exception:
            pass
        print(f"3D panel ready lim=±{lim:.0f} m  z=[{zmin:.1f},{zmax:.1f}] m")
    else:
        ax_ctx.set_facecolor(PANEL)
        ax_ctx.set_title("Receiver in context", color=TEXT, fontsize=11, pad=8)
        pad = float(map_half_m) / 111_320.0
        clat = max(1e-6, math.cos(math.radians(meta["lat"])))
        ax_ctx.set_xlim(meta["lon"] - pad / clat, meta["lon"] + pad / clat)
        ax_ctx.set_ylim(meta["lat"] - pad, meta["lat"] + pad)
        if dem_img is not None and extent is not None:
            ax_ctx.imshow(dem_img, origin="lower", extent=extent, cmap="terrain", alpha=0.75, aspect="auto", zorder=0)
            ax_ctx.set_xlim(extent[0], extent[1])
            ax_ctx.set_ylim(extent[2], extent[3])
        if basemap:
            try:
                import contextily as cx

                cx.add_basemap(
                    ax_ctx, crs="EPSG:4326", source=cx.providers.Esri.WorldImagery, zoom="auto", attribution=False
                )
                print("basemap: Esri.WorldImagery")
            except Exception as exc:
                print("basemap skip (pip install contextily):", exc)
        ax_ctx.set_xlabel("Longitude", color=MUTED)
        ax_ctx.set_ylabel("Latitude", color=MUTED)
        ax_ctx.tick_params(colors=MUTED)
        ax_ctx.grid(True, color="#ffffff22", lw=0.6)
        ax_ctx.plot([meta["lon"]], [meta["lat"]], "o", color=ACCENT, ms=10, zorder=5)
        (rx_ring,) = ax_ctx.plot(
            [meta["lon"]], [meta["lat"]], "o", mfc="none", mec=ACCENT, ms=18, mew=1.6, zorder=5
        )
        (bear_line,) = ax_ctx.plot([], [], color=ACCENT2, lw=2.2, zorder=4)
        ax_ctx.annotate(
            "RX", (meta["lon"], meta["lat"]), textcoords="offset points", xytext=(8, 8),
            color=ACCENT, fontsize=10, fontweight="bold",
        )

    # Polar
    ax_pol.set_theta_zero_location("N")
    ax_pol.set_theta_direction(-1)
    ax_pol.set_ylim(0, 90)
    ax_pol.set_yticks([0, 30, 60, 90])
    ax_pol.set_yticklabels(["90", "60", "30", "0"], color=MUTED, fontsize=8)
    ax_pol.tick_params(colors=MUTED)
    ax_pol.grid(True, color="#ffffff33", lw=0.7)
    ax_pol.set_title("Obstruction skyplot", color=TEXT, fontsize=11, pad=14)
    (pol_sky,) = ax_pol.plot([], [], color=ACCENT, lw=1.8, zorder=3)
    (pol_ray,) = ax_pol.plot([], [], color=ACCENT2, lw=2.0, zorder=4)
    ax_pol.plot([0], [0], "o", color="#ff5c5c", ms=5, zorder=5)
    pol_fill = None
    track_artists: list = []

    # FOV
    ax_fov.set_xlim(0, 360)
    ax_fov.set_ylim(0, max(55.0, float(np.nanmax(el_s)) * 1.2))
    zero_lbl = {0.0: "N", 45.0: "NE", 90.0: "E"}.get(fov0, f"{fov0:g}°")
    ax_fov.set_xlabel(f"360° FOV (Starting direction {zero_lbl})", color=MUTED)
    ax_fov.set_ylabel("LOS elevation [deg]", color=MUTED)
    ax_fov.set_title("Skyline mask (buildings ± terrain)", color=TEXT, fontsize=11, pad=8)
    ax_fov.tick_params(colors=MUTED)
    ax_fov.grid(True, color="#ffffff22", lw=0.6)
    (fov_line,) = ax_fov.plot([], [], color=ACCENT, lw=2.2)
    (fov_cursor,) = ax_fov.plot([], [], color=ACCENT2, lw=1.6)
    fov_fill = None

    status = fig.text(0.02, 0.02, "", color=MUTED, fontsize=10)
    intro_box = FancyBboxPatch(
        (0.18, 0.35), 0.64, 0.28, transform=fig.transFigure,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        facecolor="#152038", edgecolor=ACCENT, lw=1.5, zorder=20,
    )
    fig.patches.append(intro_box)
    intro_txt = fig.text(0.5, 0.52, "From receiver to sky mask", ha="center", va="center", color=TEXT, fontsize=20, fontweight="bold", zorder=21)
    intro_sub = fig.text(
        0.5, 0.44, "Ray-trace buildings + DEM horizon → GNSS obstruction mask",
        ha="center", va="center", color=MUTED, fontsize=12, zorder=21,
    )

    def _phase(frame: int):
        for i in range(5):
            if edges[i] <= frame < edges[i + 1]:
                return i, (frame - edges[i]) / max(1, phase_frames[i] - 1)
        return 4, 1.0

    def _bearing_end(az_from_n_deg: float):
        dist_m = min(0.75 * float(map_half_m), 350.0)
        dlat = (dist_m * math.cos(math.radians(az_from_n_deg))) / 111_320.0
        dlon = (dist_m * math.sin(math.radians(az_from_n_deg))) / (
            111_320.0 * max(1e-6, math.cos(math.radians(meta["lat"])))
        )
        return meta["lon"] + dlon, meta["lat"] + dlat

    def _ray_enu(az_from_n_deg: float, el_deg: float, length_m: float = 180.0):
        azr = math.radians(az_from_n_deg)
        elr = math.radians(el_deg)
        e = length_m * math.sin(azr) * math.cos(elr)
        n_ = length_m * math.cos(azr) * math.cos(elr)
        u = 1.5 + length_m * math.sin(elr)
        return e, n_, u

    def update(frame: int):
        nonlocal pol_fill, fov_fill
        phase, u = _phase(frame)
        show_intro = phase == 0
        intro_box.set_visible(show_intro)
        intro_txt.set_visible(show_intro)
        intro_sub.set_visible(show_intro)
        for ax in (ax_ctx, ax_pol, ax_fov):
            ax.set_visible(not show_intro)

        if rx_ring is not None:
            rx_ring.set_markersize(14 + 7 * (0.5 + 0.5 * math.sin(frame / max(1, fps) * 5.0)))

        # Slow orbit on 3D during locate + sweep
        if use_3d and phase >= 1:
            azim = -60 + (frame / max(1, n_frames - 1)) * 75.0
            elev = 26 + 6 * math.sin(frame / max(1, fps) * 0.7)
            ax_ctx.view_init(elev=elev, azim=azim)

        if phase == 0:
            status.set_text("Intro")
            return ()
        if phase == 1:
            status.set_text("1 · 3D city around RX" if use_3d else "1 · Locate receiver")
            if bear_line is not None:
                bear_line.set_data([], [])
            if ray3d is not None:
                ray3d.set_data_3d([0, 0], [0, 0], [1.5, 1.5])
            pol_sky.set_data([], [])
            pol_ray.set_data([], [])
            fov_line.set_data([], [])
            fov_cursor.set_data([], [])
            if pol_fill is not None:
                pol_fill.remove()
                pol_fill = None
            if fov_fill is not None:
                fov_fill.remove()
                fov_fill = None
            return ()

        prog = u if phase == 2 else 1.0
        if phase == 2:
            status.set_text("2 · Sweep azimuth · build skyline mask")
        elif phase == 3:
            status.set_text("3 · Satellite arcs over mask")
        else:
            status.set_text("4 · Final obstruction mask")

        k = min(n, max(1, int(round(prog * (n - 1))) + 1))
        az_now = float(az_s[k - 1])
        el_now = float(el_s[k - 1])
        if bear_line is not None:
            lon1, lat1 = _bearing_end(az_now)
            bear_line.set_data([meta["lon"], lon1], [meta["lat"], lat1])
        if ray3d is not None:
            e, nn, uu = _ray_enu(az_now, max(el_now, 5.0), length_m=min(0.85 * float(map_half_m), 200.0))
            ray3d.set_data_3d([0, e], [0, nn], [1.5, uu])

        th = np.radians(az_s[:k])
        rr = 90.0 - el_s[:k]
        pol_sky.set_data(th, rr)
        pol_ray.set_data([math.radians(az_now)] * 2, [0.0, 90.0 - el_now])
        if pol_fill is not None:
            pol_fill.remove()
        if k >= 2:
            r_hor = np.full_like(rr, 90.0)
            pol_fill = ax_pol.fill(
                np.concatenate([th, th[::-1]]),
                np.concatenate([rr, r_hor[::-1]]),
                color=MASK,
                alpha=0.55,
                zorder=1,
            )[0]

        fov_line.set_data(az_fov_s[:k], el_s[:k])
        fov_cursor.set_data([az_fov_s[k - 1]] * 2, [0.0, el_s[k - 1]])
        if fov_fill is not None:
            fov_fill.remove()
        if k >= 2:
            fov_fill = ax_fov.fill_between(az_fov_s[:k], 0.0, el_s[:k], color=MASK, alpha=0.35, zorder=1)

        if tracks and phase >= 3:
            for a in track_artists:
                try:
                    a.remove()
                except Exception:
                    pass
            track_artists.clear()
            alpha = 0.3 + 0.7 * (u if phase == 3 else 1.0)
            before = set(ax_pol.lines)
            draw_sat_tracks(ax_pol, tracks, mono_color=ACCENT2, label=False)
            for ln in ax_pol.lines:
                if ln not in before and ln not in (pol_sky, pol_ray):
                    ln.set_alpha(alpha)
                    track_artists.append(ln)
        return ()

    # Keep a FuncAnimation only for compatibility; we render frame-by-frame for progress logs.
    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps, blit=False)
    return fig, anim, n_frames, update


def _fmt_eta(seconds: float) -> str:
    s = max(0, int(round(seconds)))
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def save_animation_progress(
    fig: plt.Figure,
    update,
    *,
    n_frames: int,
    out: Path,
    fps: int,
    dpi: int,
    log_every: int = 10,
) -> Path:
    """Render frames one-by-one with remaining-frame + ETA logs."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    writers = []
    try:
        writers.append(
            (
                out,
                animation.FFMpegWriter(fps=fps, bitrate=4500, metadata={"artist": "gnss_gpu"}),
            )
        )
    except Exception:
        pass
    writers.append((out.with_suffix(".gif"), animation.PillowWriter(fps=fps)))

    last_err: Exception | None = None
    for path, writer in writers:
        try:
            print(f"[render] writer={type(writer).__name__}  frames={n_frames}  fps={fps}  → {path}")
            t0 = time.perf_counter()
            with writer.saving(fig, str(path), dpi=dpi):
                for i in range(n_frames):
                    t_frame = time.perf_counter()
                    update(i)
                    fig.canvas.draw()
                    writer.grab_frame()
                    dt = time.perf_counter() - t_frame
                    done = i + 1
                    left = n_frames - done
                    elapsed = time.perf_counter() - t0
                    avg = elapsed / done
                    eta = avg * left
                    if done == 1 or done == n_frames or done % max(1, log_every) == 0:
                        print(
                            f"[render] frame {done}/{n_frames}  left={left}  "
                            f"this={dt:.2f}s  avg={avg:.2f}s/frame  "
                            f"elapsed={_fmt_eta(elapsed)}  ETA={_fmt_eta(eta)}",
                            flush=True,
                        )
            print(f"[render] done {path} ({path.stat().st_size / 1e6:.2f} MB)", flush=True)
            return path
        except Exception as exc:
            last_err = exc
            print(f"[render] failed with {type(writer).__name__}: {exc}", flush=True)
            continue
    raise RuntimeError(f"Could not save animation: {last_err}")


def _self_check() -> None:
    _setup_ffmpeg()
    az = np.arange(360.0)
    el = 12.0 + 35.0 * (np.sin(np.radians(az - 45.0)) ** 2)
    meta = {"id": "demo", "lat": 44.42, "lon": 8.93, "alt": 50.0}
    fig, _anim, n_frames, update = build_animation(
        meta, az, el,
        fov_zero_deg=45.0, dem_path=None, geoid="none", map_half_m=400.0,
        basemap=False, tracks=None, title="Self-check", fps=12, mesh_enu=None, terrain_tex=None,
    )
    out = _SCRIPT_DIR / "results" / "_anim_selfcheck.mp4"
    out = save_animation_progress(fig, update, n_frames=min(n_frames, 24), out=out, fps=12, dpi=80, log_every=5)
    plt.close(fig)
    assert out.exists() and out.stat().st_size > 1000
    print("self-check OK:", out)


def main() -> int:
    args = _parse_args()
    if args.self_check:
        _self_check()
        return 0
    if args.out_mp4 is None:
        raise SystemExit("--out-mp4 is required (or pass --self-check)")

    _setup_ffmpeg()
    meta, az, el, tracks = _load_or_compute(args)
    print(
        f"RX {meta['id']}: {meta['lat']:.5f},{meta['lon']:.5f} h={meta['alt']:.1f} m  "
        f"mask max={float(np.max(el)):.1f}°  tracks={0 if not tracks else len(tracks)}"
    )

    rx = _lla_deg_to_ecef(meta["lat"], meta["lon"], meta["alt"])
    mesh_enu = None
    if args.triangles is not None and not args.no_mesh_3d and Path(args.triangles).is_file():
        try:
            tri = np.load(args.triangles)
            mesh_enu = _local_tris_enu(
                tri, rx, radius_m=float(args.mesh_radius_m), max_tris=int(args.max_tris)
            )
        except Exception as exc:
            print("3D mesh load skip:", exc)
    elif args.triangles is None:
        print("No --triangles → buildings skipped (DEM+imagery still possible)")

    terrain_tex = None
    if (
        not args.no_terrain_texture
        and args.dem_path is not None
        and Path(args.dem_path).is_file()
    ):
        try:
            sampler = NativeDemSampler(
                args.dem_path, geoid_correction=parse_geoid_correction_arg(str(args.geoid_correction))
            )
            half = float(args.mesh_radius_m)
            terrain_tex = _terrain_enu_textured(
                sampler,
                lat0=meta["lat"],
                lon0=meta["lon"],
                rx_ecef=rx,
                half_m=half,
                n=int(args.terrain_grid_n),
                zoom=int(args.imagery_zoom),
            )
        except Exception as exc:
            print("terrain+imagery skip:", exc)

    fig, _anim, n_frames, update = build_animation(
        meta, az, el,
        fov_zero_deg=float(args.fov_zero_deg),
        dem_path=args.dem_path,
        geoid=str(args.geoid_correction),
        map_half_m=float(args.mesh_radius_m),
        basemap=bool(args.basemap),
        tracks=tracks,
        title=str(args.title),
        fps=int(args.fps),
        mesh_enu=mesh_enu,
        terrain_tex=terrain_tex,
    )
    out = save_animation_progress(
        fig,
        update,
        n_frames=n_frames,
        out=Path(args.out_mp4),
        fps=int(args.fps),
        dpi=int(args.dpi),
        log_every=max(1, int(args.fps) // 2),  # ~2 logs/sec of video
    )
    plt.close(fig)
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
