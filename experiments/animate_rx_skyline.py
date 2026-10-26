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
    --mesh-radius-m 250 --max-tris 12000 \\
    --terrain-grid-n 64 --imagery-zoom 18 --bake-views 36 \\
    --out-mp4 /content/los_work/results/rx_mask_anim.mp4

Top-left 3D panel: DEM + Esri WorldImagery + OSM buildings (ENU).
Tiles download once; orbit is pre-baked to bitmaps so encode is not ~3 fps 3D redraw.
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
from mpl_toolkits.mplot3d import proj3d
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
    p.add_argument(
        "--view-half-m",
        type=float,
        default=120.0,
        help="3D camera half-extent [m] around RX (zoom; smaller = closer). Mesh/DEM can be larger.",
    )
    p.add_argument("--max-tris", type=int, default=12000, help="Max triangles in 3D panel")
    p.add_argument(
        "--terrain-grid-n",
        type=int,
        default=112,
        help="DEM+imagery grid resolution for 3D ground (higher = sharper sat texture, slower bake)",
    )
    p.add_argument("--imagery-zoom", type=int, default=19, help="Esri WorldImagery zoom for 3D ground texture")
    p.add_argument("--bake-views", type=int, default=36, help="Pre-bake this many 3D orbit views (fast playback)")
    p.add_argument("--no-terrain-texture", action="store_true", help="Skip satellite-textured DEM under buildings")
    p.add_argument("--basemap", action="store_true", help="2D fallback: Esri tiles via contextily (if no mesh)")
    p.add_argument("--no-mesh-3d", action="store_true", help="Force 2D map even if --triangles is set")
    p.add_argument("--live-3d", action="store_true", help="Slow: redraw matplotlib 3D every frame (debug)")
    p.add_argument("--out-mp4", type=Path, default=None)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--dpi", type=int, default=120)
    p.add_argument("--title", type=str, default="GNSS obstruction mask")
    p.add_argument("--self-check", action="store_true")
    return p.parse_args()


def _align_mesh_enu_to_dem(
    mesh_enu: np.ndarray,
    terrain_tex: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    sit_on_top_m: float = 0.4,
) -> tuple[np.ndarray, float]:
    """Shift buildings in Up so footprint bottoms match DEM (viz; fixes geoid mismatch).

    Returns ``(mesh_enu_shifted, delta_u)`` so overlays (hit markers) can match.
    """
    Ee, Nn, Uu, _fcols = terrain_tex
    verts = np.asarray(mesh_enu, dtype=np.float64).reshape(-1, 3)
    if verts.size == 0 or not np.isfinite(Uu).any():
        return mesh_enu, 0.0
    u15 = float(np.nanpercentile(verts[:, 2], 15))
    ground = verts[verts[:, 2] <= u15 + 1.5]
    if ground.shape[0] < 30:
        ground = verts
    rng = np.random.default_rng(0)
    take = min(500, ground.shape[0])
    g = ground[rng.choice(ground.shape[0], size=take, replace=False)]
    e_flat, n_flat, u_flat = Ee.ravel(), Nn.ravel(), Uu.ravel()
    finite = np.isfinite(u_flat)
    e_flat, n_flat, u_flat = e_flat[finite], n_flat[finite], u_flat[finite]
    if e_flat.size == 0:
        return mesh_enu, 0.0
    deltas = np.empty(g.shape[0], dtype=np.float64)
    for i, (e, n, u) in enumerate(g):
        k = int(np.argmin((e_flat - e) ** 2 + (n_flat - n) ** 2))
        deltas[i] = float(u_flat[k]) - float(u)
    delta = float(np.median(deltas))
    shift = delta + float(sit_on_top_m)
    print(
        f"[3d] mesh↔DEM align: median ΔU={delta:+.1f} m "
        f"(buildings were {'under' if delta > 0 else 'above'} terrain) → shift +{sit_on_top_m:.1f} m seat",
        flush=True,
    )
    out = np.asarray(mesh_enu, dtype=np.float64).copy()
    out[:, :, 2] += shift
    return out, shift


def _draw_cesiumish_3d(
    ax,
    *,
    mesh_enu: np.ndarray | None,
    terrain_tex: tuple | None,
    map_half_m: float,
    view_half_m: float | None = None,
) -> tuple[float, float, float]:
    """Draw textured DEM + buildings on an Axes3D. Returns (lim, zmin, zmax).

    ``view_half_m`` forces the camera box (zoom). Content may extend past it.
    """
    content_lim = float(map_half_m)
    zmin, zmax = -5.0, 40.0
    # Matplotlib 3D depth sorting is broken with plot_surface; disable auto zorder
    # so buildings (higher zorder) are not painted under the DEM texture.
    try:
        ax.computed_zorder = False
    except Exception:
        pass
    if terrain_tex is not None:
        Ee, Nn, Uu, fcols = terrain_tex
        ax.plot_surface(
            Ee,
            Nn,
            Uu,
            facecolors=fcols,
            linewidth=0,
            antialiased=False,
            shade=False,
            rstride=1,
            cstride=1,
            zorder=1,
        )
        content_lim = max(content_lim, float(np.nanmax(np.abs(Ee))), float(np.nanmax(np.abs(Nn))))
        zmin = min(zmin, float(np.nanmin(Uu)) - 2.0)
        zmax = max(zmax, float(np.nanmax(Uu)) + 5.0)
    if mesh_enu is not None and mesh_enu.size > 0:
        faces = [mesh_enu[i] for i in range(mesh_enu.shape[0])]
        coll = Poly3DCollection(
            faces,
            facecolors=(0.82, 0.86, 0.90, 0.78),
            edgecolors=(0.12, 0.14, 0.18, 0.45),
            linewidths=0.15,
        )
        coll.set_zorder(5)
        ax.add_collection3d(coll)
        content_lim = max(content_lim, float(np.nanmax(np.abs(mesh_enu[:, :, :2]))) * 1.02)
        zmax = max(zmax, float(np.nanmax(mesh_enu[:, :, 2])) * 1.08)
        zmin = min(zmin, float(np.nanmin(mesh_enu[:, :, 2])) - 1.0)
    if view_half_m is not None and float(view_half_m) > 0:
        lim = max(float(view_half_m), 25.0)
    else:
        lim = max(content_lim, 40.0)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(zmin, max(zmax, zmin + 25.0))
    ax.set_facecolor("#0a1520")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor("#1a2740")
    ax.tick_params(colors=MUTED, labelsize=6)
    ax.set_xlabel("E", color=MUTED, labelpad=-4)
    ax.set_ylabel("N", color=MUTED, labelpad=-4)
    ax.set_zlabel("U", color=MUTED, labelpad=-4)
    ax.plot([0], [0], [1.5], "o", color=ACCENT, ms=6, zorder=10)
    try:
        ax.set_box_aspect((1, 1, 0.55 if lim <= 150 else 0.45))
    except Exception:
        pass
    return lim, zmin, zmax


def _enu_to_bake_xy(e: float, n: float, u: float, view: dict) -> tuple[float, float]:
    """Project ENU → pixel xy on baked RGBA (imshow origin=upper, extent=(0,w,h,0))."""
    xd, yd, _zd = proj3d.proj_transform(e, n, u, view["M"])
    v = view["trans"] @ np.array([xd, yd, 1.0], dtype=np.float64)
    # transData is display coords (y up from bottom); imshow upper has y down from top.
    return float(v[0]), float(view["h"] - v[1])


def prebake_3d_orbit(
    *,
    mesh_enu: np.ndarray | None,
    terrain_tex: tuple | None,
    map_half_m: float,
    view_half_m: float | None = None,
    n_views: int = 36,
    dpi: int = 110,
    elev: float = 32.0,
) -> tuple[list[np.ndarray], list[dict]]:
    """Render orbit once → RGBA frames + per-view proj (for orange ray overlay)."""
    n_views = max(8, int(n_views))
    vhalf = float(view_half_m) if view_half_m is not None else float(map_half_m)
    print(
        f"[3d] pre-bake {n_views} views  view=±{vhalf:.0f} m  "
        f"(satellite/DEM already loaded once; CPU rasterize, not re-download)…",
        flush=True,
    )
    t0 = time.perf_counter()
    fig = plt.figure(figsize=(5.6, 4.3), dpi=dpi, facecolor=BG)
    ax = fig.add_subplot(111, projection="3d")
    lim, _z0, _z1 = _draw_cesiumish_3d(
        ax,
        mesh_enu=mesh_enu,
        terrain_tex=terrain_tex,
        map_half_m=map_half_m,
        view_half_m=view_half_m,
    )
    frames: list[np.ndarray] = []
    views: list[dict] = []
    azims = np.linspace(-60.0, 25.0, n_views)
    # Slightly higher elev when zoomed so roofs don't dominate the frame.
    elev_use = float(elev) if lim > 150 else max(float(elev), 38.0)
    for i, azim in enumerate(azims):
        ax.view_init(elev=elev_use, azim=float(azim))
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba()).copy())
        w, h = fig.canvas.get_width_height()
        views.append(
            {
                "M": np.asarray(ax.get_proj(), dtype=np.float64).copy(),
                "trans": np.asarray(ax.transData.get_matrix(), dtype=np.float64).copy(),
                "w": int(w),
                "h": int(h),
            }
        )
        if (i + 1) % 9 == 0 or i + 1 == n_views:
            print(f"[3d] bake {i + 1}/{n_views}", flush=True)
    plt.close(fig)
    print(f"[3d] bake done in {time.perf_counter() - t0:.1f}s — playback will reuse these bitmaps", flush=True)
    return frames, views


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
    print(f"[imagery] download Esri tiles once for bbox ({n}x{n} DEM)…", flush=True)
    t_dl = time.perf_counter()
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
    print(f"[imagery] done in {time.perf_counter() - t_dl:.1f}s (cached for whole video)", flush=True)

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
    bake_views: int = 36,
    live_3d: bool = False,
    bake_dpi: int = 110,
    view_half_m: float | None = 120.0,
):
    az = np.asarray(az_deg, dtype=np.float64)
    el = np.clip(np.asarray(el_deg, dtype=np.float64), 0.0, 90.0)
    fov0 = float(fov_zero_deg) % 360.0
    az_fov = np.mod(az - fov0, 360.0)
    order = np.argsort(az_fov)
    az_s, el_s, az_fov_s = az[order], el[order], az_fov[order]
    n = len(az_s)
    has_3d = (mesh_enu is not None and mesh_enu.size > 0) or (terrain_tex is not None)
    # Default: bake orbit once → imshow (fast). --live-3d keeps slow matplotlib 3D.
    use_baked = has_3d and not live_3d
    use_live_3d = has_3d and live_3d

    t_intro, t_map, t_sweep, t_tracks, t_hold = 1.6, 3.5, 9.5, 2.8, 2.2
    if not tracks:
        t_hold += t_tracks
        t_tracks = 0.0
    phase_frames = [max(1, int(round(d * fps))) for d in (t_intro, t_map, t_sweep, t_tracks, t_hold)]
    edges = np.cumsum([0] + phase_frames)
    n_frames = int(edges[-1])

    baked_frames: list[np.ndarray] | None = None
    baked_views: list[dict] | None = None
    if use_baked:
        baked_frames, baked_views = prebake_3d_orbit(
            mesh_enu=mesh_enu,
            terrain_tex=terrain_tex,
            map_half_m=float(map_half_m),
            view_half_m=view_half_m,
            n_views=int(bake_views),
            dpi=int(bake_dpi),
        )

    dem_img = extent = None
    if (not has_3d) and dem_path is not None and Path(dem_path).is_file():
        try:
            sampler = NativeDemSampler(dem_path, geoid_correction=parse_geoid_correction_arg(geoid))
            lons, lats, Z = _dem_grid(sampler, meta["lat"], meta["lon"], float(map_half_m), n=90)
            dem_img, extent = Z, [float(lons.min()), float(lons.max()), float(lats.min()), float(lats.max())]
        except Exception as exc:
            print("DEM backdrop skip:", exc)

    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": TEXT})
    fig = plt.figure(figsize=(12.8, 7.2), facecolor=BG)
    gs = fig.add_gridspec(
        2, 2, height_ratios=[2.1, 0.75], width_ratios=[1.35, 1.0], hspace=0.22, wspace=0.16
    )
    if use_live_3d:
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

    # --- Top-left: baked 3D bitmaps / live 3D / 2D map ---
    bear_line = rx_ring = None
    ray3d = None
    ray2d = None
    ctx_im = None
    bake_idx = 0
    if use_baked and baked_frames:
        ax_ctx.set_facecolor("#0a1520")
        ax_ctx.set_title("3D scene (DEM + satellite + buildings)", color=TEXT, fontsize=11, pad=8)
        bh, bw = baked_frames[0].shape[:2]
        # buffer_rgba is top-row-first → origin=upper; extent y goes bh→0
        ctx_im = ax_ctx.imshow(
            baked_frames[0],
            origin="upper",
            extent=(0, bw, bh, 0),
            aspect="auto",
            interpolation="bilinear",
            zorder=1,
        )
        (ray2d,) = ax_ctx.plot([], [], color=ACCENT2, lw=2.6, solid_capstyle="round", zorder=5)
        ax_ctx.set_xlim(0, bw)
        ax_ctx.set_ylim(bh, 0)
        ax_ctx.set_xticks([])
        ax_ctx.set_yticks([])
        for spine in ax_ctx.spines.values():
            spine.set_visible(False)
        print(f"3D panel: {len(baked_frames)} pre-baked views + projected orange ray overlay")
    elif use_live_3d:
        ax_ctx.set_facecolor("#0a1520")
        ax_ctx.set_title("3D scene (DEM + satellite + buildings) [live]", color=TEXT, fontsize=11, pad=8)
        _draw_cesiumish_3d(
            ax_ctx,
            mesh_enu=mesh_enu,
            terrain_tex=terrain_tex,
            map_half_m=float(map_half_m),
            view_half_m=view_half_m,
        )
        (ray3d,) = ax_ctx.plot([0, 0], [0, 0], [1.5, 1.5], color=ACCENT2, lw=2.4, zorder=11)
        ax_ctx.view_init(elev=32, azim=-55)
        print("3D panel: live matplotlib (slow; omit --live-3d to bake)")
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

        # 3D orbit: swap pre-baked bitmap, or live view_init (slow)
        nonlocal bake_idx
        if has_3d and phase >= 1:
            t = frame / max(1, n_frames - 1)
            if ctx_im is not None and baked_frames:
                bake_idx = min(len(baked_frames) - 1, int(t * (len(baked_frames) - 1)))
                ctx_im.set_data(baked_frames[bake_idx])
            elif use_live_3d:
                azim = -60 + t * 75.0
                elev = 26 + 6 * math.sin(frame / max(1, fps) * 0.7)
                ax_ctx.view_init(elev=elev, azim=azim)

        if phase == 0:
            status.set_text("Intro")
            return ()
        if phase == 1:
            status.set_text("1 · 3D city around RX" if has_3d else "1 · Locate receiver")
            if bear_line is not None:
                bear_line.set_data([], [])
            if ray3d is not None:
                ray3d.set_data_3d([0, 0], [0, 0], [1.5, 1.5])
            if ray2d is not None:
                ray2d.set_data([], [])
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
        e, nn, uu = _ray_enu(az_now, max(el_now, 5.0), length_m=min(0.85 * float(map_half_m), 200.0))
        if ray3d is not None:
            ray3d.set_data_3d([0, e], [0, nn], [1.5, uu])
        if ray2d is not None and baked_views:
            view = baked_views[bake_idx]
            x0, y0 = _enu_to_bake_xy(0.0, 0.0, 1.5, view)
            x1, y1 = _enu_to_bake_xy(e, nn, uu, view)
            ray2d.set_data([x0, x1], [y0, y1])

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
    # Live logs even when stdout is piped (Colab subprocess).
    try:
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
        sys.stderr.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    except Exception:
        pass

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
            # Pack texels into the zoomed view (not the full mesh radius).
            half = min(
                float(args.mesh_radius_m),
                max(float(args.view_half_m) * 1.35, 80.0),
            )
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

    if mesh_enu is not None and terrain_tex is not None:
        mesh_enu, _ = _align_mesh_enu_to_dem(mesh_enu, terrain_tex)

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
        bake_views=int(args.bake_views),
        live_3d=bool(args.live_3d),
        bake_dpi=min(110, int(args.dpi)),
        view_half_m=float(args.view_half_m),
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
