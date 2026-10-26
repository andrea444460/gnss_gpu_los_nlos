#!/usr/bin/env python3
"""Presentation MP4: RX → sweeping skyline mask → obstruction skyplot.

Colab deps (install once)::

  !pip install -q imageio imageio-ffmpeg contextily pillow

Example (mask già calcolata)::

  python experiments/animate_rx_skyline.py \\
    --mask-json /content/los_work/results/rx_skyline_mask.json \\
    --point-index 0 \\
    --dem-path /content/los_work/results/genova_area_dem.tif \\
    --geoid-correction egm96 \\
    --nav /content/los_work/data/BRDC00IGS_R_20262820000_01D_MN.rnx \\
    --utc 2026-10-09T12:00:00Z \\
    --basemap \\
    --out-mp4 /content/los_work/results/rx_mask_anim.mp4
"""

from __future__ import annotations

import argparse
import json
import math
import sys
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
from plot_rx_skyline import (
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
    p.add_argument("--basemap", action="store_true", help="OSM/Esri tiles via contextily")
    p.add_argument("--out-mp4", type=Path, default=None)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--dpi", type=int, default=120)
    p.add_argument("--title", type=str, default="GNSS obstruction mask")
    p.add_argument("--self-check", action="store_true")
    return p.parse_args()


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
):
    az = np.asarray(az_deg, dtype=np.float64)
    el = np.clip(np.asarray(el_deg, dtype=np.float64), 0.0, 90.0)
    fov0 = float(fov_zero_deg) % 360.0
    az_fov = np.mod(az - fov0, 360.0)
    order = np.argsort(az_fov)
    az_s, el_s, az_fov_s = az[order], el[order], az_fov[order]
    n = len(az_s)

    t_intro, t_map, t_sweep, t_tracks, t_hold = 1.6, 3.2, 9.5, 2.8, 2.2
    if not tracks:
        t_hold += t_tracks
        t_tracks = 0.0
    phase_frames = [max(1, int(round(d * fps))) for d in (t_intro, t_map, t_sweep, t_tracks, t_hold)]
    edges = np.cumsum([0] + phase_frames)
    n_frames = int(edges[-1])

    dem_img = extent = None
    if dem_path is not None and Path(dem_path).is_file():
        try:
            sampler = NativeDemSampler(dem_path, geoid_correction=parse_geoid_correction_arg(geoid))
            lons, lats, Z = _dem_grid(sampler, meta["lat"], meta["lon"], float(map_half_m), n=90)
            dem_img, extent = Z, [float(lons.min()), float(lons.max()), float(lats.min()), float(lats.max())]
        except Exception as exc:
            print("DEM backdrop skip:", exc)

    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": TEXT})
    fig = plt.figure(figsize=(12.8, 7.2), facecolor=BG)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1.0], hspace=0.30, wspace=0.22)
    ax_map = fig.add_subplot(gs[0, 0])
    ax_pol = fig.add_subplot(gs[0, 1], projection="polar")
    ax_fov = fig.add_subplot(gs[1, :])
    for ax in (ax_map, ax_fov):
        ax.set_facecolor(PANEL)
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

    # Map
    ax_map.set_title("Receiver in context", color=TEXT, fontsize=11, pad=8)
    pad = float(map_half_m) / 111_320.0
    ax_map.set_xlim(meta["lon"] - pad / max(1e-6, math.cos(math.radians(meta["lat"]))), meta["lon"] + pad / max(1e-6, math.cos(math.radians(meta["lat"]))))
    ax_map.set_ylim(meta["lat"] - pad, meta["lat"] + pad)
    if dem_img is not None and extent is not None:
        ax_map.imshow(dem_img, origin="lower", extent=extent, cmap="terrain", alpha=0.75, aspect="auto", zorder=0)
        ax_map.set_xlim(extent[0], extent[1])
        ax_map.set_ylim(extent[2], extent[3])
    if basemap:
        try:
            import contextily as cx

            cx.add_basemap(ax_map, crs="EPSG:4326", source=cx.providers.Esri.WorldImagery, zoom="auto", attribution=False)
            print("basemap: Esri.WorldImagery")
        except Exception as exc:
            print("basemap skip (pip install contextily):", exc)
    ax_map.set_xlabel("Longitude", color=MUTED)
    ax_map.set_ylabel("Latitude", color=MUTED)
    ax_map.tick_params(colors=MUTED)
    ax_map.grid(True, color="#ffffff22", lw=0.6)
    ax_map.plot([meta["lon"]], [meta["lat"]], "o", color=ACCENT, ms=10, zorder=5)
    (rx_ring,) = ax_map.plot([meta["lon"]], [meta["lat"]], "o", mfc="none", mec=ACCENT, ms=18, mew=1.6, zorder=5)
    (bear_line,) = ax_map.plot([], [], color=ACCENT2, lw=2.2, zorder=4)
    ax_map.annotate("RX", (meta["lon"], meta["lat"]), textcoords="offset points", xytext=(8, 8), color=ACCENT, fontsize=10, fontweight="bold")

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
        # ~350 m ray on map
        dist_m = min(0.75 * float(map_half_m), 350.0)
        dlat = (dist_m * math.cos(math.radians(az_from_n_deg))) / 111_320.0
        dlon = (dist_m * math.sin(math.radians(az_from_n_deg))) / (
            111_320.0 * max(1e-6, math.cos(math.radians(meta["lat"])))
        )
        return meta["lon"] + dlon, meta["lat"] + dlat

    def update(frame: int):
        nonlocal pol_fill, fov_fill
        phase, u = _phase(frame)
        show_intro = phase == 0
        intro_box.set_visible(show_intro)
        intro_txt.set_visible(show_intro)
        intro_sub.set_visible(show_intro)
        for ax in (ax_map, ax_pol, ax_fov):
            ax.set_visible(not show_intro)

        rx_ring.set_markersize(14 + 7 * (0.5 + 0.5 * math.sin(frame / max(1, fps) * 5.0)))

        if phase == 0:
            status.set_text("Intro")
            return ()
        if phase == 1:
            status.set_text("1 · Locate receiver")
            bear_line.set_data([], [])
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
        lon1, lat1 = _bearing_end(az_now)
        bear_line.set_data([meta["lon"], lon1], [meta["lat"], lat1])

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

    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps, blit=False)
    return fig, anim, n_frames


def _self_check() -> None:
    _setup_ffmpeg()
    az = np.arange(360.0)
    el = 12.0 + 35.0 * (np.sin(np.radians(az - 45.0)) ** 2)
    meta = {"id": "demo", "lat": 44.42, "lon": 8.93, "alt": 50.0}
    fig, anim, _n = build_animation(
        meta, az, el,
        fov_zero_deg=45.0, dem_path=None, geoid="none", map_half_m=400.0,
        basemap=False, tracks=None, title="Self-check", fps=12,
    )
    out = _SCRIPT_DIR / "results" / "_anim_selfcheck.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        writer = animation.FFMpegWriter(fps=12, bitrate=2500)
        anim.save(out, writer=writer, dpi=80)
    except Exception:
        out = out.with_suffix(".gif")
        anim.save(out, writer=animation.PillowWriter(fps=12), dpi=80)
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
    fig, anim, n_frames = build_animation(
        meta, az, el,
        fov_zero_deg=float(args.fov_zero_deg),
        dem_path=args.dem_path,
        geoid=str(args.geoid_correction),
        map_half_m=float(args.map_half_m),
        basemap=bool(args.basemap),
        tracks=tracks,
        title=str(args.title),
        fps=int(args.fps),
    )
    out = Path(args.out_mp4)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Rendering {n_frames} frames → {out}")
    try:
        writer = animation.FFMpegWriter(fps=int(args.fps), bitrate=4500, metadata={"artist": "gnss_gpu"})
        anim.save(out, writer=writer, dpi=int(args.dpi))
    except Exception as exc:
        print("MP4 failed, GIF fallback:", exc)
        out = out.with_suffix(".gif")
        anim.save(out, writer=animation.PillowWriter(fps=int(args.fps)), dpi=int(args.dpi))
    plt.close(fig)
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
