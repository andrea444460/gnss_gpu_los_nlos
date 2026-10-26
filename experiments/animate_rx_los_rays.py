#!/usr/bin/env python3
"""Presentation MP4: LOS/NLOS ray tracing — one satellite at a time, then all.

Reuses the pre-baked 3D city panel from ``animate_rx_skyline`` and colours each
RX→sat ray green (LOS / ``check_los`` clear) or red (NLOS / building hit).

Example::

  python experiments/animate_rx_los_rays.py \\
    --mask-json /content/los_work/results/rx_skyline_mask.json \\
    --triangles /content/los_work/results/genova_osm_triangles.npy \\
    --dem-path /content/los_work/results/genova_area_dem.tif \\
    --geoid-correction egm96 \\
    --nav /content/los_work/data/BRDC00IGS_R_20262820000_01D_MN.rnx \\
    --utc 2026-10-09T12:00:00Z \\
    --point-index 0 \\
    --out-mp4 /content/los_work/results/rx_los_rays_p0.mp4
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
from matplotlib.patches import FancyBboxPatch, Patch

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
from gnss_gpu.urban_signal_sim import _sat_elevation_azimuth
from label_los_nlos_coords import _lla_deg_to_ecef, _parse_utc, _utc_to_gps_week_tow

import animate_rx_skyline as sky

LOS = "#3dd6c6"
NLOS = "#ff5c5c"
BG = sky.BG
PANEL = sky.PANEL
TEXT = sky.TEXT
MUTED = sky.MUTED
ACCENT = sky.ACCENT


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mask-json", type=Path, default=None)
    p.add_argument("--point-id", type=str, default=None)
    p.add_argument("--point-index", type=int, default=0)
    p.add_argument("--coords", type=Path, default=None)
    p.add_argument("--triangles", type=Path, required=False)
    p.add_argument("--dem-path", type=Path, default=None)
    p.add_argument("--geoid-correction", default="egm96")
    p.add_argument("--nav", type=Path, default=None)
    p.add_argument("--utc", type=str, default=None)
    p.add_argument("--systems", type=str, default="G,R,E,C,J")
    p.add_argument("--elevation-mask-deg", type=float, default=10.0)
    p.add_argument("--max-sats", type=int, default=14, help="Cap visible sats in the video")
    p.add_argument("--ray-length-m", type=float, default=160.0)
    p.add_argument("--mesh-radius-m", type=float, default=250.0)
    p.add_argument(
        "--view-half-m",
        type=float,
        default=100.0,
        help="3D camera half-extent [m] (zoom; smaller = closer to RX)",
    )
    p.add_argument("--max-tris", type=int, default=12000)
    p.add_argument("--terrain-grid-n", type=int, default=64)
    p.add_argument("--imagery-zoom", type=int, default=18)
    p.add_argument("--bake-views", type=int, default=36)
    p.add_argument("--no-terrain-texture", action="store_true")
    p.add_argument("--out-mp4", type=Path, default=None)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--dpi", type=int, default=120)
    p.add_argument("--title", type=str, default="LOS / NLOS ray tracing")
    p.add_argument("--self-check", action="store_true")
    return p.parse_args()


def _load_rx_meta(args: argparse.Namespace) -> dict:
    if args.mask_json is not None and Path(args.mask_json).is_file():
        meta = json.loads(Path(args.mask_json).read_text(encoding="utf-8"))
        recs = meta.get("receivers") or []
        if not recs:
            raise SystemExit("mask-json has no receivers")
        if args.point_id:
            rec = next((r for r in recs if str(r.get("id")) == str(args.point_id)), None)
            if rec is None:
                raise SystemExit(f"point-id {args.point_id!r} not in mask")
        else:
            rec = recs[int(args.point_index) % len(recs)]
        return {
            "id": str(rec.get("id", f"p{args.point_index}")),
            "lat": float(rec["lat_deg"]),
            "lon": float(rec["lon_deg"]),
            "alt": float(rec["alt_m"]),
        }
    if args.coords is None:
        raise SystemExit("Need --mask-json or --coords")
    from label_los_nlos_coords import load_coords

    pts = load_coords(args.coords, 1.5)
    pt = pts[int(args.point_index) % len(pts)]
    alt = float(pt["alt"])
    if args.dem_path is not None and Path(args.dem_path).is_file():
        dem = NativeDemSampler(
            args.dem_path, geoid_correction=parse_geoid_correction_arg(str(args.geoid_correction))
        )
        alt = float(dem.sample(pt["lat"], pt["lon"])) + 1.5
    return {"id": str(pt["id"]), "lat": float(pt["lat"]), "lon": float(pt["lon"]), "alt": alt}


def _collect_sats(
    *,
    rx_ecef: np.ndarray,
    nav: Path,
    utc: str,
    systems: set[str],
    elev_mask_deg: float,
    bvh: BVHAccelerator,
    max_sats: int,
) -> list[dict]:
    _w, tow = _utc_to_gps_week_tow(_parse_utc(utc))
    eph = Ephemeris(read_nav_rinex_multi(nav, systems=sorted(systems)))
    catalog = [
        str(p).strip().upper()
        for p in eph.available_prns
        if str(p).strip() and str(p).strip().upper()[0] in systems
    ]
    sat_ecef, _clk, used = eph.compute(float(tow), prn_list=catalog)
    if len(used) == 0:
        return []
    sats = np.asarray(sat_ecef, dtype=np.float64).reshape(-1, 3)
    el, az = _sat_elevation_azimuth(rx_ecef, sats)
    mask = float(elev_mask_deg) * math.pi / 180.0
    vis = np.where(el >= mask)[0]
    if vis.size == 0:
        return []
    los = np.asarray(bvh.check_los(rx_ecef, sats[vis]), dtype=bool).ravel()
    rows: list[dict] = []
    for j, i in enumerate(vis):
        rows.append(
            {
                "id": str(used[i]).strip().upper(),
                "az_deg": float(math.degrees(az[i])),
                "el_deg": float(math.degrees(el[i])),
                "is_los": bool(los[j]),
            }
        )
    rows.sort(key=lambda r: r["az_deg"])
    if len(rows) > int(max_sats):
        # Keep a spread across azimuth, not just the first N.
        idx = np.linspace(0, len(rows) - 1, int(max_sats)).round().astype(int)
        rows = [rows[i] for i in sorted(set(idx.tolist()))]
    return rows


def _tip_enu(az_deg: float, el_deg: float, length_m: float) -> tuple[float, float, float]:
    azr = math.radians(az_deg)
    elr = math.radians(el_deg)
    e = length_m * math.sin(azr) * math.cos(elr)
    n = length_m * math.cos(azr) * math.cos(elr)
    u = 1.5 + length_m * math.sin(elr)
    return e, n, u


def build_animation(
    meta: dict,
    sats: list[dict],
    *,
    mesh_enu: np.ndarray | None,
    terrain_tex: tuple | None,
    map_half_m: float,
    view_half_m: float,
    ray_length_m: float,
    bake_views: int,
    bake_dpi: int,
    title: str,
    fps: int,
):
    n_sat = len(sats)
    if n_sat == 0:
        raise SystemExit("No visible satellites to animate")

    t_intro, t_city, t_per, t_all, t_hold = 1.4, 2.0, 0.85, 3.2, 2.0
    phase_frames = [
        max(1, int(round(t_intro * fps))),
        max(1, int(round(t_city * fps))),
        max(1, int(round(t_per * fps))) * n_sat,
        max(1, int(round(t_all * fps))),
        max(1, int(round(t_hold * fps))),
    ]
    per_frames = max(1, int(round(t_per * fps)))
    edges = np.cumsum([0] + phase_frames)
    n_frames = int(edges[-1])

    print(f"[los] baking 3D panel for {n_sat} sats…", flush=True)
    baked_frames, baked_views = sky.prebake_3d_orbit(
        mesh_enu=mesh_enu,
        terrain_tex=terrain_tex,
        map_half_m=float(map_half_m),
        view_half_m=float(view_half_m),
        n_views=int(bake_views),
        dpi=int(bake_dpi),
    )

    tips = [_tip_enu(s["az_deg"], s["el_deg"], float(ray_length_m)) for s in sats]
    n_los = sum(1 for s in sats if s["is_los"])
    n_nlos = n_sat - n_los

    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": TEXT})
    fig = plt.figure(figsize=(12.8, 7.2), facecolor=BG)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1.0], hspace=0.30, wspace=0.22)
    ax_ctx = fig.add_subplot(gs[0, 0])
    ax_pol = fig.add_subplot(gs[0, 1], projection="polar")
    ax_log = fig.add_subplot(gs[1, :])
    ax_log.set_facecolor(PANEL)
    ax_pol.set_facecolor(PANEL)

    fig.suptitle(title, color=TEXT, fontsize=16, fontweight="bold", y=0.98)
    fig.text(
        0.5,
        0.935,
        f"RX {meta['id']}  ·  {meta['lat']:.5f}°, {meta['lon']:.5f}°  ·  h={meta['alt']:.1f} m  ·  "
        f"BVH check_los  ·  {n_los} LOS / {n_nlos} NLOS",
        ha="center",
        color=MUTED,
        fontsize=11,
    )

    bh, bw = baked_frames[0].shape[:2]
    ctx_im = ax_ctx.imshow(
        baked_frames[0],
        origin="upper",
        extent=(0, bw, bh, 0),
        aspect="auto",
        interpolation="bilinear",
        zorder=1,
    )
    ax_ctx.set_title("3D · ray RX → satellite", color=TEXT, fontsize=11, pad=8)
    ax_ctx.set_xlim(0, bw)
    ax_ctx.set_ylim(bh, 0)
    ax_ctx.set_xticks([])
    ax_ctx.set_yticks([])
    for spine in ax_ctx.spines.values():
        spine.set_visible(False)

    ray_lines = []
    for s in sats:
        (ln,) = ax_ctx.plot(
            [],
            [],
            color=LOS if s["is_los"] else NLOS,
            lw=2.4,
            solid_capstyle="round",
            zorder=5,
            alpha=0.95,
        )
        ray_lines.append(ln)

    ax_pol.set_theta_zero_location("N")
    ax_pol.set_theta_direction(-1)
    ax_pol.set_ylim(0, 90)
    ax_pol.set_yticks([0, 30, 60, 90])
    ax_pol.set_yticklabels(["90", "60", "30", "0"], color=MUTED, fontsize=8)
    ax_pol.tick_params(colors=MUTED)
    ax_pol.grid(True, color="#ffffff33", lw=0.7)
    ax_pol.set_title("Skyplot · LOS green / NLOS red", color=TEXT, fontsize=11, pad=14)
    ax_pol.plot([0], [0], "o", color="#ff5c5c", ms=5, zorder=5)
    scat_los = ax_pol.scatter([], [], c=LOS, s=70, zorder=6, edgecolors="white", linewidths=0.6)
    scat_nlos = ax_pol.scatter([], [], c=NLOS, s=70, marker="s", zorder=6, edgecolors="white", linewidths=0.6)
    ax_pol.legend(
        handles=[Patch(color=LOS, label="LOS"), Patch(color=NLOS, label="NLOS")],
        loc="lower left",
        fontsize=8,
        facecolor=PANEL,
        edgecolor="#2a3550",
        labelcolor=TEXT,
    )

    ax_log.set_xlim(0, 1)
    ax_log.set_ylim(0, 1)
    ax_log.axis("off")
    ax_log.set_title("Ray-trace log (buildings BVH)", color=TEXT, fontsize=11, pad=8)
    log_txt = ax_log.text(
        0.02,
        0.92,
        "",
        va="top",
        ha="left",
        color=TEXT,
        fontsize=11,
        family="DejaVu Sans Mono",
        linespacing=1.45,
        transform=ax_log.transAxes,
    )

    status = fig.text(0.02, 0.02, "", color=MUTED, fontsize=10)
    intro_box = FancyBboxPatch(
        (0.18, 0.35),
        0.64,
        0.28,
        transform=fig.transFigure,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        facecolor="#152038",
        edgecolor=ACCENT,
        lw=1.5,
        zorder=20,
    )
    fig.patches.append(intro_box)
    intro_txt = fig.text(
        0.5,
        0.52,
        "LOS / NLOS via ray tracing",
        ha="center",
        va="center",
        color=TEXT,
        fontsize=20,
        fontweight="bold",
        zorder=21,
    )
    intro_sub = fig.text(
        0.5,
        0.44,
        "Shoot a ray from RX to each satellite · green = clear · red = building hit",
        ha="center",
        va="center",
        color=MUTED,
        fontsize=12,
        zorder=21,
    )

    def _phase(frame: int):
        for i in range(5):
            if edges[i] <= frame < edges[i + 1]:
                return i, (frame - edges[i]) / max(1, phase_frames[i] - 1)
        return 4, 1.0

    def _set_skyplot(n_show: int):
        if n_show <= 0:
            scat_los.set_offsets(np.empty((0, 2)))
            scat_nlos.set_offsets(np.empty((0, 2)))
            return
        shown = sats[:n_show]
        los_pts = [(math.radians(s["az_deg"]), 90.0 - s["el_deg"]) for s in shown if s["is_los"]]
        nlos_pts = [(math.radians(s["az_deg"]), 90.0 - s["el_deg"]) for s in shown if not s["is_los"]]
        scat_los.set_offsets(np.array(los_pts) if los_pts else np.empty((0, 2)))
        scat_nlos.set_offsets(np.array(nlos_pts) if nlos_pts else np.empty((0, 2)))

    def _set_rays(n_show: int, view: dict):
        x0, y0 = sky._enu_to_bake_xy(0.0, 0.0, 1.5, view)
        for i, ln in enumerate(ray_lines):
            if i >= n_show:
                ln.set_data([], [])
                continue
            e, nn, uu = tips[i]
            x1, y1 = sky._enu_to_bake_xy(e, nn, uu, view)
            ln.set_data([x0, x1], [y0, y1])

    def _log_text(n_show: int) -> str:
        lines = []
        for i, s in enumerate(sats[:n_show]):
            tag = "LOS " if s["is_los"] else "NLOS"
            lines.append(
                f"{s['id']:<5}  az={s['az_deg']:6.1f}°  el={s['el_deg']:5.1f}°  →  {tag}"
            )
        if not lines:
            return "(waiting for first satellite…)"
        return "\n".join(lines[-10:])  # keep panel readable

    def update(frame: int):
        phase, u = _phase(frame)
        show_intro = phase == 0
        intro_box.set_visible(show_intro)
        intro_txt.set_visible(show_intro)
        intro_sub.set_visible(show_intro)
        for ax in (ax_ctx, ax_pol, ax_log):
            ax.set_visible(not show_intro)

        t = frame / max(1, n_frames - 1)
        bake_idx = min(len(baked_frames) - 1, int(t * (len(baked_frames) - 1)))
        ctx_im.set_data(baked_frames[bake_idx])
        view = baked_views[bake_idx]

        if phase == 0:
            status.set_text("Intro")
            return ()
        if phase == 1:
            status.set_text("1 · Receiver in the city mesh")
            _set_rays(0, view)
            _set_skyplot(0)
            log_txt.set_text("Ready to shoot rays through the building BVH…")
            return ()

        if phase == 2:
            # Reveal sats one-by-one (keep previous).
            local = frame - edges[2]
            n_show = min(n_sat, local // per_frames + 1)
            cur = sats[n_show - 1]
            tag = "LOS" if cur["is_los"] else "NLOS"
            status.set_text(f"2 · Ray {n_show}/{n_sat}: {cur['id']} → {tag}")
            _set_rays(n_show, view)
            _set_skyplot(n_show)
            log_txt.set_text(_log_text(n_show))
            return ()

        # All together + hold
        status.set_text(
            f"3 · All rays  ·  {n_los} LOS (clear) / {n_nlos} NLOS (blocked)"
            if phase == 3
            else f"4 · Final  ·  {n_los} LOS / {n_nlos} NLOS"
        )
        _set_rays(n_sat, view)
        _set_skyplot(n_sat)
        log_txt.set_text(_log_text(n_sat) + f"\n\nBVH check_los done · {n_los} clear, {n_nlos} obstructed")
        return ()

    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps, blit=False)
    return fig, anim, n_frames, update


def _self_check() -> None:
    sky._setup_ffmpeg()
    meta = {"id": "demo", "lat": 44.42, "lon": 8.93, "alt": 50.0}
    sats = [
        {"id": "G01", "az_deg": 40.0, "el_deg": 35.0, "is_los": True},
        {"id": "G12", "az_deg": 120.0, "el_deg": 22.0, "is_los": False},
        {"id": "E05", "az_deg": 210.0, "el_deg": 48.0, "is_los": True},
        {"id": "C06", "az_deg": 300.0, "el_deg": 18.0, "is_los": False},
    ]
    # Tiny fake bake: empty terrain, no mesh → still runs prebake with empty scene
    fig, _a, n_frames, update = build_animation(
        meta,
        sats,
        mesh_enu=None,
        terrain_tex=None,
        map_half_m=200.0,
        view_half_m=80.0,
        ray_length_m=90.0,
        bake_views=8,
        bake_dpi=72,
        title="Self-check LOS rays",
        fps=10,
    )
    out = _SCRIPT_DIR / "results" / "_anim_los_selfcheck.gif"
    out = sky.save_animation_progress(
        fig, update, n_frames=min(n_frames, 20), out=out, fps=10, dpi=72, log_every=5
    )
    plt.close(fig)
    assert out.exists() and out.stat().st_size > 500
    print("self-check OK:", out)


def main() -> int:
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
        raise SystemExit("--out-mp4 required (or --self-check)")
    if args.nav is None or args.utc is None:
        raise SystemExit("--nav and --utc required")
    if args.triangles is None or not Path(args.triangles).is_file():
        raise SystemExit("--triangles .npy required for BVH ray tracing")

    sky._setup_ffmpeg()
    meta = _load_rx_meta(args)
    rx = _lla_deg_to_ecef(meta["lat"], meta["lon"], meta["alt"])
    tri = np.load(args.triangles)
    bvh = BVHAccelerator(BuildingModel(tri).triangles)
    systems = {s.strip().upper() for s in str(args.systems).split(",") if s.strip()}
    sats = _collect_sats(
        rx_ecef=rx,
        nav=Path(args.nav),
        utc=str(args.utc),
        systems=systems,
        elev_mask_deg=float(args.elevation_mask_deg),
        bvh=bvh,
        max_sats=int(args.max_sats),
    )
    print(
        f"RX {meta['id']}: {len(sats)} sats above {args.elevation_mask_deg:g}°  "
        f"LOS={sum(s['is_los'] for s in sats)}  NLOS={sum(not s['is_los'] for s in sats)}",
        flush=True,
    )

    mesh_enu = sky._local_tris_enu(
        tri, rx, radius_m=float(args.mesh_radius_m), max_tris=int(args.max_tris)
    )
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
            terrain_tex = sky._terrain_enu_textured(
                sampler,
                lat0=meta["lat"],
                lon0=meta["lon"],
                rx_ecef=rx,
                half_m=float(args.mesh_radius_m),
                n=int(args.terrain_grid_n),
                zoom=int(args.imagery_zoom),
            )
        except Exception as exc:
            print("terrain+imagery skip:", exc)
    if mesh_enu is not None and terrain_tex is not None:
        mesh_enu = sky._align_mesh_enu_to_dem(mesh_enu, terrain_tex)

    fig, _anim, n_frames, update = build_animation(
        meta,
        sats,
        mesh_enu=mesh_enu,
        terrain_tex=terrain_tex,
        map_half_m=float(args.mesh_radius_m),
        view_half_m=float(args.view_half_m),
        ray_length_m=min(float(args.ray_length_m), 0.95 * float(args.view_half_m)),
        bake_views=int(args.bake_views),
        bake_dpi=min(110, int(args.dpi)),
        title=str(args.title),
        fps=int(args.fps),
    )
    out = sky.save_animation_progress(
        fig,
        update,
        n_frames=n_frames,
        out=Path(args.out_mp4),
        fps=int(args.fps),
        dpi=int(args.dpi),
        log_every=max(1, int(args.fps) // 2),
    )
    plt.close(fig)
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
