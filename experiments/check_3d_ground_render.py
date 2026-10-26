#!/usr/bin/env python3
"""Local smoke-check: render ONE 3D ground frame and verify terrain covers the camera.

Uses local Luigi Ferraris mesh+DEM by default. Writes a PNG and prints PASS/FAIL checks
(no full MP4).

Example::

  python experiments/check_3d_ground_render.py
  python experiments/check_3d_ground_render.py --view-half-m 100 --terrain-grid-n 128
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

from gnss_gpu.io.dem_native import NativeDemSampler, parse_geoid_correction_arg
from label_los_nlos_coords import _lla_deg_to_ecef

import animate_rx_skyline as sky


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--triangles",
        type=Path,
        default=_SCRIPT_DIR / "results" / "luigi_osm_triangles.npy",
    )
    p.add_argument(
        "--dem-path",
        type=Path,
        default=_SCRIPT_DIR / "results" / "dem_test_luigi.tif",
    )
    p.add_argument("--geoid-correction", default="egm96")
    p.add_argument("--lat", type=float, default=None, help="Override RX lat (default: mesh centroid)")
    p.add_argument("--lon", type=float, default=None, help="Override RX lon")
    p.add_argument("--alt", type=float, default=None, help="Override RX alt ellipsoidal [m]")
    p.add_argument("--view-half-m", type=float, default=100.0)
    p.add_argument("--terrain-grid-n", type=int, default=128)
    p.add_argument("--imagery-zoom", type=int, default=19)
    p.add_argument("--max-tris", type=int, default=12000)
    p.add_argument("--dpi", type=int, default=140)
    p.add_argument(
        "--out",
        type=Path,
        default=_SCRIPT_DIR / "results" / "_check_3d_ground.png",
    )
    return p.parse_args()


def _ecef_centroid_lla(tri: np.ndarray) -> tuple[float, float, float]:
    c = np.asarray(tri, dtype=np.float64).mean(axis=(0, 1))
    x, y, z = float(c[0]), float(c[1]), float(c[2])
    a = 6378137.0
    e2 = 6.69437999014e-3
    lon = math.atan2(y, x)
    pxy = math.hypot(x, y)
    lat = math.atan2(z, pxy * (1.0 - e2))
    for _ in range(8):
        n = a / math.sqrt(1.0 - e2 * math.sin(lat) ** 2)
        lat = math.atan2(z + e2 * n * math.sin(lat), pxy)
    n = a / math.sqrt(1.0 - e2 * math.sin(lat) ** 2)
    alt = pxy / math.cos(lat) - n
    return math.degrees(lat), math.degrees(lon), float(alt)


def main() -> int:
    args = _parse_args()
    if not Path(args.triangles).is_file():
        raise SystemExit(f"missing triangles: {args.triangles}")
    if not Path(args.dem_path).is_file():
        raise SystemExit(f"missing DEM: {args.dem_path}")

    tri = np.load(args.triangles)
    lat, lon, alt0 = _ecef_centroid_lla(tri)
    if args.lat is not None:
        lat = float(args.lat)
    if args.lon is not None:
        lon = float(args.lon)
    alt = float(args.alt) if args.alt is not None else alt0 + 1.5
    view = float(args.view_half_m)
    print(f"RX lat={lat:.6f} lon={lon:.6f} alt={alt:.2f}  view=±{view:.0f} m", flush=True)

    rx = _lla_deg_to_ecef(lat, lon, alt)
    mesh_enu = sky._local_tris_enu(
        tri, rx, radius_m=max(view * 1.6, 120.0), max_tris=int(args.max_tris)
    )
    if mesh_enu is None or mesh_enu.size == 0:
        raise SystemExit("no mesh triangles near RX")

    sampler = NativeDemSampler(
        args.dem_path, geoid_correction=parse_geoid_correction_arg(str(args.geoid_correction))
    )
    terrain_tex = sky._terrain_enu_textured(
        sampler,
        lat0=lat,
        lon0=lon,
        rx_ecef=rx,
        half_m=max(view, 60.0),
        n=int(args.terrain_grid_n),
        zoom=int(args.imagery_zoom),
    )
    if terrain_tex is None:
        raise SystemExit("terrain+imagery failed")

    mesh_enu, shift = sky._align_mesh_enu_to_dem(mesh_enu, terrain_tex)
    print(f"mesh align ΔU={shift:+.2f} m", flush=True)

    Ee, Nn, Uu, fcols = terrain_tex
    terr_span = max(float(np.nanmax(np.abs(Ee))), float(np.nanmax(np.abs(Nn))))
    mesh_before = mesh_enu.shape[0]
    mesh_clip = sky._clip_mesh_enu_to_lim(mesh_enu, view)
    cents = mesh_clip.mean(axis=1)
    mesh_rmax = float(np.hypot(cents[:, 0], cents[:, 1]).max()) if mesh_clip.size else 0.0

    checks = []
    ok_span = terr_span >= view * 0.98
    checks.append(("terrain_span_covers_view", ok_span, f"span=±{terr_span:.1f} m vs view=±{view:.1f} m"))
    # Centroids may sit slightly outside; overhang is OK. Just ensure we kept something.
    ok_clip = mesh_clip.shape[0] >= max(10, mesh_before // 20)
    checks.append(
        (
            "mesh_keeps_visible_buildings",
            ok_clip,
            f"rmax={mesh_rmax:.1f} m  tris {mesh_before}→{mesh_clip.shape[0]} (overhang OK)",
        )
    )
    ok_tex = fcols.shape[0] >= int(args.terrain_grid_n) - 2
    checks.append(("texture_grid", ok_tex, f"facecolors={fcols.shape}"))

    print("[check] baking one view…", flush=True)
    frames, _views = sky.prebake_3d_orbit(
        mesh_enu=mesh_enu,
        terrain_tex=terrain_tex,
        map_half_m=view,
        view_half_m=view,
        n_views=1,
        dpi=int(args.dpi),
    )
    rgba = frames[0]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(out, rgba)
    print(f"[check] wrote {out}  shape={rgba.shape}", flush=True)

    # Pixel coverage over the whole cropped frame (should be mostly scene, not black pad).
    rgb = rgba[:, :, :3].astype(np.float32)
    lum = rgb.mean(axis=2)
    bright = float((lum > 25.0).mean())
    # A square patch in perspective is a diamond in an AABB → bright_frac≈0.45–0.55 is normal.
    ok_fill = bright >= 0.40
    checks.append(("frame_filled", ok_fill, f"bright_frac={bright:.2f} (want ≥0.40)"))

    # Texture should span most of the frame width (not a tiny central stamp).
    mid = lum[int(0.25 * lum.shape[0]) : int(0.75 * lum.shape[0]), :]
    col_hit = mid.mean(axis=0) > 40.0
    if col_hit.any():
        c0, c1 = int(np.where(col_hit)[0][0]), int(np.where(col_hit)[0][-1])
        width_frac = (c1 - c0 + 1) / max(1, lum.shape[1])
    else:
        width_frac = 0.0
    ok_width = width_frac >= 0.80
    checks.append(("sat_ground_width", ok_width, f"width_frac={width_frac:.2f} (want ≥0.80)"))

    print("\n=== RESULTS ===")
    failed = 0
    for name, ok, detail in checks:
        flag = "PASS" if ok else "FAIL"
        if not ok:
            failed += 1
        print(f"  {flag}  {name}: {detail}")
    print(f"\nPNG: {out.resolve()}")
    if failed:
        print(f"FAILED ({failed} checks)")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
