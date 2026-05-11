#!/usr/bin/env python3
"""Prepare a trajectory CSV for ``build_3d_visualization.py`` and run the LOS viewer.

Your planner CSV has lat/lon (degrees WGS84). Optionally each row may include an **ellipsoidal**
height column (WGS84 ``h`` in metres); if absent, ``--alt-m`` applies to every row.

This script:

  1. Converts each point to ECEF using per-row or default ellipsoidal height.
  2. Assigns synthetic GPS times: ``GPS TOW`` advances by ``--dt-s`` per row (wrapped in-week).
  3. Writes an UrbanNav-compatible reference CSV (``GPS Week`` + ``GPS TOW`` + ECEF).
  4. Optionally runs ``build_3d_visualization.py`` with ``--triangles-npy`` (your stadium mesh in ECEF).

Mesh requirement
----------------
Ray tracing uses ``triangles.npy`` shape ``[N,3,3]`` in **ECEF metres**, aligned with WGS84 —
same convention as the HK pipeline (``prepare_*_mesh.py``). A raw Cesium ``tileset.json``
folder is **not** consumed directly by this script; convert your stadium geometry to ``.npy``
first (e.g. export glTF from your toolchain and transform vertices to ECEF).

Example (Genoa, after you have ``ferraris_triangles.npy`` and ``BRDC*.rnx``)::

    python experiments/build_stadium_visibility_viz.py ^
      --latlon-csv \"C:/Users/Me/Desktop/luigiFerrarisTrajectory.csv\" ^
      --triangles-npy \"C:/Users/Me/Desktop/ferraris_triangles.npy\" ^
      --nav \"C:/Users/Me/Desktop/BRDC00IGS_R_20240890000_01D_MN.rnx\" ^
      --out-html \"C:/Users/Me/Desktop/luigi_ferraris_viz.html\" ^
      --alt-m 58 ^
      --gps-week 2318 ^
      --tow-start-s 43200 ^
      --dt-s 1 ^
      --area-name LuigiFerraris ^
      --cesium-ion-token \"$CESIUM_ION_TOKEN\"

Use ``--dry-run`` to only write the reference CSV and print the suggested command.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _normalize_row(row: dict) -> dict:
    return {k.strip(): v for k, v in row.items()}


def _detect_lat_lon(keys: set[str]) -> tuple[str, str]:
    kl = {k.strip().lower(): k for k in keys}
    lat_candidates = ("latitude", "lat", "latitudine", "lat_deg", "phi")
    lon_candidates = ("longitude", "lon", "lng", "longitudine", "lon_deg", "lambda")
    lat_k = next((kl[c] for c in lat_candidates if c in kl), None)
    lon_k = next((kl[c] for c in lon_candidates if c in kl), None)
    if lat_k is None or lon_k is None:
        raise ValueError(
            f"Could not find latitude/longitude columns in CSV header: {sorted(keys)}"
        )
    return lat_k, lon_k


def _detect_optional_alt_key(keys: set[str]) -> str | None:
    """Return CSV header key for WGS84 ellipsoidal height [m] if present."""
    kl = {k.strip().lower(): k for k in keys}
    candidates = (
        "altitudine ellissoidale (m)",
        "alt ellipsoidale (m)",
        "quota ellissoidica (m)",
        "altitudine (m)",
        "ellipsoidal height (m)",
        "height (m)",
        "altitude (m)",
        "altitude",
        "alt_m",
        "alt",
        "h_ellipsoid",
        "ellipsoidal_height",
    )
    return next((kl[c] for c in candidates if c in kl), None)


def _lla_to_ecef(lat_deg: float, lon_deg: float, alt_m: float) -> tuple[float, float, float]:
    a = 6378137.0
    f = 1.0 / 298.257223563
    e2 = 2.0 * f - f * f
    lat = math.radians(float(lat_deg))
    lon = math.radians(float(lon_deg))
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)
    n = a / math.sqrt(1.0 - e2 * sin_lat * sin_lat)
    x = (n + float(alt_m)) * cos_lat * cos_lon
    y = (n + float(alt_m)) * cos_lat * sin_lon
    z = (n * (1.0 - e2) + float(alt_m)) * sin_lat
    return float(x), float(y), float(z)


def _unwrap_tow(tow_s: float) -> float:
    t = float(tow_s)
    while t < 0.0:
        t += 604800.0
    while t >= 604800.0:
        t -= 604800.0
    return t


def _gps_week_tow_from_utc(dt_utc: datetime) -> tuple[int, float]:
    dt_utc = dt_utc.astimezone(timezone.utc)
    gps_epoch = datetime(1980, 1, 6, tzinfo=timezone.utc)
    secs = (dt_utc - gps_epoch).total_seconds()
    return int(secs // 604800), secs % 604800.0


def read_lat_lon_alt_csv(path: Path, *, default_alt_m: float) -> tuple[list[tuple[float, float, float]], bool]:
    """Return ``[(lat, lon, alt_m), ...]`` and whether altitudes came from the CSV."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = [_normalize_row(r) for r in csv.DictReader(f)]
    if not rows:
        raise RuntimeError(f"No data rows in {path}")
    hdr = set(rows[0].keys())
    lat_k, lon_k = _detect_lat_lon(hdr)
    alt_k = _detect_optional_alt_key(hdr)
    out: list[tuple[float, float, float]] = []
    for r in rows:
        lat = float(r[lat_k])
        lon = float(r[lon_k])
        if alt_k is not None and str(r.get(alt_k, "")).strip() != "":
            h_m = float(r[alt_k])
        else:
            h_m = float(default_alt_m)
        out.append((lat, lon, h_m))
    per_row = alt_k is not None
    return out, per_row


def write_urbannav_reference_csv(
    *,
    out_path: Path,
    lat_lon_alt: list[tuple[float, float, float]],
    gps_week: int,
    tow_start_s: float,
    dt_s: float,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tow0 = _unwrap_tow(tow_start_s)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "GPS Week",
                "GPS TOW (s)",
                "ECEF X (m)",
                "ECEF Y (m)",
                "ECEF Z (m)",
            ]
        )
        for i, (lat, lon, alt_m) in enumerate(lat_lon_alt):
            tow = _unwrap_tow(tow0 + i * float(dt_s))
            x, y, z = _lla_to_ecef(lat, lon, alt_m)
            w.writerow([int(gps_week), f"{tow:.6f}", f"{x:.6f}", f"{y:.6f}", f"{z:.6f}"])


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--latlon-csv", type=Path, required=True, help="CSV with lat/lon columns (degrees)")
    p.add_argument(
        "--reference-out",
        type=Path,
        default="",
        help="Where to write UrbanNav-style reference CSV (default: beside --out-html)",
    )
    p.add_argument(
        "--alt-m",
        type=float,
        default=58.0,
        help="Default WGS84 ellipsoidal height h [m] when the CSV has no altitude column "
        "(Italy NW coast / Genoa ground-level is often ~50–65 m; tune vs terrain).",
    )
    p.add_argument("--gps-week", type=int, default=0, help="GPS week number (default: use --start-utc)")
    p.add_argument("--tow-start-s", type=float, default=0.0, help="GPS TOW for row 0 [s] (default: use --start-utc)")
    p.add_argument("--dt-s", type=float, default=1.0, help="Synthetic time step between CSV rows [s]")
    p.add_argument(
        "--start-utc",
        type=str,
        default="",
        help="ISO datetime UTC for first row (e.g. 2024-03-29T12:00:00Z). Overrides --gps-week/--tow-start-s.",
    )
    p.add_argument("--triangles-npy", type=Path, required=True, help="Stadium mesh triangles.npy (ECEF)")
    p.add_argument("--nav", type=Path, required=True, help="RINEX navigation file (.rnx / .nav)")
    p.add_argument("--out-html", type=Path, required=True, help="Output HTML path")
    p.add_argument("--area-name", type=str, default="Stadium")
    p.add_argument("--dry-run", action="store_true", help="Only write reference CSV; do not run viz")
    p.add_argument("--python", type=str, default=sys.executable, help="Python executable for subprocess")
    p.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])

    viz = p.add_argument_group("Forwarded to build_3d_visualization.py")
    viz.add_argument("--n-epochs", type=int, default=500)
    viz.add_argument("--traj-step", type=float, default=1.0)
    viz.add_argument("--elevation-mask-deg", type=float, default=10.0)
    viz.add_argument("--eph-batch-chunk", type=int, default=64)
    viz.add_argument("--epoch-min-interval-s", type=float, default=0.0)
    viz.add_argument("--cesium-ion-token", type=str, default="")
    viz.add_argument("--viz-multipath", action="store_true")
    viz.add_argument("--atmo-bending-lite", action="store_true")
    viz.add_argument("--export-mesh-glb", action="store_true")
    viz.add_argument("--plateau-glb-radius-m", type=float, default=800.0)
    viz.add_argument("--plateau-glb-max-tris", type=int, default=800_000)

    return p.parse_args()


def main() -> None:
    args = _parse_args()
    lat_lon_alt, alt_from_csv = read_lat_lon_alt_csv(
        args.latlon_csv.resolve(), default_alt_m=float(args.alt_m)
    )
    lats = [p[0] for p in lat_lon_alt]
    lons = [p[1] for p in lat_lon_alt]
    hs = [p[2] for p in lat_lon_alt]
    print(
        f"[stadium] loaded trajectory CSV: {args.latlon_csv.resolve()} "
        f"(points={len(lat_lon_alt)}, lat=[{min(lats):.6f},{max(lats):.6f}], lon=[{min(lons):.6f},{max(lons):.6f}], "
        f"ellipsoid_h=[{min(hs):.2f},{max(hs):.2f}] m, alt={'CSV column' if alt_from_csv else 'default --alt-m'})",
        flush=True,
    )

    if args.start_utc.strip():
        dt = datetime.fromisoformat(args.start_utc.replace("Z", "+00:00"))
        gps_week, tow_start = _gps_week_tow_from_utc(dt)
        print(
            f"[stadium] start time from --start-utc={args.start_utc} -> GPS week={gps_week}, tow_start={tow_start:.3f}s",
            flush=True,
        )
    else:
        gps_week = int(args.gps_week)
        tow_start = float(args.tow_start_s)
        if gps_week <= 0:
            raise SystemExit("Provide --gps-week and --tow-start-s, or --start-utc.")
        print(
            f"[stadium] start time from explicit GPS week/tow -> week={gps_week}, tow_start={tow_start:.3f}s",
            flush=True,
        )

    ref_out = args.reference_out
    if not str(ref_out):
        ref_out = args.out_html.with_name(args.out_html.stem + "_reference.csv")

    write_urbannav_reference_csv(
        out_path=ref_out.resolve(),
        lat_lon_alt=lat_lon_alt,
        gps_week=gps_week,
        tow_start_s=tow_start,
        dt_s=float(args.dt_s),
    )
    tow_end = _unwrap_tow(tow_start + (len(lat_lon_alt) - 1) * float(args.dt_s))
    print(
        f"[stadium] wrote reference trajectory: {ref_out.resolve()} "
        f"(points={len(lat_lon_alt)}, dt={float(args.dt_s):g}s, tow=[{_unwrap_tow(tow_start):.3f},{tow_end:.3f}])",
        flush=True,
    )

    build_py = (args.repo_root / "experiments" / "build_3d_visualization.py").resolve()
    py_path = str(args.repo_root / "python")
    cmd = [
        str(args.python),
        str(build_py),
        "--area-name",
        str(args.area_name),
        "--reference-csv",
        str(ref_out.resolve()),
        "--triangles-npy",
        str(args.triangles_npy.resolve()),
        "--nav",
        str(args.nav.resolve()),
        "--out-html",
        str(args.out_html.resolve()),
        "--n-epochs",
        str(int(args.n_epochs)),
        "--traj-step",
        str(float(args.traj_step)),
        "--elevation-mask-deg",
        str(float(args.elevation_mask_deg)),
        "--eph-batch-chunk",
        str(int(args.eph_batch_chunk)),
        "--epoch-min-interval-s",
        str(float(args.epoch_min_interval_s)),
        "--plateau-glb-radius-m",
        str(float(args.plateau_glb_radius_m)),
        "--plateau-glb-max-tris",
        str(int(args.plateau_glb_max_tris)),
    ]
    if args.cesium_ion_token.strip():
        cmd += ["--cesium-ion-token", args.cesium_ion_token.strip()]
    if args.viz_multipath:
        cmd.append("--viz-multipath")
    if args.atmo_bending_lite:
        cmd.append("--atmo-bending-lite")
    if args.export_mesh_glb:
        cmd.append("--export-mesh-glb")

    print(f"[stadium] repo-root={args.repo_root.resolve()}", flush=True)
    print(f"[stadium] triangles-npy={args.triangles_npy.resolve()}", flush=True)
    print(f"[stadium] nav={args.nav.resolve()}", flush=True)
    print(f"[stadium] out-html={args.out_html.resolve()}", flush=True)
    print("\n[stadium] suggested viewer command:\n  " + " ".join(cmd), flush=True)

    if args.dry_run:
        print("[stadium] dry-run enabled: skipping viewer execution.", flush=True)
        return

    # Force a non-interactive backend in subprocesses (important in notebooks/headless runs).
    env_full = {
        **os.environ,
        "PYTHONPATH": py_path,
        "PYTHONUNBUFFERED": "1",
        "MPLBACKEND": "Agg",
    }
    print("[stadium] starting build_3d_visualization.py ...", flush=True)
    r = subprocess.run(cmd, env=env_full, cwd=str(args.repo_root))
    print(f"[stadium] viewer process exit code: {r.returncode}", flush=True)
    if r.returncode in (-11, 139):
        print(
            "[stadium] hint: segfault detected (-11/139). Try a lighter run: "
            "decimate triangles, reduce --n-epochs, and keep --viz-multipath/--export-mesh-glb disabled.",
            flush=True,
        )
    elif r.returncode in (-9, 137):
        print(
            "[stadium] hint: process killed (-9/137), often OOM. "
            "Use fewer triangles and lower --n-epochs.",
            flush=True,
        )
    raise SystemExit(r.returncode)


if __name__ == "__main__":
    main()
