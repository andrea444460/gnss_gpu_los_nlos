#!/usr/bin/env python3
"""Colab pipeline: geometry-based LOS/NLOS ray tracing for permanent station TERF.

TERF (marker ``TERF``, 9-char ``TERF00CYP``) is a fixed GNSS station in Cyprus.
This script prepares mesh + reference track + per-day LOS/NLOS labels using
``export_los_labels_from_urbannav.py`` (BVH ray tracing on OSM buildings).

Run on Google Colab with **GPU runtime** (T4 is enough).

Quick start on Colab
--------------------

1. Setup (first notebook cells)::

    # Runtime → Change runtime type → GPU
    !nvidia-smi
    !git clone https://github.com/YOUR_USER/gnss_gpu.git /content/gnss_gpu
    %cd /content/gnss_gpu
    !pip install -q numpy matplotlib folium branca pyproj rasterio requests scipy
    !apt-get -qq install -y cmake
    !mkdir -p build && cd build && cmake .. -DCMAKE_CUDA_ARCHITECTURES=75 && make -j$(nproc)

2. Upload RINEX OBS files (``TERF00CYP_*_MO.rnx``) to ``/content/terf_work/data/``.

3. Run::

    import os
    os.environ["PYTHONPATH"] = "python:build"
    !python experiments/run_terf_permanent_station_colab.py \\
        --work-dir /content/terf_work \\
        --phase all

Phases
------

- ``setup``   — environment checklist
- ``mesh``    — fetch OSM buildings around station → ``terf_osm_triangles.npy``
- ``nav``     — download matching daily BRDC (IGS) for each OBS day
- ``labels``  — fixed-position reference + LOS/NLOS CSV per OBS file
- ``viz``     — Cesium 3D viewer: per-epoch rays for OBS-tracked satellites only
- ``summary`` — aggregate stats JSON across produced label files
- ``all``     — mesh → nav → labels → (optional viz) → summary

Data layout under ``--work-dir``::

    data/TERF00CYP_R_20261920000_01D_30S_MO.rnx   # user OBS (upload)
    data/BRDC00IGS_R_20261920000_01D_MN.rnx       # auto-downloaded NAV
    results/terf_osm_triangles.npy
    results/reference_2026192.csv
    results/terf_los_labels_2026192_gc.csv
    results/terf_los_viz_2026192.html
    results/terf_pipeline_summary.json
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_EXPERIMENTS = _ROOT / "experiments"

# TERF00CYP approximate position (from RINEX header); used for OSM bbox if OBS not parsed yet.
TERF_ECEF_DEFAULT = (4367788.7241, 2893757.1063, 3625318.6571)
TERF_BBOX_BUFFER_DEG = 0.015  # ~1.5 km

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = 2.0 * WGS84_F - WGS84_F * WGS84_F

_OBS_NAME_RE = re.compile(
    r"^(?P<station>[A-Z0-9]+)_R_(?P<year>\d{4})(?P<doy>\d{3})\d+_01D_.*\.rnx$",
    re.IGNORECASE,
)


def _run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    print("\n>>> " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(_ROOT), env=env)


def _python_env() -> dict[str, str]:
    env = os.environ.copy()
    py_paths = [str(_ROOT / "python"), str(_ROOT / "build")]
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(py_paths + ([existing] if existing else []))
    return env


def _ecef_to_lla_deg(x: float, y: float, z: float) -> tuple[float, float, float]:
    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    lat = math.atan2(z, p * (1.0 - WGS84_E2))
    for _ in range(6):
        n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * math.sin(lat) ** 2)
        lat = math.atan2(z + WGS84_E2 * n * math.sin(lat), p)
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * math.sin(lat) ** 2)
    h = p / math.cos(lat) - n
    return math.degrees(lat), math.degrees(lon), h


def _parse_obs_meta(path: Path) -> dict[str, int | str]:
    m = _OBS_NAME_RE.match(path.name)
    if not m:
        raise ValueError(f"Unrecognized OBS filename (expected TERF00CYP_R_YYYYDOY...): {path.name}")
    return {
        "station": m.group("station").upper(),
        "year": int(m.group("year")),
        "doy": int(m.group("doy")),
        "tag": f"{m.group('year')}{m.group('doy')}",
    }


def _paths(work: Path) -> dict[str, Path]:
    return {
        "work": work,
        "data": work / "data",
        "results": work / "results",
        "mesh_cache": work / "results" / "terf_osm_cache.json",
        "mesh_tri": work / "results" / "terf_osm_triangles.npy",
        "mesh_glb": work / "results" / "terf_osm.glb",
        "summary": work / "results" / "terf_pipeline_summary.json",
    }


def _discover_obs(data_dir: Path) -> list[Path]:
    obs = sorted(data_dir.glob("TERF*.rnx")) + sorted(data_dir.glob("TERF*.obs"))
    obs = [p for p in obs if "MN" not in p.name.upper() and "BRDC" not in p.name.upper()]
    if not obs:
        raise FileNotFoundError(
            f"No TERF OBS files in {data_dir}. Upload TERF00CYP_R_*_MO.rnx files first."
        )
    return obs


def _read_ecef_from_obs(obs_path: Path) -> tuple[float, float, float]:
    sys.path.insert(0, str(_ROOT / "python"))
    from gnss_gpu.io.rinex import read_rinex_obs

    hdr = read_rinex_obs(obs_path).header.approx_position
    x, y, z = float(hdr[0]), float(hdr[1]), float(hdr[2])
    if math.sqrt(x * x + y * y + z * z) < 1.0:
        return TERF_ECEF_DEFAULT
    return x, y, z


def _station_bbox(ecef: tuple[float, float, float], buffer_deg: float) -> dict[str, float]:
    lat, lon, _h = _ecef_to_lla_deg(*ecef)
    return {
        "south": lat - buffer_deg,
        "west": lon - buffer_deg,
        "north": lat + buffer_deg,
        "east": lon + buffer_deg,
        "lat": lat,
        "lon": lon,
    }


def _brdc_url(year: int, doy: int) -> str:
    doy3 = f"{doy:03d}"
    name = f"BRDC00IGS_R_{year}{doy3}0000_01D_MN.rnx.gz"
    return f"https://igs.bkg.bund.de/root_ftp/IGS/BRDC/{year}/{doy3}/{name}"


def _download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  reuse {dest}")
        return
    print(f"  download {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    if dest.suffix == ".gz":
        out = dest.with_suffix("")
        with gzip.open(tmp, "rb") as fi, open(out, "wb") as fo:
            shutil.copyfileobj(fi, fo)
        tmp.unlink(missing_ok=True)
        print(f"  wrote {out}")
    else:
        tmp.rename(dest)
        print(f"  wrote {dest}")


def phase_setup() -> None:
    print("=== TERF Colab setup checklist ===")
    print("1. Runtime: GPU enabled (T4)")
    print("2. Repo at:", _ROOT)
    print("3. Build: mkdir build && cmake .. -DCMAKE_CUDA_ARCHITECTURES=75 && make -j")
    print("4. PYTHONPATH=python:build")
    print("5. Upload TERF OBS to <work-dir>/data/")
    try:
        import gnss_gpu._bvh  # noqa: F401

        print("OK: gnss_gpu._bvh importable")
    except Exception as exc:
        print(f"WARN: gnss_gpu._bvh not importable ({exc}) — build CUDA extensions first")


def phase_mesh(paths: dict[str, Path], obs_files: list[Path], *, buffer_deg: float) -> dict[str, float]:
    if paths["mesh_tri"].exists():
        print(f"reuse mesh {paths['mesh_tri']}")
        ecef = _read_ecef_from_obs(obs_files[0])
        return _station_bbox(ecef, buffer_deg)

    ecef = _read_ecef_from_obs(obs_files[0])
    bbox = _station_bbox(ecef, buffer_deg)
    paths["results"].mkdir(parents=True, exist_ok=True)
    env = _python_env()
    _run(
        [
            sys.executable,
            str(_EXPERIMENTS / "fetch_osm_buildings_bbox.py"),
            "--south",
            str(bbox["south"]),
            "--west",
            str(bbox["west"]),
            "--north",
            str(bbox["north"]),
            "--east",
            str(bbox["east"]),
            "--cache-json",
            str(paths["mesh_cache"]),
            "--out-triangles",
            str(paths["mesh_tri"]),
            "--export-glb",
            "--out-glb",
            str(paths["mesh_glb"]),
        ],
        env=env,
    )
    return bbox


def phase_nav(paths: dict[str, Path], obs_files: list[Path]) -> list[dict]:
    nav_info: list[dict] = []
    for obs_path in obs_files:
        meta = _parse_obs_meta(obs_path)
        year, doy = int(meta["year"]), int(meta["doy"])
        gz_name = f"BRDC00IGS_R_{year}{doy:03d}0000_01D_MN.rnx.gz"
        nav_rnx = paths["data"] / f"BRDC00IGS_R_{year}{doy:03d}0000_01D_MN.rnx"
        _download_file(_brdc_url(year, doy), paths["data"] / gz_name)
        if not nav_rnx.exists():
            raise FileNotFoundError(f"NAV not found after download: {nav_rnx}")
        nav_info.append({"obs": obs_path, "nav": nav_rnx, **meta})
    return nav_info


def _label_summary_csv(csv_path: Path) -> dict:
    n = los = nlos = vis = 0
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            n += 1
            if int(row.get("is_los", 0)):
                los += 1
            else:
                nlos += 1
            if int(row.get("is_visible", 0)):
                vis += 1
    return {
        "rows": n,
        "los": los,
        "nlos": nlos,
        "visible": vis,
        "nlos_frac": (nlos / n) if n else 0.0,
    }


def phase_labels(
    paths: dict[str, Path],
    nav_info: list[dict],
    *,
    systems: str,
    epoch_step: int,
    max_epochs: int,
    batch_size: int,
) -> list[dict]:
    env = _python_env()
    if not paths["mesh_tri"].exists():
        raise FileNotFoundError(f"Mesh missing: {paths['mesh_tri']}. Run --phase mesh first.")

    produced: list[dict] = []
    for item in nav_info:
        obs_path: Path = item["obs"]
        nav_path: Path = item["nav"]
        tag = str(item["tag"])
        ref_csv = paths["results"] / f"reference_{tag}.csv"
        out_csv = paths["results"] / f"terf_los_labels_{tag}_gc.csv"

        _run(
            [
                sys.executable,
                str(_EXPERIMENTS / "make_permanent_station_reference.py"),
                "--obs-path",
                str(obs_path),
                "--output-csv",
                str(ref_csv),
            ],
            env=env,
        )
        cmd = [
            sys.executable,
            str(_EXPERIMENTS / "export_los_labels_from_urbannav.py"),
            "--obs-path",
            str(obs_path),
            "--nav-path",
            str(nav_path),
            "--reference-csv",
            str(ref_csv),
            "--triangles-npy",
            str(paths["mesh_tri"]),
            "--output-csv",
            str(out_csv),
            "--systems",
            systems,
            "--batch-size",
            str(batch_size),
            "--epoch-step",
            str(epoch_step),
        ]
        if max_epochs > 0:
            cmd += ["--max-epochs", str(max_epochs)]
        _run(cmd, env=env)
        stats = _label_summary_csv(out_csv)
        produced.append(
            {
                "tag": tag,
                "obs": str(obs_path),
                "nav": str(nav_path),
                "reference_csv": str(ref_csv),
                "labels_csv": str(out_csv),
                **stats,
            }
        )
    return produced


def phase_viz(
    paths: dict[str, Path],
    nav_info: list[dict],
    *,
    n_epochs: int,
    epoch_min_interval_s: float,
    traj_step: float,
    obs_match_tol_s: float,
    elevation_mask_deg: float,
    viz_multipath: bool,
    export_mesh_glb: bool,
    cesium_ion_token: str,
    viz_tags: set[str] | None,
) -> list[dict]:
    """Build Cesium HTML with OBS-filtered satellite rays (like Odaiba viewer)."""
    env = _python_env()
    if not paths["mesh_tri"].exists():
        raise FileNotFoundError(f"Mesh missing: {paths['mesh_tri']}. Run --phase mesh first.")

    produced: list[dict] = []
    for item in nav_info:
        tag = str(item["tag"])
        if viz_tags is not None and tag not in viz_tags:
            continue

        obs_path: Path = Path(item["obs"])
        nav_path: Path = Path(item["nav"])
        ref_csv = paths["results"] / f"reference_{tag}.csv"
        if not ref_csv.exists():
            _run(
                [
                    sys.executable,
                    str(_EXPERIMENTS / "make_permanent_station_reference.py"),
                    "--obs-path",
                    str(obs_path),
                    "--output-csv",
                    str(ref_csv),
                ],
                env=env,
            )

        out_html = paths["results"] / f"terf_los_viz_{tag}.html"
        cmd = [
            sys.executable,
            str(_EXPERIMENTS / "build_3d_visualization_obs.py"),
            "--area-name",
            f"TERF {tag}",
            "--reference-csv",
            str(ref_csv),
            "--triangles-npy",
            str(paths["mesh_tri"]),
            "--nav",
            str(nav_path),
            "--obs",
            str(obs_path),
            "--out-html",
            str(out_html),
            "--n-epochs",
            str(int(n_epochs)),
            "--traj-step",
            str(float(traj_step)),
            "--epoch-min-interval-s",
            str(float(epoch_min_interval_s)),
            "--obs-match-tol-s",
            str(float(obs_match_tol_s)),
            "--elevation-mask-deg",
            str(float(elevation_mask_deg)),
            "--plateau-glb-radius-m",
            "1500",
            "--plateau-glb-max-tris",
            "50000",
        ]
        if export_mesh_glb:
            cmd.append("--export-mesh-glb")
        if viz_multipath:
            cmd.append("--viz-multipath")
        if cesium_ion_token.strip():
            cmd += ["--cesium-ion-token", cesium_ion_token.strip()]
        _run(cmd, env=env)
        produced.append(
            {
                "tag": tag,
                "html": str(out_html),
                "reference_csv": str(ref_csv),
                "obs": str(obs_path),
            }
        )
    if not produced:
        raise RuntimeError("No viz outputs produced (check --viz-day filter).")
    return produced


def phase_summary(paths: dict[str, Path], produced: list[dict], bbox: dict[str, float], viz_outputs: list[dict] | None = None) -> None:
    summary = {
        "station": "TERF00CYP",
        "marker": "TERF",
        "ecef_m": list(_read_ecef_from_obs(Path(produced[0]["obs"]))) if produced else list(TERF_ECEF_DEFAULT),
        "bbox_deg": bbox,
        "mesh_triangles_npy": str(paths["mesh_tri"]),
        "days": produced,
    }
    if viz_outputs:
        summary["viz_html"] = viz_outputs
    if produced:
        total_rows = sum(d["rows"] for d in produced)
        total_nlos = sum(d["nlos"] for d in produced)
        summary["aggregate"] = {
            "rows": total_rows,
            "nlos": total_nlos,
            "nlos_frac": total_nlos / total_rows if total_rows else 0.0,
        }
    paths["results"].mkdir(parents=True, exist_ok=True)
    paths["summary"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote summary -> {paths['summary']}")


def _seed_data_from_repo(work_data: Path, repo_data: Path) -> None:
    if not repo_data.is_dir():
        return
    work_data.mkdir(parents=True, exist_ok=True)
    for src in sorted(repo_data.glob("TERF*.rnx")):
        dst = work_data / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
            print(f"  copied {src.name} -> {dst}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Colab pipeline: TERF permanent station LOS/NLOS ray tracing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--phase",
        choices=("setup", "mesh", "nav", "labels", "viz", "summary", "all"),
        default="all",
    )
    p.add_argument("--work-dir", type=Path, default=Path("/content/terf_work"))
    p.add_argument(
        "--repo-data-dir",
        type=Path,
        default=_EXPERIMENTS / "data" / "TERF",
        help="Optional local/repo TERF RINEX folder to copy into work-dir/data",
    )
    p.add_argument("--systems", type=str, default="G,C", help="GNSS systems to label (default G,C)")
    p.add_argument("--epoch-step", type=int, default=1, help="Process every Nth OBS epoch (use 30 for quick test)")
    p.add_argument("--max-epochs", type=int, default=0, help="Cap epochs per day (0 = all)")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--bbox-buffer-deg", type=float, default=TERF_BBOX_BUFFER_DEG)
    p.add_argument("--skip-copy", action="store_true", help="Do not copy from --repo-data-dir")
    p.add_argument("--with-viz", action="store_true", help="Include 3D OBS viewer when --phase all")
    p.add_argument(
        "--viz-day",
        type=str,
        default="",
        help="Only viz this day tag (e.g. 2026192). Empty = all OBS days.",
    )
    p.add_argument("--n-epochs-viz", type=int, default=24, help="Viz epochs spread along the day")
    p.add_argument(
        "--epoch-min-interval-s",
        type=float,
        default=600.0,
        help="Min GPS spacing between consecutive viz epochs (default 600 s)",
    )
    p.add_argument("--traj-step-viz", type=float, default=1.0, help="Reference CSV row stride for viz")
    p.add_argument(
        "--obs-match-tol-s",
        type=float,
        default=20.0,
        help="Max |ΔGPS TOW| for matching OBS epoch (TERF sampling is 30 s)",
    )
    p.add_argument("--elevation-mask-deg", type=float, default=10.0)
    p.add_argument("--viz-multipath", action="store_true", help="Draw reflection paths in viz HTML")
    p.add_argument("--no-export-mesh-glb", action="store_true", help="Skip OSM mesh GLB sidecar in viz")
    p.add_argument(
        "--cesium-ion-token",
        type=str,
        default=os.environ.get("CESIUM_ION_TOKEN", ""),
        help="Cesium ion token for terrain (or set CESIUM_ION_TOKEN)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    paths = _paths(args.work_dir.resolve())
    paths["work"].mkdir(parents=True, exist_ok=True)
    paths["data"].mkdir(parents=True, exist_ok=True)

    if args.phase == "setup":
        phase_setup()
        return

    if not args.skip_copy:
        _seed_data_from_repo(paths["data"], args.repo_data_dir.resolve())

    obs_files = _discover_obs(paths["data"])
    print(f"Found {len(obs_files)} OBS file(s) in {paths['data']}")

    bbox: dict[str, float] | None = None
    nav_info: list[dict] | None = None
    produced: list[dict] | None = None
    viz_outputs: list[dict] | None = None
    viz_tags: set[str] | None = None
    if str(args.viz_day).strip():
        viz_tags = {str(args.viz_day).strip()}

    if args.phase in ("mesh", "all"):
        bbox = phase_mesh(paths, obs_files, buffer_deg=float(args.bbox_buffer_deg))

    if args.phase in ("nav", "labels", "viz", "all"):
        nav_info = phase_nav(paths, obs_files)

    if args.phase in ("labels", "all"):
        if nav_info is None:
            nav_info = phase_nav(paths, obs_files)
        produced = phase_labels(
            paths,
            nav_info,
            systems=str(args.systems),
            epoch_step=int(args.epoch_step),
            max_epochs=int(args.max_epochs),
            batch_size=int(args.batch_size),
        )

    if args.phase in ("viz",) or (args.phase == "all" and args.with_viz):
        if nav_info is None:
            nav_info = phase_nav(paths, obs_files)
        viz_outputs = phase_viz(
            paths,
            nav_info,
            n_epochs=int(args.n_epochs_viz),
            epoch_min_interval_s=float(args.epoch_min_interval_s),
            traj_step=float(args.traj_step_viz),
            obs_match_tol_s=float(args.obs_match_tol_s),
            elevation_mask_deg=float(args.elevation_mask_deg),
            viz_multipath=bool(args.viz_multipath),
            export_mesh_glb=not bool(args.no_export_mesh_glb),
            cesium_ion_token=str(args.cesium_ion_token),
            viz_tags=viz_tags,
        )

    if args.phase in ("summary", "all"):
        if produced is None:
            # Re-scan existing label CSVs
            produced = []
            for csv_path in sorted(paths["results"].glob("terf_los_labels_*_gc.csv")):
                m = re.search(r"terf_los_labels_(\d+)_gc\.csv", csv_path.name)
                tag = m.group(1) if m else csv_path.stem
                produced.append({"tag": tag, "labels_csv": str(csv_path), **_label_summary_csv(csv_path)})
        if bbox is None:
            ecef = _read_ecef_from_obs(obs_files[0])
            bbox = _station_bbox(ecef, float(args.bbox_buffer_deg))
        phase_summary(paths, produced or [], bbox, viz_outputs=viz_outputs)

    print("\n=== Done ===")
    print("Work directory:", paths["work"])
    for key in ("mesh_tri", "mesh_glb", "summary"):
        p = paths[key]
        if p.exists():
            print(f"  {key}: {p}")
    if viz_outputs:
        for v in viz_outputs:
            print(f"  viz: {v['html']}")


if __name__ == "__main__":
    main()
