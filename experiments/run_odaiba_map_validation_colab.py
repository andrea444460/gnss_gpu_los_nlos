#!/usr/bin/env python3
"""Colab-friendly end-to-end pipeline: UrbanNav Odaiba area map vs RINEX validation.

Run on Google Colab (GPU runtime recommended).  Does **not** require local execution.

Quick start on Colab
--------------------

1. **Setup** (first notebook cells)::

    # Runtime → Change runtime type → GPU (T4 is fine)
    !nvidia-smi

    # Clone (replace with your fork URL if needed)
    !git clone https://github.com/YOUR_USER/gnss_gpu.git /content/gnss_gpu
    %cd /content/gnss_gpu

    # Python deps for this pipeline
    !pip install -q numpy matplotlib folium branca pyproj rasterio requests scipy

    # Build CUDA extensions (T4 = sm_75)
    !apt-get -qq install -y cmake
    !mkdir -p build && cd build && cmake .. -DCMAKE_CUDA_ARCHITECTURES=75 && make -j$(nproc)
    # Extensions are imported from build/ via PYTHONPATH below

2. **Run full pipeline**::

    import os, sys
    os.environ["PYTHONPATH"] = "python:build"
    !python experiments/run_odaiba_map_validation_colab.py --work-dir /content/odaiba_work

3. **Download results** from ``/content/odaiba_work/results/`` (CSV, PNG, HTML).

Phases (run individually if needed)
-------------------------------------

- ``setup``     — print environment checklist
- ``fetch``     — UrbanNav Odaiba + PLATEAU subset (HTTP range download)
- ``mesh``      — PLATEAU GML → ``triangles.npy``
- ``manifest``  — bbox + RINEX time window JSON
- ``map``       — ``build_area_los_nlos_map.py`` on Odaiba bbox
- ``validate``  — compare map vs RINEX along trajectory
- ``all``       — fetch → mesh → manifest → map → validate

Notes
-----

- Area map uses ``--terrain-preprocess-mode cpu`` by default (no terrain CUDA dependency).
- BVH ray tracing still needs the compiled ``_bvh`` extension (GPU build above).
- PLATEAU Tokyo uses ``geoid_correction=egm96`` (orthometric → ellipsoidal).
- First fetch can take several minutes (remote ZIP over HTTP).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_EXPERIMENTS = _ROOT / "experiments"


def _run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    print("\n>>> " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(_ROOT), env=env)


def _python_env() -> dict[str, str]:
    env = os.environ.copy()
    py_paths = [str(_ROOT / "python"), str(_ROOT / "build")]
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(py_paths + ([existing] if existing else []))
    return env


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Colab pipeline: Odaiba area map vs UrbanNav RINEX validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--phase",
        choices=("setup", "fetch", "mesh", "manifest", "map", "validate", "all"),
        default="all",
    )
    p.add_argument("--work-dir", type=Path, default=Path("/content/odaiba_work"))
    p.add_argument("--skip-fetch", action="store_true", help="Reuse existing data under work-dir")
    p.add_argument("--map-step-m", type=float, default=25.0)
    p.add_argument("--map-dt-s", type=float, default=300.0)
    p.add_argument("--epoch-step", type=int, default=5, help="Validation RINEX epoch stride (5 ≈ faster on Colab)")
    p.add_argument("--instantaneous-geometry", action="store_true")
    p.add_argument("--max-neighbor-m", type=float, default=30.0)
    return p.parse_args()


def _paths(work: Path) -> dict[str, Path]:
    return {
        "work": work,
        "urbannav": work / "data" / "urbannav" / "Odaiba",
        "plateau": work / "data" / "plateau_odaiba",
        "results": work / "results",
        "triangles": work / "results" / "odaiba_triangles.npy",
        "manifest": work / "results" / "odaiba_run_manifest.json",
        "area_csv": work / "results" / "odaiba_area_points.csv",
        "area_html": work / "results" / "odaiba_area_map.html",
        "val_prefix": work / "results" / "odaiba_map_vs_rinex",
    }


def phase_setup() -> None:
    print("=== Colab setup checklist ===")
    print("1. Runtime: GPU enabled")
    print("2. Repo at:", _ROOT)
    print("3. Build extensions: mkdir build && cmake .. -DCMAKE_CUDA_ARCHITECTURES=75 && make -j")
    print("4. PYTHONPATH=python:build")
    print("5. pip install numpy matplotlib folium branca pyproj rasterio requests scipy")
    try:
        import gnss_gpu._bvh  # noqa: F401
        print("OK: gnss_gpu._bvh importable")
    except Exception as exc:
        print(f"WARN: gnss_gpu._bvh not importable ({exc}) — build CUDA extensions first")


def phase_fetch(paths: dict[str, Path]) -> None:
    paths["urbannav"].mkdir(parents=True, exist_ok=True)
    paths["plateau"].mkdir(parents=True, exist_ok=True)
    env = _python_env()
    _run(
        [sys.executable, str(_EXPERIMENTS / "fetch_urbannav_subset.py"),
         "--run", "Odaiba", "--output-dir", str(paths["urbannav"].parent)],
        env=env,
    )
    _run(
        [sys.executable, str(_EXPERIMENTS / "fetch_plateau_subset.py"),
         "--run-dir", str(paths["urbannav"]),
         "--output-dir", str(paths["plateau"]),
         "--preset", "tokyo23"],
        env=env,
    )


def phase_mesh(paths: dict[str, Path]) -> None:
    env = _python_env()
    paths["results"].mkdir(parents=True, exist_ok=True)
    _run(
        [
            sys.executable,
            str(_EXPERIMENTS / "plateau_dir_to_triangles_npy.py"),
            "--plateau-dir", str(paths["plateau"]),
            "--plateau-zone", "9",
            "--geoid-correction", "egm96",
            "--output-npy", str(paths["triangles"]),
            "--manifest-json", str(paths["results"] / "odaiba_triangles_manifest.json"),
        ],
        env=env,
    )


def phase_manifest(paths: dict[str, Path], map_dt_s: float) -> dict:
    env = _python_env()
    _run(
        [
            sys.executable,
            str(_EXPERIMENTS / "urbannav_run_manifest.py"),
            "--run-dir", str(paths["urbannav"]),
            "--rover-source", "trimble",
            "--dt-s", str(map_dt_s),
            "--out-json", str(paths["manifest"]),
        ],
        env=env,
    )
    return json.loads(paths["manifest"].read_text(encoding="utf-8"))


def phase_map(paths: dict[str, Path], manifest: dict, map_step_m: float, map_dt_s: float) -> None:
    env = _python_env()
    bbox = manifest["bbox"]
    rinex = manifest["rinex"]
    suggestions = manifest.get("area_map_suggestions", {})
    nav_path = paths["urbannav"] / "base.nav"

    cmd = [
        sys.executable,
        str(_EXPERIMENTS / "build_area_los_nlos_map.py"),
        "--south", str(bbox["south"]),
        "--west", str(bbox["west"]),
        "--north", str(bbox["north"]),
        "--east", str(bbox["east"]),
        "--triangles-npy", str(paths["triangles"]),
        "--nav", str(nav_path),
        "--out-csv", str(paths["area_csv"]),
        "--step-m", str(map_step_m),
        "--start-utc", str(suggestions.get("start_utc", rinex["start_utc"])),
        "--duration-s", str(suggestions.get("duration_s", rinex["duration_s"])),
        "--dt-s", str(map_dt_s),
        "--elevation-mask-deg", "10",
        "--nav-systems", "G,E,J,C",
        "--terrain-preprocess-mode", "cpu",
        "--point-batch-chunk", "64",
        "--eph-batch-chunk", "8",
    ]
    _run(cmd, env=env)

    _run(
        [
            sys.executable,
            str(_EXPERIMENTS / "render_area_los_map.py"),
            "--csv", str(paths["area_csv"]),
            "--out-html", str(paths["area_html"]),
            "--metric", "mean_n_los",
            "--stride", "2",
        ],
        env=env,
    )


def phase_validate(
    paths: dict[str, Path],
    *,
    epoch_step: int,
    max_neighbor_m: float,
    instantaneous: bool,
    map_dt_s: float,
) -> None:
    env = _python_env()
    cmd = [
        sys.executable,
        str(_EXPERIMENTS / "validate_area_map_vs_urbannav_rinex.py"),
        "--area-csv", str(paths["area_csv"]),
        "--run-dir", str(paths["urbannav"]),
        "--rover-source", "trimble",
        "--nav-systems", "G,E,J,C",
        "--max-neighbor-m", str(max_neighbor_m),
        "--epoch-step", str(epoch_step),
        "--map-dt-s", str(map_dt_s),
        "--out-prefix", str(paths["val_prefix"]),
    ]
    if instantaneous:
        cmd += ["--instantaneous-geometry", "--triangles-npy", str(paths["triangles"])]
    _run(cmd, env=env)


def main() -> None:
    args = _parse_args()
    paths = _paths(args.work_dir.resolve())
    paths["work"].mkdir(parents=True, exist_ok=True)

    if args.phase == "setup":
        phase_setup()
        return

    if args.phase in ("fetch", "all") and not args.skip_fetch:
        phase_fetch(paths)
    elif args.phase in ("mesh", "manifest", "map", "validate", "all") and not paths["urbannav"].exists():
        raise FileNotFoundError(
            f"UrbanNav data missing at {paths['urbannav']}. Run --phase fetch first."
        )

    manifest: dict | None = None
    if args.phase in ("mesh", "all"):
        phase_mesh(paths)

    if args.phase in ("manifest", "map", "all"):
        manifest = phase_manifest(paths, float(args.map_dt_s))
    elif paths["manifest"].exists():
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))

    if args.phase in ("map", "all"):
        if manifest is None:
            raise RuntimeError("manifest required for map phase")
        phase_map(paths, manifest, float(args.map_step_m), float(args.map_dt_s))

    if args.phase in ("validate", "all"):
        phase_validate(
            paths,
            epoch_step=int(args.epoch_step),
            max_neighbor_m=float(args.max_neighbor_m),
            instantaneous=bool(args.instantaneous_geometry),
            map_dt_s=float(args.map_dt_s),
        )

    print("\n=== Done ===")
    print("Results directory:", paths["results"])
    for label, p in paths.items():
        if label in ("work", "val_prefix"):
            continue
        if p.suffix and p.exists():
            print(f"  {label}: {p}")


if __name__ == "__main__":
    main()
