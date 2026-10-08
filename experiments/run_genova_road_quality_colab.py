#!/usr/bin/env python3
"""One-shot Genova road-quality ``.rqz`` pipeline for Google Colab (GPU).

Assumes the CUDA extensions are already built (``cmake`` + ``make``) and
``PYTHONPATH=python:build``.

Steps:
1. download daily BRDC NAV (BKG IGS)
2. fetch OSM building triangles for the bbox
3. run ``build_road_quality_rqz.py`` (BVH LOS → compact ``.rqz``)

Example after Colab setup::

    !python experiments/run_genova_road_quality_colab.py \\
        --work-dir /content/genova_rqz \\
        --year 2026 --doy 120 \\
        --duration-s 86400 --dt-s 300 --arterial-only
"""

from __future__ import annotations

import argparse
import gzip
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_EXPERIMENTS = _ROOT / "experiments"

# Genova urban extract (matches routing GUI / generate_road_quality defaults).
DEFAULT_SOUTH = 44.3850
DEFAULT_WEST = 8.8800
DEFAULT_NORTH = 44.4450
DEFAULT_EAST = 8.9800

_BRDC_PRODUCTS = ("BRDC00WRD_R", "BRDC00IGS_R", "BRD400DLR_S")


def _python_env() -> dict[str, str]:
    env = os.environ.copy()
    py_paths = [str(_ROOT / "python"), str(_ROOT / "build")]
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(py_paths + ([existing] if existing else []))
    env["MPLBACKEND"] = "Agg"
    return env


def _run(cmd: list[str]) -> None:
    print("\n>>> " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(_ROOT), env=_python_env())


def _brdc_rnx_name(year: int, doy: int, product: str = _BRDC_PRODUCTS[0]) -> str:
    return f"{product}_{year}{doy:03d}0000_01D_MN.rnx"


def _brdc_gz_name(year: int, doy: int, product: str = _BRDC_PRODUCTS[0]) -> str:
    return _brdc_rnx_name(year, doy, product) + ".gz"


def _brdc_urls(year: int, doy: int, product: str) -> list[str]:
    doy3 = f"{doy:03d}"
    name = _brdc_gz_name(year, doy, product)
    return [
        f"https://igs.bkg.bund.de/root_ftp/IGS/BRDC/{year}/{doy3}/{name}",
        f"https://igs.bkg.bund.de/root_ftp/EUREF/BRDC/{year}/{doy3}/{name}",
    ]


def _download_file(url: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    out_rnx = dest.with_suffix("") if dest.suffix == ".gz" else dest
    if out_rnx.exists() and out_rnx.stat().st_size > 0:
        print(f"  reuse {out_rnx}")
        return True
    print(f"  download {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=120) as resp:
            data = resp.read()
    except urllib.error.HTTPError as exc:
        tmp.unlink(missing_ok=True)
        if exc.code == 404:
            print(f"  404 {url}")
            return False
        raise
    if dest.suffix == ".gz" and (len(data) < 2 or data[:2] != b"\x1f\x8b"):
        print(f"  skip non-gzip payload from {url}")
        return False
    tmp.write_bytes(data)
    try:
        if dest.suffix == ".gz":
            with gzip.open(tmp, "rb") as fi, open(out_rnx, "wb") as fo:
                shutil.copyfileobj(fi, fo)
            tmp.unlink(missing_ok=True)
            print(f"  wrote {out_rnx}")
        else:
            tmp.rename(dest)
            print(f"  wrote {dest}")
    except (gzip.BadGzipFile, OSError) as exc:
        tmp.unlink(missing_ok=True)
        out_rnx.unlink(missing_ok=True)
        print(f"  skip corrupt gzip from {url}: {exc}")
        return False
    return out_rnx.exists() and out_rnx.stat().st_size > 0


def _resolve_nav(data_dir: Path, year: int, doy: int, *, max_day_delta: int = 3) -> Path:
    deltas = [0] + [d for k in range(1, max_day_delta + 1) for d in (-k, k)]
    for delta in deltas:
        alt = int(doy) + int(delta)
        if alt < 1 or alt > 366:
            continue
        for product in _BRDC_PRODUCTS:
            rnx = data_dir / _brdc_rnx_name(year, alt, product)
            if rnx.exists() and rnx.stat().st_size > 0:
                print(f"  reuse NAV {rnx}")
                return rnx
            gz = data_dir / _brdc_gz_name(year, alt, product)
            for url in _brdc_urls(year, alt, product):
                if _download_file(url, gz) and rnx.exists():
                    if alt != doy:
                        print(f"  NAV fallback: DOY {alt:03d} for requested {doy:03d}")
                    return rnx
    raise FileNotFoundError(
        f"Could not download BRDC for {year} DOY {doy:03d}. "
        f"Upload a *MN.rnx into {data_dir}."
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--work-dir", type=Path, default=Path("/content/genova_rqz"))
    p.add_argument("--south", type=float, default=DEFAULT_SOUTH)
    p.add_argument("--west", type=float, default=DEFAULT_WEST)
    p.add_argument("--north", type=float, default=DEFAULT_NORTH)
    p.add_argument("--east", type=float, default=DEFAULT_EAST)
    p.add_argument("--year", type=int, required=True, help="NAV calendar year (e.g. 2026)")
    p.add_argument("--doy", type=int, required=True, help="Day of year for BRDC NAV")
    p.add_argument("--dt-s", type=float, default=300.0)
    p.add_argument("--duration-s", type=float, default=86400.0)
    p.add_argument("--step-m", type=float, default=30.0)
    p.add_argument("--tile-size-m", type=float, default=12000.0)
    p.add_argument("--eph-batch-chunk", type=int, default=32)
    p.add_argument("--point-batch-chunk", type=int, default=256)
    p.add_argument("--arterial-only", action="store_true", default=True)
    p.add_argument("--no-arterial-only", action="store_true")
    p.add_argument("--include-pedestrian", action="store_true")
    p.add_argument("--skip-mesh", action="store_true", help="Reuse existing triangles npy")
    p.add_argument("--nav", type=Path, default=None, help="Optional local NAV override")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    work = args.work_dir.resolve()
    data = work / "data"
    out_dir = work / "results"
    data.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    tri = out_dir / "genova_osm_triangles.npy"
    cache = out_dir / "genova_osm_cache.json"
    dem = out_dir / "genova_area_dem.tif"
    rqz = out_dir / "genova_quality_24h.rqz"
    metrics = out_dir / "metrics.csv"

    # Sanity: CUDA BVH
    sys.path.insert(0, str(_ROOT / "python"))
    sys.path.insert(0, str(_ROOT / "build"))
    try:
        import gnss_gpu._bvh  # noqa: F401

        print("OK: gnss_gpu._bvh importable", flush=True)
    except Exception as exc:
        raise SystemExit(
            f"gnss_gpu._bvh not importable ({exc}). "
            "Build CUDA extensions first (cmake/make) and set PYTHONPATH=python:build."
        ) from exc

    if args.nav is not None:
        nav = args.nav.resolve()
        if not nav.exists():
            raise FileNotFoundError(nav)
    else:
        nav = _resolve_nav(data, int(args.year), int(args.doy))

    if not args.skip_mesh or not tri.exists():
        _run(
            [
                sys.executable,
                str(_EXPERIMENTS / "fetch_osm_buildings_bbox.py"),
                "--south",
                str(args.south),
                "--west",
                str(args.west),
                "--north",
                str(args.north),
                "--east",
                str(args.east),
                "--cache-json",
                str(cache),
                "--out-triangles",
                str(tri),
            ]
        )
    else:
        print(f"reuse mesh {tri}", flush=True)

    cmd = [
        sys.executable,
        str(_EXPERIMENTS / "build_road_quality_rqz.py"),
        "--south",
        str(args.south),
        "--west",
        str(args.west),
        "--north",
        str(args.north),
        "--east",
        str(args.east),
        "--triangles-npy",
        str(tri),
        "--nav",
        str(nav),
        "--out",
        str(rqz),
        "--step-m",
        str(args.step_m),
        "--dt-s",
        str(args.dt_s),
        "--duration-s",
        str(args.duration_s),
        "--rx-alt-mode",
        "dem",
        "--tile-size-m",
        str(args.tile_size_m),
        "--eph-batch-chunk",
        str(args.eph_batch_chunk),
        "--point-batch-chunk",
        str(args.point_batch_chunk),
        "--dem-auto-download",
        "--dem-auto-out",
        str(dem),
        "--metrics-csv",
        str(metrics),
    ]
    if args.include_pedestrian:
        cmd.append("--include-pedestrian")
    if not args.no_arterial_only:
        cmd.append("--arterial-only")

    _run(cmd)
    print(f"\nDONE: {rqz}  ({rqz.stat().st_size/1024:.1f} KB)", flush=True)
    print(f"metrics: {metrics}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
