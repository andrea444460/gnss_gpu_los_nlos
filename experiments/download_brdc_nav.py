#!/usr/bin/env python3
"""Download daily BRDC NAV from BKG for each OBS day (exact DOY, no fallback).

Example::

    python experiments/download_brdc_nav.py \\
        --obs-glob "experiments/data/GNSS Stations/*_MO.rnx" \\
        --output-dir "experiments/data/GNSS Stations"
"""

from __future__ import annotations

import argparse
import gzip
import re
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

_OBS_TAG_RE = re.compile(r"_(\d{4})(\d{3})\d+_01D_", re.I)
_BRDC_PRODUCTS = ("BRDC00WRD_R", "BRDC00IGS_R", "BRD400DLR_S")


def _brdc_rnx_name(year: int, doy: int, product: str) -> str:
    return f"{product}_{year}{doy:03d}0000_01D_MN.rnx"


def _brdc_gz_name(year: int, doy: int, product: str) -> str:
    return f"{_brdc_rnx_name(year, doy, product)}.gz"


def _brdc_urls(year: int, doy: int, product: str) -> list[str]:
    doy3 = f"{doy:03d}"
    name = _brdc_gz_name(year, doy, product)
    return [
        f"https://igs.bkg.bund.de/root_ftp/IGS/BRDC/{year}/{doy3}/{name}",
        f"https://igs.bkg.bund.de/root_ftp/EUREF/BRDC/{year}/{doy3}/{name}",
    ]


def _obs_days(obs_paths: list[Path]) -> list[tuple[int, int]]:
    days: set[tuple[int, int]] = set()
    for p in obs_paths:
        m = _OBS_TAG_RE.search(p.name)
        if m:
            days.add((int(m.group(1)), int(m.group(2))))
    return sorted(days)


def _download_file(url: str, dest_gz: Path) -> bool:
    out_rnx = dest_gz.with_suffix("")
    if out_rnx.exists() and out_rnx.stat().st_size > 0:
        print(f"  reuse {out_rnx.name}")
        return True
    print(f"  download {url}")
    try:
        with urllib.request.urlopen(url, timeout=120) as resp:
            data = resp.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            print(f"  404 {url}")
            return False
        raise
    if len(data) < 2 or data[:2] != b"\x1f\x8b":
        print(f"  skip non-gzip {url}")
        return False
    dest_gz.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest_gz.with_suffix(dest_gz.suffix + ".part")
    tmp.write_bytes(data)
    try:
        with gzip.open(tmp, "rb") as fi, open(out_rnx, "wb") as fo:
            shutil.copyfileobj(fi, fo)
    except (gzip.BadGzipFile, OSError) as exc:
        tmp.unlink(missing_ok=True)
        out_rnx.unlink(missing_ok=True)
        print(f"  corrupt gzip: {exc}")
        return False
    finally:
        tmp.unlink(missing_ok=True)
    ok = out_rnx.exists() and out_rnx.stat().st_size > 0
    if ok:
        print(f"  wrote {out_rnx.name} ({out_rnx.stat().st_size // 1024} KB)")
    return ok


def download_brdc(year: int, doy: int, output_dir: Path) -> Path | None:
    for product in _BRDC_PRODUCTS:
        rnx = output_dir / _brdc_rnx_name(year, doy, product)
        if rnx.exists() and rnx.stat().st_size > 0:
            print(f"  reuse {rnx.name}")
            return rnx
        gz = output_dir / _brdc_gz_name(year, doy, product)
        for url in _brdc_urls(year, doy, product):
            if _download_file(url, gz) and rnx.exists():
                return rnx
    return None


def main() -> None:
    p = argparse.ArgumentParser(description="Download BRDC NAV for OBS days (exact DOY).")
    p.add_argument("--obs-glob", type=str, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()

    from glob import glob

    obs_paths = [Path(x) for x in sorted(glob(args.obs_glob))]
    if not obs_paths:
        raise SystemExit(f"No OBS matched {args.obs_glob}")

    days = _obs_days(obs_paths)
    print(f"OBS days: {len(days)} -> {days}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    failed: list[tuple[int, int]] = []
    for year, doy in days:
        print(f"\nDOY {year}{doy:03d}:")
        nav = download_brdc(year, doy, args.output_dir)
        if nav is None:
            failed.append((year, doy))

    if failed:
        raise SystemExit(f"Failed to download NAV for: {failed}")
    print(f"\nAll NAV ready in {args.output_dir}")


if __name__ == "__main__":
    main()
