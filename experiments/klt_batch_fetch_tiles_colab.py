#!/usr/bin/env python3
"""Batch HK tile fetch for multiple KLT gt.csv files (Colab-friendly).

Why this script:
- Uses ONE shared tile cache directory across all KLT trips (no duplicate downloads).
- Calls existing fetch_hk3d_tiles_from_gt.py per dataset to build per-trip manifests.
- Optionally extracts only tiles referenced by each dataset manifest.
- Can validate and remove corrupted ZIPs from cache before running.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Iterable


def _validate_and_prune_bad_zips(zip_dir: Path) -> int:
    zip_dir.mkdir(parents=True, exist_ok=True)
    bad = 0
    for z in sorted(zip_dir.glob("*.zip")):
        try:
            with zipfile.ZipFile(z) as zf:
                zf.testzip()
        except Exception:
            print(f"[zip-check] removing corrupted zip: {z}")
            try:
                z.unlink()
            except OSError as exc:
                print(f"[zip-check] warning: could not remove {z}: {exc}")
            bad += 1
    return bad


def _read_manifest_zip_names(manifest_csv: Path) -> list[str]:
    out: list[str] = []
    with manifest_csv.open("r", encoding="utf-8", newline="") as fh:
        r = csv.DictReader(fh)
        for row in r:
            url = (row.get("url") or "").strip()
            if not url:
                continue
            name = os.path.basename(url.split("?", 1)[0]).strip()
            if name.lower().endswith(".zip"):
                out.append(name)
    # stable dedup
    seen: set[str] = set()
    uniq: list[str] = []
    for n in out:
        if n in seen:
            continue
        seen.add(n)
        uniq.append(n)
    return uniq


def _extract_zip_to_subdir(zip_path: Path, out_subdir: Path, force: bool) -> bool:
    marker = out_subdir / ".extract_ok"
    if marker.exists() and not force:
        return False
    out_subdir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_subdir)
    marker.write_text("ok\n", encoding="utf-8")
    return True


def _extract_for_dataset(manifest_csv: Path, shared_zip_dir: Path, extract_dir: Path, force: bool) -> tuple[int, int]:
    names = _read_manifest_zip_names(manifest_csv)
    processed = 0
    extracted_now = 0
    for i, name in enumerate(names, start=1):
        z = shared_zip_dir / name
        if not z.exists():
            print(f"[extract {i}/{len(names)}] missing zip in shared cache: {name}")
            continue
        out_sub = extract_dir / z.stem
        try:
            did = _extract_zip_to_subdir(z, out_sub, force=force)
        except zipfile.BadZipFile:
            print(f"[extract {i}/{len(names)}] bad zip: {name}")
            continue
        processed += 1
        if did:
            extracted_now += 1
            print(f"[extract {i}/{len(names)}] {name} -> {out_sub}")
        else:
            print(f"[extract {i}/{len(names)}] skip {name} (already extracted)")
    return processed, extracted_now


def _iter_gt_files(klt_labels_dir: Path) -> Iterable[Path]:
    return sorted(klt_labels_dir.glob("*_gt.csv"))


def _run_fetch(
    fetch_script: Path,
    gt_csv: Path,
    catalog_gml: Path,
    shared_zip_dir: Path,
    out_manifest_csv: Path,
    margin_deg: float,
    fmt: str,
    max_tiles: int,
) -> int:
    cmd = [
        sys.executable,
        "-u",
        str(fetch_script),
        "--gt-csv",
        str(gt_csv),
        "--catalog-gml",
        str(catalog_gml),
        "--output-dir",
        str(shared_zip_dir),
        "--out-manifest-csv",
        str(out_manifest_csv),
        "--margin-deg",
        str(margin_deg),
        "--format",
        fmt,
        "--max-tiles",
        str(max_tiles),
    ]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    assert p.stdout is not None
    for line in p.stdout:
        print(line, end="", flush=True)
    return p.wait()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch HK tile fetch for all KLT *_gt.csv with shared cache")
    p.add_argument("--klt-labels-dir", type=Path, required=True, help="Directory containing *_gt.csv files")
    p.add_argument("--catalog-gml", type=Path, required=True, help="Local resolved catalog GML path")
    p.add_argument("--shared-zip-dir", type=Path, required=True, help="Shared tile ZIP cache directory")
    p.add_argument("--manifests-dir", type=Path, required=True, help="Output directory for per-dataset manifest CSVs")
    p.add_argument("--extract-root", type=Path, default=Path(""), help="If set, extract per dataset to this root")
    p.add_argument("--force-extract", action="store_true", help="Re-extract even when marker exists")
    p.add_argument("--prune-bad-zips", action="store_true", help="Validate and delete corrupted zips in shared cache")
    p.add_argument("--margin-deg", type=float, default=0.003)
    p.add_argument("--format", dest="fmt", choices=("gltf", "fbx", "max"), default="gltf")
    p.add_argument("--max-tiles", type=int, default=0, help="0 means all selected tiles")
    p.add_argument(
        "--fetch-script",
        type=Path,
        default=Path(__file__).resolve().parent / "fetch_hk3d_tiles_from_gt.py",
        help="Path to fetch_hk3d_tiles_from_gt.py",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    klt_labels_dir = args.klt_labels_dir.resolve()
    catalog_gml = args.catalog_gml.resolve()
    shared_zip_dir = args.shared_zip_dir.resolve()
    manifests_dir = args.manifests_dir.resolve()
    fetch_script = args.fetch_script.resolve()

    if not klt_labels_dir.exists():
        raise FileNotFoundError(f"klt labels dir not found: {klt_labels_dir}")
    if not catalog_gml.exists():
        raise FileNotFoundError(f"catalog gml not found: {catalog_gml}")
    if not fetch_script.exists():
        raise FileNotFoundError(f"fetch script not found: {fetch_script}")

    shared_zip_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    if bool(args.prune_bad_zips):
        n_bad = _validate_and_prune_bad_zips(shared_zip_dir)
        print(f"[zip-check] removed bad zips: {n_bad}")

    gt_files = list(_iter_gt_files(klt_labels_dir))
    if not gt_files:
        raise RuntimeError(f"no *_gt.csv files found in {klt_labels_dir}")
    print(f"Found GT files: {len(gt_files)}")

    failed: list[str] = []
    for i, gt in enumerate(gt_files, start=1):
        ds = gt.name.replace("_gt.csv", "")
        manifest = manifests_dir / f"hk_tiles_manifest_{ds}.csv"
        print(f"\n=== [{i}/{len(gt_files)}] {ds} ===")
        rc = _run_fetch(
            fetch_script=fetch_script,
            gt_csv=gt,
            catalog_gml=catalog_gml,
            shared_zip_dir=shared_zip_dir,
            out_manifest_csv=manifest,
            margin_deg=float(args.margin_deg),
            fmt=str(args.fmt),
            max_tiles=int(args.max_tiles),
        )
        if rc != 0:
            print(f"[ERROR] fetch failed for {ds} (rc={rc})")
            failed.append(ds)
            continue

        if str(args.extract_root):
            ext_dir = args.extract_root.resolve() / ds
            n_proc, n_now = _extract_for_dataset(
                manifest_csv=manifest,
                shared_zip_dir=shared_zip_dir,
                extract_dir=ext_dir,
                force=bool(args.force_extract),
            )
            print(f"[extract] dataset={ds} processed={n_proc} newly_extracted={n_now} dir={ext_dir}")

    print("\nDone.")
    if failed:
        print(f"Failed datasets ({len(failed)}): {failed}")
        raise SystemExit(1)
    print("All datasets completed.")


if __name__ == "__main__":
    main()

