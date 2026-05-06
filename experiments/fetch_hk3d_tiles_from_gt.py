#!/usr/bin/env python3
"""Fetch Hong Kong 3D map tile ZIPs from a GT trajectory.

Given:
- a GT trajectory CSV (e.g., KLT gt.csv with columns: time,lat,lon,alt),
- a Hong Kong 3D catalog GML (Non-textured models converted catalog),

this script computes a buffered trajectory bbox, selects intersecting tiles,
and downloads their ZIP URLs (glTF/FBX/MAX) to a local folder.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
import urllib.request
import urllib.error
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import pandas as pd

NS = {
    "gml": "http://www.opengis.net/gml",
    "fme": "http://www.safe.com/gml/fme",
}

URL_FIELD_MAP = {
    "gltf": "Format_glTF",
    "fbx": "Format_FBX",
    "max": "Format_MAX",
}

DEFAULT_KLT_LABEL_IDS = [
    "0610_KLT1_203",
    "0610_KLT2_209",
    "0610_KLT3_404",
    "1109_KLT1_421",
    "1109_KLT2_294",
    "1109_KLT2_301",
    "1109_KLT3_666",
    "1116_KLT1_184",
    "1116_KLT2_169",
    "1116_KLT3_649",
]


def _download_with_progress(url: str, dst: Path, label: str = "") -> None:
    """Download URL to path with lightweight console progress logging."""
    prefix = f"{label} " if label else ""
    max_attempts = 5
    last_err: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        state = {"last_pct": -10, "last_mb": -1}

        def _hook(block_count: int, block_size: int, total_size: int) -> None:
            downloaded = block_count * block_size
            downloaded_mb = int(downloaded / (1024 * 1024))
            if total_size > 0:
                pct = int(min(100, (downloaded * 100) / total_size))
                # Print every 10% (and always at completion)
                if pct >= state["last_pct"] + 10 or pct == 100:
                    total_mb = max(1, int(total_size / (1024 * 1024)))
                    print(f"{prefix}progress: {pct:3d}% ({downloaded_mb} / {total_mb} MiB)")
                    state["last_pct"] = pct
            else:
                # Unknown total size fallback
                if downloaded_mb >= state["last_mb"] + 20:
                    print(f"{prefix}downloaded: {downloaded_mb} MiB")
                    state["last_mb"] = downloaded_mb

        try:
            urllib.request.urlretrieve(url, dst, reporthook=_hook)
            return
        except Exception as exc:
            last_err = exc
            # Remove potentially partial file before retry.
            try:
                if dst.exists():
                    dst.unlink()
            except OSError:
                pass
            if attempt >= max_attempts:
                break
            wait_s = min(2 ** attempt, 20)
            print(f"{prefix}download failed (attempt {attempt}/{max_attempts}): {exc}")
            print(f"{prefix}retrying in {wait_s}s ...")
            time.sleep(wait_s)

    raise RuntimeError(f"{prefix}download failed after {max_attempts} attempts: {last_err}")


def _parse_poslist_2d(text: str) -> list[tuple[float, float]]:
    vals = [float(x) for x in text.strip().split()]
    if len(vals) < 4 or len(vals) % 2 != 0:
        return []
    pts = []
    for i in range(0, len(vals), 2):
        lat = vals[i]
        lon = vals[i + 1]
        pts.append((lat, lon))
    return pts


def _bbox_from_points(points: list[tuple[float, float]]) -> tuple[float, float, float, float] | None:
    if not points:
        return None
    lats = [p[0] for p in points]
    lons = [p[1] for p in points]
    return min(lats), min(lons), max(lats), max(lons)


def _intersects(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> bool:
    a_min_lat, a_min_lon, a_max_lat, a_max_lon = a
    b_min_lat, b_min_lon, b_max_lat, b_max_lon = b
    return not (
        a_max_lat < b_min_lat
        or b_max_lat < a_min_lat
        or a_max_lon < b_min_lon
        or b_max_lon < a_min_lon
    )


def _trajectory_bbox_from_gt(gt_csv: Path, lat_col: int, lon_col: int, margin_deg: float) -> tuple[float, float, float, float]:
    df = pd.read_csv(gt_csv, header=None)
    lat = pd.to_numeric(df.iloc[:, lat_col], errors="coerce").dropna()
    lon = pd.to_numeric(df.iloc[:, lon_col], errors="coerce").dropna()
    if lat.empty or lon.empty:
        raise RuntimeError(f"No valid lat/lon values in {gt_csv}")
    return (
        float(lat.min() - margin_deg),
        float(lon.min() - margin_deg),
        float(lat.max() + margin_deg),
        float(lon.max() + margin_deg),
    )


def _select_tiles(catalog_gml: Path, query_bbox: tuple[float, float, float, float], url_field: str) -> list[dict[str, str]]:
    root = ET.parse(catalog_gml).getroot()
    rows: list[dict[str, str]] = []
    for feat in root.findall(".//fme:Nontextured_models", NS):
        sheet = (feat.findtext("fme:SHEETNO", default="", namespaces=NS) or "").strip()
        url = (feat.findtext(f"fme:{url_field}", default="", namespaces=NS) or "").strip()
        if not url:
            continue
        poslist_el = feat.find(".//gml:posList", NS)
        if poslist_el is None or not (poslist_el.text or "").strip():
            continue
        pts = _parse_poslist_2d(poslist_el.text or "")
        tile_bbox = _bbox_from_points(pts)
        if tile_bbox is None or not _intersects(tile_bbox, query_bbox):
            continue
        rows.append(
            {
                "sheetno": sheet,
                "url": url,
                "tile_min_lat": f"{tile_bbox[0]:.9f}",
                "tile_min_lon": f"{tile_bbox[1]:.9f}",
                "tile_max_lat": f"{tile_bbox[2]:.9f}",
                "tile_max_lon": f"{tile_bbox[3]:.9f}",
            }
        )
    # Deduplicate by URL while preserving order.
    out: list[dict[str, str]] = []
    seen = set()
    for r in rows:
        u = r["url"]
        if u in seen:
            continue
        seen.add(u)
        out.append(r)
    return out


def _download_tiles(
    rows: list[dict[str, str]],
    out_dir: Path,
    max_tiles: int,
    continue_on_download_error: bool = True,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for i, r in enumerate(rows, start=1):
        if max_tiles > 0 and n >= max_tiles:
            break
        url = r["url"]
        fname = os.path.basename(url.split("?", 1)[0])
        dst = out_dir / fname
        if dst.exists() and dst.stat().st_size > 0:
            print(f"[{i}/{len(rows)}] skip {fname} (exists {dst.stat().st_size} bytes)")
            n += 1
            continue
        print(f"[{i}/{len(rows)}] download {fname} ...")
        try:
            _download_with_progress(url, dst, label=f"[{i}/{len(rows)}] {fname}")
            print(f"[{i}/{len(rows)}] done {fname} ({dst.stat().st_size} bytes)")
            n += 1
        except Exception as exc:
            print(f"[{i}/{len(rows)}] failed {fname}: {exc}")
            if not continue_on_download_error:
                raise
            try:
                if dst.exists():
                    dst.unlink()
            except OSError:
                pass
    return n


def _extract_tiles(zip_dir: Path, extract_dir: Path, force_extract: bool = False) -> tuple[int, int]:
    """Extract downloaded tile ZIPs into per-tile subfolders.

    Returns:
        (n_processed, n_extracted_now)
    """
    extract_dir.mkdir(parents=True, exist_ok=True)
    zips = sorted(zip_dir.glob("*.zip"))
    n_processed = 0
    n_extracted_now = 0
    for i, z in enumerate(zips, start=1):
        out_subdir = extract_dir / z.stem
        marker = out_subdir / ".extract_ok"
        if marker.exists() and not force_extract:
            print(f"[extract {i}/{len(zips)}] skip {z.name} (already extracted)")
            n_processed += 1
            continue
        out_subdir.mkdir(parents=True, exist_ok=True)
        print(f"[extract {i}/{len(zips)}] {z.name} -> {out_subdir}")
        with zipfile.ZipFile(z) as zf:
            zf.extractall(out_subdir)
        marker.write_text("ok\n", encoding="utf-8")
        n_processed += 1
        n_extracted_now += 1
    return n_processed, n_extracted_now


def _to_dropbox_direct_download(url: str) -> str:
    """Normalize a Dropbox share URL to a direct-download form (dl=1)."""
    parts = urlsplit(url)
    q = dict(parse_qsl(parts.query, keep_blank_values=True))
    q["dl"] = "1"
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(q), parts.fragment))


def _join_dropbox_path(base_url: str, suffix_path: str) -> str:
    """Append a path suffix to a Dropbox shared URL path preserving query."""
    parts = urlsplit(base_url.strip())
    base_path = parts.path.rstrip("/")
    suffix = suffix_path.strip().lstrip("/")
    joined_path = f"{base_path}/{suffix}" if suffix else base_path
    return urlunsplit((parts.scheme, parts.netloc, joined_path, parts.query, parts.fragment))


def _download_all_klt_label_gt_csvs(
    klt_root_dropbox_url: str,
    dataset_ids: list[str],
    output_dir: Path,
    force_downloads: bool,
    skip_names: set[str],
    dropbox_access_token: str = "",
) -> int:
    """Download gt.csv for multiple KLT label datasets from one root Dropbox URL.

    This mode intentionally fetches only gt.csv files (per dataset) so large files
    like image.zip and nlos.pkl are not downloaded.
    """
    root = klt_root_dropbox_url.strip()
    if not root:
        raise ValueError("--klt-root-dropbox-url is required for --download-all-klt-labels mode")
    token = dropbox_access_token.strip()

    output_dir.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    for i, ds in enumerate(dataset_ids, start=1):
        ds_clean = ds.strip()
        if not ds_clean:
            continue
        gt_name = "gt.csv"
        if gt_name.lower() in skip_names:
            print(f"[KLT {i}/{len(dataset_ids)}] skip {ds_clean}/{gt_name} (excluded)")
            continue
        dst = output_dir / f"{ds_clean}_gt.csv"
        if dst.exists() and dst.stat().st_size > 0 and not force_downloads:
            print(f"[KLT {i}/{len(dataset_ids)}] reuse {dst.name} ({dst.stat().st_size} bytes)")
            n_ok += 1
            continue

        print(f"[KLT {i}/{len(dataset_ids)}] download {ds_clean}/gt.csv")
        try:
            if token:
                # Prefer Dropbox API: robust even when anonymous shared links return "No Access" HTML.
                api_url = "https://content.dropboxapi.com/2/sharing/get_shared_link_file"
                rel_path = f"/label/{ds_clean}/gt.csv"
                api_arg = {"url": root, "path": rel_path}
                req = urllib.request.Request(
                    api_url,
                    data=b"",
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Dropbox-API-Arg": json.dumps(api_arg),
                        "Content-Type": "text/plain; charset=utf-8",
                    },
                    method="POST",
                )
                try:
                    with urllib.request.urlopen(req) as resp:
                        payload = resp.read()
                except urllib.error.HTTPError as http_err:
                    body = ""
                    try:
                        body = http_err.read().decode("utf-8", errors="ignore")
                    except Exception:
                        body = "<unable to read error body>"
                    raise RuntimeError(
                        f"Dropbox API HTTP {http_err.code} for {ds_clean}/gt.csv. "
                        f"Response body: {body}"
                    ) from http_err
                dst.write_bytes(payload)
                # Guardrail: Dropbox web error pages are HTML, not CSV.
                head = dst.read_text(encoding="utf-8", errors="ignore")[:256].lstrip().lower()
                if head.startswith("<!doctype html") or head.startswith("<html"):
                    raise RuntimeError(
                        "Dropbox API returned HTML content; verify token/scopes and shared-link access."
                    )
            else:
                # Public-link fallback.
                url = _join_dropbox_path(root, f"label/{ds_clean}/gt.csv")
                direct_url = _to_dropbox_direct_download(url)
                _download_with_progress(direct_url, dst, label=f"[KLT {i}/{len(dataset_ids)}]")
            print(f"[KLT {i}/{len(dataset_ids)}] done {dst.name} ({dst.stat().st_size} bytes)")
            n_ok += 1
        except Exception as exc:
            print(f"[KLT {i}/{len(dataset_ids)}] failed {ds_clean}/gt.csv: {exc}")
    return n_ok


def _resolve_gt_csv(
    gt_csv: Path | None,
    gt_dropbox_url: str,
    work_dir: Path,
    force_downloads: bool,
) -> Path:
    """Resolve gt.csv from local path or Dropbox URL.

    If Dropbox returns a ZIP/folder bundle, this function extracts and returns the
    first file named `gt.csv` found recursively.
    """
    if gt_csv is not None:
        p = gt_csv.resolve()
        if not p.exists():
            raise FileNotFoundError(f"GT CSV not found: {p}")
        return p

    if not gt_dropbox_url.strip():
        raise ValueError("Provide either --gt-csv or --gt-dropbox-url")

    work_dir.mkdir(parents=True, exist_ok=True)
    cached_gt_csv = work_dir / "gt.csv"
    if cached_gt_csv.exists() and cached_gt_csv.stat().st_size > 0 and not force_downloads:
        print(f"Reusing cached GT CSV: {cached_gt_csv} ({cached_gt_csv.stat().st_size} bytes)")
        return cached_gt_csv

    direct_url = _to_dropbox_direct_download(gt_dropbox_url.strip())
    dl_path = work_dir / "gt_download.bin"
    if dl_path.exists() and dl_path.stat().st_size > 0 and not force_downloads:
        print(f"Reusing cached GT archive/blob: {dl_path} ({dl_path.stat().st_size} bytes)")
    else:
        print(f"Downloading GT from Dropbox: {direct_url}")
        _download_with_progress(direct_url, dl_path, label="[GT]")
        print(f"GT raw download saved: {dl_path} ({dl_path.stat().st_size} bytes)")

    # Direct CSV response.
    if dl_path.suffix.lower() == ".csv":
        return dl_path
    try:
        head = dl_path.read_bytes()[:4]
    except OSError:
        head = b""
    if head.startswith(b"PK\x03\x04"):
        with zipfile.ZipFile(dl_path) as zf:
            names = [n for n in zf.namelist() if n.lower().endswith("/gt.csv") or n.lower().endswith("gt.csv")]
            if not names:
                raise RuntimeError("Dropbox archive downloaded, but no gt.csv found inside")
            names.sort()
            pick = names[0]
            out_csv = work_dir / "gt.csv"
            with zf.open(pick) as src, out_csv.open("wb") as dst:
                dst.write(src.read())
            print(f"Resolved gt.csv from archive member: {pick}")
            return out_csv

    # Fallback: maybe CSV body with generic filename.
    txt = dl_path.read_text(encoding="utf-8", errors="ignore")
    if "," in txt and "\n" in txt:
        out_csv = work_dir / "gt.csv"
        out_csv.write_text(txt, encoding="utf-8")
        return out_csv

    raise RuntimeError("Could not resolve gt.csv from provided Dropbox URL")


def _resolve_catalog_gml(
    catalog_gml: Path | None,
    catalog_gml_url: str,
    work_dir: Path,
    force_downloads: bool,
) -> Path:
    """Resolve catalog GML from local path or URL."""
    if catalog_gml is not None:
        p = catalog_gml.resolve()
        if p.exists():
            return p
        raw = str(catalog_gml)
        if ":" in raw and "\\" in raw and os.name != "nt":
            raise FileNotFoundError(
                f"Catalog GML looks like a Windows path on Linux/Colab: {raw}. "
                "Use a Colab/Linux path (e.g. /content/...gml) or provide --catalog-gml-url."
            )
        raise FileNotFoundError(f"Catalog GML not found: {p}")

    if not catalog_gml_url.strip():
        raise ValueError("Provide either --catalog-gml or --catalog-gml-url")

    work_dir.mkdir(parents=True, exist_ok=True)
    parts = urlsplit(catalog_gml_url.strip())
    name = os.path.basename(parts.path) or "catalog.gml"
    if not name.lower().endswith(".gml"):
        name = "catalog.gml"
    out = work_dir / name
    if out.exists() and out.stat().st_size > 0 and not force_downloads:
        print(f"Reusing cached catalog GML: {out} ({out.stat().st_size} bytes)")
        # Continue below to normalize cached payload into a parseable GML if needed.
    else:
        print(f"Downloading catalog GML: {catalog_gml_url.strip()}")
        _download_with_progress(catalog_gml_url.strip(), out, label="[CATALOG]")
        print(f"Catalog GML saved: {out} ({out.stat().st_size} bytes)")

    # Some endpoints return ZIP or non-.gml filename payloads.
    try:
        head = out.read_bytes()[:4]
    except OSError:
        head = b""
    if head.startswith(b"PK\x03\x04"):
        with zipfile.ZipFile(out) as zf:
            gml_members = [n for n in zf.namelist() if n.lower().endswith(".gml") or n.lower().endswith(".xml")]
            if not gml_members:
                raise RuntimeError("Catalog archive downloaded, but no .gml/.xml file found inside")
            gml_members.sort()
            pick = gml_members[0]
            out_gml = work_dir / Path(pick).name
            with zf.open(pick) as src, out_gml.open("wb") as dst:
                dst.write(src.read())
            print(f"Resolved catalog GML from archive member: {pick}")
            return out_gml

    # Heuristic: if download is HTML, fail with explicit guidance.
    text_head = out.read_text(encoding="utf-8", errors="ignore")[:512].lstrip().lower()
    if text_head.startswith("<!doctype html") or text_head.startswith("<html"):
        raise RuntimeError(
            "Catalog URL returned HTML instead of GML. Use a direct download URL, "
            "or provide --catalog-gml with a local .gml file."
        )

    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch HK 3D tile ZIPs intersecting GT trajectory bbox")
    p.add_argument("--gt-csv", type=Path, help="Local GT CSV path (e.g., KLT gt.csv)")
    p.add_argument(
        "--gt-dropbox-url",
        type=str,
        default="",
        help="Dropbox shared URL to GT folder/file; gt.csv is auto-resolved and downloaded",
    )
    p.add_argument("--catalog-gml", type=Path, help="Local HK 3D catalog GML path")
    p.add_argument(
        "--catalog-gml-url",
        type=str,
        default="",
        help="URL to HK 3D catalog GML (downloaded when --catalog-gml is not provided)",
    )
    p.add_argument("--output-dir", type=Path, required=True, help="Destination directory for downloaded ZIPs")
    p.add_argument(
        "--extract-dir",
        type=Path,
        default=Path(""),
        help="If set, extract downloaded ZIPs into this directory (per-tile subfolders)",
    )
    p.add_argument(
        "--extract",
        action="store_true",
        help="Enable ZIP extraction step after download",
    )
    p.add_argument(
        "--force-extract",
        action="store_true",
        help="Re-extract even if extraction marker exists",
    )
    p.add_argument("--out-manifest-csv", type=Path, default=Path("experiments/results/hk_tiles_manifest.csv"))
    p.add_argument(
        "--work-dir",
        type=Path,
        default=Path("experiments/data/_hk3d_tmp"),
        help="Working directory for temporary downloads (including gt.csv from Dropbox)",
    )
    p.add_argument(
        "--force-downloads",
        action="store_true",
        help="Force re-download of GT/Catalog even if cached files exist",
    )
    p.add_argument(
        "--download-all-klt-labels",
        action="store_true",
        help="Batch-download gt.csv for all KLT label subdatasets from one Dropbox root URL",
    )
    p.add_argument(
        "--klt-root-dropbox-url",
        type=str,
        default="",
        help="Dropbox shared root URL containing config/data/label folders for KLT",
    )
    p.add_argument(
        "--klt-label-ids",
        type=str,
        default=",".join(DEFAULT_KLT_LABEL_IDS),
        help="Comma-separated KLT label IDs for batch mode (default: known 10 IDs)",
    )
    p.add_argument(
        "--klt-label-output-dir",
        type=Path,
        default=Path("experiments/data/klt_labels"),
        help="Output directory for batch-downloaded *gt.csv files in KLT mode",
    )
    p.add_argument(
        "--klt-skip-names",
        type=str,
        default="image.zip,nlos.pkl",
        help="Comma-separated file names to exclude in KLT mode",
    )
    p.add_argument(
        "--dropbox-access-token",
        type=str,
        default=os.environ.get("DROPBOX_ACCESS_TOKEN", ""),
        help="Dropbox access token for API download in batch KLT mode (or set DROPBOX_ACCESS_TOKEN)",
    )
    p.add_argument("--lat-col", type=int, default=1, help="0-based latitude column index in gt.csv (default 1)")
    p.add_argument("--lon-col", type=int, default=2, help="0-based longitude column index in gt.csv (default 2)")
    p.add_argument(
        "--margin-deg",
        type=float,
        default=0.003,
        help="BBox margin in degrees around GT track (default 0.003 ~ 300m lat)",
    )
    p.add_argument(
        "--format",
        dest="fmt",
        choices=("gltf", "fbx", "max"),
        default="gltf",
        help="Which download URL field to use from catalog (default gltf)",
    )
    p.add_argument(
        "--max-tiles",
        type=int,
        default=0,
        help="Download at most N tiles (0 means all selected)",
    )
    p.add_argument(
        "--print-links",
        action="store_true",
        help="Print selected tile URLs",
    )
    p.add_argument(
        "--strict-downloads",
        action="store_true",
        help="Fail fast if any tile download fails (default: continue with remaining tiles)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if bool(args.download_all_klt_labels):
        ids = [x.strip() for x in str(args.klt_label_ids).split(",") if x.strip()]
        skip_names = {x.strip().lower() for x in str(args.klt_skip_names).split(",") if x.strip()}
        n = _download_all_klt_label_gt_csvs(
            klt_root_dropbox_url=str(args.klt_root_dropbox_url),
            dataset_ids=ids,
            output_dir=args.klt_label_output_dir.resolve(),
            force_downloads=bool(args.force_downloads),
            skip_names=skip_names,
            dropbox_access_token=str(args.dropbox_access_token),
        )
        print(f"KLT batch completed: downloaded/reused {n} gt.csv files")
        print(f"KLT output dir: {args.klt_label_output_dir.resolve()}")
        return

    work_dir = args.work_dir.resolve()
    gt_csv = _resolve_gt_csv(args.gt_csv, args.gt_dropbox_url, work_dir, bool(args.force_downloads))
    catalog_gml = _resolve_catalog_gml(args.catalog_gml, args.catalog_gml_url, work_dir, bool(args.force_downloads))

    query_bbox = _trajectory_bbox_from_gt(
        gt_csv=gt_csv,
        lat_col=int(args.lat_col),
        lon_col=int(args.lon_col),
        margin_deg=float(args.margin_deg),
    )
    url_field = URL_FIELD_MAP[args.fmt]
    rows = _select_tiles(catalog_gml, query_bbox, url_field=url_field)

    args.out_manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_manifest_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "sheetno",
                "url",
                "tile_min_lat",
                "tile_min_lon",
                "tile_max_lat",
                "tile_max_lon",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    print(f"GT: {gt_csv}")
    print(f"Catalog: {catalog_gml}")
    print(
        "Query bbox: "
        f"lat[{query_bbox[0]:.6f},{query_bbox[2]:.6f}] "
        f"lon[{query_bbox[1]:.6f},{query_bbox[3]:.6f}]"
    )
    print(f"Selected tiles: {len(rows)}")
    print(f"Manifest CSV: {args.out_manifest_csv.resolve()}")

    if args.print_links:
        for r in rows:
            print(r["url"])

    n_downloaded = _download_tiles(
        rows,
        args.output_dir.resolve(),
        max_tiles=int(args.max_tiles),
        continue_on_download_error=not bool(args.strict_downloads),
    )
    print(f"Downloaded/skipped tiles: {n_downloaded}")
    print(f"Output dir: {args.output_dir.resolve()}")

    if bool(args.extract) or bool(args.extract_dir):
        if str(args.extract_dir):
            extract_dir = args.extract_dir.resolve()
        else:
            extract_dir = args.output_dir.resolve().parent / f"{args.output_dir.resolve().name}_extracted"
        n_proc, n_now = _extract_tiles(
            zip_dir=args.output_dir.resolve(),
            extract_dir=extract_dir,
            force_extract=bool(args.force_extract),
        )
        print(f"Extracted ZIPs: processed={n_proc}, newly_extracted={n_now}")
        print(f"Extract dir: {extract_dir}")


if __name__ == "__main__":
    main()
