#!/usr/bin/env python3
"""Patch KLT viewer HTMLs to highlight geometric false-NLOS rays in yellow.

False-NLOS definition (for selected constellations):
    geometric is_los == 0 AND GT is_los == 1

The script computes keys "(rounded gps_tow)|sat_id" and injects a JS lookup
table into each HTML. During rendering, matching rays are colored yellow and
labelled as FNLOS.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _norm_sat(s: object) -> str | None:
    x = str(s).upper().strip()
    if len(x) == 3 and x[0] in "GRECJIS" and x[1:].isdigit():
        return x
    return None


def _find_gt_file(gt_dir: Path, dataset_id: str) -> Path | None:
    candidates = [
        gt_dir / f"klt_{dataset_id}_features_labels_labelMinus18Auto.csv",
        gt_dir / f"klt_{dataset_id}_features_labels_labelMinus18.csv",
        gt_dir / f"klt_{dataset_id}_features_labels.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _false_nlos_keys(
    geom_csv: Path,
    gt_csv: Path,
    offset_s: int,
    systems: set[str],
) -> list[str]:
    g = pd.read_csv(geom_csv)
    t = pd.read_csv(gt_csv)

    g["sat_id"] = g["sat_id"].map(_norm_sat)
    t["sat_id"] = t["sat_id"].map(_norm_sat)
    g = g[g["sat_id"].notna()].copy()
    t = t[t["sat_id"].notna()].copy()
    g = g[g["sat_id"].str[0].isin(systems)].copy()
    t = t[t["sat_id"].str[0].isin(systems)].copy()
    if g.empty or t.empty:
        return []

    g["tow_sec"] = np.round(pd.to_numeric(g["gps_tow"], errors="coerce")).astype("Int64")
    g = g[g["tow_sec"].notna()].copy()
    t["gps_tow_base"] = (pd.to_numeric(t["label_timestamp"], errors="coerce") - 315964800 + 18) % 604800.0
    t = t[t["gps_tow_base"].notna()].copy()
    t["tow_sec"] = np.round(t["gps_tow_base"] + int(offset_s)).astype("Int64")
    t = t[t["tow_sec"].notna()].copy()
    if g.empty or t.empty:
        return []

    gg = g.groupby(["tow_sec", "sat_id"], as_index=False)["is_los"].mean()
    gg["is_los"] = (gg["is_los"] >= 0.5).astype(int)
    tt = t.groupby(["tow_sec", "sat_id"], as_index=False)["is_los"].mean()
    tt["is_los"] = (tt["is_los"] >= 0.5).astype(int)
    m = gg.merge(tt, on=["tow_sec", "sat_id"], suffixes=("_geom", "_gt"))
    fn = m[(m["is_los_geom"] == 0) & (m["is_los_gt"] == 1)]
    keys = sorted({f"{int(r.tow_sec)}|{r.sat_id}" for r in fn.itertuples()})
    return keys


def _patch_html_text(s: str, keys: list[str]) -> str:
    arr = "[" + ",".join("'" + k + "'" for k in keys) + "]"
    snippet = (
        "const FALSE_NLOS_KEYS = new Set("
        + arr
        + ");\n"
        + "function isFalseNlosRay(epoch, ray) {\n"
        + "  if (!epoch || !ray || epoch.gpsTow == null || !ray.prn) return false;\n"
        + "  const tow = Math.round(Number(epoch.gpsTow));\n"
        + "  if (!Number.isFinite(tow)) return false;\n"
        + "  const key = String(tow) + '|' + String(ray.prn).trim().toUpperCase();\n"
        + "  return FALSE_NLOS_KEYS.has(key);\n"
        + "}\n\n"
    )

    if "const FALSE_NLOS_KEYS = new Set(" in s:
        a = s.find("const FALSE_NLOS_KEYS = new Set(")
        b = s.find("function paintFrame() {", a)
        if b > a:
            s = s[:a] + snippet + s[b:]
    else:
        s = s.replace("function paintFrame() {", snippet + "function paintFrame() {", 1)

    old_ray_block = (
        "        const color = ray.los ? Cesium.Color.fromCssColorString('#00d4aa')\n"
        "                              : Cesium.Color.fromCssColorString('#ff6b6b');\n"
        "        const widthPx = ray.los ? 4 : 5;"
    )
    new_ray_block = (
        "        const isFnlos = isFalseNlosRay(epoch, ray);\n"
        "        const color = isFnlos\n"
        "          ? Cesium.Color.fromCssColorString('#ffd400')\n"
        "          : (ray.los ? Cesium.Color.fromCssColorString('#00d4aa')\n"
        "                     : Cesium.Color.fromCssColorString('#ff6b6b'));\n"
        "        const widthPx = isFnlos ? 6 : (ray.los ? 4 : 5);"
    )
    s = s.replace(old_ray_block, new_ray_block)

    old_label_block = (
        "          const color = ray.los ? Cesium.Color.fromCssColorString('#00d4aa')\n"
        "                                : Cesium.Color.fromCssColorString('#ff6b6b');\n"
        "          le.show = true;\n"
        "          le.position = Cesium.Cartesian3.fromDegrees(ray.end[1], ray.end[0], ray.end[2]);\n"
        "          le.label.text = new Cesium.ConstantProperty('PRN' + ray.prn + (ray.los ? ' LOS' : ' NLOS'));"
    )
    new_label_block = (
        "          const isFnlos = isFalseNlosRay(epoch, ray);\n"
        "          const color = isFnlos\n"
        "            ? Cesium.Color.fromCssColorString('#ffd400')\n"
        "            : (ray.los ? Cesium.Color.fromCssColorString('#00d4aa')\n"
        "                       : Cesium.Color.fromCssColorString('#ff6b6b'));\n"
        "          le.show = true;\n"
        "          le.position = Cesium.Cartesian3.fromDegrees(ray.end[1], ray.end[0], ray.end[2]);\n"
        "          le.label.text = new Cesium.ConstantProperty('PRN' + ray.prn + (isFnlos ? ' FNLOS' : (ray.los ? ' LOS' : ' NLOS')));"
    )
    s = s.replace(old_label_block, new_label_block)
    return s


def _dataset_id_from_html_name(html_path: Path) -> str | None:
    name = html_path.name
    if not name.endswith("_viz.html"):
        return None
    return name[: -len("_viz.html")]


def main() -> None:
    p = argparse.ArgumentParser(description="Patch KLT viewer HTMLs with false-NLOS highlighting.")
    p.add_argument("--html-root", type=Path, required=True, help="Directory containing *_viz.html files")
    p.add_argument("--geom-dir", type=Path, required=True, help="Directory containing *_los_labels.csv")
    p.add_argument("--gt-dir", type=Path, required=True, help="Directory containing klt_* feature-label CSVs")
    p.add_argument("--dataset-id", type=str, default="", help="Optional single dataset id to patch")
    p.add_argument("--offset-s", type=int, default=18, help="GT->geom time offset in seconds (default +18)")
    p.add_argument("--systems", type=str, default="G,C", help="Constellations to consider (default G,C)")
    p.add_argument("--backup", action="store_true", help="Write .bak backup before patching each HTML")
    args = p.parse_args()

    systems = {s.strip().upper() for s in args.systems.split(",") if s.strip()}
    htmls = sorted(args.html_root.glob("*_viz.html"))
    if args.dataset_id:
        htmls = [h for h in htmls if h.name == f"{args.dataset_id}_viz.html"]

    if not htmls:
        raise RuntimeError(f"No target HTML files found in {args.html_root}")

    patched = 0
    skipped = 0
    for html in htmls:
        ds = _dataset_id_from_html_name(html)
        if not ds:
            skipped += 1
            continue
        geom = args.geom_dir / f"{ds}_los_labels.csv"
        gt = _find_gt_file(args.gt_dir, ds)
        if not geom.exists() or gt is None:
            print(f"[skip] {ds}: missing geom or GT")
            skipped += 1
            continue

        keys = _false_nlos_keys(geom, gt, offset_s=args.offset_s, systems=systems)
        text = html.read_text(encoding="utf-8")
        new_text = _patch_html_text(text, keys)
        if args.backup:
            html.with_suffix(html.suffix + ".bak").write_text(text, encoding="utf-8")
        html.write_text(new_text, encoding="utf-8")
        patched += 1
        print(f"[ok] {ds}: patched false-NLOS keys={len(keys)}")

    print(f"Done. patched={patched}, skipped={skipped}")


if __name__ == "__main__":
    main()
