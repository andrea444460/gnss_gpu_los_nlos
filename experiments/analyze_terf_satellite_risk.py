#!/usr/bin/env python3
"""Numeric comparison: which satellites are NLOS/multipath-prone and at fixed sky directions.

Uses geometry labels CSV (``terf_los_labels_*_gc.csv`` from Colab labels phase) when
available. Optionally adds C/N0 fade proxy from RINEX OBS.

Outputs:
  - ``terf_sat_risk_summary.csv``  — one row per PRN (all days pooled)
  - ``terf_sat_risk_by_day.csv``   — PRN × day
  - ``terf_sat_risk_az_bins.csv``  — NLOS / fade rate by azimuth bin
  - ``terf_sat_risk_report.json``

Example (labels from Colab work dir copied locally)::

    python experiments/analyze_terf_satellite_risk.py \\
        --labels-glob /content/terf_work/results/terf_los_labels_*_gc.csv

Local C/N0-only proxy (no labels)::

    python experiments/analyze_terf_satellite_risk.py \\
        --obs-glob experiments/data/TERF/TERF00CYP_R_*_MO.rnx \\
        --nav-glob experiments/data/TERF/BRDC*.rnx \\
        --reference-csv experiments/data/TERF/reference_2026192.csv \\
        --output-dir experiments/results/terf_sat_risk
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_ROOT / "python"))

_TAG_RE = re.compile(r"terf_los_labels_(\d+)_gc\.csv$", re.I)


def _circular_mean_deg(az: np.ndarray) -> float:
    if az.size == 0:
        return float("nan")
    r = np.radians(az)
    return float(np.degrees(np.arctan2(np.mean(np.sin(r)), np.mean(np.cos(r)))) % 360.0)


def _circular_concentration(az: np.ndarray) -> float:
    """1 = all same direction, 0 = uniform spread."""
    if az.size == 0:
        return float("nan")
    r = np.radians(az)
    return float(np.hypot(np.mean(np.sin(r)), np.mean(np.cos(r))))


def _circular_std_deg(az: np.ndarray) -> float:
    R = _circular_concentration(az)
    if not np.isfinite(R) or R <= 1e-9:
        return 180.0
    return float(np.degrees(np.sqrt(max(0.0, -2.0 * math.log(R)))))


def _tag_from_labels_path(p: Path) -> str:
    m = _TAG_RE.search(p.name)
    return m.group(1) if m else p.stem


def _load_labels_csv(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                rows.append(
                    {
                        "tag": _tag_from_labels_path(path),
                        "prn": str(row.get("sat_id") or row.get("prn", "")).strip().upper(),
                        "tow": float(row["gps_tow"]),
                        "is_los": int(row["is_los"]),
                        "is_visible": int(row.get("is_visible", 1)),
                        "az": float(row["azimuth_deg"]),
                        "el": float(row["elevation_deg"]),
                        "excess_delay_m": float(row.get("excess_delay_m") or 0.0),
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue
    return rows


@dataclass
class PrnStats:
    prn: str
    n_rows: int = 0
    n_nlos: int = 0
    n_los: int = 0
    az_nlos: list[float] = field(default_factory=list)
    el_nlos: list[float] = field(default_factory=list)
    excess_nlos: list[float] = field(default_factory=list)
    tags: set[str] = field(default_factory=set)

    def add(self, row: dict) -> None:
        self.n_rows += 1
        self.tags.add(str(row["tag"]))
        if int(row["is_los"]):
            self.n_los += 1
        else:
            self.n_nlos += 1
            self.az_nlos.append(float(row["az"]))
            self.el_nlos.append(float(row["el"]))
            self.excess_nlos.append(float(row.get("excess_delay_m", 0.0)))

    def summary(self) -> dict:
        az = np.asarray(self.az_nlos, dtype=np.float64)
        el = np.asarray(self.el_nlos, dtype=np.float64)
        nlos_frac = self.n_nlos / self.n_rows if self.n_rows else 0.0
        return {
            "prn": self.prn,
            "n_rows": self.n_rows,
            "n_nlos": self.n_nlos,
            "n_los": self.n_los,
            "nlos_frac": nlos_frac,
            "n_days": len(self.tags),
            "mean_az_nlos_deg": _circular_mean_deg(az) if az.size else None,
            "az_nlos_std_deg": _circular_std_deg(az) if az.size else None,
            "az_nlos_concentration": _circular_concentration(az) if az.size else None,
            "mean_el_nlos_deg": float(np.mean(el)) if el.size else None,
            "std_el_nlos_deg": float(np.std(el)) if el.size else None,
            "mean_excess_delay_nlos_m": float(np.mean(self.excess_nlos)) if self.excess_nlos else None,
        }


def _aggregate_labels(all_rows: list[dict]) -> tuple[dict[str, PrnStats], dict[tuple[str, str], PrnStats]]:
    pooled: dict[str, PrnStats] = {}
    by_day: dict[tuple[str, str], PrnStats] = {}
    for row in all_rows:
        prn = row["prn"]
        tag = row["tag"]
        if prn not in pooled:
            pooled[prn] = PrnStats(prn=prn)
        pooled[prn].add(row)
        key = (tag, prn)
        if key not in by_day:
            by_day[key] = PrnStats(prn=prn)
        by_day[key].add(row)
    return pooled, by_day


def _az_bin_summary(all_rows: list[dict], *, bin_deg: float, min_el: float) -> list[dict]:
    bins: dict[int, dict] = defaultdict(lambda: {"n": 0, "nlos": 0, "els": []})
    n_bins = max(1, int(round(360.0 / bin_deg)))
    for row in all_rows:
        if float(row["el"]) < min_el:
            continue
        b = int(float(row["az"]) // bin_deg) % n_bins
        bins[b]["n"] += 1
        if not int(row["is_los"]):
            bins[b]["nlos"] += 1
        bins[b]["els"].append(float(row["el"]))
    out = []
    for b in sorted(bins):
        d = bins[b]
        n = d["n"]
        out.append(
            {
                "az_lo_deg": b * bin_deg,
                "az_hi_deg": (b + 1) * bin_deg,
                "n_samples": n,
                "n_nlos": d["nlos"],
                "nlos_frac": d["nlos"] / n if n else 0.0,
                "mean_el_deg": float(np.mean(d["els"])) if d["els"] else None,
            }
        )
    out.sort(key=lambda r: (-r["nlos_frac"], -r["n_nlos"]))
    return out


def _cnr_proxy_rows(
    obs_path: Path,
    nav_path: Path,
    rx: np.ndarray,
    tag: str,
    *,
    min_el_deg: float,
    fade_db: float,
) -> list[dict]:
    from gnss_gpu.ephemeris import Ephemeris
    from gnss_gpu.io.nav_rinex import read_nav_rinex_multi
    from gnss_gpu.io.rinex import read_rinex_obs
    from gnss_gpu.urban_signal_sim import _sat_elevation_azimuth

    GPS_EPOCH = datetime(1980, 1, 6)

    def gps_tow(dt: datetime) -> float:
        return (dt - GPS_EPOCH).total_seconds() % 604800.0

    def preferred_cnr(odict: dict) -> float | None:
        for k, v in odict.items():
            if str(k).upper().startswith("S"):
                try:
                    fv = float(v)
                    if np.isfinite(fv) and fv > 0:
                        return fv
                except (TypeError, ValueError):
                    pass
        return None

    def norm_prn(s: str) -> str:
        s = s.strip().upper()
        if len(s) >= 3 and s[0] in "GREJCI" and s[1:].isdigit():
            return f"{s[0]}{int(s[1:]):02d}"
        return s

    obs = read_rinex_obs(obs_path)
    eph = Ephemeris(read_nav_rinex_multi(nav_path))
    by_prn: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)

    for ep in obs.epochs:
        tow = float(gps_tow(ep.time))
        for sat_id, odict in ep.observations.items():
            cnr = preferred_cnr(odict)
            if cnr is None:
                continue
            prn = norm_prn(str(sat_id))
            try:
                sat_ecef, _, used = eph.compute(tow, [prn])
            except Exception:
                continue
            if not used:
                continue
            el_rad, az_rad = _sat_elevation_azimuth(rx, sat_ecef[0:1])
            el = float(np.degrees(el_rad[0]))
            if el < min_el_deg:
                continue
            az = float(np.degrees(az_rad[0])) % 360.0
            by_prn[prn].append((tow, az, el, float(cnr)))

    proxy: list[dict] = []
    for prn, series in by_prn.items():
        series.sort(key=lambda x: x[0])
        cnrs = np.array([s[3] for s in series])
        med = float(np.median(cnrs))
        for tow, az, el, cnr in series:
            is_fade = int(cnr < med - float(fade_db))
            proxy.append(
                {
                    "tag": tag,
                    "prn": prn,
                    "tow": tow,
                    "is_los": 1 - is_fade,
                    "is_visible": 1,
                    "az": az,
                    "el": el,
                    "excess_delay_m": 0.0,
                }
            )
    return proxy


def _match_nav(obs_path: Path, nav_paths: list[Path]) -> Path | None:
    m = re.search(r"_(\d{7})\d+_01D_", obs_path.name)
    if not m:
        return None
    tag = m.group(1)
    for p in nav_paths:
        if tag in p.name:
            return p
    return None


def _tag_from_obs(obs_path: Path) -> str:
    m = re.search(r"_(\d{7})\d+_01D_", obs_path.name)
    return m.group(1) if m else obs_path.stem


def _load_rx(reference_csv: Path) -> np.ndarray:
    with open(reference_csv, newline="", encoding="utf-8") as fh:
        row = next(csv.DictReader(fh))
    return np.array(
        [float(row["ECEF X (m)"]), float(row["ECEF Y (m)"]), float(row["ECEF Z (m)"])],
        dtype=np.float64,
    )


def _rank_sticky_nlos(pooled: dict[str, PrnStats], *, min_nlos: int) -> list[dict]:
    rows = []
    for st in pooled.values():
        if st.n_nlos < min_nlos:
            continue
        s = st.summary()
        s["risk_score"] = float(s["nlos_frac"] * (s["az_nlos_concentration"] or 0.0))
        rows.append(s)
    rows.sort(key=lambda r: (-r["risk_score"], -r["nlos_frac"], -r["n_nlos"]))
    return rows


def _day_consistency(by_day: dict[tuple[str, str], PrnStats]) -> list[dict]:
    per_prn: dict[str, list[dict]] = defaultdict(list)
    for (tag, prn), st in by_day.items():
        s = st.summary()
        if s["mean_az_nlos_deg"] is not None:
            per_prn[prn].append({"tag": tag, "az": s["mean_az_nlos_deg"], "el": s["mean_el_nlos_deg"], "nlos_frac": s["nlos_frac"]})
    out = []
    for prn, days in per_prn.items():
        if len(days) < 2:
            continue
        az = np.array([d["az"] for d in days], dtype=np.float64)
        el = np.array([d["el"] for d in days if d["el"] is not None], dtype=np.float64)
        out.append(
            {
                "prn": prn,
                "n_days_with_nlos": len(days),
                "az_std_across_days_deg": _circular_std_deg(az),
                "az_concentration_across_days": _circular_concentration(az),
                "el_std_across_days_deg": float(np.std(el)) if el.size else None,
                "mean_nlos_frac_across_days": float(np.mean([d["nlos_frac"] for d in days])),
                "day_tags": [d["tag"] for d in days],
            }
        )
    out.sort(key=lambda r: (-(r["az_concentration_across_days"] or 0), -r["mean_nlos_frac_across_days"]))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="TERF satellite NLOS/multipath risk comparison.")
    p.add_argument("--labels-glob", type=str, default="", help="e.g. .../terf_los_labels_*_gc.csv")
    p.add_argument("--obs-glob", type=str, default="", help="C/N0 proxy if no labels")
    p.add_argument("--nav-glob", type=str, default="", help="BRDC for C/N0 proxy")
    p.add_argument("--reference-csv", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=Path("experiments/results/terf_sat_risk"))
    p.add_argument("--az-bin-deg", type=float, default=10.0)
    p.add_argument("--min-el-deg", type=float, default=20.0)
    p.add_argument("--cnr-fade-db", type=float, default=6.0, help="C/N0 proxy fade threshold vs PRN median")
    p.add_argument("--min-nlos-count", type=int, default=20, help="Min NLOS rows to rank a PRN")
    args = p.parse_args()

    all_rows: list[dict] = []
    source = "labels"

    if str(args.labels_glob).strip():
        paths = sorted(Path(".").glob(str(args.labels_glob))) if not Path(args.labels_glob).is_absolute() else sorted(Path("/").glob(str(args.labels_glob)))
        # glob from cwd - fix: use Path().glob with full pattern
        from glob import glob

        paths = [Path(x) for x in sorted(glob(str(args.labels_glob)))]
        if not paths:
            print(f"No labels matched: {args.labels_glob}")
        for lp in paths:
            chunk = _load_labels_csv(lp)
            print(f"  labels {lp.name}: {len(chunk)} rows")
            all_rows.extend(chunk)

    if not all_rows and str(args.obs_glob).strip():
        from glob import glob

        if not args.reference_csv:
            raise SystemExit("--reference-csv required for C/N0 proxy mode")
        rx = _load_rx(args.reference_csv)
        obs_paths = [Path(x) for x in sorted(glob(str(args.obs_glob)))]
        nav_paths = [Path(x) for x in sorted(glob(str(args.nav_glob)))] if args.nav_glob else []
        source = "cnr_proxy"
        print(f"C/N0 proxy mode: {len(obs_paths)} OBS, fade >= {args.cnr_fade_db} dB below PRN median")
        for obs_path in obs_paths:
            nav = _match_nav(obs_path, nav_paths)
            if nav is None:
                print(f"  skip {obs_path.name}: no NAV")
                continue
            tag = _tag_from_obs(obs_path)
            chunk = _cnr_proxy_rows(
                obs_path,
                nav,
                rx,
                tag,
                min_el_deg=float(args.min_el_deg),
                fade_db=float(args.cnr_fade_db),
            )
            print(f"  {obs_path.name}: {len(chunk)} proxy rows (nav={nav.name})")
            all_rows.extend(chunk)

    if not all_rows:
        raise SystemExit("No data. Pass --labels-glob or --obs-glob + --nav-glob + --reference-csv.")

    pooled, by_day = _aggregate_labels(all_rows)
    summary_rows = _rank_sticky_nlos(pooled, min_nlos=int(args.min_nlos_count))
    day_rows = []
    for (tag, prn), st in sorted(by_day.items()):
        row = st.summary()
        row["tag"] = tag
        day_rows.append(row)
    day_rows.sort(key=lambda r: (-r["nlos_frac"], r["prn"], r["tag"]))
    az_bins = _az_bin_summary(all_rows, bin_deg=float(args.az_bin_deg), min_el=float(args.min_el_deg))
    consistency = _day_consistency(by_day)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = args.output_dir / "terf_sat_risk_summary.csv"
    by_day_csv = args.output_dir / "terf_sat_risk_by_day.csv"
    az_csv = args.output_dir / "terf_sat_risk_az_bins.csv"
    consist_csv = args.output_dir / "terf_sat_consistency_by_prn.csv"

    def write_csv(path: Path, rows: list[dict]) -> None:
        if not rows:
            return
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    write_csv(summary_csv, summary_rows)
    write_csv(by_day_csv, day_rows)
    write_csv(az_csv, az_bins)
    write_csv(consist_csv, consistency)

    report = {
        "source": source,
        "n_rows": len(all_rows),
        "n_prns": len(pooled),
        "top_nlos_prns": summary_rows[:15],
        "top_sticky_directions": consistency[:15],
        "top_azimuth_bins": az_bins[:10],
        "outputs": {
            "summary_csv": str(summary_csv),
            "by_day_csv": str(by_day_csv),
            "az_bins_csv": str(az_csv),
            "consistency_csv": str(consist_csv),
        },
    }
    report_json = args.output_dir / "terf_sat_risk_report.json"
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nSource: {source} ({len(all_rows)} rows, {len(pooled)} PRNs)")
    if source == "cnr_proxy":
        print("  (C/N0 fade proxy — for geometry NLOS use labels CSV from Colab)")
    print("\nTop PRNs by NLOS/fade fraction × direction concentration:")
    for r in summary_rows[:10]:
        az = r.get("mean_az_nlos_deg")
        conc = r.get("az_nlos_concentration")
        print(
            f"  {r['prn']:4s}  nlos={r['nlos_frac']*100:5.1f}%  n={r['n_nlos']:5d}  "
            f"az_nlos~{az:6.1f} deg  concentration={conc:.2f}  days={r['n_days']}"
        )
    print("\nPRNs with NLOS/fades always from same direction (across days):")
    for r in consistency[:8]:
        print(
            f"  {r['prn']:4s}  az_std={r['az_std_across_days_deg']:5.1f} deg  "
            f"conc={r['az_concentration_across_days']:.2f}  "
            f"mean_nlos={r['mean_nlos_frac_across_days']*100:.1f}%  days={r['day_tags']}"
        )
    print(f"\nWrote {summary_csv}")
    print(f"Wrote {report_json}")


if __name__ == "__main__":
    main()
