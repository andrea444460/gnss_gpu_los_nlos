#!/usr/bin/env python3
"""Per-station C/N0 fade proxy + sticky-direction satellite risk (multi-station).

Same logic as analyze_terf_satellite_risk.py (fade >= N dB below PRN median, el>=min),
run separately for each permanent station in a data directory.

Example::

    python experiments/analyze_gnss_stations_sat_risk_batch.py \\
        --data-dir "experiments/data/GNSS Stations" \\
        --crd-path "experiments/data/GNSS Stations/NEW_CYCLOPS_STATIONS.CRD" \\
        --output-dir experiments/results/gnss_stations_sat_risk
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from glob import glob
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_EXPERIMENTS = Path(__file__).resolve().parent
if str(_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_ROOT / "python"))
if str(_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS))

from gnss_gpu.ephemeris import Ephemeris
from gnss_gpu.io.nav_rinex import read_nav_rinex_multi
from gnss_gpu.io.rinex import read_rinex_obs
from gnss_gpu.urban_signal_sim import _sat_elevation_azimuth

import analyze_terf_satellite_risk as risk
from download_brdc_nav import download_brdc, _obs_days

_STATION_RE = re.compile(r"^([A-Z0-9]{4})00CYP_R_", re.I)
_CRD_RE = re.compile(
    r"^\s*\d+\s+([A-Z0-9]{4})\s+\S+\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)",
    re.I,
)
GPS_EPOCH = datetime(1980, 1, 6)
_CNR_PREF = ("S1C", "S1X", "S1P", "S1W", "S2W", "S2C", "S2X", "S5Q", "S5X", "S7Q", "S7X", "S8Q", "S8X")


def _load_crd(path: Path) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = _CRD_RE.match(line)
        if not m:
            continue
        code = m.group(1).upper()
        out[code] = np.array([float(m.group(2)), float(m.group(3)), float(m.group(4))], dtype=np.float64)
    return out


def _group_obs(data_dir: Path) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for p in sorted(data_dir.glob("*00CYP_R_*_MO.rnx")):
        m = _STATION_RE.match(p.name)
        if m:
            groups[m.group(1).upper()].append(p)
    return dict(groups)


def _preferred_cnr(obs_dict: dict) -> float | None:
    upper = {str(k).upper(): k for k in obs_dict}
    for code in _CNR_PREF:
        k = upper.get(code)
        if k is None:
            continue
        try:
            v = float(obs_dict[k])
        except (TypeError, ValueError):
            continue
        if math.isfinite(v) and v > 0.0:
            return v
    for uk, orig in sorted(upper.items()):
        if not uk.startswith("S"):
            continue
        try:
            v = float(obs_dict[orig])
        except (TypeError, ValueError):
            continue
        if math.isfinite(v) and v > 0.0:
            return v
    return None


def _normalize_prn(sat_id: str) -> str:
    s = str(sat_id).strip().upper()
    if len(s) >= 3 and s[0] in "GREJCI" and s[1:].isdigit():
        return f"{s[0]}{int(s[1:]):02d}"
    return s


def _cnr_proxy_arrays_fast(
    obs_path: Path,
    eph: Ephemeris,
    rx: np.ndarray,
    tag: str,
    *,
    min_el_deg: float,
    fade_db: float,
    epoch_step: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return prn, tow, is_los, az, el arrays for one OBS day."""
    step = max(1, int(epoch_step))
    obs = read_rinex_obs(obs_path)
    chunk_prn: list[str] = []
    chunk_tow: list[float] = []
    chunk_az: list[float] = []
    chunk_el: list[float] = []
    chunk_cnr: list[float] = []

    for ei, ep in enumerate(obs.epochs):
        if step > 1 and (ei % step) != 0:
            continue
        tow = float((ep.time - GPS_EPOCH).total_seconds() % 604800.0)
        prns: list[str] = []
        cnrs: list[float] = []
        for sat_id, odict in ep.observations.items():
            cnr = _preferred_cnr(odict)
            if cnr is None:
                continue
            prns.append(_normalize_prn(str(sat_id)))
            cnrs.append(float(cnr))
        if not prns:
            continue
        try:
            sat_ecef, _, used = eph.compute(tow, prn_list=prns)
        except Exception:
            continue
        if not used:
            continue
        used_map = {str(p).strip().upper(): i for i, p in enumerate(used)}
        sel_idx: list[int] = []
        sel_prn: list[str] = []
        sel_cnr: list[float] = []
        for prn, cnr in zip(prns, cnrs):
            j = used_map.get(prn)
            if j is None:
                continue
            sel_idx.append(j)
            sel_prn.append(prn)
            sel_cnr.append(cnr)
        if not sel_idx:
            continue
        sat_sel = np.asarray(sat_ecef[sel_idx], dtype=np.float64)
        el_rad, az_rad = _sat_elevation_azimuth(rx, sat_sel)
        el_deg = np.degrees(el_rad)
        az_deg = np.degrees(az_rad) % 360.0
        for k in range(len(sel_prn)):
            if float(el_deg[k]) < min_el_deg:
                continue
            chunk_prn.append(sel_prn[k])
            chunk_tow.append(tow)
            chunk_az.append(float(az_deg[k]))
            chunk_el.append(float(el_deg[k]))
            chunk_cnr.append(sel_cnr[k])

    empty_f = np.array([], dtype=np.float64)
    if not chunk_tow:
        return (
            np.array([], dtype="U8"),
            empty_f,
            np.array([], dtype=np.int8),
            empty_f.astype(np.float32),
            empty_f.astype(np.float32),
        )

    prn_arr = np.asarray(chunk_prn, dtype="U8")
    tow_arr = np.asarray(chunk_tow, dtype=np.float64)
    az_arr = np.asarray(chunk_az, dtype=np.float32)
    el_arr = np.asarray(chunk_el, dtype=np.float32)
    cnr_arr = np.asarray(chunk_cnr, dtype=np.float64)

    unique, inv = np.unique(prn_arr, return_inverse=True)
    med = np.empty(unique.size, dtype=np.float64)
    fade = float(fade_db)
    for i in range(unique.size):
        med[i] = float(np.median(cnr_arr[inv == i]))
    is_los = (cnr_arr >= med[inv] - fade).astype(np.int8)
    return prn_arr, tow_arr, is_los, az_arr, el_arr


def _cnr_proxy_rows_fast(
    obs_path: Path,
    eph: Ephemeris,
    rx: np.ndarray,
    tag: str,
    *,
    min_el_deg: float,
    fade_db: float,
    epoch_step: int = 2,
) -> list[dict]:
    from gnss_stations_proxy_cache import _rows_from_arrays

    arrays = _cnr_proxy_arrays_fast(
        obs_path,
        eph,
        rx,
        tag,
        min_el_deg=min_el_deg,
        fade_db=fade_db,
        epoch_step=epoch_step,
    )
    return _rows_from_arrays(tag, *arrays)


def _sticky_table(
    summary_rows: list[dict],
    consistency: list[dict],
    *,
    min_days: int,
    min_concentration: float,
    max_az_std: float,
) -> list[dict]:
    by_prn = {r["prn"]: r for r in consistency}
    rows: list[dict] = []
    for s in summary_rows:
        prn = s["prn"]
        c = by_prn.get(prn)
        if c is None:
            continue
        n_days = int(c["n_days_with_nlos"])
        conc = float(c["az_concentration_across_days"])
        az_std = float(c["az_std_across_days_deg"])
        if n_days < min_days or conc < min_concentration or az_std > max_az_std:
            continue
        rows.append(
            {
                "prn": prn,
                "fade_pct": round(100.0 * float(s["nlos_frac"]), 1),
                "az_fade_deg": round(float(s["mean_az_nlos_deg"]), 0) if s["mean_az_nlos_deg"] is not None else None,
                "n_days": n_days,
                "n_days_total": int(s["n_days"]),
                "az_std_across_days_deg": round(az_std, 1),
                "az_concentration": round(conc, 2),
                "mean_el_fade_deg": round(float(s["mean_el_nlos_deg"]), 1) if s["mean_el_nlos_deg"] is not None else None,
                "n_fade_samples": int(s["n_nlos"]),
                "risk_score": round(float(s.get("risk_score", 0.0)), 3),
            }
        )
    rows.sort(key=lambda r: (-r["fade_pct"], -r["az_concentration"]))
    return rows


def _analyze_station(
    station: str,
    obs_paths: list[Path],
    rx: np.ndarray,
    nav_paths: list[Path],
    *,
    min_el_deg: float,
    fade_db: float,
    epoch_step: int,
    cache_dir: Path | None,
    no_cache: bool,
    min_nlos_count: int,
    az_bin_deg: float,
    min_days: int,
    min_concentration: float,
    max_az_std: float,
) -> dict:
    from gnss_stations_proxy_cache import get_proxy_rows

    eph_cache: dict[Path, Ephemeris] = {}
    all_rows: list[dict] = []
    for obs_path in obs_paths:
        nav = risk._match_nav(obs_path, nav_paths)
        if nav is None:
            print(f"  skip {obs_path.name}: no NAV for same day")
            continue
        if nav not in eph_cache:
            eph_cache[nav] = Ephemeris(read_nav_rinex_multi(nav))
        tag = risk._tag_from_obs(obs_path)
        chunk = get_proxy_rows(
            obs_path,
            eph_cache[nav],
            rx,
            tag,
            min_el_deg=min_el_deg,
            fade_db=fade_db,
            epoch_step=epoch_step,
            cache_dir=cache_dir,
            no_cache=no_cache,
            compute_fn=_cnr_proxy_arrays_fast,
        )
        print(f"  {obs_path.name}: {len(chunk)} rows (nav={nav.name})")
        all_rows.extend(chunk)

    if not all_rows:
        return {"station": station, "error": "no data"}

    pooled, by_day = risk._aggregate_labels(all_rows)
    summary_rows = risk._rank_sticky_nlos(pooled, min_nlos=min_nlos_count)
    consistency = risk._day_consistency(by_day)
    az_bins = risk._az_bin_summary(all_rows, bin_deg=az_bin_deg, min_el=min_el_deg)
    sticky = _sticky_table(
        summary_rows,
        consistency,
        min_days=min_days,
        min_concentration=min_concentration,
        max_az_std=max_az_std,
    )

    day_rows = []
    for (tag, prn), st in sorted(by_day.items()):
        row = st.summary()
        row["tag"] = tag
        row["station"] = station
        day_rows.append(row)

    return {
        "station": station,
        "n_rows": len(all_rows),
        "n_prns": len(pooled),
        "n_days": len({r["tag"] for r in all_rows}),
        "summary_rows": summary_rows,
        "consistency": consistency,
        "sticky_prns": sticky,
        "az_bins": az_bins,
        "by_day": day_rows,
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _format_station_report(station: str, sticky: list[dict], n_days: int, fade_db: float) -> str:
    lines = [
        f"## {station} — proxy C/N0 (fade >= {fade_db:.0f} dB sotto mediana PRN, el >= 20 deg)",
        "",
        "Satelliti piu a rischio e sempre dalla stessa direzione:",
        "",
        "| PRN | % fade | Azimut | El fade | Giorni | az std | conc | Note |",
        "|-----|--------|--------|---------|--------|--------|------|------|",
    ]
    for r in sticky[:12]:
        note = f"{r['n_days']}/{n_days} giorni"
        if r["az_std_across_days_deg"] <= 2.0:
            note += f", az std {r['az_std_across_days_deg']:.1f} deg"
        el_s = f"{r['mean_el_fade_deg']:.0f} deg" if r.get("mean_el_fade_deg") is not None else "—"
        lines.append(
            f"| {r['prn']} | {r['fade_pct']:.1f}% | ~{r['az_fade_deg']:.0f} deg | {el_s} | "
            f"{r['n_days']}/{n_days} | {r['az_std_across_days_deg']:.1f} deg | "
            f"{r['az_concentration']:.2f} | {note} |"
        )
    if not sticky:
        lines.append("| — | — | — | — | — | — | — | nessun PRN sticky |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="Per-station sticky satellite fade risk.")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--crd-path", type=Path, required=True)
    p.add_argument("--terf-reference-csv", type=Path, default=Path("experiments/data/TERF/reference_2026192.csv"))
    p.add_argument("--output-dir", type=Path, default=Path("experiments/results/gnss_stations_sat_risk"))
    p.add_argument("--download-nav", action="store_true")
    p.add_argument("--min-el-deg", type=float, default=20.0)
    p.add_argument("--cnr-fade-db", type=float, default=6.0)
    p.add_argument("--epoch-step", type=int, default=2, help="Use every Nth OBS epoch (2=60s on 30s data)")
    p.add_argument("--cache-dir", type=Path, default=Path("experiments/results/gnss_stations_sat_risk/proxy_cache"))
    p.add_argument("--no-cache", action="store_true", help="Disable NPZ proxy cache")
    p.add_argument("--min-nlos-count", type=int, default=20)
    p.add_argument("--az-bin-deg", type=float, default=10.0)
    p.add_argument("--min-days", type=int, default=3, help="Min days with fades for sticky table")
    p.add_argument("--min-concentration", type=float, default=0.95)
    p.add_argument("--max-az-std-deg", type=float, default=5.0)
    args = p.parse_args()

    coords = _load_crd(args.crd_path)
    terf_rx = None
    if args.terf_reference_csv.is_file():
        terf_rx = risk._load_rx(args.terf_reference_csv)

    if args.download_nav:
        obs_for_nav = [Path(x) for x in sorted(glob(str(args.data_dir / "*_MO.rnx")))]
        for year, doy in _obs_days(obs_for_nav):
            print(f"NAV check {year}{doy:03d} ...")
            if download_brdc(year, doy, args.data_dir) is None:
                raise SystemExit(f"Could not download NAV for {year} DOY {doy:03d}")

    nav_paths = [Path(x) for x in sorted(glob(str(args.data_dir / "BRDC*.rnx")))]
    if not nav_paths:
        raise SystemExit(f"No BRDC in {args.data_dir}")

    groups = _group_obs(args.data_dir)
    if not groups:
        raise SystemExit(f"No OBS in {args.data_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    all_sticky: list[dict] = []
    report_md_parts = ["# GNSS Stations — sticky satellite fade risk\n"]

    for station in sorted(groups):
        obs_paths = groups[station]
        print(f"\n=== {station} ({len(obs_paths)} days) ===")
        if station in coords:
            rx = coords[station]
        elif station == "TERF" and terf_rx is not None:
            rx = terf_rx
        else:
            from analyze_gnss_stations_cnr_batch import _rx_from_obs_header

            try:
                rx = _rx_from_obs_header(obs_paths[0])
            except ValueError as exc:
                print(f"  skip: {exc}")
                continue

        result = _analyze_station(
            station,
            obs_paths,
            rx,
            nav_paths,
            min_el_deg=float(args.min_el_deg),
            fade_db=float(args.cnr_fade_db),
            epoch_step=int(args.epoch_step),
            cache_dir=None if args.no_cache else args.cache_dir,
            no_cache=bool(args.no_cache),
            min_nlos_count=int(args.min_nlos_count),
            az_bin_deg=float(args.az_bin_deg),
            min_days=min(int(args.min_days), max(1, len(obs_paths) - 1)),
            min_concentration=float(args.min_concentration),
            max_az_std=float(args.max_az_std_deg),
        )
        if "error" in result:
            print(f"  {result['error']}")
            continue

        out_st = args.output_dir / station
        out_st.mkdir(parents=True, exist_ok=True)
        for r in result["sticky_prns"]:
            r["station"] = station
        all_sticky.extend(result["sticky_prns"])

        _write_csv(out_st / "sat_risk_summary.csv", result["summary_rows"])
        _write_csv(out_st / "sat_risk_consistency.csv", result["consistency"])
        _write_csv(out_st / "sat_risk_sticky_prns.csv", result["sticky_prns"])
        _write_csv(out_st / "sat_risk_by_day.csv", result["by_day"])
        _write_csv(out_st / "sat_risk_az_bins.csv", result["az_bins"])

        report = {
            "station": station,
            "n_days": result["n_days"],
            "n_rows": result["n_rows"],
            "fade_db_threshold": float(args.cnr_fade_db),
            "sticky_prns": result["sticky_prns"],
            "top_summary": result["summary_rows"][:15],
            "top_az_bins": result["az_bins"][:10],
        }
        (out_st / "sat_risk_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

        md = _format_station_report(station, result["sticky_prns"], result["n_days"], float(args.cnr_fade_db))
        (out_st / "sat_risk_report.md").write_text(md, encoding="utf-8")
        report_md_parts.append(md)

        print(f"\n  Sticky PRNs ({station}):")
        for r in result["sticky_prns"][:8]:
            el_s = f"{r['mean_el_fade_deg']:.0f}" if r.get("mean_el_fade_deg") is not None else "—"
            print(
                f"    {r['prn']:4s}  fade={r['fade_pct']:5.1f}%  az~{r['az_fade_deg']:6.0f} deg  "
                f"el~{el_s} deg  {r['n_days']}/{result['n_days']} days  conc={r['az_concentration']:.2f}  "
                f"az_std={r['az_std_across_days_deg']:.1f} deg"
            )

    _write_csv(args.output_dir / "all_stations_sticky_prns.csv", all_sticky)
    (args.output_dir / "all_stations_report.md").write_text("\n".join(report_md_parts), encoding="utf-8")

    print(f"\nDone in {time.perf_counter() - t0:.1f}s -> {args.output_dir}")


if __name__ == "__main__":
    main()
