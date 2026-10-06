#!/usr/bin/env python3
"""Check if C/N0 fade times (GPS TOW) align across permanent stations.

For each sticky PRN, compares when fades occur (fade >= N dB below PRN median)
across stations on the same OBS days.

Example::

    python experiments/analyze_gnss_stations_fade_times.py \\
        --data-dir "experiments/data/GNSS Stations" \\
        --crd-path "experiments/data/GNSS Stations/NEW_CYCLOPS_STATIONS.CRD" \\
        --output-dir experiments/results/gnss_stations_sat_risk/fade_time_sync
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
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

import analyze_gnss_stations_sat_risk_batch as sat
import analyze_terf_satellite_risk as risk

_STATION_RE = re.compile(r"^([A-Z0-9]{4})00CYP_R_(\d{7})\d+_01D_", re.I)
GPS_EPOCH = datetime(1980, 1, 6, tzinfo=timezone.utc)


def _group_obs(data_dir: Path) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for p in sorted(data_dir.glob("*00CYP_R_*_MO.rnx")):
        m = _STATION_RE.match(p.name)
        if m:
            groups[m.group(1).upper()].append(p)
    return dict(groups)


def _fade_events_by_prn(
    rows: list[dict],
) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if int(r["is_los"]) == 0:
            out[r["prn"]].append(
                {
                    "tag": r["tag"],
                    "tow": float(r["tow"]),
                    "az": float(r["az"]),
                    "el": float(r["el"]),
                }
            )
    return out


def _tow_to_utc_hour(tow: float, tag: str) -> float:
    """Approx UTC hour-of-day from GPS week tag + TOW (30s OBS, no leap-sec fine)."""
    year = int(tag[:4])
    doy = int(tag[4:7])
    dt0 = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=doy - 1)
    dt = GPS_EPOCH + timedelta(seconds=float(tow)) + (dt0 - GPS_EPOCH)
    # tag anchors calendar day; TOW is within GPS week — use modulo for time-of-day
    sec = float(tow) % 86400.0
    return sec / 3600.0


def _bin_tow(tow: float, bin_s: float) -> int:
    return int(round(float(tow) / bin_s))


def _overlap_fraction(tows_a: np.ndarray, tows_b: np.ndarray, *, bin_s: float) -> float:
    if tows_a.size == 0 or tows_b.size == 0:
        return 0.0
    ba = {_bin_tow(t, bin_s) for t in tows_a}
    bb = {_bin_tow(t, bin_s) for t in tows_b}
    if not ba or not bb:
        return 0.0
    inter = len(ba & bb)
    union = len(ba | bb)
    return inter / union if union else 0.0


def _mean_pairwise_overlap(station_tows: dict[str, np.ndarray], *, bin_s: float) -> float:
    names = sorted(station_tows)
    if len(names) < 2:
        return 0.0
    vals = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            vals.append(_overlap_fraction(station_tows[a], station_tows[b], bin_s=bin_s))
    return float(np.mean(vals)) if vals else 0.0


def _fade_mask(prn: np.ndarray, is_los: np.ndarray, target: str) -> np.ndarray:
    return (prn == target) & (is_los == 0)


def main() -> None:
    p = argparse.ArgumentParser(description="Compare fade times across GNSS stations.")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--crd-path", type=Path, required=True)
    p.add_argument("--terf-reference-csv", type=Path, default=Path("experiments/data/TERF/reference_2026192.csv"))
    p.add_argument("--output-dir", type=Path, default=Path("experiments/results/gnss_stations_sat_risk/fade_time_sync"))
    p.add_argument("--min-el-deg", type=float, default=20.0)
    p.add_argument("--cnr-fade-db", type=float, default=6.0)
    p.add_argument("--epoch-step", type=int, default=2, help="Use every Nth OBS epoch (2=60s on 30s data)")
    p.add_argument("--cache-dir", type=Path, default=Path("experiments/results/gnss_stations_sat_risk/proxy_cache"))
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--tow-bin-s", type=float, default=60.0, help="TOW bin for overlap (default 60s)")
    p.add_argument("--prns", type=str, default="G08,G24,G27,G32,G18", help="Comma-separated PRNs to check")
    args = p.parse_args()

    coords = sat._load_crd(args.crd_path)
    terf_rx = risk._load_rx(args.terf_reference_csv) if args.terf_reference_csv.is_file() else None
    nav_paths = [Path(x) for x in sorted(glob(str(args.data_dir / "BRDC*.rnx")))]
    groups = _group_obs(args.data_dir)
    prns = [x.strip().upper() for x in args.prns.split(",") if x.strip()]

    # station -> tag -> arrays (prn, tow, is_los, az, el)
    data: dict[str, dict[str, tuple[np.ndarray, ...]]] = {}
    eph_cache: dict[Path, Ephemeris] = {}

    from gnss_stations_proxy_cache import get_proxy_arrays

    t0 = time.perf_counter()
    for station, obs_paths in sorted(groups.items()):
        if station in coords:
            rx = coords[station]
        elif station == "TERF" and terf_rx is not None:
            rx = terf_rx
        else:
            from analyze_gnss_stations_cnr_batch import _rx_from_obs_header

            rx = _rx_from_obs_header(obs_paths[0])
        data[station] = {}
        for obs_path in obs_paths:
            nav = risk._match_nav(obs_path, nav_paths)
            if nav is None:
                continue
            if nav not in eph_cache:
                eph_cache[nav] = Ephemeris(read_nav_rinex_multi(nav))
            tag = risk._tag_from_obs(obs_path)
            prn, tow, is_los, az, el = get_proxy_arrays(
                obs_path,
                eph_cache[nav],
                rx,
                tag,
                min_el_deg=float(args.min_el_deg),
                fade_db=float(args.cnr_fade_db),
                epoch_step=int(args.epoch_step),
                cache_dir=None if args.no_cache else args.cache_dir,
                no_cache=bool(args.no_cache),
                compute_fn=sat._cnr_proxy_arrays_fast,
            )
            data[station][tag] = (prn, tow, is_los, az, el)
            print(f"  {station} {tag}: {tow.size} rows")

    tags = sorted({tag for st in data.values() for tag in st})
    stations = sorted(data.keys())

    prn_reports = []
    overlap_rows = []
    detail_rows = []

    for prn in prns:
        # Per day: collect fade TOW per station
        day_overlaps = []
        station_all_tows: dict[str, list[float]] = {s: [] for s in stations}

        for tag in tags:
            st_tows: dict[str, np.ndarray] = {}
            for station in stations:
                packed = data.get(station, {}).get(tag)
                if packed is None:
                    st_tows[station] = np.array([], dtype=np.float64)
                    continue
                prn_arr, tow_arr, is_los_arr, az_arr, el_arr = packed
                mask = _fade_mask(prn_arr, is_los_arr, prn)
                tows = tow_arr[mask]
                st_tows[station] = tows
                station_all_tows[station].extend(tows.tolist())
                if mask.any():
                    for tow_v, az_v, el_v in zip(tow_arr[mask], az_arr[mask], el_arr[mask]):
                        detail_rows.append(
                            {
                                "prn": prn,
                                "tag": tag,
                                "station": station,
                                "tow_s": round(float(tow_v), 1),
                                "utc_hour": round(_tow_to_utc_hour(float(tow_v), tag), 3),
                                "az_deg": round(float(az_v), 1),
                                "el_deg": round(float(el_v), 1),
                            }
                        )
            ov = _mean_pairwise_overlap(st_tows, bin_s=float(args.tow_bin_s))
            day_overlaps.append({"tag": tag, "overlap_jaccard": round(ov, 3)})
            overlap_rows.append(
                {
                    "prn": prn,
                    "tag": tag,
                    "mean_pairwise_tow_overlap": round(ov, 3),
                    "n_stations": len([s for s in stations if st_tows.get(s, np.array([])).size > 0]),
                }
            )

        pooled_overlap = _mean_pairwise_overlap(
            {s: np.asarray(v, dtype=np.float64) for s, v in station_all_tows.items()},
            bin_s=float(args.tow_bin_s),
        )

        # Hour-of-day per station (pooled all days)
        st_hours: dict[str, np.ndarray] = {}
        for station in stations:
            hrs: list[float] = []
            for tag in tags:
                packed = data.get(station, {}).get(tag)
                if packed is None:
                    continue
                prn_arr, tow_arr, is_los_arr, _, _ = packed
                mask = _fade_mask(prn_arr, is_los_arr, prn)
                if mask.any():
                    sec = np.mod(tow_arr[mask].astype(np.float64), 86400.0) / 3600.0
                    hrs.extend(sec.tolist())
            st_hours[station] = np.asarray(hrs, dtype=np.float64)

        mean_hours = {s: float(np.mean(h)) if h.size else None for s, h in st_hours.items()}
        std_hours = {s: float(np.std(h)) if h.size else None for s, h in st_hours.items()}

        prn_reports.append(
            {
                "prn": prn,
                "pooled_mean_pairwise_overlap": round(pooled_overlap, 3),
                "per_day_overlap": day_overlaps,
                "mean_fade_utc_hour_by_station": mean_hours,
                "std_fade_utc_hour_by_station": std_hours,
                "n_fade_events_by_station": {s: len(station_all_tows[s]) for s in stations},
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "fade_time_overlap_by_prn_day.csv", "w", newline="", encoding="utf-8") as fh:
        if overlap_rows:
            w = csv.DictWriter(fh, fieldnames=list(overlap_rows[0].keys()))
            w.writeheader()
            w.writerows(overlap_rows)

    with open(args.output_dir / "fade_time_events.csv", "w", newline="", encoding="utf-8") as fh:
        if detail_rows:
            w = csv.DictWriter(fh, fieldnames=list(detail_rows[0].keys()))
            w.writeheader()
            w.writerows(detail_rows)

    summary = {
        "stations": stations,
        "tags": tags,
        "tow_bin_s": float(args.tow_bin_s),
        "fade_db": float(args.cnr_fade_db),
        "prn_reports": prn_reports,
        "interpretation": (
            "overlap_jaccard ~1 = fade bins coincide across stations; "
            "~0 = fades at different times. Same PRN pass times differ by geometry; "
            "high overlap suggests common obstruction window."
        ),
    }
    (args.output_dir / "fade_time_sync_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nDone in {time.perf_counter() - t0:.1f}s")
    print("\nPooled TOW overlap (Jaccard on 60s bins) — same fade timing across stations?")
    for r in prn_reports:
        print(f"\n  {r['prn']}  pooled_overlap={r['pooled_mean_pairwise_overlap']:.2f}")
        for tag, ov in [(d["tag"], d["overlap_jaccard"]) for d in r["per_day_overlap"]]:
            print(f"    day {tag}: overlap={ov:.2f}")
        print("    mean fade UTC hour by station:")
        for st in stations:
            mh = r["mean_fade_utc_hour_by_station"].get(st)
            sh = r["std_fade_utc_hour_by_station"].get(st)
            n = r["n_fade_events_by_station"].get(st, 0)
            if mh is not None:
                print(f"      {st:4s}  n={n:4d}  mean_hour={mh:5.1f}h  std={sh:.1f}h")


if __name__ == "__main__":
    main()
