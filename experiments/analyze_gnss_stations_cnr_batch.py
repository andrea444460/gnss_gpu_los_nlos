#!/usr/bin/env python3
"""Elevation-normalized C/N0 residual analysis for multiple permanent GNSS stations.

Pools all OBS days per station, subtracts elevation model, ranks PRN spikes and
azimuth bins with negative residuals (obstruction / fade proxy).

Example::

    python experiments/analyze_gnss_stations_cnr_batch.py \\
        --data-dir "experiments/data/GNSS Stations" \\
        --crd-path "experiments/data/GNSS Stations/NEW_CYCLOPS_STATIONS.CRD" \\
        --nav-glob "experiments/data/GNSS Stations/BRDC*.rnx" \\
        --output-dir experiments/results/gnss_stations_cnr
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import defaultdict
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

import analyze_terf_cnr_residual_fft as cnr
from download_brdc_nav import download_brdc, _obs_days

_STATION_RE = re.compile(r"^([A-Z0-9]{4})00CYP_R_(\d{7})\d+_01D_", re.I)
_CRD_RE = re.compile(
    r"^\s*\d+\s+([A-Z0-9]{4})\s+\S+\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)",
    re.I,
)


def _load_crd(path: Path) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = _CRD_RE.match(line)
        if not m:
            continue
        code = m.group(1).upper()
        out[code] = np.array([float(m.group(2)), float(m.group(3)), float(m.group(4))], dtype=np.float64)
    return out


def _load_terf_reference(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    return cnr._load_rx(path)


def _rx_from_obs_header(obs_path: Path) -> np.ndarray:
    from gnss_gpu.io.rinex import read_rinex_obs

    obs = read_rinex_obs(obs_path)
    xyz = np.asarray(obs.header.approx_position, dtype=np.float64).reshape(3)
    if np.all(np.isfinite(xyz)) and np.linalg.norm(xyz) > 1e6:
        return xyz
    raise ValueError(f"No valid APPROX POSITION in {obs_path.name}")


def _group_obs(data_dir: Path) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for p in sorted(data_dir.glob("*_R_*_MO.rnx")):
        m = _STATION_RE.match(p.name)
        if not m:
            continue
        groups[m.group(1).upper()].append(p)
    return dict(groups)


def _az_bin_residuals(
    az: np.ndarray,
    el: np.ndarray,
    residual: np.ndarray,
    *,
    bin_deg: float,
    min_el: float,
    spike_db: float,
) -> list[dict]:
    n_bins = max(1, int(round(360.0 / bin_deg)))
    bins: dict[int, dict] = defaultdict(lambda: {"n": 0, "n_spike": 0, "res": [], "el": []})
    for a, e, r in zip(az, el, residual):
        if float(e) < min_el:
            continue
        b = int(float(a) // bin_deg) % n_bins
        bins[b]["n"] += 1
        bins[b]["res"].append(float(r))
        bins[b]["el"].append(float(e))
        if float(r) < -spike_db:
            bins[b]["n_spike"] += 1
    rows = []
    for b in sorted(bins):
        d = bins[b]
        n = d["n"]
        res = np.asarray(d["res"], dtype=np.float64)
        rows.append(
            {
                "az_lo_deg": b * bin_deg,
                "az_hi_deg": (b + 1) * bin_deg,
                "n_samples": n,
                "n_spike_below_db": d["n_spike"],
                "spike_frac": d["n_spike"] / n if n else 0.0,
                "mean_residual_db": float(np.mean(res)),
                "median_residual_db": float(np.median(res)),
                "min_residual_db": float(np.min(res)),
                "mean_el_deg": float(np.mean(d["el"])),
            }
        )
    rows.sort(key=lambda r: (-r["spike_frac"], r["min_residual_db"]))
    return rows


def _analyze_station(
    station: str,
    obs_paths: list[Path],
    rx: np.ndarray,
    nav_paths: list[Path],
    *,
    min_el: float,
    max_el: float,
    el_bin_deg: float,
    epoch_step: int,
    spike_db: float,
    az_bin_deg: float,
) -> dict:
    eph_cache: dict[Path, Ephemeris] = {}
    all_tag: list[np.ndarray] = []
    all_prn: list[np.ndarray] = []
    all_tow: list[np.ndarray] = []
    all_az: list[np.ndarray] = []
    all_el: list[np.ndarray] = []
    all_cnr: list[np.ndarray] = []
    global_prn: dict[str, int] = {}
    tag_names: list[str] = []

    for obs_path in obs_paths:
        nav = cnr._match_nav(obs_path, nav_paths)
        if nav is None:
            print(f"  skip {obs_path.name}: no NAV")
            continue
        if nav not in eph_cache:
            eph_cache[nav] = Ephemeris(read_nav_rinex_multi(nav))
        arrays, local_prn = cnr._collect_samples(
            obs_path,
            eph_cache[nav],
            rx,
            min_el=min_el,
            max_el=max_el,
            epoch_step=epoch_step,
        )
        if arrays[2].size == 0:
            print(f"  {obs_path.name}: 0 samples")
            continue
        tag = cnr._tag_from_obs(obs_path.name)
        tag_id = len(tag_names)
        tag_names.append(tag)
        _, prn_idx, tow, az, el, cnr_arr = arrays
        prn_codes = np.empty(prn_idx.size, dtype=np.int16)
        for prn, code in local_prn.items():
            if prn not in global_prn:
                global_prn[prn] = len(global_prn)
            prn_codes[prn_idx == code] = global_prn[prn]
        all_tag.append(np.full(tow.size, tag_id, dtype=np.int16))
        all_prn.append(prn_codes)
        all_tow.append(tow)
        all_az.append(az)
        all_el.append(el)
        all_cnr.append(cnr_arr)
        print(f"  {obs_path.name}: {tow.size} samples (nav={nav.name})")

    if not all_tow:
        return {"station": station, "error": "no samples"}

    tag_arr = np.concatenate(all_tag)
    prn_arr = np.concatenate(all_prn)
    tow_arr = np.concatenate(all_tow)
    az_arr = np.concatenate(all_az)
    el_arr = np.concatenate(all_el)
    cnr_arr = np.concatenate(all_cnr)
    code_to_prn = [""] * len(global_prn)
    for prn, code in global_prn.items():
        code_to_prn[code] = prn

    centers, medians = cnr._build_el_model(el_arr, cnr_arr, el_bin_deg=el_bin_deg, min_count=50)
    fallback = float(np.median(cnr_arr))
    model_cnr = cnr._interp_model(el_arr, centers, medians, fallback)
    residual = cnr_arr - model_cnr

    spike_rows = []
    for code, prn in enumerate(code_to_prn):
        m = prn_arr == code
        v = residual[m]
        if v.size == 0:
            continue
        az_v = az_arr[m]
        spike_rows.append(
            {
                "station": station,
                "prn": prn,
                "n_samples": int(v.size),
                "mean_residual_db": float(np.mean(v)),
                "std_residual_db": float(np.std(v)),
                "min_residual_db": float(np.min(v)),
                "frac_below_minus3db": float(np.mean(v < -3.0)),
                "n_spike_samples": int(np.count_nonzero(v < -spike_db)),
                "mean_az_spike_deg": float(np.mean(az_v[v < -spike_db])) if np.any(v < -spike_db) else None,
            }
        )
    spike_rows.sort(key=lambda r: (-r["frac_below_minus3db"], r["min_residual_db"]))

    az_rows = _az_bin_residuals(
        az_arr, el_arr, residual, bin_deg=az_bin_deg, min_el=min_el, spike_db=spike_db
    )
    for r in az_rows:
        r["station"] = station

    by_day = []
    for ti, tname in enumerate(tag_names):
        m = tag_arr == ti
        v = residual[m]
        by_day.append(
            {
                "station": station,
                "tag": tname,
                "n_samples": int(v.size),
                "mean_residual_db": float(np.mean(v)),
                "frac_below_minus3db": float(np.mean(v < -3.0)),
                "min_residual_db": float(np.min(v)),
            }
        )

    top_az = az_rows[0] if az_rows else {}
    return {
        "station": station,
        "n_samples": int(tow_arr.size),
        "n_days": len(tag_names),
        "tags": tag_names,
        "rx_ecef": [float(x) for x in rx],
        "spike_rows": spike_rows,
        "az_rows": az_rows,
        "by_day": by_day,
        "top_az_bin": top_az,
        "top_spike_prns": spike_rows[:10],
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Multi-station C/N0 residual spike analysis.")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--crd-path", type=Path, required=True)
    p.add_argument("--nav-glob", type=str, default="", help="Default: <data-dir>/BRDC*.rnx")
    p.add_argument("--download-nav", action="store_true", help="Download missing BRDC for each OBS day")
    p.add_argument("--terf-reference-csv", type=Path, default=Path("experiments/data/TERF/reference_2026192.csv"))
    p.add_argument("--output-dir", type=Path, default=Path("experiments/results/gnss_stations_cnr"))
    p.add_argument("--min-el-deg", type=float, default=20.0)
    p.add_argument("--max-el-deg", type=float, default=85.0)
    p.add_argument("--el-bin-deg", type=float, default=2.0)
    p.add_argument("--epoch-step", type=int, default=2)
    p.add_argument("--spike-db", type=float, default=3.0)
    p.add_argument("--az-bin-deg", type=float, default=10.0)
    args = p.parse_args()

    coords = _load_crd(args.crd_path)
    terf_rx = _load_terf_reference(args.terf_reference_csv)
    nav_glob = str(args.nav_glob).strip() or str(args.data_dir / "BRDC*.rnx")

    if args.download_nav:
        obs_for_nav = [Path(x) for x in sorted(glob(str(args.data_dir / "*_MO.rnx")))]
        for year, doy in _obs_days(obs_for_nav):
            print(f"NAV check {year}{doy:03d} ...")
            if download_brdc(year, doy, args.data_dir) is None:
                raise SystemExit(f"Could not download NAV for {year} DOY {doy:03d}")

    nav_paths = [Path(x) for x in sorted(glob(nav_glob))]
    if not nav_paths:
        raise SystemExit(f"No NAV matched {args.nav_glob}")

    groups = _group_obs(args.data_dir)
    if not groups:
        raise SystemExit(f"No OBS in {args.data_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    station_summaries: list[dict] = []
    all_spike: list[dict] = []
    all_az: list[dict] = []

    for station in sorted(groups):
        obs_paths = groups[station]
        print(f"\n=== {station} ({len(obs_paths)} days) ===")
        if station in coords:
            rx = coords[station]
        elif station == "TERF" and terf_rx is not None:
            rx = terf_rx
        else:
            try:
                rx = _rx_from_obs_header(obs_paths[0])
                print(f"  RX from RINEX header: {rx.tolist()}")
            except ValueError as exc:
                print(f"  skip {station}: {exc}")
                continue

        result = _analyze_station(
            station,
            obs_paths,
            rx,
            nav_paths,
            min_el=float(args.min_el_deg),
            max_el=float(args.max_el_deg),
            el_bin_deg=float(args.el_bin_deg),
            epoch_step=int(args.epoch_step),
            spike_db=float(args.spike_db),
            az_bin_deg=float(args.az_bin_deg),
        )
        if "error" in result:
            print(f"  {result['error']}")
            continue

        out_st = args.output_dir / station
        out_st.mkdir(parents=True, exist_ok=True)
        spike_csv = out_st / "cnr_residual_spike_ranking.csv"
        az_csv = out_st / "cnr_residual_az_bins.csv"
        day_csv = out_st / "cnr_residual_by_day.csv"

        for path, rows in (
            (spike_csv, result["spike_rows"]),
            (az_csv, result["az_rows"]),
            (day_csv, result["by_day"]),
        ):
            if rows:
                with open(path, "w", newline="", encoding="utf-8") as fh:
                    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                    w.writeheader()
                    w.writerows(rows)

        report = {
            "station": station,
            "n_samples": result["n_samples"],
            "n_days": result["n_days"],
            "tags": result["tags"],
            "rx_ecef": result["rx_ecef"],
            "top_spike_prns": result["top_spike_prns"],
            "top_az_bins": result["az_rows"][:10],
        }
        (out_st / "cnr_residual_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

        top_az = result["top_az_bin"]
        station_summaries.append(
            {
                "station": station,
                "n_days": result["n_days"],
                "n_samples": result["n_samples"],
                "frac_below_minus3db": float(
                    np.mean([r["frac_below_minus3db"] for r in result["spike_rows"]])
                )
                if result["spike_rows"]
                else 0.0,
                "top_az_lo_deg": top_az.get("az_lo_deg"),
                "top_az_hi_deg": top_az.get("az_hi_deg"),
                "top_az_spike_frac": top_az.get("spike_frac"),
                "top_az_min_residual_db": top_az.get("min_residual_db"),
                "top_prn": result["top_spike_prns"][0]["prn"] if result["top_spike_prns"] else None,
                "top_prn_spike_frac": result["top_spike_prns"][0]["frac_below_minus3db"]
                if result["top_spike_prns"]
                else None,
            }
        )
        all_spike.extend(result["spike_rows"])
        all_az.extend(result["az_rows"])

        print(f"  samples={result['n_samples']} days={result['n_days']}")
        if result["top_spike_prns"]:
            t = result["top_spike_prns"][0]
            print(
                f"  top PRN {t['prn']}: spike_frac={t['frac_below_minus3db']*100:.1f}% "
                f"min={t['min_residual_db']:.1f} dB"
            )
        if top_az:
            print(
                f"  top az bin {top_az['az_lo_deg']:.0f}-{top_az['az_hi_deg']:.0f}: "
                f"spike_frac={top_az['spike_frac']*100:.1f}% min={top_az['min_residual_db']:.1f} dB"
            )

    compare_csv = args.output_dir / "stations_comparison.csv"
    if station_summaries:
        with open(compare_csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(station_summaries[0].keys()))
            w.writeheader()
            w.writerows(station_summaries)

    if all_spike:
        with open(args.output_dir / "all_stations_spike_ranking.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(all_spike[0].keys()))
            w.writeheader()
            w.writerows(all_spike)

    if all_az:
        with open(args.output_dir / "all_stations_az_bins.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(all_az[0].keys()))
            w.writeheader()
            w.writerows(all_az)

    summary = {
        "stations": station_summaries,
        "elapsed_s": time.perf_counter() - t0,
        "outputs": {
            "comparison_csv": str(compare_csv),
            "per_station_dirs": str(args.output_dir / "<STATION>"),
        },
    }
    (args.output_dir / "batch_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nDone in {summary['elapsed_s']:.1f}s -> {args.output_dir}")
    print("\nStation comparison (top az bin with most negative residuals):")
    for r in station_summaries:
        print(
            f"  {r['station']:4s}  days={r['n_days']}  samples={r['n_samples']:6d}  "
            f"top_az={r['top_az_lo_deg']:.0f}-{r['top_az_hi_deg']:.0f} deg  "
            f"spike_frac={r['top_az_spike_frac']*100:5.1f}%  "
            f"top_prn={r['top_prn']}"
        )


if __name__ == "__main__":
    main()
