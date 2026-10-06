#!/usr/bin/env python3
"""Per-station C/N0 skyplot (one day): tracks colored by PRN-normalized C/N0.

Normalization: C/N0 - median(C/N0) per satellite for that day (el >= min_el).

Example::

    python experiments/plot_gnss_stations_cnr_skyplot_batch.py \\
        --data-dir "experiments/data/GNSS Stations" \\
        --crd-path "experiments/data/GNSS Stations/NEW_CYCLOPS_STATIONS.CRD" \\
        --skyplot-day 2026192 \\
        --output-dir experiments/results/gnss_stations_sat_risk
"""

from __future__ import annotations

import argparse
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

import analyze_terf_satellite_risk as risk
import plot_terf_cnr_skyplot as sky

_STATION_RE = re.compile(r"^([A-Z0-9]{4})00CYP_R_(\d{7})\d+_01D_", re.I)
_CRD_RE = re.compile(
    r"^\s*\d+\s+([A-Z0-9]{4})\s+\S+\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)",
    re.I,
)
GPS_EPOCH = datetime(1980, 1, 6)


def _load_crd(path: Path) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = _CRD_RE.match(line)
        if not m:
            continue
        code = m.group(1).upper()
        out[code] = np.array([float(m.group(2)), float(m.group(3)), float(m.group(4))], dtype=np.float64)
    return out


def _collect_tracks_fast(
    obs_path: Path,
    eph: Ephemeris,
    rx: np.ndarray,
    *,
    min_el_deg: float,
    max_el_deg: float,
    epoch_step: int,
) -> dict[str, dict[str, np.ndarray]]:
    from gnss_gpu.io.rinex import read_rinex_obs
    from gnss_gpu.urban_signal_sim import _sat_elevation_azimuth

    step = max(1, int(epoch_step))
    obs = read_rinex_obs(obs_path)
    raw: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)

    for ei, ep in enumerate(obs.epochs):
        if step > 1 and (ei % step) != 0:
            continue
        tow = float((ep.time - GPS_EPOCH).total_seconds() % 604800.0)
        prns: list[str] = []
        cnrs: list[float] = []
        for sat_id, odict in ep.observations.items():
            cnr = sky._preferred_cnr(odict)
            if cnr is None:
                continue
            prns.append(sky._normalize_prn(str(sat_id)))
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
            el = float(el_deg[k])
            if el < min_el_deg or el > max_el_deg:
                continue
            raw[sel_prn[k]].append((tow, float(az_deg[k]), el, float(sel_cnr[k])))

    tracks: dict[str, dict[str, np.ndarray]] = {}
    for prn, rows in raw.items():
        rows.sort(key=lambda x: x[0])
        cnr = np.array([r[3] for r in rows], dtype=np.float64)
        med = float(np.median(cnr))
        tracks[prn] = {
            "tow": np.array([r[0] for r in rows], dtype=np.float64),
            "az": np.array([r[1] for r in rows], dtype=np.float64),
            "el": np.array([r[2] for r in rows], dtype=np.float64),
            "cnr": cnr - med,
        }
    return tracks


def main() -> None:
    p = argparse.ArgumentParser(description="Per-station PRN-normalized C/N0 skyplot.")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--crd-path", type=Path, required=True)
    p.add_argument("--terf-reference-csv", type=Path, default=Path("experiments/data/TERF/reference_2026192.csv"))
    p.add_argument("--output-dir", type=Path, default=Path("experiments/results/gnss_stations_sat_risk"))
    p.add_argument("--skyplot-day", type=str, default="2026192", help="OBS tag YYYYDOY")
    p.add_argument("--min-el-deg", type=float, default=20.0)
    p.add_argument("--max-el-deg", type=float, default=85.0)
    p.add_argument("--epoch-step", type=int, default=2)
    p.add_argument("--residual-clip-db", type=float, default=8.0)
    p.add_argument("--linewidth", type=float, default=2.0)
    args = p.parse_args()

    coords = _load_crd(args.crd_path)
    terf_rx = risk._load_rx(args.terf_reference_csv) if args.terf_reference_csv.is_file() else None
    nav_paths = [Path(x) for x in sorted(glob(str(args.data_dir / "BRDC*.rnx")))]
    if not nav_paths:
        raise SystemExit(f"No BRDC in {args.data_dir}")

    day_tag = str(args.skyplot_day).strip()
    obs_by_station: dict[str, Path] = {}
    for pth in sorted(args.data_dir.glob("*00CYP_R_*_MO.rnx")):
        m = _STATION_RE.match(pth.name)
        if not m or m.group(2) != day_tag:
            continue
        obs_by_station[m.group(1).upper()] = pth

    if not obs_by_station:
        raise SystemExit(f"No OBS for day {day_tag} in {args.data_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    for station in sorted(obs_by_station):
        obs_path = obs_by_station[station]
        nav = risk._match_nav(obs_path, nav_paths)
        if nav is None:
            print(f"skip {station}: no NAV for {day_tag}")
            continue
        if station in coords:
            rx = coords[station]
        elif station == "TERF" and terf_rx is not None:
            rx = terf_rx
        else:
            from analyze_gnss_stations_cnr_batch import _rx_from_obs_header

            rx = _rx_from_obs_header(obs_path)

        print(f"{station}: {obs_path.name} + {nav.name}")
        eph = Ephemeris(read_nav_rinex_multi(nav))
        tracks = _collect_tracks_fast(
            obs_path,
            eph,
            rx,
            min_el_deg=float(args.min_el_deg),
            max_el_deg=float(args.max_el_deg),
            epoch_step=int(args.epoch_step),
        )
        n_pts = sum(int(d["tow"].size) for d in tracks.values())
        out_dir = args.output_dir / station
        out_dir.mkdir(parents=True, exist_ok=True)
        out_png = out_dir / f"cnr_skyplot_prn_norm_{day_tag}.png"
        clip = float(args.residual_clip_db)
        sky._plot_tracks(
            tracks,
            out_png,
            title=f"{station} — C/N0 - PRN median ({day_tag})\nel {args.min_el_deg:g}-{args.max_el_deg:g} deg",
            cmap="coolwarm",
            vmin=-clip,
            vmax=clip,
            linewidth=float(args.linewidth),
            max_tow_gap_s=120.0,
            mark_points=False,
            cnr_mode="normalized",
            cbar_label="C/N0 - PRN median [dB-Hz]",
        )
        print(f"  {len(tracks)} PRNs, {n_pts} pts -> {out_png}")

    print(f"Done in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
