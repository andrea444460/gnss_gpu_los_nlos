#!/usr/bin/env python3
"""Export geometry-based LOS/NLOS labels from UrbanNav RINEX epochs.

This utility aligns RINEX observation epochs with receiver trajectory samples
from ``reference.csv`` and computes per-satellite LOS/NLOS labels using:

  1) broadcast ephemeris from NAV RINEX,
  2) PLATEAU 3D buildings (with BVH acceleration),
  3) geometric ray-tracing in ``UrbanSignalSimulator``.

Output is a flat CSV suitable for ML evaluation/benchmarking.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "python"))

from gnss_gpu.bvh import BVHAccelerator
from gnss_gpu.ephemeris import Ephemeris
from gnss_gpu.io.nav_rinex import read_nav_rinex_multi
from gnss_gpu.io.plateau import PlateauLoader
from gnss_gpu.io.rinex import read_rinex_obs
from gnss_gpu.urban_signal_sim import UrbanSignalSimulator

GPS_EPOCH = datetime(1980, 1, 6)
GPS_WEEK_SECONDS = 604800.0


@dataclass
class ReferenceTrack:
    weeks: np.ndarray
    tows: np.ndarray
    rx_ecef: np.ndarray


def _datetime_to_gps_week_tow(dt: datetime) -> tuple[int, float]:
    delta_s = (dt - GPS_EPOCH).total_seconds()
    week = int(delta_s // GPS_WEEK_SECONDS)
    tow = float(delta_s - week * GPS_WEEK_SECONDS)
    return week, tow


def _normalize_row_keys(row: dict[str, str]) -> dict[str, str]:
    return {k.strip(): v for k, v in row.items()}


def _load_reference_track(reference_csv: Path) -> ReferenceTrack:
    weeks: list[int] = []
    tows: list[float] = []
    rx: list[list[float]] = []
    with open(reference_csv, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            r = _normalize_row_keys(row)
            weeks.append(int(float(r["GPS Week"])))
            tows.append(float(r["GPS TOW (s)"]))
            rx.append(
                [
                    float(r["ECEF X (m)"]),
                    float(r["ECEF Y (m)"]),
                    float(r["ECEF Z (m)"]),
                ]
            )
    if not tows:
        raise ValueError(f"reference track is empty: {reference_csv}")
    return ReferenceTrack(
        weeks=np.asarray(weeks, dtype=np.int32),
        tows=np.asarray(tows, dtype=np.float64),
        rx_ecef=np.asarray(rx, dtype=np.float64),
    )


def _nearest_reference_xyz(track: ReferenceTrack, week: int, tow: float) -> np.ndarray | None:
    mask = track.weeks == week
    if not np.any(mask):
        return None
    tows = track.tows[mask]
    idx_local = int(np.argmin(np.abs(tows - tow)))
    idx_global = np.where(mask)[0][idx_local]
    return track.rx_ecef[idx_global]


def _parse_sat_id(sat_id: str) -> tuple[str, int] | None:
    sat = sat_id.strip().upper()
    if len(sat) < 2:
        return None
    system = sat[0]
    prn_text = sat[1:]
    if not prn_text.isdigit():
        return None
    return system, int(prn_text)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export geometric LOS/NLOS labels for UrbanNav RINEX epochs")
    parser.add_argument("--obs-path", type=Path, required=True, help="RINEX observation file (.obs)")
    parser.add_argument("--nav-path", type=Path, required=True, help="RINEX navigation file (.nav)")
    parser.add_argument("--reference-csv", type=Path, required=True, help="UrbanNav reference.csv (ECEF trajectory)")
    parser.add_argument("--plateau-dir", type=Path, required=True, help="Directory containing PLATEAU .gml tiles")
    parser.add_argument("--output-csv", type=Path, required=True, help="Output labels CSV path")
    parser.add_argument("--obs-code", type=str, default="C1C", help="Observation code to require in OBS file")
    parser.add_argument(
        "--systems",
        type=str,
        default="G",
        help="Comma-separated systems to label (e.g. G or G,E,J)",
    )
    parser.add_argument("--plateau-zone", type=int, default=9, help="PLATEAU plane-rectangular coordinate zone")
    parser.add_argument("--epoch-step", type=int, default=1, help="Use every Nth epoch from OBS")
    parser.add_argument("--max-epochs", type=int, default=0, help="0 means all, otherwise cap processed epochs")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    systems = tuple(s.strip().upper() for s in args.systems.split(",") if s.strip())

    print("[1/5] Loading input files...")
    obs = read_rinex_obs(args.obs_path)
    nav_messages = read_nav_rinex_multi(args.nav_path, systems=systems)
    eph = Ephemeris(nav_messages)
    track = _load_reference_track(args.reference_csv)
    print(f"  epochs in OBS: {len(obs.epochs)}")
    print(f"  ephemeris satellites: {len(eph.available_prns)}")
    print(f"  reference samples: {len(track.tows)}")

    print("[2/5] Loading PLATEAU and building BVH...")
    loader = PlateauLoader(zone=args.plateau_zone)
    building = loader.load_directory(str(args.plateau_dir))
    bvh = BVHAccelerator.from_building_model(building)
    usim = UrbanSignalSimulator(building_model=bvh, noise_floor_db=-35.0)
    print(f"  triangles: {len(building.triangles)}")
    print(f"  bvh nodes: {bvh.n_nodes}")

    print("[3/5] Processing epochs...")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    processed_epochs = 0
    skipped_no_rx = 0
    skipped_no_obs = 0
    total_epochs = len(obs.epochs)
    t_start = time.time()
    last_log_t = t_start
    last_log_rows = 0

    with open(args.output_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "gps_week",
                "gps_tow",
                "sat_id",
                "system",
                "prn",
                "obs_code",
                "pseudorange_m",
                "is_los",
                "is_visible",
                "elevation_deg",
                "azimuth_deg",
                "excess_delay_m",
                "rx_x_m",
                "rx_y_m",
                "rx_z_m",
            ],
        )
        writer.writeheader()

        for epoch_idx, epoch in enumerate(obs.epochs):
            if args.epoch_step > 1 and (epoch_idx % args.epoch_step != 0):
                continue
            if args.max_epochs > 0 and processed_epochs >= args.max_epochs:
                break

            gps_week, gps_tow = _datetime_to_gps_week_tow(epoch.time)
            rx_xyz = _nearest_reference_xyz(track, gps_week, gps_tow)
            if rx_xyz is None:
                skipped_no_rx += 1
                continue

            selected_sat_ids: list[str] = []
            selected_pr: list[float] = []
            for sat_id, sat_obs in epoch.observations.items():
                sat = _parse_sat_id(sat_id)
                if sat is None:
                    continue
                system, _ = sat
                if system not in systems:
                    continue
                pr = float(sat_obs.get(args.obs_code, 0.0))
                if not np.isfinite(pr) or pr <= 0.0:
                    continue
                selected_sat_ids.append(sat_id.strip().upper())
                selected_pr.append(pr)

            if not selected_sat_ids:
                skipped_no_obs += 1
                continue

            sat_ecef, sat_clk, used_ids = eph.compute(
                gps_tow,
                prn_list=selected_sat_ids,
                obs_codes=[args.obs_code] * len(selected_sat_ids),
            )
            if len(used_ids) == 0:
                continue

            prn_ints: list[int] = []
            for sat_id in used_ids:
                parsed = _parse_sat_id(str(sat_id))
                prn_ints.append(parsed[1] if parsed is not None else 0)

            result = usim.compute_epoch(
                rx_ecef=rx_xyz,
                sat_ecef=sat_ecef,
                sat_clk=sat_clk,
                prn_list=prn_ints,
            )

            pr_by_sat = {sid: pr for sid, pr in zip(selected_sat_ids, selected_pr)}
            for i, sat_id in enumerate(used_ids):
                sat_str = str(sat_id).strip().upper()
                parsed = _parse_sat_id(sat_str)
                if parsed is None:
                    continue
                system, prn = parsed
                elev_deg = math.degrees(float(result["elevations"][i]))
                az_deg = math.degrees(float(result["azimuths"][i]))
                if az_deg < 0.0:
                    az_deg += 360.0
                writer.writerow(
                    {
                        "gps_week": gps_week,
                        "gps_tow": f"{gps_tow:.3f}",
                        "sat_id": sat_str,
                        "system": system,
                        "prn": prn,
                        "obs_code": args.obs_code,
                        "pseudorange_m": f"{pr_by_sat.get(sat_str, float('nan')):.3f}",
                        "is_los": int(bool(result["is_los"][i])),
                        "is_visible": int(bool(result["visible"][i])),
                        "elevation_deg": f"{elev_deg:.3f}",
                        "azimuth_deg": f"{az_deg:.3f}",
                        "excess_delay_m": f"{float(result['excess_delays'][i]):.3f}",
                        "rx_x_m": f"{float(rx_xyz[0]):.3f}",
                        "rx_y_m": f"{float(rx_xyz[1]):.3f}",
                        "rx_z_m": f"{float(rx_xyz[2]):.3f}",
                    }
                )
                written += 1

            processed_epochs += 1
            now = time.time()
            if processed_epochs % 25 == 0 or (now - last_log_t) >= 15.0:
                elapsed = max(now - t_start, 1e-6)
                eps = processed_epochs / elapsed
                rows_per_s = (written - last_log_rows) / max(now - last_log_t, 1e-6)
                if args.max_epochs > 0:
                    total_target = min(args.max_epochs, total_epochs)
                else:
                    total_target = total_epochs
                pct = 100.0 * processed_epochs / max(total_target, 1)
                remaining = max(total_target - processed_epochs, 0)
                eta_s = remaining / max(eps, 1e-9)
                eta_m = eta_s / 60.0
                print(
                    f"  epoch {processed_epochs}/{total_target} ({pct:.1f}%)"
                    f" | rows={written}"
                    f" | speed={eps:.2f} ep/s"
                    f" | rows/s={rows_per_s:.1f}"
                    f" | ETA={eta_m:.1f} min"
                )
                last_log_t = now
                last_log_rows = written

    print("[4/5] Done.")
    print(f"  processed epochs: {processed_epochs}")
    print(f"  skipped (no matching reference week): {skipped_no_rx}")
    print(f"  skipped (no usable observations): {skipped_no_obs}")
    print(f"  written rows: {written}")
    print(f"  output: {args.output_csv}")
    print("[5/5] You can now compare this CSV against your ML predictions.")


if __name__ == "__main__":
    main()
