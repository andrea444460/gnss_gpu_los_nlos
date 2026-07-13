#!/usr/bin/env python3
"""Build a fixed-position reference.csv for a permanent GNSS station.

Reads epoch times from a RINEX OBS file and repeats the same ECEF coordinates
for every epoch (from RINEX header or CLI override).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_ROOT / "python"))

from gnss_gpu.io.rinex import read_rinex_obs

GPS_EPOCH_SECONDS = 315964800.0  # 1980-01-06 minus 1970-01-01
GPS_WEEK_SECONDS = 604800.0
LEAP_SECONDS = 18.0


def _datetime_to_gps_week_tow(dt) -> tuple[int, float]:
    unix_s = dt.timestamp()
    gps_s = unix_s - GPS_EPOCH_SECONDS + LEAP_SECONDS
    week = int(gps_s // GPS_WEEK_SECONDS)
    tow = float(gps_s - week * GPS_WEEK_SECONDS)
    return week, tow


def main() -> None:
    p = argparse.ArgumentParser(description="Fixed ECEF reference.csv from RINEX OBS epochs.")
    p.add_argument("--obs-path", type=Path, required=True)
    p.add_argument("--output-csv", type=Path, required=True)
    p.add_argument("--ecef-x", type=float, default=None)
    p.add_argument("--ecef-y", type=float, default=None)
    p.add_argument("--ecef-z", type=float, default=None)
    args = p.parse_args()

    obs = read_rinex_obs(args.obs_path)
    if not obs.epochs:
        raise SystemExit(f"No epochs in {args.obs_path}")

    xyz = obs.header.approx_position.astype(float)
    if args.ecef_x is not None:
        xyz[0] = args.ecef_x
    if args.ecef_y is not None:
        xyz[1] = args.ecef_y
    if args.ecef_z is not None:
        xyz[2] = args.ecef_z

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["GPS Week", "GPS TOW (s)", "ECEF X (m)", "ECEF Y (m)", "ECEF Z (m)"])
        for ep in obs.epochs:
            week, tow = _datetime_to_gps_week_tow(ep.time)
            w.writerow([week, f"{tow:.6f}", f"{xyz[0]:.4f}", f"{xyz[1]:.4f}", f"{xyz[2]:.4f}"])

    print(f"Wrote {len(obs.epochs)} fixed-position rows -> {args.output_csv}")
    print(f"  ECEF = [{xyz[0]:.4f}, {xyz[1]:.4f}, {xyz[2]:.4f}]")


if __name__ == "__main__":
    main()
