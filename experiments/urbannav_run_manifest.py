#!/usr/bin/env python3
"""Derive area-map parameters from an UrbanNav run (bbox, RINEX time window).

Writes a JSON manifest used by the Colab Odaiba validation pipeline.

Example::

    PYTHONPATH=python python experiments/urbannav_run_manifest.py \\
      --run-dir experiments/data/urbannav/Odaiba \\
      --buffer-m 400 \\
      --out-json experiments/results/odaiba_run_manifest.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_ROOT / "python"))

from gnss_gpu.io.rinex import read_rinex_obs
from gnss_gpu.io.urbannav import UrbanNavLoader
from gnss_gpu.urban_signal_sim import ecef_to_lla


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--rover-source", type=str, default="trimble")
    p.add_argument("--buffer-m", type=float, default=400.0, help="BBox padding around trajectory [m].")
    p.add_argument("--dt-s", type=float, default=300.0, help="Suggested ephemeris sampling step for area map.")
    p.add_argument("--out-json", type=Path, required=True)
    return p.parse_args()


def _bbox_from_latlon(lat_deg: np.ndarray, lon_deg: np.ndarray, buffer_m: float) -> dict[str, float]:
    lat = np.asarray(lat_deg, dtype=np.float64)
    lon = np.asarray(lon_deg, dtype=np.float64)
    lat0 = float(np.mean(lat))
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = max(111_320.0 * math.cos(math.radians(lat0)), 1e-6)
    dlat = float(buffer_m) / m_per_deg_lat
    dlon = float(buffer_m) / m_per_deg_lon
    return {
        "south": float(np.min(lat) - dlat),
        "west": float(np.min(lon) - dlon),
        "north": float(np.max(lat) + dlat),
        "east": float(np.max(lon) + dlon),
    }


def _utc_iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main() -> None:
    args = _parse_args()
    run_dir = args.run_dir.resolve()
    loader = UrbanNavLoader(run_dir)

    gt_times, gt_ecef = loader.load_ground_truth()
    if gt_ecef.size == 0:
        raise ValueError(f"empty ground truth in {run_dir}")

    lats, lons, _alts = [], [], []
    for xyz in gt_ecef:
        lat, lon, alt = ecef_to_lla(float(xyz[0]), float(xyz[1]), float(xyz[2]))
        lats.append(math.degrees(lat))
        lons.append(math.degrees(lon))
        _alts.append(float(alt))

    bbox = _bbox_from_latlon(np.array(lats), np.array(lons), float(args.buffer_m))

    rover_obs = loader.load_rover_obs(rover_source=str(args.rover_source))
    if rover_obs is None or not rover_obs.epochs:
        raise FileNotFoundError(f"rover obs not found in {run_dir}")

    t_first = rover_obs.epochs[0].time
    t_last = rover_obs.epochs[-1].time
    duration_s = max(1.0, (t_last - t_first).total_seconds())
    interval = float(rover_obs.header.interval or 1.0)

    nav_path = loader._find_base_nav()
    rover_path = loader._find_rover_obs(rover_source=str(args.rover_source))

    manifest = {
        "run_dir": str(run_dir),
        "rover_source": str(args.rover_source),
        "reference_csv": str((run_dir / "reference.csv").resolve()),
        "nav_path": str(nav_path.resolve()) if nav_path else "",
        "rover_obs_path": str(rover_path.resolve()) if rover_path else "",
        "bbox": bbox,
        "buffer_m": float(args.buffer_m),
        "rinex": {
            "n_epochs": int(len(rover_obs.epochs)),
            "interval_s": interval,
            "start_utc": _utc_iso(t_first),
            "end_utc": _utc_iso(t_last),
            "duration_s": float(duration_s),
        },
        "area_map_suggestions": {
            "start_utc": _utc_iso(t_first),
            "duration_s": float(duration_s),
            "dt_s": float(args.dt_s),
            "elevation_mask_deg": 10.0,
            "nav_systems": "G,E,J,C",
            "step_m": 25.0,
            "terrain_preprocess_mode": "cpu",
        },
        "gt": {
            "n_points": int(gt_ecef.shape[0]),
            "tow_min": float(np.nanmin(gt_times)),
            "tow_max": float(np.nanmax(gt_times)),
        },
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"\nwritten: {args.out_json}")


if __name__ == "__main__":
    main()
