#!/usr/bin/env python3
"""Polar skyplot: satellite passes colored by C/N0.

Each arc is one PRN flyover. Default = **absolute C/N0** (dB-Hz), easiest to read.
Use --min-el-deg 20 to drop the horizon fade.

Example::

    python experiments/plot_terf_cnr_skyplot.py \\
        --obs-path experiments/data/TERF/TERF00CYP_R_20261920000_01D_30S_MO.rnx \\
        --nav-path experiments/data/TERF/BRDC00IGS_R_20261920000_01D_MN.rnx \\
        --reference-csv experiments/data/TERF/reference_2026192.csv \\
        --output experiments/results/terf_cnr_audit/terf_cnr_tracks_2026192.png
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_ROOT / "python"))

from gnss_gpu.ephemeris import Ephemeris
from gnss_gpu.io.nav_rinex import read_nav_rinex_multi
from gnss_gpu.io.rinex import read_rinex_obs
from gnss_gpu.urban_signal_sim import _sat_elevation_azimuth

GPS_EPOCH = datetime(1980, 1, 6)
_CNR_PREF = ("S1C", "S1X", "S1P", "S1W", "S2W", "S2C", "S2X", "S5Q", "S5X", "S7Q", "S7X", "S8Q", "S8X")


def _gps_tow(dt: datetime) -> float:
    return (dt - GPS_EPOCH).total_seconds() % 604800.0


def _normalize_prn(sat_id: str) -> str:
    s = str(sat_id).strip().upper()
    if len(s) >= 3 and s[0] in "GREJCI" and s[1:].isdigit():
        return f"{s[0]}{int(s[1:]):02d}"
    return s


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
        if np.isfinite(v) and v > 0.0:
            return v
    for uk, orig in sorted(upper.items()):
        if not uk.startswith("S"):
            continue
        try:
            v = float(obs_dict[orig])
        except (TypeError, ValueError):
            continue
        if np.isfinite(v) and v > 0.0:
            return v
    return None


def _load_rx(reference_csv: Path) -> np.ndarray:
    with open(reference_csv, newline="", encoding="utf-8") as fh:
        row = next(csv.DictReader(fh))
    return np.array(
        [float(row["ECEF X (m)"]), float(row["ECEF Y (m)"]), float(row["ECEF Z (m)"])],
        dtype=np.float64,
    )


def _collect_tracks(
    obs_path: Path,
    eph: Ephemeris,
    rx: np.ndarray,
    *,
    min_el_deg: float,
    max_el_deg: float,
    epoch_step: int,
) -> dict[str, dict[str, np.ndarray]]:
    obs = read_rinex_obs(obs_path)
    raw: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)

    for ei, ep in enumerate(obs.epochs):
        if epoch_step > 1 and (ei % epoch_step) != 0:
            continue
        tow = float(_gps_tow(ep.time))
        for sat_id, odict in ep.observations.items():
            cnr = _preferred_cnr(odict)
            if cnr is None:
                continue
            prn = _normalize_prn(str(sat_id))
            try:
                sat_ecef, _clk, used = eph.compute(tow, [prn])
            except Exception:
                continue
            if not used:
                continue
            el_rad, az_rad = _sat_elevation_azimuth(rx, sat_ecef[0:1])
            el = float(np.degrees(el_rad[0]))
            if el < float(min_el_deg) or el > float(max_el_deg):
                continue
            az = float(np.degrees(az_rad[0])) % 360.0
            raw[prn].append((tow, az, el, float(cnr)))

    tracks: dict[str, dict[str, np.ndarray]] = {}
    for prn, rows in raw.items():
        rows.sort(key=lambda x: x[0])
        tracks[prn] = {
            "tow": np.array([r[0] for r in rows], dtype=np.float64),
            "az": np.array([r[1] for r in rows], dtype=np.float64),
            "el": np.array([r[2] for r in rows], dtype=np.float64),
            "cnr": np.array([r[3] for r in rows], dtype=np.float64),
        }
    return tracks


def _split_passes(
    tow: np.ndarray,
    az: np.ndarray,
    *,
    max_tow_gap_s: float,
    max_az_jump_deg: float,
) -> list[slice]:
    n = tow.size
    if n == 0:
        return []
    breaks = [0]
    for i in range(1, n):
        dt = float(tow[i] - tow[i - 1])
        if dt < -302400.0:
            dt += 604800.0
        elif dt > 302400.0:
            dt -= 604800.0
        daz = abs(float(az[i] - az[i - 1]))
        if daz > 180.0:
            daz = 360.0 - daz
        if dt > float(max_tow_gap_s) or daz > float(max_az_jump_deg):
            breaks.append(i)
    breaks.append(n)
    return [slice(breaks[k], breaks[k + 1]) for k in range(len(breaks) - 1) if breaks[k + 1] > breaks[k]]


def _color_values(cnr: np.ndarray, mode: str) -> np.ndarray:
    if mode in ("absolute", "normalized"):
        return np.asarray(cnr, dtype=np.float64)
    if mode == "per_pass":
        return np.asarray(cnr, dtype=np.float64) - float(np.median(cnr))
    if mode == "per_prn":
        return np.asarray(cnr, dtype=np.float64) - float(np.median(cnr))
    raise ValueError(mode)


def _plot_tracks(
    tracks: dict[str, dict[str, np.ndarray]],
    out_path: Path,
    *,
    title: str,
    cmap: str,
    vmin: float | None,
    vmax: float | None,
    linewidth: float,
    max_tow_gap_s: float,
    mark_points: bool,
    cnr_mode: str,
    cbar_label: str,
) -> None:
    all_cnr: list[float] = []
    for data in tracks.values():
        all_cnr.extend(data["cnr"].tolist())
    if not all_cnr:
        raise SystemExit("No points in elevation range.")

    if cnr_mode == "absolute":
        lo = 35.0 if vmin is None else float(vmin)
        hi = 50.0 if vmax is None else float(vmax)
    elif cnr_mode == "normalized":
        p = 8.0 if vmin is None else max(abs(float(vmin)), abs(float(vmax)))
        lo, hi = -p, p
        if vmin is not None:
            lo = float(vmin)
        if vmax is not None:
            hi = float(vmax)
    else:
        p = 8.0 if vmin is None else max(abs(float(vmin)), abs(float(vmax)))
        lo, hi = -p, p
        if vmin is not None:
            lo = float(vmin)
        if vmax is not None:
            hi = float(vmax)

    norm = plt.Normalize(vmin=lo, vmax=hi)
    cmap_obj = plt.get_cmap("coolwarm" if cnr_mode == "normalized" else cmap)

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "polar"})
    fig.patch.set_facecolor("#f8f8f8")
    ax.set_facecolor("#ffffff")

    n_seg = 0
    for prn in sorted(tracks):
        data = tracks[prn]
        for sl in _split_passes(
            data["tow"],
            data["az"],
            max_tow_gap_s=max_tow_gap_s,
            max_az_jump_deg=45.0,
        ):
            az = data["az"][sl]
            el = data["el"][sl]
            cnr = data["cnr"][sl]
            if cnr.size < 2:
                continue
            colors = _color_values(cnr, cnr_mode)
            theta = np.radians(az)
            r = 90.0 - el
            for i in range(len(theta) - 1):
                cval = 0.5 * (float(colors[i]) + float(colors[i + 1]))
                ax.plot(
                    theta[i : i + 2],
                    r[i : i + 2],
                    color=cmap_obj(norm(cval)),
                    linewidth=float(linewidth),
                    solid_capstyle="round",
                    zorder=2,
                )
                n_seg += 1
            if mark_points:
                ax.scatter(
                    theta,
                    r,
                    c=colors,
                    cmap=cmap_obj,
                    norm=norm,
                    s=18,
                    linewidths=0.2,
                    edgecolors="#333",
                    alpha=0.9,
                    zorder=4,
                )

    print(f"  drawn segments: {n_seg}")

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 90)
    ax.set_yticks([0, 15, 30, 45, 60, 75, 90])
    ax.set_yticklabels(["90", "75", "60", "45", "30", "15", "0"])
    ax.set_title(title, va="bottom", fontsize=11, pad=16)
    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, pad=0.1, label=cbar_label)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="C/N0 skyplot — one arc per satellite pass.")
    p.add_argument("--obs-path", type=Path, required=True)
    p.add_argument("--nav-path", type=Path, required=True)
    p.add_argument("--reference-csv", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--min-el-deg", type=float, default=20.0, help="Skip below this elevation (default 20)")
    p.add_argument("--max-el-deg", type=float, default=90.0)
    p.add_argument("--epoch-step", type=int, default=1)
    p.add_argument("--obs-interval-s", type=float, default=30.0)
    p.add_argument("--max-tow-gap-s", type=float, default=0.0)
    p.add_argument(
        "--cnr-mode",
        choices=("absolute", "per_pass"),
        default="absolute",
        help="absolute = raw C/N0; per_pass = minus median of each flyover arc",
    )
    p.add_argument("--cmap", type=str, default="turbo")
    p.add_argument("--linewidth", type=float, default=2.5)
    p.add_argument("--mark-points", action="store_true")
    p.add_argument("--title", type=str, default="")
    p.add_argument("--vmin", type=float, default=None)
    p.add_argument("--vmax", type=float, default=None)
    args = p.parse_args()

    print(f"Loading OBS {args.obs_path} ...")
    rx = _load_rx(args.reference_csv)
    eph = Ephemeris(read_nav_rinex_multi(args.nav_path))
    tracks = _collect_tracks(
        args.obs_path,
        eph,
        rx,
        min_el_deg=float(args.min_el_deg),
        max_el_deg=float(args.max_el_deg),
        epoch_step=max(1, int(args.epoch_step)),
    )
    n_pts = sum(int(d["tow"].size) for d in tracks.values())
    print(f"  tracks: {len(tracks)} PRNs, {n_pts} points (el {args.min_el_deg}-{args.max_el_deg} deg)")

    step = max(1, int(args.epoch_step))
    max_gap = float(args.max_tow_gap_s)
    if max_gap <= 0.0:
        max_gap = 1.5 * step * float(args.obs_interval_s)

    mode = str(args.cnr_mode)
    if mode == "absolute":
        cbar = "C/N0 [dB-Hz]"
        subtitle = f"absolute C/N0, elevation {args.min_el_deg:g}-{args.max_el_deg:g} deg"
    else:
        cbar = "C/N0 - pass median [dB-Hz]"
        subtitle = "dip along each flyover vs that pass median"

    title = args.title.strip() or f"{subtitle}\n{args.obs_path.stem}"
    _plot_tracks(
        tracks,
        args.output,
        title=title,
        cmap=str(args.cmap),
        vmin=args.vmin,
        vmax=args.vmax,
        linewidth=float(args.linewidth),
        max_tow_gap_s=max_gap,
        mark_points=bool(args.mark_points),
        cnr_mode=mode,
        cbar_label=cbar,
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
