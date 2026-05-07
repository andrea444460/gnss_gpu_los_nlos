#!/usr/bin/env python3
"""Build GNSS skyplot from RINEX OBS+NAV using gnss_gpu ephemeris."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "python"))

from gnss_gpu.ephemeris import Ephemeris
from gnss_gpu.io.nav_rinex import read_nav_rinex_multi
from gnss_gpu.io.rinex import read_rinex_obs

GPS_EPOCH = datetime(1980, 1, 6)
C_LIGHT = 299792458.0

SYS_COLORS = {
    "G": "#1f77b4",  # GPS
    "R": "#d62728",  # GLONASS
    "E": "#2ca02c",  # Galileo
    "C": "#ff7f0e",  # BeiDou
    "J": "#9467bd",  # QZSS
    "I": "#8c564b",  # IRNSS
}


def _datetime_to_gps_tow(dt: datetime) -> float:
    return float((dt - GPS_EPOCH).total_seconds() % 604800.0)


def _ecef_to_lla(x: float, y: float, z: float) -> tuple[float, float, float]:
    a = 6378137.0
    f = 1.0 / 298.257223563
    e2 = 2.0 * f - f * f
    lon = math.atan2(y, x)
    p = math.sqrt(x * x + y * y)
    lat = math.atan2(z, p * (1.0 - e2))
    for _ in range(8):
        sin_lat = math.sin(lat)
        n = a / math.sqrt(1.0 - e2 * sin_lat * sin_lat)
        lat = math.atan2(z + e2 * n * sin_lat, p)
    sin_lat = math.sin(lat)
    n = a / math.sqrt(1.0 - e2 * sin_lat * sin_lat)
    alt = p / max(math.cos(lat), 1e-12) - n
    return lat, lon, alt


def _sat_el_az(rx_ecef: np.ndarray, sat_ecef: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rx = np.asarray(rx_ecef, dtype=np.float64).reshape(3)
    sat = np.asarray(sat_ecef, dtype=np.float64).reshape(-1, 3)
    lat, lon, _ = _ecef_to_lla(float(rx[0]), float(rx[1]), float(rx[2]))
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    sin_lon, cos_lon = math.sin(lon), math.cos(lon)
    r = np.array(
        [
            [-sin_lon, cos_lon, 0.0],
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
            [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],
        ],
        dtype=np.float64,
    )
    enu = (r @ (sat - rx).T).T
    e, n, u = enu[:, 0], enu[:, 1], enu[:, 2]
    el = np.arctan2(u, np.sqrt(e * e + n * n))
    az = np.arctan2(e, n)
    az = np.mod(az, 2.0 * np.pi)
    return el, az


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build skyplot from RINEX OBS+NAV")
    p.add_argument("--obs", type=Path, required=True, help="RINEX observation file (.obs)")
    p.add_argument("--nav", type=Path, required=True, help="RINEX navigation file (.nav/.rnx)")
    p.add_argument("--systems", type=str, default="G,R,E,C,J", help="Comma-separated systems to include")
    p.add_argument("--epoch-step", type=int, default=5, help="Use every Nth OBS epoch")
    p.add_argument("--max-epochs", type=int, default=0, help="Cap processed epochs (0=all)")
    p.add_argument("--elev-mask-deg", type=float, default=0.0, help="Minimum elevation to plot")
    p.add_argument(
        "--style",
        type=str,
        default="prn-tracks",
        choices=("scatter", "prn-tracks"),
        help="Plot style: scatter cloud or per-PRN tracks with labels",
    )
    p.add_argument(
        "--mono-color",
        action="store_true",
        help="Use one color for all systems (orange-like style in your screenshot).",
    )
    p.add_argument("--out-png", type=Path, required=True, help="Output skyplot image path")
    p.add_argument(
        "--out-prn-csv",
        type=Path,
        default=None,
        help="Optional CSV with per-PRN counts/elevation stats",
    )
    return p.parse_args()


_PR_CODE_PRIORITY = (
    "C1C",
    "C1X",
    "C1P",
    "C1W",
    "C2I",
    "C2X",
    "C2P",
    "C2W",
    "C5X",
    "C5Q",
    "C6I",
    "C6X",
    "C7Q",
    "C7X",
    "C8X",
)


def _pick_pseudorange_m(sat_obs: dict[str, float]) -> float | None:
    for code in _PR_CODE_PRIORITY:
        if code not in sat_obs:
            continue
        try:
            pr = float(sat_obs[code])
        except (TypeError, ValueError):
            continue
        if np.isfinite(pr) and pr > 0.0:
            return pr
    for code, val in sat_obs.items():
        if not str(code).startswith("C"):
            continue
        try:
            pr = float(val)
        except (TypeError, ValueError):
            continue
        if np.isfinite(pr) and pr > 0.0:
            return pr
    return None


def _solve_spp(rx0: np.ndarray, sat_ecef: np.ndarray, pr_corr: np.ndarray, n_iter: int = 8) -> np.ndarray | None:
    """Simple iterative least-squares SPP: unknowns are x,y,z and receiver clock bias in meters."""
    x = np.zeros(4, dtype=np.float64)
    x[:3] = np.asarray(rx0, dtype=np.float64).reshape(3)
    for _ in range(n_iter):
        r = sat_ecef - x[:3]
        rho = np.linalg.norm(r, axis=1)
        if np.any(~np.isfinite(rho)) or np.any(rho < 1.0):
            return None
        pred = rho + x[3]
        v = pr_corr - pred
        h = np.empty((sat_ecef.shape[0], 4), dtype=np.float64)
        h[:, :3] = -(r / rho[:, None])
        h[:, 3] = 1.0
        try:
            dx, *_ = np.linalg.lstsq(h, v, rcond=None)
        except np.linalg.LinAlgError:
            return None
        x += dx
        if np.linalg.norm(dx[:3]) < 1e-3:
            break
    return x[:3]


def _estimate_rx_from_obs(obs, eph: Ephemeris, systems: tuple[str, ...], prn_catalog: list[str]) -> np.ndarray | None:
    allowed = set(systems)
    for ep in obs.epochs[: min(len(obs.epochs), 120)]:
        sat_ids: list[str] = []
        pr_vals: list[float] = []
        for sat_id, sat_obs in ep.observations.items():
            sid = str(sat_id).strip().upper()
            if len(sid) < 2 or sid[0] not in allowed:
                continue
            pr = _pick_pseudorange_m(sat_obs)
            if pr is None:
                continue
            sat_ids.append(sid)
            pr_vals.append(pr)
        if len(sat_ids) < 6:
            continue
        tow = _datetime_to_gps_tow(ep.time)
        sat_ecef, sat_clk, used_prns = eph.compute(tow, prn_list=sat_ids)
        if len(used_prns) < 6:
            continue
        sid_to_i = {str(s).strip().upper(): i for i, s in enumerate(used_prns)}
        sel_idx: list[int] = []
        sel_pr: list[float] = []
        for sid, pr in zip(sat_ids, pr_vals):
            i = sid_to_i.get(sid)
            if i is None:
                continue
            sel_idx.append(i)
            sel_pr.append(pr)
        if len(sel_idx) < 6:
            continue
        sat_sel = np.asarray(sat_ecef[sel_idx], dtype=np.float64)
        clk_sel = np.asarray(sat_clk[sel_idx], dtype=np.float64)
        pr_corr = np.asarray(sel_pr, dtype=np.float64) + C_LIGHT * clk_sel
        rx = _solve_spp(np.zeros(3, dtype=np.float64), sat_sel, pr_corr, n_iter=10)
        if rx is not None and np.linalg.norm(rx) > 5.0e6 and np.linalg.norm(rx) < 7.5e6:
            return rx
    return None


def main() -> None:
    args = _parse_args()
    systems = tuple(s.strip().upper() for s in args.systems.split(",") if s.strip())
    obs = read_rinex_obs(args.obs)
    nav = read_nav_rinex_multi(args.nav, systems=systems)
    eph = Ephemeris(nav)
    prn_catalog = [str(p).strip().upper() for p in eph.available_prns]
    if not prn_catalog:
        raise RuntimeError("No satellites available from NAV in selected systems.")

    rx_ecef = np.asarray(obs.header.approx_position, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(rx_ecef)) or np.linalg.norm(rx_ecef) < 1.0:
        rx_est = _estimate_rx_from_obs(obs, eph, systems, prn_catalog)
        if rx_est is None:
            raise RuntimeError(
                "OBS header has no valid APPROX POSITION XYZ and automatic SPP fallback failed."
            )
        rx_ecef = rx_est
        print(
            "OBS header has no valid APPROX POSITION XYZ; "
            f"using estimated RX ECEF: [{rx_ecef[0]:.1f}, {rx_ecef[1]:.1f}, {rx_ecef[2]:.1f}]"
        )

    az_by_sys: dict[str, list[float]] = {k: [] for k in systems}
    r_by_sys: dict[str, list[float]] = {k: [] for k in systems}
    points_by_prn: dict[str, list[tuple[float, float]]] = {}
    stats_by_prn: dict[str, dict[str, float]] = {}
    n_epochs = 0

    for i, ep in enumerate(obs.epochs):
        if args.epoch_step > 1 and i % args.epoch_step != 0:
            continue
        tow = _datetime_to_gps_tow(ep.time)
        sat_ecef, _sat_clk, used_prns = eph.compute(tow, prn_list=prn_catalog)
        if len(used_prns) == 0:
            continue
        el, az = _sat_el_az(rx_ecef, sat_ecef)
        for j, prn in enumerate(used_prns):
            sid = str(prn).strip().upper()
            sys_id = sid[0] if sid else "?"
            if sys_id not in az_by_sys:
                continue
            elev_deg = float(np.degrees(el[j]))
            if elev_deg < float(args.elev_mask_deg):
                continue
            az_by_sys[sys_id].append(float(az[j]))
            r_by_sys[sys_id].append(90.0 - elev_deg)
            points_by_prn.setdefault(sid, []).append((float(az[j]), 90.0 - elev_deg))
            rec = stats_by_prn.get(sid)
            if rec is None:
                rec = {
                    "system": sys_id,
                    "count": 0.0,
                    "elev_sum": 0.0,
                    "elev_min": elev_deg,
                    "elev_max": elev_deg,
                }
                stats_by_prn[sid] = rec
            rec["count"] += 1.0
            rec["elev_sum"] += elev_deg
            rec["elev_min"] = min(rec["elev_min"], elev_deg)
            rec["elev_max"] = max(rec["elev_max"], elev_deg)
        n_epochs += 1
        if args.max_epochs > 0 and n_epochs >= args.max_epochs:
            break

    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlim(0, 90)
    ax.set_rticks([0, 30, 60, 90])
    ax.set_yticklabels(["90°", "60°", "30°", "0°"])
    ax.grid(True, alpha=0.35)

    plotted_any = False
    mono_color = "#f0a000"
    if args.style == "scatter":
        for sys_id in systems:
            az_vals = az_by_sys.get(sys_id, [])
            r_vals = r_by_sys.get(sys_id, [])
            if not az_vals:
                continue
            plotted_any = True
            color = mono_color if args.mono_color else SYS_COLORS.get(sys_id, "#7f7f7f")
            ax.scatter(
                np.asarray(az_vals, dtype=np.float64),
                np.asarray(r_vals, dtype=np.float64),
                s=8,
                alpha=0.55,
                c=color,
                label=f"{sys_id} ({len(az_vals)})",
                linewidths=0.0,
            )
    else:
        # Per-PRN short tracks with endpoint labels (style similar to the provided screenshot).
        for sat_id in sorted(points_by_prn.keys()):
            pts = points_by_prn[sat_id]
            if not pts:
                continue
            plotted_any = True
            sys_id = sat_id[0]
            color = mono_color if args.mono_color else SYS_COLORS.get(sys_id, "#7f7f7f")
            az_vals = np.asarray([p[0] for p in pts], dtype=np.float64)
            r_vals = np.asarray([p[1] for p in pts], dtype=np.float64)
            ax.plot(az_vals, r_vals, "-", color=color, alpha=0.9, linewidth=1.4)
            ax.plot(az_vals[-1], r_vals[-1], "o", color=color, markersize=2.8)
            # Small angular/radial offset for readability.
            label_theta = float(az_vals[-1] + np.deg2rad(1.2))
            label_r = float(max(min(r_vals[-1] + 1.2, 90.0), 0.0))
            ax.text(label_theta, label_r, sat_id, fontsize=7, ha="left", va="center", color="#333333")

    title = "Skyplot from RINEX OBS+NAV"
    if args.elev_mask_deg > -89.0:
        title += f" (elev >= {args.elev_mask_deg:.1f} deg)"
    ax.set_title(title, pad=18)
    if plotted_any and args.style == "scatter" and not args.mono_color:
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)

    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=160)
    plt.close(fig)

    if args.out_prn_csv is not None:
        args.out_prn_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_prn_csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "sat_id",
                    "system",
                    "count",
                    "elev_mean_deg",
                    "elev_min_deg",
                    "elev_max_deg",
                ],
            )
            writer.writeheader()
            for sat_id in sorted(stats_by_prn.keys()):
                rec = stats_by_prn[sat_id]
                c = max(int(rec["count"]), 1)
                writer.writerow(
                    {
                        "sat_id": sat_id,
                        "system": rec["system"],
                        "count": int(rec["count"]),
                        "elev_mean_deg": f"{(rec['elev_sum'] / c):.3f}",
                        "elev_min_deg": f"{rec['elev_min']:.3f}",
                        "elev_max_deg": f"{rec['elev_max']:.3f}",
                    }
                )

    print(f"Saved skyplot: {args.out_png}")
    print(f"Processed epochs: {n_epochs}")
    if args.out_prn_csv is not None:
        print(f"Saved per-PRN stats: {args.out_prn_csv}")


if __name__ == "__main__":
    main()
