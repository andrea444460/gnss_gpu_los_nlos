#!/usr/bin/env python3
"""Monte Carlo Kalman filter along **moving trajectories** on the pitch.

Unlike ``stadium_pitch_grid_kalman_monte_carlo.py`` (one static point per grid cell),
this script walks several paths across the field. At each second:

  - the **true** position follows the path (operator moving on the pitch);
  - measurement noise ``R`` comes from the **nearest grid cell's PDOP** (from daily CSV);
  - a constant-velocity Kalman filter predicts/updates state ``[x,y,vx,vy]``.

Example::

    python experiments/stadium_pitch_kalman_trajectory_mc.py \\
        --grid-csv "C:/Users/Me/Desktop/Stadio ray Tracing/luigi_pitch_grid_daily.csv" \\
        --perimeter-csv experiments/data/LuigiFerraris/luigiFerrarisTrajectory.csv \\
        --n-mc 100 --duration-s 60 \\
        --out-csv "C:/Users/Me/Desktop/Stadio ray Tracing/kf_trajectories.csv"
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

# Reuse KF core from the static-cell script (same file, no gnss_gpu package).
from stadium_pitch_grid_kalman_monte_carlo import (
    P0,
    F,
    Q,
    is_fix_zone,
    kalman_predict,
    kalman_update,
    measurement_noise_from_pdop,
    _fval,
)

# ---------------------------------------------------------------------------
# Pitch / grid helpers
# ---------------------------------------------------------------------------


def latlon_to_enu_m(lat_deg: float, lon_deg: float, lat0_deg: float, lon0_deg: float) -> tuple[float, float]:
    """Local ENU [m] from WGS84 degree differences (small-area flat Earth)."""
    lat0_rad = math.radians(lat0_deg)
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * math.cos(lat0_rad)
    e = (float(lon_deg) - float(lon0_deg)) * m_per_deg_lon
    n = (float(lat_deg) - float(lat0_deg)) * m_per_deg_lat
    return float(e), float(n)


def load_grid_cells(grid_csv: Path) -> tuple[list[dict], float, float]:
    with open(grid_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty grid CSV: {grid_csv}")
    lats = [float(r["lat_deg"]) for r in rows]
    lons = [float(r["lon_deg"]) for r in rows]
    lat0 = float(np.mean(lats))
    lon0 = float(np.mean(lons))
    cells: list[dict] = []
    for r in rows:
        e, n = latlon_to_enu_m(float(r["lat_deg"]), float(r["lon_deg"]), lat0, lon0)
        pdop = _fval(r.get("mean_pdop_los", ""))
        cells.append(
            {
                "row": int(r["row"]),
                "col": int(r["col"]),
                "e": e,
                "n": n,
                "pdop": float(pdop) if pdop is not None and math.isfinite(pdop) else None,
            }
        )
    return cells, lat0, lon0


def nearest_cell(cells: list[dict], e: float, n: float) -> dict:
    best = cells[0]
    d2min = float("inf")
    for c in cells:
        d2 = (c["e"] - e) ** 2 + (c["n"] - n) ** 2
        if d2 < d2min:
            d2min = d2
            best = c
    return best


def resample_polyline_xy(waypoints: np.ndarray, n_samples: int) -> np.ndarray:
    """Uniform spacing along polyline; returns (n_samples, 2) [e, n]."""
    wp = np.asarray(waypoints, dtype=np.float64).reshape(-1, 2)
    if wp.shape[0] == 1:
        return np.repeat(wp, int(n_samples), axis=0)
    seg = np.linalg.norm(np.diff(wp, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    if total < 1e-6:
        return np.repeat(wp[:1], int(n_samples), axis=0)
    targets = np.linspace(0.0, total, int(n_samples))
    out = np.empty((int(n_samples), 2), dtype=np.float64)
    j = 0
    for i, s in enumerate(targets):
        while j + 1 < len(cum) - 1 and cum[j + 1] < s:
            j += 1
        seg_len = max(cum[j + 1] - cum[j], 1e-12)
        alpha = (s - cum[j]) / seg_len
        out[i] = (1.0 - alpha) * wp[j] + alpha * wp[j + 1]
    return out


# ---------------------------------------------------------------------------
# Trajectory catalogue
# ---------------------------------------------------------------------------


def grid_line_trajectories(cells: list[dict]) -> list[tuple[str, np.ndarray]]:
    """Rows and columns of the 3x5 grid: straight lines across the pitch."""
    by_rc = {(c["row"], c["col"]): c for c in cells}
    row_ids = sorted({c["row"] for c in cells})
    col_ids = sorted({c["col"] for c in cells})
    trajs: list[tuple[str, np.ndarray]] = []

    for ri in row_ids:
        pts = np.array([[by_rc[(ri, ci)]["e"], by_rc[(ri, ci)]["n"]] for ci in col_ids])
        trajs.append((f"grid_row_{ri}_W_to_E", pts))

    for ci in col_ids:
        pts = np.array([[by_rc[(ri, ci)]["e"], by_rc[(ri, ci)]["n"]] for ri in row_ids])
        trajs.append((f"grid_col_{ci}_S_to_N", pts))

    # Diagonals between corners
    trajs.append(
        (
            "diag_SW_NE",
            np.array(
                [
                    [by_rc[(row_ids[0], col_ids[0])]["e"], by_rc[(row_ids[0], col_ids[0])]["n"]],
                    [by_rc[(row_ids[-1], col_ids[-1])]["e"], by_rc[(row_ids[-1], col_ids[-1])]["n"]],
                ]
            ),
        )
    )
    trajs.append(
        (
            "diag_SE_NW",
            np.array(
                [
                    [by_rc[(row_ids[0], col_ids[-1])]["e"], by_rc[(row_ids[0], col_ids[-1])]["n"]],
                    [by_rc[(row_ids[-1], col_ids[0])]["e"], by_rc[(row_ids[-1], col_ids[0])]["n"]],
                ]
            ),
        )
    )
    return trajs


def perimeter_trajectory(perimeter_csv: Path, lat0: float, lon0: float) -> tuple[str, np.ndarray]:
    with open(perimeter_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    pts = []
    for r in rows:
        lat_k = next((k for k in r if k.strip().lower().startswith("lat")), None)
        lon_k = next((k for k in r if "lon" in k.strip().lower()), None)
        if lat_k is None or lon_k is None:
            continue
        e, n = latlon_to_enu_m(float(r[lat_k]), float(r[lon_k]), lat0, lon0)
        pts.append([e, n])
    if len(pts) < 2:
        raise ValueError(f"perimeter CSV has too few points: {perimeter_csv}")
    # Close the loop
    pts.append(pts[0])
    return "perimeter_loop", np.asarray(pts, dtype=np.float64)


# ---------------------------------------------------------------------------
# KF along a moving path
# ---------------------------------------------------------------------------


def simulate_kf_along_path(
    path_en: np.ndarray,
    cells: list[dict],
    *,
    rng: np.random.Generator,
    sigma_fix_m: float,
    sigma_float_m: float,
    fix_row: int,
    fix_cols: set[int],
    pdop_fallback: float,
) -> tuple[float, float, np.ndarray]:
    """One MC run along ``path_en`` (T,2). Returns mean/max error [m] and per-step errors."""
    x = np.zeros(4, dtype=np.float64)
    P = P0.copy()
    errs = np.empty(path_en.shape[0], dtype=np.float64)

    for t in range(path_en.shape[0]):
        tx, ty = float(path_en[t, 0]), float(path_en[t, 1])
        cell = nearest_cell(cells, tx, ty)
        pdop = cell["pdop"] if cell["pdop"] is not None else float(pdop_fallback)
        fix = is_fix_zone(cell["row"], cell["col"], fix_row=fix_row, fix_cols=fix_cols)
        R = measurement_noise_from_pdop(
            pdop,
            sigma_fix_m=sigma_fix_m,
            sigma_float_m=sigma_float_m,
            is_fix_zone=fix,
        )

        x = F @ x
        P = kalman_predict(P)
        noise = rng.standard_normal(2)
        z = np.array(
            [tx + math.sqrt(R[0, 0]) * noise[0], ty + math.sqrt(R[1, 1]) * noise[1]],
            dtype=np.float64,
        )
        x, P = kalman_update(x, P, z, R)
        errs[t] = math.sqrt((x[0] - tx) ** 2 + (x[1] - ty) ** 2)

    return float(np.mean(errs)), float(np.max(errs)), errs


def run_trajectory_monte_carlo(
    traj_id: str,
    waypoints: np.ndarray,
    cells: list[dict],
    *,
    n_mc: int,
    duration_s: int,
    sigma_fix_m: float,
    sigma_float_m: float,
    fix_row: int,
    fix_cols: set[int],
    pdop_fallback: float,
    seed: int,
) -> dict:
    path = resample_polyline_xy(waypoints, int(duration_s))
    fix_set = set(int(c) for c in fix_cols)
    rng = np.random.default_rng(int(seed))
    mean_errs = []
    max_errs = []
    for _ in range(int(n_mc)):
        m, mx, _ = simulate_kf_along_path(
            path,
            cells,
            rng=rng,
            sigma_fix_m=sigma_fix_m,
            sigma_float_m=sigma_float_m,
            fix_row=fix_row,
            fix_cols=fix_set,
            pdop_fallback=pdop_fallback,
        )
        mean_errs.append(m)
        max_errs.append(mx)
    return {
        "trajectory_id": traj_id,
        "n_waypoints": int(waypoints.shape[0]),
        "duration_s": int(duration_s),
        "path_length_m": float(
            np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)) if path.shape[0] > 1 else 0.0
        ),
        "mc_mean_error_m": float(np.mean(mean_errs)),
        "mc_std_error_m": float(np.std(mean_errs)),
        "mc_max_error_mean_m": float(np.mean(max_errs)),
        "mc_mean_error_cm": float(np.mean(mean_errs)) * 100.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grid-csv", type=Path, required=True, help="luigi_pitch_grid_daily.csv")
    ap.add_argument(
        "--perimeter-csv",
        type=Path,
        default=None,
        help="Optional lat/lon loop (e.g. luigiFerrarisTrajectory.csv)",
    )
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--n-mc", type=int, default=100)
    ap.add_argument("--duration-s", type=int, default=60)
    ap.add_argument("--sigma-fix-m", type=float, default=0.012)
    ap.add_argument("--sigma-float-m", type=float, default=0.25)
    ap.add_argument("--pdop-fallback", type=float, default=50.0)
    ap.add_argument("--fix-row", type=int, default=1)
    ap.add_argument("--fix-cols", type=str, default="2,3")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cells, lat0, lon0 = load_grid_cells(args.grid_csv.resolve())
    fix_cols = tuple(int(x.strip()) for x in str(args.fix_cols).split(",") if x.strip())

    trajectories = grid_line_trajectories(cells)
    if args.perimeter_csv is not None:
        trajectories.append(perimeter_trajectory(args.perimeter_csv.resolve(), lat0, lon0))

    print(f"[traj-mc] {len(trajectories)} trajectories, n_mc={int(args.n_mc)}, T={int(args.duration_s)}s", flush=True)

    results: list[dict] = []
    for ti, (tid, wpts) in enumerate(trajectories):
        row = run_trajectory_monte_carlo(
            tid,
            wpts,
            cells,
            n_mc=int(args.n_mc),
            duration_s=int(args.duration_s),
            sigma_fix_m=float(args.sigma_fix_m),
            sigma_float_m=float(args.sigma_float_m),
            fix_row=int(args.fix_row),
            fix_cols=fix_cols,
            pdop_fallback=float(args.pdop_fallback),
            seed=int(args.seed) + ti * 50_000,
        )
        results.append(row)
        print(
            f"  {tid}: len={row['path_length_m']:.0f}m "
            f"err_mean={row['mc_mean_error_cm']:.1f}+/-{row['mc_std_error_m']*100:.1f}cm "
            f"max_mean={row['mc_max_error_mean_m']:.2f}m",
            flush=True,
        )

    out = args.out_csv.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(results[0].keys())
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"[traj-mc] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
