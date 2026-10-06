#!/usr/bin/env python3
"""Monte Carlo Kalman filter precision map on the stadium pitch grid.

Reads ``luigi_pitch_grid_daily.csv`` (mean PDOP per cell) and, for each grid point,
runs many independent 60 s simulations of a constant-velocity Kalman filter with
position measurements whose noise scales with local PDOP.

Standalone (numpy + stdlib csv only for the core; optional matplotlib for PNG).

Example::

    python experiments/stadium_pitch_grid_kalman_monte_carlo.py \\
        --input-csv "C:/Users/Andrea/Desktop/Stadio ray Tracing/luigi_pitch_grid_daily.csv" \\
        --n-mc 200 \\
        --out-csv "C:/Users/Andrea/Desktop/Stadio ray Tracing/luigi_pitch_grid_kf_mc.csv" \\
        --out-png "C:/Users/Andrea/Desktop/Stadio ray Tracing/luigi_pitch_grid_kf_mc.png"
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

# --- Kalman model (ENU position + velocity, 1 Hz) ---------------------------------

DT = 1.0
F = np.array(
    [[1.0, 0.0, DT, 0.0],
     [0.0, 1.0, 0.0, DT],
     [0.0, 0.0, 1.0, 0.0],
     [0.0, 0.0, 0.0, 1.0]],
    dtype=np.float64,
)
P0 = np.diag([1.0, 1.0, 0.1, 0.1]).astype(np.float64)
Q = np.diag([0.01, 0.01, 0.02, 0.02]).astype(np.float64)
H_POS = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float64)
I4 = np.eye(4, dtype=np.float64)


def get_geometry_matrix_for_position() -> np.ndarray:
    """Observation matrix: measure horizontal position (x, y) from state [x,y,vx,vy]."""
    return H_POS.copy()


def measurement_noise_from_pdop(
    pdop: float,
    *,
    sigma_fix_m: float,
    sigma_float_m: float,
    is_fix_zone: bool,
) -> np.ndarray:
    """Map PDOP + ambiguity state to 2×2 measurement covariance R [m²]."""
    sigma0 = float(sigma_fix_m) if is_fix_zone else float(sigma_float_m)
    # Phase/code noise scaled by geometry: σ_xy ≈ PDOP · σ₀
    s = max(float(pdop), 0.5) * sigma0
    var = s * s
    return np.diag([var, var]).astype(np.float64)


def kalman_predict(P: np.ndarray) -> np.ndarray:
    return F @ P @ F.T + Q


def kalman_update(
    x: np.ndarray, P: np.ndarray, z: np.ndarray, R: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    H = get_geometry_matrix_for_position()
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.solve(S, np.eye(2, dtype=np.float64))
    innov = z - H @ x
    x_new = x + K @ innov
    IKH = I4 - K @ H
    P_new = IKH @ P @ IKH.T + K @ R @ K.T
    return x_new, P_new


def horizontal_precision_1sigma(P: np.ndarray) -> float:
    sx = math.sqrt(max(P[0, 0], 0.0))
    sy = math.sqrt(max(P[1, 1], 0.0))
    return math.sqrt(sx * sx + sy * sy)


def simulate_kf_trajectory_seconds(
    n_seconds: int,
    R: np.ndarray,
    *,
    rng: np.random.Generator,
    true_xy: tuple[float, float] = (0.0, 0.0),
) -> tuple[float, float]:
    """One MC run: static truth at ``true_xy``, noisy position fixes every second.

    Returns (covariance-based 1σ [m], horizontal estimation error [m]).
    """
    x = np.zeros(4, dtype=np.float64)
    P = P0.copy()
    tx, ty = true_xy
    for _ in range(int(n_seconds)):
        x = F @ x
        P = kalman_predict(P)
        noise = rng.standard_normal(2)
        z = np.array(
            [tx + math.sqrt(R[0, 0]) * noise[0], ty + math.sqrt(R[1, 1]) * noise[1]],
            dtype=np.float64,
        )
        x, P = kalman_update(x, P, z, R)
    cov_prec = horizontal_precision_1sigma(P)
    est_err = math.sqrt((x[0] - tx) ** 2 + (x[1] - ty) ** 2)
    return cov_prec, est_err


def monte_carlo_cell_precision(
    pdop: float,
    *,
    n_mc: int,
    n_seconds: int,
    sigma_fix_m: float,
    sigma_float_m: float,
    is_fix_zone: bool,
    seed: int,
) -> tuple[float, float, float, float]:
    """Return (mean/std cov 1σ [m], mean/std estimation error [m]) after ``n_seconds``."""
    R = measurement_noise_from_pdop(
        pdop,
        sigma_fix_m=sigma_fix_m,
        sigma_float_m=sigma_float_m,
        is_fix_zone=is_fix_zone,
    )
    rng = np.random.default_rng(seed)
    cov_s = np.empty(int(n_mc), dtype=np.float64)
    err_s = np.empty(int(n_mc), dtype=np.float64)
    for k in range(int(n_mc)):
        cov_s[k], err_s[k] = simulate_kf_trajectory_seconds(
            n_seconds, R, rng=rng, true_xy=(0.0, 0.0)
        )
    return (
        float(np.mean(cov_s)),
        float(np.std(cov_s)),
        float(np.mean(err_s)),
        float(np.std(err_s)),
    )


def _fval(raw: str) -> float | None:
    s = (raw or "").strip()
    if not s:
        return None
    return float(s)


def load_pitch_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    return rows


def is_fix_zone(row: int, col: int, *, fix_row: int, fix_cols: set[int]) -> bool:
    return int(row) == int(fix_row) and int(col) in fix_cols


def run_grid_monte_carlo(
    rows: list[dict],
    *,
    n_mc: int,
    n_seconds: int,
    sigma_fix_m: float,
    sigma_float_m: float,
    fix_row: int,
    fix_cols: tuple[int, ...],
    pdop_fallback: float,
    base_seed: int,
) -> list[dict]:
    fix_set = set(int(c) for c in fix_cols)
    out: list[dict] = []
    for idx, r in enumerate(rows):
        ri = int(r["row"])
        ci = int(r["col"])
        pdop = _fval(r.get("mean_pdop_los", ""))
        if pdop is None or not math.isfinite(pdop):
            pdop_use = float(pdop_fallback)
            pdop_missing = True
        else:
            pdop_use = float(pdop)
            pdop_missing = False

        fix = is_fix_zone(ri, ci, fix_row=fix_row, fix_cols=fix_set)
        seed = int(base_seed) + idx * 10_000 + ri * 100 + ci
        cov_mean, cov_std, err_mean, err_std = monte_carlo_cell_precision(
            pdop_use,
            n_mc=n_mc,
            n_seconds=n_seconds,
            sigma_fix_m=sigma_fix_m,
            sigma_float_m=sigma_float_m,
            is_fix_zone=fix,
            seed=seed,
        )
        out.append(
            {
                "row": ri,
                "col": ci,
                "lat_deg": r.get("lat_deg", ""),
                "lon_deg": r.get("lon_deg", ""),
                "mean_pdop_los": "" if pdop_missing else f"{pdop_use:.4f}",
                "kf_cov_precision_mean_m": cov_mean,
                "kf_cov_precision_std_m": cov_std,
                "kf_est_error_mean_m": err_mean,
                "kf_est_error_std_m": err_std,
                "kf_map_precision_m": err_mean,
                "kf_map_precision_cm": err_mean * 100.0,
                "is_fix_zone": int(fix),
                "pdop_imputed": int(pdop_missing),
            }
        )
        print(
            f"  cell row={ri} col={ci} pdop={pdop_use:.2f} "
            f"{'FIX' if fix else 'FLOAT'} -> cov={cov_mean*100:.1f}cm "
            f"MC err={err_mean*100:.1f}+/-{err_std*100:.1f}cm",
            flush=True,
        )
    return out


def write_results_csv(path: Path, results: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "row",
        "col",
        "lat_deg",
        "lon_deg",
        "mean_pdop_los",
        "kf_cov_precision_mean_m",
        "kf_cov_precision_std_m",
        "kf_est_error_mean_m",
        "kf_est_error_std_m",
        "kf_map_precision_m",
        "kf_map_precision_cm",
        "is_fix_zone",
        "pdop_imputed",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow(
                {
                    "row": r["row"],
                    "col": r["col"],
                    "lat_deg": r["lat_deg"],
                    "lon_deg": r["lon_deg"],
                    "mean_pdop_los": r["mean_pdop_los"],
                    "kf_cov_precision_mean_m": f"{r['kf_cov_precision_mean_m']:.6f}",
                    "kf_cov_precision_std_m": f"{r['kf_cov_precision_std_m']:.6f}",
                    "kf_est_error_mean_m": f"{r['kf_est_error_mean_m']:.6f}",
                    "kf_est_error_std_m": f"{r['kf_est_error_std_m']:.6f}",
                    "kf_map_precision_m": f"{r['kf_map_precision_m']:.6f}",
                    "kf_map_precision_cm": f"{r['kf_map_precision_cm']:.2f}",
                    "is_fix_zone": r["is_fix_zone"],
                    "pdop_imputed": r["pdop_imputed"],
                }
            )


def results_to_display_grid(results: list[dict]) -> tuple[np.ndarray, list[list[str]]]:
    """Build (n_rows, n_cols) arrays for pitch-field plot; row 0 = North."""
    row_ids = sorted({int(r["row"]) for r in results})
    col_ids = sorted({int(r["col"]) for r in results})
    n_rows, n_cols = len(row_ids), len(col_ids)
    rmap = {rid: i for i, rid in enumerate(row_ids)}
    cmap = {cid: j for j, cid in enumerate(col_ids)}

    grid_csv = np.full((n_rows, n_cols), np.nan, dtype=np.float64)
    grid_disp = np.full((n_rows, n_cols), np.nan, dtype=np.float64)
    labels = [["—" for _ in range(n_cols)] for _ in range(n_rows)]

    for r in results:
        ri = rmap[int(r["row"])]
        ci = cmap[int(r["col"])]
        val_m = float(r["kf_map_precision_m"])
        grid_csv[ri, ci] = val_m
        grid_disp[n_rows - 1 - ri, ci] = val_m  # flip for origin=upper
        if val_m >= 1.0:
            labels[n_rows - 1 - ri][ci] = f"{val_m:.1f} m"
        else:
            labels[n_rows - 1 - ri][ci] = f"{val_m*100:.0f} cm"

    return grid_disp, labels


def plot_kf_map(
    grid: np.ndarray,
    labels: list[list[str]],
    *,
    out_png: Path,
    title: str,
    vmin_m: float,
    vmax_m: float,
) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        from matplotlib.patches import Circle
    except ImportError as exc:
        raise SystemExit("matplotlib required for --out-png: pip install matplotlib") from exc

    n_rows, n_cols = grid.shape
    extent = [-40.0, 40.0, -60.0, 60.0]
    y_centers = np.linspace(40.0, -40.0, n_rows)
    x_centers = np.linspace(-32.0, 32.0, n_cols)
    fig_bg = "#1a1d24"

    fig, ax = plt.subplots(figsize=(7, 8), facecolor=fig_bg)
    ax.set_facecolor(fig_bg)
    cmap = plt.colormaps["RdYlGn_r"].copy()
    cmap.set_bad(color="#1f2421")

    im = ax.imshow(
        grid,
        cmap=cmap,
        norm=Normalize(vmin=float(vmin_m), vmax=float(vmax_m)),
        extent=extent,
        aspect="auto",
        origin="upper",
        interpolation="nearest",
    )

    for i in range(n_rows):
        for j in range(n_cols):
            ax.text(
                x_centers[j],
                y_centers[i],
                labels[i][j],
                color="white",
                ha="center",
                va="center",
                fontweight="bold",
                fontsize=10,
                zorder=4,
            )

    ax.axhline(0, color="white", linewidth=1, alpha=0.7, zorder=2)
    ax.add_patch(
        Circle((0, 0), 15, edgecolor="white", facecolor="none", linewidth=1, alpha=0.7, zorder=2)
    )
    ax.set_xticks([-20, 0, 20])
    ax.set_yticks([-40, -20, 0, 20, 40])
    ax.tick_params(axis="both", colors="#cfd2d6", labelcolor="#e8eaed", labelsize=11, pad=6)
    ax.set_xlabel("East [m]", color="#cfd2d6", fontsize=12, labelpad=12)
    ax.set_ylabel("North [m]", color="#cfd2d6", fontsize=12, labelpad=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("white")

    fig.text(0.5, 0.97, title, ha="center", va="top", color="white", fontsize=16, fontweight="bold")
    fig.text(
        0.5,
        0.925,
        "Precisione orizzontale KF (Monte Carlo, 1σ)",
        ha="center",
        va="top",
        color="#e0e4e8",
        fontsize=13,
    )

    cbar = fig.colorbar(im, ax=ax, pad=0.04, shrink=0.88)
    cbar.set_label("Errore stimato [m]", color="white", fontsize=12, labelpad=10)
    cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white")
    fig.subplots_adjust(left=0.14, right=0.78, top=0.86, bottom=0.14)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, facecolor=fig_bg)
    plt.close(fig)
    print(f"Saved {out_png}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-csv", type=Path, required=True)
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--out-png", type=Path, default=None)
    ap.add_argument("--n-mc", type=int, default=200, help="Monte Carlo runs per cell")
    ap.add_argument("--duration-s", type=int, default=60, help="Simulation length [s]")
    ap.add_argument("--sigma-fix-m", type=float, default=0.012)
    ap.add_argument("--sigma-float-m", type=float, default=0.25)
    ap.add_argument(
        "--pdop-fallback",
        type=float,
        default=50.0,
        help="PDOP used when cell has no mean_pdop_los in CSV",
    )
    ap.add_argument("--fix-row", type=int, default=1, help="CSV row index for RTK FIX zone")
    ap.add_argument("--fix-cols", type=str, default="2,3", help="CSV col indices for FIX")
    ap.add_argument("--plot-vmin-m", type=float, default=0.05)
    ap.add_argument("--plot-vmax-m", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--title", type=str, default="Stadio Luigi Ferraris")
    args = ap.parse_args()

    inp = args.input_csv.resolve()
    rows = load_pitch_csv(inp)
    fix_cols = tuple(int(x.strip()) for x in str(args.fix_cols).split(",") if x.strip())

    print(
        f"[kf-mc] cells={len(rows)} n_mc={int(args.n_mc)} duration={int(args.duration_s)}s",
        flush=True,
    )
    results = run_grid_monte_carlo(
        rows,
        n_mc=int(args.n_mc),
        n_seconds=int(args.duration_s),
        sigma_fix_m=float(args.sigma_fix_m),
        sigma_float_m=float(args.sigma_float_m),
        fix_row=int(args.fix_row),
        fix_cols=fix_cols,
        pdop_fallback=float(args.pdop_fallback),
        base_seed=int(args.seed),
    )

    out_csv = args.out_csv.resolve()
    write_results_csv(out_csv, results)
    print(f"[kf-mc] wrote {out_csv}", flush=True)

    if args.out_png:
        grid, labels = results_to_display_grid(results)
        plot_kf_map(
            grid,
            labels,
            out_png=args.out_png.resolve(),
            title=str(args.title),
            vmin_m=float(args.plot_vmin_m),
            vmax_m=float(args.plot_vmax_m),
        )


if __name__ == "__main__":
    main()
