#!/usr/bin/env python3
"""Plot results from ``stadium_pitch_kalman_trajectory_mc.py``.

Produces:
  - ``*_traj_map.png`` — all paths on the pitch, colour = MC mean error
  - ``*_traj_bars.png`` — bar chart with uncertainty bars

Example::

    python experiments/plot_stadium_kalman_trajectories.py \\
        --results-csv "C:/Users/Me/Desktop/Stadio ray Tracing/kf_trajectories.csv" \\
        --grid-csv "C:/Users/Me/Desktop/Stadio ray Tracing/luigi_pitch_grid_daily.csv" \\
        --perimeter-csv experiments/data/LuigiFerraris/luigiFerrarisTrajectory.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.patches import Circle
except ImportError as exc:
    raise SystemExit("pip install matplotlib") from exc

from stadium_pitch_kalman_trajectory_mc import (
    grid_line_trajectories,
    load_grid_cells,
    perimeter_trajectory,
)


def load_results_csv(path: Path) -> dict[str, dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out: dict[str, dict] = {}
    for r in rows:
        tid = r["trajectory_id"].strip()
        out[tid] = {
            "mean_m": float(r["mc_mean_error_m"]),
            "std_m": float(r["mc_std_error_m"]),
            "mean_cm": float(r["mc_mean_error_cm"]),
            "max_m": float(r["mc_max_error_mean_m"]),
        }
    return out


def build_trajectory_catalog(
    grid_csv: Path,
    perimeter_csv: Path | None,
) -> list[tuple[str, np.ndarray]]:
    cells, lat0, lon0 = load_grid_cells(grid_csv.resolve())
    trajs = grid_line_trajectories(cells)
    if perimeter_csv is not None:
        trajs.append(perimeter_trajectory(perimeter_csv.resolve(), lat0, lon0))
    return trajs


def _style_pitch_ax(ax, all_en: np.ndarray, *, fig_bg: str = "#1a1d24", pad_m: float = 12.0) -> None:
    ax.set_facecolor(fig_bg)
    if all_en.size >= 2:
        e = all_en[:, 0]
        n = all_en[:, 1]
        ax.set_xlim(float(e.min()) - pad_m, float(e.max()) + pad_m)
        ax.set_ylim(float(n.min()) - pad_m, float(n.max()) + pad_m)
    else:
        ax.set_xlim(-45, 45)
        ax.set_ylim(-65, 65)
    ax.axhline(0, color="white", linewidth=1, alpha=0.5, zorder=1)
    ax.add_patch(
        Circle((0, 0), 15, edgecolor="white", facecolor="none", linewidth=1, alpha=0.5, zorder=1)
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("East [m]", color="#cfd2d6", fontsize=12, labelpad=8)
    ax.set_ylabel("North [m]", color="#cfd2d6", fontsize=12, labelpad=8)
    ax.tick_params(colors="#cfd2d6", labelsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor("white")


def plot_trajectory_map(
    trajectories: list[tuple[str, np.ndarray]],
    results: dict[str, dict],
    *,
    grid_csv: Path,
    out_png: Path,
    title: str,
    vmin_m: float,
    vmax_m: float,
) -> None:
    fig_bg = "#1a1d24"
    fig, ax = plt.subplots(figsize=(8, 9), facecolor=fig_bg)

    all_pts: list[np.ndarray] = []
    for tid, wpts in trajectories:
        if tid in results:
            all_pts.append(np.asarray(wpts, dtype=np.float64))
    cells, _, _ = load_grid_cells(grid_csv.resolve())
    for c in cells:
        all_pts.append(np.array([[c["e"], c["n"]]], dtype=np.float64))
    all_en = np.vstack(all_pts) if all_pts else np.zeros((0, 2))
    _style_pitch_ax(ax, all_en, fig_bg=fig_bg)

    cmap = plt.colormaps["RdYlGn_r"]
    norm = Normalize(vmin=float(vmin_m), vmax=float(vmax_m))

    for tid, wpts in trajectories:
        if tid not in results:
            continue
        err_m = results[tid]["mean_m"]
        col = cmap(norm(err_m))
        wp = np.asarray(wpts, dtype=np.float64)
        ax.plot(
            wp[:, 0],
            wp[:, 1],
            color=col,
            linewidth=3.0,
            solid_capstyle="round",
            zorder=3,
            label=f"{tid} ({err_m*100:.0f} cm)",
        )
        ax.scatter(wp[0, 0], wp[0, 1], c="white", s=28, zorder=4, edgecolors=col, linewidths=1.5)
        ax.scatter(wp[-1, 0], wp[-1, 1], color=col, s=40, marker="s", zorder=4)

    for c in cells:
        ax.plot(c["e"], c["n"], "o", color="#556", markersize=5, zorder=2)

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, shrink=0.88)
    cbar.set_label("Errore medio KF [m]", color="white", fontsize=11)
    cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white")

    fig.text(0.5, 0.97, title, ha="center", va="top", color="white", fontsize=16, fontweight="bold")
    fig.text(
        0.5,
        0.925,
        "Traiettorie sul campo (colore = errore MC medio, 60 s)",
        ha="center",
        va="top",
        color="#e0e4e8",
        fontsize=12,
    )

    fig.subplots_adjust(left=0.12, right=0.82, top=0.86, bottom=0.10)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, facecolor=fig_bg)
    plt.close(fig)
    print(f"Saved {out_png}", flush=True)


def plot_trajectory_bars(
    trajectories: list[tuple[str, np.ndarray]],
    results: dict[str, dict],
    *,
    out_png: Path,
    title: str,
) -> None:
    ids = []
    means_cm = []
    stds_cm = []
    for tid, _ in trajectories:
        if tid not in results:
            continue
        ids.append(tid.replace("grid_", "").replace("_", "\n"))
        means_cm.append(results[tid]["mean_cm"])
        stds_cm.append(results[tid]["std_m"] * 100.0)

    order = np.argsort(means_cm)
    ids = [ids[i] for i in order]
    means_cm = [means_cm[i] for i in order]
    stds_cm = [stds_cm[i] for i in order]

    fig_bg = "#1a1d24"
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=fig_bg)
    ax.set_facecolor("#252a32")
    x = np.arange(len(ids))
    colors = plt.colormaps["RdYlGn_r"](Normalize(vmin=min(means_cm), vmax=max(means_cm))(means_cm))

    ax.bar(x, means_cm, yerr=stds_cm, color=colors, edgecolor="white", linewidth=0.6, capsize=4, error_kw={"ecolor": "#aaa"})
    ax.set_xticks(x)
    ax.set_xticklabels(ids, rotation=45, ha="right", fontsize=9, color="#ddd")
    ax.set_ylabel("Errore medio KF [cm]", color="#cfd2d6", fontsize=12)
    ax.tick_params(colors="#cfd2d6")
    for spine in ax.spines.values():
        spine.set_edgecolor("#555")

    fig.text(0.5, 0.98, title, ha="center", va="top", color="white", fontsize=15, fontweight="bold")
    fig.text(
        0.5,
        0.93,
        "Confronto traiettorie (barre = media MC, whisker = std)",
        ha="center",
        va="top",
        color="#e0e4e8",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, facecolor=fig_bg)
    plt.close(fig)
    print(f"Saved {out_png}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-csv", type=Path, required=True)
    ap.add_argument("--grid-csv", type=Path, required=True)
    ap.add_argument("--perimeter-csv", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--title", type=str, default="Stadio Luigi Ferraris")
    ap.add_argument("--plot-vmin-m", type=float, default=0.3)
    ap.add_argument("--plot-vmax-m", type=float, default=5.0)
    args = ap.parse_args()

    results_path = args.results_csv.resolve()
    out_dir = args.out_dir.resolve() if args.out_dir else results_path.parent
    stem = results_path.stem

    results = load_results_csv(results_path)
    trajectories = build_trajectory_catalog(args.grid_csv.resolve(), args.perimeter_csv)

    plot_trajectory_map(
        trajectories,
        results,
        grid_csv=args.grid_csv.resolve(),
        out_png=out_dir / f"{stem}_map.png",
        title=str(args.title),
        vmin_m=float(args.plot_vmin_m),
        vmax_m=float(args.plot_vmax_m),
    )
    plot_trajectory_bars(
        trajectories,
        results,
        out_png=out_dir / f"{stem}_bars.png",
        title=str(args.title),
    )


if __name__ == "__main__":
    main()
