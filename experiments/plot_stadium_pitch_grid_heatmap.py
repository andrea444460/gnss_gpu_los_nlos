#!/usr/bin/env python3
"""Plot colored heatmaps on the football pitch from ``stadium_pitch_grid_daily_metrics`` CSV.

Example::

    python experiments/plot_stadium_pitch_grid_heatmap.py \\
        --input-csv ~/Desktop/luigi_pitch_grid_daily.csv \\
        --out-png ~/Desktop/luigi_pitch_grid_heatmap.png

Writes a combined 3-panel PNG plus separate files ``*_los.png``, ``*_nlos.png``, ``*_pdop.png``,
a styled PDOP pitch view ``*_pdop_campo.png``, and RTK horizontal precision ``*_precision.png``
(unless ``--combined-only`` / respective ``--no-*`` flags).
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import Circle
    from matplotlib.colors import Normalize
except ImportError as exc:
    raise SystemExit("matplotlib required: pip install matplotlib") from exc


def _fval(raw: str) -> float:
    s = (raw or "").strip()
    if not s:
        return float("nan")
    return float(s)


def load_grid_csv(path: Path) -> dict:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty CSV: {path}")

    row_ids = sorted({int(r["row"]) for r in rows})
    col_ids = sorted({int(r["col"]) for r in rows})
    n_rows = len(row_ids)
    n_cols = len(col_ids)
    row_index = {rid: i for i, rid in enumerate(row_ids)}
    col_index = {cid: i for i, cid in enumerate(col_ids)}

    def _grid(field: str) -> np.ndarray:
        g = np.full((n_rows, n_cols), np.nan, dtype=np.float64)
        for r in rows:
            g[row_index[int(r["row"])], col_index[int(r["col"])]] = _fval(r[field])
        return g

    lats = np.full((n_rows, n_cols), np.nan, dtype=np.float64)
    lons = np.full((n_rows, n_cols), np.nan, dtype=np.float64)
    for r in rows:
        ri = row_index[int(r["row"])]
        ci = col_index[int(r["col"])]
        lats[ri, ci] = float(r["lat_deg"])
        lons[ri, ci] = float(r["lon_deg"])

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "row_ids": row_ids,
        "col_ids": col_ids,
        "lats": lats,
        "lons": lons,
        "mean_n_los": _grid("mean_n_los"),
        "mean_n_nlos": _grid("mean_n_nlos"),
        "mean_pdop_los": _grid("mean_pdop_los"),
        "mean_hdop_los": _grid("mean_hdop_los"),
        "n_epochs_pdop_ok": _grid("n_epochs_pdop_ok"),
    }


def _pitch_extent_m(lats: np.ndarray, lons: np.ndarray) -> tuple[float, float, float, float]:
    """Approximate local ENU span [m] from lat/lon grid corners."""
    lat0 = float(np.nanmean(lats))
    lon0 = float(np.nanmean(lons))
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat0))

    e = (lons - lon0) * m_per_deg_lon
    n = (lats - lat0) * m_per_deg_lat
    pad = 6.0
    return (
        float(np.nanmin(e) - pad),
        float(np.nanmax(e) + pad),
        float(np.nanmin(n) - pad),
        float(np.nanmax(n) + pad),
    )


def _draw_pitch(ax, extent: tuple[float, float, float, float]) -> None:
    xmin, xmax, ymin, ymax = extent
    ax.add_patch(
        mpatches.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            fill=False,
            edgecolor="#ffffff",
            linewidth=2.0,
            linestyle="-",
            zorder=2,
        )
    )
    mid_y = 0.5 * (ymin + ymax)
    ax.plot([xmin, xmax], [mid_y, mid_y], color="#ffffff", linewidth=1.2, alpha=0.65, zorder=2)
    cx = 0.5 * (xmin + xmax)
    circle = mpatches.Circle((cx, mid_y), 0.12 * (ymax - ymin), fill=False, edgecolor="#ffffff", linewidth=1.0, alpha=0.65, zorder=2)
    ax.add_patch(circle)


def _annotate_cells(
    ax,
    grid: np.ndarray,
    extent: tuple[float, float, float, float],
    fmt: str,
    *,
    text_color: str = "white",
) -> None:
    xmin, xmax, ymin, ymax = extent
    n_rows, n_cols = grid.shape
    dx = (xmax - xmin) / n_cols
    dy = (ymax - ymin) / n_rows
    for ri in range(n_rows):
        for ci in range(n_cols):
            val = grid[ri, ci]
            if not math.isfinite(val):
                txt = "—"
            else:
                txt = fmt.format(val)
            cx = xmin + (ci + 0.5) * dx
            cy = ymin + (ri + 0.5) * dy
            ax.text(cx, cy, txt, ha="center", va="center", fontsize=9, color=text_color, fontweight="bold", zorder=4)


def _heatmap_panel(
    ax,
    grid: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    *,
    title: str,
    cmap: str,
    vmin: float | None,
    vmax: float | None,
    cbar_label: str,
    value_fmt: str,
    reverse_cmap: bool = False,
    value_color: str = "white",
) -> None:
    extent = _pitch_extent_m(lats, lons)
    xmin, xmax, ymin, ymax = extent
    n_rows, n_cols = grid.shape
    data = np.asarray(grid, dtype=np.float64)
    cmap_obj = plt.get_cmap(cmap)
    if reverse_cmap:
        cmap_obj = cmap_obj.reversed()

    valid = data[np.isfinite(data)]
    if valid.size == 0:
        ax.set_title(title)
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center", color="white")
        return

    lo = float(np.nanmin(valid)) if vmin is None else vmin
    hi = float(np.nanmax(valid)) if vmax is None else vmax
    if hi <= lo:
        hi = lo + 1.0

    im = ax.imshow(
        data,
        origin="lower",
        extent=[xmin, xmax, ymin, ymax],
        cmap=cmap_obj,
        norm=Normalize(vmin=lo, vmax=hi),
        aspect="equal",
        interpolation="nearest",
        alpha=0.95,
    )
    _draw_pitch(ax, extent)
    _annotate_cells(ax, grid, extent, value_fmt, text_color=value_color)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_title(title, fontsize=12, color="white", pad=8)
    ax.set_xlabel("East [m]", color="#ccc")
    ax.set_ylabel("North [m]", color="#ccc")
    ax.tick_params(colors="#aaa")
    for spine in ax.spines.values():
        spine.set_color("#555")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, color="#ccc")
    cbar.ax.yaxis.set_tick_params(color="#aaa")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#aaa")


PANEL_SPECS = (
    {
        "key": "mean_n_los",
        "suffix": "los",
        "title": "Mean LOS satellites",
        "cmap": "YlGn",
        "vmin": 0.0,
        "vmax": None,
        "cbar_label": "count",
        "value_fmt": "{:.1f}",
        "reverse_cmap": False,
        "value_color": "black",
    },
    {
        "key": "mean_n_nlos",
        "suffix": "nlos",
        "title": "Mean NLOS satellites",
        "cmap": "YlOrRd",
        "vmin": 0.0,
        "vmax": None,
        "cbar_label": "count",
        "value_fmt": "{:.1f}",
        "reverse_cmap": False,
    },
    {
        "key": "mean_pdop_los",
        "suffix": "pdop",
        "title": "Mean PDOP (LOS only)",
        "cmap": "RdYlGn",
        "vmin": 1.0,
        "vmax": 20.0,
        "cbar_label": "PDOP",
        "value_fmt": "{:.1f}",
        "reverse_cmap": True,
    },
)


def plot_single_heatmap(data: dict, *, spec: dict, suptitle: str, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.6), facecolor="#1a1d26")
    ax.set_facecolor("#243024")
    _heatmap_panel(
        ax,
        data[spec["key"]],
        data["lats"],
        data["lons"],
        title=spec["title"],
        cmap=spec["cmap"],
        vmin=spec["vmin"],
        vmax=spec["vmax"],
        cbar_label=spec["cbar_label"],
        value_fmt=spec["value_fmt"],
        reverse_cmap=bool(spec["reverse_cmap"]),
        value_color=str(spec.get("value_color", "white")),
    )
    fig.suptitle(suptitle, fontsize=13, color="white", y=0.98)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved {out_png}", flush=True)


def plot_heatmaps(data: dict, *, title: str, out_png: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), facecolor="#1a1d26")
    for ax in axes:
        ax.set_facecolor("#243024")

    for ax, spec in zip(axes, PANEL_SPECS):
        _heatmap_panel(
            ax,
            data[spec["key"]],
            data["lats"],
            data["lons"],
            title=spec["title"],
            cmap=spec["cmap"],
            vmin=spec["vmin"],
            vmax=spec["vmax"],
            cbar_label=spec["cbar_label"],
            value_fmt=spec["value_fmt"],
            reverse_cmap=bool(spec["reverse_cmap"]),
            value_color=str(spec.get("value_color", "white")),
        )

    fig.suptitle(title, fontsize=14, color="white", y=1.02)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved {out_png}", flush=True)


def _pdop_display_grid(data: dict) -> np.ndarray:
    """PDOP grid with row 0 = North (for ``origin='upper'`` pitch maps)."""
    grid = np.asarray(data["mean_pdop_los"], dtype=np.float64)
    return np.flipud(grid)


def _render_pitch_field(
    values: np.ndarray,
    labels: list[list[str]],
    *,
    suptitle: str,
    subtitle: str,
    out_png: Path,
    vmin: float,
    vmax: float,
    cbar_label: str,
    label_fontsize: float = 11.0,
) -> None:
    """Shared dark pitch grid (same layout as ``*_pdop_campo.png``)."""
    n_rows, n_cols = values.shape
    extent = [-40.0, 40.0, -60.0, 60.0]
    y_centers = np.linspace(40.0, -40.0, n_rows)
    x_centers = np.linspace(-32.0, 32.0, n_cols)

    fig_bg = "#1a1d24"
    fig, ax = plt.subplots(figsize=(7, 8), facecolor=fig_bg)
    ax.set_facecolor(fig_bg)

    cmap = plt.colormaps["RdYlGn_r"].copy()
    cmap.set_bad(color="#1f2421")

    im = ax.imshow(
        values,
        cmap=cmap,
        norm=Normalize(vmin=float(vmin), vmax=float(vmax)),
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
                fontsize=label_fontsize,
                zorder=4,
            )

    ax.axhline(0, color="white", linewidth=1, alpha=0.7, zorder=2)
    ax.add_patch(
        Circle((0, 0), 15, edgecolor="white", facecolor="none", linewidth=1, alpha=0.7, zorder=2)
    )

    ax.set_xticks([-20, 0, 20])
    ax.set_yticks([-40, -20, 0, 20, 40])
    ax.tick_params(
        axis="both",
        colors="#cfd2d6",
        labelcolor="#e8eaed",
        labelsize=11,
        length=4,
        width=1,
        pad=6,
    )
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ax.set_xlabel("East [m]", color="#cfd2d6", fontsize=12, labelpad=12)
    ax.set_ylabel("North [m]", color="#cfd2d6", fontsize=12, labelpad=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("white")

    # Two-line header (avoid suptitle + ax.title overlap after tight_layout).
    fig.text(
        0.5,
        0.97,
        suptitle,
        ha="center",
        va="top",
        color="white",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.925,
        subtitle,
        ha="center",
        va="top",
        color="#e0e4e8",
        fontsize=13,
    )

    cbar = fig.colorbar(im, ax=ax, pad=0.04, shrink=0.88)
    cbar.set_label(cbar_label, color="white", fontsize=12, labelpad=10)
    cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white")

    fig.subplots_adjust(left=0.14, right=0.78, top=0.86, bottom=0.14)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, facecolor=fig_bg)
    plt.close(fig)
    print(f"Saved {out_png}", flush=True)


def plot_pdop_field_style(
    data: dict,
    *,
    suptitle: str,
    out_png: Path,
    vmin: float = 1.0,
    vmax: float = 20.0,
) -> None:
    """PDOP-only pitch heatmap (RdYlGn_r, NaN = dark, origin upper, North +40 at top)."""
    pdop_display = _pdop_display_grid(data)
    n_rows, n_cols = pdop_display.shape

    labels: list[list[str]] = []
    for i in range(n_rows):
        row_labels: list[str] = []
        for j in range(n_cols):
            val = pdop_display[i, j]
            if not math.isfinite(val):
                row_labels.append("—")
            else:
                row_labels.append(f"{val:.1f}")
        labels.append(row_labels)

    _render_pitch_field(
        pdop_display,
        labels,
        suptitle=suptitle,
        subtitle="Mean PDOP (LOS only)",
        out_png=out_png,
        vmin=vmin,
        vmax=vmax,
        cbar_label="PDOP",
    )


def plot_precision_field_style(
    data: dict,
    *,
    suptitle: str,
    out_png: Path,
    vmin_m: float = 0.05,
    vmax_m: float = 5.0,
    fix_display_row: int = 1,
    fix_display_cols: tuple[int, ...] = (2, 3),
    sigma_fix_m: float = 0.012,
    sigma_float_m: float = 0.25,
) -> None:
    """Horizontal error @ 95%% from PDOP: RTK FIX at pitch centre, FLOAT elsewhere.

    ``err_2d_95[m] = PDOP * 0.58 * sigma_0 * 2`` (HDOP ~ 0.58·PDOP, 95%% ≈ 2σ).
    """
    pdop_display = _pdop_display_grid(data)
    n_rows, n_cols = pdop_display.shape
    precision_data = np.full_like(pdop_display, np.nan, dtype=np.float64)
    labels: list[list[str]] = [["" for _ in range(n_cols)] for _ in range(n_rows)]

    sf_fix = 0.58 * float(sigma_fix_m) * 2.0
    sf_float = 0.58 * float(sigma_float_m) * 2.0
    fix_cols_set = set(int(c) for c in fix_display_cols)

    for i in range(n_rows):
        for j in range(n_cols):
            pdop = pdop_display[i, j]
            if not math.isfinite(pdop):
                labels[i][j] = "—"
                continue
            if i == int(fix_display_row) and j in fix_cols_set:
                val_m = float(pdop) * sf_fix
                precision_data[i, j] = val_m
                labels[i][j] = f"{val_m * 100:.1f} cm\n(FIX)"
            else:
                val_m = float(pdop) * sf_float
                precision_data[i, j] = val_m
                if val_m >= 1.0:
                    labels[i][j] = f"{val_m:.1f} m\n(FLOAT)"
                else:
                    labels[i][j] = f"{val_m * 100:.0f} cm\n(FLOAT)"

    _render_pitch_field(
        precision_data,
        labels,
        suptitle=suptitle,
        subtitle="Stima Errore Orizzontale RTK (95% Confidenza)",
        out_png=out_png,
        vmin=vmin_m,
        vmax=vmax_m,
        cbar_label="Errore di Posizionamento Stimato [m]",
        label_fontsize=10.0,
    )


def plot_separate_heatmaps(data: dict, *, title: str, out_base: Path) -> list[Path]:
    stem = out_base.stem
    if stem.endswith("_heatmap"):
        stem = stem[: -len("_heatmap")]
    parent = out_base.parent
    saved: list[Path] = []
    for spec in PANEL_SPECS:
        path = parent / f"{stem}_{spec['suffix']}.png"
        plot_single_heatmap(data, spec=spec, suptitle=title, out_png=path)
        saved.append(path)
    return saved


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-csv", type=Path, required=True)
    ap.add_argument("--out-png", type=Path, default=None)
    ap.add_argument("--title", type=str, default="Stadio Luigi Ferraris")
    ap.add_argument(
        "--combined-only",
        action="store_true",
        help="Only write the 3-panel combined PNG (skip separate *_los/_nlos/_pdop files).",
    )
    ap.add_argument(
        "--separate-only",
        action="store_true",
        help="Only write separate PNGs (skip combined 3-panel figure).",
    )
    ap.add_argument(
        "--no-pdop-campo",
        action="store_true",
        help="Skip the styled PDOP pitch figure (*_pdop_campo.png).",
    )
    ap.add_argument("--pdop-vmin", type=float, default=1.0)
    ap.add_argument("--pdop-vmax", type=float, default=20.0)
    ap.add_argument(
        "--no-precision",
        action="store_true",
        help="Skip RTK horizontal precision map (*_precision.png).",
    )
    ap.add_argument("--precision-vmin-m", type=float, default=0.05)
    ap.add_argument("--precision-vmax-m", type=float, default=5.0)
    ap.add_argument(
        "--precision-only",
        action="store_true",
        help="Only write the RTK precision pitch figure.",
    )
    args = ap.parse_args()

    inp = args.input_csv.resolve()
    out = args.out_png.resolve() if args.out_png else inp.with_name(inp.stem + "_heatmap.png")
    data = load_grid_csv(inp)
    title = str(args.title)
    stem = out.stem
    if stem.endswith("_heatmap"):
        stem = stem[: -len("_heatmap")]

    if args.precision_only:
        plot_precision_field_style(
            data,
            suptitle=title,
            out_png=out.parent / f"{stem}_precision.png",
            vmin_m=float(args.precision_vmin_m),
            vmax_m=float(args.precision_vmax_m),
        )
        return

    if not args.separate_only:
        plot_heatmaps(data, title=title, out_png=out)
    if not args.combined_only:
        plot_separate_heatmaps(data, title=title, out_base=out)
        if not args.no_pdop_campo:
            plot_pdop_field_style(
                data,
                suptitle=title,
                out_png=out.parent / f"{stem}_pdop_campo.png",
                vmin=float(args.pdop_vmin),
                vmax=float(args.pdop_vmax),
            )
        if not args.no_precision:
            plot_precision_field_style(
                data,
                suptitle=title,
                out_png=out.parent / f"{stem}_precision.png",
                vmin_m=float(args.precision_vmin_m),
                vmax_m=float(args.precision_vmax_m),
            )


if __name__ == "__main__":
    main()
