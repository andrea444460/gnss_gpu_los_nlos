#!/usr/bin/env python3
"""Plot a CHCNAV/CoPre-style .pos trajectory on an interactive Folium map."""

from __future__ import annotations

import argparse
from pathlib import Path

import folium
import numpy as np

Q_COLORS = {1: "#00c853", 2: "#ffab00", 3: "#ff5252", 4: "#9e9e9e", 5: "#7e57c2"}
TRACK_COLORS = ["#1565c0", "#e65100", "#6a1b9a", "#00838f", "#c62828"]


def read_pos(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read CHCNAV/CoPre or RTKLIB .pos trajectories."""
    lats: list[float] = []
    lons: list[float] = []
    qs: list[int] = []
    rtklib = False
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            if s.startswith("%"):
                if "latitude(deg)" in s or "program" in s.lower():
                    rtklib = True
                continue
            if s.startswith("Date"):
                continue
            parts = s.split()
            if len(parts) < 6:
                continue
            try:
                if rtklib:
                    lats.append(float(parts[2]))
                    lons.append(float(parts[3]))
                    qs.append(int(parts[5]))
                else:
                    lats.append(float(parts[2]))
                    lons.append(float(parts[3]))
                    qs.append(int(parts[5]))
            except ValueError:
                continue
    if not lats:
        raise ValueError(f"no points parsed from {path}")
    return np.asarray(lats), np.asarray(lons), np.asarray(qs)


def read_chcnav_pos(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  return read_pos(path)


def _segments_by_q(
    lat_arr: np.ndarray, lon_arr: np.ndarray, q_arr: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray, int]]:
    segs: list[tuple[np.ndarray, np.ndarray, int]] = []
    if len(lat_arr) == 0:
        return segs
    start = 0
    for i in range(1, len(q_arr)):
        if q_arr[i] != q_arr[i - 1]:
            segs.append((lat_arr[start : i + 1], lon_arr[start : i + 1], int(q_arr[start])))
            start = i
    segs.append((lat_arr[start:], lon_arr[start:], int(q_arr[start])))
    return segs


def build_map(
    lats: np.ndarray,
    lons: np.ndarray,
    qs: np.ndarray,
    *,
    title: str,
    max_points: int = 8000,
    line_color: str | None = None,
) -> folium.Map:
    stride = max(1, len(lats) // max_points)
    idx = np.arange(0, len(lats), stride)
    if idx[-1] != len(lats) - 1:
        idx = np.append(idx, len(lats) - 1)

    m = folium.Map(location=[float(lats.mean()), float(lons.mean())], zoom_start=15)
    folium.TileLayer("OpenStreetMap", name="OSM").add_to(m)
    folium.TileLayer("CartoDB positron", name="CartoDB").add_to(m)

    for slat, slon, q in _segments_by_q(lats[idx], lons[idx], qs[idx]):
        folium.PolyLine(
            locations=list(zip(slat.tolist(), slon.tolist())),
            color=line_color or Q_COLORS.get(q, "#2196f3"),
            weight=4,
            opacity=0.85,
            tooltip=f"{title} Q={q}" if line_color else f"Q={q}",
        ).add_to(m)

    folium.Marker(
        [float(lats[0]), float(lons[0])],
        popup="Start",
        icon=folium.Icon(color="green", icon="play"),
    ).add_to(m)
    folium.Marker(
        [float(lats[-1]), float(lons[-1])],
        popup="End",
        icon=folium.Icon(color="red", icon="stop"),
    ).add_to(m)
    m.fit_bounds([[float(lats.min()), float(lons.min())], [float(lats.max()), float(lons.max())]])

    q1 = int((qs == 1).sum())
    q2 = int((qs == 2).sum())
    other = int((~np.isin(qs, [1, 2])).sum())
    color_line = (
        f'<span style="color:{line_color}">&#9632;</span> track &nbsp;'
        if line_color
        else (
            '<span style="color:#00c853">&#9632;</span> Q1 &nbsp;'
            '<span style="color:#ffab00">&#9632;</span> Q2 &nbsp;'
            '<span style="color:#ff5252">&#9632;</span> Q3+ &nbsp;'
            '<span style="color:#2196f3">&#9632;</span> other'
        )
    )
    legend_html = f"""
<div style="position: fixed; bottom: 24px; left: 24px; z-index: 9999;
background: white; padding: 10px 12px; border-radius: 6px;
box-shadow: 0 1px 6px rgba(0,0,0,.25); font: 13px sans-serif;">
<b>{title}</b><br>
Points: {len(lats)}<br>
Q1 fix: {q1} &nbsp; Q2 float: {q2} &nbsp; other: {other}<br>
{color_line}
</div>
"""
    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl().add_to(m)
    return m


def add_track(
    m: folium.Map,
    lats: np.ndarray,
    lons: np.ndarray,
    qs: np.ndarray,
    *,
    name: str,
    color: str,
    max_points: int = 8000,
) -> None:
    stride = max(1, len(lats) // max_points)
    idx = np.arange(0, len(lats), stride)
    if idx[-1] != len(lats) - 1:
        idx = np.append(idx, len(lats) - 1)
    grp = folium.FeatureGroup(name=name, show=True)
    folium.PolyLine(
        locations=list(zip(lats[idx].tolist(), lons[idx].tolist())),
        color=color,
        weight=4,
        opacity=0.85,
        tooltip=name,
    ).add_to(grp)
    folium.CircleMarker(
        [float(lats[0]), float(lons[0])],
        radius=5,
        color=color,
        fill=True,
        popup=f"{name} start",
    ).add_to(grp)
    folium.CircleMarker(
        [float(lats[-1]), float(lons[-1])],
        radius=5,
        color=color,
        fill=True,
        popup=f"{name} end",
    ).add_to(grp)
    grp.add_to(m)


def build_compare_map(
    tracks: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]],
    *,
    title: str,
    max_points: int = 8000,
) -> folium.Map:
    all_lats = np.concatenate([t[1] for t in tracks])
    all_lons = np.concatenate([t[2] for t in tracks])
    m = folium.Map(
        location=[float(all_lats.mean()), float(all_lons.mean())],
        zoom_start=15,
    )
    folium.TileLayer("OpenStreetMap", name="OSM").add_to(m)
    folium.TileLayer("CartoDB positron", name="CartoDB").add_to(m)
    rows = []
    for i, (name, lats, lons, qs) in enumerate(tracks):
        color = TRACK_COLORS[i % len(TRACK_COLORS)]
        add_track(m, lats, lons, qs, name=name, color=color, max_points=max_points)
        q1 = int((qs == 1).sum())
        q2 = int((qs == 2).sum())
        rows.append(
            f'<span style="color:{color}">&#9632;</span> {name}: '
            f"{len(lats)} pts, Q1={q1}, Q2={q2}"
        )
    m.fit_bounds(
        [
            [float(all_lats.min()), float(all_lons.min())],
            [float(all_lats.max()), float(all_lons.max())],
        ]
    )
    legend_html = f"""
<div style="position: fixed; bottom: 24px; left: 24px; z-index: 9999;
background: white; padding: 10px 12px; border-radius: 6px;
box-shadow: 0 1px 6px rgba(0,0,0,.25); font: 13px sans-serif;">
<b>{title}</b><br>
{"<br>".join(rows)}
</div>
"""
    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl().add_to(m)
    return m


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pos", type=Path, action="append", required=True, help="Input .pos file")
    p.add_argument("--label", action="append", default=[], help="Legend label (same order as --pos)")
    p.add_argument(
        "--out-html",
        type=Path,
        default=None,
        help="Output HTML (default: next to first input)",
    )
    args = p.parse_args()
    paths = [x.resolve() for x in args.pos]
    labels = args.label or [p.name for p in paths]
    if len(labels) != len(paths):
        raise SystemExit("need one --label per --pos file")
    if len(paths) == 1:
        out = args.out_html or paths[0].with_name(f"{paths[0].stem}_map.html")
        lats, lons, qs = read_pos(paths[0])
        m = build_map(lats, lons, qs, title=labels[0])
        m.save(str(out))
        print(f"Wrote {out}")
        print(f"Points: {len(lats)}")
        print(
            f"Bounds: lat [{lats.min():.6f}, {lats.max():.6f}] "
            f"lon [{lons.min():.6f}, {lons.max():.6f}]"
        )
        return

    out = args.out_html or paths[0].with_name(
        "_vs_".join(p.stem for p in paths) + "_map.html"
    )
    tracks = []
    for path, label in zip(paths, labels):
        lats, lons, qs = read_pos(path)
        tracks.append((label, lats, lons, qs))
    m = build_compare_map(tracks, title=" / ".join(labels))
    m.save(str(out))
    print(f"Wrote {out}")
    for name, lats, lons, _ in tracks:
        print(
            f"{name}: {len(lats)} pts, "
            f"lat [{lats.min():.6f}, {lats.max():.6f}], "
            f"lon [{lons.min():.6f}, {lons.max():.6f}]"
        )


if __name__ == "__main__":
    main()
