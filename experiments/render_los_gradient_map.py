#!/usr/bin/env python3
"""Render LOS/NLOS CSV points as a gradient heatmap in Folium."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import branca
import folium
import numpy as np
from folium.plugins import HeatMap
from branca.element import MacroElement


def _load_rows(csv_path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                rows.append(
                    {
                        "lat": float(row["lat_deg"]),
                        "lon": float(row["lon_deg"]),
                        "los": float(row["mean_n_los"]),
                        "nlos": float(row.get("mean_n_nlos", 0.0)),
                    }
                )
            except Exception:
                continue
    return rows


class BasemapToggleControl(MacroElement):
    """Leaflet control to hide/show basemap tile pane."""

    def __init__(self, *, position: str = "topright"):
        super().__init__()
        self._name = "BasemapToggleControl"
        self.position = position
        self._template = branca.element.Template(
            """
            {% macro script(this, kwargs) %}
            (function() {
              var map = {{this._parent.get_name()}};
              var visible = true;
              var control = L.control({position: '{{this.position}}'});
              control.onAdd = function() {
                var div = L.DomUtil.create('div', 'leaflet-bar');
                var a = L.DomUtil.create('a', '', div);
                a.href = '#';
                a.title = 'Toggle basemap';
                a.innerHTML = 'Map';
                a.style.width = '42px';
                a.style.textAlign = 'center';
                a.style.background = '#fff';
                a.style.font = '12px sans-serif';
                a.style.lineHeight = '26px';
                L.DomEvent.on(a, 'click', function(e) {
                  L.DomEvent.stop(e);
                  visible = !visible;
                  var pane = map.getPane('tilePane');
                  if (pane) {
                    pane.style.display = visible ? 'block' : 'none';
                  }
                  a.style.opacity = visible ? '1.0' : '0.6';
                });
                return div;
              };
              control.addTo(map);
            })();
            {% endmacro %}
            """
        )


def main() -> None:
    p = argparse.ArgumentParser(description="Render gradient heatmap from LOS/NLOS CSV")
    p.add_argument("--csv", type=Path, required=True, help="Input CSV from build_area_los_nlos_map.py")
    p.add_argument("--out-html", type=Path, required=True, help="Output gradient map HTML")
    p.add_argument("--metric", choices=("los", "nlos"), default="los", help="Metric used for heat intensity")
    p.add_argument("--radius", type=int, default=16, help="Heat kernel radius")
    p.add_argument("--blur", type=int, default=20, help="Heat blur")
    p.add_argument("--min-opacity", type=float, default=0.35, help="Minimum heat opacity")
    p.add_argument("--point-layer", action="store_true", help="Overlay original points for debugging")
    args = p.parse_args()

    rows = _load_rows(args.csv)
    if not rows:
        raise RuntimeError("No valid rows parsed from CSV.")

    lat0 = float(np.mean([r["lat"] for r in rows]))
    lon0 = float(np.mean([r["lon"] for r in rows]))
    m = folium.Map(location=[lat0, lon0], zoom_start=14, tiles="CartoDB positron")

    key = "los" if args.metric == "los" else "nlos"
    vals = np.asarray([r[key] for r in rows], dtype=np.float64)
    vmin = float(np.nanpercentile(vals, 5))
    vmax = float(np.nanpercentile(vals, 95))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(np.nanmin(vals))
        vmax = float(np.nanmax(vals) + 1.0)

    heat = []
    for r in rows:
        w = (r[key] - vmin) / (vmax - vmin)
        w = float(np.clip(w, 0.0, 1.0))
        heat.append([r["lat"], r["lon"], w])

    gradient = {
        0.0: "#313695",
        0.25: "#74add1",
        0.5: "#ffffbf",
        0.75: "#f46d43",
        1.0: "#a50026",
    }

    HeatMap(
        heat,
        name=f"{args.metric.upper()} gradient",
        radius=int(args.radius),
        blur=int(args.blur),
        min_opacity=float(args.min_opacity),
        max_zoom=17,
        gradient=gradient,
    ).add_to(m)

    if args.point_layer:
        for r in rows:
            folium.CircleMarker(
                location=[r["lat"], r["lon"]],
                radius=1,
                weight=0,
                fill=True,
                fill_opacity=0.55,
                color="#111111",
            ).add_to(m)

    # Add an in-map UI toggle to hide/show basemap tiles.
    m.add_child(BasemapToggleControl(position="topright"))
    folium.LayerControl().add_to(m)
    args.out_html.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(args.out_html))
    print(f"rows={len(rows)} metric={args.metric} saved={args.out_html}")


if __name__ == "__main__":
    main()
