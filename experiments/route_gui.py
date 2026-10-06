#!/usr/bin/env python3
"""Minimal Leaflet GUI to inspect contracted / time-extended GNSS routes.

Default: offline Genova street fixture (real street-shaped polylines + names).
Optional live Overpass if the network allows it.

    PYTHONPATH=python python experiments/route_gui.py --port 8765
    PYTHONPATH=python python experiments/route_gui.py --demo overpass
    PYTHONPATH=python python experiments/route_gui.py --demo synthetic

Open http://127.0.0.1:8765 — click origin, click destination, Route.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_ROOT / "python"))

from gnss_gpu.io.osm_roads import (  # noqa: E402
    BBox,
    RoadEdge,
    RoadGraph,
    build_directed_road_graph,
    fetch_roads_overpass,
)
from gnss_gpu.routing import (  # noqa: E402
    CostParams,
    build_te_from_timeseries,
    path_to_geojson,
    prepare_contracted_graph,
    route_contracted_latlon,
    route_te_latlon,
    te_path_to_geojson,
)
from gnss_gpu.routing_graph import (  # noqa: E402
    make_demo_spatial_graph,
    quality_at,
    synthesize_quality_timeseries,
)

# Genova centro storico — small enough for Overpass, real street geometry
DEFAULT_BBOX = BBox(south=44.4000, west=8.9200, north=44.4140, east=8.9450)
FIXTURE_ROADS = (
    Path(__file__).resolve().parents[1] / "python" / "gnss_gpu" / "fixtures" / "genova_centro_roads.json"
)


def _hdop_color(hdop: float) -> str:
    """Green (good) → amber → red (bad) from HDOP ∈ [1, 8]."""
    if not math.isfinite(hdop):
        return "#6b7280"
    t = max(0.0, min(1.0, (float(hdop) - 1.0) / 7.0))
    # Piecewise: green → yellow → orange → red
    if t < 0.33:
        u = t / 0.33
        r, g, b = int(40 + 200 * u), int(180 + 40 * u), int(70 * (1.0 - u))
    elif t < 0.66:
        u = (t - 0.33) / 0.33
        r, g, b = int(240), int(220 - 100 * u), int(40)
    else:
        u = (t - 0.66) / 0.34
        r, g, b = int(240 - 40 * u), int(120 - 90 * u), int(40)
    return f"#{r:02x}{g:02x}{b:02x}"


def _quality_color(hdop: float, n_los: float) -> str:
    """Blend HDOP + LOS count into one display score (higher = worse)."""
    if not math.isfinite(hdop):
        return _hdop_color(hdop)
    # Fewer LOS satellites worsen the displayed quality.
    los_pen = 0.0
    if math.isfinite(n_los):
        los_pen = max(0.0, (8.0 - float(n_los)) / 8.0) * 3.0
    return _hdop_color(float(hdop) + los_pen)


CAR_HIGHWAYS = frozenset(
    {
        "motorway",
        "trunk",
        "primary",
        "secondary",
        "tertiary",
        "unclassified",
        "residential",
        "living_street",
        "service",
        "motorway_link",
        "trunk_link",
        "primary_link",
        "secondary_link",
        "tertiary_link",
    }
)


def _filter_car_ways(roads: list[dict]) -> list[dict]:
    """Keep only motor-vehicle highway classes (no footway/pedestrian/cycleway)."""
    out = []
    for w in roads:
        hw = str((w.get("tags") or {}).get("highway", "")).strip().lower()
        if hw in CAR_HIGHWAYS:
            out.append(w)
    return out


def _graph_from_roads(roads: list[dict], source: str):
    roads = _filter_car_ways(roads)
    graph = build_directed_road_graph(roads)
    if not graph.edges:
        raise RuntimeError("no road edges built")
    samples = synthesize_quality_timeseries(graph)
    return graph, samples, source, roads


def _load_fixture_graph():
    payload = json.loads(FIXTURE_ROADS.read_text(encoding="utf-8"))
    roads = [el for el in payload.get("elements", []) if el.get("type") == "way"]
    roads = _filter_car_ways(roads)
    return _graph_from_roads(roads, f"fixture:{FIXTURE_ROADS.name} ({len(roads)} car ways)")


def _load_overpass_graph(bbox: BBox):
    roads = fetch_roads_overpass(
        bbox,
        include_pedestrian=False,
        timeout_s=45,
        max_attempts_per_endpoint=2,
    )
    return _graph_from_roads(
        roads,
        f"overpass:{bbox.south},{bbox.west},{bbox.north},{bbox.east} ({len(roads)} ways)",
    )


def _way_quality(samples: dict, way_id: int, t_s: float = 0.0) -> tuple[float, float]:
    seq = samples.get(way_id) or []
    if not seq:
        return float("nan"), float("nan")
    # pick nearest sample at/before t_s
    best = seq[0]
    for row in seq:
        if row[0] <= t_s:
            best = row
        else:
            break
    return float(best[1]), float(best[2])


class DemoState:
    def __init__(self, *, demo: str = "fixture", bbox: BBox = DEFAULT_BBOX) -> None:
        self.hdop_step = 0.5
        self.n_los_step = 1.0
        self.source = ""
        self.bbox = bbox
        self.roads: list[dict] = []
        if demo == "synthetic":
            self.spatial, self.samples = make_demo_spatial_graph()
            self.source = "synthetic rectangular block (unit-test only)"
            self.roads = []
        elif demo == "overpass":
            try:
                self.spatial, self.samples, self.source, self.roads = _load_overpass_graph(bbox)
            except Exception as exc:  # noqa: BLE001
                print(f"WARNING: Overpass failed ({exc}); using offline fixture", flush=True)
                self.spatial, self.samples, self.source, self.roads = _load_fixture_graph()
                self.source += f" [overpass fallback: {exc}]"
        else:
            self.spatial, self.samples, self.source, self.roads = _load_fixture_graph()

        # Stamp initial quality onto edges for contraction helpers
        for e in self.spatial.edges:
            seq = self.samples.get(e.way_id) or []
            if seq:
                e.mean_hdop, e.mean_n_los = seq[0][1], seq[0][2]

        self.te = None
        self.timelines = None
        self._te_ready = False
        self.contracted = None  # built lazily — map display needs only OSM ways + samples

    def _ensure_contracted(self, t_s: float = 0.0):
        """Contracted routing graph for a quality snapshot (cached for t≈0)."""
        if t_s <= 1e-9 and self.contracted is not None:
            return self.contracted
        cg = prepare_contracted_graph(
            self._snapshot(t_s), hdop_step=self.hdop_step, n_los_step=self.n_los_step
        )
        if t_s <= 1e-9:
            self.contracted = cg
        return cg

    def _ensure_timelines(self) -> None:
        if self.timelines is not None:
            return
        from gnss_gpu.routing_graph import attach_timelines_by_way

        self.timelines = attach_timelines_by_way(
            self.spatial,
            self.samples,
            hdop_step=self.hdop_step,
            n_los_step=self.n_los_step,
        )

    def _ensure_te(self):
        if self._te_ready and self.te is not None:
            return self.te
        print("Building time-extended graph (first TE use)…", flush=True)
        self.te, self.timelines = build_te_from_timeseries(
            self.spatial,
            self.samples,
            hdop_step=self.hdop_step,
            n_los_step=self.n_los_step,
            wait_cost=0.0,
            contract_per_layer=True,
        )
        self._te_ready = True
        print(
            f"TE ready: nodes={len(self.te.nodes)} edges={len(self.te.edges)} layers={len(self.te.layers)}",
            flush=True,
        )
        return self.te

    def _layer_time(self, layer: int) -> float:
        """Mid-time of a quality layer without requiring the full TE graph."""
        self._ensure_timelines()
        from gnss_gpu.routing_graph import build_layer_boundaries

        bounds = build_layer_boundaries(self.timelines or [])
        if len(bounds) < 2:
            return 0.0
        layers = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1) if bounds[i + 1] > bounds[i]]
        if not layers:
            return 0.0
        layer = max(0, min(int(layer), len(layers) - 1))
        t0, t1 = layers[layer]
        return 0.5 * (t0 + t1)

    def _layer_meta(self) -> list[dict]:
        self._ensure_timelines()
        from gnss_gpu.routing_graph import build_layer_boundaries

        bounds = build_layer_boundaries(self.timelines or [])
        out = []
        idx = 0
        for i in range(len(bounds) - 1):
            if bounds[i + 1] <= bounds[i]:
                continue
            out.append({"index": idx, "t0_s": bounds[i], "t1_s": bounds[i + 1]})
            idx += 1
        return out or [{"index": 0, "t0_s": 0.0, "t1_s": 1.0}]

    def _snapshot(self, t_s: float) -> RoadGraph:
        self._ensure_timelines()
        edges = []
        for e, tl in zip(self.spatial.edges, self.timelines):
            h, n = quality_at(tl, t_s)
            edges.append(
                RoadEdge(
                    u=e.u,
                    v=e.v,
                    length_m=e.length_m,
                    way_id=e.way_id,
                    highway=e.highway,
                    name=e.name,
                    geometry=list(e.geometry),
                    mean_hdop=h,
                    mean_n_los=n,
                )
            )
        return RoadGraph(nodes=dict(self.spatial.nodes), edges=edges)

    def _edges_to_features(self, edges, *, style: str = "fine") -> list[dict]:
        feats = []
        seen: set[tuple[int, int, int]] = set()
        for e in edges:
            if len(e.geometry) < 2:
                continue
            a, b = sorted((e.u, e.v))
            key = (a, b, e.way_id)
            if key in seen:
                continue
            seen.add(key)
            feats.append(
                {
                    "type": "Feature",
                    "properties": {
                        "way_id": e.way_id,
                        "length_m": e.length_m,
                        "mean_hdop": e.mean_hdop,
                        "mean_n_los": e.mean_n_los,
                        "color": _quality_color(e.mean_hdop, e.mean_n_los),
                        "name": e.name,
                        "highway": e.highway,
                        "style": style,
                    },
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [[lon, lat] for lat, lon in e.geometry],
                    },
                }
            )
        return feats

    def graph_geojson(self, *, mode: str, layer: int, show_contracted_overlay: bool = False) -> dict:
        """Map layer: full OSM way polylines (match basemap), not chord stubs.

        Routing still uses the contracted graph; display uses original way
        geometry so streets sit on the OpenStreetMap tiles.
        """
        t_s = self._layer_time(layer)

        feats: list[dict] = []
        if self.roads:
            for way in self.roads:
                geom = way.get("geometry") or []
                if len(geom) < 2:
                    continue
                tags = way.get("tags") or {}
                wid = int(way.get("id", -1))
                hdop, n_los = _way_quality(self.samples, wid, t_s)
                feats.append(
                    {
                        "type": "Feature",
                        "properties": {
                            "way_id": wid,
                            "length_m": None,
                            "mean_hdop": hdop,
                            "mean_n_los": n_los,
                            "color": _quality_color(hdop, n_los),
                            "name": str(tags.get("name", "")),
                            "highway": str(tags.get("highway", "")),
                            "style": "fine",
                        },
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [
                                [float(p["lon"]), float(p["lat"])] for p in geom
                            ],
                        },
                    }
                )
        else:
            fine = self._snapshot(t_s)
            feats.extend(self._edges_to_features(fine.edges, style="fine"))

        if show_contracted_overlay:
            fine = self._snapshot(t_s)
            cg = self._ensure_contracted(t_s) if t_s <= 1e-9 else prepare_contracted_graph(
                fine, hdop_step=self.hdop_step, n_los_step=self.n_los_step
            )
            feats.extend(self._edges_to_features(cg.graph.edges, style="contracted"))
        return {"type": "FeatureCollection", "features": feats}

    def meta(self) -> dict:
        return {
            "source": self.source,
            "n_spatial_nodes": len(self.spatial.nodes),
            "n_spatial_edges_fine": len(self.spatial.edges),
            "n_contracted_edges_t0": (
                len(self.contracted.graph.edges) if self.contracted is not None else None
            ),
            "n_te_nodes": len(self.te.nodes) if self.te is not None else None,
            "n_te_edges": len(self.te.edges) if self.te is not None else None,
            "te_built": bool(self._te_ready),
            "layers": self._layer_meta(),
            "note": (
                "Map draws full OSM way centerlines (same coords as the basemap). "
                "Routing snaps to routable contracted nodes (not orphan mid-edge nodes). "
                "Yellow path is expanded via the member map. TE graph builds lazily."
            ),
        }


STATE: DemoState | None = None

HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>GNSS route lab</title>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  :root { --bg:#1a1f24; --panel:#24303a; --ink:#e8eef2; --accent:#3dbb7a; --muted:#8aa0b0; }
  * { box-sizing: border-box; }
  html, body { margin:0; height:100%; font:14px/1.4 "IBM Plex Sans", system-ui, sans-serif; background:var(--bg); color:var(--ink); }
  #wrap { display:grid; grid-template-columns: 320px 1fr; height:100%; }
  aside { padding:16px; background:linear-gradient(160deg,#24303a,#1a222a); border-right:1px solid #31404c; overflow:auto; }
  h1 { font-size:18px; margin:0 0 8px; font-weight:600; letter-spacing:.02em; }
  p.note { color:var(--muted); font-size:12px; margin:0 0 14px; }
  label { display:block; margin:10px 0 4px; color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.04em; }
  input, select, button { width:100%; padding:8px 10px; border-radius:6px; border:1px solid #3a4b58; background:#152028; color:var(--ink); }
  button { background:var(--accent); color:#062416; border:none; font-weight:600; cursor:pointer; margin-top:12px; }
  button.secondary { background:#31404c; color:var(--ink); }
  #map { height:100%; }
  #stats { margin-top:14px; font-size:12px; white-space:pre-wrap; background:#152028; padding:10px; border-radius:6px; color:#c5d6e0; max-height:40vh; overflow:auto; }
  .row { display:grid; grid-template-columns:1fr 1fr; gap:8px; }
  #legend { margin:12px 0; padding:10px; background:#152028; border-radius:6px; font-size:12px; color:#c5d6e0; }
  #legend h2 { margin:0 0 6px; font-size:12px; font-weight:600; letter-spacing:.04em; text-transform:uppercase; color:var(--muted); }
  #legend .formula { color:var(--muted); font-size:11px; margin:0 0 8px; line-height:1.35; }
  #legend table { width:100%; border-collapse:collapse; }
  #legend td { padding:4px 0; vertical-align:middle; }
  #legend .swatch { width:18px; height:12px; border-radius:2px; display:inline-block; margin-right:8px; border:1px solid #3a4b58; }
  #legend .param { color:#e8eef2; }
  #legend .hint { color:var(--muted); font-size:11px; }
</style>
</head>
<body>
<div id="wrap">
  <aside>
    <h1>GNSS route lab</h1>
    <p class="note">Click A, then B, then Route. Change <b>Layer</b> to see quality over time.</p>
    <div id="legend">
      <h2>Legenda colore archi</h2>
      <p class="formula">Colore = score GNSS ≈ HDOP + 3·max(0, (8 − n<sub>LOS</sub>)/8).<br/>Verde = buona qualità, rosso = scarsa.</p>
      <table>
        <tr><td><span class="swatch" style="background:#53bc36"></span><span class="param">HDOP ≈ 1–2</span></td><td class="hint">n<sub>LOS</sub> ≥ 8 · ottima</td></tr>
        <tr><td><span class="swatch" style="background:#f0d328"></span><span class="param">HDOP ≈ 3–4</span></td><td class="hint">n<sub>LOS</sub> ≈ 5–7 · media</td></tr>
        <tr><td><span class="swatch" style="background:#f07d28"></span><span class="param">HDOP ≈ 5–6</span></td><td class="hint">n<sub>LOS</sub> ≈ 3–4 · scarsa</td></tr>
        <tr><td><span class="swatch" style="background:#c81e28"></span><span class="param">HDOP ≥ 7</span></td><td class="hint">n<sub>LOS</sub> ≤ 2 · pessima</td></tr>
        <tr><td><span class="swatch" style="background:#6b7280"></span><span class="param">n/d</span></td><td class="hint">nessuna metrica</td></tr>
      </table>
    </div>
    <p id="hint" class="note" style="color:#3dbb7a">Click anywhere on the map to place A (origin).</p>
    <label>Mode</label>
    <select id="mode">
      <option value="spatial">Spatial routing</option>
      <option value="te">Time-extended routing</option>
    </select>
    <label><input id="overlay" type="checkbox"/> show contracted overlay (dashed)</label>
    <label>Layer</label>
    <input id="layer" type="number" min="0" value="0"/>
    <div class="row">
      <div>
        <label>alpha</label>
        <input id="alpha" type="number" step="0.1" value="1"/>
      </div>
      <div>
        <label>beta</label>
        <input id="beta" type="number" step="0.1" value="0.5"/>
      </div>
    </div>
    <div class="row">
      <div>
        <label>start layer (TE)</label>
        <input id="startLayer" type="number" min="-1" value="-1" title="-1 = any"/>
      </div>
      <div>
        <label>algorithm</label>
        <select id="algo"><option value="dijkstra">Dijkstra</option><option value="astar">A*</option></select>
      </div>
    </div>
    <button id="btnRoute">Route A → B</button>
    <button id="btnClear" class="secondary">Clear markers</button>
    <button id="btnReload" class="secondary">Reload graph</button>
    <div id="stats">Loading…</div>
  </aside>
  <div id="map"></div>
</div>
<script>
const map = L.map('map');
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  maxZoom: 19, attribution: '&copy; OpenStreetMap'
}).addTo(map);
let edgeLayer = L.layerGroup().addTo(map);
let pathLayer = L.layerGroup().addTo(map);
let markerLayer = L.layerGroup().addTo(map);
let markers = [];
let origin = null, dest = null;

function setHint(){
  const el = document.getElementById('hint');
  if (!el) return;
  if (!origin) el.textContent = 'Click anywhere on the map to place A (origin).';
  else if (!dest) el.textContent = 'Click anywhere else to place B (destination).';
  else el.textContent = 'A and B set — press Route, or Clear to reset.';
}

function placeMarker(latlng, kind){
  const color = kind === 'A' ? '#3dbb7a' : '#e85d4c';
  const m = L.circleMarker(latlng, {
    radius: 9,
    color,
    fillColor: color,
    fillOpacity: 1,
    weight: 2,
    interactive: false,
  }).bindTooltip(kind, {permanent: true, direction: 'top', offset: [0, -10]});
  m.addTo(markerLayer);
  markers.push(m);
  return m;
}

async function loadMeta(){
  const m = await (await fetch('/api/meta')).json();
  document.getElementById('layer').max = Math.max(0, (m.layers||[]).length-1);
  document.getElementById('stats').textContent = JSON.stringify(m, null, 2);
  setHint();
}

async function loadGraph(){
  const mode = document.getElementById('mode').value;
  const layer = parseInt(document.getElementById('layer').value||'0',10);
  const overlay = document.getElementById('overlay').checked;
  const g = await (await fetch(`/api/graph?mode=${mode}&layer=${layer}&overlay=${overlay}`)).json();
  edgeLayer.clearLayers();
  const layer2 = L.geoJSON(g, {
    // Roads must NOT capture clicks, otherwise B can only be placed where
    // there is no green polyline (felt like "only on A").
    interactive: false,
    style: f => {
      const contracted = f.properties.style === 'contracted';
      return {
        color: contracted ? '#9ec9ff' : (f.properties.color || '#3dbb7a'),
        weight: contracted ? 2 : 5,
        opacity: contracted ? 0.85 : 0.9,
        dashArray: contracted ? '6 6' : null,
      };
    },
  }).addTo(edgeLayer);
  if (g.features.length) map.fitBounds(layer2.getBounds(), {padding:[30,30]});
}

map.on('click', (e) => {
  if (!origin) {
    origin = e.latlng;
    placeMarker(origin, 'A');
  } else if (!dest) {
    dest = e.latlng;
    placeMarker(dest, 'B');
  } else {
    // third click moves B
    dest = e.latlng;
    if (markers.length >= 2) markerLayer.removeLayer(markers.pop());
    placeMarker(dest, 'B');
  }
  setHint();
});

document.getElementById('btnClear').onclick = () => {
  origin = dest = null;
  markerLayer.clearLayers();
  markers = [];
  pathLayer.clearLayers();
  setHint();
};

document.getElementById('btnReload').onclick = () => loadGraph();
document.getElementById('mode').onchange = () => loadGraph();
document.getElementById('layer').onchange = () => loadGraph();
document.getElementById('overlay').onchange = () => loadGraph();

document.getElementById('btnRoute').onclick = async () => {
  if (!origin || !dest) { alert('Click origin A, then destination B anywhere on the map'); return; }
  const body = {
    lat_a: origin.lat, lon_a: origin.lng,
    lat_b: dest.lat, lon_b: dest.lng,
    alpha: parseFloat(document.getElementById('alpha').value),
    beta: parseFloat(document.getElementById('beta').value),
    mode: document.getElementById('mode').value,
    layer: parseInt(document.getElementById('layer').value||'0',10),
    start_layer: parseInt(document.getElementById('startLayer').value||'-1',10),
    algorithm: document.getElementById('algo').value,
  };
  document.getElementById('stats').textContent = 'Routing…';
  let res;
  try {
    res = await (await fetch('/api/route', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)})).json();
  } catch (err) {
    document.getElementById('stats').textContent = String(err);
    alert('Route request failed: ' + err);
    return;
  }
  pathLayer.clearLayers();
  if (!res.ok) {
    document.getElementById('stats').textContent = JSON.stringify(res, null, 2);
    alert(res.error || 'no path');
    return;
  }
  L.geoJSON(res.path, {
    interactive: false,
    style: { color:'#f5d76e', weight:7, opacity:0.95 },
  }).addTo(pathLayer);
  document.getElementById('stats').textContent = JSON.stringify(res.summary, null, 2);
};

loadMeta().then(loadGraph);
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj: object) -> None:
        self._send(code, json.dumps(obj).encode("utf-8"), "application/json")

    def do_GET(self) -> None:  # noqa: N802
        assert STATE is not None
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            self._send(200, HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/meta":
            self._json(200, STATE.meta())
            return
        if parsed.path == "/api/graph":
            qs = parse_qs(parsed.query)
            mode = qs.get("mode", ["spatial"])[0]
            layer = int(qs.get("layer", ["0"])[0])
            overlay = qs.get("overlay", ["false"])[0].lower() == "true"
            self._json(
                200,
                STATE.graph_geojson(mode=mode, layer=layer, show_contracted_overlay=overlay),
            )
            return
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        assert STATE is not None
        parsed = urlparse(self.path)
        if parsed.path != "/api/route":
            self._json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        try:
            req = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self._json(400, {"ok": False, "error": "bad json"})
            return

        params = CostParams(
            alpha=float(req.get("alpha", 1.0)),
            beta=float(req.get("beta", 0.5)),
        )
        lat_a = float(req["lat_a"])
        lon_a = float(req["lon_a"])
        lat_b = float(req["lat_b"])
        lon_b = float(req["lon_b"])
        mode = str(req.get("mode", "spatial"))
        layer = int(req.get("layer", 0))
        start_layer = int(req.get("start_layer", -1))
        algorithm = str(req.get("algorithm", "dijkstra"))

        if mode == "te":
            te = STATE._ensure_te()
            sl = None if start_layer < 0 else start_layer
            route = route_te_latlon(
                te,
                lat_a,
                lon_a,
                lat_b,
                lon_b,
                params,
                start_layer=sl,
            )
            if route is None:
                self._json(200, {"ok": False, "error": "no TE path"})
                return
            path = te_path_to_geojson(te, route, properties={"kind": "te"})
            summary = {
                "mode": "te",
                "source": STATE.source,
                "length_m": route.length_m,
                "total_cost": route.total_cost,
                "mean_hdop": route.mean_hdop,
                "mean_n_los": route.mean_n_los,
                "layers": route.layers,
                "spatial_node_ids": route.spatial_node_ids,
                "n_travel_contracted": sum(
                    1 for i in route.edge_indices if te.edges[i].kind == "travel"
                ),
                "n_travel_fine": sum(
                    len(te.edges[i].fine_edge_indices)
                    for i in route.edge_indices
                    if te.edges[i].kind == "travel"
                ),
                "start_spatial": route.spatial_node_ids[0] if route.spatial_node_ids else None,
                "goal_spatial": route.spatial_node_ids[-1] if route.spatial_node_ids else None,
                "start_layer": sl,
            }
            self._json(200, {"ok": True, "path": path, "summary": summary})
            return

        t_s = STATE._layer_time(layer)
        cg = STATE._ensure_contracted(t_s)
        route = route_contracted_latlon(
            cg, lat_a, lon_a, lat_b, lon_b, params, algorithm=algorithm
        )
        if route is None:
            self._json(200, {"ok": False, "error": "no spatial path"})
            return
        path = path_to_geojson(cg.fine, route, properties={"kind": "spatial_expanded"})
        summary = {
            "mode": "spatial",
            "source": STATE.source,
            "routed_on": "contracted",
            "displayed_as": "fine_expanded",
            "layer": layer,
            "t_s": t_s,
            "length_m": route.length_m,
            "total_cost": route.total_cost,
            "mean_hdop": route.mean_hdop,
            "mean_n_los": route.mean_n_los,
            "n_edges_contracted": len(route.edge_indices),
            "n_edges_fine": len(route.fine_edge_indices),
            "node_ids_contracted": route.node_ids,
            "node_ids_fine": route.fine_node_ids,
            "n_graph_edges_contracted": len(cg.graph.edges),
            "n_graph_edges_fine": len(cg.fine.edges),
        }
        self._json(200, {"ok": True, "path": path, "summary": summary})

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))


class _ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


def main(argv: list[str] | None = None) -> int:
    global STATE
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument(
        "--demo",
        choices=("fixture", "overpass", "synthetic"),
        default="fixture",
        help="fixture=offline Genova streets (default); overpass=live OSM; synthetic=tiny rectangle",
    )
    p.add_argument(
        "--bbox",
        default=None,
        help="south,west,north,east (default: Genova centro snippet)",
    )
    args = p.parse_args(argv)
    bbox = DEFAULT_BBOX
    if args.bbox:
        parts = [float(x) for x in args.bbox.replace(" ", "").split(",")]
        bbox = BBox(south=parts[0], west=parts[1], north=parts[2], east=parts[3])

    print(f"Loading graph (demo={args.demo})…", flush=True)
    STATE = DemoState(demo=args.demo, bbox=bbox)
    print(f"source: {STATE.source}", flush=True)
    print(
        f"nodes={len(STATE.spatial.nodes)} edges={len(STATE.spatial.edges)} "
        f"(contraction/TE lazy)",
        flush=True,
    )

    httpd = _ReusableThreadingHTTPServer((args.host, args.port), Handler)
    print(f"GNSS route GUI at http://{args.host}:{args.port}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nbye", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
