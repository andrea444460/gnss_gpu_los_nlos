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
    dijkstra_te_route,
    path_to_geojson,
    prepare_contracted_graph,
    route_contracted_latlon,
    snap_nearest_node,
    te_path_to_geojson,
)
from gnss_gpu.routing_graph import (  # noqa: E402
    make_demo_spatial_graph,
    quality_at,
    synthesize_quality_timeseries,
)

# Genova centro storico — small enough for Overpass, real street geometry
DEFAULT_BBOX = BBox(south=44.4055, west=8.9305, north=44.4085, east=8.9355)
FIXTURE_ROADS = (
    Path(__file__).resolve().parents[1] / "python" / "gnss_gpu" / "fixtures" / "genova_centro_roads.json"
)


def _hdop_color(hdop: float) -> str:
    if not math.isfinite(hdop):
        return "#888888"
    t = max(0.0, min(1.0, (hdop - 1.0) / 7.0))
    r = int(255 * t)
    g = int(200 * (1.0 - t))
    return f"#{r:02x}{g:02x}40"


def _graph_from_roads(roads: list[dict], source: str):
    graph = build_directed_road_graph(roads)
    if not graph.edges:
        raise RuntimeError("no road edges built")
    samples = synthesize_quality_timeseries(graph)
    return graph, samples, source


def _load_fixture_graph() -> tuple[RoadGraph, dict[int, list[tuple[float, float, float]]], str]:
    payload = json.loads(FIXTURE_ROADS.read_text(encoding="utf-8"))
    roads = [el for el in payload.get("elements", []) if el.get("type") == "way"]
    return _graph_from_roads(roads, f"fixture:{FIXTURE_ROADS.name} ({len(roads)} ways)")


def _load_overpass_graph(bbox: BBox) -> tuple[RoadGraph, dict[int, list[tuple[float, float, float]]], str]:
    roads = fetch_roads_overpass(
        bbox,
        include_pedestrian=False,
        timeout_s=25,
        max_attempts_per_endpoint=1,
    )
    return _graph_from_roads(
        roads,
        f"overpass:{bbox.south},{bbox.west},{bbox.north},{bbox.east} ({len(roads)} ways)",
    )


class DemoState:
    def __init__(self, *, demo: str = "fixture", bbox: BBox = DEFAULT_BBOX) -> None:
        self.hdop_step = 0.5
        self.n_los_step = 1.0
        self.source = ""
        self.bbox = bbox
        if demo == "synthetic":
            self.spatial, self.samples = make_demo_spatial_graph()
            self.source = "synthetic rectangular block (unit-test only)"
        elif demo == "overpass":
            try:
                self.spatial, self.samples, self.source = _load_overpass_graph(bbox)
            except Exception as exc:  # noqa: BLE001
                print(f"WARNING: Overpass failed ({exc}); using offline fixture", flush=True)
                self.spatial, self.samples, self.source = _load_fixture_graph()
                self.source += f" [overpass fallback: {exc}]"
        else:
            self.spatial, self.samples, self.source = _load_fixture_graph()

        # Stamp initial quality onto edges for contraction helpers
        for e in self.spatial.edges:
            seq = self.samples.get(e.way_id) or []
            if seq:
                e.mean_hdop, e.mean_n_los = seq[0][1], seq[0][2]

        self.te, self.timelines = build_te_from_timeseries(
            self.spatial,
            self.samples,
            hdop_step=self.hdop_step,
            n_los_step=self.n_los_step,
            wait_cost=0.0,
            contract_per_layer=True,
        )
        self.contracted = prepare_contracted_graph(
            self._snapshot(0.0), hdop_step=self.hdop_step, n_los_step=self.n_los_step
        )

    def _snapshot(self, t_s: float) -> RoadGraph:
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
                        "color": _hdop_color(e.mean_hdop),
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
        """Always emit fine (uncontracted) street geometry for the map.

        Routing still uses the contracted graph; the map must stay recognizable.
        Optional overlay draws contracted edges as dashed lines.
        """
        feats: list[dict] = []
        if mode == "te":
            # Draw fine spatial streets colored by this layer's quality
            t_s = 0.0
            if self.te.layers:
                layer = max(0, min(layer, len(self.te.layers) - 1))
                t_s = 0.5 * (self.te.layers[layer].t0_s + self.te.layers[layer].t1_s)
            fine = self._snapshot(t_s)
            feats.extend(self._edges_to_features(fine.edges, style="fine"))
            if show_contracted_overlay:
                # contracted travel edges for this TE layer
                te_travel = [
                    e for e in self.te.edges if e.kind == "travel" and e.layer == layer
                ]
                # Build fake RoadEdge-like objects for overlay
                overlay = []
                from gnss_gpu.io.osm_roads import RoadEdge as RE

                for e in te_travel:
                    overlay.append(
                        RE(
                            u=e.u,
                            v=e.v,
                            length_m=e.length_m,
                            way_id=e.way_id,
                            highway=e.highway,
                            name=e.name,
                            geometry=list(e.geometry),
                            mean_hdop=e.mean_hdop,
                            mean_n_los=e.mean_n_los,
                        )
                    )
                feats.extend(self._edges_to_features(overlay, style="contracted"))
            return {"type": "FeatureCollection", "features": feats}

        t_s = 0.0
        if self.te.layers:
            layer = max(0, min(layer, len(self.te.layers) - 1))
            t_s = 0.5 * (self.te.layers[layer].t0_s + self.te.layers[layer].t1_s)
        fine = self._snapshot(t_s)
        feats.extend(self._edges_to_features(fine.edges, style="fine"))
        if show_contracted_overlay:
            cg = prepare_contracted_graph(
                fine, hdop_step=self.hdop_step, n_los_step=self.n_los_step
            )
            feats.extend(self._edges_to_features(cg.graph.edges, style="contracted"))
        return {"type": "FeatureCollection", "features": feats}

    def meta(self) -> dict:
        return {
            "source": self.source,
            "n_spatial_nodes": len(self.spatial.nodes),
            "n_spatial_edges_fine": len(self.spatial.edges),
            "n_contracted_edges_t0": len(self.contracted.graph.edges),
            "n_te_nodes": len(self.te.nodes),
            "n_te_edges": len(self.te.edges),
            "layers": [
                {"index": ly.index, "t0_s": ly.t0_s, "t1_s": ly.t1_s} for ly in self.te.layers
            ],
            "note": (
                "Map always shows fine (uncontracted) street geometry. "
                "Routing runs on the contracted graph; the path is expanded back "
                "via the member map so the yellow route follows recognizable streets. "
                "Optional overlay draws contracted edges dashed."
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
</style>
</head>
<body>
<div id="wrap">
  <aside>
    <h1>GNSS route lab</h1>
    <p class="note">Map = fine streets (always). Routing = contracted graph, then path expanded back. Click A then B.</p>
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
let markers = [];
let origin = null, dest = null;

async function loadMeta(){
  const m = await (await fetch('/api/meta')).json();
  document.getElementById('layer').max = Math.max(0, (m.layers||[]).length-1);
  document.getElementById('stats').textContent = JSON.stringify(m, null, 2);
}

async function loadGraph(){
  const mode = document.getElementById('mode').value;
  const layer = parseInt(document.getElementById('layer').value||'0',10);
  const overlay = document.getElementById('overlay').checked;
  const g = await (await fetch(`/api/graph?mode=${mode}&layer=${layer}&overlay=${overlay}`)).json();
  edgeLayer.clearLayers();
  const layer2 = L.geoJSON(g, {
    style: f => {
      const contracted = f.properties.style === 'contracted';
      return {
        color: contracted ? '#9ec9ff' : (f.properties.color || '#3dbb7a'),
        weight: contracted ? 2 : 5,
        opacity: contracted ? 0.85 : 0.9,
        dashArray: contracted ? '6 6' : null,
      };
    },
    onEachFeature: (f,l) => l.bindPopup(
      `${f.properties.name || '(unnamed)'}<br>highway=${f.properties.highway||''}<br>way ${f.properties.way_id}<br>HDOP ${Number(f.properties.mean_hdop).toFixed(2)}<br>nLOS ${Number(f.properties.mean_n_los).toFixed(1)}<br>${Number(f.properties.length_m).toFixed(1)} m<br style="${f.properties.style}"`
    )
  }).addTo(edgeLayer);
  if (g.features.length) map.fitBounds(layer2.getBounds(), {padding:[30,30]});
}

map.on('click', (e) => {
  if (!origin) {
    origin = e.latlng;
    markers.push(L.circleMarker(origin, {radius:8, color:'#3dbb7a', fillColor:'#3dbb7a', fillOpacity:1}).addTo(map).bindTooltip('A'));
  } else if (!dest) {
    dest = e.latlng;
    markers.push(L.circleMarker(dest, {radius:8, color:'#e85d4c', fillColor:'#e85d4c', fillOpacity:1}).addTo(map).bindTooltip('B'));
  }
});

document.getElementById('btnClear').onclick = () => {
  origin = dest = null;
  markers.forEach(m => map.removeLayer(m)); markers = [];
  pathLayer.clearLayers();
};

document.getElementById('btnReload').onclick = () => loadGraph();
document.getElementById('mode').onchange = () => loadGraph();
document.getElementById('layer').onchange = () => loadGraph();
document.getElementById('overlay').onchange = () => loadGraph();

document.getElementById('btnRoute').onclick = async () => {
  if (!origin || !dest) { alert('Click origin and destination on the map'); return; }
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
  const res = await (await fetch('/api/route', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)})).json();
  pathLayer.clearLayers();
  if (!res.ok) {
    document.getElementById('stats').textContent = JSON.stringify(res, null, 2);
    alert(res.error || 'no path');
    return;
  }
  L.geoJSON(res.path, { style: { color:'#f5d76e', weight:7, opacity:0.95 } }).addTo(pathLayer);
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
            sa = snap_nearest_node(STATE.spatial, lat_a, lon_a)
            sb = snap_nearest_node(STATE.spatial, lat_b, lon_b)
            sl = None if start_layer < 0 else start_layer
            route = dijkstra_te_route(STATE.te, sa, sb, params, start_layer=sl)
            if route is None:
                self._json(200, {"ok": False, "error": "no TE path"})
                return
            path = te_path_to_geojson(STATE.te, route, properties={"kind": "te"})
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
                    1 for i in route.edge_indices if STATE.te.edges[i].kind == "travel"
                ),
                "n_travel_fine": sum(
                    len(STATE.te.edges[i].fine_edge_indices)
                    for i in route.edge_indices
                    if STATE.te.edges[i].kind == "travel"
                ),
                "start_spatial": sa,
                "goal_spatial": sb,
            }
            self._json(200, {"ok": True, "path": path, "summary": summary})
            return

        t_s = 0.0
        if STATE.te.layers:
            layer = max(0, min(layer, len(STATE.te.layers) - 1))
            t_s = 0.5 * (STATE.te.layers[layer].t0_s + STATE.te.layers[layer].t1_s)
        fine = STATE._snapshot(t_s)
        cg = prepare_contracted_graph(
            fine, hdop_step=STATE.hdop_step, n_los_step=STATE.n_los_step
        )
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
    print(f"nodes={len(STATE.spatial.nodes)} edges={len(STATE.spatial.edges)} layers={len(STATE.te.layers)}", flush=True)

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"GNSS route GUI at http://{args.host}:{args.port}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nbye", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
