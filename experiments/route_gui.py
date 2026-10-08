#!/usr/bin/env python3
"""Minimal Leaflet GUI for GNSS-aware spatial routing (time-extended deferred).

Default: cached OSM Genova car ways. Map paints from pregenerated GeoJSON
immediately; the routing graph loads in the background.

    PYTHONPATH=python python experiments/route_gui.py --port 8765
    PYTHONPATH=python python experiments/route_gui.py --demo overpass
    PYTHONPATH=python python experiments/route_gui.py --pregenerate-only
    PYTHONPATH=python python experiments/route_gui.py --demo synthetic

Open http://127.0.0.1:8765 — click origin, click destination, Route.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_ROOT / "python"))

from gnss_gpu.io.osm_cache import (  # noqa: E402
    default_cache_dir,
    display_cache_path,
    fetch_roads_cached,
    filter_car_ways,
    read_display_cache_bytes,
    write_display_cache,
)
from gnss_gpu.io.road_quality_pack import read_road_quality_pack  # noqa: E402
from gnss_gpu.io.osm_roads import (  # noqa: E402
    BBox,
    RoadEdge,
    RoadGraph,
    build_directed_road_graph,
)
from gnss_gpu.routing import (  # noqa: E402
    CostParams,
    path_to_geojson,
    prepare_contracted_graph,
    route_contracted_latlon,
)
from gnss_gpu.routing_graph import (  # noqa: E402
    make_demo_spatial_graph,
    quality_at,
    synthesize_quality_timeseries,
)

# Genova urban (car-only), loaded via disk-cached Overpass tiles
DEFAULT_BBOX = BBox(south=44.3850, west=8.8800, north=44.4450, east=8.9800)
FIXTURE_ROADS = (
    Path(__file__).resolve().parents[1] / "python" / "gnss_gpu" / "fixtures" / "genova_centro_roads.json"
)


DISPLAY_ARTERIAL = frozenset(
    {
        "motorway",
        "trunk",
        "primary",
        "secondary",
        "tertiary",
        "motorway_link",
        "trunk_link",
        "primary_link",
        "secondary_link",
        "tertiary_link",
    }
)


def _simplify_lonlat(
    coords: list[list[float]],
    *,
    min_step_m: float = 20.0,
) -> list[list[float]]:
    """Drop nearly-collinear display points (routing still uses full geometry)."""
    if len(coords) <= 2:
        return [[round(coords[0][0], 5), round(coords[0][1], 5)],
                [round(coords[-1][0], 5), round(coords[-1][1], 5)]] if len(coords) >= 2 else coords
    kept = [[round(coords[0][0], 5), round(coords[0][1], 5)]]
    last_lat, last_lon = coords[0][1], coords[0][0]
    for lon, lat in coords[1:-1]:
        dy = (lat - last_lat) * 111_320.0
        dx = (lon - last_lon) * 82_000.0
        if (dx * dx + dy * dy) >= (min_step_m * min_step_m):
            kept.append([round(lon, 5), round(lat, 5)])
            last_lat, last_lon = lat, lon
    end = coords[-1]
    kept.append([round(end[0], 5), round(end[1], 5)])
    return kept if len(kept) >= 2 else [kept[0], [round(end[0], 5), round(end[1], 5)]]


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


def _filter_car_ways(roads: list[dict]) -> list[dict]:
    return filter_car_ways(roads)


def _default_quality_pack_path() -> Path:
    return default_cache_dir() / "genova_quality_24h.rqz"


def _load_quality_samples(graph: RoadGraph, roads: list[dict]) -> tuple[dict, str]:
    """Prefer compact ``.rqz`` pack when present; else synthetic short timeline."""
    pack_path = Path(
        os.environ.get("GNSS_GPU_QUALITY_PACK", "").strip() or _default_quality_pack_path()
    )
    if pack_path.is_file():
        pack = read_road_quality_pack(pack_path)
        samples = pack.to_samples_dict()
        # Keep only ways present in this extract.
        way_ids = {int(w["id"]) for w in roads}
        samples = {wid: seq for wid, seq in samples.items() if wid in way_ids}
        if samples:
            return samples, f"quality-pack:{pack_path.name} ({len(samples)} ways, {pack.horizon_s/3600:.0f}h)"
    return synthesize_quality_timeseries(graph), "synthetic-short"


def _graph_from_roads(roads: list[dict], source: str):
    roads = _filter_car_ways(roads)
    graph = build_directed_road_graph(roads)
    if not graph.edges:
        raise RuntimeError("no road edges built")
    samples, qsrc = _load_quality_samples(graph, roads)
    return graph, samples, f"{source} | {qsrc}", roads


def _load_fixture_graph():
    payload = json.loads(FIXTURE_ROADS.read_text(encoding="utf-8"))
    roads = [el for el in payload.get("elements", []) if el.get("type") == "way"]
    roads = _filter_car_ways(roads)
    return _graph_from_roads(roads, f"fixture:{FIXTURE_ROADS.name} ({len(roads)} car ways)")


def _load_overpass_graph(bbox: BBox, *, force_refresh: bool = False):
    roads, info = fetch_roads_cached(
        bbox,
        car_only=True,
        include_pedestrian=False,
        force_refresh=force_refresh,
        tile_size_m=3000.0,
        timeout_s=45,
        max_attempts_per_endpoint=2,
    )
    hit = "cache-hit" if info.get("cache_hit") else "cache-miss"
    src = (
        f"overpass+{hit}:{bbox.south},{bbox.west},{bbox.north},{bbox.east} "
        f"({len(roads)} car ways)"
    )
    return _graph_from_roads(roads, src)


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
    def __init__(
        self,
        *,
        demo: str = "fixture",
        bbox: BBox = DEFAULT_BBOX,
        eager: bool = False,
    ) -> None:
        self.demo = demo
        self.hdop_step = 0.5
        self.n_los_step = 1.0
        self.source = "loading…"
        self.bbox = bbox
        self.roads: list[dict] = []
        self.spatial = RoadGraph(nodes={}, edges=[])
        self.samples: dict = {}
        self.te = None
        self.timelines = None
        self._te_ready = False
        self.contracted = None  # built lazily on first route
        self._display_bytes: dict[tuple[str, int], bytes] = {}
        self._graph_ready = False
        self._graph_error: str | None = None
        self._lock = threading.RLock()
        self._bg: threading.Thread | None = None

        # Map can paint from disk before the routing graph exists.
        self._hydrate_display_cache(details=("arterial", "full"), layers=(0,))
        has_map = ("arterial", 0) in self._display_bytes

        if demo == "synthetic" or eager or not has_map:
            if not has_map and demo != "synthetic":
                print(
                    "no display cache yet — building graph+map "
                    "(next start will be instant; or run --pregenerate-only)",
                    flush=True,
                )
            self._load_graph_blocking()
            if self.roads and demo != "synthetic":
                self.pregenerate_display(details=("arterial", "full"), layers=(0,))
        else:
            print(
                "map ready from display cache; routing graph loads in background…",
                flush=True,
            )
            self._bg = threading.Thread(
                target=self._load_graph_blocking, name="graph-loader", daemon=True
            )
            self._bg.start()

    def _hydrate_display_cache(
        self,
        *,
        details: tuple[str, ...],
        layers: tuple[int, ...],
    ) -> None:
        for detail in details:
            for layer in layers:
                key = (detail, int(layer))
                path = self._display_path(detail, layer)
                raw = read_display_cache_bytes(path)
                if raw is None:
                    continue
                self._display_bytes[key] = raw
                print(
                    f"display cache hit {detail} L{layer}: "
                    f"{path.name} ({len(raw)/1e6:.2f} MB)",
                    flush=True,
                )

    def _load_graph_blocking(self) -> None:
        demo = self.demo
        bbox = self.bbox
        t0 = time.perf_counter()
        try:
            if demo == "synthetic":
                spatial, samples = make_demo_spatial_graph()
                source = "synthetic rectangular block (unit-test only)"
                roads: list[dict] = []
            elif demo == "overpass":
                try:
                    spatial, samples, source, roads = _load_overpass_graph(bbox)
                except Exception as exc:  # noqa: BLE001
                    print(f"WARNING: Overpass failed ({exc}); using offline fixture", flush=True)
                    spatial, samples, source, roads = _load_fixture_graph()
                    source += f" [overpass fallback: {exc}]"
            else:
                spatial, samples, source, roads = _load_fixture_graph()

            for e in spatial.edges:
                seq = samples.get(e.way_id) or []
                if seq:
                    e.mean_hdop, e.mean_n_los = seq[0][1], seq[0][2]

            with self._lock:
                self.spatial = spatial
                self.samples = samples
                self.source = source
                self.roads = roads
                self.contracted = None
                self.timelines = None
                self._graph_error = None
                self._graph_ready = True
            print(
                f"routing graph ready: nodes={len(spatial.nodes)} "
                f"edges={len(spatial.edges)} in {time.perf_counter()-t0:.2f}s "
                f"({source})",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self._graph_error = str(exc)
                self._graph_ready = False
                self.source = f"graph load failed: {exc}"
            print(f"ERROR building routing graph: {exc}", flush=True)

    @property
    def graph_ready(self) -> bool:
        return self._graph_ready

    def wait_until_ready(self, timeout_s: float | None = 120.0) -> bool:
        if self._graph_ready:
            return True
        if self._bg is not None:
            self._bg.join(timeout=timeout_s)
        return self._graph_ready

    def _display_path(self, detail: str, layer: int) -> Path:
        return display_cache_path(
            self.bbox,
            detail=detail,
            layer=layer,
            cache_dir=default_cache_dir(),
        )

    def pregenerate_display(
        self,
        *,
        details: tuple[str, ...] = ("arterial", "full"),
        layers: tuple[int, ...] = (0,),
        force: bool = False,
    ) -> None:
        """Build and disk-cache map GeoJSON so /api/graph is a file read."""
        if not self._graph_ready:
            self.wait_until_ready()
        for detail in details:
            for layer in layers:
                key = (detail, int(layer))
                path = self._display_path(detail, layer)
                if not force and path.is_file():
                    raw = read_display_cache_bytes(path)
                    if raw is not None:
                        self._display_bytes[key] = raw
                        print(
                            f"display cache hit {detail} L{layer}: "
                            f"{path.name} ({len(raw)/1e6:.2f} MB)",
                            flush=True,
                        )
                        continue
                print(f"pregenerating display {detail} L{layer}…", flush=True)
                t0 = time.perf_counter()
                geo = self._build_graph_geojson(
                    mode="spatial",
                    layer=layer,
                    show_contracted_overlay=False,
                    detail=detail,
                )
                write_display_cache(path, geo)
                raw = path.read_bytes()
                self._display_bytes[key] = raw
                print(
                    f"  wrote {path.name} features={len(geo['features'])} "
                    f"{len(raw)/1e6:.2f} MB in {time.perf_counter()-t0:.2f}s",
                    flush=True,
                )

    def _ensure_contracted(self, t_s: float = 0.0):
        """Contracted routing graph for a quality snapshot (cached for t≈0)."""
        if not self._graph_ready:
            raise RuntimeError("routing graph still loading")
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

    def _sample_time_range(self) -> tuple[float, float]:
        """Min/max sample times available for the time scrubber."""
        t_max = 0.0
        for seq in self.samples.values():
            if seq:
                t_max = max(t_max, float(seq[-1][0]))
        return 0.0, t_max

    def graph_geojson(
        self,
        *,
        mode: str,
        layer: int,
        show_contracted_overlay: bool = False,
        detail: str = "arterial",
        t_s: float | None = None,
    ) -> dict:
        """Return map GeoJSON; prefer pregenerated disk/memory cache when possible."""
        use_cache = (
            not show_contracted_overlay
            and (t_s is None or float(t_s) <= 1e-9)
        )
        if use_cache:
            key = (detail, int(layer))
            raw = self._display_bytes.get(key)
            if raw is None:
                path = self._display_path(detail, layer)
                raw = read_display_cache_bytes(path)
                if raw is not None:
                    self._display_bytes[key] = raw
            if raw is not None:
                return json.loads(raw.decode("utf-8"))
            if not self._graph_ready:
                return {"type": "FeatureCollection", "features": []}
            return self._build_graph_geojson(
                mode=mode,
                layer=layer,
                show_contracted_overlay=False,
                detail=detail,
                t_s=0.0,
            )

        if t_s is not None and not show_contracted_overlay:
            if not self._graph_ready:
                return {"type": "FeatureCollection", "features": []}
            return self._build_graph_geojson(
                mode=mode,
                layer=layer,
                show_contracted_overlay=False,
                detail=detail,
                t_s=float(t_s),
            )

        # Overlay on: reuse cached/base roads, then append a capped contracted set.
        base = self.graph_geojson(
            mode=mode,
            layer=layer,
            show_contracted_overlay=False,
            detail=detail,
            t_s=t_s,
        )
        if not self._graph_ready:
            return base
        # TE disabled in GUI — contracted overlay uses the t=0 snapshot.
        cg = self._ensure_contracted(0.0)
        contracted_edges = sorted(
            cg.graph.edges, key=lambda e: float(e.length_m), reverse=True
        )[:2500]
        feats = list(base.get("features") or [])
        feats.extend(self._edges_to_features(contracted_edges, style="contracted"))
        return {"type": "FeatureCollection", "features": feats}

    def graph_geojson_bytes(
        self,
        *,
        layer: int,
        detail: str = "arterial",
        show_contracted_overlay: bool = False,
        t_s: float | None = None,
    ) -> bytes | None:
        """Raw cached JSON bytes for fast HTTP responses (None if unavailable)."""
        if show_contracted_overlay:
            return None
        if t_s is not None and float(t_s) > 1e-9:
            return None
        key = (detail, int(layer))
        raw = self._display_bytes.get(key)
        if raw is not None:
            return raw
        path = self._display_path(detail, layer)
        raw = read_display_cache_bytes(path)
        if raw is not None:
            self._display_bytes[key] = raw
        return raw

    def _build_graph_geojson(
        self,
        *,
        mode: str,
        layer: int,
        show_contracted_overlay: bool = False,
        detail: str = "arterial",
        t_s: float | None = None,
    ) -> dict:
        """Map overlay GeoJSON. Routing still uses the full contracted graph.

        ``detail``:
          - ``arterial`` (default): major roads only (~fast Leaflet draw)
          - ``full``: all car ways (slow for city-scale)
        """
        if t_s is None:
            t_s = self._layer_time(layer)

        feats: list[dict] = []
        if self.roads:
            for way in self.roads:
                tags = way.get("tags") or {}
                hw = str(tags.get("highway", "")).strip().lower()
                if detail != "full" and hw not in DISPLAY_ARTERIAL:
                    continue
                geom = way.get("geometry") or []
                if len(geom) < 2:
                    continue
                wid = int(way.get("id", -1))
                hdop, n_los = _way_quality(self.samples, wid, float(t_s))
                raw_coords = [[float(p["lon"]), float(p["lat"])] for p in geom]
                feats.append(
                    {
                        "type": "Feature",
                        "properties": {
                            "way_id": wid,
                            "mean_hdop": hdop,
                            "mean_n_los": n_los,
                            "color": _quality_color(hdop, n_los),
                            "style": "fine",
                            "t_s": float(t_s),
                        },
                        "geometry": {
                            "type": "LineString",
                            "coordinates": _simplify_lonlat(raw_coords),
                        },
                    }
                )
        else:
            fine = self._snapshot(float(t_s))
            feats.extend(self._edges_to_features(fine.edges, style="fine"))

        if show_contracted_overlay:
            fine = self._snapshot(float(t_s))
            cg = (
                self._ensure_contracted(float(t_s))
                if float(t_s) <= 1e-9
                else prepare_contracted_graph(
                    fine, hdop_step=self.hdop_step, n_los_step=self.n_los_step
                )
            )
            # Cap overlay size — full contracted city graph freezes Leaflet and
            # made unchecking the box look broken while a huge draw was in flight.
            contracted_edges = sorted(
                cg.graph.edges, key=lambda e: float(e.length_m), reverse=True
            )[:2500]
            feats.extend(self._edges_to_features(contracted_edges, style="contracted"))
        return {"type": "FeatureCollection", "features": feats}

    def meta(self) -> dict:
        # TE disabled in GUI: do not build timelines on every page load.
        ready = self._graph_ready
        t_min, t_max = self._sample_time_range() if ready else (0.0, 0.0)
        return {
            "ready": ready,
            "source": self.source,
            "n_spatial_nodes": len(self.spatial.nodes) if ready else 0,
            "n_spatial_edges_fine": len(self.spatial.edges) if ready else 0,
            "n_contracted_edges_t0": (
                len(self.contracted.graph.edges)
                if ready and self.contracted is not None
                else None
            ),
            "layers": [{"index": 0, "t0_s": 0.0, "t1_s": 1.0}],
            "t_min_s": t_min,
            "t_max_s": t_max,
            "display_cached": sorted(f"{d}/L{layer}" for d, layer in self._display_bytes),
            "error": self._graph_error,
            "note": (
                "Map paints from pregenerated GeoJSON immediately. "
                "Optional time scrubber recolors roads from the GNSS timeline. "
                "Path is black."
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
  #timePanel { display:none; margin-top:8px; padding:10px; background:#152028; border-radius:6px; }
  #timePanel.active { display:block; }
  #timeSlider { width:100%; padding:0; accent-color:var(--accent); }
  #timeLabel { color:var(--ink); font-variant-numeric: tabular-nums; }
  button.active-toggle { background:#2a6b4f; color:#e8eef2; }
</style>
</head>
<body>
<div id="wrap">
  <aside>
    <h1>GNSS route lab</h1>
    <p class="note">Click A, then B, then Route. Path in black. (Time-extended routing is disabled for now.)</p>
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
    <label><input id="overlay" type="checkbox"/> show contracted overlay (dashed)</label>
    <label><input id="fullRoads" type="checkbox"/> show all streets (slow)</label>
    <button id="btnTime" class="secondary" disabled>Enable time scrubber</button>
    <div id="timePanel">
      <label>GNSS time <span id="timeLabel">t = 0 s</span></label>
      <input id="timeSlider" type="range" min="0" max="200" step="1" value="0"/>
      <p class="note" style="margin:6px 0 0">Move to recolor roads from the quality timeline (north degrades after ~100 s).</p>
    </div>
    <label>Quality layer (routing costs)</label>
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
        <label>algorithm</label>
        <select id="algo"><option value="astar">A*</option><option value="dijkstra">Dijkstra</option></select>
      </div>
      <div></div>
    </div>
    <button id="btnRoute">Route A → B</button>
    <button id="btnClear" class="secondary">Clear markers</button>
    <button id="btnReload" class="secondary">Reload graph</button>
    <div id="stats">Loading…</div>
  </aside>
  <div id="map"></div>
</div>
<script>
const map = L.map('map', { preferCanvas: true, zoomControl: true });
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  maxZoom: 19, attribution: '&copy; OpenStreetMap'
}).addTo(map);
let edgeLayer = L.layerGroup().addTo(map);
let pathLayer = L.layerGroup().addTo(map);
let markerLayer = L.layerGroup().addTo(map);
let markers = [];
let origin = null, dest = null;
let didFit = false;
let graphAbort = null;
let graphReqId = 0;
let timeEnabled = false;
let timeDebounce = null;
let tMinS = 0;
let tMaxS = 200;

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

function currentTimeS(){
  if (!timeEnabled) return null;
  return parseFloat(document.getElementById('timeSlider').value || '0');
}

function updateTimeLabel(){
  const t = parseFloat(document.getElementById('timeSlider').value || '0');
  document.getElementById('timeLabel').textContent = 't = ' + t.toFixed(0) + ' s';
}

function setTimeEnabled(on){
  timeEnabled = !!on;
  const btn = document.getElementById('btnTime');
  const panel = document.getElementById('timePanel');
  btn.textContent = timeEnabled ? 'Disable time scrubber' : 'Enable time scrubber';
  btn.classList.toggle('active-toggle', timeEnabled);
  panel.classList.toggle('active', timeEnabled);
  if (!timeEnabled) {
    document.getElementById('timeSlider').value = String(tMinS);
    updateTimeLabel();
  }
  loadGraph();
}

async function loadMeta(){
  const m = await (await fetch('/api/meta')).json();
  document.getElementById('layer').max = Math.max(0, (m.layers||[]).length-1);
  tMinS = Number.isFinite(m.t_min_s) ? m.t_min_s : 0;
  tMaxS = Number.isFinite(m.t_max_s) && m.t_max_s > tMinS ? m.t_max_s : 200;
  const slider = document.getElementById('timeSlider');
  slider.min = String(tMinS);
  slider.max = String(tMaxS);
  if (parseFloat(slider.value) < tMinS || parseFloat(slider.value) > tMaxS) {
    slider.value = String(tMinS);
  }
  updateTimeLabel();
  const status = m.ready
    ? `Routing ready\\n${m.source}\\nnodes=${m.n_spatial_nodes} edges=${m.n_spatial_edges_fine}\\nGNSS timeline: ${tMinS.toFixed(0)}–${tMaxS.toFixed(0)} s`
    : `Map ready — routing graph still loading…\\n${m.source || ''}`;
  document.getElementById('stats').textContent = status;
  document.getElementById('btnRoute').disabled = !m.ready;
  document.getElementById('btnTime').disabled = !m.ready;
  setHint();
  return m;
}

async function loadGraph(){
  const mode = 'spatial';
  const layer = parseInt(document.getElementById('layer').value||'0',10);
  const overlay = document.getElementById('overlay').checked;
  const detail = document.getElementById('fullRoads').checked ? 'full' : 'arterial';
  const tS = currentTimeS();
  // Cancel in-flight overlay draws so unchecking can take effect immediately.
  if (graphAbort) {
    try { graphAbort.abort(); } catch (_) {}
  }
  graphAbort = (typeof AbortController !== 'undefined') ? new AbortController() : null;
  const reqId = ++graphReqId;
  edgeLayer.clearLayers();
  const tTag = (tS === null) ? '' : (' @ t=' + tS.toFixed(0) + 's');
  document.getElementById('stats').textContent = 'Loading road layer (' + detail + (overlay ? '+overlay' : '') + tTag + ')…';
  const t0 = performance.now();
  let g;
  try {
    let url = `/api/graph?mode=${mode}&layer=${layer}&overlay=${overlay}&detail=${detail}`;
    if (tS !== null) url += `&t_s=${encodeURIComponent(tS)}`;
    const resp = await fetch(url, graphAbort ? { signal: graphAbort.signal } : undefined);
    g = await resp.json();
  } catch (err) {
    if (err && err.name === 'AbortError') return;
    document.getElementById('stats').textContent = 'Graph load failed: ' + err;
    return;
  }
  if (reqId !== graphReqId) return; // stale response
  edgeLayer.clearLayers();
  const tFetch = performance.now() - t0;
  const t1 = performance.now();
  const layer2 = L.geoJSON(g, {
    interactive: false,
    renderer: L.canvas({ padding: 0.5 }),
    style: f => {
      const contracted = f.properties.style === 'contracted';
      return {
        color: contracted ? '#9ec9ff' : (f.properties.color || '#3dbb7a'),
        weight: contracted ? 2 : 3,
        opacity: contracted ? 0.85 : 0.8,
        dashArray: contracted ? '6 6' : null,
      };
    },
  }).addTo(edgeLayer);
  const tDraw = performance.now() - t1;
  if (reqId !== graphReqId) return;
  if (g.features.length && !didFit) {
    map.fitBounds(layer2.getBounds(), {padding:[30,30]});
    didFit = true;
  }
  document.getElementById('stats').textContent =
    `Display: ${g.features.length} ways (${detail}${overlay ? '+overlay' : ''}${tTag})\\n` +
    `fetch ${tFetch.toFixed(0)}ms / draw ${tDraw.toFixed(0)}ms\\n` +
    `Routing still uses the full car network.`;
}

async function pollReady(){
  for (let i = 0; i < 120; i++) {
    const m = await loadMeta();
    if (m.ready) return;
    await new Promise(r => setTimeout(r, 500));
  }
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
document.getElementById('layer').onchange = () => loadGraph();
document.getElementById('overlay').onchange = () => loadGraph();
document.getElementById('fullRoads').onchange = () => loadGraph();
document.getElementById('btnTime').onclick = () => setTimeEnabled(!timeEnabled);
document.getElementById('timeSlider').addEventListener('input', () => {
  updateTimeLabel();
  if (!timeEnabled) return;
  if (timeDebounce) clearTimeout(timeDebounce);
  timeDebounce = setTimeout(() => loadGraph(), 80);
});

document.getElementById('btnRoute').onclick = async () => {
  if (!origin || !dest) { alert('Click origin A, then destination B anywhere on the map'); return; }
  const body = {
    lat_a: origin.lat, lon_a: origin.lng,
    lat_b: dest.lat, lon_b: dest.lng,
    alpha: parseFloat(document.getElementById('alpha').value),
    beta: parseFloat(document.getElementById('beta').value),
    mode: 'spatial',
    layer: parseInt(document.getElementById('layer').value||'0',10),
    algorithm: document.getElementById('algo').value,
  };
  document.getElementById('stats').textContent = 'Routing…';
  let res;
  try {
    const resp = await fetch('/api/route', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
    res = await resp.json();
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
    style: { color:'#111111', weight:8, opacity:1 },
  }).addTo(pathLayer);
  document.getElementById('stats').textContent = JSON.stringify(res.summary, null, 2);
};

loadGraph().then(() => { loadMeta(); pollReady(); });
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
            detail = qs.get("detail", ["arterial"])[0]
            t_raw = qs.get("t_s", [None])[0]
            t_s = float(t_raw) if t_raw is not None and str(t_raw).strip() != "" else None
            if not overlay:
                raw = STATE.graph_geojson_bytes(
                    layer=layer,
                    detail=detail,
                    show_contracted_overlay=False,
                    t_s=t_s,
                )
                if raw is not None:
                    self._send(200, raw, "application/json")
                    return
            self._json(
                200,
                STATE.graph_geojson(
                    mode=mode,
                    layer=layer,
                    show_contracted_overlay=overlay,
                    detail=detail,
                    t_s=t_s,
                ),
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
        algorithm = str(req.get("algorithm", "dijkstra"))

        if mode == "te":
            self._json(
                200,
                {
                    "ok": False,
                    "error": "time-extended routing is disabled for now (spatial only)",
                },
            )
            return

        if not STATE.graph_ready:
            # Brief wait so a just-opened tab can route once bg load finishes.
            STATE.wait_until_ready(timeout_s=2.0)
        if not STATE.graph_ready:
            self._json(
                503,
                {
                    "ok": False,
                    "error": "routing graph still loading — map is ready, retry in a moment",
                    "source": STATE.source,
                },
            )
            return

        # TE disabled in GUI: use t=0 so first route does not build timelines.
        t_s = 0.0
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
        default="overpass",
        help="overpass=cached OSM Genova (default); fixture=bundled snippet; synthetic=tiny rectangle",
    )
    p.add_argument(
        "--bbox",
        default=None,
        help="south,west,north,east (default: Genova urban core)",
    )
    p.add_argument(
        "--force-refresh",
        action="store_true",
        help="Ignore OSM disk cache and re-fetch from Overpass",
    )
    p.add_argument(
        "--pregenerate-only",
        action="store_true",
        help="Build OSM+display caches and exit (no HTTP server)",
    )
    args = p.parse_args(argv)
    bbox = DEFAULT_BBOX
    if args.bbox:
        parts = [float(x) for x in args.bbox.replace(" ", "").split(",")]
        bbox = BBox(south=parts[0], west=parts[1], north=parts[2], east=parts[3])

    print(f"Loading GUI (demo={args.demo})…", flush=True)
    if args.demo == "overpass" and args.force_refresh:
        print("Force-refreshing Overpass cache…", flush=True)
        _load_overpass_graph(bbox, force_refresh=True)
    eager = bool(args.pregenerate_only)
    STATE = DemoState(demo=args.demo, bbox=bbox, eager=eager)
    print(f"source: {STATE.source}", flush=True)
    if STATE.graph_ready:
        print(
            f"nodes={len(STATE.spatial.nodes)} edges={len(STATE.spatial.edges)} "
            f"(contraction lazy; TE disabled; display pregenerated)",
            flush=True,
        )
    else:
        print(
            "HTTP starting now — map from display cache; routing graph still loading…",
            flush=True,
        )
    if args.pregenerate_only:
        STATE.wait_until_ready()
        STATE.pregenerate_display(details=("arterial", "full"), layers=(0,), force=True)
        print("pregenerate done", flush=True)
        return 0

    httpd = _ReusableThreadingHTTPServer((args.host, args.port), Handler)
    print(f"GNSS route GUI at http://{args.host}:{args.port}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nbye", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
