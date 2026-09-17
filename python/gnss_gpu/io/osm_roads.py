"""OSM roads fetch, sampling, and directed navigable graph."""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

try:
    import requests
except ImportError:  # pragma: no cover - optional until Overpass fetch
    requests = None  # type: ignore[assignment]


OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OVERPASS_ENDPOINTS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
)


@dataclass(frozen=True)
class BBox:
    south: float
    west: float
    north: float
    east: float

    def to_overpass(self) -> str:
        return f"{self.south:.7f},{self.west:.7f},{self.north:.7f},{self.east:.7f}"


@dataclass
class RoadNode:
    node_id: int
    lat_deg: float
    lon_deg: float


@dataclass
class RoadEdge:
    u: int
    v: int
    length_m: float
    way_id: int
    highway: str = ""
    name: str = ""
    geometry: list[tuple[float, float]] = field(default_factory=list)
    mean_hdop: float = float("nan")
    mean_n_los: float = float("nan")


@dataclass
class RoadGraph:
    nodes: dict[int, RoadNode]
    edges: list[RoadEdge]

    def adjacency(self) -> dict[int, list[tuple[int, int]]]:
        """Map u -> list of (edge_index, v)."""
        adj: dict[int, list[tuple[int, int]]] = {nid: [] for nid in self.nodes}
        for i, e in enumerate(self.edges):
            adj.setdefault(e.u, []).append((i, e.v))
        return adj


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp * 0.5) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl * 0.5) ** 2
    return 2.0 * r * math.asin(min(1.0, math.sqrt(a)))


def _interpolate_latlon(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return a + (b - a) * float(t)


def build_roads_overpass_query(
    bbox: BBox,
    *,
    include_pedestrian: bool = False,
) -> str:
    """Build Overpass query for roads in bbox."""
    bb = bbox.to_overpass()
    if include_pedestrian:
        filt = 'way["highway"]'
    else:
        filt = (
            'way["highway"]'
            '["highway"!="footway"]'
            '["highway"!="path"]'
            '["highway"!="steps"]'
            '["highway"!="cycleway"]'
            '["highway"!="pedestrian"]'
            '["highway"!="corridor"]'
            '["highway"!="track"]'
        )
    return f"[out:json][timeout:60];({filt}({bb}););out body geom;"


def fetch_roads_overpass(
    bbox: BBox,
    *,
    include_pedestrian: bool = False,
    timeout_s: int = 120,
    user_agent: str = "gnss_gpu_osm_roads/1.0",
    endpoints: tuple[str, ...] = OVERPASS_ENDPOINTS,
    max_attempts_per_endpoint: int = 5,
) -> list[dict]:
    """Fetch road ways from Overpass for a bbox."""
    if requests is None:
        raise ImportError("requests is required for Overpass fetch (pip install requests)")
    query = build_roads_overpass_query(bbox, include_pedestrian=include_pedestrian)
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "User-Agent": user_agent,
        "Accept": "application/json,text/plain,*/*",
    }
    payload = None
    last_error: Exception | None = None
    for endpoint in endpoints:
        for attempt in range(max(1, int(max_attempts_per_endpoint))):
            try:
                resp = requests.post(
                    endpoint,
                    data=query,
                    headers=headers,
                    timeout=timeout_s,
                )
                if resp.status_code == 429:
                    wait_s = min(120.0, 2.0**attempt + random.uniform(0.0, 1.0))
                    time.sleep(wait_s)
                    continue
                resp.raise_for_status()
                payload = resp.json()
                break
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                wait_s = min(60.0, 2.0**attempt + random.uniform(0.0, 1.0))
                time.sleep(wait_s)
        if payload is not None:
            break
    if payload is None:
        if last_error is not None:
            raise last_error
        raise RuntimeError("Overpass road fetch failed without a specific exception.")

    dedup: dict[int, dict] = {}
    for el in payload.get("elements", []):
        if el.get("type") != "way":
            continue
        eid = el.get("id")
        if isinstance(eid, int):
            dedup[eid] = el
    return list(dedup.values())


def way_geometry_to_line_latlon(geom: list[dict]) -> np.ndarray | None:
    pts = []
    for g in geom:
        try:
            pts.append([float(g["lat"]), float(g["lon"])])
        except (TypeError, ValueError, KeyError):
            return None
    arr = np.asarray(pts, dtype=np.float64)
    if arr.shape[0] < 2:
        return None
    return arr


def sample_line_every_meters(line_latlon: np.ndarray, step_m: float) -> np.ndarray:
    """Sample points along polyline approximately every step_m."""
    if line_latlon.ndim != 2 or line_latlon.shape[1] != 2:
        raise ValueError("line_latlon must have shape [N,2]")
    if line_latlon.shape[0] < 2:
        return line_latlon.copy()
    step = max(1.0, float(step_m))
    out = [line_latlon[0]]
    carry = 0.0
    for i in range(line_latlon.shape[0] - 1):
        a = line_latlon[i]
        b = line_latlon[i + 1]
        seg_m = haversine_m(float(a[0]), float(a[1]), float(b[0]), float(b[1]))
        if seg_m <= 0.0:
            continue
        dist = step - carry
        while dist < seg_m:
            t = dist / seg_m
            out.append(_interpolate_latlon(a, b, t))
            dist += step
        carry = seg_m - (dist - step)
        if carry >= step:
            carry = 0.0
    if np.linalg.norm(out[-1] - line_latlon[-1]) > 1e-12:
        out.append(line_latlon[-1])
    return np.asarray(out, dtype=np.float64)


def sample_road_points(
    roads: Iterable[dict],
    *,
    step_m: float = 25.0,
    dedup_decimals: int = 6,
) -> list[dict]:
    """Return sampled points list from road way elements."""
    points: list[dict] = []
    seen: set[tuple[int, int]] = set()
    for way in roads:
        geom = way.get("geometry")
        if not isinstance(geom, list):
            continue
        line = way_geometry_to_line_latlon(geom)
        if line is None:
            continue
        sampled = sample_line_every_meters(line, step_m=step_m)
        tags = way.get("tags", {}) if isinstance(way.get("tags"), dict) else {}
        for p in sampled:
            k = (
                int(round(float(p[0]) * (10**dedup_decimals))),
                int(round(float(p[1]) * (10**dedup_decimals))),
            )
            if k in seen:
                continue
            seen.add(k)
            points.append(
                {
                    "way_id": int(way.get("id", -1)),
                    "highway": str(tags.get("highway", "")),
                    "name": str(tags.get("name", "")),
                    "lat_deg": float(p[0]),
                    "lon_deg": float(p[1]),
                }
            )
    return points


def split_bbox_into_tiles(bbox: BBox, *, tile_size_m: float) -> list[BBox]:
    """Split bbox into smaller bbox tiles with approx tile_size_m dimensions."""
    if tile_size_m <= 0:
        return [bbox]
    lat_c = 0.5 * (bbox.south + bbox.north)
    dlat = float(tile_size_m) / 111_320.0
    dlon = float(tile_size_m) / (111_320.0 * max(1e-6, math.cos(math.radians(lat_c))))
    lats = np.arange(bbox.south, bbox.north + dlat, dlat)
    lons = np.arange(bbox.west, bbox.east + dlon, dlon)
    tiles: list[BBox] = []
    for i in range(len(lats) - 1):
        for j in range(len(lons) - 1):
            tiles.append(
                BBox(
                    south=float(lats[i]),
                    west=float(lons[j]),
                    north=float(min(lats[i + 1], bbox.north)),
                    east=float(min(lons[j + 1], bbox.east)),
                )
            )
    return tiles


def _coord_key(lat: float, lon: float, decimals: int) -> tuple[int, int]:
    scale = 10**decimals
    return int(round(lat * scale)), int(round(lon * scale))


def oneway_direction(tags: dict) -> str:
    """Return 'forward', 'backward', or 'both' from OSM tags."""
    oneway = str(tags.get("oneway", "")).strip().lower()
    junction = str(tags.get("junction", "")).strip().lower()
    highway = str(tags.get("highway", "")).strip().lower()
    if oneway in ("yes", "true", "1"):
        return "forward"
    if oneway in ("-1", "reverse"):
        return "backward"
    if oneway in ("no", "false", "0"):
        return "both"
    if junction == "roundabout" or highway in ("motorway", "motorway_link"):
        return "forward"
    return "both"


def _polyline_length_m(line: np.ndarray) -> float:
    total = 0.0
    for i in range(line.shape[0] - 1):
        total += haversine_m(
            float(line[i, 0]),
            float(line[i, 1]),
            float(line[i + 1, 0]),
            float(line[i + 1, 1]),
        )
    return total


def build_directed_road_graph(
    roads: Iterable[dict],
    *,
    dedup_decimals: int = 6,
) -> RoadGraph:
    """Build a directed road graph honouring oneway / roundabout tags.

    Nodes are merged by rounded lat/lon so way endpoints that meet share a node.
    Each consecutive geometry segment becomes one directed edge (or two if both-way).
    """
    key_to_id: dict[tuple[int, int], int] = {}
    nodes: dict[int, RoadNode] = {}
    edges: list[RoadEdge] = []

    def _node_for(lat: float, lon: float) -> int:
        k = _coord_key(lat, lon, dedup_decimals)
        nid = key_to_id.get(k)
        if nid is not None:
            return nid
        nid = len(nodes)
        key_to_id[k] = nid
        nodes[nid] = RoadNode(node_id=nid, lat_deg=float(lat), lon_deg=float(lon))
        return nid

    for way in roads:
        geom = way.get("geometry")
        if not isinstance(geom, list):
            continue
        line = way_geometry_to_line_latlon(geom)
        if line is None:
            continue
        tags = way.get("tags", {}) if isinstance(way.get("tags"), dict) else {}
        way_id = int(way.get("id", -1))
        highway = str(tags.get("highway", ""))
        name = str(tags.get("name", ""))
        direction = oneway_direction(tags)

        node_ids = [_node_for(float(line[i, 0]), float(line[i, 1])) for i in range(line.shape[0])]
        for i in range(len(node_ids) - 1):
            u = node_ids[i]
            v = node_ids[i + 1]
            if u == v:
                continue
            seg = line[i : i + 2]
            length_m = _polyline_length_m(seg)
            if length_m <= 0.0:
                continue
            geom_fwd = [(float(seg[0, 0]), float(seg[0, 1])), (float(seg[1, 0]), float(seg[1, 1]))]
            geom_rev = list(reversed(geom_fwd))
            if direction in ("forward", "both"):
                edges.append(
                    RoadEdge(
                        u=u,
                        v=v,
                        length_m=length_m,
                        way_id=way_id,
                        highway=highway,
                        name=name,
                        geometry=geom_fwd,
                    )
                )
            if direction in ("backward", "both"):
                edges.append(
                    RoadEdge(
                        u=v,
                        v=u,
                        length_m=length_m,
                        way_id=way_id,
                        highway=highway,
                        name=name,
                        geometry=geom_rev,
                    )
                )

    return RoadGraph(nodes=nodes, edges=edges)
