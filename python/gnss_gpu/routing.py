"""GNSS-quality-aware road routing (Dijkstra / A*)."""

from __future__ import annotations

import csv
import heapq
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from gnss_gpu.io.osm_roads import RoadEdge, RoadGraph, haversine_m


@dataclass(frozen=True)
class CostParams:
    alpha: float = 1.0
    beta: float = 0.5
    h0: float = 2.0
    n_star: float = 8.0
    p_max: float = 5.0


@dataclass
class RouteResult:
    node_ids: list[int]
    edge_indices: list[int]
    total_cost: float
    length_m: float
    mean_hdop: float
    mean_n_los: float


def clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def edge_gnss_penalty(edge: RoadEdge, params: CostParams) -> float:
    """Non-negative GNSS penalty factor (unitless)."""
    hdop = float(edge.mean_hdop)
    n_los = float(edge.mean_n_los)
    pen = 0.0
    if math.isfinite(hdop) and params.h0 > 0.0 and params.alpha != 0.0:
        pen += params.alpha * clip(hdop / params.h0, 0.0, params.p_max)
    if math.isfinite(n_los) and params.n_star > 0.0 and params.beta != 0.0:
        pen += params.beta * clip((params.n_star - n_los) / params.n_star, 0.0, 1.0)
    return max(0.0, pen)


def edge_cost(edge: RoadEdge, params: CostParams) -> float:
    """c(e) = L(e) * (1 + GNSS penalty). Always >= length_m >= 0."""
    return float(edge.length_m) * (1.0 + edge_gnss_penalty(edge, params))


def aggregate_quality_onto_edges(
    graph: RoadGraph,
    points: Iterable[dict],
    *,
    hdop_key: str = "mean_hdop",
    n_los_key: str = "mean_n_los",
    way_id_key: str = "way_id",
) -> RoadGraph:
    """Attach mean HDOP / n_LOS per way_id onto matching edges (in-place)."""
    sums: dict[int, list[float]] = {}
    counts: dict[int, list[int]] = {}
    for p in points:
        try:
            wid = int(p[way_id_key])
        except (KeyError, TypeError, ValueError):
            continue
        hdop = p.get(hdop_key, float("nan"))
        n_los = p.get(n_los_key, float("nan"))
        try:
            hdop_f = float(hdop)
        except (TypeError, ValueError):
            hdop_f = float("nan")
        try:
            n_los_f = float(n_los)
        except (TypeError, ValueError):
            n_los_f = float("nan")
        if wid not in sums:
            sums[wid] = [0.0, 0.0]
            counts[wid] = [0, 0]
        if math.isfinite(hdop_f):
            sums[wid][0] += hdop_f
            counts[wid][0] += 1
        if math.isfinite(n_los_f):
            sums[wid][1] += n_los_f
            counts[wid][1] += 1

    for e in graph.edges:
        sc = sums.get(e.way_id)
        cc = counts.get(e.way_id)
        if sc is None or cc is None:
            continue
        if cc[0] > 0:
            e.mean_hdop = sc[0] / cc[0]
        if cc[1] > 0:
            e.mean_n_los = sc[1] / cc[1]
    return graph


def load_quality_points_csv(path: Path | str) -> list[dict]:
    """Load road-point quality CSV (expects way_id + mean_hdop and/or mean_n_los)."""
    rows: list[dict] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def snap_nearest_node(graph: RoadGraph, lat_deg: float, lon_deg: float) -> int:
    best_id = -1
    best_d = float("inf")
    for nid, node in graph.nodes.items():
        d = haversine_m(lat_deg, lon_deg, node.lat_deg, node.lon_deg)
        if d < best_d:
            best_d = d
            best_id = nid
    if best_id < 0:
        raise ValueError("RoadGraph has no nodes")
    return best_id


def _reconstruct(
    came_from: dict[int, tuple[int, int]],
    start: int,
    goal: int,
) -> tuple[list[int], list[int]]:
    """Return (node_ids, edge_indices) from came_from[v] = (u, edge_index)."""
    if goal == start:
        return [start], []
    nodes_rev = [goal]
    edges_rev: list[int] = []
    cur = goal
    while cur != start:
        if cur not in came_from:
            raise RuntimeError("incomplete path reconstruction")
        prev, ei = came_from[cur]
        edges_rev.append(ei)
        nodes_rev.append(prev)
        cur = prev
    nodes_rev.reverse()
    edges_rev.reverse()
    return nodes_rev, edges_rev


def _path_metrics(graph: RoadGraph, edge_indices: list[int], total_cost: float) -> RouteResult:
    length = 0.0
    hdop_w = 0.0
    nlos_w = 0.0
    w_h = 0.0
    w_n = 0.0
    nodes: list[int] = []
    if edge_indices:
        nodes.append(graph.edges[edge_indices[0]].u)
        for ei in edge_indices:
            e = graph.edges[ei]
            nodes.append(e.v)
            length += e.length_m
            if math.isfinite(e.mean_hdop):
                hdop_w += e.mean_hdop * e.length_m
                w_h += e.length_m
            if math.isfinite(e.mean_n_los):
                nlos_w += e.mean_n_los * e.length_m
                w_n += e.length_m
    mean_hdop = hdop_w / w_h if w_h > 0 else float("nan")
    mean_n_los = nlos_w / w_n if w_n > 0 else float("nan")
    return RouteResult(
        node_ids=nodes,
        edge_indices=list(edge_indices),
        total_cost=float(total_cost),
        length_m=float(length),
        mean_hdop=float(mean_hdop),
        mean_n_los=float(mean_n_los),
    )


def dijkstra_route(
    graph: RoadGraph,
    start: int,
    goal: int,
    params: CostParams | None = None,
) -> RouteResult | None:
    """Shortest path with GNSS-weighted edge costs."""
    if start not in graph.nodes or goal not in graph.nodes:
        raise KeyError("start/goal node not in graph")
    params = params or CostParams()
    if start == goal:
        return RouteResult([start], [], 0.0, 0.0, float("nan"), float("nan"))

    adj = graph.adjacency()
    dist = {start: 0.0}
    came_from: dict[int, tuple[int, int]] = {}
    heap: list[tuple[float, int]] = [(0.0, start)]
    visited: set[int] = set()

    while heap:
        d_u, u = heapq.heappop(heap)
        if u in visited:
            continue
        visited.add(u)
        if u == goal:
            nodes, edges = _reconstruct(came_from, start, goal)
            result = _path_metrics(graph, edges, d_u)
            result.node_ids = nodes
            return result
        for ei, v in adj.get(u, []):
            if v in visited:
                continue
            nd = d_u + edge_cost(graph.edges[ei], params)
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                came_from[v] = (u, ei)
                heapq.heappush(heap, (nd, v))
    return None


def astar_route(
    graph: RoadGraph,
    start: int,
    goal: int,
    params: CostParams | None = None,
) -> RouteResult | None:
    """A* with haversine heuristic (admissible: heuristic <= true length <= cost)."""
    if start not in graph.nodes or goal not in graph.nodes:
        raise KeyError("start/goal node not in graph")
    params = params or CostParams()
    if start == goal:
        return RouteResult([start], [], 0.0, 0.0, float("nan"), float("nan"))

    goal_node = graph.nodes[goal]

    def h(nid: int) -> float:
        n = graph.nodes[nid]
        return haversine_m(n.lat_deg, n.lon_deg, goal_node.lat_deg, goal_node.lon_deg)

    adj = graph.adjacency()
    g_score = {start: 0.0}
    came_from: dict[int, tuple[int, int]] = {}
    heap: list[tuple[float, int]] = [(h(start), start)]
    closed: set[int] = set()

    while heap:
        _, u = heapq.heappop(heap)
        if u in closed:
            continue
        closed.add(u)
        if u == goal:
            nodes, edges = _reconstruct(came_from, start, goal)
            result = _path_metrics(graph, edges, g_score[u])
            result.node_ids = nodes
            return result
        for ei, v in adj.get(u, []):
            if v in closed:
                continue
            tentative = g_score[u] + edge_cost(graph.edges[ei], params)
            if tentative < g_score.get(v, float("inf")):
                g_score[v] = tentative
                came_from[v] = (u, ei)
                heapq.heappush(heap, (tentative + h(v), v))
    return None


def route_latlon(
    graph: RoadGraph,
    lat_a: float,
    lon_a: float,
    lat_b: float,
    lon_b: float,
    params: CostParams | None = None,
    *,
    algorithm: str = "astar",
) -> RouteResult | None:
    start = snap_nearest_node(graph, lat_a, lon_a)
    goal = snap_nearest_node(graph, lat_b, lon_b)
    if algorithm == "dijkstra":
        return dijkstra_route(graph, start, goal, params)
    if algorithm == "astar":
        return astar_route(graph, start, goal, params)
    raise ValueError(f"unknown algorithm: {algorithm}")


def path_to_geojson(
    graph: RoadGraph,
    route: RouteResult,
    *,
    properties: dict | None = None,
) -> dict:
    coords: list[list[float]] = []
    for ei in route.edge_indices:
        e = graph.edges[ei]
        for lat, lon in e.geometry:
            pt = [lon, lat]
            if not coords or coords[-1] != pt:
                coords.append(pt)
    if not coords and route.node_ids:
        for nid in route.node_ids:
            n = graph.nodes[nid]
            coords.append([n.lon_deg, n.lat_deg])
    props = {
        "length_m": route.length_m,
        "total_cost": route.total_cost,
        "mean_hdop": route.mean_hdop,
        "mean_n_los": route.mean_n_los,
        "n_edges": len(route.edge_indices),
    }
    if properties:
        props.update(properties)
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": props,
                "geometry": {"type": "LineString", "coordinates": coords},
            }
        ],
    }


def write_geojson(path: Path | str, geojson: dict) -> None:
    Path(path).write_text(json.dumps(geojson, indent=2), encoding="utf-8")


def edges_to_csv(graph: RoadGraph, path: Path | str) -> None:
    fieldnames = [
        "u",
        "v",
        "length_m",
        "way_id",
        "highway",
        "name",
        "mean_hdop",
        "mean_n_los",
        "lat_u",
        "lon_u",
        "lat_v",
        "lon_v",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for e in graph.edges:
            nu = graph.nodes[e.u]
            nv = graph.nodes[e.v]
            w.writerow(
                {
                    "u": e.u,
                    "v": e.v,
                    "length_m": f"{e.length_m:.3f}",
                    "way_id": e.way_id,
                    "highway": e.highway,
                    "name": e.name,
                    "mean_hdop": "" if not math.isfinite(e.mean_hdop) else f"{e.mean_hdop:.4f}",
                    "mean_n_los": "" if not math.isfinite(e.mean_n_los) else f"{e.mean_n_los:.4f}",
                    "lat_u": f"{nu.lat_deg:.7f}",
                    "lon_u": f"{nu.lon_deg:.7f}",
                    "lat_v": f"{nv.lat_deg:.7f}",
                    "lon_v": f"{nv.lon_deg:.7f}",
                }
            )
