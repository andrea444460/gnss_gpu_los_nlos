"""GNSS-quality-aware road routing (Dijkstra / A*)."""

from __future__ import annotations

import csv
import heapq
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from gnss_gpu.io.osm_roads import RoadEdge, RoadGraph, haversine_m
from gnss_gpu.routing_graph import (
    ContractedGraph,
    QualityInterval,
    TimeExtendedEdge,
    TimeExtendedGraph,
    attach_timelines_by_way,
    build_time_extended_graph,
    contract_same_quality_edges,
    expand_contracted_route,
)


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
    layers: list[int] = field(default_factory=list)
    spatial_node_ids: list[int] = field(default_factory=list)
    # After expand: fine (uncontracted) edge indices for map display
    fine_edge_indices: list[int] = field(default_factory=list)
    fine_node_ids: list[int] = field(default_factory=list)
    fine_geometry: list[tuple[float, float]] = field(default_factory=list)


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


def te_edge_cost(edge: TimeExtendedEdge, params: CostParams) -> float:
    """Cost for a time-extended edge (wait uses length_m as wait penalty)."""
    if edge.kind == "wait":
        return max(0.0, float(edge.length_m))
    # reuse RoadEdge penalty fields
    tmp = RoadEdge(
        u=edge.u,
        v=edge.v,
        length_m=edge.length_m,
        way_id=edge.way_id,
        mean_hdop=edge.mean_hdop,
        mean_n_los=edge.mean_n_los,
    )
    return edge_cost(tmp, params)


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


def load_quality_timeseries_csv(path: Path | str) -> dict[int, list[tuple[float, float, float]]]:
    """Load per-way time series: columns way_id,t_s,mean_hdop,mean_n_los."""
    by_way: dict[int, list[tuple[float, float, float]]] = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                wid = int(row["way_id"])
                t_s = float(row.get("t_s", row.get("time_s", 0.0)))
                hdop = float(row["mean_hdop"])
                n_los = float(row["mean_n_los"])
            except (KeyError, TypeError, ValueError):
                continue
            by_way[wid].append((t_s, hdop, n_los))
    for wid in by_way:
        by_way[wid].sort(key=lambda x: x[0])
    return dict(by_way)


def prepare_contracted_graph(
    graph: RoadGraph,
    *,
    hdop_step: float = 0.5,
    n_los_step: float = 1.0,
) -> ContractedGraph:
    """Contract consecutive same-direction same-quality degree-2 chains.

    Returns a :class:`ContractedGraph`: route on ``.graph``, draw ``.fine``,
    expand the path with :func:`expand_route_result`.
    """
    return contract_same_quality_edges(graph, hdop_step=hdop_step, n_los_step=n_los_step)


def expand_route_result(contracted: ContractedGraph, route: RouteResult) -> RouteResult:
    """Fill fine_* fields by unpacking contracted edges via the member map."""
    fine_edges, fine_nodes, geom = expand_contracted_route(contracted, route.edge_indices)
    route.fine_edge_indices = fine_edges
    route.fine_node_ids = fine_nodes
    route.fine_geometry = geom
    return route


def route_contracted_latlon(
    contracted: ContractedGraph,
    lat_a: float,
    lon_a: float,
    lat_b: float,
    lon_b: float,
    params: CostParams | None = None,
    *,
    algorithm: str = "astar",
    snap_k: int = 24,
) -> RouteResult | None:
    """Route on the contracted graph, then expand to fine street geometry."""
    route = route_latlon(
        contracted.graph,
        lat_a,
        lon_a,
        lat_b,
        lon_b,
        params,
        algorithm=algorithm,
        snap_k=snap_k,
    )
    if route is None:
        return None
    if route.fine_geometry and not route.edge_indices:
        # Trivial same-node snap already filled fine_* fields.
        return route
    return expand_route_result(contracted, route)


def build_te_from_timeseries(
    graph: RoadGraph,
    samples_by_way: dict[int, list[tuple[float, float, float]]],
    *,
    hdop_step: float = 0.5,
    n_los_step: float = 1.0,
    wait_cost: float = 0.0,
    contract_per_layer: bool = True,
) -> tuple[TimeExtendedGraph, list[list[QualityInterval]]]:
    timelines = attach_timelines_by_way(
        graph, samples_by_way, hdop_step=hdop_step, n_los_step=n_los_step
    )
    te = build_time_extended_graph(
        graph,
        timelines,
        wait_cost=wait_cost,
        hdop_step=hdop_step,
        n_los_step=n_los_step,
        contract_per_layer=contract_per_layer,
    )
    return te, timelines


def snap_nearest_node(
    graph: RoadGraph,
    lat_deg: float,
    lon_deg: float,
    *,
    candidates: Iterable[int] | None = None,
) -> int:
    """Nearest node; optionally restrict to ``candidates`` (e.g. routable only)."""
    pool = candidates if candidates is not None else graph.nodes.keys()
    best_id = -1
    best_d = float("inf")
    for nid in pool:
        node = graph.nodes[int(nid)]
        d = haversine_m(lat_deg, lon_deg, node.lat_deg, node.lon_deg)
        if d < best_d:
            best_d = d
            best_id = int(nid)
    if best_id < 0:
        raise ValueError("RoadGraph has no candidate nodes")
    return best_id


def routable_nodes(graph: RoadGraph) -> set[int]:
    """Nodes incident to at least one directed edge (safe snap targets)."""
    live: set[int] = set()
    for e in graph.edges:
        live.add(e.u)
        live.add(e.v)
    return live


def nearest_routable_nodes(
    graph: RoadGraph,
    lat_deg: float,
    lon_deg: float,
    *,
    k: int = 12,
    candidates: set[int] | None = None,
) -> list[tuple[float, int]]:
    """k nearest nodes with incident edges, as (distance_m, node_id)."""
    pool = candidates if candidates is not None else routable_nodes(graph)
    scored: list[tuple[float, int]] = []
    for nid in pool:
        n = graph.nodes[nid]
        scored.append((haversine_m(lat_deg, lon_deg, n.lat_deg, n.lon_deg), nid))
    scored.sort(key=lambda t: t[0])
    return scored[: max(1, int(k))]


def weak_components(graph: RoadGraph) -> dict[int, int]:
    """Map node_id -> weakly-connected component id (ignore direction)."""
    und: dict[int, set[int]] = {}
    for e in graph.edges:
        und.setdefault(e.u, set()).add(e.v)
        und.setdefault(e.v, set()).add(e.u)
    comp: dict[int, int] = {}
    cid = 0
    for seed in und:
        if seed in comp:
            continue
        stack = [seed]
        comp[seed] = cid
        while stack:
            u = stack.pop()
            for v in und.get(u, ()):
                if v not in comp:
                    comp[v] = cid
                    stack.append(v)
        cid += 1
    return comp


def route_latlon(
    graph: RoadGraph,
    lat_a: float,
    lon_a: float,
    lat_b: float,
    lon_b: float,
    params: CostParams | None = None,
    *,
    algorithm: str = "astar",
    snap_k: int = 24,
) -> RouteResult | None:
    """Snap A/B to nearby routable nodes (same weak component) then shortest path.

    Snaps stay local to the click — do not jump to a distant \"main\" component,
    which made nearby clicks on side streets look broken.
    """
    params = params or CostParams()
    comps = weak_components(graph)
    if not comps:
        return None

    cands_a = nearest_routable_nodes(graph, lat_a, lon_a, k=snap_k)
    cands_b = nearest_routable_nodes(graph, lat_b, lon_b, k=snap_k)
    if not cands_a or not cands_b:
        return None

    best: RouteResult | None = None
    best_score = float("inf")
    for da, sa in cands_a:
        for db, sb in cands_b:
            if sa == sb:
                continue
            if comps.get(sa) != comps.get(sb):
                continue
            if algorithm == "dijkstra":
                route = dijkstra_route(graph, sa, sb, params)
            elif algorithm == "astar":
                route = astar_route(graph, sa, sb, params)
            else:
                raise ValueError(f"unknown algorithm: {algorithm}")
            if route is None:
                continue
            score = da + db + 0.05 * route.total_cost
            if score < best_score:
                best_score = score
                best = route

    # Very close clicks often share the nearest node — try 1st A × 2nd+ B, etc.
    if best is None and cands_a and cands_b:
        sa = cands_a[0][1]
        for db, sb in cands_b[1:]:
            if comps.get(sa) != comps.get(sb):
                continue
            if algorithm == "dijkstra":
                route = dijkstra_route(graph, sa, sb, params)
            else:
                route = astar_route(graph, sa, sb, params)
            if route is not None:
                return route
        # Still nothing: trivial zero-length path at shared snap (same place).
        if cands_a[0][1] == cands_b[0][1]:
            nid = cands_a[0][1]
            return RouteResult(
                node_ids=[nid],
                edge_indices=[],
                total_cost=0.0,
                length_m=0.0,
                mean_hdop=float("nan"),
                mean_n_los=float("nan"),
                fine_node_ids=[nid],
                fine_edge_indices=[],
                fine_geometry=[(graph.nodes[nid].lat_deg, graph.nodes[nid].lon_deg)],
            )
    return best


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


def dijkstra_te_route(
    te: TimeExtendedGraph,
    start_spatial: int,
    goal_spatial: int,
    params: CostParams | None = None,
    *,
    start_layer: int | None = None,
) -> RouteResult | None:
    """Dijkstra on a time-extended graph.

    Start: (start_spatial, start_layer) or any layer if start_layer is None
    (super-source via zero-cost injection). Goal: any layer at goal_spatial.
    """
    params = params or CostParams()
    starts: list[int] = []
    if start_layer is not None:
        key = (start_spatial, start_layer)
        if key not in te.index:
            return None
        starts = [te.index[key]]
    else:
        starts = [
            te.index[(start_spatial, layer.index)]
            for layer in te.layers
            if (start_spatial, layer.index) in te.index
        ]
    goals = {
        te.index[(goal_spatial, layer.index)]
        for layer in te.layers
        if (goal_spatial, layer.index) in te.index
    }
    if not starts or not goals:
        return None

    adj = te.adjacency()
    dist: dict[int, float] = {s: 0.0 for s in starts}
    came_from: dict[int, tuple[int, int]] = {}
    heap: list[tuple[float, int]] = [(0.0, s) for s in starts]
    heapq.heapify(heap)
    visited: set[int] = set()
    best_goal: int | None = None
    best_cost = float("inf")

    while heap:
        d_u, u = heapq.heappop(heap)
        if u in visited:
            continue
        visited.add(u)
        if u in goals:
            best_goal = u
            best_cost = d_u
            break
        for ei, v in adj.get(u, []):
            if v in visited:
                continue
            nd = d_u + te_edge_cost(te.edges[ei], params)
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                came_from[v] = (u, ei)
                heapq.heappush(heap, (nd, v))

    if best_goal is None:
        return None

    nodes_rev = [best_goal]
    edges_rev: list[int] = []
    cur = best_goal
    while cur not in starts:
        if cur not in came_from:
            break
        prev, ei = came_from[cur]
        edges_rev.append(ei)
        nodes_rev.append(prev)
        cur = prev
    nodes_rev.reverse()
    edges_rev.reverse()

    length = 0.0
    hdop_w = 0.0
    nlos_w = 0.0
    w_h = 0.0
    w_n = 0.0
    layers: list[int] = []
    spatial_ids: list[int] = []
    for nid in nodes_rev:
        n = te.nodes[nid]
        layers.append(n.layer)
        if not spatial_ids or spatial_ids[-1] != n.spatial_id:
            spatial_ids.append(n.spatial_id)
    for ei in edges_rev:
        e = te.edges[ei]
        if e.kind != "travel":
            continue
        length += e.length_m
        if math.isfinite(e.mean_hdop):
            hdop_w += e.mean_hdop * e.length_m
            w_h += e.length_m
        if math.isfinite(e.mean_n_los):
            nlos_w += e.mean_n_los * e.length_m
            w_n += e.length_m

    return RouteResult(
        node_ids=nodes_rev,
        edge_indices=edges_rev,
        total_cost=float(best_cost),
        length_m=float(length),
        mean_hdop=float(hdop_w / w_h) if w_h > 0 else float("nan"),
        mean_n_los=float(nlos_w / w_n) if w_n > 0 else float("nan"),
        layers=layers,
        spatial_node_ids=spatial_ids,
    )


def te_spatial_ids(te: TimeExtendedGraph) -> set[int]:
    """Spatial node ids that appear in the TE index (safe TE snap targets)."""
    return {sid for sid, _layer in te.index.keys()}


def nearest_te_spatial_nodes(
    te: TimeExtendedGraph,
    lat_deg: float,
    lon_deg: float,
    *,
    k: int = 12,
) -> list[tuple[float, int]]:
    scored: list[tuple[float, int]] = []
    for sid in te_spatial_ids(te):
        n = te.spatial.nodes[sid]
        scored.append((haversine_m(lat_deg, lon_deg, n.lat_deg, n.lon_deg), sid))
    scored.sort(key=lambda t: t[0])
    return scored[: max(1, int(k))]


def route_te_latlon(
    te: TimeExtendedGraph,
    lat_a: float,
    lon_a: float,
    lat_b: float,
    lon_b: float,
    params: CostParams | None = None,
    *,
    start_layer: int | None = None,
    snap_k: int = 12,
) -> RouteResult | None:
    """Snap A/B to TE-present spatial nodes (k nearest) then Dijkstra-TE."""
    params = params or CostParams()
    cands_a = nearest_te_spatial_nodes(te, lat_a, lon_a, k=snap_k)
    cands_b = nearest_te_spatial_nodes(te, lat_b, lon_b, k=snap_k)
    if not cands_a or not cands_b:
        return None

    best: RouteResult | None = None
    best_score = float("inf")
    for da, sa in cands_a:
        for db, sb in cands_b:
            if sa == sb:
                continue
            route = dijkstra_te_route(
                te, sa, sb, params, start_layer=start_layer
            )
            if route is None:
                continue
            score = da + db + 0.05 * route.total_cost
            if score < best_score:
                best_score = score
                best = route
    return best


def te_path_to_geojson(
    te: TimeExtendedGraph,
    route: RouteResult,
    *,
    properties: dict | None = None,
) -> dict:
    coords: list[list[float]] = []
    # Prefer fine spatial geometry via member map (recognizable streets)
    for ei in route.edge_indices:
        e = te.edges[ei]
        if e.kind != "travel":
            continue
        if e.fine_edge_indices:
            for fei in e.fine_edge_indices:
                for lat, lon in te.spatial.edges[fei].geometry:
                    pt = [lon, lat]
                    if not coords or coords[-1] != pt:
                        coords.append(pt)
        else:
            for lat, lon in e.geometry:
                pt = [lon, lat]
                if not coords or coords[-1] != pt:
                    coords.append(pt)
    if not coords:
        for nid in route.node_ids:
            n = te.nodes[nid]
            pt = [n.lon_deg, n.lat_deg]
            if not coords or coords[-1] != pt:
                coords.append(pt)
    props = {
        "length_m": route.length_m,
        "total_cost": route.total_cost,
        "mean_hdop": route.mean_hdop,
        "mean_n_los": route.mean_n_los,
        "n_edges_contracted": len([i for i in route.edge_indices if te.edges[i].kind == "travel"]),
        "n_edges_fine": sum(
            len(te.edges[i].fine_edge_indices)
            for i in route.edge_indices
            if te.edges[i].kind == "travel"
        ),
        "layers": route.layers,
        "spatial_node_ids": route.spatial_node_ids,
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


def path_to_geojson(
    graph: RoadGraph,
    route: RouteResult,
    *,
    properties: dict | None = None,
) -> dict:
    coords: list[list[float]] = []
    # Prefer expanded fine geometry when present
    if route.fine_geometry:
        for lat, lon in route.fine_geometry:
            coords.append([lon, lat])
    else:
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
        "n_edges_fine": len(route.fine_edge_indices) or len(route.edge_indices),
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
