"""Edge contraction and time-extended GNSS road graphs.

Spatial edges are *not* one-per-GNSS-sample. Fine OSM segments get a quality
label; consecutive same-direction same-quality runs through degree-2 nodes are
then contracted into a single edge.

Time extension adds a layer whenever the quantized GNSS quality of any edge
changes. Nodes become (spatial_node, layer); waiting edges advance time.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

from gnss_gpu.io.osm_roads import RoadEdge, RoadGraph, RoadNode, haversine_m


@dataclass(frozen=True)
class QualityKey:
    """Quantized GNSS quality used for contraction / layer boundaries."""

    hdop_bin: int
    n_los_bin: int


@dataclass
class QualityInterval:
    """Constant-quality interval on one spatial edge / way."""

    t0_s: float
    t1_s: float
    mean_hdop: float
    mean_n_los: float

    def key(self, hdop_step: float = 0.5, n_los_step: float = 1.0) -> QualityKey:
        return quantize_quality(self.mean_hdop, self.mean_n_los, hdop_step, n_los_step)


@dataclass
class TimedRoadEdge(RoadEdge):
    """Spatial edge carrying a piecewise-constant quality timeline."""

    timeline: list[QualityInterval] = field(default_factory=list)


@dataclass
class TimeLayer:
    index: int
    t0_s: float
    t1_s: float


@dataclass
class TimeExtendedNode:
    node_id: int  # packed id in the TE graph
    spatial_id: int
    layer: int
    lat_deg: float
    lon_deg: float


@dataclass
class TimeExtendedEdge:
    u: int
    v: int
    length_m: float
    way_id: int
    kind: str  # "travel" | "wait"
    layer: int
    mean_hdop: float = float("nan")
    mean_n_los: float = float("nan")
    geometry: list[tuple[float, float]] = field(default_factory=list)
    highway: str = ""
    name: str = ""
    # Fine (uncontracted) spatial edge indices covered by this travel edge.
    fine_edge_indices: list[int] = field(default_factory=list)


@dataclass
class TimeExtendedGraph:
    layers: list[TimeLayer]
    nodes: dict[int, TimeExtendedNode]
    edges: list[TimeExtendedEdge]
    spatial: RoadGraph
    # (spatial_id, layer) -> te node id
    index: dict[tuple[int, int], int] = field(default_factory=dict)

    def adjacency(self) -> dict[int, list[tuple[int, int]]]:
        adj: dict[int, list[tuple[int, int]]] = {nid: [] for nid in self.nodes}
        for i, e in enumerate(self.edges):
            adj.setdefault(e.u, []).append((i, e.v))
        return adj


def quantize_quality(
    hdop: float,
    n_los: float,
    hdop_step: float = 0.5,
    n_los_step: float = 1.0,
) -> QualityKey:
    h_bin = -1 if not math.isfinite(hdop) else int(math.floor(hdop / max(1e-9, hdop_step)))
    n_bin = -1 if not math.isfinite(n_los) else int(math.floor(n_los / max(1e-9, n_los_step)))
    return QualityKey(hdop_bin=h_bin, n_los_bin=n_bin)


def qualities_equal(
    a_hdop: float,
    a_n_los: float,
    b_hdop: float,
    b_n_los: float,
    *,
    hdop_step: float = 0.5,
    n_los_step: float = 1.0,
) -> bool:
    return quantize_quality(a_hdop, a_n_los, hdop_step, n_los_step) == quantize_quality(
        b_hdop, b_n_los, hdop_step, n_los_step
    )


def _undirected_degree(graph: RoadGraph) -> dict[int, int]:
    """Count unique undirected neighbors (junction detection)."""
    neigh: dict[int, set[int]] = {nid: set() for nid in graph.nodes}
    for e in graph.edges:
        neigh.setdefault(e.u, set()).add(e.v)
        neigh.setdefault(e.v, set()).add(e.u)
    return {nid: len(s) for nid, s in neigh.items()}


@dataclass
class ContractedGraph:
    """Routing graph after quality/direction contraction, with expand map.

    ``graph`` is what Dijkstra/A* should run on.
    ``fine`` is the original detailed road network (for map display).
    ``members[i]`` lists fine-edge indices that contracted edge ``i`` covers,
    in travel order — used to expand a routed path back to recognizable streets.
    """

    graph: RoadGraph
    fine: RoadGraph
    members: list[list[int]]


def contract_same_quality_edges(
    graph: RoadGraph,
    *,
    hdop_step: float = 0.5,
    n_los_step: float = 1.0,
) -> ContractedGraph:
    """Merge consecutive same-direction same-quality degree-2 chains.

    Junctions are nodes with undirected neighbor count != 2. Bidirectional
    OSM edges are OK: at an intermediate node we continue to the *other*
    neighbor (not the predecessor), requiring a directed edge that keeps the
    same ``way_id`` and quantized (HDOP, n_LOS).

    Returns a :class:`ContractedGraph` so callers can route on the compact
    graph and then expand the path with :func:`expand_contracted_route`.
    """
    if not graph.edges:
        empty = RoadGraph(nodes=dict(graph.nodes), edges=[])
        return ContractedGraph(graph=empty, fine=graph, members=[])

    undeg = _undirected_degree(graph)
    neighbors: dict[int, set[int]] = {nid: set() for nid in graph.nodes}
    edge_uv: dict[tuple[int, int], int] = {}
    for i, e in enumerate(graph.edges):
        neighbors.setdefault(e.u, set()).add(e.v)
        neighbors.setdefault(e.v, set()).add(e.u)
        edge_uv[(e.u, e.v)] = i

    def _same(e0: RoadEdge, e1: RoadEdge) -> bool:
        if e0.way_id != e1.way_id:
            return False
        return qualities_equal(
            e0.mean_hdop,
            e0.mean_n_los,
            e1.mean_hdop,
            e1.mean_n_los,
            hdop_step=hdop_step,
            n_los_step=n_los_step,
        )

    def _continue(prev: int, node: int, ref: RoadEdge) -> int | None:
        """Next edge index node→w, or None if chain must stop."""
        if undeg.get(node, 0) != 2:
            return None
        opts = [w for w in neighbors.get(node, set()) if w != prev]
        if len(opts) != 1:
            return None
        w = opts[0]
        ei = edge_uv.get((node, w))
        if ei is None:
            return None
        if not _same(ref, graph.edges[ei]):
            return None
        return ei

    used: set[int] = set()
    new_edges: list[RoadEdge] = []
    members: list[list[int]] = []

    for start_ei, start_e in enumerate(graph.edges):
        if start_ei in used:
            continue
        chain = [start_ei]
        # Expand backward: pred → u → v  (u must be undirected degree-2)
        while True:
            head = graph.edges[chain[0]]
            if undeg.get(head.u, 0) != 2:
                break
            opts = [w for w in neighbors.get(head.u, set()) if w != head.v]
            if len(opts) != 1:
                break
            pred = opts[0]
            prev_ei = edge_uv.get((pred, head.u))
            if prev_ei is None or prev_ei in used:
                break
            if not _same(graph.edges[prev_ei], head):
                break
            chain.insert(0, prev_ei)

        # Expand forward
        while True:
            tail = graph.edges[chain[-1]]
            next_ei = _continue(tail.u, tail.v, tail)
            if next_ei is None or next_ei in used:
                break
            chain.append(next_ei)

        for ei in chain:
            used.add(ei)

        first = graph.edges[chain[0]]
        last = graph.edges[chain[-1]]
        geom: list[tuple[float, float]] = []
        length = 0.0
        for ei in chain:
            e = graph.edges[ei]
            length += e.length_m
            for pt in e.geometry:
                if not geom or geom[-1] != pt:
                    geom.append(pt)
        new_edges.append(
            RoadEdge(
                u=first.u,
                v=last.v,
                length_m=length,
                way_id=first.way_id,
                highway=first.highway,
                name=first.name,
                geometry=geom,
                mean_hdop=first.mean_hdop,
                mean_n_los=first.mean_n_los,
            )
        )
        members.append(list(chain))

    # Keep *all* fine nodes in the contracted graph's node dict so snap/display
    # can still reference intermediate coordinates; adjacency only uses endpoints.
    return ContractedGraph(
        graph=RoadGraph(nodes=dict(graph.nodes), edges=new_edges),
        fine=graph,
        members=members,
    )


def expand_contracted_route(
    contracted: ContractedGraph,
    route_edge_indices: list[int],
) -> tuple[list[int], list[int], list[tuple[float, float]]]:
    """Expand contracted route edges → fine edge indices, node ids, polyline.

    Returns ``(fine_edge_indices, fine_node_ids, geometry_latlon)``.
    """
    fine_edges: list[int] = []
    geom: list[tuple[float, float]] = []
    for cei in route_edge_indices:
        for fei in contracted.members[cei]:
            fine_edges.append(fei)
            e = contracted.fine.edges[fei]
            for pt in e.geometry:
                if not geom or geom[-1] != pt:
                    geom.append(pt)
    nodes: list[int] = []
    if fine_edges:
        nodes.append(contracted.fine.edges[fine_edges[0]].u)
        for fei in fine_edges:
            nodes.append(contracted.fine.edges[fei].v)
    return fine_edges, nodes, geom


def collapse_quality_timeline(
    samples: list[tuple[float, float, float]],
    *,
    hdop_step: float = 0.5,
    n_los_step: float = 1.0,
) -> list[QualityInterval]:
    """Collapse (t, hdop, n_los) samples into constant-quality intervals.

    ``samples`` must be sorted by time. Interval boundaries are exactly the
    times when the quantized quality key changes.
    """
    if not samples:
        return []
    ordered = sorted(samples, key=lambda x: x[0])
    intervals: list[QualityInterval] = []
    t0, h0, n0 = ordered[0]
    cur_key = quantize_quality(h0, n0, hdop_step, n_los_step)
    acc_h = [h0]
    acc_n = [n0]
    for t, h, n in ordered[1:]:
        key = quantize_quality(h, n, hdop_step, n_los_step)
        if key != cur_key:
            intervals.append(
                QualityInterval(
                    t0_s=float(t0),
                    t1_s=float(t),
                    mean_hdop=float(sum(acc_h) / len(acc_h)),
                    mean_n_los=float(sum(acc_n) / len(acc_n)),
                )
            )
            t0, h0, n0 = t, h, n
            cur_key = key
            acc_h = [h]
            acc_n = [n]
        else:
            acc_h.append(h)
            acc_n.append(n)
    # open-ended last interval: extend by last sample dt or +1s
    t_end = ordered[-1][0]
    if t_end <= t0:
        t_end = t0 + 1.0
    intervals.append(
        QualityInterval(
            t0_s=float(t0),
            t1_s=float(t_end if t_end > t0 else t0 + 1.0),
            mean_hdop=float(sum(acc_h) / len(acc_h)),
            mean_n_los=float(sum(acc_n) / len(acc_n)),
        )
    )
    # If only one sample, give a unit interval
    if len(intervals) == 1 and intervals[0].t1_s <= intervals[0].t0_s:
        intervals[0] = QualityInterval(
            t0_s=intervals[0].t0_s,
            t1_s=intervals[0].t0_s + 1.0,
            mean_hdop=intervals[0].mean_hdop,
            mean_n_los=intervals[0].mean_n_los,
        )
    return intervals


def build_layer_boundaries(
    edge_timelines: Iterable[list[QualityInterval]],
) -> list[float]:
    """Sorted unique times where any edge's quality interval starts/ends."""
    times: set[float] = set()
    for tl in edge_timelines:
        for iv in tl:
            times.add(float(iv.t0_s))
            times.add(float(iv.t1_s))
    return sorted(times)


def quality_at(timeline: list[QualityInterval], t_s: float) -> tuple[float, float]:
    if not timeline:
        return float("nan"), float("nan")
    for iv in timeline:
        if iv.t0_s <= t_s < iv.t1_s:
            return iv.mean_hdop, iv.mean_n_los
    # clamp to last
    last = timeline[-1]
    if t_s >= last.t1_s:
        return last.mean_hdop, last.mean_n_los
    return timeline[0].mean_hdop, timeline[0].mean_n_los


def attach_timelines_by_way(
    graph: RoadGraph,
    samples_by_way: dict[int, list[tuple[float, float, float]]],
    *,
    hdop_step: float = 0.5,
    n_los_step: float = 1.0,
) -> list[list[QualityInterval]]:
    """Return parallel list of timelines, one per graph.edges entry."""
    out: list[list[QualityInterval]] = []
    cache: dict[int, list[QualityInterval]] = {}
    for e in graph.edges:
        if e.way_id not in cache:
            cache[e.way_id] = collapse_quality_timeline(
                samples_by_way.get(e.way_id, []),
                hdop_step=hdop_step,
                n_los_step=n_los_step,
            )
        tl = cache[e.way_id]
        out.append(tl)
        if tl:
            # set static fields to first-layer values for contraction helpers
            e.mean_hdop = tl[0].mean_hdop
            e.mean_n_los = tl[0].mean_n_los
    return out


def build_time_extended_graph(
    spatial: RoadGraph,
    timelines: list[list[QualityInterval]],
    *,
    wait_cost: float = 0.0,
    hdop_step: float = 0.5,
    n_los_step: float = 1.0,
    contract_per_layer: bool = True,
) -> TimeExtendedGraph:
    """Build a time-extended graph; layers split at GNSS quality change times.

    - Travel edges stay inside one layer and use that layer's quality.
    - Wait edges (u,k) -> (u,k+1) cost ``wait_cost`` (default 0).
    - If ``contract_per_layer``, spatial edges are contracted using that layer's
      quality before being lifted into the TE graph (per layer independently
      for travel edges; node set remains the union of contracted endpoints plus
      all original spatial nodes that appear).
    """
    if len(timelines) != len(spatial.edges):
        raise ValueError("timelines must align with spatial.edges")

    bounds = build_layer_boundaries(timelines)
    if len(bounds) < 2:
        # static single layer
        bounds = [0.0, 1.0]
    layers = [
        TimeLayer(index=i, t0_s=bounds[i], t1_s=bounds[i + 1])
        for i in range(len(bounds) - 1)
        if bounds[i + 1] > bounds[i]
    ]
    if not layers:
        layers = [TimeLayer(0, 0.0, 1.0)]

    te_nodes: dict[int, TimeExtendedNode] = {}
    index: dict[tuple[int, int], int] = {}
    te_edges: list[TimeExtendedEdge] = []

    def _te_node(spatial_id: int, layer: int) -> int:
        key = (spatial_id, layer)
        nid = index.get(key)
        if nid is not None:
            return nid
        sn = spatial.nodes[spatial_id]
        nid = len(te_nodes)
        index[key] = nid
        te_nodes[nid] = TimeExtendedNode(
            node_id=nid,
            spatial_id=spatial_id,
            layer=layer,
            lat_deg=sn.lat_deg,
            lon_deg=sn.lon_deg,
        )
        return nid

    for layer in layers:
        t_mid = 0.5 * (layer.t0_s + layer.t1_s)
        # Materialize a spatial snapshot with layer quality
        snap_edges: list[RoadEdge] = []
        for e, tl in zip(spatial.edges, timelines):
            h, n = quality_at(tl, t_mid)
            snap_edges.append(
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
        snap = RoadGraph(nodes=dict(spatial.nodes), edges=snap_edges)
        member_map: list[list[int]] | None = None
        if contract_per_layer:
            contracted = contract_same_quality_edges(
                snap, hdop_step=hdop_step, n_los_step=n_los_step
            )
            # Remap members: contracted.members index into snap.edges, which
            # align 1:1 with spatial.edges for this snapshot.
            member_map = contracted.members
            snap = contracted.graph
        for i, e in enumerate(snap.edges):
            if e.u not in spatial.nodes or e.v not in spatial.nodes:
                continue
            u = _te_node(e.u, layer.index)
            v = _te_node(e.v, layer.index)
            fine_idx = list(member_map[i]) if member_map is not None else []
            # Geometry from fine segments when available
            geom = list(e.geometry)
            if fine_idx:
                geom = []
                for fei in fine_idx:
                    for pt in spatial.edges[fei].geometry:
                        if not geom or geom[-1] != pt:
                            geom.append(pt)
            te_edges.append(
                TimeExtendedEdge(
                    u=u,
                    v=v,
                    length_m=e.length_m,
                    way_id=e.way_id,
                    kind="travel",
                    layer=layer.index,
                    mean_hdop=e.mean_hdop,
                    mean_n_los=e.mean_n_los,
                    geometry=geom,
                    highway=e.highway,
                    name=e.name,
                    fine_edge_indices=fine_idx,
                )
            )

    # Waiting edges between layers for every spatial node that appears
    spatial_ids = sorted({n.spatial_id for n in te_nodes.values()})
    for sid in spatial_ids:
        for layer in layers[:-1]:
            # create nodes even if isolated in a layer
            u = _te_node(sid, layer.index)
            v = _te_node(sid, layer.index + 1)
            te_edges.append(
                TimeExtendedEdge(
                    u=u,
                    v=v,
                    length_m=0.0,
                    way_id=-1,
                    kind="wait",
                    layer=layer.index,
                    mean_hdop=float("nan"),
                    mean_n_los=float("nan"),
                    geometry=[],
                )
            )

    # Store wait_cost on wait edges via length_m abuse? Better: keep length 0 and
    # handle wait_cost in the router. Annotate with mean_hdop unused; store cost
    # in length_m as wait_cost for simplicity in generic Dijkstra.
    if wait_cost != 0.0:
        for e in te_edges:
            if e.kind == "wait":
                e.length_m = float(wait_cost)

    return TimeExtendedGraph(
        layers=layers,
        nodes=te_nodes,
        edges=te_edges,
        spatial=spatial,
        index=index,
    )


def synthesize_quality_timeseries(
    graph: RoadGraph,
    *,
    lat_split: float | None = None,
) -> dict[int, list[tuple[float, float, float]]]:
    """Invent piecewise GNSS timelines so the GUI works without a real sim.

    Spatial pattern (visible already at t=0):
    - driveable arterials (secondary/tertiary) stay good
    - residential/service are mixed
    - pedestrian/footway corridors act as urban canyons (worse HDOP / fewer LOS)
    - a soft N/S + E/W gradient adds spatial texture

    Temporal pattern: northern half of the map degrades after t≈100s so the
    time-extended layers show a real quality change.
    """
    if lat_split is None:
        lats = [n.lat_deg for n in graph.nodes.values()]
        lat_split = float(sorted(lats)[len(lats) // 2]) if lats else 0.0
    lons = [n.lon_deg for n in graph.nodes.values()]
    lon0 = float(min(lons)) if lons else 0.0
    lon1 = float(max(lons)) if lons else 1.0
    lon_span = max(1e-9, lon1 - lon0)

    # Base (hdop, n_los) by highway class — canyon-like for walkways.
    class_base: dict[str, tuple[float, float]] = {
        "motorway": (0.9, 12.0),
        "trunk": (1.0, 12.0),
        "primary": (1.1, 11.0),
        "secondary": (1.3, 11.0),
        "tertiary": (1.6, 10.0),
        "residential": (2.4, 8.0),
        "living_street": (2.8, 7.0),
        "unclassified": (2.6, 8.0),
        "service": (3.2, 6.0),
        "cycleway": (3.8, 5.0),
        "pedestrian": (4.5, 4.0),
        "footway": (5.5, 3.0),
        "path": (5.0, 3.5),
    }

    samples: dict[int, list[tuple[float, float, float]]] = {}
    for e in graph.edges:
        if e.way_id in samples:
            continue
        if e.geometry:
            mid_lat = 0.5 * (e.geometry[0][0] + e.geometry[-1][0])
            mid_lon = 0.5 * (e.geometry[0][1] + e.geometry[-1][1])
        else:
            nu, nv = graph.nodes[e.u], graph.nodes[e.v]
            mid_lat = 0.5 * (nu.lat_deg + nv.lat_deg)
            mid_lon = 0.5 * (nu.lon_deg + nv.lon_deg)

        h0, n0 = class_base.get(str(e.highway or "").lower(), (3.0, 6.0))
        # Soft spatial texture so nearby streets aren't identical.
        east = (mid_lon - lon0) / lon_span
        north = 1.0 if mid_lat >= lat_split else 0.0
        # Deterministic jitter from way_id (stable across runs).
        jitter = ((int(e.way_id) * 1103515245 + 12345) & 0x7FFF) / 32767.0
        hdop_t0 = max(0.8, h0 + 1.8 * east + 0.6 * north + 0.9 * (jitter - 0.5))
        nlos_t0 = max(1.0, n0 - 2.0 * east - 1.0 * north - 1.5 * (jitter - 0.5))

        if mid_lat >= lat_split:
            # Northern corridors degrade over time (urban canyon worsens).
            samples[e.way_id] = [
                (0.0, hdop_t0, nlos_t0),
                (50.0, hdop_t0 + 0.3, max(1.0, nlos_t0 - 0.5)),
                (100.0, min(9.0, hdop_t0 + 3.5), max(1.0, nlos_t0 - 4.0)),
                (150.0, min(9.5, hdop_t0 + 4.0), max(1.0, nlos_t0 - 5.0)),
                (200.0, min(9.5, hdop_t0 + 4.0), max(1.0, nlos_t0 - 5.0)),
            ]
        else:
            samples[e.way_id] = [
                (0.0, hdop_t0, nlos_t0),
                (100.0, hdop_t0 + 0.15, max(1.0, nlos_t0 - 0.2)),
                (200.0, hdop_t0 + 0.3, max(1.0, nlos_t0 - 0.4)),
            ]
    return samples


def make_demo_spatial_graph() -> tuple[RoadGraph, dict[int, list[tuple[float, float, float]]]]:
    """Small synthetic city block with time-varying GNSS quality for GUI/tests.

    Layout (lat/lon degrees, ~111m per 0.001)::

        0 ---- way1 ---- 1 ---- way2 ---- 2
        |                |                |
       way3             way4             way5
        |                |                |
        3 ---- way6 ---- 4 ---- way7 ---- 5

    way1 becomes bad GNSS after t=100; way6 stays good. Short path 0→2 via top
    is bad later; bottom detour stays cleaner.
    """
    coords = {
        0: (44.4060, 8.9320),
        1: (44.4060, 8.9330),
        2: (44.4060, 8.9340),
        3: (44.4050, 8.9320),
        4: (44.4050, 8.9330),
        5: (44.4050, 8.9340),
        # micro-nodes on way1 for contraction demo
        10: (44.4060, 8.93233),
        11: (44.4060, 8.93266),
    }
    nodes = {i: RoadNode(i, lat, lon) for i, (lat, lon) in coords.items()}

    def _edge(u: int, v: int, way_id: int) -> RoadEdge:
        a, b = coords[u], coords[v]
        length = haversine_m(a[0], a[1], b[0], b[1])
        return RoadEdge(
            u=u,
            v=v,
            length_m=length,
            way_id=way_id,
            highway="residential",
            geometry=[a, b],
        )

    # Bidirectional streets (way1 is the micro-chain 0-10-11-1, not a single edge)
    pairs = [
        (1, 2, 2),
        (0, 3, 3),
        (1, 4, 4),
        (2, 5, 5),
        (3, 4, 6),
        (4, 5, 7),
    ]
    edges: list[RoadEdge] = []
    for u, v, wid in pairs:
        edges.append(_edge(u, v, wid))
        edges.append(_edge(v, u, wid))

    chain = [0, 10, 11, 1]
    for i in range(len(chain) - 1):
        u, v = chain[i], chain[i + 1]
        edges.append(_edge(u, v, 1))
        edges.append(_edge(v, u, 1))

    # Time series: most ways good always; way1/way2 good then bad after t=100
    samples: dict[int, list[tuple[float, float, float]]] = {}
    for wid in (1, 2, 3, 4, 5, 6, 7):
        if wid in (1, 2):
            samples[wid] = [
                (0.0, 1.5, 10.0),
                (50.0, 1.6, 10.0),
                (100.0, 7.0, 3.0),
                (150.0, 7.5, 2.0),
                (200.0, 7.5, 2.0),
            ]
        else:
            samples[wid] = [
                (0.0, 1.2, 11.0),
                (100.0, 1.3, 11.0),
                (200.0, 1.4, 10.0),
            ]

    return RoadGraph(nodes=nodes, edges=edges), samples
