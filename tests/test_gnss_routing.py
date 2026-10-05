"""Toy checks for GNSS-aware Dijkstra/A* routing, contraction, time-extended."""

from __future__ import annotations

from gnss_gpu.io.osm_roads import (
    RoadEdge,
    RoadGraph,
    RoadNode,
    build_directed_road_graph,
    oneway_direction,
)
from gnss_gpu.routing import (
    CostParams,
    aggregate_quality_onto_edges,
    astar_route,
    build_te_from_timeseries,
    dijkstra_route,
    dijkstra_te_route,
    edge_cost,
    prepare_contracted_graph,
)
from gnss_gpu.routing_graph import (
    collapse_quality_timeline,
    contract_same_quality_edges,
    make_demo_spatial_graph,
    quantize_quality,
)


def _toy_graph() -> RoadGraph:
    nodes = {
        0: RoadNode(0, 0.0, 0.0),
        1: RoadNode(1, 0.0, 0.002),
        2: RoadNode(2, 0.001, 0.001),
    }
    edges = [
        RoadEdge(0, 1, length_m=100.0, way_id=1, mean_hdop=8.0, mean_n_los=2.0, geometry=[(0.0, 0.0), (0.0, 0.002)]),
        RoadEdge(0, 2, length_m=120.0, way_id=2, mean_hdop=1.0, mean_n_los=10.0, geometry=[(0.0, 0.0), (0.001, 0.001)]),
        RoadEdge(2, 1, length_m=120.0, way_id=3, mean_hdop=1.0, mean_n_los=10.0, geometry=[(0.001, 0.001), (0.0, 0.002)]),
    ]
    return RoadGraph(nodes=nodes, edges=edges)


def test_oneway_direction_tags():
    assert oneway_direction({"oneway": "yes"}) == "forward"
    assert oneway_direction({"oneway": "-1"}) == "backward"
    assert oneway_direction({"oneway": "no"}) == "both"
    assert oneway_direction({"junction": "roundabout"}) == "forward"
    assert oneway_direction({"highway": "residential"}) == "both"


def test_build_directed_graph_respects_oneway():
    roads = [
        {
            "id": 10,
            "type": "way",
            "tags": {"highway": "residential", "oneway": "yes"},
            "geometry": [
                {"lat": 0.0, "lon": 0.0},
                {"lat": 0.0, "lon": 0.001},
            ],
        },
        {
            "id": 11,
            "type": "way",
            "tags": {"highway": "residential"},
            "geometry": [
                {"lat": 0.0, "lon": 0.001},
                {"lat": 0.001, "lon": 0.001},
            ],
        },
    ]
    g = build_directed_road_graph(roads)
    oneway_edges = [e for e in g.edges if e.way_id == 10]
    assert len(oneway_edges) == 1
    both_edges = [e for e in g.edges if e.way_id == 11]
    assert len(both_edges) == 2


def test_distance_only_prefers_short_oneway():
    g = _toy_graph()
    params = CostParams(alpha=0.0, beta=0.0)
    route = dijkstra_route(g, 0, 1, params)
    assert route is not None
    assert route.node_ids == [0, 1]
    assert abs(route.length_m - 100.0) < 1e-9


def test_high_alpha_prefers_longer_good_gnss():
    g = _toy_graph()
    params = CostParams(alpha=2.0, beta=0.0, h0=2.0, p_max=5.0)
    assert edge_cost(g.edges[0], params) > edge_cost(g.edges[1], params) + edge_cost(g.edges[2], params)
    route = dijkstra_route(g, 0, 1, params)
    assert route is not None
    assert route.node_ids == [0, 2, 1]
    assert abs(route.length_m - 240.0) < 1e-9


def test_astar_matches_dijkstra_on_toy():
    g = _toy_graph()
    params = CostParams(alpha=1.0, beta=0.5, h0=2.0, n_star=8.0)
    d = dijkstra_route(g, 0, 1, params)
    a = astar_route(g, 0, 1, params)
    assert d is not None and a is not None
    assert d.node_ids == a.node_ids
    assert abs(d.total_cost - a.total_cost) < 1e-9


def test_aggregate_quality_by_way_id():
    g = _toy_graph()
    for e in g.edges:
        e.mean_hdop = float("nan")
        e.mean_n_los = float("nan")
    points = [
        {"way_id": 1, "mean_hdop": 6.0, "mean_n_los": 3.0},
        {"way_id": 1, "mean_hdop": 8.0, "mean_n_los": 1.0},
        {"way_id": 2, "mean_hdop": 2.0, "mean_n_los": 9.0},
    ]
    aggregate_quality_onto_edges(g, points)
    e0 = next(e for e in g.edges if e.way_id == 1)
    assert abs(e0.mean_hdop - 7.0) < 1e-9
    assert abs(e0.mean_n_los - 2.0) < 1e-9


def test_no_path_against_oneway():
    nodes = {
        0: RoadNode(0, 0.0, 0.0),
        1: RoadNode(1, 0.0, 0.001),
    }
    edges = [
        RoadEdge(0, 1, length_m=50.0, way_id=1, mean_hdop=1.0, mean_n_los=8.0),
    ]
    g = RoadGraph(nodes=nodes, edges=edges)
    assert dijkstra_route(g, 1, 0, CostParams()) is None


def test_contract_merges_same_quality_chain():
    nodes = {
        0: RoadNode(0, 0.0, 0.0),
        1: RoadNode(1, 0.0, 0.001),
        2: RoadNode(2, 0.0, 0.002),
    }
    edges = [
        RoadEdge(0, 1, 100.0, way_id=7, mean_hdop=2.0, mean_n_los=8.0, geometry=[(0.0, 0.0), (0.0, 0.001)]),
        RoadEdge(1, 2, 100.0, way_id=7, mean_hdop=2.1, mean_n_los=8.0, geometry=[(0.0, 0.001), (0.0, 0.002)]),
    ]
    g = RoadGraph(nodes=nodes, edges=edges)
    c = contract_same_quality_edges(g, hdop_step=0.5, n_los_step=1.0)
    assert len(c.graph.edges) == 1
    assert c.graph.edges[0].u == 0 and c.graph.edges[0].v == 2
    assert abs(c.graph.edges[0].length_m - 200.0) < 1e-9
    assert c.members[0] == [0, 1]
    assert len(c.fine.edges) == 2  # fine graph unchanged


def test_contract_stops_when_quality_changes():
    nodes = {
        0: RoadNode(0, 0.0, 0.0),
        1: RoadNode(1, 0.0, 0.001),
        2: RoadNode(2, 0.0, 0.002),
    }
    edges = [
        RoadEdge(0, 1, 100.0, way_id=7, mean_hdop=1.0, mean_n_los=10.0, geometry=[(0.0, 0.0), (0.0, 0.001)]),
        RoadEdge(1, 2, 100.0, way_id=7, mean_hdop=6.0, mean_n_los=3.0, geometry=[(0.0, 0.001), (0.0, 0.002)]),
    ]
    g = RoadGraph(nodes=nodes, edges=edges)
    c = contract_same_quality_edges(g, hdop_step=0.5, n_los_step=1.0)
    assert len(c.graph.edges) == 2


def test_expand_route_restores_fine_geometry():
    from gnss_gpu.routing import CostParams, dijkstra_route, expand_route_result

    nodes = {
        0: RoadNode(0, 0.0, 0.0),
        1: RoadNode(1, 0.0, 0.001),
        2: RoadNode(2, 0.0, 0.002),
    }
    edges = [
        RoadEdge(0, 1, 100.0, way_id=7, mean_hdop=1.0, mean_n_los=10.0, geometry=[(0.0, 0.0), (0.0, 0.001)]),
        RoadEdge(1, 2, 100.0, way_id=7, mean_hdop=1.0, mean_n_los=10.0, geometry=[(0.0, 0.001), (0.0, 0.002)]),
    ]
    fine = RoadGraph(nodes=nodes, edges=edges)
    cg = contract_same_quality_edges(fine)
    assert len(cg.graph.edges) == 1
    route = dijkstra_route(cg.graph, 0, 2, CostParams(alpha=0, beta=0))
    assert route is not None
    assert route.edge_indices == [0]  # one contracted edge
    expand_route_result(cg, route)
    assert route.fine_edge_indices == [0, 1]
    assert route.fine_node_ids == [0, 1, 2]
    assert len(route.fine_geometry) == 3


def test_demo_contraction_reduces_micro_edges():
    spatial, samples = make_demo_spatial_graph()
    for e in spatial.edges:
        seq = samples[e.way_id]
        e.mean_hdop, e.mean_n_los = seq[0][1], seq[0][2]
    raw_way1 = [e for e in spatial.edges if e.way_id == 1]
    assert len(raw_way1) == 6  # 3 micro * 2 dirs
    c = prepare_contracted_graph(spatial)
    c_way1 = [e for e in c.graph.edges if e.way_id == 1]
    assert len(c_way1) == 2  # one per direction
    # map still has all fine edges
    assert len(c.fine.edges) == len(spatial.edges)


def test_collapse_timeline_on_quality_change():
    samples = [
        (0.0, 1.0, 10.0),
        (50.0, 1.2, 10.0),
        (100.0, 6.0, 3.0),
        (150.0, 6.4, 3.0),  # same HDOP bin as 6.0 with step 0.5
    ]
    ivs = collapse_quality_timeline(samples, hdop_step=0.5, n_los_step=1.0)
    assert len(ivs) == 2
    assert ivs[0].t0_s == 0.0 and ivs[0].t1_s == 100.0
    assert ivs[1].t0_s == 100.0
    assert quantize_quality(ivs[0].mean_hdop, ivs[0].mean_n_los, 0.5, 1.0) != quantize_quality(
        ivs[1].mean_hdop, ivs[1].mean_n_los, 0.5, 1.0
    )


def test_time_extended_prefers_detour_after_quality_drop():
    spatial, samples = make_demo_spatial_graph()
    te, _ = build_te_from_timeseries(spatial, samples, wait_cost=0.0, contract_per_layer=True)
    assert len(te.layers) >= 2

    # Force start in a late layer where top corridor is bad
    late = te.layers[-1].index
    params = CostParams(alpha=3.0, beta=0.5, h0=2.0, p_max=5.0)
    route = dijkstra_te_route(te, 0, 2, params, start_layer=late)
    assert route is not None
    # Should avoid pure top 0-1-2 if GNSS penalty is high — go via bottom
    assert 3 in route.spatial_node_ids or 4 in route.spatial_node_ids
    # Travel edges carry fine member map for display unpacking
    travel = [te.edges[i] for i in route.edge_indices if te.edges[i].kind == "travel"]
    assert travel and all(e.fine_edge_indices for e in travel)
