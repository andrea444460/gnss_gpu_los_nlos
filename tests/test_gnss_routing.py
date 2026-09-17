"""Toy checks for GNSS-aware Dijkstra/A* routing."""

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
    dijkstra_route,
    edge_cost,
)


def _toy_graph() -> RoadGraph:
    """
    Diamond with oneway short-but-bad GNSS vs long-but-good GNSS.

        0 --(100m, HDOP=8, oneway)--> 1
        |                              |
     (150m, HDOP=1)               (same as left, reverse not allowed on top)
        |                              |
        2 --(100m, HDOP=1)-----------> 1

    Actually simpler 3-node:

        A --short bad--> B
        A --long good--> C --good--> B

    with A->B direct oneway, and A->C->B bidirectional good GNSS.
    """
    nodes = {
        0: RoadNode(0, 0.0, 0.0),
        1: RoadNode(1, 0.0, 0.002),  # ~222 m east
        2: RoadNode(2, 0.001, 0.001),  # detour
    }
    # lengths chosen explicitly (ignore geo distance for cost tests)
    edges = [
        RoadEdge(0, 1, length_m=100.0, way_id=1, mean_hdop=8.0, mean_n_los=2.0, geometry=[(0.0, 0.0), (0.0, 0.002)]),
        RoadEdge(0, 2, length_m=120.0, way_id=2, mean_hdop=1.0, mean_n_los=10.0, geometry=[(0.0, 0.0), (0.001, 0.001)]),
        RoadEdge(2, 1, length_m=120.0, way_id=3, mean_hdop=1.0, mean_n_los=10.0, geometry=[(0.001, 0.001), (0.0, 0.002)]),
        # oneway: no edge 1 -> 0
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
    pairs = {(e.u, e.v) for e in g.edges}
    # oneway way 10: only one direction between its endpoints
    oneway_edges = [e for e in g.edges if e.way_id == 10]
    assert len(oneway_edges) == 1
    # bidirectional way 11: two directed edges
    both_edges = [e for e in g.edges if e.way_id == 11]
    assert len(both_edges) == 2
    assert len(pairs) == 3


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
    # short edge cost: 100 * (1 + 2 * clip(8/2,0,5)) = 100 * (1 + 2*4) = 900
    # long path: 2 * 120 * (1 + 2 * clip(1/2,0,5)) = 240 * (1 + 1) = 480
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
    # clear and re-aggregate
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
    e1 = next(e for e in g.edges if e.way_id == 2)
    assert abs(e1.mean_hdop - 2.0) < 1e-9


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
