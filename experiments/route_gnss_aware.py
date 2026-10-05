#!/usr/bin/env python3
"""GNSS-quality-aware A*/Dijkstra routing on an OSM road graph.

Pipeline:
1. Fetch (or load) OSM ways in a bbox → directed graph (oneway-aware)
2. Optionally attach mean_hdop / mean_n_los from an area-map quality CSV
3. Route A→B with cost  L * (1 + alpha*HDOP_pen + beta*nLOS_pen)
4. Also emit distance-only route (alpha=beta=0) for comparison

Example (synthetic quality CSV not required)::

    PYTHONPATH=python python experiments/route_gnss_aware.py \\
      --bbox 44.405,8.930,44.408,8.935 \\
      --from 44.4055,8.9310 --to 44.4075,8.9340 \\
      --alpha 1 --beta 0.5 \\
      --out-prefix experiments/results/route_demo
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_ROOT / "python"))

from gnss_gpu.io.osm_roads import BBox, build_directed_road_graph, fetch_roads_overpass
from gnss_gpu.routing import (
    CostParams,
    aggregate_quality_onto_edges,
    edges_to_csv,
    load_quality_points_csv,
    path_to_geojson,
    prepare_contracted_graph,
    route_latlon,
    write_geojson,
)


def _parse_latlon(s: str) -> tuple[float, float]:
    parts = s.replace(" ", "").split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"expected lat,lon got {s!r}")
    return float(parts[0]), float(parts[1])


def _parse_bbox(s: str) -> BBox:
    parts = s.replace(" ", "").split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("bbox must be south,west,north,east")
    south, west, north, east = (float(x) for x in parts)
    return BBox(south=south, west=west, north=north, east=east)


def _summarize(label: str, route) -> dict:
    if route is None:
        return {"label": label, "ok": False}
    return {
        "label": label,
        "ok": True,
        "length_m": route.length_m,
        "total_cost": route.total_cost,
        "mean_hdop": route.mean_hdop,
        "mean_n_los": route.mean_n_los,
        "n_edges": len(route.edge_indices),
        "node_ids": route.node_ids,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bbox", type=_parse_bbox, required=True, help="south,west,north,east")
    p.add_argument("--from", dest="origin", type=_parse_latlon, required=True, help="lat,lon")
    p.add_argument("--to", dest="dest", type=_parse_latlon, required=True, help="lat,lon")
    p.add_argument("--quality-csv", type=Path, default=None, help="area-map points with way_id + metrics")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--h0", type=float, default=2.0)
    p.add_argument("--n-star", type=float, default=8.0)
    p.add_argument("--p-max", type=float, default=5.0)
    p.add_argument("--algorithm", choices=("astar", "dijkstra"), default="astar")
    p.add_argument("--include-pedestrian", action="store_true")
    p.add_argument(
        "--contract",
        action="store_true",
        help="merge consecutive same-direction same-quality degree-2 edges",
    )
    p.add_argument("--out-prefix", type=Path, required=True)
    args = p.parse_args(argv)

    roads = fetch_roads_overpass(args.bbox, include_pedestrian=args.include_pedestrian)
    graph = build_directed_road_graph(roads)
    if args.quality_csv is not None:
        points = load_quality_points_csv(args.quality_csv)
        aggregate_quality_onto_edges(graph, points)
    if args.contract:
        graph = prepare_contracted_graph(graph)

    params = CostParams(
        alpha=args.alpha,
        beta=args.beta,
        h0=args.h0,
        n_star=args.n_star,
        p_max=args.p_max,
    )
    distance_only = CostParams(alpha=0.0, beta=0.0, h0=args.h0, n_star=args.n_star, p_max=args.p_max)

    lat_a, lon_a = args.origin
    lat_b, lon_b = args.dest
    gnss_route = route_latlon(
        graph, lat_a, lon_a, lat_b, lon_b, params, algorithm=args.algorithm
    )
    dist_route = route_latlon(
        graph, lat_a, lon_a, lat_b, lon_b, distance_only, algorithm=args.algorithm
    )

    out = Path(args.out_prefix)
    out.parent.mkdir(parents=True, exist_ok=True)
    edges_to_csv(graph, out.with_name(out.name + "_edges.csv"))

    summary = {
        "bbox": {
            "south": args.bbox.south,
            "west": args.bbox.west,
            "north": args.bbox.north,
            "east": args.bbox.east,
        },
        "origin": {"lat": lat_a, "lon": lon_a},
        "dest": {"lat": lat_b, "lon": lon_b},
        "n_nodes": len(graph.nodes),
        "n_edges": len(graph.edges),
        "params": {
            "alpha": params.alpha,
            "beta": params.beta,
            "h0": params.h0,
            "n_star": params.n_star,
            "p_max": params.p_max,
            "algorithm": args.algorithm,
        },
        "gnss_aware": _summarize("gnss_aware", gnss_route),
        "distance_only": _summarize("distance_only", dist_route),
    }
    out.with_name(out.name + "_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    if gnss_route is not None:
        write_geojson(
            out.with_name(out.name + "_gnss.geojson"),
            path_to_geojson(graph, gnss_route, properties={"kind": "gnss_aware"}),
        )
    if dist_route is not None:
        write_geojson(
            out.with_name(out.name + "_distance.geojson"),
            path_to_geojson(graph, dist_route, properties={"kind": "distance_only"}),
        )

    print(json.dumps(summary, indent=2))
    if gnss_route is None:
        print("ERROR: no GNSS-aware path found", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
