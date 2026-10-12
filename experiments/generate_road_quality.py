#!/usr/bin/env python3
"""Generate a compact per-way GNSS quality pack (``.rqz``) for a road extract.

The pack stores **quantized change-points only** (not one sample per epoch),
so a day of city data stays small. Values are synthetic unless ``--from-csv``
is provided (real pipeline can write the same format later).

Example::

    PYTHONPATH=python python experiments/generate_road_quality.py \\
      --horizon-h 24 --dt-s 300 --arterial-only \\
      --out python/gnss_gpu/fixtures/cache/genova_quality_24h.rqz
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_ROOT / "python"))

from gnss_gpu.io.osm_cache import (  # noqa: E402
    CAR_HIGHWAYS,
    fetch_roads_cached,
    filter_car_ways,
)
from gnss_gpu.io.osm_roads import BBox  # noqa: E402
from gnss_gpu.io.road_quality_pack import (  # noqa: E402
    naive_dense_bytes,
    pack_from_samples,
    read_road_quality_pack,
    write_road_quality_pack,
)
from gnss_gpu.routing import load_quality_timeseries_csv  # noqa: E402

DEFAULT_BBOX = BBox(south=44.3850, west=8.8800, north=44.4450, east=8.9800)

ARTERIAL = frozenset(
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


def _way_mid(way: dict) -> tuple[float, float]:
    geom = way.get("geometry") or []
    if not geom:
        return 0.0, 0.0
    a, b = geom[0], geom[-1]
    return 0.5 * (float(a["lat"]) + float(b["lat"])), 0.5 * (
        float(a["lon"]) + float(b["lon"])
    )


def synthesize_horizon_samples(
    roads: list[dict],
    *,
    horizon_s: float,
    dt_s: float,
    bbox: BBox,
) -> dict[int, list[tuple[float, float, float]]]:
    """Lightweight synthetic diurnal GNSS field on a fixed time grid.

    Not a substitute for raycast LOS — only fills the compact pack format for
    demos/tests until a real generator writes the same ``.rqz``.
    """
    lat0, lat1 = bbox.south, bbox.north
    lon0, lon1 = bbox.west, bbox.east
    lat_span = max(1e-9, lat1 - lat0)
    lon_span = max(1e-9, lon1 - lon0)
    n_steps = int(horizon_s / dt_s) + 1
    times = [i * dt_s for i in range(n_steps)]

    class_base = {
        "motorway": (0.9, 12.0),
        "trunk": (1.0, 12.0),
        "primary": (1.1, 11.0),
        "secondary": (1.3, 11.0),
        "tertiary": (1.6, 10.0),
        "residential": (2.4, 8.0),
        "unclassified": (2.6, 8.0),
        "service": (3.2, 6.0),
    }

    out: dict[int, list[tuple[float, float, float]]] = {}
    for way in roads:
        wid = int(way["id"])
        tags = way.get("tags") or {}
        hw = str(tags.get("highway", "")).strip().lower()
        h0, n0 = class_base.get(hw, (3.0, 6.0))
        mid_lat, mid_lon = _way_mid(way)
        east = (mid_lon - lon0) / lon_span
        north = (mid_lat - lat0) / lat_span
        jitter = ((wid * 1103515245 + 12345) & 0x7FFF) / 32767.0
        seq: list[tuple[float, float, float]] = []
        for t in times:
            # Diurnal: worse mid-day in denser/east/north; mild night recovery.
            day = 0.5 - 0.5 * math.cos(2.0 * math.pi * (t / max(horizon_s, 1.0)))
            h = h0 + 1.5 * east + 0.8 * north + 2.2 * day * (0.4 + 0.6 * east) + 0.5 * (jitter - 0.5)
            n = n0 - 1.5 * east - 1.0 * north - 3.0 * day * (0.5 + 0.5 * north) - 0.8 * (jitter - 0.5)
            seq.append(
                (
                    float(t),
                    float(max(0.8, min(9.5, h))),
                    float(max(1.0, min(14.0, n))),
                )
            )
        out[wid] = seq
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bbox", default=None, help="south,west,north,east")
    p.add_argument("--horizon-h", type=float, default=24.0, help="Horizon hours (default 24)")
    p.add_argument("--dt-s", type=float, default=300.0, help="Sample grid step seconds (default 300)")
    p.add_argument("--hdop-step", type=float, default=0.5)
    p.add_argument("--n-los-step", type=float, default=1.0)
    p.add_argument("--arterial-only", action="store_true", help="Only arterial highways")
    p.add_argument("--from-csv", type=Path, default=None, help="Optional dense/sparse CSV input")
    p.add_argument(
        "--out",
        type=Path,
        default=Path("python/gnss_gpu/fixtures/cache/genova_quality_24h.rqz"),
    )
    args = p.parse_args(argv)

    bbox = DEFAULT_BBOX
    if args.bbox:
        parts = [float(x) for x in args.bbox.replace(" ", "").split(",")]
        bbox = BBox(south=parts[0], west=parts[1], north=parts[2], east=parts[3])

    horizon_s = float(args.horizon_h) * 3600.0
    dt_s = float(args.dt_s)
    t0 = time.perf_counter()

    if args.from_csv is not None:
        samples = load_quality_timeseries_csv(args.from_csv)
        source = f"csv:{args.from_csv}"
        n_ways_src = len(samples)
        print(f"loaded CSV ways={n_ways_src}", flush=True)
    else:
        roads, info = fetch_roads_cached(
            bbox, car_only=True, include_pedestrian=False, tile_size_m=3000.0
        )
        roads = filter_car_ways(roads)
        if args.arterial_only:
            roads = [
                w
                for w in roads
                if str((w.get("tags") or {}).get("highway", "")).lower() in ARTERIAL
            ]
        print(
            f"roads={len(roads)} cache_hit={info.get('cache_hit')} "
            f"horizon={args.horizon_h}h dt={dt_s}s arterial={args.arterial_only}",
            flush=True,
        )
        samples = synthesize_horizon_samples(
            roads, horizon_s=horizon_s, dt_s=dt_s, bbox=bbox
        )
        source = "synthetic_diurnal_v1"
        n_ways_src = len(samples)

    pack = pack_from_samples(
        samples,
        bbox=[bbox.south, bbox.west, bbox.north, bbox.east],
        t0_s=0.0,
        horizon_s=horizon_s,
        dt_s=dt_s,
        hdop_step=args.hdop_step,
        n_los_step=args.n_los_step,
        source=source,
        meta={
            "arterial_only": bool(args.arterial_only),
            "car_highways": sorted(CAR_HIGHWAYS),
        },
    )
    out = write_road_quality_pack(args.out, pack)
    # Round-trip sanity
    rt = read_road_quality_pack(out)
    size = out.stat().st_size
    n_epochs = int(horizon_s / dt_s) + 1
    dense = naive_dense_bytes(n_ways_src, n_epochs)
    print(
        f"wrote {out}  ({size/1024:.1f} KB)  ways={pack.n_ways} runs={pack.n_runs} "
        f"avg_runs/way={pack.n_runs/max(1,pack.n_ways):.2f}",
        flush=True,
    )
    print(
        f"vs naive dense float table ≈ {dense/1e6:.1f} MB  "
        f"compression ≈ {dense/max(1,size):.0f}×   "
        f"roundtrip_ways={rt.n_ways}  elapsed={time.perf_counter()-t0:.2f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
