"""Disk cache for Overpass road extracts (GUI + tests).

Cache files live under ``GNSS_GPU_OSM_CACHE`` if set, otherwise
``python/gnss_gpu/fixtures/cache/``. Keys are stable hashes of bbox + filters.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Callable

from gnss_gpu.io.osm_roads import BBox, fetch_roads_overpass, split_bbox_into_tiles

# Car-only highway classes used when normalizing cached extracts.
CAR_HIGHWAYS = frozenset(
    {
        "motorway",
        "trunk",
        "primary",
        "secondary",
        "tertiary",
        "unclassified",
        "residential",
        "living_street",
        "service",
        "motorway_link",
        "trunk_link",
        "primary_link",
        "secondary_link",
        "tertiary_link",
    }
)


def default_cache_dir() -> Path:
    env = os.environ.get("GNSS_GPU_OSM_CACHE", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return (
        Path(__file__).resolve().parents[1] / "fixtures" / "cache"
    ).resolve()


def cache_key(
    bbox: BBox,
    *,
    car_only: bool = True,
    include_pedestrian: bool = False,
    tile_size_m: float | None = None,
) -> str:
    payload = {
        "south": round(bbox.south, 5),
        "west": round(bbox.west, 5),
        "north": round(bbox.north, 5),
        "east": round(bbox.east, 5),
        "car_only": bool(car_only),
        "include_pedestrian": bool(include_pedestrian),
        "tile_size_m": tile_size_m,
        "v": 1,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def cache_path(bbox: BBox, cache_dir: Path | None = None, **kwargs) -> Path:
    root = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    return root / f"osm_roads_{cache_key(bbox, **kwargs)}.json"


def filter_car_ways(roads: list[dict]) -> list[dict]:
    out: list[dict] = []
    for w in roads:
        tags = w.get("tags") or {}
        hw = str(tags.get("highway", "")).strip().lower()
        if hw not in CAR_HIGHWAYS:
            continue
        geom = w.get("geometry") or []
        if len(geom) < 2:
            continue
        # Compact tags kept for graph build / display.
        t = {k: tags[k] for k in ("highway", "name", "oneway", "junction") if k in tags}
        out.append(
            {
                "type": "way",
                "id": int(w["id"]),
                "tags": t,
                "geometry": [
                    {"lat": float(p["lat"]), "lon": float(p["lon"])} for p in geom
                ],
            }
        )
    return out


def write_roads_cache(
    path: Path,
    *,
    bbox: BBox,
    roads: list[dict],
    meta: dict | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "bbox": [bbox.south, bbox.west, bbox.north, bbox.east],
        "n_ways": len(roads),
        "cached_at_unix": time.time(),
        "meta": meta or {},
        "elements": roads,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(path)
    return path


def read_roads_cache(path: Path) -> tuple[list[dict], dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    roads = [el for el in payload.get("elements", []) if el.get("type") == "way"]
    meta = {
        "bbox": payload.get("bbox"),
        "n_ways": payload.get("n_ways", len(roads)),
        "cached_at_unix": payload.get("cached_at_unix"),
        "meta": payload.get("meta") or {},
        "path": str(path),
    }
    return roads, meta


def fetch_roads_cached(
    bbox: BBox,
    *,
    cache_dir: Path | None = None,
    car_only: bool = True,
    include_pedestrian: bool = False,
    force_refresh: bool = False,
    tile_size_m: float | None = 1200.0,
    timeout_s: int = 90,
    max_attempts_per_endpoint: int = 3,
    fetch_fn: Callable[..., list[dict]] | None = None,
) -> tuple[list[dict], dict]:
    """Load roads from disk cache, or fetch Overpass (optionally tiled) and store.

    ``fetch_fn`` is injectable for unit tests (defaults to ``fetch_roads_overpass``).
    Returns ``(roads, info)`` where ``info`` has ``cache_hit``, ``path``, ``n_ways``.
    """
    path = cache_path(
        bbox,
        cache_dir,
        car_only=car_only,
        include_pedestrian=include_pedestrian,
        tile_size_m=tile_size_m,
    )
    if path.is_file() and not force_refresh:
        roads, meta = read_roads_cache(path)
        if car_only:
            roads = filter_car_ways(roads)
        return roads, {
            "cache_hit": True,
            "path": str(path),
            "n_ways": len(roads),
            "bbox": [bbox.south, bbox.west, bbox.north, bbox.east],
            **meta,
        }

    fetcher = fetch_fn or fetch_roads_overpass
    tiles = (
        split_bbox_into_tiles(bbox, tile_size_m=float(tile_size_m))
        if tile_size_m and tile_size_m > 0
        else [bbox]
    )
    dedup: dict[int, dict] = {}
    for tile in tiles:
        chunk = fetcher(
            tile,
            include_pedestrian=include_pedestrian,
            timeout_s=timeout_s,
            max_attempts_per_endpoint=max_attempts_per_endpoint,
        )
        for w in chunk:
            wid = w.get("id")
            if isinstance(wid, int):
                dedup[wid] = w

    roads = list(dedup.values())
    if car_only:
        roads = filter_car_ways(roads)

    write_roads_cache(
        path,
        bbox=bbox,
        roads=roads,
        meta={
            "car_only": car_only,
            "include_pedestrian": include_pedestrian,
            "tile_size_m": tile_size_m,
            "n_tiles": len(tiles),
        },
    )
    return roads, {
        "cache_hit": False,
        "path": str(path),
        "n_ways": len(roads),
        "bbox": [bbox.south, bbox.west, bbox.north, bbox.east],
        "n_tiles": len(tiles),
    }
