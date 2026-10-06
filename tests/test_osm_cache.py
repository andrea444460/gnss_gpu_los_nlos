"""Tests for OSM Overpass disk cache (no network on cache hit)."""

from __future__ import annotations

from pathlib import Path

from gnss_gpu.io.osm_cache import (
    cache_path,
    fetch_roads_cached,
    filter_car_ways,
    read_roads_cache,
    write_roads_cache,
)
from gnss_gpu.io.osm_roads import BBox


def _fake_way(wid: int, highway: str = "residential") -> dict:
    return {
        "type": "way",
        "id": wid,
        "tags": {"highway": highway, "name": f"Way {wid}"},
        "geometry": [
            {"lat": 44.40, "lon": 8.93},
            {"lat": 44.401, "lon": 8.931},
        ],
    }


def test_fetch_roads_cached_writes_and_hits(tmp_path: Path):
    bbox = BBox(south=44.40, west=8.93, north=44.41, east=8.94)
    calls = {"n": 0}

    def fake_fetch(tile, **kwargs):
        calls["n"] += 1
        return [_fake_way(1), _fake_way(2, "footway"), _fake_way(3, "secondary")]

    roads1, info1 = fetch_roads_cached(
        bbox,
        cache_dir=tmp_path,
        car_only=True,
        tile_size_m=None,
        fetch_fn=fake_fetch,
    )
    assert info1["cache_hit"] is False
    assert calls["n"] == 1
    assert {int(w["id"]) for w in roads1} == {1, 3}  # footway dropped
    assert Path(info1["path"]).is_file()

    roads2, info2 = fetch_roads_cached(
        bbox,
        cache_dir=tmp_path,
        car_only=True,
        tile_size_m=None,
        fetch_fn=fake_fetch,
    )
    assert info2["cache_hit"] is True
    assert calls["n"] == 1  # no second network call
    assert len(roads2) == 2


def test_force_refresh_bypasses_cache(tmp_path: Path):
    bbox = BBox(south=44.40, west=8.93, north=44.41, east=8.94)
    calls = {"n": 0}

    def fake_fetch(tile, **kwargs):
        calls["n"] += 1
        return [_fake_way(10 + calls["n"])]

    fetch_roads_cached(
        bbox, cache_dir=tmp_path, tile_size_m=None, fetch_fn=fake_fetch, car_only=True
    )
    roads, info = fetch_roads_cached(
        bbox,
        cache_dir=tmp_path,
        tile_size_m=None,
        fetch_fn=fake_fetch,
        car_only=True,
        force_refresh=True,
    )
    assert info["cache_hit"] is False
    assert calls["n"] == 2
    assert roads[0]["id"] == 12


def test_filter_car_ways_drops_pedestrian():
    roads = [_fake_way(1, "residential"), _fake_way(2, "pedestrian"), _fake_way(3, "cycleway")]
    kept = filter_car_ways(roads)
    assert [w["id"] for w in kept] == [1]


def test_write_read_roundtrip(tmp_path: Path):
    bbox = BBox(44.1, 8.1, 44.2, 8.2)
    path = tmp_path / "sample.json"
    ways = filter_car_ways([_fake_way(5, "tertiary")])
    write_roads_cache(path, bbox=bbox, roads=ways, meta={"k": 1})
    roads, meta = read_roads_cache(path)
    assert len(roads) == 1
    assert meta["n_ways"] == 1
    assert cache_path(bbox, tmp_path).name.startswith("osm_roads_")
