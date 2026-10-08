"""Shared pytest helpers for OSM disk cache (no live Overpass in unit tests)."""

from __future__ import annotations

from pathlib import Path

import pytest

from gnss_gpu.io.osm_cache import fetch_roads_cached, write_roads_cache
from gnss_gpu.io.osm_roads import BBox


@pytest.fixture
def osm_cache_dir(tmp_path: Path) -> Path:
    """Isolated cache directory for a single test (never touches fixtures/cache)."""
    d = tmp_path / "osm_cache"
    d.mkdir()
    return d


@pytest.fixture
def genova_bbox() -> BBox:
    return BBox(south=44.3950, west=8.9000, north=44.4250, east=8.9600)


@pytest.fixture
def seeded_osm_cache(osm_cache_dir: Path, genova_bbox: BBox) -> Path:
    """Pre-seed a tiny car-only extract so tests can assert cache-hit without network."""
    ways = [
        {
            "type": "way",
            "id": 1001,
            "tags": {"highway": "residential", "name": "Via Test"},
            "geometry": [
                {"lat": 44.4070, "lon": 8.9340},
                {"lat": 44.4075, "lon": 8.9345},
            ],
        }
    ]
    from gnss_gpu.io.osm_cache import cache_path

    path = cache_path(
        genova_bbox,
        osm_cache_dir,
        car_only=True,
        include_pedestrian=False,
        tile_size_m=2500.0,
    )
    write_roads_cache(path, bbox=genova_bbox, roads=ways, meta={"test_seed": True})
    return path


@pytest.fixture
def cached_roads(seeded_osm_cache: Path, osm_cache_dir: Path, genova_bbox: BBox):
    """Load roads via ``fetch_roads_cached`` (must be cache-hit; fetch_fn would fail)."""

    def boom(*_a, **_k):
        raise AssertionError("Overpass fetch must not run on cache hit")

    roads, info = fetch_roads_cached(
        genova_bbox,
        cache_dir=osm_cache_dir,
        car_only=True,
        tile_size_m=2500.0,
        fetch_fn=boom,
    )
    assert info["cache_hit"] is True
    assert len(roads) >= 1
    return roads, info
