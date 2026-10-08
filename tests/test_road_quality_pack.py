"""Tests for compact road quality pack (.rqz)."""

from __future__ import annotations

from pathlib import Path

from gnss_gpu.io.road_quality_pack import (
    pack_from_samples,
    read_road_quality_pack,
    write_road_quality_pack,
)


def test_pack_rle_and_roundtrip(tmp_path: Path):
    # Same quantized key for first samples, then a clear jump → 2 runs.
    samples = {
        10: [
            (0.0, 1.0, 10.0),
            (60.0, 1.1, 10.0),
            (120.0, 1.2, 10.0),
            (180.0, 6.0, 3.0),
            (240.0, 6.2, 3.0),
        ],
        11: [
            (0.0, 2.0, 8.0),
            (300.0, 2.0, 8.0),
        ],
    }
    pack = pack_from_samples(
        samples,
        bbox=[44.0, 8.0, 45.0, 9.0],
        t0_s=0.0,
        horizon_s=300.0,
        dt_s=60.0,
        hdop_step=0.5,
        n_los_step=1.0,
        source="test",
    )
    assert pack.n_ways == 2
    by_id = {w.way_id: w for w in pack.ways}
    assert len(by_id[10].runs) == 2
    assert len(by_id[11].runs) == 1

    path = tmp_path / "q.rqz"
    write_road_quality_pack(path, pack)
    assert path.stat().st_size < 2000
    rt = read_road_quality_pack(path)
    assert rt.n_ways == 2
    d = rt.to_samples_dict()
    assert 10 in d and d[10][0][0] == 0.0
    # Last sentinel at horizon
    assert d[10][-1][0] == 300.0
