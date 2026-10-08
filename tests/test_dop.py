"""Tests for NumPy HDOP helpers."""

from __future__ import annotations

import math

import numpy as np

from gnss_gpu.dop import hdop_from_los, n_los_and_hdop_chunk


def _ecef_from_lla(lat_deg: float, lon_deg: float, alt_m: float = 0.0) -> np.ndarray:
    a = 6378137.0
    e2 = 6.69437999014e-3
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    sin_lon, cos_lon = math.sin(lon), math.cos(lon)
    n = a / math.sqrt(1.0 - e2 * sin_lat * sin_lat)
    return np.array(
        [
            (n + alt_m) * cos_lat * cos_lon,
            (n + alt_m) * cos_lat * sin_lon,
            (n * (1.0 - e2) + alt_m) * sin_lat,
        ],
        dtype=np.float64,
    )


def test_hdop_needs_four_sats():
    rx = _ecef_from_lla(44.41, 8.93, 50.0)
    # Three sky directions → NaN
    sats = np.array(
        [
            rx + np.array([2e7, 0.0, 1e7]),
            rx + np.array([0.0, 2e7, 1e7]),
            rx + np.array([-2e7, 0.0, 1e7]),
        ],
        dtype=np.float64,
    )
    assert math.isnan(hdop_from_los(rx, sats, np.array([True, True, True])))


def test_hdop_finite_with_good_geometry():
    rx = _ecef_from_lla(44.41, 8.93, 50.0)
    offsets = [
        [2.0e7, 0.0, 1.5e7],
        [0.0, 2.0e7, 1.5e7],
        [-2.0e7, 0.0, 1.5e7],
        [0.0, -2.0e7, 1.5e7],
        [1.0e7, 1.0e7, 2.0e7],
    ]
    sats = np.array([rx + np.asarray(o, dtype=np.float64) for o in offsets], dtype=np.float64)
    mask = np.ones(len(offsets), dtype=bool)
    h = hdop_from_los(rx, sats, mask)
    assert math.isfinite(h)
    assert 0.5 < h < 20.0


def test_n_los_and_hdop_chunk_shape():
    rx = np.stack([_ecef_from_lla(44.41, 8.93, 50.0), _ecef_from_lla(44.42, 8.94, 60.0)])
    offsets = [
        [2.0e7, 0.0, 1.5e7],
        [0.0, 2.0e7, 1.5e7],
        [-2.0e7, 0.0, 1.5e7],
        [0.0, -2.0e7, 1.5e7],
        [1.0e7, 1.0e7, 2.0e7],
    ]
    base = rx[0]
    sats = np.stack([np.array([base + np.asarray(o, dtype=np.float64) for o in offsets])] * 2)
    # sats: (2 epochs, 5 sats, 3)
    mask = np.ones((2, 2, 5), dtype=bool)
    mask[1, 1, :] = False
    n_los, hdop = n_los_and_hdop_chunk(rx, sats, mask)
    assert n_los.shape == (2, 2)
    assert hdop.shape == (2, 2)
    assert n_los[0, 0] == 5.0
    assert n_los[1, 1] == 0.0
    assert math.isfinite(hdop[0, 0])
    assert math.isnan(hdop[1, 1])
