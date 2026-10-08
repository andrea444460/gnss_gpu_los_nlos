"""Lightweight DOP helpers (CPU / NumPy)."""

from __future__ import annotations

import math

import numpy as np


def _ecef_to_enu_matrix(lat_rad: float, lon_rad: float) -> np.ndarray:
    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)
    sin_lon = math.sin(lon_rad)
    cos_lon = math.cos(lon_rad)
    return np.array(
        [
            [-sin_lon, cos_lon, 0.0],
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
            [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],
        ],
        dtype=np.float64,
    )


def ecef_to_lla_rad(rx_ecef: np.ndarray) -> tuple[float, float, float]:
    """Approximate WGS84 ECEF → (lat_rad, lon_rad, alt_m)."""
    x, y, z = (float(rx_ecef[0]), float(rx_ecef[1]), float(rx_ecef[2]))
    a = 6378137.0
    e2 = 6.69437999014e-3
    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    lat = math.atan2(z, p * (1.0 - e2))
    for _ in range(6):
        sin_lat = math.sin(lat)
        n = a / math.sqrt(1.0 - e2 * sin_lat * sin_lat)
        alt = p / max(1e-12, math.cos(lat)) - n
        lat = math.atan2(z, p * (1.0 - e2 * n / (n + alt)))
    sin_lat = math.sin(lat)
    n = a / math.sqrt(1.0 - e2 * sin_lat * sin_lat)
    alt = p / max(1e-12, math.cos(lat)) - n
    return lat, lon, alt


def hdop_from_los(
    rx_ecef: np.ndarray,
    sat_ecef: np.ndarray,
    los_mask: np.ndarray,
    *,
    min_sats: int = 4,
) -> float:
    """HDOP from LOS satellite geometry at one receiver epoch.

    Uses the standard 4-column ENU+clock geometry matrix on LOS sats only.
    Returns NaN when fewer than ``min_sats`` are available or GᵀG is singular.
    """
    rx = np.asarray(rx_ecef, dtype=np.float64).reshape(3)
    sats = np.asarray(sat_ecef, dtype=np.float64).reshape(-1, 3)
    mask = np.asarray(los_mask, dtype=bool).reshape(-1)
    idx = np.flatnonzero(mask)
    if idx.size < int(min_sats):
        return float("nan")

    lat, lon, _alt = ecef_to_lla_rad(rx)
    r_enu = _ecef_to_enu_matrix(lat, lon)
    d = sats[idx] - rx
    rng = np.linalg.norm(d, axis=1)
    ok = np.isfinite(rng) & (rng > 1.0)
    if int(np.count_nonzero(ok)) < int(min_sats):
        return float("nan")
    u_ecef = d[ok] / rng[ok, None]
    enu = (r_enu @ u_ecef.T).T  # [n, 3] east,north,up
    g = np.column_stack((-enu[:, 0], -enu[:, 1], -enu[:, 2], np.ones(enu.shape[0])))
    try:
        q = np.linalg.inv(g.T @ g)
    except np.linalg.LinAlgError:
        return float("nan")
    h2 = float(q[0, 0] + q[1, 1])
    if not math.isfinite(h2) or h2 < 0.0:
        return float("nan")
    return math.sqrt(h2)


def n_los_and_hdop_chunk(
    rx_ecef: np.ndarray,
    sat_ecef: np.ndarray,
    los_vis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-point / per-epoch n_LOS counts and HDOP.

    Parameters
    ----------
    rx_ecef : (n_p, 3)
    sat_ecef : (n_t, n_sat, 3)
    los_vis : (n_p, n_t, n_sat)  True where sat is elevation/terrain-visible AND LOS

    Returns
    -------
    n_los : (n_p, n_t) float64
    hdop : (n_p, n_t) float64
    """
    rx = np.asarray(rx_ecef, dtype=np.float64)
    sat = np.asarray(sat_ecef, dtype=np.float64)
    mask = np.asarray(los_vis, dtype=bool)
    n_p, n_t, n_sat = mask.shape
    if rx.shape != (n_p, 3):
        raise ValueError(f"rx_ecef shape {rx.shape} != {(n_p, 3)}")
    if sat.shape != (n_t, n_sat, 3):
        raise ValueError(f"sat_ecef shape {sat.shape} != {(n_t, n_sat, 3)}")

    n_los = np.sum(mask, axis=2).astype(np.float64)
    hdop = np.full((n_p, n_t), np.nan, dtype=np.float64)
    for pi in range(n_p):
        rx_p = rx[pi]
        for ti in range(n_t):
            hdop[pi, ti] = hdop_from_los(rx_p, sat[ti], mask[pi, ti])
    return n_los, hdop
