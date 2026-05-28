"""CUDA terrain/elevation preprocess helpers."""

from __future__ import annotations

import numpy as np

try:
    from gnss_gpu._gnss_gpu_terrain import terrain_prefilter_batch as _terrain_prefilter_batch

    _HAS_TERRAIN_CUDA = True
except ImportError:
    _HAS_TERRAIN_CUDA = False


def has_terrain_cuda() -> bool:
    return _HAS_TERRAIN_CUDA


def terrain_prefilter_batch(
    rx_ecef: np.ndarray,
    sat_ecef: np.ndarray,
    dem_h: np.ndarray,
    *,
    dem_lat0_deg: float,
    dem_lon0_deg: float,
    dem_lat_step_deg: float,
    dem_lon_step_deg: float,
    max_distance_m: float,
    sample_step_m: float,
    elevation_mask_rad: float,
    margin_rad: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run CUDA preprocess and return boolean masks + masked satellites."""
    if not _HAS_TERRAIN_CUDA:
        raise RuntimeError("gnss_gpu terrain CUDA module not available; rebuild with CUDA bindings.")

    rx = np.asarray(rx_ecef, dtype=np.float64, order="C").reshape(-1, 3)
    sat = np.asarray(sat_ecef, dtype=np.float64, order="C")
    if sat.ndim != 3 or sat.shape[2] != 3:
        raise ValueError(f"sat_ecef must have shape [N,n_sat,3], got {sat.shape}")
    if sat.shape[0] != rx.shape[0]:
        raise ValueError("rx_ecef and sat_ecef must share leading N.")

    dem = np.asarray(dem_h, dtype=np.float32, order="C")
    if dem.ndim != 2:
        raise ValueError(f"dem_h must have shape [H,W], got {dem.shape}")

    vis_i, tblk_i, tblk_vis_i, sat_masked = _terrain_prefilter_batch(
        rx,
        sat,
        dem,
        float(dem_lat0_deg),
        float(dem_lon0_deg),
        float(dem_lat_step_deg),
        float(dem_lon_step_deg),
        float(max_distance_m),
        float(sample_step_m),
        float(elevation_mask_rad),
        float(margin_rad),
    )
    visible = np.asarray(vis_i, dtype=bool)
    terrain_blocked = np.asarray(tblk_i, dtype=bool)
    terrain_blocked_visible = np.asarray(tblk_vis_i, dtype=bool)
    sat_masked = np.asarray(sat_masked, dtype=np.float64)
    return visible, terrain_blocked, terrain_blocked_visible, sat_masked
