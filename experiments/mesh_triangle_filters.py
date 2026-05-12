"""Mesh triangle filters for stadium / 3D Tiles triangle soups (ECEF ``[N,3,3]``)."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_PYTHON = Path(__file__).resolve().parents[1] / "python"
if _REPO_PYTHON.is_dir() and str(_REPO_PYTHON) not in sys.path:
    sys.path.insert(0, str(_REPO_PYTHON))

import numpy as np

from gnss_gpu.urban_signal_sim import ecef_to_lla


def triangle_areas_m2(tri: np.ndarray) -> np.ndarray:
    """Triangle areas [m²] for array shaped ``(n, 3, 3)``."""
    t = np.asarray(tri, dtype=np.float64).reshape(-1, 3, 3)
    e0 = t[:, 1, :] - t[:, 0, :]
    e1 = t[:, 2, :] - t[:, 0, :]
    return 0.5 * np.linalg.norm(np.cross(e0, e1), axis=1)


def remove_bottom_closure_slab(
    tri: np.ndarray,
    *,
    alt_quantile: float = 0.03,
    min_area_m2: float = 120.0,
    alt_slack_m: float = 0.05,
) -> tuple[np.ndarray, int]:
    """Drop large triangles whose centroid sits in the lowest ellipsoidal-height band.

    Many georeferenced glTF / b3dm stadium shells include a watertight **bottom cap** (few large
    facets). Ray tracing then intersects that slab (fake bounce / NLOS). This heuristic keeps
    small triangles (stands, tessellated pitch) and removes huge low-altitude plates.

    Parameters
    ----------
    alt_quantile
        Centroid ellipsoidal heights below this quantile are the low band (default 3%).
    min_area_m2
        Only triangles with area ≥ this are candidates (default 120 m²).
    alt_slack_m
        Treat ``<= threshold + slack'' as low band (numerical noise).

    Returns
    -------
    (tri_kept, n_removed)
    """
    tri = np.asarray(tri, dtype=np.float64)
    if tri.ndim != 3 or tri.shape[1:] != (3, 3):
        raise ValueError(f"expected tri shape (n, 3, 3), got {tri.shape}")
    n = tri.shape[0]
    if n == 0:
        return tri, 0

    centroids = tri.mean(axis=1)
    alts = np.empty(n, dtype=np.float64)
    for i in range(n):
        _lat, _lon, h = ecef_to_lla(
            float(centroids[i, 0]),
            float(centroids[i, 1]),
            float(centroids[i, 2]),
        )
        alts[i] = float(h)

    areas = triangle_areas_m2(tri)
    aq = float(np.clip(float(alt_quantile), 0.001, 0.25))
    ath = float(np.quantile(alts, aq))
    drop = (alts <= ath + float(alt_slack_m)) & (areas >= float(min_area_m2))
    n_drop = int(np.sum(drop))
    if n_drop == 0:
        return tri, 0
    keep = ~drop
    if not np.any(keep):
        raise ValueError(
            "bottom-slab filter removed all triangles; loosen --bottom-alt-quantile or "
            "--bottom-min-area-m2"
        )
    return tri[keep], n_drop
