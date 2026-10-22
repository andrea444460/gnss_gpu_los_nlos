"""Terrain horizon prefilter from DEM.

Builds a fast azimuth->minimum elevation horizon profile around a receiver
position and uses it to reject satellites hidden by terrain relief.

DEM vertical datum must match the receiver ECEF altitude (typically WGS84
ellipsoidal). AWS elevation tiles are orthometric: pass
``geoid_correction="egm96"`` (or a constant N) so sampled heights become
``h_ell = H + N``. Comparing raw H to ellipsoidal RX altitude makes the
terrain look ~N metres too low and yields ``terrain_blocked≈0``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from gnss_gpu.urban_signal_sim import _sat_elevation_azimuth, ecef_to_lla


@dataclass(frozen=True)
class HorizonConfig:
    max_distance_m: float = 20_000.0
    sample_step_m: float = 60.0
    azimuth_step_deg: float = 2.0
    cache_resolution_m: float = 50.0
    margin_deg: float = 0.0


class TerrainHorizonMask:
    """Compute and cache horizon masks from a DEM raster."""

    def __init__(
        self,
        dem_path: str | Path,
        config: HorizonConfig | None = None,
        *,
        geoid_correction=None,
    ):
        try:
            import rasterio
            from pyproj import Transformer
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "Terrain horizon prefilter requires rasterio and pyproj "
                "(pip install rasterio pyproj)."
            ) from exc

        from gnss_gpu.io.dem_native import _normalize_dem_crs, parse_geoid_correction_arg
        from gnss_gpu.io.plateau import _resolve_geoid_correction

        self._rasterio = rasterio
        self._Transformer = Transformer
        self.dem_path = str(dem_path)
        self.cfg = config or HorizonConfig()

        self.ds = rasterio.open(self.dem_path)
        self.crs = _normalize_dem_crs(self.ds.crs)
        self._to_dem_crs = Transformer.from_crs("EPSG:4326", self.crs, always_xy=True)
        self._cache: dict[tuple[int, int, int], np.ndarray] = {}

        # Accept CLI-style strings ("egm96", "48") or Plateau GeoidCorrection.
        if isinstance(geoid_correction, str):
            geoid_correction = parse_geoid_correction_arg(geoid_correction)
        self._geoid_fn: Callable[[float, float], float] | None
        self._geoid_label: str
        self._geoid_fn, self._geoid_label = _resolve_geoid_correction(geoid_correction)

        az_step_rad = math.radians(self.cfg.azimuth_step_deg)
        n_az = max(8, int(round((2.0 * math.pi) / max(1e-6, az_step_rad))))
        self.azimuth_bins_rad = np.linspace(0.0, 2.0 * math.pi, num=n_az, endpoint=False)
        self._margin_rad = math.radians(self.cfg.margin_deg)

    def _cache_key(self, rx_ecef: np.ndarray) -> tuple[int, int, int]:
        q = max(1.0, float(self.cfg.cache_resolution_m))
        x, y, z = np.asarray(rx_ecef, dtype=np.float64).reshape(3)
        return int(round(x / q)), int(round(y / q)), int(round(z / q))

    def _sample_dem_heights(self, lats_deg: np.ndarray, lons_deg: np.ndarray) -> np.ndarray:
        """Sample DEM; apply geoid so values are ellipsoidal metres when configured."""
        lats = np.asarray(lats_deg, dtype=np.float64).ravel()
        lons = np.asarray(lons_deg, dtype=np.float64).ravel()
        xs, ys = self._to_dem_crs.transform(lons, lats)
        pts = [(float(x), float(y)) for x, y in zip(xs, ys)]
        vals = np.array([v[0] for v in self.ds.sample(pts)], dtype=np.float64)
        vals[~np.isfinite(vals)] = np.nan
        if self._geoid_fn is not None:
            for i in range(vals.size):
                if np.isfinite(vals[i]):
                    vals[i] = float(vals[i]) + float(self._geoid_fn(float(lats[i]), float(lons[i])))
        return vals

    def _compute_horizon_profile_rad(self, rx_ecef: np.ndarray) -> np.ndarray:
        rx = np.asarray(rx_ecef, dtype=np.float64).reshape(3)
        lat_rad, lon_rad, alt_m = ecef_to_lla(float(rx[0]), float(rx[1]), float(rx[2]))
        lat_deg = math.degrees(lat_rad)
        lon_deg = math.degrees(lon_rad)

        dists = np.arange(
            float(self.cfg.sample_step_m),
            float(self.cfg.max_distance_m) + float(self.cfg.sample_step_m),
            float(self.cfg.sample_step_m),
            dtype=np.float64,
        )
        if dists.size == 0:
            return np.full_like(self.azimuth_bins_rad, -np.pi / 2.0, dtype=np.float64)

        cos_lat = max(1e-6, math.cos(lat_rad))
        out = np.full(self.azimuth_bins_rad.shape, -np.pi / 2.0, dtype=np.float64)

        for i, az in enumerate(self.azimuth_bins_rad):
            north = dists * math.cos(float(az))
            east = dists * math.sin(float(az))
            lats = lat_deg + north / 111_320.0
            lons = lon_deg + east / (111_320.0 * cos_lat)
            z_dem = self._sample_dem_heights(lats, lons)
            valid = np.isfinite(z_dem)
            if not np.any(valid):
                continue
            angles = np.arctan2(z_dem[valid] - alt_m, dists[valid])
            out[i] = float(np.max(angles))
        return out

    def horizon_profile_rad(self, rx_ecef: np.ndarray) -> np.ndarray:
        key = self._cache_key(rx_ecef)
        prof = self._cache.get(key)
        if prof is None:
            prof = self._compute_horizon_profile_rad(rx_ecef)
            self._cache[key] = prof
        return prof

    def terrain_visible_mask(self, rx_ecef: np.ndarray, sat_ecef: np.ndarray) -> np.ndarray:
        """Return bool mask: True where satellite is above terrain horizon."""
        sat = np.asarray(sat_ecef, dtype=np.float64).reshape(-1, 3)
        if sat.shape[0] == 0:
            return np.zeros((0,), dtype=bool)
        el_rad, az_rad = _sat_elevation_azimuth(np.asarray(rx_ecef, dtype=np.float64), sat)
        az = np.mod(az_rad, 2.0 * np.pi)
        prof = self.horizon_profile_rad(rx_ecef)
        n = prof.shape[0]
        idx = np.floor((az / (2.0 * np.pi)) * n).astype(int)
        idx = np.clip(idx, 0, n - 1)
        horizon_at_az = prof[idx]
        return el_rad >= (horizon_at_az + self._margin_rad)
