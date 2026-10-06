"""Sample elevations directly from a GeoTIFF DEM in its native CRS (no WGS84 reproject)."""

from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import rasterio
from pyproj import Transformer

from gnss_gpu.io.plateau import GeoidCorrection, _resolve_geoid_correction


def parse_geoid_correction_arg(value: str) -> GeoidCorrection:
    """Parse CLI ``--geoid-correction``: ``none``, ``egm96``, or a constant float [m]."""
    if value is None or str(value).strip().lower() in ("none", ""):
        return None
    if str(value).lower() == "egm96":
        return "egm96"
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(
            f"geoid_correction: expected 'none', 'egm96', or float; got {value!r}"
        ) from exc

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = 2.0 * WGS84_F - WGS84_F * WGS84_F


def lla_deg_to_ecef(lat_deg: float, lon_deg: float, alt_m: float) -> np.ndarray:
    lat = math.radians(float(lat_deg))
    lon = math.radians(float(lon_deg))
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    x = (n + alt_m) * cos_lat * cos_lon
    y = (n + alt_m) * cos_lat * sin_lon
    z = (n * (1.0 - WGS84_E2) + alt_m) * sin_lat
    return np.asarray([x, y, z], dtype=np.float64)


class NativeDemSampler:
    """Read and sample a DEM GeoTIFF without reprojecting the elevation grid.

    Satellite/radar DEMs (SRTM, Copernicus, TanDEM-X, …) already carry WGS84
    ellipsoidal heights — use ``geoid_correction=None`` (default).

    Orthometric DEMs (national LiDAR, PLATEAU, …) need ``geoid_correction``
    (e.g. ``"egm96"``) so ``h_ell = H_orth + N(lat, lon)``.
    """

    def __init__(
        self,
        dem_path: str | Path,
        *,
        geoid_correction: GeoidCorrection = None,
    ) -> None:
        self.path = Path(dem_path)
        self._ds = rasterio.open(str(self.path))
        self.crs = self._ds.crs
        self.transform = self._ds.transform
        self.nodata = self._ds.nodata
        self.heights = np.asarray(self._ds.read(1), dtype=np.float64)
        self.rows, self.cols = self.heights.shape
        self._to_dem = Transformer.from_crs("EPSG:4326", self.crs, always_xy=True)
        self._to_wgs84 = Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True)
        self._geoid_fn, self.geoid_label = _resolve_geoid_correction(geoid_correction)
        valid = np.isfinite(self.heights)
        if self.nodata is not None and math.isfinite(float(self.nodata)):
            valid &= self.heights != float(self.nodata)
        valid &= self.heights > -200.0
        if not np.any(valid):
            raise ValueError(f"DEM has no valid samples: {self.path}")
        self.fallback_ortho_m = float(np.median(self.heights[valid]))
        self.min_ortho_m = float(np.min(self.heights[valid]))
        self.max_ortho_m = float(np.max(self.heights[valid]))
        self.min_m = self.min_ortho_m
        self.max_m = self.max_ortho_m
        self.fallback_m = self.fallback_ortho_m
        if self._geoid_fn is not None:
            south, west, north, east = rasterio.transform.array_bounds(
                self.rows, self.cols, self.transform
            )
            n_vals = []
            for x, y in ((west, south), (east, south), (west, north), (east, north)):
                lon, lat = self._to_wgs84.transform(x, y)
                n_vals.append(float(self._geoid_fn(float(lat), float(lon))))
            n_med = float(np.median(n_vals))
            self.min_m = float(self.min_ortho_m + min(n_vals))
            self.max_m = float(self.max_ortho_m + max(n_vals))
            self.fallback_m = float(self.fallback_ortho_m + n_med)

    def close(self) -> None:
        self._ds.close()

    def _sanitize_z_ortho(self, z: float) -> float:
        if self.nodata is not None and z == float(self.nodata):
            return self.fallback_ortho_m
        if not math.isfinite(z) or z <= -200.0:
            return self.fallback_ortho_m
        return float(z)

    def _to_ellipsoidal(self, lat_deg: float, lon_deg: float, h_ortho_m: float) -> float:
        h = float(h_ortho_m)
        if self._geoid_fn is not None:
            h += float(self._geoid_fn(float(lat_deg), float(lon_deg)))
        return h

    def sample(self, lat_deg: float, lon_deg: float) -> float:
        x, y = self._to_dem.transform(float(lon_deg), float(lat_deg))
        col_f, row_f = ~self.transform * (x, y)
        if not (math.isfinite(col_f) and math.isfinite(row_f)):
            return self.fallback_m
        if col_f < 0.0 or row_f < 0.0 or col_f >= (self.cols - 1) or row_f >= (self.rows - 1):
            return self.fallback_m
        c0 = int(math.floor(col_f))
        r0 = int(math.floor(row_f))
        c1 = c0 + 1
        r1 = r0 + 1
        fc = col_f - c0
        fr = row_f - r0
        z00 = self._sanitize_z_ortho(self.heights[r0, c0])
        z01 = self._sanitize_z_ortho(self.heights[r0, c1])
        z10 = self._sanitize_z_ortho(self.heights[r1, c0])
        z11 = self._sanitize_z_ortho(self.heights[r1, c1])
        z0 = (1.0 - fc) * z00 + fc * z01
        z1 = (1.0 - fc) * z10 + fc * z11
        h_ortho = float((1.0 - fr) * z0 + fr * z1)
        return self._to_ellipsoidal(lat_deg, lon_deg, h_ortho)

    def sample_many(self, lats_deg: np.ndarray, lons_deg: np.ndarray) -> np.ndarray:
        lats = np.asarray(lats_deg, dtype=np.float64).ravel()
        lons = np.asarray(lons_deg, dtype=np.float64).ravel()
        return np.asarray([self.sample(float(la), float(lo)) for la, lo in zip(lats, lons)], dtype=np.float64)

    def pixel_to_latlon(self, row: int, col: int) -> tuple[float, float]:
        x, y = self.transform * (float(col) + 0.5, float(row) + 0.5)
        lon, lat = self._to_wgs84.transform(x, y)
        return float(lat), float(lon)

    def to_triangles_ecef(self, *, cell_stride: int = 1) -> np.ndarray:
        stride = max(1, int(cell_stride))
        tris: list[np.ndarray] = []
        for r in range(0, self.rows - stride, stride):
            r1 = r + stride
            for c in range(0, self.cols - stride, stride):
                c1 = c + stride
                z00 = self._to_ellipsoidal(*self.pixel_to_latlon(r, c), self._sanitize_z_ortho(self.heights[r, c]))
                z01 = self._to_ellipsoidal(*self.pixel_to_latlon(r, c1), self._sanitize_z_ortho(self.heights[r, c1]))
                z10 = self._to_ellipsoidal(*self.pixel_to_latlon(r1, c), self._sanitize_z_ortho(self.heights[r1, c]))
                z11 = self._to_ellipsoidal(*self.pixel_to_latlon(r1, c1), self._sanitize_z_ortho(self.heights[r1, c1]))
                lat00, lon00 = self.pixel_to_latlon(r, c)
                lat01, lon01 = self.pixel_to_latlon(r, c1)
                lat10, lon10 = self.pixel_to_latlon(r1, c)
                lat11, lon11 = self.pixel_to_latlon(r1, c1)
                p00 = lla_deg_to_ecef(lat00, lon00, z00)
                p01 = lla_deg_to_ecef(lat01, lon01, z01)
                p10 = lla_deg_to_ecef(lat10, lon10, z10)
                p11 = lla_deg_to_ecef(lat11, lon11, z11)
                tris.append(np.stack([p00, p01, p11], axis=0))
                tris.append(np.stack([p00, p11, p10], axis=0))
        if not tris:
            raise ValueError("Native DEM ground mesh is empty.")
        return np.asarray(tris, dtype=np.float64)

    def summary(self) -> dict:
        south, west, north, east = rasterio.transform.array_bounds(self.rows, self.cols, self.transform)
        lat_sw, lon_sw = self._to_wgs84.transform(west, south)
        lat_ne, lon_ne = self._to_wgs84.transform(east, north)
        return {
            "path": str(self.path),
            "crs": str(self.crs),
            "rows": int(self.rows),
            "cols": int(self.cols),
            "south": float(min(lat_sw, lat_ne)),
            "north": float(max(lat_sw, lat_ne)),
            "west": float(min(lon_sw, lon_ne)),
            "east": float(max(lon_sw, lon_ne)),
            "min": self.min_m,
            "max": self.max_m,
            "fallback": self.fallback_m,
            "nodata": self.nodata,
            "geoid": self.geoid_label,
            "heightDatum": (
                "orthometric+geoid" if self._geoid_fn is not None else "satellite_ellipsoidal"
            ),
        }
