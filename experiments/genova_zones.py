"""Shared Genoa study-zone definitions (centro storico vs coastal strip)."""

from __future__ import annotations

import numpy as np

LAT_M = 111_320.0

DEFAULT_CENTRO_BBOX = dict(south=44.4038, west=8.9290, north=44.4108, east=8.9415)
DEFAULT_COASTAL_BUFFER_M = 150.0
DEFAULT_LON_BIN_M = 100.0
# Drop hinterland road samples north of this latitude (city + coast focus).
DEFAULT_MAP_CLIP_NORTH = 44.46
# Eastern (Levante) coastal strip for compact square figure (east of centro / port).
DEFAULT_COAST_LEVANTE_LON = 8.96
# Centro storico → Nervi corridor (waterfront roads incl. Corso Italia / Via Cinque Maggio).
DEFAULT_COAST_CORRIDOR_WEST_LON = DEFAULT_CENTRO_BBOX["west"]
DEFAULT_COAST_CORRIDOR_EAST_LON = 9.035
COAST_CORRIDOR_EXTRA_STREET_TERMS: tuple[str, ...] = (
    "corso italia",
    "via cinque maggio",
)


def filter_map_rows(rows: list[dict], *, north_max: float | None = DEFAULT_MAP_CLIP_NORTH) -> list[dict]:
    if north_max is None:
        return rows
    return [r for r in rows if float(r["lat_deg"]) <= float(north_max)]


def filter_coast_levante(rows: list[dict], *, min_lon: float = DEFAULT_COAST_LEVANTE_LON) -> list[dict]:
    return [r for r in rows if float(r["lon_deg"]) >= float(min_lon)]


def _matches_corridor_extra_street(name: str) -> bool:
    n = str(name).lower()
    return any(term in n for term in COAST_CORRIDOR_EXTRA_STREET_TERMS)


def filter_coast_corridor_rows(
    rows: list[dict],
    zones: np.ndarray,
    *,
    centro_bbox: dict[str, float] = DEFAULT_CENTRO_BBOX,
    west_lon: float = DEFAULT_COAST_CORRIDOR_WEST_LON,
    east_lon: float = DEFAULT_COAST_CORRIDOR_EAST_LON,
) -> list[dict]:
    """Coastal corridor: ``costiera`` (150 m buffer) plus named arterials in range.

    Extra streets (Corso Italia, Via Cinque Maggio) are unioned in without widening
    the global coastal buffer.
    """
    if len(rows) != len(zones):
        raise ValueError("rows and zones length mismatch")

    seen: set[tuple[int, int]] = set()
    out: list[dict] = []
    bb = centro_bbox
    for r, z in zip(rows, zones):
        lon = float(r["lon_deg"])
        lat = float(r["lat_deg"])
        if lon < float(west_lon) or lon > float(east_lon):
            continue
        if (
            bb["south"] <= lat <= bb["north"]
            and bb["west"] <= lon <= bb["east"]
        ):
            continue
        name = str(r.get("name", ""))
        if z != "costiera" and not _matches_corridor_extra_street(name):
            continue
        key = (round(lat * 1e6), round(lon * 1e6))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def _in_centro_mask(
    lats: np.ndarray,
    lons: np.ndarray,
    centro_bbox: dict[str, float],
) -> np.ndarray:
    return (
        (lats >= centro_bbox["south"])
        & (lats <= centro_bbox["north"])
        & (lons >= centro_bbox["west"])
        & (lons <= centro_bbox["east"])
    )


def build_southern_envelope(
    lons: np.ndarray,
    lats: np.ndarray,
    *,
    centro_bbox: dict[str, float] | None = None,
    lon_bin_m: float = DEFAULT_LON_BIN_M,
    smooth_bins: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Southernmost road latitude per longitude bin (proxy for the waterfront)."""
    mask = np.ones(len(lons), dtype=bool)
    if centro_bbox is not None:
        mask &= ~_in_centro_mask(lats, lons, centro_bbox)

    lons_u = lons[mask]
    lats_u = lats[mask]
    if len(lons_u) == 0:
        raise ValueError("no points available to build southern envelope")

    cos_lat = np.cos(np.radians(float(np.median(lats_u))))
    lon_bin_deg = max(lon_bin_m / (LAT_M * cos_lat), 1e-6)
    lon_min = float(lons_u.min())
    idx = np.floor((lons_u - lon_min) / lon_bin_deg).astype(np.int64)
    n_bins = int(idx.max()) + 1

    bin_lons: list[float] = []
    bin_lats: list[float] = []
    for b in range(n_bins):
        m = idx == b
        if not np.any(m):
            continue
        bin_lons.append(lon_min + (b + 0.5) * lon_bin_deg)
        bin_lats.append(float(lats_u[m].min()))

    env_lon = np.asarray(bin_lons, dtype=np.float64)
    env_lat = np.asarray(bin_lats, dtype=np.float64)
    order = np.argsort(env_lon)
    env_lon = env_lon[order]
    env_lat = env_lat[order]

    if smooth_bins > 1 and len(env_lat) >= smooth_bins:
        half = smooth_bins // 2
        smoothed = np.empty_like(env_lat)
        for i in range(len(env_lat)):
            lo = max(0, i - half)
            hi = min(len(env_lat), i + half + 1)
            smoothed[i] = env_lat[lo:hi].min()
        env_lat = smoothed

    env_lat = _clamp_envelope_spikes(env_lat, max_step_m=180.0)
    return env_lon, env_lat


def _clamp_envelope_spikes(env_lat: np.ndarray, *, max_step_m: float) -> np.ndarray:
    """Limit northward jumps between adjacent bins (inland outliers without waterfront roads)."""
    out = env_lat.copy()
    max_step = max_step_m / LAT_M
    for i in range(1, len(out)):
        if out[i] - out[i - 1] > max_step:
            out[i] = out[i - 1] + max_step * 0.15
    for i in range(len(out) - 2, -1, -1):
        if out[i] - out[i + 1] > max_step:
            out[i] = out[i + 1] + max_step * 0.15
    return out


def landward_dist_from_envelope_m(
    lats: np.ndarray,
    lons: np.ndarray,
    env_lon: np.ndarray,
    env_lat: np.ndarray,
) -> np.ndarray:
    """Distance north from the southern envelope, in metres (Genoa: sea is south)."""
    coast_lat = np.interp(lons, env_lon, env_lat)
    return (lats - coast_lat) * LAT_M


def classify_zones(
    lats: np.ndarray,
    lons: np.ndarray,
    *,
    centro_bbox: dict[str, float],
    coastal_buffer_m: float,
    lon_bin_m: float = DEFAULT_LON_BIN_M,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    zones = np.full(len(lats), "other", dtype=object)
    in_centro = _in_centro_mask(lats, lons, centro_bbox)
    zones[in_centro] = "centro_storico"

    env_lon, env_lat = build_southern_envelope(
        lons, lats, centro_bbox=centro_bbox, lon_bin_m=lon_bin_m,
    )
    landward_m = landward_dist_from_envelope_m(lats, lons, env_lon, env_lat)
    valid_lon = (lons >= env_lon.min()) & (lons <= env_lon.max())
    coastal = (
        valid_lon
        & (landward_m >= -5.0)
        & (landward_m <= coastal_buffer_m)
        & ~in_centro
    )
    zones[coastal] = "costiera"
    return zones, (env_lon, env_lat)
