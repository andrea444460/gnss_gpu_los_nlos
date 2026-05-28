"""OSM building corridor fetch + mesh conversion utilities.

This module fetches OpenStreetMap building footprints from Overpass, builds an
extruded triangle mesh, and converts vertices to ECEF coordinates for LOS/NLOS
ray tracing.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import requests

from gnss_gpu.io.urbannav import UrbanNavLoader
from gnss_gpu.urban_signal_sim import ecef_to_lla


WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = 2.0 * WGS84_F - WGS84_F * WGS84_F
OVERPASS_URL = "https://overpass-api.de/api/interpreter"


@dataclass(frozen=True)
class BBox:
    south: float
    west: float
    north: float
    east: float

    def as_tuple(self) -> tuple[float, float, float, float]:
        return self.south, self.west, self.north, self.east

    def to_overpass(self) -> str:
        s, w, n, e = self.as_tuple()
        return f"{s:.7f},{w:.7f},{n:.7f},{e:.7f}"


def _lla_deg_to_ecef(lat_deg: float, lon_deg: float, alt_m: float) -> np.ndarray:
    lat = math.radians(float(lat_deg))
    lon = math.radians(float(lon_deg))
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    x = (n + float(alt_m)) * cos_lat * cos_lon
    y = (n + float(alt_m)) * cos_lat * sin_lon
    z = (n * (1.0 - WGS84_E2) + float(alt_m)) * sin_lat
    return np.array([x, y, z], dtype=np.float64)


def _parse_float_from_tag(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    # OSM values are often like "12", "12.5", "12 m", "12;14".
    text = text.replace(",", ".").replace("m", " ")
    for token in text.replace(";", " ").split():
        try:
            return float(token)
        except ValueError:
            continue
    return None


def infer_building_height_m(
    tags: dict[str, object] | None,
    *,
    levels_height_m: float = 3.0,
    default_height_m: float = 10.0,
) -> float:
    h, _source = _infer_building_height_and_source(
        tags,
        levels_height_m=levels_height_m,
        default_height_m=default_height_m,
    )
    return h


def _infer_building_height_and_source(
    tags: dict[str, object] | None,
    *,
    levels_height_m: float = 3.0,
    default_height_m: float = 10.0,
) -> tuple[float, str]:
    tags = tags or {}
    h = _parse_float_from_tag(tags.get("height"))
    if h is not None and h > 0.0:
        return h, "height"
    lv = _parse_float_from_tag(tags.get("building:levels"))
    if lv is not None and lv > 0.0:
        return lv * float(levels_height_m), "levels"
    return float(default_height_m), "default"


def load_trajectory_latlon(reference_csv: Path) -> np.ndarray:
    """Load trajectory from UrbanNav-style CSV and return [N,2] lat/lon degrees."""
    loader = UrbanNavLoader(reference_csv.parent)
    try:
        _times, ecef = loader.load_ground_truth(filepath=reference_csv)
        out = []
        for i in range(ecef.shape[0]):
            lat, lon, _alt = ecef_to_lla(float(ecef[i, 0]), float(ecef[i, 1]), float(ecef[i, 2]))
            out.append([math.degrees(lat), math.degrees(lon)])
        if out:
            return np.asarray(out, dtype=np.float64)
    except (ValueError, FileNotFoundError, OSError):
        pass

    # Fallback to lat/lon headers or gt.csv [ts,lat,lon,alt]
    with open(reference_csv, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if rows:
        keys = {k.strip(): k for k in rows[0].keys()}
        lat_key = next(
            (
                keys[k]
                for k in keys
                if k.lower() in {"lat", "latitude", "latitude (deg)", "lat_deg", "latitudine"}
            ),
            None,
        )
        lon_key = next(
            (
                keys[k]
                for k in keys
                if k.lower() in {"lon", "longitude", "longitude (deg)", "lng", "lon_deg", "longitudine"}
            ),
            None,
        )
        if lat_key and lon_key:
            ll = []
            for r in rows:
                try:
                    ll.append([float(r[lat_key]), float(r[lon_key])])
                except (TypeError, ValueError, KeyError):
                    continue
            if ll:
                return np.asarray(ll, dtype=np.float64)

    with open(reference_csv, newline="", encoding="utf-8") as fh:
        raw = list(csv.reader(fh))
    ll = []
    for row in raw:
        if len(row) < 3:
            continue
        try:
            ll.append([float(row[1]), float(row[2])])
        except ValueError:
            continue
    if not ll:
        raise ValueError(f"Could not parse trajectory lat/lon from {reference_csv}")
    return np.asarray(ll, dtype=np.float64)


def _meters_to_lat_deg(meters: float) -> float:
    return float(meters) / 111_320.0


def _meters_to_lon_deg(meters: float, lat_deg: float) -> float:
    cos_lat = max(1e-6, math.cos(math.radians(lat_deg)))
    return float(meters) / (111_320.0 * cos_lat)


def build_corridor_tiles(
    latlon_deg: np.ndarray,
    *,
    buffer_m: float,
    tile_step_m: float = 500.0,
) -> list[BBox]:
    """Create overlapping bbox tiles approximating a buffered trajectory corridor."""
    if latlon_deg.ndim != 2 or latlon_deg.shape[1] != 2:
        raise ValueError("latlon_deg must have shape [N,2]")
    if len(latlon_deg) == 0:
        return []

    lat_step = _meters_to_lat_deg(tile_step_m)
    key_set: set[tuple[int, int]] = set()
    for lat, lon in latlon_deg:
        lon_step = _meters_to_lon_deg(tile_step_m, float(lat))
        lat_idx = int(math.floor(float(lat) / max(lat_step, 1e-12)))
        lon_idx = int(math.floor(float(lon) / max(lon_step, 1e-12)))
        key_set.add((lat_idx, lon_idx))

    boxes: list[BBox] = []
    for lat_idx, lon_idx in sorted(key_set):
        lat_c = (lat_idx + 0.5) * lat_step
        lon_step = _meters_to_lon_deg(tile_step_m, lat_c)
        lon_c = (lon_idx + 0.5) * lon_step
        dlat = _meters_to_lat_deg(buffer_m + 0.5 * tile_step_m)
        dlon = _meters_to_lon_deg(buffer_m + 0.5 * tile_step_m, lat_c)
        boxes.append(BBox(lat_c - dlat, lon_c - dlon, lat_c + dlat, lon_c + dlon))
    return boxes


def build_overpass_query(bbox: BBox) -> str:
    bb = bbox.to_overpass()
    return (
        f"[out:json][timeout:60];"
        f"(way[\"building\"]({bb});relation[\"building\"]({bb}););"
        f"out body geom;"
    )


def fetch_buildings_overpass(
    bboxes: Iterable[BBox],
    *,
    timeout_s: int = 120,
    user_agent: str = "gnss_gpu_osm_fetch/1.0",
) -> list[dict]:
    """Fetch building ways/relations from Overpass and deduplicate by (type,id)."""
    dedup: dict[tuple[str, int], dict] = {}
    session = requests.Session()
    headers = {"Content-Type": "application/x-www-form-urlencoded", "User-Agent": user_agent}
    for bbox in bboxes:
        query = build_overpass_query(bbox)
        resp = session.post(OVERPASS_URL, data=query, headers=headers, timeout=timeout_s)
        resp.raise_for_status()
        payload = resp.json()
        for el in payload.get("elements", []):
            typ = str(el.get("type", ""))
            eid = el.get("id", None)
            if typ not in {"way", "relation"} or not isinstance(eid, int):
                continue
            dedup[(typ, eid)] = el
    return list(dedup.values())


def _ring_area_xy(ring_xy: np.ndarray) -> float:
    x = ring_xy[:, 0]
    y = ring_xy[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _remove_duplicate_last(ring: np.ndarray) -> np.ndarray:
    if len(ring) >= 2 and np.allclose(ring[0], ring[-1]):
        return ring[:-1]
    return ring


def _earclip_indices(poly_xy: np.ndarray) -> list[tuple[int, int, int]]:
    """Triangulate simple polygon using ear clipping."""
    n = poly_xy.shape[0]
    if n < 3:
        return []
    verts = list(range(n))
    tris: list[tuple[int, int, int]] = []
    ccw = _ring_area_xy(poly_xy) > 0

    def is_convex(a: int, b: int, c: int) -> bool:
        pa, pb, pc = poly_xy[a], poly_xy[b], poly_xy[c]
        cross = (pb[0] - pa[0]) * (pc[1] - pa[1]) - (pb[1] - pa[1]) * (pc[0] - pa[0])
        return cross > 0 if ccw else cross < 0

    def point_in_tri(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
        v0 = c - a
        v1 = b - a
        v2 = p - a
        den = v0[0] * v1[1] - v1[0] * v0[1]
        if abs(den) < 1e-12:
            return False
        u = (v2[0] * v1[1] - v1[0] * v2[1]) / den
        v = (v0[0] * v2[1] - v2[0] * v0[1]) / den
        return u >= 0.0 and v >= 0.0 and (u + v) <= 1.0

    guard = 0
    while len(verts) > 3 and guard < 10_000:
        guard += 1
        ear_found = False
        m = len(verts)
        for i in range(m):
            ia = verts[(i - 1) % m]
            ib = verts[i]
            ic = verts[(i + 1) % m]
            if not is_convex(ia, ib, ic):
                continue
            a, b, c = poly_xy[ia], poly_xy[ib], poly_xy[ic]
            blocked = False
            for j in verts:
                if j in (ia, ib, ic):
                    continue
                if point_in_tri(poly_xy[j], a, b, c):
                    blocked = True
                    break
            if blocked:
                continue
            tris.append((ia, ib, ic))
            del verts[i]
            ear_found = True
            break
        if not ear_found:
            break

    if len(verts) == 3:
        tris.append((verts[0], verts[1], verts[2]))
    return tris


def _way_geometry_to_ring_latlon(geom: list[dict]) -> np.ndarray | None:
    pts = []
    for g in geom:
        try:
            pts.append([float(g["lat"]), float(g["lon"])])
        except (KeyError, TypeError, ValueError):
            return None
    ring = np.asarray(pts, dtype=np.float64)
    ring = _remove_duplicate_last(ring)
    if ring.shape[0] < 3:
        return None
    return ring


def _relation_outer_rings(rel: dict) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for mem in rel.get("members", []):
        if mem.get("type") != "way":
            continue
        if mem.get("role") not in {"outer", ""}:
            continue
        geom = mem.get("geometry")
        if not isinstance(geom, list):
            continue
        ring = _way_geometry_to_ring_latlon(geom)
        if ring is not None:
            out.append(ring)
    return out


def _ring_to_xy_m(ring_latlon: np.ndarray, ref_lat_deg: float, ref_lon_deg: float) -> np.ndarray:
    x = (ring_latlon[:, 1] - ref_lon_deg) * 111_320.0 * math.cos(math.radians(ref_lat_deg))
    y = (ring_latlon[:, 0] - ref_lat_deg) * 111_320.0
    return np.column_stack([x, y])


def buildings_to_triangles_ecef(
    elements: Iterable[dict],
    *,
    levels_height_m: float = 3.0,
    default_height_m: float = 10.0,
    base_alt_m: float = 0.0,
    stats: dict[str, int] | None = None,
) -> np.ndarray:
    """Convert OSM elements into extruded ECEF triangle mesh [N,3,3]."""
    elems = list(elements)
    if not elems:
        return np.empty((0, 3, 3), dtype=np.float64)

    # Reference for planar triangulation.
    all_lats = []
    all_lons = []
    for e in elems:
        geom = e.get("geometry")
        if isinstance(geom, list):
            for g in geom:
                if "lat" in g and "lon" in g:
                    all_lats.append(float(g["lat"]))
                    all_lons.append(float(g["lon"]))
    if not all_lats:
        return np.empty((0, 3, 3), dtype=np.float64)
    ref_lat = float(np.mean(all_lats))
    ref_lon = float(np.mean(all_lons))

    tris: list[np.ndarray] = []

    if stats is not None:
        stats.clear()
        stats.update(
            {
                "total_elements_considered": 0,
                "meshed_buildings": 0,
                "height_defined_count": 0,
                "height_generated_count": 0,
                "generated_from_levels_count": 0,
                "generated_default_count": 0,
                "skipped_no_valid_geometry_count": 0,
            }
        )

    def add_extruded_ring(ring_latlon: np.ndarray, height_m: float) -> bool:
        ring_xy = _ring_to_xy_m(ring_latlon, ref_lat, ref_lon)
        tri_idx = _earclip_indices(ring_xy)
        if not tri_idx:
            return False
        bottom = np.asarray(
            [_lla_deg_to_ecef(float(lat), float(lon), base_alt_m) for lat, lon in ring_latlon],
            dtype=np.float64,
        )
        top = np.asarray(
            [_lla_deg_to_ecef(float(lat), float(lon), base_alt_m + height_m) for lat, lon in ring_latlon],
            dtype=np.float64,
        )

        # Roof triangles.
        for ia, ib, ic in tri_idx:
            tris.append(np.stack([top[ia], top[ib], top[ic]], axis=0))
        # Bottom triangles (reverse winding).
        for ia, ib, ic in tri_idx:
            tris.append(np.stack([bottom[ic], bottom[ib], bottom[ia]], axis=0))

        # Wall quads -> two triangles per edge.
        n = ring_latlon.shape[0]
        for i in range(n):
            j = (i + 1) % n
            tris.append(np.stack([bottom[i], bottom[j], top[j]], axis=0))
            tris.append(np.stack([bottom[i], top[j], top[i]], axis=0))
        return True

    for el in elems:
        tags = el.get("tags", {}) if isinstance(el.get("tags"), dict) else {}
        h, source = _infer_building_height_and_source(
            tags,
            levels_height_m=levels_height_m,
            default_height_m=default_height_m,
        )
        if h <= 0.0:
            continue

        if stats is not None:
            stats["total_elements_considered"] += 1

        has_valid_geometry = False
        if el.get("type") == "way" and isinstance(el.get("geometry"), list):
            ring = _way_geometry_to_ring_latlon(el["geometry"])
            if ring is not None:
                has_valid_geometry = add_extruded_ring(ring, h) or has_valid_geometry
        elif el.get("type") == "relation":
            for ring in _relation_outer_rings(el):
                has_valid_geometry = add_extruded_ring(ring, h) or has_valid_geometry

        if stats is not None:
            if has_valid_geometry:
                stats["meshed_buildings"] += 1
                if source == "height":
                    stats["height_defined_count"] += 1
                else:
                    stats["height_generated_count"] += 1
                    if source == "levels":
                        stats["generated_from_levels_count"] += 1
                    else:
                        stats["generated_default_count"] += 1
            else:
                stats["skipped_no_valid_geometry_count"] += 1

    if not tris:
        return np.empty((0, 3, 3), dtype=np.float64)
    return np.asarray(tris, dtype=np.float64)
