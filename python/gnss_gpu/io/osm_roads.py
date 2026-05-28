"""OSM roads fetch and road-point sampling utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import requests


OVERPASS_URL = "https://overpass-api.de/api/interpreter"


@dataclass(frozen=True)
class BBox:
    south: float
    west: float
    north: float
    east: float

    def to_overpass(self) -> str:
        return f"{self.south:.7f},{self.west:.7f},{self.north:.7f},{self.east:.7f}"


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp * 0.5) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl * 0.5) ** 2
    return 2.0 * r * math.asin(min(1.0, math.sqrt(a)))


def _interpolate_latlon(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return a + (b - a) * float(t)


def build_roads_overpass_query(
    bbox: BBox,
    *,
    include_pedestrian: bool = False,
) -> str:
    """Build Overpass query for roads in bbox."""
    bb = bbox.to_overpass()
    if include_pedestrian:
        filt = "way[\"highway\"]"
    else:
        # Keep road network focused on navigation use-cases.
        filt = (
            "way[\"highway\"]"
            "[\"highway\"!=\"footway\"]"
            "[\"highway\"!=\"path\"]"
            "[\"highway\"!=\"steps\"]"
            "[\"highway\"!=\"cycleway\"]"
            "[\"highway\"!=\"pedestrian\"]"
            "[\"highway\"!=\"corridor\"]"
            "[\"highway\"!=\"track\"]"
        )
    return f"[out:json][timeout:60];({filt}({bb}););out body geom;"


def fetch_roads_overpass(
    bbox: BBox,
    *,
    include_pedestrian: bool = False,
    timeout_s: int = 120,
    user_agent: str = "gnss_gpu_osm_roads/1.0",
) -> list[dict]:
    """Fetch road ways from Overpass for a bbox."""
    query = build_roads_overpass_query(bbox, include_pedestrian=include_pedestrian)
    resp = requests.post(
        OVERPASS_URL,
        data=query,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": user_agent,
            "Accept": "application/json,text/plain,*/*",
        },
        timeout=timeout_s,
    )
    resp.raise_for_status()
    payload = resp.json()
    dedup: dict[int, dict] = {}
    for el in payload.get("elements", []):
        if el.get("type") != "way":
            continue
        eid = el.get("id")
        if isinstance(eid, int):
            dedup[eid] = el
    return list(dedup.values())


def way_geometry_to_line_latlon(geom: list[dict]) -> np.ndarray | None:
    pts = []
    for g in geom:
        try:
            pts.append([float(g["lat"]), float(g["lon"])])
        except (TypeError, ValueError, KeyError):
            return None
    arr = np.asarray(pts, dtype=np.float64)
    if arr.shape[0] < 2:
        return None
    return arr


def sample_line_every_meters(line_latlon: np.ndarray, step_m: float) -> np.ndarray:
    """Sample points along polyline approximately every step_m."""
    if line_latlon.ndim != 2 or line_latlon.shape[1] != 2:
        raise ValueError("line_latlon must have shape [N,2]")
    if line_latlon.shape[0] < 2:
        return line_latlon.copy()
    step = max(1.0, float(step_m))
    out = [line_latlon[0]]
    carry = 0.0
    for i in range(line_latlon.shape[0] - 1):
        a = line_latlon[i]
        b = line_latlon[i + 1]
        seg_m = _haversine_m(float(a[0]), float(a[1]), float(b[0]), float(b[1]))
        if seg_m <= 0.0:
            continue
        dist = step - carry
        while dist < seg_m:
            t = dist / seg_m
            out.append(_interpolate_latlon(a, b, t))
            dist += step
        carry = seg_m - (dist - step)
        if carry >= step:
            carry = 0.0
    if np.linalg.norm(out[-1] - line_latlon[-1]) > 1e-12:
        out.append(line_latlon[-1])
    return np.asarray(out, dtype=np.float64)


def sample_road_points(
    roads: Iterable[dict],
    *,
    step_m: float = 25.0,
    dedup_decimals: int = 6,
) -> list[dict]:
    """Return sampled points list from road way elements."""
    points: list[dict] = []
    seen: set[tuple[int, int]] = set()
    for way in roads:
        geom = way.get("geometry")
        if not isinstance(geom, list):
            continue
        line = way_geometry_to_line_latlon(geom)
        if line is None:
            continue
        sampled = sample_line_every_meters(line, step_m=step_m)
        tags = way.get("tags", {}) if isinstance(way.get("tags"), dict) else {}
        for p in sampled:
            k = (
                int(round(float(p[0]) * (10**dedup_decimals))),
                int(round(float(p[1]) * (10**dedup_decimals))),
            )
            if k in seen:
                continue
            seen.add(k)
            points.append(
                {
                    "way_id": int(way.get("id", -1)),
                    "highway": str(tags.get("highway", "")),
                    "name": str(tags.get("name", "")),
                    "lat_deg": float(p[0]),
                    "lon_deg": float(p[1]),
                }
            )
    return points


def split_bbox_into_tiles(bbox: BBox, *, tile_size_m: float) -> list[BBox]:
    """Split bbox into smaller bbox tiles with approx tile_size_m dimensions."""
    if tile_size_m <= 0:
        return [bbox]
    lat_c = 0.5 * (bbox.south + bbox.north)
    dlat = float(tile_size_m) / 111_320.0
    dlon = float(tile_size_m) / (111_320.0 * max(1e-6, math.cos(math.radians(lat_c))))
    lats = np.arange(bbox.south, bbox.north + dlat, dlat)
    lons = np.arange(bbox.west, bbox.east + dlon, dlon)
    tiles: list[BBox] = []
    for i in range(len(lats) - 1):
        for j in range(len(lons) - 1):
            tiles.append(
                BBox(
                    south=float(lats[i]),
                    west=float(lons[j]),
                    north=float(min(lats[i + 1], bbox.north)),
                    east=float(min(lons[j + 1], bbox.east)),
                )
            )
    return tiles
