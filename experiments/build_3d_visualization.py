#!/usr/bin/env python3
"""Generate 3D LOS/NLOS geometry visualization + recording helper.

Creates a standalone HTML file using CesiumJS (free tier) that shows:
  - 3D globe with terrain + buildings (Cesium OSM Buildings)
  - Receiver trajectory (yellow line)
  - Per-epoch satellite rays (green=LOS, red=NLOS)
  - Animated flythrough

Then uses Playwright to record a video of the visualization.

Optional ``--filter-by-obs`` keeps only satellites that have a non-zero **code pseudorange**
(``C*``) in the rover **RINEX .obs** at the nearest epoch (GPS TOW match within ``--obs-match-tol-s``).
By default the generator also embeds **C/N₀** time series (``S*`` in the same ``.obs``) so clicking a ray
in the HTML viewer opens a chart with an epoch cursor (disable with ``--no-cnr-embed``).

Optional ``--viz-multipath`` adds polylines for first-order specular reflections
(satellite→bounce→receiver): **orange** when LOS also has a reflection sketch, **red** when the
satellite is NLOS and the reflection replaces the hidden direct ray (direct through-building stub
is omitted). With ephemeris batching + BVH, multipath uses ``compute_multipath_batch``
(one launch per chunk); otherwise per-epoch ``compute_multipath`` / CPU mesh fallback.

Optional ``--export-plateau-glb`` writes a **ROI-filtered** PLATEAU mesh next to the HTML
(``*_plateau.glb``). Vertices are ECEF offsets from the trajectory centroid; Cesium uses
``Matrix4.fromTranslation(pivotEcef)`` only (no ENU frame), matching the ray geometry.
Use ``--plateau-glb-radius-m`` / ``--plateau-glb-max-tris`` to control size, or
``--plateau-glb-full-mesh`` for all triangles (still capped by ``max-tris``). Serve over HTTP.

The receiver trajectory is sourced from UrbanNav. Satellite geometry uses the
**broadcast RINEX navigation** file next to the trajectory (``base.nav``) or
``--nav``; constellation blocks **G/R/E/J/C/I** from that file are parsed (SBAS omitted).
LOS/NLOS rays
are computed in Python against the **local PLATEAU CityGML mesh** (--plateau-dir).
``PlateauLoader`` applies orthometric→ellipsoidal height correction by default (see ``gnss_gpu.io.plateau``), so mesh and RX ECEF agree without extra mesh shifts; ``--export-plateau-glb`` exports the same geometry used for ray tracing.
By default satellites below **10° elevation** are excluded (``--elevation-mask-deg``); use ``0`` for horizon-only or ``-90`` to disable.
Optional ``--epoch-min-interval-s`` > 0 enforces a minimum GPS-time gap between consecutive viz epochs; default is off (evenly spaced trajectory rows).
Ephemeris positions are evaluated with ``Ephemeris.compute_batch`` in chunks (``--eph-batch-chunk``, default 64).
When the compiled BVH module exposes ``check_los_batch``, LOS intersection uses one CUDA launch per ephemeris chunk instead of per-epoch ``UrbanSignalSimulator.compute_epoch`` ray tests.
The HTML viewer
uses **Cesium Ion imagery + World Terrain + OSM Buildings** for context only:
those layers are **not** the same geometry as PLATEAU, so footprints/heights can
differ from what the simulator intersected.

**Google Colab / CLI**

For terrain + OSM buildings you need a **Cesium ion** access token (free tier):
https://cesium.com/ion/signup — then either:

  export CESIUM_ION_TOKEN="your_token"
  python experiments/build_3d_visualization.py --area-name Shinjuku \\
      --plateau-dir experiments/data/plateau_shinjuku \\
      --reference-csv experiments/data/urbannav/Shinjuku/Shinjuku/reference.csv \\
      --out-html cesium_shinjuku.html

Or pass ``--cesium-ion-token ...`` (injected into the HTML).

Without a token the viewer still runs with the default ellipsoid (no world terrain / OSM buildings).

Receiver markers and LOS/NLOS rays use **trajectory ellipsoid heights** embedded in the JSON so polylines match Python ray geometry relative to the PLATEAU GLB; they are **not** snapped to Ion terrain (terrain vs survey heights would skew intersections visually).

Satellite rays use a **reused pool of polyline ``Entity`` objects**: each epoch only updates positions/materials instead of destroying and rebuilding geometry.
"""

import argparse
import base64
import binascii
import csv
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from gnss_gpu.ephemeris import Ephemeris
from gnss_gpu.io.nav_rinex import read_nav_rinex_multi
from gnss_gpu.io.rinex import read_rinex_obs
from gnss_gpu.io.plateau import PlateauLoader
from gnss_gpu.raytrace import BuildingModel
from gnss_gpu.viz.plateau_glb import export_plateau_roi_glb
from gnss_gpu.bvh import BVHAccelerator
from gnss_gpu.urban_signal_sim import (
    UrbanSignalSimulator,
    _sat_elevation_azimuth,
    apply_atmo_bending_lite,
    virtual_satellite_ecef_lite,
    ecef_to_lla,
)

# RINEX 3 navigation systems retained for this viewer (SBAS ``S`` excluded for now).
RINEX_NAV_SYSTEMS_VIZ = ("G", "R", "E", "J", "C", "I")


def _normalize_csv_row(row: dict) -> dict:
    return {k.strip(): v for k, v in row.items()}


def _lla_deg_to_ecef(lat_deg: float, lon_deg: float, alt_m: float) -> np.ndarray:
    """Convert geodetic degrees to ECEF meters (WGS-84)."""
    a = 6378137.0
    f = 1.0 / 298.257223563
    e2 = 2.0 * f - f * f
    lat = math.radians(float(lat_deg))
    lon = math.radians(float(lon_deg))
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)
    n = a / math.sqrt(1.0 - e2 * sin_lat * sin_lat)
    x = (n + float(alt_m)) * cos_lat * cos_lon
    y = (n + float(alt_m)) * cos_lat * sin_lon
    z = (n * (1.0 - e2) + float(alt_m)) * sin_lat
    return np.array([x, y, z], dtype=np.float64)


def _to_gps_tow_seconds(ts: float) -> float:
    """Normalize time to GPS TOW-like seconds for ephemeris calls."""
    t = float(ts)
    if t < 0.0:
        return t % 604800.0
    # If already in TOW range, keep as is.
    if t <= 604800.0:
        return t
    # Likely absolute timestamp (GPS/Unix-like); fold to week seconds.
    return t % 604800.0


def _unix_to_gps_tow_seconds(unix_s: float, leap_seconds: float = 18.0) -> float:
    """Convert Unix seconds to GPS TOW [s] using a fixed leap-second offset."""
    gps_unix_offset = 315964800.0  # 1980-01-06 - 1970-01-01
    gps_seconds = float(unix_s) - gps_unix_offset + float(leap_seconds)
    return gps_seconds % 604800.0


def load_trajectory(csv_path, step=200):
    """Return positions, GPS TOW times, and raw CSV row count.

    Points loaded ≈ ``ceil(n_csv_rows / step)`` (every ``step``-th row, zero-based).
    """
    with open(csv_path, encoding="utf-8") as f:
        rows_dict = [_normalize_csv_row(r) for r in csv.DictReader(f)]

    positions: list[list[float]] = []
    times: list[float] = []

    # UrbanNav reference.csv mode (ECEF columns present).
    if rows_dict and {"ECEF X (m)", "ECEF Y (m)", "ECEF Z (m)", "GPS TOW (s)"} <= set(rows_dict[0].keys()):
        n_csv = len(rows_dict)
        for i in range(0, n_csv, step):
            r = rows_dict[i]
            positions.append(
                [
                    float(r["ECEF X (m)"]),
                    float(r["ECEF Y (m)"]),
                    float(r["ECEF Z (m)"]),
                ]
            )
            times.append(_to_gps_tow_seconds(float(r["GPS TOW (s)"])))
        return np.array(positions), np.array(times), int(n_csv)

    # KLT gt.csv mode: headerless rows [timestamp, lat, lon, alt].
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.reader(f)
        raw_rows = [r for r in reader if r]
    n_csv = len(raw_rows)
    for i in range(0, n_csv, step):
        r = raw_rows[i]
        if len(r) < 4:
            continue
        ts = float(r[0])
        lat = float(r[1])
        lon = float(r[2])
        alt = float(r[3])
        ecef = _lla_deg_to_ecef(lat, lon, alt)
        positions.append([float(ecef[0]), float(ecef[1]), float(ecef[2])])
        # KLT gt.csv stores Unix timestamps in the first column.
        times.append(_unix_to_gps_tow_seconds(ts))
    if not positions:
        raise ValueError(f"Could not parse trajectory CSV format: {csv_path}")
    print("  Trajectory parser: detected gt.csv-style [timestamp,lat,lon,alt], converted to ECEF.")
    return np.array(positions), np.array(times), int(n_csv)


def trajectory_row_stride(step_arg: float) -> int:
    """Convert ``--traj-step`` to an integer CSV row stride (minimum every row).

    ``0`` (or ``0.0``) means **no downsampling** — same as stride ``1`` (every row).
    Other values are rounded to an integer; stride is always at least ``1``.
    """
    x = float(step_arg)
    if x < 0:
        raise ValueError("--traj-step must be >= 0 (0 = use every CSV row)")
    if x == 0:
        return 1
    return max(1, int(round(x)))


def _gps_seconds_of_week(dt: datetime) -> float:
    """GPS time-of-week [s] consistent with :mod:`gnss_gpu.io.nav_rinex` broadcast messages."""
    gps_epoch = datetime(1980, 1, 6)
    return (dt - gps_epoch).total_seconds() % 604800.0


def _infer_nav_date_from_filename(nav_path: str) -> Optional[datetime]:
    """Infer NAV date from common BRDC filename token _YYYYDDD."""
    name = os.path.basename(nav_path)
    m = re.search(r"_(\d{4})(\d{3})", name)
    if not m:
        return None
    year = int(m.group(1))
    doy = int(m.group(2))
    try:
        return datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=doy - 1)
    except ValueError:
        return None


def _infer_ref_start_datetime_utc(ref_csv: str) -> Optional[datetime]:
    """Infer trajectory absolute date from UrbanNav reference.csv or gt.csv."""
    p = Path(ref_csv)
    if not p.exists():
        return None
    with open(p, newline="", encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        rows = [_normalize_csv_row(r) for r in rd]
    if rows and "GPS Week" in rows[0] and "GPS TOW (s)" in rows[0]:
        try:
            w = int(float(rows[0]["GPS Week"]))
            tow = float(rows[0]["GPS TOW (s)"])
            gps_epoch = datetime(1980, 1, 6, tzinfo=timezone.utc)
            return gps_epoch + timedelta(seconds=w * 604800.0 + tow)
        except Exception:
            return None
    # gt.csv style [unix_s, lat, lon, alt]
    with open(p, newline="", encoding="utf-8") as fh:
        r2 = csv.reader(fh)
        for row in r2:
            if row and len(row) >= 1:
                try:
                    return datetime.fromtimestamp(float(row[0]), tz=timezone.utc)
                except Exception:
                    continue
    return None


def _infer_obs_start_datetime_utc(obs_path: Optional[str]) -> Optional[datetime]:
    if not obs_path:
        return None
    try:
        obs = read_rinex_obs(Path(obs_path))
        if not obs.epochs:
            return None
        return obs.epochs[0].time.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _validate_nav_date_sanity(nav_file: str, ref_csv: str, obs_rover_path: Optional[str]) -> None:
    nav_dt = _infer_nav_date_from_filename(nav_file)
    ref_dt = _infer_ref_start_datetime_utc(ref_csv)
    obs_dt = _infer_obs_start_datetime_utc(obs_rover_path)
    if nav_dt is None:
        return
    checks = []
    if ref_dt is not None:
        checks.append(("reference", ref_dt))
    if obs_dt is not None:
        checks.append(("obs", obs_dt))
    for label, d in checks:
        dd = abs((d - nav_dt).total_seconds()) / 86400.0
        if dd > 14.0:
            raise RuntimeError(
                f"NAV date sanity check failed: nav filename date {nav_dt.date()} is far from "
                f"{label} date {d.date()} (|Δ|={dd:.1f} days). Use a matching NAV/BRDC file."
            )


def _normalize_rinex_sat_id(raw: str) -> str:
    s = str(raw).strip().upper().replace(" ", "")
    if len(s) >= 2 and s[0] in "GREJCIS" and s[1:].isdigit():
        return f"{s[0]}{int(s[1:]):02d}"
    return s


def _rinex_sat_id_from_prn(prn) -> str:
    """Map ephemeris PRN entry to RINEX satellite id (e.g. ``G05``, ``E13``)."""
    if isinstance(prn, str):
        p = prn.strip().upper()
        if len(p) >= 2 and p[0] in "GREJCIS":
            rest = p[1:]
            if rest.isdigit():
                return f"{p[0]}{int(rest):02d}"
            return p
        if p.isdigit():
            return f"G{int(p):02d}"
        return _normalize_rinex_sat_id(p)
    return f"G{int(prn):02d}"


class _ObsTowPrnLookup:
    """Nearest-neighbour OBS epochs by GPS TOW; satellite ids from code pseudoranges."""

    def __init__(self, obs_path: str, match_tol_s: float):
        obs = read_rinex_obs(Path(obs_path))
        rows: list[tuple[float, frozenset[str]]] = []
        for ep in obs.epochs:
            sats: set[str] = set()
            for sat_id, odict in ep.observations.items():
                for k, v in odict.items():
                    if not str(k).startswith("C"):
                        continue
                    try:
                        fv = float(v)
                    except (TypeError, ValueError):
                        continue
                    if abs(fv) > 1e-6:
                        sats.add(_normalize_rinex_sat_id(str(sat_id)))
                        break
            if sats:
                rows.append((_gps_seconds_of_week(ep.time), frozenset(sats)))
        rows.sort(key=lambda x: x[0])
        self._tows = np.array([r[0] for r in rows], dtype=np.float64) if rows else np.zeros(0)
        self._sets = [r[1] for r in rows]
        self._tol = float(match_tol_s)

    @property
    def n_epochs(self) -> int:
        return len(self._sets)

    @property
    def tow_min(self) -> Optional[float]:
        if self._tows.size == 0:
            return None
        return float(np.min(self._tows))

    @property
    def tow_max(self) -> Optional[float]:
        if self._tows.size == 0:
            return None
        return float(np.max(self._tows))

    def nearest_sat_ids(self, tow_csv: float) -> Optional[frozenset[str]]:
        """Satellite ids observed at nearest OBS epoch, or ``None`` if outside tolerance."""
        if not self._sets:
            return frozenset()
        idx = int(np.searchsorted(self._tows, float(tow_csv)))
        best_d = float("inf")
        best: Optional[frozenset[str]] = None
        for j in (idx - 1, idx):
            if 0 <= j < len(self._sets):
                d = abs(_tow_delta_sec(float(self._tows[j]), float(tow_csv)))
                if d < best_d:
                    best_d = d
                    best = self._sets[j]
        if best is None or best_d > self._tol:
            return None
        return best


def _obs_keep_ids_for_tow(
    tow_csv: float,
    obs_lookup: _ObsTowPrnLookup,
    strict: bool,
    warn_state: dict,
) -> Optional[frozenset[str]]:
    """RINEX satellite ids with code observations at the nearest OBS epoch.

    Returns:
        ``None`` — no OBS epoch within tolerance and not ``strict``; caller skips OBS filtering.
        ``frozenset`` — caller keeps only these ids (possibly empty, e.g. strict no-match).
    """
    allowed = obs_lookup.nearest_sat_ids(tow_csv)
    if allowed is None:
        if strict:
            return frozenset()
        if not warn_state.get("warned_nomatch"):
            print(
                "  OBS filter: some trajectory times have no OBS epoch within tolerance — "
                "showing all NAV sats for those epochs (use --obs-strict to hide instead)."
            )
            warn_state["warned_nomatch"] = True
        return None
    return allowed


def _apply_obs_visibility_mask(
    visible: np.ndarray,
    used_prns,
    tow_csv: float,
    obs_lookup: _ObsTowPrnLookup,
    strict: bool,
    warn_state: dict,
) -> None:
    keep = _obs_keep_ids_for_tow(tow_csv, obs_lookup, strict, warn_state)
    if keep is None:
        return
    n_sat = int(visible.shape[0])
    if len(keep) == 0:
        visible[:] = False
        return
    for i in range(n_sat):
        if visible[i]:
            sid = _rinex_sat_id_from_prn(used_prns[i])
            if sid not in keep:
                visible[i] = False


def _preferred_cnr_dbhz(obs_dict: dict) -> Optional[float]:
    """Pick one usable carrier strength observation (RINEX ``S*``, typically dB-Hz)."""
    pref_order = (
        "S1C",
        "S1X",
        "S1P",
        "S1W",
        "S2W",
        "S2C",
        "S2X",
        "S5Q",
        "S5X",
        "S7Q",
        "S7X",
        "S8Q",
        "S8X",
    )
    upper_keys = {str(k).upper(): k for k in obs_dict}
    for code in pref_order:
        k = upper_keys.get(code)
        if k is None:
            continue
        try:
            v = float(obs_dict[k])
        except (TypeError, ValueError):
            continue
        if np.isfinite(v) and v > 0.0:
            return v
    for uk, orig in sorted(upper_keys.items()):
        if not uk.startswith("S"):
            continue
        try:
            v = float(obs_dict[orig])
        except (TypeError, ValueError):
            continue
        if np.isfinite(v) and v > 0.0:
            return v
    return None


def _extract_cnr_series_from_obs(
    obs_path: str,
    *,
    max_points_per_prn: int = 2500,
) -> dict[str, list[list[float]]]:
    """Return mapping rinex_sat_id → ``[[gps_tow_s, cnr_dbhz], ...]``, downsampled for HTML embed."""
    obs = read_rinex_obs(Path(obs_path))
    raw: dict[str, list[tuple[float, float]]] = {}
    for ep in obs.epochs:
        tow = float(_gps_seconds_of_week(ep.time))
        for sat_id, odict in ep.observations.items():
            cnr = _preferred_cnr_dbhz(odict)
            if cnr is None:
                continue
            sid = _normalize_rinex_sat_id(str(sat_id))
            raw.setdefault(sid, []).append((tow, float(cnr)))
    out: dict[str, list[list[float]]] = {}
    cap = max(50, int(max_points_per_prn))
    for sid, pairs in raw.items():
        pairs.sort(key=lambda x: x[0])
        if len(pairs) > cap:
            idx = np.linspace(0, len(pairs) - 1, cap, dtype=np.int64)
            pairs = [pairs[int(i)] for i in idx]
        out[sid] = [[round(float(t), 4), round(float(v), 3)] for t, v in pairs]
    return out


def _resolve_obs_path(traj_csv: str, obs_opt: str) -> Optional[str]:
    """Return rover ``.obs`` path or ``None`` if ``obs_opt`` empty and no default file."""
    o = (obs_opt or "").strip()
    if o:
        p = os.path.abspath(o)
        if not os.path.isfile(p):
            raise FileNotFoundError(f"RINEX observation file not found: {p}")
        return p
    d = os.path.dirname(os.path.abspath(traj_csv))
    for name in ("rover_ublox.obs", "rover_trimble.obs", "rover.obs"):
        cand = os.path.join(d, name)
        if os.path.isfile(cand):
            return cand
    return None


def _cnr_obs_path_for_embed(ref_csv: str, obs_cli: str, *, embed: bool) -> Optional[str]:
    """Resolve rover OBS path for embedding C/N₀ series (optional ``--obs``, else default names)."""
    if not embed:
        return None
    o = (obs_cli or "").strip()
    if o:
        p = os.path.abspath(o)
        return p if os.path.isfile(p) else None
    return _resolve_obs_path(ref_csv, "") or None


def _resolve_nav_path(traj_csv: str, nav_path: Optional[str]) -> str:
    """Return navigation file path; default ``<reference_csv_dir>/base.nav``."""
    if nav_path:
        p = os.path.abspath(nav_path)
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Navigation file not found: {p}")
        return p
    cand = os.path.join(os.path.dirname(os.path.abspath(traj_csv)), "base.nav")
    if os.path.isfile(cand):
        return cand
    raise FileNotFoundError(
        f"No navigation file: pass --nav or place base.nav next to the trajectory ({cand})."
    )


def _sat_display_id(prn) -> str:
    if isinstance(prn, str):
        return prn
    return f"G{int(prn):02d}"


def _tow_delta_sec(t_new: float, t_old: float) -> float:
    """GPS TOW difference [s]; unwrap a single week rollover."""
    d = float(t_new) - float(t_old)
    if d < -302400.0:
        d += 604800.0
    elif d > 302400.0:
        d -= 604800.0
    return d


def _cumulative_track_time_s(times: np.ndarray) -> np.ndarray:
    """Monotonic time along trajectory [s] from the first row (handles TOW jumps)."""
    t = np.asarray(times, dtype=np.float64)
    if t.size == 0:
        return t
    offs = np.zeros_like(t, dtype=np.float64)
    for i in range(1, t.shape[0]):
        offs[i] = offs[i - 1] + _tow_delta_sec(float(t[i]), float(t[i - 1]))
    return offs


def select_viz_epoch_indices(
    times: np.ndarray,
    n_epochs: int,
    min_interval_s: float,
) -> tuple[np.ndarray, dict]:
    """Pick ``min(n_epochs, n_rows)`` trajectory indices spread along time.

    When ``min_interval_s`` > 0, builds the maximal **1 / min_interval_s** ladder along GPS time
    (greedy forward on ``times``), then subsamples ``need`` rungs uniformly so consecutive picks stay
    ≥ ``min_interval_s`` apart while targeting ``n_epochs`` epochs. If the trajectory span or row
    density cannot supply that many spaced slots, returns fewer and reports via the dict.

    Returns:
        (indices, info) where ``info`` has keys ``requested``, ``used``, ``max_spaced_slots``,
        ``traj_rows``, ``span_s``.
    """
    times = np.asarray(times, dtype=np.float64)
    n = int(times.shape[0])
    info = {
        "requested": max(1, int(n_epochs)),
        "used": 0,
        "max_spaced_slots": 0,
        "traj_rows": n,
        "span_s": 0.0,
    }
    if n == 0:
        return np.array([], dtype=int), info

    need_target = max(1, min(int(n_epochs), n))
    info["requested"] = int(n_epochs)

    if min_interval_s <= 0.0:
        out = np.linspace(0, n - 1, need_target, dtype=int)
        info["used"] = int(out.shape[0])
        info["max_spaced_slots"] = n
        offs = _cumulative_track_time_s(times)
        info["span_s"] = float(offs[-1]) if offs.size else 0.0
        return out, info

    offs = _cumulative_track_time_s(times)
    span = float(offs[-1])
    if span <= 0.0:
        span = abs(float(times[-1] - times[0]))
    info["span_s"] = span

    greedy_idx: list[int] = []
    last_t: Optional[float] = None
    for i in range(n):
        tt = float(offs[i])
        if last_t is None or tt >= last_t + float(min_interval_s) - 1e-9:
            greedy_idx.append(i)
            last_t = tt

    mf = len(greedy_idx)
    info["max_spaced_slots"] = mf
    if mf == 0:
        info["used"] = 1
        return np.array([0], dtype=int), info

    need_eff = min(need_target, mf)
    info["used"] = need_eff

    if need_eff < need_target:
        print(
            f"  WARNING: asked {need_target} viz epochs at ≥{float(min_interval_s):g}s GPS spacing; "
            f"only {mf} spaced slots exist (~{span:.0f}s span, {n} traj rows after step). Using {need_eff}."
        )

    if need_eff == 1:
        return np.asarray([greedy_idx[0]], dtype=int), info

    slot_i = np.linspace(0, mf - 1, need_eff)
    slot_i = np.round(slot_i).astype(np.int64)
    slot_i = np.clip(slot_i, 0, mf - 1)
    for k in range(1, need_eff):
        if slot_i[k] <= slot_i[k - 1]:
            slot_i[k] = min(slot_i[k - 1] + 1, mf - 1)

    if slot_i[-1] >= mf:
        slot_i = np.clip(slot_i, 0, mf - 1)

    return np.asarray([greedy_idx[int(s)] for s in slot_i], dtype=int), info


def _validate_time_alignment_for_viz(
    times_tow: np.ndarray,
    eph: Ephemeris,
    prn_catalog,
    obs_lookup: Optional[_ObsTowPrnLookup] = None,
) -> None:
    """Fail early for obvious time-base mismatches before expensive geometry."""
    if times_tow.size == 0:
        raise RuntimeError("Trajectory has no usable time samples.")
    t_min = float(np.min(times_tow))
    t_max = float(np.max(times_tow))
    if t_max - t_min < 1e-6:
        raise RuntimeError("Trajectory time span is zero; check reference/gt CSV.")

    probe_n = min(8, max(2, int(times_tow.size)))
    probe_tows = np.linspace(t_min, t_max, num=probe_n, dtype=np.float64)
    sat_probe, _, _ = eph.compute_batch(probe_tows, prn_list=prn_catalog)
    if sat_probe.size == 0 or sat_probe.shape[1] == 0:
        raise RuntimeError(
            "NAV ephemeris provides no satellites for trajectory time range "
            f"[{t_min:.1f}, {t_max:.1f}] TOW. Check NAV file/date."
        )

    if obs_lookup is not None and obs_lookup.n_epochs > 0:
        o_min = obs_lookup.tow_min
        o_max = obs_lookup.tow_max
        if o_min is not None and o_max is not None:
            overlap = max(0.0, min(t_max, o_max) - max(t_min, o_min))
            if overlap < 1.0:
                raise RuntimeError(
                    "Trajectory and OBS time windows have almost no overlap: "
                    f"traj=[{t_min:.1f},{t_max:.1f}] obs=[{o_min:.1f},{o_max:.1f}]"
                )


def compute_all_epochs(
    area_name,
    plateau_dir,
    traj_csv,
    n_epochs=15,
    step=200,
    plateau_zone=9,
    nav_path: Optional[str] = None,
    elevation_mask_deg: float = 10.0,
    eph_batch_chunk: int = 64,
    epoch_min_interval_s: float = 0.0,
    viz_multipath: bool = False,
    multipath_min_delay_m: float = 0.5,
    obs_rover_path: Optional[str] = None,
    obs_match_tol_s: float = 1.0,
    obs_strict: bool = False,
    cnr_obs_path: Optional[str] = None,
    cnr_max_points_per_prn: int = 2500,
    atmo_bending_lite: bool = False,
    atmo_pressure_hpa: float = 1010.0,
    atmo_temp_c: float = 10.0,
    triangles_npy: Optional[str] = None,
):
    """Compute LOS/NLOS for all epochs and return visualization data.

    ``elevation_mask_deg`` is passed to :class:`~gnss_gpu.urban_signal_sim.UrbanSignalSimulator`
    (default 10°; only satellites at or above that elevation get rays and LOS counts).

    Satellite positions use :meth:`~gnss_gpu.ephemeris.Ephemeris.compute_batch`
    in windows of ``eph_batch_chunk`` epochs (default 64). Batch keeps only PRNs
    with a valid ephemeris block for *every* epoch in that window; smaller windows
    preserve more satellites on long runs. Set ``eph_batch_chunk`` to 0 for one
    batch over the whole timeline (fastest, strictest PRN intersection).

    LOS rays against PLATEAU use :meth:`~gnss_gpu.bvh.BVHAccelerator.check_los_batch`
    when available (same CUDA extension as single-ray LOS); otherwise
    :class:`~gnss_gpu.urban_signal_sim.UrbanSignalSimulator` per epoch.

    Optional ``viz_multipath`` adds first-order specular reflection segments
    (satellite → bounce → receiver) via :meth:`~gnss_gpu.bvh.BVHAccelerator.compute_multipath`
    with CPU mesh fallback when BVH multipath is unavailable (needs compiled raytrace).

    Optional ``obs_rover_path`` filters satellites to those with code pseudorange in rover OBS.
    With BVH batching, OBS is applied to ``visible_batch`` before LOS/multipath so unused PRNs
    are not ray-traced.

    Optional ``cnr_obs_path``: when set, embed C/N0 time series from that RINEX observation file
    (``S*`` observations) under ``cnrByPrn`` for the HTML ray-click chart.

    ``epoch_min_interval_s``: if > 0, minimum GPS-time spacing between consecutive viz epochs
    (uniform subsample along a greedy ladder); if ``0`` (default), epochs are evenly spaced in
    **row index** after ``step`` (same as legacy ``np.linspace``).
    """
    print(f"[{area_name}] Loading geometry...")
    if triangles_npy:
        tri = np.asarray(np.load(triangles_npy), dtype=np.float64)
        if tri.ndim != 3 or tri.shape[1:] != (3, 3):
            raise ValueError(f"--triangles-npy must have shape [N,3,3], got {tri.shape}")
        building = BuildingModel(tri)
        print(f"  source: triangles npy ({triangles_npy})")
    else:
        loader = PlateauLoader(zone=int(plateau_zone))
        building = loader.load_directory(plateau_dir)
        print(f"  source: PLATEAU directory ({plateau_dir})")
    print(f"  {len(building.triangles)} triangles")

    multipath_warned = False
    multipath_batch_warned = False

    positions, times, n_csv_rows = load_trajectory(traj_csv, step=step)
    n_loaded = len(times)
    print(
        f"  Trajectory CSV: {n_csv_rows} rows; --traj-step={step} → {n_loaded} positions loaded "
        f"(cannot viz more epochs than this)."
    )
    indices, sel_info = select_viz_epoch_indices(times, n_epochs, epoch_min_interval_s)
    if int(n_epochs) > n_loaded:
        sug = max(1, n_csv_rows // max(int(n_epochs), 1))
        approx_pts = (n_csv_rows + sug - 1) // sug
        print(
            f"  Note: --n-epochs={int(n_epochs)} > loaded positions ({n_loaded}). "
            f"Lower --traj-step (e.g. --traj-step {sug} → ~{approx_pts} positions) or reduce --n-epochs."
        )
    if epoch_min_interval_s > 0:
        print(
            f"  Viz epochs: {len(indices)} (GPS Δt ≥ {epoch_min_interval_s:g}s between consecutive; "
            f"{sel_info['traj_rows']} traj rows, ~{sel_info['span_s']:.0f}s span)"
        )
    else:
        print(f"  Viz epochs: {len(indices)} (evenly spaced rows; no min Δt constraint)")

    print(f"[{area_name}] Building BVH...")
    bvh = BVHAccelerator.from_building_model(building)
    has_bvh_los_batch = hasattr(bvh, "check_los_batch")
    print(f"  BVH LOS batch path: {'enabled' if has_bvh_los_batch else 'disabled (fallback per-epoch)'}")

    nav_file = _resolve_nav_path(traj_csv, nav_path)
    _validate_nav_date_sanity(nav_file, traj_csv, obs_rover_path)
    print(f"[{area_name}] Loading broadcast ephemeris: {nav_file}")
    nav_messages = read_nav_rinex_multi(nav_file, systems=RINEX_NAV_SYSTEMS_VIZ)
    eph = Ephemeris(nav_messages)
    prn_catalog = eph.available_prns
    print(f"  Satellites in NAV: {len(prn_catalog)}")
    effective_elevation_mask_deg = float(elevation_mask_deg)
    if obs_rover_path:
        # When OBS is provided, use observation availability as the visibility gate:
        # keep all elevations and let OBS filtering decide what to show.
        effective_elevation_mask_deg = -90.0
        print(
            "  Elevation mask: disabled (-90°) because OBS filtering is active "
            f"({obs_rover_path})"
        )
    else:
        print(f"  Elevation mask: {effective_elevation_mask_deg:g}° above horizon")
    if viz_multipath:
        print(
            f"  Multipath viz: enabled (delay ≥ {multipath_min_delay_m:g} m); "
            "uses BVH batched multipath when LOS batch path is active, else per-epoch GPU/CPU."
        )
    if atmo_bending_lite:
        print(
            "  Atmo bending lite: enabled "
            f"(P={float(atmo_pressure_hpa):.1f} hPa, T={float(atmo_temp_c):.1f} C)"
        )
    else:
        print("  Atmo bending lite: disabled")

    usim = UrbanSignalSimulator(
        building_model=bvh,
        noise_floor_db=-35,
        elevation_mask_deg=effective_elevation_mask_deg,
        atmo_bending_lite=bool(atmo_bending_lite),
        atmo_pressure_hpa=float(atmo_pressure_hpa),
        atmo_temp_c=float(atmo_temp_c),
    )

    obs_lookup: Optional[_ObsTowPrnLookup] = None
    obs_warn_state: dict = {}
    if obs_rover_path:
        obs_lookup = _ObsTowPrnLookup(obs_rover_path, obs_match_tol_s)
        print(
            f"[{area_name}] OBS PRN filter: {obs_rover_path} "
            f"(match nearest OBS epoch if |ΔGPS TOW|≤{obs_match_tol_s:g}s; strict={obs_strict})"
        )
        print(f"  OBS epochs with code pseudorange: {obs_lookup.n_epochs}")

    _validate_time_alignment_for_viz(
        times_tow=np.asarray(times, dtype=np.float64),
        eph=eph,
        prn_catalog=prn_catalog,
        obs_lookup=obs_lookup,
    )

    cnr_by_prn: dict[str, list] = {}
    cnr_obs_source = ""
    if cnr_obs_path:
        try:
            cnr_by_prn = _extract_cnr_series_from_obs(
                cnr_obs_path,
                max_points_per_prn=int(cnr_max_points_per_prn),
            )
            cnr_obs_source = os.path.basename(cnr_obs_path)
            print(
                f"[{area_name}] C/N0 embed: {cnr_obs_path} "
                f"({len(cnr_by_prn)} PRNs, ≤{int(cnr_max_points_per_prn)} samples/PRN)"
            )
        except Exception as e:
            print(f"  C/N0 embed skipped: {e}")

    idx_list = [int(i) for i in indices]
    if eph_batch_chunk <= 0:
        chunk_bounds = [(0, len(idx_list))]
    else:
        step_sz = max(1, int(eph_batch_chunk))
        chunk_bounds = [
            (a, min(a + step_sz, len(idx_list)))
            for a in range(0, len(idx_list), step_sz)
        ]

    epochs_data = []
    log_every = max(1, min(50, len(idx_list) // 15 or 1))
    fi = 0

    for cb_start, cb_end in chunk_bounds:
        chunk_idx = idx_list[cb_start:cb_end]
        tow_chunk = np.asarray([float(times[ei]) for ei in chunk_idx], dtype=np.float64)
        sat_b, clk_b, used_prns = eph.compute_batch(tow_chunk, prn_list=prn_catalog)
        n_sat_chunk = int(sat_b.shape[1])

        if n_sat_chunk == 0:
            sat_rows = None
            clk_rows = None
        else:
            sat_rows = sat_b
            clk_rows = clk_b

        los_batch = None
        el_batch = None
        visible_batch = None
        mp_delays_batch = None
        mp_refl_batch = None
        if (
            sat_rows is not None
            and has_bvh_los_batch
            and len(chunk_idx) > 0
        ):
            rx_blk = np.ascontiguousarray(positions[chunk_idx], dtype=np.float64)
            sat_blk = np.ascontiguousarray(sat_rows, dtype=np.float64)
            sat_ray_blk = sat_blk.copy()
            n_b = sat_blk.shape[0]
            n_sat_blk = sat_blk.shape[1]
            el_batch = np.zeros((n_b, n_sat_blk), dtype=np.float64)
            for i in range(n_b):
                el_batch[i], _ = _sat_elevation_azimuth(rx_blk[i], sat_blk[i])
            if usim.atmo_bending_lite:
                el_batch = apply_atmo_bending_lite(
                    el_batch,
                    pressure_hpa=usim.atmo_pressure_hpa,
                    temp_c=usim.atmo_temp_c,
                )
                for i in range(n_b):
                    sat_ray_blk[i] = virtual_satellite_ecef_lite(
                        rx_blk[i],
                        sat_blk[i],
                        pressure_hpa=usim.atmo_pressure_hpa,
                        temp_c=usim.atmo_temp_c,
                    )
            visible_batch = el_batch >= usim.elevation_mask_rad
            if obs_lookup is not None:
                for i in range(n_b):
                    keep = _obs_keep_ids_for_tow(
                        float(times[chunk_idx[i]]),
                        obs_lookup,
                        obs_strict,
                        obs_warn_state,
                    )
                    if keep is None:
                        continue
                    row = visible_batch[i]
                    if len(keep) == 0:
                        row[:] = False
                        continue
                    for j in range(n_sat_blk):
                        if row[j] and _rinex_sat_id_from_prn(used_prns[j]) not in keep:
                            row[j] = False
            sat_work = sat_ray_blk.copy()
            sat_work[~visible_batch] = np.nan
            los_batch = np.asarray(bvh.check_los_batch(rx_blk, sat_work), dtype=bool)
            if viz_multipath:
                try:
                    mp_delays_batch, mp_refl_batch = bvh.compute_multipath_batch(rx_blk, sat_work)
                    mp_delays_batch = np.asarray(mp_delays_batch, dtype=np.float64)
                    mp_refl_batch = np.asarray(mp_refl_batch, dtype=np.float64)
                except Exception as _e:
                    mp_delays_batch = None
                    mp_refl_batch = None
                    if not multipath_batch_warned:
                        print(
                            f"  BVH multipath batch unavailable ({_e}); "
                            "falling back to per-epoch multipath for this run."
                        )
                        multipath_batch_warned = True

        for local_i, ei in enumerate(chunk_idx):
            fi += 1
            rx = positions[ei]
            t = times[ei] - times[0]

            if sat_rows is None:
                sat_ecef, sat_clk, used_prns_i = eph.compute(
                    float(times[ei]), prn_list=prn_catalog
                )
            else:
                sat_ecef = sat_rows[local_i]
                sat_clk = clk_rows[local_i]
                used_prns_i = used_prns
            sat_ecef_ray = sat_ecef

            n_sat = int(np.asarray(sat_ecef).reshape(-1, 3).shape[0])
            if n_sat == 0:
                lat, lon, alt = ecef_to_lla(*rx)
                epochs_data.append({
                    "rx": [math.degrees(lat), math.degrees(lon), alt],
                    "rays": [],
                    "reflections": [],
                    "n_los": 0,
                    "n_nlos": 0,
                    "t": t,
                    "gpsTow": float(times[ei]),
                })
                if fi % log_every == 0 or fi == len(idx_list):
                    print(f"  [{fi}/{len(idx_list)}] t={t:.0f}s — no ephemeris at this TOW")
                continue

            if los_batch is not None:
                visible = np.asarray(visible_batch[local_i], dtype=bool).copy()
                el = el_batch[local_i]
                is_los = np.ones(n_sat, dtype=bool)
                vis_idx = np.where(visible)[0]
                if len(vis_idx) > 0:
                    is_los[vis_idx] = los_batch[local_i, vis_idx]
                if usim.atmo_bending_lite:
                    sat_ecef_ray = virtual_satellite_ecef_lite(
                        rx,
                        sat_ecef,
                        pressure_hpa=usim.atmo_pressure_hpa,
                        temp_c=usim.atmo_temp_c,
                    )
            else:
                result = usim.compute_epoch(
                    rx_ecef=rx,
                    sat_ecef=sat_ecef,
                    sat_clk=sat_clk,
                    prn_list=used_prns_i,
                )
                visible = np.asarray(result["visible"], dtype=bool).copy()
                el = result["elevations"]
                is_los = np.asarray(result["is_los"], dtype=bool)
                if usim.atmo_bending_lite:
                    sat_ecef_ray = virtual_satellite_ecef_lite(
                        rx,
                        sat_ecef,
                        pressure_hpa=usim.atmo_pressure_hpa,
                        temp_c=usim.atmo_temp_c,
                    )

            if obs_lookup is not None:
                _apply_obs_visibility_mask(
                    visible,
                    used_prns_i,
                    float(times[ei]),
                    obs_lookup,
                    obs_strict,
                    obs_warn_state,
                )

            n_los_ep = int(np.sum(is_los & visible))
            n_nlos_ep = int(np.sum(~is_los & visible))

            reflections = []
            sat_mat = np.ascontiguousarray(np.asarray(sat_ecef_ray), dtype=np.float64).reshape(n_sat, 3)
            if viz_multipath and n_sat > 0:
                delays_mp = None
                refl_pts_mp = None
                if mp_delays_batch is not None and mp_refl_batch is not None:
                    delays_mp = mp_delays_batch[local_i]
                    refl_pts_mp = mp_refl_batch[local_i]
                else:
                    try:
                        delays_mp, refl_pts_mp = bvh.compute_multipath(rx, sat_mat)
                    except Exception:
                        try:
                            delays_mp, refl_pts_mp = building.compute_multipath(rx, sat_mat)
                        except Exception as _e:
                            if not multipath_warned:
                                print(
                                    "  Multipath viz skipped (compile/install raytrace or BVH with multipath): "
                                    f"{_e}"
                                )
                                multipath_warned = True
                if delays_mp is not None and refl_pts_mp is not None:
                    delays_mp = np.asarray(delays_mp, dtype=np.float64).reshape(-1)
                    refl_pts_mp = np.asarray(refl_pts_mp, dtype=np.float64).reshape(-1, 3)
                    thr = float(multipath_min_delay_m)
                    for i in range(n_sat):
                        if not visible[i]:
                            continue
                        dmp = float(delays_mp[i])
                        if dmp < thr:
                            continue
                        R = refl_pts_mp[i]
                        if not np.all(np.isfinite(R)):
                            continue
                        sat_i = sat_mat[i]
                        v = sat_i - R
                        nv = float(np.linalg.norm(v))
                        if nv < 1.0:
                            continue
                        v = v / nv
                        inc_end = R + v * min(5000.0, nv * 0.999)
                        ila, ilo, ih = ecef_to_lla(
                            float(inc_end[0]), float(inc_end[1]), float(inc_end[2])
                        )
                        bla, blo, bh = ecef_to_lla(float(R[0]), float(R[1]), float(R[2]))
                        reflections.append(
                            {
                                "prn": _sat_display_id(used_prns_i[i]),
                                "delay_m": dmp,
                                "inc": [math.degrees(ila), math.degrees(ilo), ih],
                                "bounce": [math.degrees(bla), math.degrees(blo), bh],
                                "nlosMp": bool(not is_los[i]),
                            }
                        )

            hide_direct_prns = {r["prn"] for r in reflections if r.get("nlosMp")}
            lat, lon, alt = ecef_to_lla(*rx)
            epoch = {
                "rx": [math.degrees(lat), math.degrees(lon), alt],
                "rays": [],
                "reflections": reflections,
                "n_los": n_los_ep,
                "n_nlos": n_nlos_ep,
                "t": t,
                "gpsTow": float(times[ei]),
            }
            for i in range(n_sat):
                if not visible[i]:
                    continue
                prn_str = _sat_display_id(used_prns_i[i])
                if prn_str in hide_direct_prns:
                    continue
                direction = sat_ecef_ray[i] - rx
                dist = np.linalg.norm(direction)
                ray_end = rx + direction / dist * 5000
                re_lat, re_lon, re_alt = ecef_to_lla(*ray_end)
                epoch["rays"].append({
                    "prn": prn_str,
                    "los": bool(is_los[i]),
                    "el": float(np.degrees(el[i])),
                    "end": [math.degrees(re_lat), math.degrees(re_lon), re_alt],
                })
            epochs_data.append(epoch)
            if fi % log_every == 0 or fi == len(idx_list):
                print(
                    f"  [{fi}/{len(idx_list)}] t={t:.0f}s "
                    f"LOS={n_los_ep} NLOS={n_nlos_ep} "
                    f"(batch PRNs={n_sat_chunk if sat_rows is not None else 'fallback'})"
                )

    # Trajectory line
    traj = []
    for p in positions:
        lat, lon, alt = ecef_to_lla(*p)
        traj.append([math.degrees(lat), math.degrees(lon), alt])

    pivot_ecef = np.mean(positions, axis=0).tolist()

    return {
        "epochs": epochs_data,
        "trajectory": traj,
        "area": area_name,
        "pivot_ecef": pivot_ecef,
        "plateauModel": None,
        "cnrByPrn": cnr_by_prn,
        "cnrObsSource": cnr_obs_source or None,
    }


def warn_if_ion_token_not_yet_valid(tok: str) -> None:
    """Warn when a JWT-shaped Ion token has ``iat`` in the future (Ion often rejects)."""
    t = (tok or "").strip()
    parts = t.split(".")
    if len(parts) != 3:
        return
    try:
        pad = "=" * ((4 - len(parts[1]) % 4) % 4)
        raw = base64.urlsafe_b64decode(parts[1] + pad)
        payload = json.loads(raw.decode("ascii"))
    except (ValueError, json.JSONDecodeError, binascii.Error, UnicodeDecodeError):
        return
    iat = payload.get("iat")
    if iat is None:
        return
    skew_s = 120.0
    if float(iat) > time.time() + skew_s:
        when = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(float(iat)))
        print(
            f"WARNING: Cesium Ion token has iat={when} (still in the future). "
            "Terrain/OSM may fail until then — mint a new token at https://ion.cesium.com/tokens"
        )


_VIEWER_EXTENSIONS_JS = r"""
function rinexSystemLetter(prnStr) {
  const s = String(prnStr || '').trim().toUpperCase();
  if (!s) return '';
  const c = s.charAt(0);
  if ('GREJCI'.indexOf(c) >= 0) return c;
  return '';
}
function collectPrnsFromDataset(ds) {
  const out = new Set();
  if (!ds || !ds.epochs) return [];
  ds.epochs.forEach(ep => {
    (ep.rays || []).forEach(r => { if (r && r.prn) out.add(String(r.prn).trim()); });
    const refl = ep.reflections;
    if (refl) refl.forEach(r => { if (r && r.prn) out.add(String(r.prn).trim()); });
  });
  return Array.from(out).sort((a, b) =>
    String(a).localeCompare(String(b), undefined, { numeric: true, sensitivity: 'base' }));
}
function prnPassesFilters(prnStr) {
  const selC = document.getElementById('selConstellation');
  const sys = rinexSystemLetter(prnStr);
  const cval = selC && selC.value;
  if (cval && sys !== cval) return false;
  const sel = document.getElementById('selSatellite');
  if (!sel || sel.selectedOptions.length === 0) return true;
  const want = new Set(Array.from(sel.selectedOptions).map(o => String(o.value)));
  return want.has(String(prnStr));
}
function rebuildSatelliteSelectorOptions() {
  const selC = document.getElementById('selConstellation');
  const selS = document.getElementById('selSatellite');
  if (!selS || typeof datasets === 'undefined' || !datasets.length) return;
  const ds = datasets[currentDataset] || datasets[0];
  const all = collectPrnsFromDataset(ds);
  const cval = selC ? selC.value : '';
  const filtered = cval ? all.filter(p => rinexSystemLetter(p) === cval) : all;
  const prev = new Set(Array.from(selS.selectedOptions).map(o => o.value));
  selS.innerHTML = '';
  filtered.forEach(p => {
    const opt = document.createElement('option');
    opt.value = p;
    opt.textContent = p;
    if (prev.has(p)) opt.selected = true;
    selS.appendChild(opt);
  });
}
let cnrChartPrn = null;
function normalizeCnrPrnKey(raw) {
  return String(raw || '').trim().toUpperCase();
}
function lookupCnrSeries(ds, prnRaw) {
  const map = (ds && ds.cnrByPrn) || {};
  const k = normalizeCnrPrnKey(prnRaw);
  if (map[k]) return map[k];
  const keys = Object.keys(map);
  for (let i = 0; i < keys.length; i++) {
    if (normalizeCnrPrnKey(keys[i]) === k) return map[keys[i]];
  }
  return null;
}
function towDeltaSec(a, b) {
  let d = Number(a) - Number(b);
  if (d < -302400.0) d += 604800.0;
  else if (d > 302400.0) d -= 604800.0;
  return d;
}
function satStateAtEpoch(ep, prnRaw) {
  const prn = normalizeCnrPrnKey(prnRaw);
  const refl = ((ep && ep.reflections) || []).filter(function (mp) {
    return normalizeCnrPrnKey(mp && mp.prn) === prn;
  });
  if (refl.some(function (mp) { return !!mp.nlosMp; })) {
    return { label: 'NLOS MP', cls: 's-nlosmp' };
  }
  if (refl.length > 0) {
    return { label: 'LOS MP', cls: 's-losmp' };
  }
  const rays = (ep && ep.rays) || [];
  for (let i = 0; i < rays.length; i++) {
    const r = rays[i];
    if (normalizeCnrPrnKey(r && r.prn) !== prn) continue;
    return r.los ? { label: 'LOS', cls: 's-los' } : { label: 'NLOS', cls: 's-nlos' };
  }
  return { label: '-', cls: '' };
}
function drawCnrChart(ds, prn, cursorTow) {
  const canvas = document.getElementById('cnrCanvas');
  const titleEl = document.getElementById('cnrTitle');
  const hintEl = document.getElementById('cnrHint');
  const statusEl = document.getElementById('cnrSatStatus');
  if (!canvas || !titleEl || !hintEl) return;
  if (statusEl) {
    const ep = (ds && ds.epochs && ds.epochs[displayedEpochIndex]) || null;
    const st = satStateAtEpoch(ep, prn);
    statusEl.textContent = st.label;
    statusEl.className = st.cls ? st.cls : '';
  }
  const series = lookupCnrSeries(ds, prn);
  const src = (ds && ds.cnrObsSource) ? ds.cnrObsSource : '';
  titleEl.textContent = 'C/N\u2080 — PRN ' + prn + (src ? ' — ' + src : '');
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const wCss = canvas.clientWidth || 680;
  const hCss = canvas.clientHeight || 240;
  canvas.width = Math.floor(wCss * dpr);
  canvas.height = Math.floor(hCss * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = '#0a0e14';
  ctx.fillRect(0, 0, wCss, hCss);
  if (!series || series.length === 0) {
    ctx.fillStyle = '#889';
    ctx.font = '14px system-ui,sans-serif';
    ctx.fillText('No C/N\u2080 (S*) observations for this PRN in embedded OBS.', 16, hCss / 2);
    hintEl.textContent =
      'Place rover .obs next to reference.csv, pass --obs, or enable CNR embed when generating HTML.';
    return;
  }
  const xs = series.map(p => Number(p[0]));
  const ys = series.map(p => Number(p[1]));
  let minX = Math.min.apply(null, xs);
  let maxX = Math.max.apply(null, xs);
  let minY = Math.min.apply(null, ys);
  let maxY = Math.max.apply(null, ys);
  if (maxY <= minY) { maxY = minY + 1; }
  const padL = 52;
  const padR = 14;
  const padT = 14;
  const padB = 36;
  const plotW = wCss - padL - padR;
  const plotH = hCss - padT - padB;
  function xScale(t) {
    return padL + ((t - minX) / (maxX - minX + 1e-9)) * plotW;
  }
  function yScale(v) {
    return padT + ((maxY - v) / (maxY - minY + 1e-9)) * plotH;
  }
  ctx.strokeStyle = '#334';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padL, padT);
  ctx.lineTo(padL, padT + plotH);
  ctx.lineTo(padL + plotW, padT + plotH);
  ctx.stroke();
  ctx.strokeStyle = '#3b7ddd';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (let i = 0; i < series.length; i++) {
    const px = xScale(xs[i]);
    const py = yScale(ys[i]);
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  }
  ctx.stroke();
  if (cursorTow != null && Number.isFinite(Number(cursorTow))) {
    const ct = Number(cursorTow);
    let cx = xScale(ct);
    if (cx < padL || cx > padL + plotW) {
      const wrapped = ct + ((towDeltaSec(minX, ct) < 0) ? 604800 : -604800);
      cx = xScale(wrapped);
    }
    ctx.strokeStyle = '#ff9f1c';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(cx, padT);
    ctx.lineTo(cx, padT + plotH);
    ctx.stroke();
  }
  ctx.fillStyle = '#aab';
  ctx.font = '11px monospace';
  ctx.fillText(minY.toFixed(0) + ' dB-Hz', 6, padT + plotH - 4);
  ctx.fillText(maxY.toFixed(0), 6, padT + 10);
  ctx.fillText('GPS TOW ' + minX.toFixed(0) + '\u2013' + maxX.toFixed(0) + ' s', padL, hCss - 10);
  hintEl.textContent =
    'Orange line: current epoch (GPS TOW). Scrub epoch here or on the main slider; click LOS/NLOS or multipath polyline for this PRN.';
}
function openCnrPanel(prn) {
  cnrChartPrn = normalizeCnrPrnKey(prn);
  const bd = document.getElementById('cnrBackdrop');
  if (bd) bd.classList.add('show');
  if (typeof syncEpochSlider === 'function') syncEpochSlider();
  refreshCnrChart();
}
function closeCnrPanel() {
  const bd = document.getElementById('cnrBackdrop');
  if (bd) bd.classList.remove('show');
  cnrChartPrn = null;
}
function refreshCnrChart() {
  if (!cnrChartPrn) return;
  const ds = datasets[currentDataset];
  const ep = ds.epochs[displayedEpochIndex];
  const tow = (ep && ep.gpsTow != null) ? ep.gpsTow : null;
  drawCnrChart(ds, cnrChartPrn, tow);
}
viewer.screenSpaceEventHandler.setInputAction(function (click) {
  const picked = viewer.scene.pick(click.position);
  if (!picked || !picked.id) return;
  const ent = picked.id;
  let prn = ent.vizPrn;
  if (
    !prn &&
    typeof pooledRayLines !== 'undefined' &&
    pooledRayLines.indexOf(ent) >= 0
  ) {
    const idx = pooledRayLines.indexOf(ent);
    const ds = datasets[currentDataset];
    const ep = ds.epochs[displayedEpochIndex];
    if (ep && ep.rays && ep.rays[idx]) prn = ep.rays[idx].prn;
  }
  if (
    !prn &&
    typeof pooledMultipathLines !== 'undefined' &&
    pooledMultipathLines.indexOf(ent) >= 0
  ) {
    const idx = pooledMultipathLines.indexOf(ent);
    const ds = datasets[currentDataset];
    const ep = ds.epochs[displayedEpochIndex];
    const reflAll = (ep && ep.reflections) || [];
    const reflList = reflAll.filter(function (mp) {
      return prnPassesFilters(mp.prn);
    });
    const chkMp = document.getElementById('chkShowMultipath');
    const showMpGeom = chkMp ? chkMp.checked : true;
    const nm = showMpGeom ? reflList.length : 0;
    if (idx >= 0 && idx < nm && reflList[idx]) prn = reflList[idx].prn;
  }
  if (!prn) return;
  openCnrPanel(prn);
}, Cesium.ScreenSpaceEventType.LEFT_CLICK);
(function wireCnrUi() {
  const btn = document.getElementById('btnCnrClose');
  if (btn) btn.addEventListener('click', closeCnrPanel);
  const bd = document.getElementById('cnrBackdrop');
  if (bd)
    bd.addEventListener('click', function (ev) {
      if (ev.target === bd) closeCnrPanel();
    });
  window.addEventListener('resize', function () {
    if (cnrChartPrn) refreshCnrChart();
  });
})();
"""


_ION_IAT_CHECK_JS = """<script>
(function () {{
  try {{
    var t = String(window.CESIUM_ION_TOKEN || '').trim();
    if (!t) return;
    var parts = t.split('.');
    if (parts.length !== 3) return;
    var b = parts[1].replace(/-/g, '+').replace(/_/g, '/');
    while (b.length % 4) b += '=';
    var j = JSON.parse(atob(b));
    if (j.iat && j.iat * 1000 > Date.now() + 120000) {{
      console.warn('[Cesium Ion] Token iat is in the future (' + new Date(j.iat * 1000).toISOString()
        + '). Requests may fail until then. Create a new token at https://ion.cesium.com/tokens');
    }}
  }} catch (e) {{}}
}})();
</script>
"""


def generate_html(datasets, output_path, cesium_ion_token: Optional[str] = None):
    """Generate standalone HTML with CesiumJS visualization."""
    export_sets = []
    for ds in datasets:
        export_sets.append({k: v for k, v in ds.items() if k != "pivot_ecef"})
    data_json = json.dumps(export_sets)
    tok = (cesium_ion_token or os.environ.get("CESIUM_ION_TOKEN", "") or "").strip()
    warn_if_ion_token_not_yet_valid(tok)
    token_script = ""
    if tok:
        token_script = (
            f"<script>window.CESIUM_ION_TOKEN = {json.dumps(tok)};</script>\n{_ION_IAT_CHECK_JS}"
        )

    html = f"""<!DOCTYPE html>
<html>
<head>
{token_script}<meta charset="utf-8">
<title>GPU Urban GNSS Signal Simulator — LOS/NLOS 3D Verification</title>
<style>
  html, body {{ margin: 0; padding: 0; width: 100%; height: 100%; overflow: hidden; }}
  #cesiumContainer {{ width: 100%; height: 100%; }}
  #fileProtocolWarn {{
    display: none;
    position: fixed; inset: 0; z-index: 1000;
    align-items: center; justify-content: center;
    background: #1a1d26; color: #e8e8e8;
    font-family: system-ui, Segoe UI, sans-serif; padding: 24px; box-sizing: border-box;
  }}
  #fileProtocolWarn .inner {{ max-width: 540px; line-height: 1.55; }}
  #fileProtocolWarn h2 {{ margin-top: 0; color: #fff; }}
  #fileProtocolWarn code {{ background: #2a3142; padding: 2px 8px; border-radius: 4px; word-break: break-all; }}
  #fileProtocolWarn pre {{ background: #0f1219; padding: 14px; border-radius: 8px; overflow-x: auto; font-size: 13px; }}
  #overlay {{
    position: absolute; top: 10px; left: 10px; z-index: 10;
    background: rgba(10, 15, 30, 0.85); color: #e0e0e0; padding: 12px 16px;
    border-radius: 8px; font-family: monospace; font-size: 13px;
    border: 1px solid #334; min-width: 220px;
  }}
  #overlay h3 {{ margin: 0 0 8px 0; color: #fff; font-size: 14px; }}
  .los {{ color: #00d4aa; }}
  .nlos {{ color: #ff6b6b; }}
  #progress {{ margin-top: 6px; font-size: 11px; color: #888; }}
  #playback {{
    margin-top: 10px; padding-top: 10px; border-top: 1px solid #445;
  }}
  #playback .row {{
    display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-top: 6px;
  }}
  #playback button {{
    cursor: pointer; font-family: monospace; font-size: 12px;
    padding: 6px 10px; border-radius: 6px; border: 1px solid #556;
    background: #252a38; color: #eee;
  }}
  #playback button:hover {{ background: #354050; }}
  #playback input[type="range"] {{ flex: 1; min-width: 140px; vertical-align: middle; }}
  #playback label {{ font-size: 11px; color: #aaa; }}
  #playback select {{
    background: #252a38; color: #eee; border: 1px solid #556;
    border-radius: 4px; padding: 4px 6px; font-family: monospace; font-size: 11px;
  }}
  #btnToggleOverlay {{
    position: fixed; top: 10px; right: 10px; z-index: 20;
    cursor: pointer; font-family: monospace; font-size: 11px;
    padding: 6px 12px; border-radius: 6px; border: 1px solid #556;
    background: rgba(22, 27, 38, 0.92); color: #ddd;
    box-shadow: 0 2px 8px rgba(0,0,0,0.35);
  }}
  #btnToggleOverlay:hover {{ background: rgba(42, 49, 66, 0.96); color: #fff; }}
  #selSatellite {{
    min-width: 8rem;
    min-height: 5.5rem;
    vertical-align: top;
  }}
  #cnrBackdrop {{
    display: none;
    position: fixed;
    inset: 0;
    z-index: 50;
    background: rgba(0, 0, 0, 0.5);
    align-items: center;
    justify-content: center;
  }}
  #cnrBackdrop.show {{ display: flex; }}
  #cnrPanel {{
    background: #121722;
    border: 1px solid #445;
    border-radius: 10px;
    padding: 14px 16px;
    width: min(720px, 92vw);
    max-height: 88vh;
    overflow: auto;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.55);
    color: #e0e0e0;
    font-family: system-ui, Segoe UI, sans-serif;
    font-size: 13px;
  }}
  #cnrPanel h4 {{ margin: 0 0 10px 0; color: #fff; font-size: 15px; }}
  #cnrCanvas {{
    width: 100%;
    height: 240px;
    display: block;
    background: #0a0e14;
    border-radius: 6px;
  }}
  #cnrHint {{ margin-top: 8px; font-size: 11px; color: #889; line-height: 1.45; }}
  #cnrEpochRow {{
    margin-bottom: 10px;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 8px;
  }}
  #cnrEpochRow button {{
    cursor: pointer;
    padding: 6px 12px;
    border-radius: 6px;
    border: 1px solid #556;
    background: #252a38;
    color: #eee;
    font-family: monospace;
    font-size: 12px;
    min-width: 4.5rem;
  }}
  #cnrEpochSlider {{ flex: 1; min-width: 160px; accent-color: #3b7ddd; }}
  #cnrSatStatus {{
    margin-left: auto;
    padding: 3px 8px;
    border-radius: 999px;
    border: 1px solid #445;
    background: #1b2333;
    color: #dbe5ff;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.02em;
    text-transform: uppercase;
  }}
  #cnrSatStatus.s-los {{ border-color: #15705e; color: #8ff3df; }}
  #cnrSatStatus.s-nlos {{ border-color: #8a3a3a; color: #ffb2b2; }}
  #cnrSatStatus.s-losmp {{ border-color: #8a5a1f; color: #ffd59b; }}
  #cnrSatStatus.s-nlosmp {{ border-color: #a24646; color: #ff9f9f; }}
</style>
</head>
<body>
<div id="fileProtocolWarn"><div class="inner">
  <h2>Use http://localhost — not file://</h2>
  <p>Browsers block Cesium <strong>Web Workers</strong> for pages opened as local files. You may see messages such as
  «Refused to cross-origin redirects of the top-level worker script» and «file: URLs are treated as unique security origins»:
  this is not your Ion token; it is the browser security model for the <code>file:</code> protocol.</p>
  <p>In this file’s folder (PowerShell or terminal):</p>
  <pre>python -m http.server 8765</pre>
  <p>Then open in your browser a URL such as <code>http://localhost:8765/cesium_odaiba.html</code> (use your actual file name from the directory listing).</p>
</div></div>
<div id="cesiumContainer"></div>
<div id="overlay">
  <h3>GPU Urban GNSS Signal Sim</h3>
  <div id="area">Loading...</div>
  <div id="stats"></div>
  <div id="progress"></div>
  <div id="vizSources" style="font-size:10px;color:#889;line-height:1.35;margin-top:8px;max-width:280px;">
    <strong>Different sources:</strong> LOS/NLOS in Python uses the local <strong>PLATEAU</strong> mesh.
    Below you see satellite imagery + terrain + <strong>OSM/Ion</strong> buildings (approximate): they will not match PLATEAU millimeter-for-millimeter.</div>
  <div id="playback">
    <div class="row">
      <button type="button" id="btnPlayPause">Pause</button>
      <button type="button" id="btnPrev" title="Previous epoch">◀ Back</button>
      <button type="button" id="btnPrev10" title="Skip back 10 epochs">◀ −10</button>
      <button type="button" id="btnNext" title="Next epoch">Next ▶</button>
      <button type="button" id="btnNext10" title="Skip forward 10 epochs">Next +10</button>
      <label>Speed <select id="speedSelect">
        <option value="4000">Slow</option>
        <option value="2500">Medium</option>
        <option value="1500" selected>Normal</option>
        <option value="800">Fast</option>
      </select></label>
      <label title="Keep the camera framed on the receiver each epoch (preserves your zoom once enabled)">
        <input type="checkbox" id="chkFollowRx"/> Follow RX
      </label>
      <label title="First-order specular satellite→bounce→RX; NLOS+reflection hides direct stub and draws multipath in red (--viz-multipath when generating)">
        <input type="checkbox" id="chkShowMultipath" checked/> Multipath
      </label>
    </div>
    <div class="row" id="osmToggleRow" style="display:none;">
      <label title="Hide Cesium Ion OSM 3D buildings (compare with PLATEAU GLB)">
        <input type="checkbox" id="chkHideOsm"/> Hide Ion OSM buildings
      </label>
    </div>
    <div class="row">
      <label for="epochSlider">Epoch</label>
      <input type="range" id="epochSlider" min="0" max="0" value="0" />
    </div>
    <div class="row" style="flex-wrap:wrap;align-items:flex-start;margin-top:8px;gap:12px;">
      <label style="display:flex;flex-direction:column;gap:4px;font-size:11px;color:#aab;">
        <span>Constellation</span>
        <select id="selConstellation" title="Limit viz to one GNSS system (RINEX letter)">
          <option value="">All</option>
          <option value="G">GPS (G)</option>
          <option value="R">GLONASS (R)</option>
          <option value="E">Galileo (E)</option>
          <option value="C">BeiDou (C)</option>
          <option value="J">QZSS (J)</option>
          <option value="I">NavIC (I)</option>
        </select>
      </label>
      <label style="display:flex;flex-direction:column;gap:4px;font-size:11px;color:#aab;">
        <span>Satellites <span style="color:#667;font-weight:normal;">(empty = all)</span></span>
        <select id="selSatellite" multiple size="5" title="Ctrl/Cmd multi-select; empty = every PRN in constellation scope"></select>
      </label>
    </div>
    <div id="playbackHint" style="font-size:10px;color:#7a8299;margin-top:6px;line-height:1.35;">
      While playing: approximate terrain and rays without PRN labels (smoother). When paused or dragging the slider: full detail.
      Click a <strong>LOS/NLOS ray</strong> or <strong>multipath polyline</strong> to open C/N₀ vs time from embedded OBS (when generated). With the C/N₀ panel open, use its epoch slider or ◀/▶ to move along the trajectory.
      <br/><strong>Epoch:</strong> <strong>\u2190</strong> / <strong>\u2192</strong> step one epoch when focus is not in an input or the satellite list.
      <strong>Tab</strong> toggles Play/Pause (same guard; <strong>Shift+Tab</strong> keeps normal focus move).
      <br/><strong>Camera:</strong> drag = orbit · <strong>Ctrl+drag</strong> = pan · wheel = zoom · right-drag = tilt.
    </div>
  </div>
</div>
<div id="cnrBackdrop" aria-hidden="true">
  <div id="cnrPanel" role="dialog" aria-labelledby="cnrTitle">
    <h4 id="cnrTitle">C/N₀</h4>
    <div id="cnrEpochRow">
      <span id="cnrEpochLabel" style="font-size:12px;color:#aab;min-width:5.5rem;"></span>
      <button type="button" id="btnCnrPrev" title="Previous epoch">◀ Prev</button>
      <button type="button" id="btnCnrNext" title="Next epoch">Next ▶</button>
      <label style="flex:1;min-width:160px;display:flex;align-items:center;gap:8px;font-size:11px;color:#aab;">
        <span style="white-space:nowrap;">Epoch</span>
        <input type="range" id="cnrEpochSlider" min="0" max="0" value="0" />
      </label>
      <span id="cnrSatStatus" title="Satellite state at current epoch">-</span>
    </div>
    <canvas id="cnrCanvas" width="680" height="240"></canvas>
    <div id="cnrHint"></div>
    <div style="margin-top:12px;text-align:right;">
      <button type="button" id="btnCnrClose" style="cursor:pointer;padding:6px 14px;border-radius:6px;border:1px solid #556;background:#252a38;color:#eee;font-family:monospace;font-size:12px;">Close</button>
    </div>
  </div>
</div>
<button type="button" id="btnToggleOverlay" title="Hide the entire info panel">Hide panel</button>
<script>
(function () {{
  var CESIUM_BASE = 'https://cesium.com/downloads/cesiumjs/releases/1.124/Build/Cesium/';
  function showFileWarn() {{
    var el = document.getElementById('fileProtocolWarn');
    if (el) el.style.display = 'flex';
    var ov = document.getElementById('overlay');
    if (ov) ov.style.display = 'none';
  }}
  function runApp() {{
// Ion token MUST be set before Viewer — otherwise default imagery/terrain stay blank.
const ionToken = String(window.CESIUM_ION_TOKEN || '').trim();
if (ionToken) {{
  Cesium.Ion.defaultAccessToken = ionToken;
}}

const viewer = new Cesium.Viewer('cesiumContainer', {{
  timeline: false,
  animation: false,
  baseLayerPicker: false,
  geocoder: false,
  homeButton: false,
  sceneModePicker: false,
  navigationHelpButton: false,
  fullscreenButton: false,
  infoBox: false,
  selectionIndicator: false,
}});

viewer.scene.globe.depthTestAgainstTerrain = true;
viewer.scene.globe.baseColor = Cesium.Color.fromCssColorString('#1a5270');
viewer.scene.globe.showGroundAtmosphere = true;
viewer.scene.globe.enableLighting = false;

(function () {{
  const sscc = viewer.scene.screenSpaceCameraController;
  sscc.translateEventTypes = [
    Cesium.CameraEventType.MIDDLE_DRAG,
    {{
      eventType: Cesium.CameraEventType.LEFT_DRAG,
      modifier: Cesium.KeyboardEventModifier.CTRL,
    }},
  ];
}})();

let osmBuildingsTileset = null;
if (ionToken) {{
  const osmRow = document.getElementById('osmToggleRow');
  if (osmRow) osmRow.style.display = 'flex';
  try {{
    viewer.clock.shouldAnimate = false;
    viewer.clock.currentTime = Cesium.JulianDate.fromIso8601('2024-06-15T03:30:00Z');
    viewer.scene.globe.enableLighting = true;
    viewer.scene.setTerrain(Cesium.Terrain.fromWorldTerrain({{
      requestWaterMask: true,
      requestVertexNormals: true,
    }}));
  }} catch (e) {{
    console.warn('World terrain failed:', e);
  }}
  Cesium.createOsmBuildingsAsync()
    .then(tileset => {{
      osmBuildingsTileset = tileset;
      viewer.scene.primitives.add(tileset);
      const chk = document.getElementById('chkHideOsm');
      if (chk && chk.checked) tileset.show = false;
    }})
    .catch(e => console.warn('OSM Buildings (check Ion token assets):', e));
}} else {{
  console.warn('No CESIUM_ION_TOKEN — terrain/buildings disabled. Open via http://localhost if Ion fails on file://');
}}

const datasets = {data_json};
const plateauSpec = datasets[0] && datasets[0].plateauModel;
if (plateauSpec && plateauSpec.url) {{
  const vs = document.getElementById('vizSources');
  if (vs) {{
    vs.innerHTML += '<br/><strong>PLATEAU GLB:</strong> Same mesh family as LOS/NLOS — hide OSM buildings to compare.';
  }}
  const pivotCart = Cesium.Cartesian3.fromDegrees(plateauSpec.lon, plateauSpec.lat, plateauSpec.height);
  const modelMatrix = Cesium.Matrix4.fromTranslation(pivotCart);
  const glbUrl = new URL(plateauSpec.url, window.location.href).href;
  Cesium.Model.fromGltfAsync({{
    url: glbUrl,
    modelMatrix,
    // POSITION holds raw ECEF offsets (parallel to world axes); default glTF Y-up correction would rotate them.
    upAxis: Cesium.Axis.Z,
    forwardAxis: Cesium.Axis.X,
  }})
    .then(model => {{
      viewer.scene.primitives.add(model);
    }})
    .catch(e => console.warn('PLATEAU GLB load failed (use http://localhost; GLB next to HTML):', e));
}}

(function () {{
  const chkHideOsm = document.getElementById('chkHideOsm');
  if (chkHideOsm) {{
    chkHideOsm.addEventListener('change', (e) => {{
      if (osmBuildingsTileset) osmBuildingsTileset.show = !e.target.checked;
    }});
  }}
  const chkMp = document.getElementById('chkShowMultipath');
  if (chkMp) {{
    chkMp.addEventListener('change', () => {{ paintFrame(); }});
  }}
  const selConst = document.getElementById('selConstellation');
  if (selConst) {{
    selConst.addEventListener('change', () => {{
      rebuildSatelliteSelectorOptions();
      paintFrame();
    }});
  }}
  const selSat = document.getElementById('selSatellite');
  if (selSat) {{
    selSat.addEventListener('change', () => {{ paintFrame(); }});
  }}
}})();
let currentDataset = 0;
let displayedEpochIndex = 0;
let rxEntityRef = null;
let pooledRayLines = [];
let pooledRayLabels = [];
let pooledMultipathLines = [];
let playing = true;
let stepMs = 1500;
const datasetGapMs = 2500;
let playbackTimer = null;
let followRxCam = false;
let lastFollowRxCartesian = null;
const followRxDeltaScratch = new Cesium.Cartesian3();

function clearPlaybackTimer() {{
  if (playbackTimer !== null) {{
    clearTimeout(playbackTimer);
    playbackTimer = null;
  }}
}}

function syncEpochSlider() {{
  const ds = datasets[currentDataset];
  const slider = document.getElementById('epochSlider');
  const n = ds.epochs.length;
  const maxIdx = Math.max(0, n - 1);
  const v = String(Math.min(displayedEpochIndex, maxIdx));
  slider.max = String(maxIdx);
  slider.value = v;
  const cnrSl = document.getElementById('cnrEpochSlider');
  if (cnrSl) {{
    cnrSl.max = String(maxIdx);
    cnrSl.value = v;
  }}
  const cnrLbl = document.getElementById('cnrEpochLabel');
  if (cnrLbl) {{
    cnrLbl.textContent = 'Epoch ' + (Math.min(displayedEpochIndex, maxIdx) + 1) + '/' + n;
  }}
}}

function paintFrame() {{
  const ds = datasets[currentDataset];
  displayedEpochIndex = Math.min(displayedEpochIndex, ds.epochs.length - 1);
  displayedEpochIndex = Math.max(0, displayedEpochIndex);
  showEpoch(ds, displayedEpochIndex);
  syncEpochSlider();
}}

function setPlayPauseUi() {{
  document.getElementById('btnPlayPause').textContent = playing ? 'Pause' : 'Play';
}}

function stepForwardCore() {{
  const ds = datasets[currentDataset];
  const n = ds.epochs.length;

  if (displayedEpochIndex >= n - 1 && datasets.length > 1) {{
    viewer.entities.removeAll();
    resetRayPools();
    currentDataset = (currentDataset + 1) % datasets.length;
    displayedEpochIndex = 0;
    drawTrajectory(datasets[currentDataset]);
    rebuildSatelliteSelectorOptions();
    const ep = datasets[currentDataset].epochs[0];
    viewer.camera.flyTo({{
      destination: Cesium.Cartesian3.fromDegrees(ep.rx[1], ep.rx[0], ep.rx[2] + 450),
      orientation: {{ heading: Cesium.Math.toRadians(20), pitch: Cesium.Math.toRadians(-38), roll: 0 }},
      duration: 2,
    }});
    lastFollowRxCartesian = null;
    return datasetGapMs;
  }}

  displayedEpochIndex = (displayedEpochIndex + 1) % n;
  return stepMs;
}}

function stepBackwardCore() {{
  const ds = datasets[currentDataset];
  const n = ds.epochs.length;

  if (displayedEpochIndex <= 0 && datasets.length > 1) {{
    viewer.entities.removeAll();
    resetRayPools();
    currentDataset = (currentDataset - 1 + datasets.length) % datasets.length;
    const prevDs = datasets[currentDataset];
    displayedEpochIndex = prevDs.epochs.length - 1;
    drawTrajectory(prevDs);
    rebuildSatelliteSelectorOptions();
    const rx = prevDs.epochs[displayedEpochIndex].rx;
    viewer.camera.flyTo({{
      destination: Cesium.Cartesian3.fromDegrees(rx[1], rx[0], rx[2] + 450),
      orientation: {{ heading: Cesium.Math.toRadians(20), pitch: Cesium.Math.toRadians(-38), roll: 0 }},
      duration: 2,
    }});
    lastFollowRxCartesian = null;
    return datasetGapMs;
  }}

  displayedEpochIndex = (displayedEpochIndex - 1 + n) % n;
  return stepMs;
}}

function goNextEpoch() {{
  const wait = stepForwardCore();
  paintFrame();
  return wait;
}}

function goPrevEpoch() {{
  stepBackwardCore();
  paintFrame();
}}

function tickPlayback() {{
  playbackTimer = null;
  if (!playing) return;
  const wait = goNextEpoch();
  if (playing && wait != null) {{
    playbackTimer = setTimeout(tickPlayback, wait);
  }}
}}

function resetRayPools() {{
  rxEntityRef = null;
  pooledRayLines = [];
  pooledRayLabels = [];
  pooledMultipathLines = [];
}}

function ensureRayLinePool(n) {{
  viewer.entities.suspendEvents();
  while (pooledRayLines.length < n) {{
    pooledRayLines.push(viewer.entities.add({{
      polyline: {{
        positions: new Cesium.ConstantProperty([
          new Cesium.Cartesian3(),
          new Cesium.Cartesian3(),
        ]),
        width: new Cesium.ConstantProperty(4),
        arcType: Cesium.ArcType.NONE,
        clampToGround: false,
        material: Cesium.Color.WHITE,
      }},
      show: false,
    }}));
  }}
  viewer.entities.resumeEvents();
}}

function ensureRayLabelPool(n) {{
  viewer.entities.suspendEvents();
  while (pooledRayLabels.length < n) {{
    pooledRayLabels.push(viewer.entities.add({{
      position: Cesium.Cartesian3.ZERO,
      label: {{
        text: '.',
        font: '11px monospace',
        fillColor: Cesium.Color.WHITE,
        outlineColor: Cesium.Color.BLACK,
        outlineWidth: 1,
        style: Cesium.LabelStyle.FILL_AND_OUTLINE,
        scale: 0.8,
      }},
      show: false,
    }}));
  }}
  viewer.entities.resumeEvents();
}}

function ensureMultipathLinePool(n) {{
  viewer.entities.suspendEvents();
  while (pooledMultipathLines.length < n) {{
    pooledMultipathLines.push(viewer.entities.add({{
      polyline: {{
        positions: new Cesium.ConstantProperty([
          new Cesium.Cartesian3(),
          new Cesium.Cartesian3(),
          new Cesium.Cartesian3(),
        ]),
        width: new Cesium.ConstantProperty(3),
        arcType: Cesium.ArcType.NONE,
        clampToGround: false,
        material: Cesium.Color.fromCssColorString('#ff9f1c'),
      }},
      show: false,
    }}));
  }}
  viewer.entities.resumeEvents();
}}

function updateEpochOverlay(ds, epoch, epochIdx) {{
  document.getElementById('area').textContent = ds.area;
  const rays = epoch.rays || [];
  const losCount = rays.filter(r => r.los && prnPassesFilters(r.prn)).length;
  const nlosCount = rays.filter(r => !r.los && prnPassesFilters(r.prn)).length;
  const mpList = (epoch.reflections || []).filter(mp => prnPassesFilters(mp.prn));
  let mpExtra = '';
  if (mpList.length > 0) {{
    mpExtra = ' | <span style="color:#ff9f1c">MP paths: ' + mpList.length + '</span>';
  }}
  document.getElementById('stats').innerHTML =
    '<span class="los">LOS: ' + losCount + '</span> | <span class="nlos">NLOS: ' + nlosCount + '</span>' + mpExtra;
  let towHint = '';
  if (epoch.gpsTow != null) {{
    towHint = '  GPS TOW ' + Number(epoch.gpsTow).toFixed(1) + ' s';
  }}
  document.getElementById('progress').textContent =
    'Epoch ' + (epochIdx+1) + '/' + ds.epochs.length + '  t=' + epoch.t.toFixed(0) + 's' + towHint;
}}

function showEpoch(ds, epochIdx) {{
  const epoch = ds.epochs[epochIdx];
  const rx = epoch.rx;
  const lon = rx[1];
  const lat = rx[0];
  const ellipsoidH = rx[2];
  const showSatLabels = !playing;

  function drawRxAndRays(rxH) {{
    viewer.entities.suspendEvents();
    try {{
      const rxPos = Cesium.Cartesian3.fromDegrees(lon, lat, rxH);
      if (!rxEntityRef) {{
        rxEntityRef = viewer.entities.add({{
          position: rxPos,
          point: {{ pixelSize: 10, color: Cesium.Color.WHITE, outlineColor: Cesium.Color.BLACK, outlineWidth: 2 }},
          label: {{ text: 'RX', font: '12px monospace', fillColor: Cesium.Color.WHITE,
                    style: Cesium.LabelStyle.FILL_AND_OUTLINE, outlineWidth: 2,
                    verticalOrigin: Cesium.VerticalOrigin.BOTTOM, pixelOffset: new Cesium.Cartesian2(0, -12) }},
        }});
      }} else {{
        rxEntityRef.position = rxPos;
      }}

      const nr = epoch.rays.length;
      ensureRayLinePool(nr);
      ensureRayLabelPool(nr);

      for (let i = 0; i < nr; i++) {{
        const ray = epoch.rays[i];
        const lineEnt = pooledRayLines[i];
        const passes = prnPassesFilters(ray.prn);
        lineEnt.vizPrn = ray.prn;
        lineEnt.allowPicking = passes;
        if (!passes) {{
          lineEnt.show = false;
          continue;
        }}
        const color = ray.los ? Cesium.Color.fromCssColorString('#00d4aa')
                              : Cesium.Color.fromCssColorString('#ff6b6b');
        const widthPx = ray.los ? 4 : 5;
        const p0 = Cesium.Cartesian3.fromDegrees(lon, lat, rxH);
        const p1 = Cesium.Cartesian3.fromDegrees(ray.end[1], ray.end[0], ray.end[2]);
        lineEnt.show = true;
        lineEnt.polyline.positions = new Cesium.ConstantProperty([p0, p1]);
        lineEnt.polyline.material = new Cesium.ColorMaterialProperty(color);
        lineEnt.polyline.width = new Cesium.ConstantProperty(widthPx);
      }}
      for (let i = nr; i < pooledRayLines.length; i++) {{
        pooledRayLines[i].show = false;
      }}

      const reflAll = epoch.reflections || [];
      const chkMp = document.getElementById('chkShowMultipath');
      const showMpGeom = chkMp ? chkMp.checked : true;
      const reflList = reflAll.filter(mp => prnPassesFilters(mp.prn));
      const nm = showMpGeom ? reflList.length : 0;
      ensureMultipathLinePool(nm);
      const orange = Cesium.Color.fromCssColorString('#ff9f1c');
      const mpNlosRed = Cesium.Color.fromCssColorString('#ff6b6b');
      for (let j = 0; j < nm; j++) {{
        const mp = reflList[j];
        const q0 = Cesium.Cartesian3.fromDegrees(mp.inc[1], mp.inc[0], mp.inc[2]);
        const q1 = Cesium.Cartesian3.fromDegrees(mp.bounce[1], mp.bounce[0], mp.bounce[2]);
        const q2 = Cesium.Cartesian3.fromDegrees(lon, lat, rxH);
        const ent = pooledMultipathLines[j];
        ent.vizPrn = mp.prn;
        ent.allowPicking = true;
        ent.show = true;
        ent.polyline.positions = new Cesium.ConstantProperty([q0, q1, q2]);
        const mpCol = mp.nlosMp ? mpNlosRed : orange;
        ent.polyline.material = new Cesium.ColorMaterialProperty(mpCol);
      }}
      for (let j = nm; j < pooledMultipathLines.length; j++) {{
        const hid = pooledMultipathLines[j];
        hid.show = false;
        hid.allowPicking = false;
      }}

      for (let i = 0; i < pooledRayLabels.length; i++) {{
        const le = pooledRayLabels[i];
        if (showSatLabels && i < nr) {{
          const ray = epoch.rays[i];
          const color = ray.los ? Cesium.Color.fromCssColorString('#00d4aa')
                                : Cesium.Color.fromCssColorString('#ff6b6b');
          le.show = true;
          le.position = Cesium.Cartesian3.fromDegrees(ray.end[1], ray.end[0], ray.end[2]);
          le.label.text = new Cesium.ConstantProperty('PRN' + ray.prn + (ray.los ? ' LOS' : ' NLOS'));
          le.label.fillColor = new Cesium.ConstantProperty(color);
        }} else {{
          le.show = false;
        }}
      }}
    }} finally {{
      viewer.entities.resumeEvents();
    }}

    updateEpochOverlay(ds, epoch, epochIdx);
    if (typeof refreshCnrChart === 'function') {{
      try {{ refreshCnrChart(); }} catch (e) {{}}
    }}
    applyFollowRxCameraIfEnabled(lon, lat, rxH);
    viewer.scene.requestRender();
  }}

  // RX + rays use trajectory ellipsoid height so polylines match Python LOS geometry.
  // Terrain-snapped heights disagree with PLATEAU GLB (also ellipsoid / survey heights).
  drawRxAndRays(ellipsoidH);
}}

function drawTrajectory(ds) {{
  const flat = [];
  ds.trajectory.forEach(p => {{ flat.push(p[1], p[0]); }});
  viewer.entities.add({{
    polyline: {{
      positions: Cesium.Cartesian3.fromDegreesArray(flat),
      width: 4,
      material: Cesium.Color.YELLOW.withAlpha(1.0),
      clampToGround: true,
    }},
  }});
}}

function frameCameraOnRx(rx) {{
  viewer.camera.cancelFlight();
  const center = Cesium.Cartesian3.fromDegrees(rx[1], rx[0], rx[2]);
  const sphere = new Cesium.BoundingSphere(center, 8000);
  viewer.camera.viewBoundingSphere(
    sphere,
    new Cesium.HeadingPitchRange(
      Cesium.Math.toRadians(48),
      Cesium.Math.toRadians(-36),
      4200
    )
  );
}}

function applyFollowRxCameraIfEnabled(lonDeg, latDeg, heightM) {{
  if (!followRxCam) return;
  viewer.camera.cancelFlight();
  const newCenter = Cesium.Cartesian3.fromDegrees(lonDeg, latDeg, heightM);
  if (!lastFollowRxCartesian) {{
    lastFollowRxCartesian = Cesium.Cartesian3.clone(newCenter);
    return;
  }}
  Cesium.Cartesian3.subtract(newCenter, lastFollowRxCartesian, followRxDeltaScratch);
  Cesium.Cartesian3.add(viewer.camera.positionWC, followRxDeltaScratch, viewer.camera.positionWC);
  Cesium.Cartesian3.clone(newCenter, lastFollowRxCartesian);
}}

drawTrajectory(datasets[0]);
frameCameraOnRx(datasets[0].epochs[0].rx);
displayedEpochIndex = 0;
rebuildSatelliteSelectorOptions();
paintFrame();
setPlayPauseUi();

document.getElementById('btnPlayPause').addEventListener('click', () => {{
  playing = !playing;
  setPlayPauseUi();
  clearPlaybackTimer();
  if (!playing) {{
    paintFrame();
  }}
  if (playing) {{
    playbackTimer = setTimeout(tickPlayback, stepMs);
  }}
}});

document.getElementById('btnNext').addEventListener('click', () => {{
  clearPlaybackTimer();
  goNextEpoch();
  if (playing) {{
    playbackTimer = setTimeout(tickPlayback, stepMs);
  }}
}});

document.getElementById('btnNext10').addEventListener('click', () => {{
  clearPlaybackTimer();
  for (let k = 0; k < 10; k++) {{
    stepForwardCore();
  }}
  paintFrame();
  if (playing) {{
    playbackTimer = setTimeout(tickPlayback, stepMs);
  }}
}});

document.getElementById('btnPrev').addEventListener('click', () => {{
  clearPlaybackTimer();
  goPrevEpoch();
  if (playing) {{
    playbackTimer = setTimeout(tickPlayback, stepMs);
  }}
}});

document.getElementById('btnPrev10').addEventListener('click', () => {{
  clearPlaybackTimer();
  for (let k = 0; k < 10; k++) {{
    stepBackwardCore();
  }}
  paintFrame();
  if (playing) {{
    playbackTimer = setTimeout(tickPlayback, stepMs);
  }}
}});

document.getElementById('speedSelect').addEventListener('change', (e) => {{
  stepMs = parseInt(e.target.value, 10);
}});

document.getElementById('chkFollowRx').addEventListener('change', (e) => {{
  followRxCam = e.target.checked;
  if (!followRxCam) {{
    lastFollowRxCartesian = null;
  }}
  if (followRxCam) {{
    paintFrame();
  }}
}});

let overlayPanelVisible = true;
document.getElementById('btnToggleOverlay').addEventListener('click', () => {{
  overlayPanelVisible = !overlayPanelVisible;
  const ov = document.getElementById('overlay');
  const btn = document.getElementById('btnToggleOverlay');
  ov.style.display = overlayPanelVisible ? '' : 'none';
  btn.textContent = overlayPanelVisible ? 'Hide panel' : 'Show panel';
}});

document.getElementById('epochSlider').addEventListener('input', (e) => {{
  clearPlaybackTimer();
  displayedEpochIndex = parseInt(e.target.value, 10);
  paintFrame();
  if (playing) {{
    playbackTimer = setTimeout(tickPlayback, stepMs);
  }}
}});

const cnrEpochSliderEl = document.getElementById('cnrEpochSlider');
if (cnrEpochSliderEl) {{
  cnrEpochSliderEl.addEventListener('input', (e) => {{
    clearPlaybackTimer();
    displayedEpochIndex = parseInt(e.target.value, 10);
    paintFrame();
    if (playing) {{
      playbackTimer = setTimeout(tickPlayback, stepMs);
    }}
  }});
}}
const btnCnrPrevEl = document.getElementById('btnCnrPrev');
if (btnCnrPrevEl) {{
  btnCnrPrevEl.addEventListener('click', () => {{
    clearPlaybackTimer();
    goPrevEpoch();
    if (playing) {{
      playbackTimer = setTimeout(tickPlayback, stepMs);
    }}
  }});
}}
const btnCnrNextEl = document.getElementById('btnCnrNext');
if (btnCnrNextEl) {{
  btnCnrNextEl.addEventListener('click', () => {{
    clearPlaybackTimer();
    goNextEpoch();
    if (playing) {{
      playbackTimer = setTimeout(tickPlayback, stepMs);
    }}
  }});
}}

window.addEventListener('keydown', (ev) => {{
  const t = ev.target;
  const tag = t && t.tagName;
  const inField = tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT';

  if (ev.key === 'Tab') {{
    if (inField || ev.shiftKey) return;
    ev.preventDefault();
    playing = !playing;
    setPlayPauseUi();
    clearPlaybackTimer();
    if (!playing) {{
      paintFrame();
    }}
    if (playing) {{
      playbackTimer = setTimeout(tickPlayback, stepMs);
    }}
    return;
  }}

  if (ev.key !== 'ArrowLeft' && ev.key !== 'ArrowRight') return;
  if (inField) return;
  ev.preventDefault();
  clearPlaybackTimer();
  if (ev.key === 'ArrowLeft') goPrevEpoch();
  else goNextEpoch();
  if (playing) {{
    playbackTimer = setTimeout(tickPlayback, stepMs);
  }}
}}, {{ passive: false }});

if (playing) {{
  playbackTimer = setTimeout(tickPlayback, stepMs);
}}
  }}
  if (location.protocol === 'file:') {{
    showFileWarn();
    return;
  }}
  var link = document.createElement('link');
  link.rel = 'stylesheet';
  link.href = CESIUM_BASE + 'Widgets/widgets.css';
  document.head.appendChild(link);
  var scr = document.createElement('script');
  scr.src = CESIUM_BASE + 'Cesium.js';
  scr.onload = function () {{ runApp(); }};
  scr.onerror = function () {{
    var inner = document.querySelector('#fileProtocolWarn .inner');
    if (inner) inner.innerHTML = '<p>Unable to load Cesium from the CDN (network or firewall).</p>';
    showFileWarn();
  }};
  document.head.appendChild(scr);
}})();
</script>
</body>
</html>"""
    html = html.replace(
        "const followRxDeltaScratch = new Cesium.Cartesian3();\n\nfunction clearPlaybackTimer",
        "const followRxDeltaScratch = new Cesium.Cartesian3();\n\n"
        + _VIEWER_EXTENSIONS_JS
        + "\n\nfunction clearPlaybackTimer",
    )
    _out_dir = os.path.dirname(os.path.abspath(output_path))
    if _out_dir:
        os.makedirs(_out_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(html)
    print(f"HTML: {output_path}")


def record_video(html_path, video_path, duration_ms=30000):
    """Record video of the visualization using Playwright."""
    import subprocess
    script = f"""
const {{ chromium }} = require('playwright');
(async () => {{
  const browser = await chromium.launch({{ headless: true }});
  const context = await browser.newContext({{
    viewport: {{ width: 1280, height: 720 }},
    recordVideo: {{ dir: '{os.path.dirname(video_path)}', size: {{ width: 1280, height: 720 }} }},
  }});
  const page = await context.newPage();
  await page.goto('file://{os.path.abspath(html_path)}');
  await page.waitForTimeout({duration_ms});
  await context.close();
  await browser.close();
  // Rename video
  const fs = require('fs');
  const files = fs.readdirSync('{os.path.dirname(video_path)}')
    .filter(f => f.endsWith('.webm'));
  if (files.length > 0) {{
    fs.renameSync(
      '{os.path.dirname(video_path)}/' + files[files.length - 1],
      '{video_path}'
    );
  }}
}})();
"""
    script_path = video_path + ".js"
    with open(script_path, "w") as f:
        f.write(script)

    print(f"Recording video ({duration_ms / 1000:.0f}s)...")
    result = subprocess.run(
        ["node", script_path],
        capture_output=True, text=True, timeout=duration_ms / 1000 + 30,
        cwd=os.path.dirname(os.path.abspath(__file__)) + "/..",
    )
    if result.returncode != 0:
        print(f"  stderr: {result.stderr[:500]}")
    os.unlink(script_path)
    if os.path.exists(video_path):
        size_mb = os.path.getsize(video_path) / 1e6
        print(f"Video: {video_path} ({size_mb:.1f} MB)")
    else:
        print("Video recording failed — HTML file can still be opened in a browser")


def _export_plateau_glb_sidecar(
    ds: dict,
    plateau_dir: str,
    plateau_zone: int,
    out_html: str,
    *,
    radius_m: float,
    max_triangles: int,
    glb_out_explicit: str,
    full_mesh: bool,
) -> None:
    """Populate ``ds[\"plateauModel\"]`` and write ``*_plateau.glb`` next to HTML."""
    pivot = np.asarray(ds["pivot_ecef"], dtype=np.float64).reshape(3)
    loader = PlateauLoader(zone=int(plateau_zone))
    building = loader.load_directory(plateau_dir)
    tri_ecef = building.triangles
    base = os.path.splitext(os.path.abspath(out_html))[0]
    glb_path = glb_out_explicit.strip() or (base + "_plateau.glb")
    n_kept, n_tot = export_plateau_roi_glb(
        tri_ecef,
        pivot,
        glb_path,
        radius_m=float(radius_m),
        max_triangles=int(max_triangles),
    )
    lat, lon, h = ecef_to_lla(float(pivot[0]), float(pivot[1]), float(pivot[2]))
    ds["plateauModel"] = {
        "url": os.path.basename(glb_path),
        "lat": math.degrees(lat),
        "lon": math.degrees(lon),
        "height": float(h),
    }
    print(f"PLATEAU GLB: {glb_path} ({n_kept} / {n_tot} triangles in ROI)")


def _export_triangles_glb_sidecar(
    ds: dict,
    triangles_npy: str,
    out_html: str,
    *,
    radius_m: float,
    max_triangles: int,
    glb_out_explicit: str,
    full_mesh: bool,
) -> None:
    """Populate ``ds['plateauModel']`` and write GLB from ``--triangles-npy`` mesh."""
    pivot = np.asarray(ds["pivot_ecef"], dtype=np.float64).reshape(3)
    tri_ecef = np.asarray(np.load(triangles_npy), dtype=np.float64)
    if tri_ecef.ndim != 3 or tri_ecef.shape[1:] != (3, 3):
        raise ValueError(f"--triangles-npy must have shape [N,3,3], got {tri_ecef.shape}")
    base = os.path.splitext(os.path.abspath(out_html))[0]
    glb_path = glb_out_explicit.strip() or (base + "_mesh.glb")
    n_kept, n_tot = export_plateau_roi_glb(
        tri_ecef,
        pivot,
        glb_path,
        radius_m=float(radius_m),
        max_triangles=int(max_triangles),
        full_mesh=bool(full_mesh),
    )
    lat, lon, h = ecef_to_lla(float(pivot[0]), float(pivot[1]), float(pivot[2]))
    ds["plateauModel"] = {
        "url": os.path.basename(glb_path),
        "lat": math.degrees(lat),
        "lon": math.degrees(lon),
        "height": float(h),
    }
    print(f"Mesh GLB: {glb_path} ({n_kept} / {n_tot} triangles in ROI)")


def _default_legacy_paths():
    root = os.path.join(os.path.dirname(__file__), "..")
    return {
        "shinjuku_plateau": os.path.join(root, "experiments/data/plateau_shinjuku"),
        "shinjuku_ref": os.path.join(root, "experiments/data/urbannav/Shinjuku/reference.csv"),
        "odaiba_plateau": os.path.join(root, "experiments/data/plateau_odaiba"),
        "odaiba_ref": os.path.join(root, "experiments/data/urbannav/Odaiba/reference.csv"),
        "out_dir": os.path.join(root, "experiments/results/los_nlos_verification"),
    }


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(description="CesiumJS LOS/NLOS ray visualization (UrbanNav + PLATEAU)")
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Build default Shinjuku + Odaiba HTML under experiments/results/ (old behaviour)",
    )
    parser.add_argument("--area-name", type=str, default="Area", help="Label shown in the viewer overlay")
    parser.add_argument("--plateau-dir", type=str, default="", help="Directory with PLATEAU CityGML tiles")
    parser.add_argument(
        "--triangles-npy",
        type=str,
        default="",
        help="Alternative mesh input: .npy triangles with shape [N,3,3] in ECEF meters",
    )
    parser.add_argument("--reference-csv", type=str, default="", help="UrbanNav reference.csv (ECEF trajectory)")
    parser.add_argument("--plateau-zone", type=int, default=9, help="PLATEAU plane zone (Tokyo area = 9)")
    parser.add_argument("--n-epochs", type=int, default=12, help="Number of epochs along trajectory")
    parser.add_argument(
        "--epoch-min-interval-s",
        type=float,
        default=0.0,
        help="If > 0, minimum GPS Δt [s] between consecutive viz epochs (spread along run). "
        "Default 0: evenly spaced row indices (legacy).",
    )
    parser.add_argument(
        "--traj-step",
        type=float,
        default=300.0,
        help="CSV row stride (≥0). 0 = every row (same as 1); default 300. Non-integers rounded.",
    )
    parser.add_argument(
        "--nav",
        type=str,
        default="",
        help="RINEX .nav file (default: base.nav in the same folder as --reference-csv)",
    )
    parser.add_argument(
        "--elevation-mask-deg",
        type=float,
        default=10.0,
        help="Minimum elevation [deg] for rays and LOS stats (default 10; 0 = horizon; -90 disables mask)",
    )
    parser.add_argument(
        "--eph-batch-chunk",
        type=int,
        default=64,
        help="Epochs per Ephemeris.compute_batch call (default 64). Smaller windows keep more PRNs on long runs; "
        "0 = one batch for the whole timeline (fastest GPU path, strictest PRN intersection).",
    )
    parser.add_argument("--out-html", type=str, default="cesium_los_nlos.html", help="Output HTML path")
    parser.add_argument(
        "--export-plateau-glb",
        action="store_true",
        help="Write ROI-filtered PLATEAU mesh as GLB beside HTML and load it in Cesium (needs http://)",
    )
    parser.add_argument(
        "--export-mesh-glb",
        action="store_true",
        help="Write ROI-filtered GLB from --triangles-npy mesh beside HTML and load it in Cesium",
    )
    parser.add_argument(
        "--plateau-glb-radius-m",
        type=float,
        default=1200.0,
        help="Include triangles whose centroid is within this distance [m] of trajectory centroid (default 1200)",
    )
    parser.add_argument(
        "--plateau-glb-max-tris",
        type=int,
        default=300_000,
        help="Maximum triangles in GLB after ROI filter / subsample cap (default 300000)",
    )
    parser.add_argument(
        "--plateau-glb-full-mesh",
        action="store_true",
        help="Export all PLATEAU triangles (ignore radius); still subsampled to --plateau-glb-max-tris",
    )
    parser.add_argument(
        "--plateau-glb-out",
        type=str,
        default="",
        help="Explicit output path for GLB (default: <out-html>_plateau.glb)",
    )
    parser.add_argument(
        "--cesium-ion-token",
        type=str,
        default=os.environ.get("CESIUM_ION_TOKEN", ""),
        help="Cesium ion token for terrain + OSM buildings (or set CESIUM_ION_TOKEN)",
    )
    parser.add_argument("--record-video", action="store_true", help="Try Playwright screen recording (needs Node)")
    parser.add_argument(
        "--viz-multipath",
        action="store_true",
        help="Include first-order specular reflection paths (sat→wall→RX) in HTML; needs BVH/raytrace multipath",
    )
    parser.add_argument(
        "--multipath-min-delay-m",
        type=float,
        default=0.5,
        help="Only draw reflections with excess path delay ≥ this [m] (default 0.5)",
    )
    parser.add_argument(
        "--filter-by-obs",
        action="store_true",
        help="Hide rays for satellites without code pseudorange (C*) in rover RINEX .obs at nearest epoch",
    )
    parser.add_argument(
        "--obs",
        type=str,
        default="",
        help="Rover RINEX observation file (default: rover_ublox.obs / rover_trimble.obs / rover.obs next to reference CSV)",
    )
    parser.add_argument(
        "--obs-match-tol-s",
        type=float,
        default=1.0,
        help="Max |ΔGPS TOW| [s] between trajectory epoch and matched OBS epoch (default 1)",
    )
    parser.add_argument(
        "--obs-strict",
        action="store_true",
        help="If no OBS epoch within tolerance, hide all satellites (default: keep NAV-only visibility)",
    )
    parser.add_argument(
        "--no-cnr-embed",
        action="store_true",
        help="Do not embed C/N₀ (S*) time series from rover .obs for ray-click chart",
    )
    parser.add_argument(
        "--atmo-bending-lite",
        action="store_true",
        help="Apply lightweight apparent-elevation correction for LOS visibility masking.",
    )
    parser.add_argument(
        "--atmo-pressure-hpa",
        type=float,
        default=1010.0,
        help="Pressure [hPa] for the lite atmospheric-bending model.",
    )
    parser.add_argument(
        "--atmo-temp-c",
        type=float,
        default=10.0,
        help="Temperature [deg C] for the lite atmospheric-bending model.",
    )
    parser.add_argument(
        "--cnr-max-points",
        type=int,
        default=2500,
        help="Max OBS samples per satellite for embedded C/N₀ series (downsampled; default 2500)",
    )
    args = parser.parse_args(argv)

    try:
        traj_stride = trajectory_row_stride(float(args.traj_step))
    except ValueError as e:
        parser.error(str(e))

    tok = args.cesium_ion_token.strip() if args.cesium_ion_token else ""
    nav_opt = args.nav.strip() if getattr(args, "nav", "") else ""
    el_mask = float(getattr(args, "elevation_mask_deg", 10.0))
    eph_chunk = int(getattr(args, "eph_batch_chunk", 64))
    epoch_gap_s = float(getattr(args, "epoch_min_interval_s", 0.0))

    obs_tol = float(getattr(args, "obs_match_tol_s", 1.0))
    obs_strict = bool(getattr(args, "obs_strict", False))
    obs_opt = (getattr(args, "obs", "") or "").strip()
    filter_obs = bool(getattr(args, "filter_by_obs", False))
    embed_cnr = not bool(getattr(args, "no_cnr_embed", False))
    cnr_max_pts = max(50, int(getattr(args, "cnr_max_points", 2500)))
    atmo_bending_lite = bool(getattr(args, "atmo_bending_lite", False))
    atmo_pressure_hpa = float(getattr(args, "atmo_pressure_hpa", 1010.0))
    atmo_temp_c = float(getattr(args, "atmo_temp_c", 10.0))

    def _obs_path_for_ref(ref_csv: str) -> Optional[str]:
        if not filter_obs:
            return None
        pth = _resolve_obs_path(ref_csv, obs_opt)
        if pth is None:
            parser.error(
                "--filter-by-obs needs a rover .obs next to the reference CSV "
                "(try rover_ublox.obs, rover_trimble.obs, rover.obs) or pass --obs PATH"
            )
        return pth

    if args.legacy:
        p = _default_legacy_paths()
        os.makedirs(p["out_dir"], exist_ok=True)
        shinjuku = compute_all_epochs(
            "Shinjuku",
            p["shinjuku_plateau"],
            p["shinjuku_ref"],
            n_epochs=max(1, args.n_epochs),
            step=traj_stride,
            plateau_zone=9,
            nav_path=nav_opt or None,
            elevation_mask_deg=el_mask,
            eph_batch_chunk=eph_chunk,
            epoch_min_interval_s=epoch_gap_s,
            obs_rover_path=_obs_path_for_ref(p["shinjuku_ref"]),
            obs_match_tol_s=obs_tol,
            obs_strict=obs_strict,
            cnr_obs_path=_cnr_obs_path_for_embed(p["shinjuku_ref"], obs_opt, embed=embed_cnr),
            cnr_max_points_per_prn=cnr_max_pts,
            atmo_bending_lite=atmo_bending_lite,
            atmo_pressure_hpa=atmo_pressure_hpa,
            atmo_temp_c=atmo_temp_c,
        )
        odaiba = compute_all_epochs(
            "Odaiba",
            p["odaiba_plateau"],
            p["odaiba_ref"],
            n_epochs=max(1, args.n_epochs),
            step=traj_stride,
            plateau_zone=9,
            nav_path=nav_opt or None,
            elevation_mask_deg=el_mask,
            eph_batch_chunk=eph_chunk,
            epoch_min_interval_s=epoch_gap_s,
            obs_rover_path=_obs_path_for_ref(p["odaiba_ref"]),
            obs_match_tol_s=obs_tol,
            obs_strict=obs_strict,
            cnr_obs_path=_cnr_obs_path_for_embed(p["odaiba_ref"], obs_opt, embed=embed_cnr),
            cnr_max_points_per_prn=cnr_max_pts,
            atmo_bending_lite=atmo_bending_lite,
            atmo_pressure_hpa=atmo_pressure_hpa,
            atmo_temp_c=atmo_temp_c,
        )
        html_path = os.path.join(p["out_dir"], "los_nlos_3d.html")
        if getattr(args, "export_plateau_glb", False):
            print("Note: --export-plateau-glb is ignored with --legacy (two areas); run without --legacy per scene.")
        generate_html([shinjuku, odaiba], html_path, cesium_ion_token=tok or None)
        print(f"HTML: {html_path}")
        if args.record_video:
            video_path = os.path.join(p["out_dir"], "los_nlos_3d.webm")
            try:
                record_video(html_path, video_path, duration_ms=35000)
            except Exception as e:
                print(f"Video recording skipped: {e}")
        if not tok:
            print("Tip: set CESIUM_ION_TOKEN or --cesium-ion-token for terrain + OSM buildings.")
        return

    if not args.reference_csv:
        parser.error("--reference-csv is required unless --legacy is set.")
    if bool(args.plateau_dir) == bool(args.triangles_npy):
        parser.error("Provide exactly one of --plateau-dir or --triangles-npy.")

    ds = compute_all_epochs(
        args.area_name,
        args.plateau_dir,
        args.reference_csv,
        n_epochs=max(1, args.n_epochs),
        step=traj_stride,
        plateau_zone=args.plateau_zone,
        nav_path=nav_opt or None,
        elevation_mask_deg=el_mask,
        eph_batch_chunk=eph_chunk,
        epoch_min_interval_s=epoch_gap_s,
        viz_multipath=bool(getattr(args, "viz_multipath", False)),
        multipath_min_delay_m=float(getattr(args, "multipath_min_delay_m", 0.5)),
        obs_rover_path=_obs_path_for_ref(args.reference_csv),
        obs_match_tol_s=obs_tol,
        obs_strict=obs_strict,
        cnr_obs_path=_cnr_obs_path_for_embed(args.reference_csv, obs_opt, embed=embed_cnr),
        cnr_max_points_per_prn=cnr_max_pts,
        atmo_bending_lite=atmo_bending_lite,
        atmo_pressure_hpa=atmo_pressure_hpa,
        atmo_temp_c=atmo_temp_c,
        triangles_npy=(args.triangles_npy.strip() or None),
    )
    out_html = os.path.abspath(args.out_html)
    _out_dir = os.path.dirname(out_html)
    if _out_dir:
        os.makedirs(_out_dir, exist_ok=True)
    export_glb = bool(getattr(args, "export_plateau_glb", False) or getattr(args, "export_mesh_glb", False))
    if export_glb:
        try:
            if args.triangles_npy:
                _export_triangles_glb_sidecar(
                    ds,
                    args.triangles_npy,
                    out_html,
                    radius_m=float(getattr(args, "plateau_glb_radius_m", 1200.0)),
                    max_triangles=int(getattr(args, "plateau_glb_max_tris", 300_000)),
                    glb_out_explicit=str(getattr(args, "plateau_glb_out", "") or ""),
                    full_mesh=bool(getattr(args, "plateau_glb_full_mesh", False)),
                )
            else:
                _export_plateau_glb_sidecar(
                    ds,
                    args.plateau_dir,
                    args.plateau_zone,
                    out_html,
                    radius_m=float(getattr(args, "plateau_glb_radius_m", 1200.0)),
                    max_triangles=int(getattr(args, "plateau_glb_max_tris", 300_000)),
                    glb_out_explicit=str(getattr(args, "plateau_glb_out", "") or ""),
                    full_mesh=bool(getattr(args, "plateau_glb_full_mesh", False)),
                )
        except Exception as e:
            print(f"PLATEAU GLB export failed: {e}")
            ds["plateauModel"] = None
    generate_html([ds], out_html, cesium_ion_token=tok or None)
    print(f"HTML: {out_html}")
    if not tok:
        print("Tip: set CESIUM_ION_TOKEN or pass --cesium-ion-token for world terrain + OSM buildings.")
    if args.record_video:
        try:
            record_video(out_html, out_html.replace(".html", ".webm"), duration_ms=25000)
        except Exception as e:
            print(f"Video recording skipped: {e}")


if __name__ == "__main__":
    main()
