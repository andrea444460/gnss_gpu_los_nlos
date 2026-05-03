#!/usr/bin/env python3
"""Generate 3D LOS/NLOS geometry visualization + recording helper.

Creates a standalone HTML file using CesiumJS (free tier) that shows:
  - 3D globe with terrain + buildings (Cesium OSM Buildings)
  - Receiver trajectory (yellow line)
  - Per-epoch satellite rays (green=LOS, red=NLOS)
  - Animated flythrough

Then uses Playwright to record a video of the visualization.

The receiver trajectory is sourced from UrbanNav. Satellite geometry uses the
**broadcast RINEX navigation** file next to the trajectory (``base.nav``) or
``--nav``; constellation blocks **G/R/E/J/C/I** from that file are parsed (SBAS omitted).
LOS/NLOS rays
are computed in Python against the **local PLATEAU CityGML mesh** (--plateau-dir).
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

During **Play**, Cesium skips per-epoch ``sampleTerrainMostDetailed`` (uses ``globe.getHeight`` only) and omits PRN endpoint labels so redraw stays light; **Pause** or the epoch slider refines terrain and restores labels.

Satellite rays use a **reused pool of polyline ``Entity`` objects**: each epoch only updates positions/materials instead of destroying and rebuilding geometry.
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from typing import Optional

import numpy as np

from gnss_gpu.ephemeris import Ephemeris
from gnss_gpu.io.nav_rinex import read_nav_rinex_multi
from gnss_gpu.io.plateau import PlateauLoader
from gnss_gpu.bvh import BVHAccelerator
from gnss_gpu.urban_signal_sim import (
    UrbanSignalSimulator,
    _sat_elevation_azimuth,
    ecef_to_lla,
)

# RINEX 3 navigation systems retained for this viewer (SBAS ``S`` excluded for now).
RINEX_NAV_SYSTEMS_VIZ = ("G", "R", "E", "J", "C", "I")


def _normalize_csv_row(row: dict) -> dict:
    return {k.strip(): v for k, v in row.items()}


def load_trajectory(csv_path, step=200):
    """Return positions, GPS TOW times, and raw CSV row count.

    Points loaded ≈ ``ceil(n_csv_rows / step)`` (every ``step``-th row, zero-based).
    """
    with open(csv_path, encoding="utf-8") as f:
        rows = [_normalize_csv_row(r) for r in csv.DictReader(f)]
    n_csv = len(rows)
    positions, times = [], []
    for i in range(0, n_csv, step):
        r = rows[i]
        positions.append(
            [
                float(r["ECEF X (m)"]),
                float(r["ECEF Y (m)"]),
                float(r["ECEF Z (m)"]),
            ]
        )
        times.append(float(r["GPS TOW (s)"]))
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

    ``epoch_min_interval_s``: if > 0, minimum GPS-time spacing between consecutive viz epochs
    (uniform subsample along a greedy ladder); if ``0`` (default), epochs are evenly spaced in
    **row index** after ``step`` (same as legacy ``np.linspace``).
    """
    print(f"[{area_name}] Loading PLATEAU...")
    loader = PlateauLoader(zone=int(plateau_zone))
    building = loader.load_directory(plateau_dir)
    print(f"  {len(building.triangles)} triangles")

    print(f"[{area_name}] Building BVH...")
    bvh = BVHAccelerator.from_building_model(building)
    has_bvh_los_batch = hasattr(bvh, "check_los_batch")
    print(f"  BVH LOS batch path: {'enabled' if has_bvh_los_batch else 'disabled (fallback per-epoch)'}")

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

    nav_file = _resolve_nav_path(traj_csv, nav_path)
    print(f"[{area_name}] Loading broadcast ephemeris: {nav_file}")
    nav_messages = read_nav_rinex_multi(nav_file, systems=RINEX_NAV_SYSTEMS_VIZ)
    eph = Ephemeris(nav_messages)
    prn_catalog = eph.available_prns
    print(f"  Satellites in NAV: {len(prn_catalog)}")
    print(f"  Elevation mask: {elevation_mask_deg:g}° above horizon")

    usim = UrbanSignalSimulator(
        building_model=bvh,
        noise_floor_db=-35,
        elevation_mask_deg=elevation_mask_deg,
    )

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
        if (
            sat_rows is not None
            and has_bvh_los_batch
            and len(chunk_idx) > 0
        ):
            rx_blk = np.ascontiguousarray(positions[chunk_idx], dtype=np.float64)
            sat_blk = np.ascontiguousarray(sat_rows, dtype=np.float64)
            n_b = sat_blk.shape[0]
            n_sat_blk = sat_blk.shape[1]
            el_batch = np.zeros((n_b, n_sat_blk), dtype=np.float64)
            for i in range(n_b):
                el_batch[i], _ = _sat_elevation_azimuth(rx_blk[i], sat_blk[i])
            visible_batch = el_batch >= usim.elevation_mask_rad
            sat_work = sat_blk.copy()
            sat_work[~visible_batch] = np.nan
            los_batch = np.asarray(bvh.check_los_batch(rx_blk, sat_work), dtype=bool)

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

            n_sat = int(np.asarray(sat_ecef).reshape(-1, 3).shape[0])
            if n_sat == 0:
                lat, lon, alt = ecef_to_lla(*rx)
                epochs_data.append({
                    "rx": [math.degrees(lat), math.degrees(lon), alt + 2],
                    "rays": [],
                    "n_los": 0,
                    "n_nlos": 0,
                    "t": t,
                })
                if fi % log_every == 0 or fi == len(idx_list):
                    print(f"  [{fi}/{len(idx_list)}] t={t:.0f}s — no ephemeris at this TOW")
                continue

            if los_batch is not None:
                visible = visible_batch[local_i]
                el = el_batch[local_i]
                is_los = np.ones(n_sat, dtype=bool)
                vis_idx = np.where(visible)[0]
                if len(vis_idx) > 0:
                    is_los[vis_idx] = los_batch[local_i, vis_idx]
                n_los_ep = int(np.sum(is_los & visible))
                n_nlos_ep = int(np.sum(~is_los & visible))
            else:
                result = usim.compute_epoch(
                    rx_ecef=rx,
                    sat_ecef=sat_ecef,
                    sat_clk=sat_clk,
                    prn_list=used_prns_i,
                )
                visible = result["visible"]
                el = result["elevations"]
                is_los = result["is_los"]
                n_los_ep = int(result["n_los"])
                n_nlos_ep = int(result["n_nlos"])

            lat, lon, alt = ecef_to_lla(*rx)
            epoch = {
                "rx": [math.degrees(lat), math.degrees(lon), alt + 2],
                "rays": [],
                "n_los": n_los_ep,
                "n_nlos": n_nlos_ep,
                "t": t,
            }
            for i in range(n_sat):
                if not visible[i]:
                    continue
                direction = sat_ecef[i] - rx
                dist = np.linalg.norm(direction)
                ray_end = rx + direction / dist * 5000
                re_lat, re_lon, re_alt = ecef_to_lla(*ray_end)
                epoch["rays"].append({
                    "prn": _sat_display_id(used_prns_i[i]),
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
        traj.append([math.degrees(lat), math.degrees(lon), alt + 1])

    return {"epochs": epochs_data, "trajectory": traj, "area": area_name}


def generate_html(datasets, output_path, cesium_ion_token: Optional[str] = None):
    """Generate standalone HTML with CesiumJS visualization."""
    data_json = json.dumps(datasets)
    tok = (cesium_ion_token or os.environ.get("CESIUM_ION_TOKEN", "") or "").strip()
    token_script = ""
    if tok:
        token_script = f"<script>window.CESIUM_ION_TOKEN = {json.dumps(tok)};</script>\n"

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
    </div>
    <div class="row">
      <label for="epochSlider">Epoch</label>
      <input type="range" id="epochSlider" min="0" max="0" value="0" />
    </div>
    <div id="playbackHint" style="font-size:10px;color:#7a8299;margin-top:6px;line-height:1.35;">
      While playing: approximate terrain and rays without PRN labels (smoother). When paused or dragging the slider: full detail.
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

if (ionToken) {{
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
    .then(tileset => {{ viewer.scene.primitives.add(tileset); }})
    .catch(e => console.warn('OSM Buildings (check Ion token assets):', e));
}} else {{
  console.warn('No CESIUM_ION_TOKEN — terrain/buildings disabled. Open via http://localhost if Ion fails on file://');
}}

const datasets = {data_json};
let currentDataset = 0;
let displayedEpochIndex = 0;
let rxEntityRef = null;
let pooledRayLines = [];
let pooledRayLabels = [];
let playing = true;
let stepMs = 1500;
const datasetGapMs = 2500;
let playbackTimer = null;
let terrainSnapGeneration = 0;
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
  slider.max = String(Math.max(0, n - 1));
  slider.value = String(Math.min(displayedEpochIndex, n - 1));
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

function updateEpochOverlay(ds, epoch, epochIdx) {{
  document.getElementById('area').textContent = ds.area;
  const losCount = epoch.rays.filter(r => r.los).length;
  const nlosCount = epoch.rays.filter(r => !r.los).length;
  document.getElementById('stats').innerHTML =
    '<span class="los">LOS: ' + losCount + '</span> | <span class="nlos">NLOS: ' + nlosCount + '</span>';
  document.getElementById('progress').textContent =
    'Epoch ' + (epochIdx+1) + '/' + ds.epochs.length + '  t=' + epoch.t.toFixed(0) + 's';
}}

function showEpoch(ds, epochIdx) {{
  terrainSnapGeneration += 1;
  const gen = terrainSnapGeneration;
  const epoch = ds.epochs[epochIdx];
  const rx = epoch.rx;
  const lon = rx[1];
  const lat = rx[0];
  const ellipsoidH = rx[2];
  const RX_ABOVE_GROUND_M = 2.0;
  // Avoid sampleTerrainMostDetailed during auto-play (network + double redraw); pause/scrub uses HQ path.
  const fastPlayback = playing;
  const showSatLabels = !playing;

  function drawRxAndRays(rxH) {{
    if (gen !== terrainSnapGeneration) return;

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
        const color = ray.los ? Cesium.Color.fromCssColorString('#00d4aa')
                              : Cesium.Color.fromCssColorString('#ff6b6b');
        const widthPx = ray.los ? 4 : 5;
        const p0 = Cesium.Cartesian3.fromDegrees(lon, lat, rxH);
        const p1 = Cesium.Cartesian3.fromDegrees(ray.end[1], ray.end[0], ray.end[2]);
        const lineEnt = pooledRayLines[i];
        lineEnt.show = true;
        lineEnt.polyline.positions = new Cesium.ConstantProperty([p0, p1]);
        lineEnt.polyline.material = new Cesium.ColorMaterialProperty(color);
        lineEnt.polyline.width = new Cesium.ConstantProperty(widthPx);
      }}
      for (let i = nr; i < pooledRayLines.length; i++) {{
        pooledRayLines[i].show = false;
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
    applyFollowRxCameraIfEnabled(lon, lat, rxH);
    viewer.scene.requestRender();
  }}

  const tp = viewer.scene.globe.terrainProvider;
  if (!tp || tp instanceof Cesium.EllipsoidTerrainProvider) {{
    drawRxAndRays(ellipsoidH);
    return;
  }}

  if (fastPlayback) {{
    const qh = viewer.scene.globe.getHeight(Cesium.Cartographic.fromDegrees(lon, lat));
    drawRxAndRays(Cesium.defined(qh) ? qh + RX_ABOVE_GROUND_M : ellipsoidH);
    return;
  }}

  const quickH = viewer.scene.globe.getHeight(Cesium.Cartographic.fromDegrees(lon, lat));
  if (Cesium.defined(quickH)) {{
    drawRxAndRays(quickH + RX_ABOVE_GROUND_M);
  }}

  Cesium.sampleTerrainMostDetailed(tp, [Cesium.Cartographic.fromDegrees(lon, lat, 0)])
    .then(function (updated) {{
      if (gen !== terrainSnapGeneration) return;
      drawRxAndRays(updated[0].height + RX_ABOVE_GROUND_M);
    }})
    .catch(function () {{
      if (gen !== terrainSnapGeneration) return;
      if (!Cesium.defined(quickH)) {{
        drawRxAndRays(ellipsoidH);
        updateEpochOverlay(ds, epoch, epochIdx);
      }}
    }});
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
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
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
        "--cesium-ion-token",
        type=str,
        default=os.environ.get("CESIUM_ION_TOKEN", ""),
        help="Cesium ion token for terrain + OSM buildings (or set CESIUM_ION_TOKEN)",
    )
    parser.add_argument("--record-video", action="store_true", help="Try Playwright screen recording (needs Node)")
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
        )
        html_path = os.path.join(p["out_dir"], "los_nlos_3d.html")
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

    if not args.plateau_dir or not args.reference_csv:
        parser.error("--plateau-dir and --reference-csv are required unless --legacy is set.")

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
    )
    out_html = os.path.abspath(args.out_html)
    _out_dir = os.path.dirname(out_html)
    if _out_dir:
        os.makedirs(_out_dir, exist_ok=True)
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
