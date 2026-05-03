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
By default there is **no elevation mask** (``--elevation-mask-deg -90``).
Ephemeris positions are evaluated with ``Ephemeris.compute_batch`` in chunks (``--eph-batch-chunk``, default 64).
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
from gnss_gpu.urban_signal_sim import UrbanSignalSimulator, ecef_to_lla

# RINEX 3 navigation systems retained for this viewer (SBAS ``S`` excluded for now).
RINEX_NAV_SYSTEMS_VIZ = ("G", "R", "E", "J", "C", "I")


def _normalize_csv_row(row: dict) -> dict:
    return {k.strip(): v for k, v in row.items()}


def load_trajectory(csv_path, step=200):
    with open(csv_path, encoding="utf-8") as f:
        rows = [_normalize_csv_row(r) for r in csv.DictReader(f)]
    positions, times = [], []
    for i in range(0, len(rows), step):
        r = rows[i]
        positions.append(
            [
                float(r["ECEF X (m)"]),
                float(r["ECEF Y (m)"]),
                float(r["ECEF Z (m)"]),
            ]
        )
        times.append(float(r["GPS TOW (s)"]))
    return np.array(positions), np.array(times)


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


def compute_all_epochs(
    area_name,
    plateau_dir,
    traj_csv,
    n_epochs=15,
    step=200,
    plateau_zone=9,
    nav_path: Optional[str] = None,
    elevation_mask_deg: float = -90.0,
    eph_batch_chunk: int = 64,
):
    """Compute LOS/NLOS for all epochs and return visualization data.

    Satellite positions use :meth:`~gnss_gpu.ephemeris.Ephemeris.compute_batch`
    in windows of ``eph_batch_chunk`` epochs (default 64). Batch keeps only PRNs
    with a valid ephemeris block for *every* epoch in that window; smaller windows
    preserve more satellites on long runs. Set ``eph_batch_chunk`` to 0 for one
    batch over the whole timeline (fastest, strictest PRN intersection).
    """
    print(f"[{area_name}] Loading PLATEAU...")
    loader = PlateauLoader(zone=int(plateau_zone))
    building = loader.load_directory(plateau_dir)
    print(f"  {len(building.triangles)} triangles")

    print(f"[{area_name}] Building BVH...")
    bvh = BVHAccelerator.from_building_model(building)

    positions, times = load_trajectory(traj_csv, step=step)
    indices = np.linspace(0, len(positions) - 1, n_epochs, dtype=int)

    nav_file = _resolve_nav_path(traj_csv, nav_path)
    print(f"[{area_name}] Loading broadcast ephemeris: {nav_file}")
    nav_messages = read_nav_rinex_multi(nav_file, systems=RINEX_NAV_SYSTEMS_VIZ)
    eph = Ephemeris(nav_messages)
    prn_catalog = eph.available_prns
    print(f"  Satellites in NAV: {len(prn_catalog)}")

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
        else:
            sat_rows = sat_b
            clk_rows = clk_b

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

            result = usim.compute_epoch(
                rx_ecef=rx,
                sat_ecef=sat_ecef,
                sat_clk=sat_clk,
                prn_list=used_prns_i,
            )

            lat, lon, alt = ecef_to_lla(*rx)
            epoch = {
                "rx": [math.degrees(lat), math.degrees(lon), alt + 2],
                "rays": [],
                "n_los": result["n_los"],
                "n_nlos": result["n_nlos"],
                "t": t,
            }
            for i in range(n_sat):
                if not result["visible"][i]:
                    continue
                direction = sat_ecef[i] - rx
                dist = np.linalg.norm(direction)
                ray_end = rx + direction / dist * 5000
                re_lat, re_lon, re_alt = ecef_to_lla(*ray_end)
                epoch["rays"].append({
                    "prn": _sat_display_id(used_prns_i[i]),
                    "los": bool(result["is_los"][i]),
                    "el": float(np.degrees(result["elevations"][i])),
                    "end": [math.degrees(re_lat), math.degrees(re_lon), re_alt],
                })
            epochs_data.append(epoch)
            if fi % log_every == 0 or fi == len(idx_list):
                print(
                    f"  [{fi}/{len(idx_list)}] t={t:.0f}s "
                    f"LOS={result['n_los']} NLOS={result['n_nlos']} "
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
</style>
</head>
<body>
<div id="fileProtocolWarn"><div class="inner">
  <h2>Serve http://localhost — non file://</h2>
  <p>I browser bloccano i <strong>Web Worker</strong> di Cesium per pagine aperte come file locale. Compare quindi
  «Refused to cross-origin redirects of the top-level worker script» e «file: URLs are treated as unique security origins»:
  non è il token Ion, è la sicurezza del protocollo <code>file:</code>.</p>
  <p>Nella cartella di questo file (PowerShell o terminale):</p>
  <pre>python -m http.server 8765</pre>
  <p>Poi apri nel browser un URL del tipo <code>http://localhost:8765/cesium_odaiba.html</code> (nome file come sulla lista).</p>
</div></div>
<div id="cesiumContainer"></div>
<div id="overlay">
  <h3>GPU Urban GNSS Signal Sim</h3>
  <div id="area">Loading...</div>
  <div id="stats"></div>
  <div id="progress"></div>
  <div id="vizSources" style="font-size:10px;color:#889;line-height:1.35;margin-top:8px;max-width:280px;">
    <strong>Fonti diverse:</strong> LOS/NLOS in Python usa la mesh <strong>PLATEAU</strong> locale.
    Qui sotto vedi satellitare + terreno + edifici <strong>OSM/Ion</strong> (approssimati): non coincidono al mm con PLATEAU.</div>
  <div id="playback">
    <div class="row">
      <button type="button" id="btnPlayPause">Pausa</button>
      <button type="button" id="btnPrev" title="Epoch precedente">◀ Indietro</button>
      <button type="button" id="btnNext" title="Epoch successiva">Avanti ▶</button>
      <label>Velocità <select id="speedSelect">
        <option value="4000">Lenta</option>
        <option value="2500">Media</option>
        <option value="1500" selected>Normale</option>
        <option value="800">Veloce</option>
      </select></label>
    </div>
    <div class="row">
      <label for="epochSlider">Epoch</label>
      <input type="range" id="epochSlider" min="0" max="0" value="0" />
    </div>
  </div>
</div>
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
let rayEntities = [];
let playing = true;
let stepMs = 1500;
const datasetGapMs = 2500;
let playbackTimer = null;
let terrainSnapGeneration = 0;

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
  document.getElementById('btnPlayPause').textContent = playing ? 'Pausa' : 'Play';
}}

function goNextEpoch() {{
  const ds = datasets[currentDataset];
  const n = ds.epochs.length;

  if (displayedEpochIndex >= n - 1 && datasets.length > 1) {{
    viewer.entities.removeAll();
    rayEntities = [];
    currentDataset = (currentDataset + 1) % datasets.length;
    displayedEpochIndex = 0;
    drawTrajectory(datasets[currentDataset]);
    const ep = datasets[currentDataset].epochs[0];
    viewer.camera.flyTo({{
      destination: Cesium.Cartesian3.fromDegrees(ep.rx[1], ep.rx[0], ep.rx[2] + 450),
      orientation: {{ heading: Cesium.Math.toRadians(20), pitch: Cesium.Math.toRadians(-38), roll: 0 }},
      duration: 2,
    }});
    paintFrame();
    return datasetGapMs;
  }}

  displayedEpochIndex = (displayedEpochIndex + 1) % n;
  paintFrame();
  return stepMs;
}}

function goPrevEpoch() {{
  const ds = datasets[currentDataset];
  const n = ds.epochs.length;

  if (displayedEpochIndex <= 0 && datasets.length > 1) {{
    viewer.entities.removeAll();
    rayEntities = [];
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
    paintFrame();
    return;
  }}

  displayedEpochIndex = (displayedEpochIndex - 1 + n) % n;
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

function clearRays() {{
  rayEntities.forEach(e => viewer.entities.remove(e));
  rayEntities = [];
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
  clearRays();
  const epoch = ds.epochs[epochIdx];
  const rx = epoch.rx;
  const lon = rx[1];
  const lat = rx[0];
  const ellipsoidH = rx[2];
  const RX_ABOVE_GROUND_M = 2.0;

  function drawRxAndRays(rxH) {{
    if (gen !== terrainSnapGeneration) return;
    rayEntities.push(viewer.entities.add({{
      position: Cesium.Cartesian3.fromDegrees(lon, lat, rxH),
      point: {{ pixelSize: 10, color: Cesium.Color.WHITE, outlineColor: Cesium.Color.BLACK, outlineWidth: 2 }},
      label: {{ text: 'RX', font: '12px monospace', fillColor: Cesium.Color.WHITE,
                style: Cesium.LabelStyle.FILL_AND_OUTLINE, outlineWidth: 2,
                verticalOrigin: Cesium.VerticalOrigin.BOTTOM, pixelOffset: new Cesium.Cartesian2(0, -12) }},
    }}));

    epoch.rays.forEach(ray => {{
      const color = ray.los ? Cesium.Color.fromCssColorString('#00d4aa').withAlpha(1.0)
                            : Cesium.Color.fromCssColorString('#ff6b6b').withAlpha(1.0);
      const width = ray.los ? 2 : 3;

      rayEntities.push(viewer.entities.add({{
        polyline: {{
          positions: Cesium.Cartesian3.fromDegreesArrayHeights([
            lon, lat, rxH,
            ray.end[1], ray.end[0], ray.end[2]
          ]),
          width: Math.max(6, width + 4),
          material: color,
          arcType: Cesium.ArcType.NONE,
          perPositionHeight: true,
          clampToGround: false,
        }},
      }}));

      rayEntities.push(viewer.entities.add({{
        position: Cesium.Cartesian3.fromDegrees(ray.end[1], ray.end[0], ray.end[2]),
        label: {{
          text: 'PRN' + ray.prn + (ray.los ? ' LOS' : ' NLOS'),
          font: '11px monospace',
          fillColor: color,
          style: Cesium.LabelStyle.FILL_AND_OUTLINE,
          outlineWidth: 1,
          scale: 0.8,
        }},
      }}));
    }});

    updateEpochOverlay(ds, epoch, epochIdx);
  }}

  const tp = viewer.scene.globe.terrainProvider;
  if (!tp || tp instanceof Cesium.EllipsoidTerrainProvider) {{
    drawRxAndRays(ellipsoidH);
    return;
  }}

  const quickH = viewer.scene.globe.getHeight(Cesium.Cartographic.fromDegrees(lon, lat));
  if (Cesium.defined(quickH)) {{
    drawRxAndRays(quickH + RX_ABOVE_GROUND_M);
  }}

  Cesium.sampleTerrainMostDetailed(tp, [Cesium.Cartographic.fromDegrees(lon, lat, 0)])
    .then(function (updated) {{
      if (gen !== terrainSnapGeneration) return;
      clearRays();
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

drawTrajectory(datasets[0]);
frameCameraOnRx(datasets[0].epochs[0].rx);
displayedEpochIndex = 0;
paintFrame();
setPlayPauseUi();

document.getElementById('btnPlayPause').addEventListener('click', () => {{
  playing = !playing;
  setPlayPauseUi();
  clearPlaybackTimer();
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

document.getElementById('btnPrev').addEventListener('click', () => {{
  clearPlaybackTimer();
  goPrevEpoch();
  if (playing) {{
    playbackTimer = setTimeout(tickPlayback, stepMs);
  }}
}});

document.getElementById('speedSelect').addEventListener('change', (e) => {{
  stepMs = parseInt(e.target.value, 10);
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
    if (inner) inner.innerHTML = '<p>Impossibile caricare Cesium dal CDN (rete o firewall).</p>';
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
    parser.add_argument("--traj-step", type=int, default=300, help="Sample every N rows from reference.csv")
    parser.add_argument(
        "--nav",
        type=str,
        default="",
        help="RINEX .nav file (default: base.nav in the same folder as --reference-csv)",
    )
    parser.add_argument(
        "--elevation-mask-deg",
        type=float,
        default=-90.0,
        help="Minimum elevation [deg] for including satellites (-90 ≈ no mask; typical receivers use ~5–10)",
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

    tok = args.cesium_ion_token.strip() if args.cesium_ion_token else ""
    nav_opt = args.nav.strip() if getattr(args, "nav", "") else ""
    el_mask = float(getattr(args, "elevation_mask_deg", -90.0))
    eph_chunk = int(getattr(args, "eph_batch_chunk", 64))

    if args.legacy:
        p = _default_legacy_paths()
        os.makedirs(p["out_dir"], exist_ok=True)
        shinjuku = compute_all_epochs(
            "Shinjuku",
            p["shinjuku_plateau"],
            p["shinjuku_ref"],
            n_epochs=12,
            step=300,
            plateau_zone=9,
            nav_path=nav_opt or None,
            elevation_mask_deg=el_mask,
            eph_batch_chunk=eph_chunk,
        )
        odaiba = compute_all_epochs(
            "Odaiba",
            p["odaiba_plateau"],
            p["odaiba_ref"],
            n_epochs=12,
            step=300,
            plateau_zone=9,
            nav_path=nav_opt or None,
            elevation_mask_deg=el_mask,
            eph_batch_chunk=eph_chunk,
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
        step=max(1, args.traj_step),
        plateau_zone=args.plateau_zone,
        nav_path=nav_opt or None,
        elevation_mask_deg=el_mask,
        eph_batch_chunk=eph_chunk,
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
