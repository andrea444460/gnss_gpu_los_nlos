#!/usr/bin/env python3
"""Build a split HTML viewer: Cesium 3D mesh + existing Folium LOS/NLOS map."""

from __future__ import annotations

import argparse
import base64
import json
import math
from pathlib import Path

import numpy as np


WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = 2.0 * WGS84_F - WGS84_F * WGS84_F


def ecef_to_lla_deg(x: float, y: float, z: float) -> tuple[float, float, float]:
    lon = math.atan2(y, x)
    p = math.sqrt(x * x + y * y)
    lat = math.atan2(z, p * (1.0 - WGS84_E2))
    for _ in range(10):
        sin_lat = math.sin(lat)
        n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
        lat = math.atan2(z + WGS84_E2 * n * sin_lat, p)
    sin_lat = math.sin(lat)
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    alt = p / max(1e-12, math.cos(lat)) - n
    return math.degrees(lat), math.degrees(lon), alt


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--triangles-npy", type=Path, required=True, help="ECEF triangles [N,3,3] for pivot")
    p.add_argument("--glb-path", type=Path, required=True, help="Building GLB (relative to --out-html dir)")
    p.add_argument("--folium-html", type=Path, required=True, help="Existing Folium map HTML")
    p.add_argument("--out-html", type=Path, required=True, help="Output combined HTML path")
    p.add_argument("--cesium-ion-token", type=str, default="", help="Optional Cesium ion token")
    p.add_argument("--title", type=str, default="Area LOS/NLOS + 3D buildings")
    p.add_argument(
        "--embed-glb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Embed GLB as data URL so file:// open works (default: on)",
    )
    p.add_argument(
        "--regenerate-glb",
        action="store_true",
        help="Rebuild GLB from --triangles-npy before generating HTML",
    )
    p.add_argument(
        "--max-glb-triangles",
        type=int,
        default=0,
        help="Triangle cap when --regenerate-glb (0 = keep all; low values break building shapes)",
    )
    return p.parse_args()


def _rel_path(from_dir: Path, target: Path) -> str:
    target = target.resolve()
    from_dir = from_dir.resolve()
    try:
        return target.relative_to(from_dir).as_posix()
    except ValueError:
        return target.as_posix()


def build_html(cfg: dict) -> str:
    cfg_json = json.dumps(cfg)
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>{cfg["title"]}</title>
  <script src="https://cesium.com/downloads/cesiumjs/releases/1.120/Build/Cesium/Cesium.js"></script>
  <link href="https://cesium.com/downloads/cesiumjs/releases/1.120/Build/Cesium/Widgets/widgets.css" rel="stylesheet" />
  <style>
    html, body {{
      width: 100%;
      height: 100%;
      margin: 0;
      padding: 0;
      overflow: hidden;
      background: #111;
      font-family: sans-serif;
    }}
    #root {{
      display: grid;
      grid-template-columns: 1fr 6px 1fr;
      width: 100%;
      height: 100%;
    }}
    #cesiumPane, #mapPane {{
      position: relative;
      min-width: 0;
      min-height: 0;
      overflow: hidden;
    }}
    #cesiumContainer {{
      width: 100%;
      height: 100%;
    }}
    #foliumFrame {{
      width: 100%;
      height: 100%;
      border: 0;
      background: #fff;
    }}
    #splitter {{
      cursor: col-resize;
      background: #333;
    }}
    #toolbar {{
      position: absolute;
      top: 10px;
      left: 10px;
      z-index: 1000;
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }}
    .btn {{
      background: rgba(0,0,0,0.65);
      color: #fff;
      border: 1px solid #666;
      border-radius: 4px;
      padding: 6px 10px;
      font-size: 12px;
      cursor: pointer;
    }}
    .btn:hover {{ background: rgba(30,30,30,0.9); }}
    #info {{
      position: absolute;
      bottom: 10px;
      left: 10px;
      z-index: 1000;
      background: rgba(0,0,0,0.65);
      color: #fff;
      padding: 8px 10px;
      font-size: 12px;
      max-width: 420px;
    }}
    #mapToolbar {{
      position: absolute;
      top: 10px;
      right: 10px;
      z-index: 1000;
      display: flex;
      gap: 8px;
    }}
    .maximized {{ grid-template-columns: 1fr !important; }}
    .maximized #mapPane, .maximized #splitter {{ display: none; }}
    .maximized-map #cesiumPane, .maximized-map #splitter {{ display: none; }}
    .maximized-map {{ grid-template-columns: 1fr !important; }}
  </style>
</head>
<body>
  <div id="root">
    <div id="cesiumPane">
      <div id="toolbar">
        <button class="btn" id="btn3dOnly">3D full</button>
        <button class="btn" id="btnSplit">Split</button>
        <button class="btn" id="btnMapOnly">Map full</button>
        <button class="btn" id="btnToggleMesh">Toggle mesh</button>
      </div>
      <div id="info">Loading 3D mesh…</div>
      <div id="cesiumContainer"></div>
    </div>
    <div id="splitter" title="Drag to resize"></div>
    <div id="mapPane">
      <div id="mapToolbar">
        <span class="btn" style="cursor:default;">2D LOS/NLOS map</span>
      </div>
      <iframe id="foliumFrame" title="LOS/NLOS map"></iframe>
    </div>
  </div>
  <script>
    const cfg = {cfg_json};
    if (cfg.token) {{
      Cesium.Ion.defaultAccessToken = cfg.token;
    }}

    const root = document.getElementById("root");
    const splitter = document.getElementById("splitter");
    const foliumFrame = document.getElementById("foliumFrame");
    foliumFrame.src = cfg.foliumUrl;

    const viewer = new Cesium.Viewer("cesiumContainer", {{
      timeline: false,
      animation: false,
      sceneModePicker: true,
      baseLayerPicker: true,
      geocoder: true,
      homeButton: true,
    }});

    const pivot = Cesium.Cartesian3.fromDegrees(cfg.lon, cfg.lat, cfg.alt);
    const modelMatrix = Cesium.Matrix4.fromTranslation(pivot);
    let meshModel = null;

    const glbSource = cfg.glbDataUrl || cfg.glbUrl;
    Cesium.Model.fromGltfAsync({{
      url: glbSource,
      modelMatrix,
      scale: 1.0,
      upAxis: Cesium.Axis.Z,
      forwardAxis: Cesium.Axis.X,
    }}).then((model) => {{
      meshModel = model;
      viewer.scene.primitives.add(model);
      const glbLabel = cfg.glbDataUrl ? "embedded GLB" : cfg.glbUrl;
      document.getElementById("info").textContent =
        "3D: " + glbLabel + " | 2D: " + cfg.foliumUrl;
    }}).catch((err) => {{
      console.error(err);
      document.getElementById("info").textContent = "GLB load failed: " + String(err);
    }});

    viewer.camera.flyTo({{
      destination: Cesium.Cartesian3.fromDegrees(cfg.lon, cfg.lat, cfg.alt + 350),
      orientation: {{ heading: 0.0, pitch: -0.65, roll: 0.0 }},
    }});

    document.getElementById("btn3dOnly").onclick = () => {{
      root.classList.remove("maximized-map");
      root.classList.add("maximized");
    }};
    document.getElementById("btnMapOnly").onclick = () => {{
      root.classList.remove("maximized");
      root.classList.add("maximized-map");
    }};
    document.getElementById("btnSplit").onclick = () => {{
      root.classList.remove("maximized", "maximized-map");
      root.style.gridTemplateColumns = "1fr 6px 1fr";
    }};
    document.getElementById("btnToggleMesh").onclick = () => {{
      if (meshModel) meshModel.show = !meshModel.show;
    }};

    let dragging = false;
    splitter.addEventListener("mousedown", () => {{ dragging = true; }});
    window.addEventListener("mouseup", () => {{ dragging = false; }});
    window.addEventListener("mousemove", (ev) => {{
      if (!dragging) return;
      const rect = root.getBoundingClientRect();
      const x = Math.max(0.2, Math.min(0.8, (ev.clientX - rect.left) / rect.width));
      root.style.gridTemplateColumns = `${{x}}fr 6px ${{1 - x}}fr`;
    }});
  </script>
</body>
</html>
"""


def main() -> None:
    args = _parse_args()
    tri = np.asarray(np.load(args.triangles_npy), dtype=np.float64)
    if tri.ndim != 3 or tri.shape[1:] != (3, 3):
        raise ValueError(f"--triangles-npy must have shape [N,3,3], got {tri.shape}")

    pivot = tri.reshape(-1, 3).mean(axis=0)
    lat, lon, alt = ecef_to_lla_deg(float(pivot[0]), float(pivot[1]), float(pivot[2]))

    if bool(args.regenerate_glb):
        import sys

        _py = Path(__file__).resolve().parents[1] / "python"
        if str(_py) not in sys.path:
            sys.path.insert(0, str(_py))
        from gnss_gpu.viz.plateau_glb import export_plateau_roi_glb  # noqa: E402

        cap = int(args.max_glb_triangles)
        if cap <= 0:
            cap = int(tri.shape[0])
        kept, total = export_plateau_roi_glb(
            tri,
            pivot,
            args.glb_path,
            full_mesh=True,
            max_triangles=cap,
        )
        print(f"[regenerate-glb] kept={kept}/{total} path={args.glb_path}")
        if kept < total:
            print(
                "[regenerate-glb] warning: triangle subsampling may produce scattered shards in 3D view",
                flush=True,
            )

    out_dir = args.out_html.parent.resolve()
    glb_rel = _rel_path(out_dir, args.glb_path)
    cfg: dict = {
        "title": str(args.title),
        "glbUrl": glb_rel,
        "foliumUrl": _rel_path(out_dir, args.folium_html),
        "lat": lat,
        "lon": lon,
        "alt": alt,
        "token": str(args.cesium_ion_token),
        "glbDataUrl": "",
    }
    if bool(args.embed_glb):
        glb_bytes = args.glb_path.read_bytes()
        if glb_bytes[:4] != b"glTF":
            raise ValueError(f"Not a GLB file: {args.glb_path}")
        cfg["glbDataUrl"] = "data:model/gltf-binary;base64," + base64.b64encode(glb_bytes).decode("ascii")
        print(f"[embed] glb bytes={len(glb_bytes):,} data_url_chars={len(cfg['glbDataUrl']):,}")

    args.out_html.parent.mkdir(parents=True, exist_ok=True)
    args.out_html.write_text(build_html(cfg), encoding="utf-8")
    print(f"saved combined html: {args.out_html}")
    print(f"pivot lat/lon/alt: {lat:.6f}, {lon:.6f}, {alt:.2f}")
    print(f"glb rel: {glb_rel}")
    print(f"folium rel: {cfg['foliumUrl']}")
    print(f"embed_glb: {bool(args.embed_glb)}")


if __name__ == "__main__":
    main()
