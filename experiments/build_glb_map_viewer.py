#!/usr/bin/env python3
"""Build a minimal Cesium HTML viewer for a GLB mesh on map.

The GLB is assumed to contain ECEF offsets from a pivot (as exported by
gnss_gpu.viz.plateau_glb.export_plateau_roi_glb). The pivot is computed from
the triangles NPY mesh and used as translation in Cesium.
"""

from __future__ import annotations

import argparse
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


def build_html(glb_url: str, lat: float, lon: float, alt: float, token: str) -> str:
    cfg = {"glbUrl": glb_url, "lat": lat, "lon": lon, "alt": alt, "token": token}
    cfg_json = json.dumps(cfg)
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>OSM GLB Map Viewer</title>
  <script src="https://cesium.com/downloads/cesiumjs/releases/1.120/Build/Cesium/Cesium.js"></script>
  <link href="https://cesium.com/downloads/cesiumjs/releases/1.120/Build/Cesium/Widgets/widgets.css" rel="stylesheet" />
  <style>
    html, body, #cesiumContainer {{ width:100%; height:100%; margin:0; padding:0; overflow:hidden; background:#111; }}
    #info {{ position:absolute; top:10px; left:10px; z-index:999; background:#0008; color:#fff; padding:8px 10px; font-family:sans-serif; font-size:12px; }}
  </style>
</head>
<body>
  <div id="info">GLB overlay on map</div>
  <div id="cesiumContainer"></div>
  <script>
    const cfg = {cfg_json};
    if (cfg.token) {{
      Cesium.Ion.defaultAccessToken = cfg.token;
    }}
    const viewer = new Cesium.Viewer("cesiumContainer", {{
      timeline: false,
      animation: false,
      sceneModePicker: true,
      baseLayerPicker: true,
      geocoder: true,
      homeButton: true
    }});
    const pivot = Cesium.Cartesian3.fromDegrees(cfg.lon, cfg.lat, cfg.alt);
    // GLB vertices are ECEF offsets, so use pure translation only.
    const modelMatrix = Cesium.Matrix4.fromTranslation(pivot);
    Cesium.Model.fromGltfAsync({{
      url: cfg.glbUrl,
      modelMatrix,
      scale: 1.0,
      upAxis: Cesium.Axis.Z,
      forwardAxis: Cesium.Axis.X
    }}).then((model) => {{
      viewer.scene.primitives.add(model);
    }}).catch((err) => {{
      console.error("GLB load failed:", err);
      const info = document.getElementById("info");
      if (info) info.textContent = "GLB load failed: " + String(err);
    }});
    viewer.camera.flyTo({{
      destination: Cesium.Cartesian3.fromDegrees(cfg.lon, cfg.lat, cfg.alt + 250),
      orientation: {{ heading: 0, pitch: -0.7, roll: 0 }}
    }});
  </script>
</body>
</html>
"""


def main() -> None:
    p = argparse.ArgumentParser(description="Generate minimal Cesium HTML for GLB map overlay")
    p.add_argument("--triangles-npy", type=Path, required=True, help="ECEF triangles [N,3,3]")
    p.add_argument("--glb-path", type=Path, required=True, help="GLB path (relative to HTML folder preferred)")
    p.add_argument("--out-html", type=Path, required=True, help="Output HTML path")
    p.add_argument("--cesium-ion-token", type=str, default="", help="Optional Cesium ion token")
    args = p.parse_args()

    tri = np.asarray(np.load(args.triangles_npy), dtype=np.float64)
    if tri.ndim != 3 or tri.shape[1:] != (3, 3):
        raise ValueError(f"--triangles-npy must have shape [N,3,3], got {tri.shape}")
    pivot = tri.reshape(-1, 3).mean(axis=0)
    lat, lon, alt = ecef_to_lla_deg(float(pivot[0]), float(pivot[1]), float(pivot[2]))

    out_dir = args.out_html.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    glb_rel = args.glb_path.name if args.glb_path.parent == out_dir else str(args.glb_path.as_posix())
    html = build_html(glb_rel, lat, lon, alt, args.cesium_ion_token)
    args.out_html.write_text(html, encoding="utf-8")
    print(f"saved html: {args.out_html}")
    print(f"pivot lat/lon/alt: {lat:.6f}, {lon:.6f}, {alt:.2f}")


if __name__ == "__main__":
    main()
