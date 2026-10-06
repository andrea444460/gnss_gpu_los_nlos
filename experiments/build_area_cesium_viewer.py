#!/usr/bin/env python3
"""Build a Cesium HTML viewer: DEM terrain + 3D buildings + colored LOS/NLOS points."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_ROOT / "python"))

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = 2.0 * WGS84_F - WGS84_F * WGS84_F

METRICS = (
    "mean_n_los",
    "mean_n_nlos",
    "mean_n_visible",
    "mean_n_terrain_blocked_visible",
)


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


def _lla_deg_to_ecef(lat_deg: float, lon_deg: float, alt_m: float) -> np.ndarray:
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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--triangles-npy", type=Path, required=True)
    p.add_argument("--glb-path", type=Path, required=True)
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--dem-path", type=Path, required=True, help="Same DEM used for mesh + rx-alt-mode")
    p.add_argument("--out-html", type=Path, required=True)
    p.add_argument("--metric", type=str, default="mean_n_los", choices=METRICS)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--max-points", type=int, default=0)
    p.add_argument("--rx-ant-height-m", type=float, default=1.5, help="Point height above DEM [m]")
    p.add_argument(
        "--dem-mesh-stride",
        type=int,
        default=2,
        help="DEM ground mesh sampling stride (1=full res, 2=half step)",
    )
    p.add_argument(
        "--dem-ground-glb",
        type=Path,
        default=None,
        help="Optional output path for DEM ground GLB (default: out-html dir)",
    )
    p.add_argument(
        "--cesium-ion-token",
        type=str,
        default=os.environ.get("CESIUM_ION_TOKEN", ""),
        help="Optional Ion token for imagery only (terrain comes from --dem-path)",
    )
    p.add_argument("--title", type=str, default="Area LOS/NLOS — Cesium")
    p.add_argument(
        "--geoid-correction",
        default="none",
        help="Only for orthometric DEMs: egm96 or constant N [m]. Satellite DEMs: none (default).",
    )
    p.add_argument(
        "--cesium-globe-offset-m",
        type=float,
        default=0.0,
        help="Raise Cesium ellipsoid terrain [m] for basemap alignment; DEM mesh unchanged.",
    )
    p.add_argument("--embed-glb", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--regenerate-glb", action="store_true")
    p.add_argument("--max-glb-triangles", type=int, default=0)
    return p.parse_args()


def _read_points(
    csv_path: Path,
    metric: str,
    stride: int,
    max_points: int,
    dem_sampler,
    rx_ant_height_m: float,
) -> tuple[dict, dict]:
    lats: list[float] = []
    lons: list[float] = []
    vals: list[float] = []
    highways: list[str] = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if (i % max(1, stride)) != 0:
                continue
            try:
                lats.append(float(row["lat_deg"]))
                lons.append(float(row["lon_deg"]))
                vals.append(float(row[metric]))
                highways.append(str(row.get("highway", "")))
            except (KeyError, TypeError, ValueError):
                continue
            if max_points > 0 and len(lats) >= max_points:
                break

    if not vals:
        raise ValueError(f"No valid points read from {csv_path}")

    dem_payload = dem_sampler.summary()
    ground_alts = dem_sampler.sample_many(np.asarray(lats), np.asarray(lons))
    point_alts = (ground_alts + float(rx_ant_height_m)).tolist()

    arr = np.asarray(vals, dtype=np.float64)
    vmin = float(np.nanpercentile(arr, 5))
    vmax = float(np.nanpercentile(arr, 95))
    if not math.isfinite(vmin) or not math.isfinite(vmax) or vmax <= vmin:
        vmin = float(np.nanmin(arr))
        vmax = float(np.nanmax(arr) + 1.0)

    points = {
        "metric": metric,
        "count": len(vals),
        "vmin": vmin,
        "vmax": vmax,
        "lats": lats,
        "lons": lons,
        "alts": point_alts,
        "vals": vals,
        "highways": highways,
    }
    return points, dem_payload


def build_html(cfg: dict) -> str:
    cfg_json = json.dumps(cfg, ensure_ascii=True)
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>{cfg["title"]}</title>
  <script src="https://cesium.com/downloads/cesiumjs/releases/1.120/Build/Cesium/Cesium.js"></script>
  <link href="https://cesium.com/downloads/cesiumjs/releases/1.120/Build/Cesium/Widgets/widgets.css" rel="stylesheet" />
  <style>
    html, body, #cesiumContainer {{ width:100%; height:100%; margin:0; padding:0; overflow:hidden; background:#111; }}
    #toolbar {{ position:absolute; top:10px; left:10px; z-index:1000; display:flex; gap:8px; flex-wrap:wrap; }}
    .btn {{ background:rgba(0,0,0,0.7); color:#fff; border:1px solid #666; border-radius:4px; padding:6px 10px; font-size:12px; cursor:pointer; }}
    #legend {{ position:absolute; bottom:12px; left:12px; z-index:1000; background:rgba(0,0,0,0.75); color:#fff; padding:10px 12px; font:12px/1.4 sans-serif; border-radius:4px; }}
    .bar {{ width:180px; height:12px; margin-top:6px; background:linear-gradient(to right, #d73027, #fee08b, #1a9850); border:1px solid #888; }}
    #info {{ position:absolute; top:10px; right:10px; z-index:1000; background:rgba(0,0,0,0.7); color:#fff; padding:8px 10px; font:12px sans-serif; border-radius:4px; max-width:320px; }}
  </style>
</head>
<body>
  <div id="toolbar">
    <button class="btn" id="btnToggleMesh">Buildings</button>
    <button class="btn" id="btnTogglePoints">Points</button>
    <button class="btn" id="btnToggleTerrain">DEM ground</button>
    <button class="btn" id="btnHome">Home</button>
  </div>
  <div id="info"></div>
  <div id="legend"></div>
  <div id="cesiumContainer"></div>
  <script>
    const cfg = {cfg_json};
    if (cfg.token) {{
      Cesium.Ion.defaultAccessToken = cfg.token;
    }}

    function colorForValue(v, vmin, vmax) {{
      const t = Math.max(0, Math.min(1, (v - vmin) / Math.max(1e-9, vmax - vmin)));
      let r, g, b;
      if (t < 0.5) {{
        const u = t * 2.0;
        r = Math.round(215 + (254 - 215) * u);
        g = Math.round(48 + (224 - 48) * u);
        b = Math.round(39 + (139 - 39) * u);
      }} else {{
        const u = (t - 0.5) * 2.0;
        r = Math.round(254 + (26 - 254) * u);
        g = Math.round(224 + (152 - 224) * u);
        b = Math.round(139 + (80 - 139) * u);
      }}
      return new Cesium.Color(r / 255, g / 255, b / 255, 0.92);
    }}

    const globeOffsetM = Number(cfg.globeOffsetM || 0);

    class OffsetEllipsoidTerrainProvider extends Cesium.EllipsoidTerrainProvider {{
      constructor(offsetM) {{
        super();
        this._offsetM = Number(offsetM) || 0;
      }}
      requestTileGeometry(x, y, level, request) {{
        const p = super.requestTileGeometry(x, y, level, request);
        if (!p || this._offsetM === 0) return p;
        return Promise.resolve(p).then((mesh) => {{
          if (!mesh || !mesh.attributes || !mesh.attributes.position) return mesh;
          const pos = mesh.attributes.position.values;
          const ellipsoid = Cesium.Ellipsoid.WGS84;
          const scratch = new Cesium.Cartographic();
          const scratchCart = new Cesium.Cartesian3();
          for (let i = 0; i < pos.length; i += 3) {{
            scratchCart.x = pos[i];
            scratchCart.y = pos[i + 1];
            scratchCart.z = pos[i + 2];
            const carto = ellipsoid.cartesianToCartographic(scratchCart, scratch);
            carto.height += this._offsetM;
            const raised = ellipsoid.cartographicToCartesian(carto, scratchCart);
            pos[i] = raised.x;
            pos[i + 1] = raised.y;
            pos[i + 2] = raised.z;
          }}
          return mesh;
        }});
      }}
    }}

    const viewer = new Cesium.Viewer("cesiumContainer", {{
      terrainProvider: new OffsetEllipsoidTerrainProvider(globeOffsetM),
      timeline: false,
      animation: false,
      sceneModePicker: true,
      baseLayerPicker: true,
      geocoder: true,
      homeButton: false,
    }});
    viewer.scene.globe.depthTestAgainstTerrain = true;

    const pivot = Cesium.Cartesian3.fromDegrees(cfg.lon, cfg.lat, cfg.alt);
    const modelMatrix = Cesium.Matrix4.fromTranslation(pivot);
    let meshModel = null;
    let groundModel = null;

    function loadGlbModel(url, onOk, onErr) {{
      return Cesium.Model.fromGltfAsync({{
        url,
        modelMatrix,
        scale: 1.0,
        upAxis: Cesium.Axis.Z,
        forwardAxis: Cesium.Axis.X,
      }}).then((model) => {{
        viewer.scene.primitives.add(model);
        onOk(model);
      }}).catch(onErr);
    }}

    const glbSource = cfg.glbDataUrl || cfg.glbUrl;
    loadGlbModel(glbSource, (m) => {{ meshModel = m; }}, (err) => console.error("Buildings GLB failed:", err));

    const groundSource = cfg.groundGlbDataUrl || cfg.groundGlbUrl;
    if (groundSource) {{
      loadGlbModel(
        groundSource,
        (m) => {{ groundModel = m; }},
        (err) => console.error("Ground GLB failed:", err),
      );
    }}

    const pts = cfg.points;
    const pointCollection = viewer.scene.primitives.add(new Cesium.PointPrimitiveCollection());
    for (let i = 0; i < pts.lats.length; i++) {{
      pointCollection.add({{
        position: Cesium.Cartesian3.fromDegrees(pts.lons[i], pts.lats[i], pts.alts[i]),
        color: colorForValue(pts.vals[i], pts.vmin, pts.vmax),
        pixelSize: 7,
      }});
    }}

    document.getElementById("legend").innerHTML =
      `<b>${{pts.metric}}</b><br/>` +
      `${{pts.count.toLocaleString()}} points (stride=${{cfg.stride}})<br/>` +
      `min ${{pts.vmin.toFixed(2)}} — max ${{pts.vmax.toFixed(2)}}` +
      `<div class="bar"></div>`;

    const dem = cfg.dem;
    document.getElementById("info").innerHTML =
      `DEM ground mesh: ${{cfg.groundTriangles.toLocaleString()}} tris<br/>` +
      `DEM native ${{dem.crs}} (${{dem.heightDatum}})<br/>` +
      `elev ${{dem.min.toFixed(0)}}–${{dem.max.toFixed(0)}} m, stride=${{cfg.demMeshStride}}<br/>` +
      `Cesium globe offset: ${{globeOffsetM.toFixed(1)}} m<br/>` +
      `Metric: ${{pts.metric}}`;

    const camAlt = Math.max(cfg.alt + 450, dem.max + 250);
    viewer.camera.flyTo({{
      destination: Cesium.Cartesian3.fromDegrees(cfg.lon, cfg.lat, camAlt),
      orientation: {{ heading: 0.0, pitch: -0.6, roll: 0.0 }},
    }});

    document.getElementById("btnToggleMesh").onclick = () => {{
      if (meshModel) meshModel.show = !meshModel.show;
    }};
    document.getElementById("btnTogglePoints").onclick = () => {{
      pointCollection.show = !pointCollection.show;
    }};
    document.getElementById("btnToggleTerrain").onclick = () => {{
      if (groundModel) groundModel.show = !groundModel.show;
    }};
    document.getElementById("btnHome").onclick = () => {{
      viewer.camera.flyTo({{
        destination: Cesium.Cartesian3.fromDegrees(cfg.lon, cfg.lat, camAlt),
        orientation: {{ heading: 0.0, pitch: -0.6, roll: 0.0 }},
      }});
    }};
  </script>
</body>
</html>
"""


def main() -> None:
    args = _parse_args()
    tri = np.asarray(np.load(args.triangles_npy), dtype=np.float64)
    if tri.ndim != 3 or tri.shape[1:] != (3, 3):
        raise ValueError(f"Expected [N,3,3], got {tri.shape}")

    pivot = tri.reshape(-1, 3).mean(axis=0)
    lat, lon, alt = ecef_to_lla_deg(float(pivot[0]), float(pivot[1]), float(pivot[2]))

    from gnss_gpu.viz.plateau_glb import export_plateau_roi_glb

    cap = int(args.max_glb_triangles) if int(args.max_glb_triangles) > 0 else int(tri.shape[0])
    kept, total = export_plateau_roi_glb(
        tri,
        pivot,
        args.glb_path,
        full_mesh=True,
        max_triangles=cap,
        coordinate_frame="ecef_offset",
    )
    print(f"[buildings-glb] kept={kept}/{total} frame=ecef_offset path={args.glb_path}")

    from gnss_gpu.io.dem_native import NativeDemSampler, parse_geoid_correction_arg

    geoid_corr = parse_geoid_correction_arg(str(args.geoid_correction))
    dem_sampler = NativeDemSampler(args.dem_path, geoid_correction=geoid_corr)
    points, dem_payload = _read_points(
        args.csv,
        str(args.metric),
        int(args.stride),
        int(args.max_points),
        dem_sampler,
        float(args.rx_ant_height_m),
    )
    print(
        f"[csv] metric={points['metric']} rendered={points['count']} "
        f"vmin={points['vmin']:.2f} vmax={points['vmax']:.2f}",
        flush=True,
    )
    print(
        f"[dem] native crs={dem_payload['crs']} geoid={dem_payload['geoid']} "
        f"datum={dem_payload['heightDatum']} grid={dem_payload['rows']}x{dem_payload['cols']} "
        f"elev={dem_payload['min']:.1f}..{dem_payload['max']:.1f} m path={args.dem_path}",
        flush=True,
    )

    ground_tri = dem_sampler.to_triangles_ecef(cell_stride=int(args.dem_mesh_stride))
    ground_glb_path = args.dem_ground_glb or args.out_html.with_name("limassol_dem_ground.glb")
    export_plateau_roi_glb(
        ground_tri,
        pivot,
        ground_glb_path,
        full_mesh=True,
        max_triangles=int(ground_tri.shape[0]),
        base_color_factor=(0.45, 0.52, 0.38, 1.0),
        coordinate_frame="ecef_offset",
    )
    print(f"[dem-ground] triangles={ground_tri.shape[0]:,} glb={ground_glb_path}", flush=True)

    cfg: dict = {
        "title": str(args.title),
        "glbUrl": args.glb_path.name,
        "glbDataUrl": "",
        "groundGlbUrl": ground_glb_path.name,
        "groundGlbDataUrl": "",
        "groundTriangles": int(ground_tri.shape[0]),
        "demMeshStride": int(args.dem_mesh_stride),
        "globeOffsetM": float(args.cesium_globe_offset_m),
        "lat": lat,
        "lon": lon,
        "alt": alt,
        "token": str(args.cesium_ion_token).strip(),
        "rxAntHeightM": float(args.rx_ant_height_m),
        "stride": int(args.stride),
        "points": points,
        "dem": dem_payload,
    }

    if args.embed_glb:
        glb_bytes = args.glb_path.read_bytes()
        if glb_bytes[:4] != b"glTF":
            raise ValueError(f"Not a GLB: {args.glb_path}")
        cfg["glbDataUrl"] = "data:model/gltf-binary;base64," + base64.b64encode(glb_bytes).decode("ascii")
        print(f"[embed] buildings glb bytes={len(glb_bytes):,}", flush=True)

        ground_bytes = ground_glb_path.read_bytes()
        cfg["groundGlbDataUrl"] = "data:model/gltf-binary;base64," + base64.b64encode(ground_bytes).decode("ascii")
        print(f"[embed] ground glb bytes={len(ground_bytes):,}", flush=True)

    args.out_html.parent.mkdir(parents=True, exist_ok=True)
    args.out_html.write_text(build_html(cfg), encoding="utf-8")
    dem_sampler.close()
    print(f"saved: {args.out_html}", flush=True)


if __name__ == "__main__":
    main()
