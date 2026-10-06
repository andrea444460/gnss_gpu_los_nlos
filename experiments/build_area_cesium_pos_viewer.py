#!/usr/bin/env python3
"""Build a Cesium viewer from an existing area map HTML + RTKLIB .pos rover track.

Duplicates the buildings/DEM scene from ``--base-html`` (embedded GLBs + token),
drops the LOS/NLOS point cloud, and draws the kinematic solution as colored
polylines (RTKLIB Q: 1=fix, 2=float, 3=sbas, 4=dgps, 5=single, 6=ppp).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_ROOT / "python"))

# RTKLIB solution quality (Q column in lat/lon/height .pos output)
Q_META: dict[int, dict[str, str]] = {
    1: {"key": "fix", "label": "RTK fixed", "color": "#2ca02c"},
    2: {"key": "float", "label": "RTK float", "color": "#ffbf00"},
    3: {"key": "sbas", "label": "SBAS", "color": "#1f77b4"},
    4: {"key": "dgps", "label": "DGPS", "color": "#17becf"},
    5: {"key": "single", "label": "Single", "color": "#d62728"},
    6: {"key": "ppp", "label": "PPP", "color": "#9467bd"},
}
Q_DEFAULT = {"key": "other", "label": "Other", "color": "#aaaaaa"}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--base-html",
        type=Path,
        required=True,
        help="Existing Cesium HTML to copy scene assets from (embedded GLBs, token, pivot)",
    )
    p.add_argument("--pos-path", type=Path, required=True, help="RTKLIB .pos file (lat/lon/height format)")
    p.add_argument("--out-html", type=Path, required=True)
    p.add_argument("--title", type=str, default="")
    p.add_argument("--pos-stride", type=int, default=1, help="Keep every Nth epoch for the track polyline")
    p.add_argument("--line-width", type=float, default=4.0, help="Polyline width in pixels")
    p.add_argument(
        "--dem-path",
        type=Path,
        default=None,
        help="DEM GeoTIFF for vertical alignment (default: from base-html or sibling limassol_area_dem.tif)",
    )
    p.add_argument(
        "--track-height-mode",
        choices=("cesium-terrain", "ellipsoidal", "auto", "dem-ground"),
        default="cesium-terrain",
        help=(
            "cesium-terrain: snap track to Cesium World Terrain + antenna (default, matches basemap); "
            "ellipsoidal: raw WGS84 .pos heights; "
            "auto/dem-ground: align to local DEM mesh instead"
        ),
    )
    p.add_argument(
        "--no-world-terrain",
        action="store_true",
        help="Keep flat ellipsoid even when a Cesium Ion token is available",
    )
    p.add_argument(
        "--rx-antenna-height-m",
        type=float,
        default=1.5,
        help="Antenna height above ground used for auto/dem-ground alignment [m]",
    )
    return p.parse_args()


def resolve_dem_path(base_html: Path, base_cfg: dict, explicit: Path | None) -> Path:
    if explicit is not None:
        if not explicit.is_file():
            raise FileNotFoundError(f"DEM not found: {explicit}")
        return explicit
    dem_info = base_cfg.get("dem") or {}
    dem_from_cfg = dem_info.get("path")
    if dem_from_cfg:
        candidate = Path(str(dem_from_cfg))
        if candidate.is_file():
            return candidate
    sibling = base_html.parent / "limassol_area_dem.tif"
    if sibling.is_file():
        return sibling
    raise FileNotFoundError(
        "DEM not found; pass --dem-path or place limassol_area_dem.tif next to --base-html"
    )


def extract_cfg_from_html(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"const cfg = (\{.*?\});\s*\n", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Could not find cfg JSON in {path}")
    return json.loads(match.group(1))


def scene_cfg_from_base(base_cfg: dict, *, title: str) -> dict:
    out = {
        "title": title or base_cfg.get("title", "RTK track — Cesium"),
        "glbUrl": base_cfg.get("glbUrl", ""),
        "glbDataUrl": base_cfg.get("glbDataUrl", ""),
        "groundGlbUrl": base_cfg.get("groundGlbUrl", ""),
        "groundGlbDataUrl": base_cfg.get("groundGlbDataUrl", ""),
        "groundTriangles": int(base_cfg.get("groundTriangles", 0)),
        "demMeshStride": int(base_cfg.get("demMeshStride", 2)),
        "globeOffsetM": float(base_cfg.get("globeOffsetM", 0.0)),
        "lat": float(base_cfg["lat"]),
        "lon": float(base_cfg["lon"]),
        "alt": float(base_cfg["alt"]),
        "token": str(base_cfg.get("token", "")),
        "dem": base_cfg.get("dem", {}),
        "lineWidth": 0.0,  # filled in main
    }
    if not out["glbDataUrl"] and not out["glbUrl"]:
        raise ValueError("base-html has no buildings GLB reference")
    return out


def parse_rtklib_pos_llh(path: Path, *, stride: int) -> list[dict]:
    """Parse RTKPOST lat/lon/height .pos (columns: date time lat lon h Q ns ...)."""
    rows: list[dict] = []
    step = max(1, int(stride))
    kept = 0
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("%"):
                continue
            parts = s.split()
            if len(parts) < 6:
                continue
            try:
                lat = float(parts[2])
                lon = float(parts[3])
                h = float(parts[4])
                q = int(float(parts[5]))
            except ValueError:
                continue
            if kept % step != 0:
                kept += 1
                continue
            kept += 1
            rows.append(
                {
                    "lat": lat,
                    "lon": lon,
                    "h": h,
                    "q": q,
                    "time": f"{parts[0]} {parts[1]}",
                }
            )
    if not rows:
        raise ValueError(f"No valid epochs in {path}")
    return rows


def segments_by_q(rows: list[dict]) -> list[dict]:
    segments: list[dict] = []
    cur_q = int(rows[0]["q"])
    lats = [float(rows[0]["lat"])]
    lons = [float(rows[0]["lon"])]
    alts = [float(rows[0]["h"])]

    def flush() -> None:
        if len(lats) < 2:
            return
        meta = Q_META.get(cur_q, Q_DEFAULT)
        segments.append(
            {
                "q": cur_q,
                "key": meta["key"],
                "label": meta["label"],
                "color": meta["color"],
                "count": len(lats),
                "lats": lats,
                "lons": lons,
                "alts": alts,
            }
        )

    for row in rows[1:]:
        q = int(row["q"])
        if q != cur_q:
            bridge = (lats[-1], lons[-1], alts[-1])
            flush()
            cur_q = q
            lats = [bridge[0], float(row["lat"])]
            lons = [bridge[1], float(row["lon"])]
            alts = [bridge[2], float(row["h"])]
        else:
            lats.append(float(row["lat"]))
            lons.append(float(row["lon"]))
            alts.append(float(row["h"]))
    flush()
    return segments


def q_summary(rows: list[dict]) -> list[dict]:
    counts: dict[int, int] = {}
    for row in rows:
        q = int(row["q"])
        counts[q] = counts.get(q, 0) + 1
    out = []
    for q in sorted(counts):
        meta = Q_META.get(q, Q_DEFAULT)
        out.append(
            {
                "q": q,
                "key": meta["key"],
                "label": meta["label"],
                "color": meta["color"],
                "count": counts[q],
            }
        )
    return out


def apply_track_heights(
    rows: list[dict],
    dem_sampler,
    *,
    mode: str,
    rx_antenna_height_m: float,
) -> dict:
    """Align RTK ellipsoidal heights with the DEM-grounded scene."""
    lats = np.asarray([float(r["lat"]) for r in rows], dtype=np.float64)
    lons = np.asarray([float(r["lon"]) for r in rows], dtype=np.float64)
    pos_h = np.asarray([float(r["h"]) for r in rows], dtype=np.float64)
    dem_h = dem_sampler.sample_many(lats, lons)
    median_delta = float(np.median(pos_h - dem_h))

    if mode == "ellipsoidal":
        vis_h = pos_h.copy()
        offset_m = 0.0
    elif mode == "dem-ground":
        vis_h = dem_h + float(rx_antenna_height_m)
        offset_m = median_delta - float(rx_antenna_height_m)
    else:
        offset_m = median_delta - float(rx_antenna_height_m)
        vis_h = pos_h - offset_m

    for i, row in enumerate(rows):
        row["h"] = float(vis_h[i])

    return {
        "heightMode": mode,
        "verticalOffsetM": float(offset_m),
        "rxAntennaHeightM": float(rx_antenna_height_m),
        "medianPosDemDeltaM": median_delta,
    }


def build_pos_track_payload(
    path: Path,
    *,
    stride: int,
    dem_sampler,
    height_mode: str,
    rx_antenna_height_m: float,
) -> dict:
    rows = parse_rtklib_pos_llh(path, stride=stride)
    if height_mode in ("auto", "dem-ground"):
        if dem_sampler is None:
            raise ValueError(f"--track-height-mode {height_mode} requires --dem-path")
        height_meta = apply_track_heights(
            rows,
            dem_sampler,
            mode=height_mode,
            rx_antenna_height_m=rx_antenna_height_m,
        )
    else:
        height_meta = {
            "heightMode": height_mode,
            "verticalOffsetM": 0.0,
            "rxAntennaHeightM": float(rx_antenna_height_m),
            "medianPosDemDeltaM": 0.0,
        }
    segments = segments_by_q(rows)
    if not segments:
        raise ValueError(f"Track has fewer than 2 epochs per quality segment in {path}")
    lats = [float(r["lat"]) for r in rows]
    lons = [float(r["lon"]) for r in rows]
    alts = [float(r["h"]) for r in rows]
    return {
        "source": str(path),
        "stride": int(stride),
        "epochs": len(rows),
        "segments": len(segments),
        "latCenter": float(sum(lats) / len(lats)),
        "lonCenter": float(sum(lons) / len(lons)),
        "altCenter": float(sum(alts) / len(alts)),
        "altMin": float(min(alts)),
        "altMax": float(max(alts)),
        "qSummary": q_summary(rows),
        "qColors": {str(q): Q_META[q] for q in Q_META},
        "qDefault": Q_DEFAULT,
        "track": segments,
        **height_meta,
    }


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
    #legend {{ position:absolute; bottom:12px; left:12px; z-index:1000; background:rgba(0,0,0,0.75); color:#fff; padding:10px 12px; font:12px/1.4 sans-serif; border-radius:4px; max-width:280px; }}
    .swatch {{ display:inline-block; width:14px; height:14px; border-radius:2px; margin-right:6px; vertical-align:middle; border:1px solid #666; }}
    #info {{ position:absolute; top:10px; right:10px; z-index:1000; background:rgba(0,0,0,0.7); color:#fff; padding:8px 10px; font:12px sans-serif; border-radius:4px; max-width:340px; }}
  </style>
</head>
<body>
  <div id="toolbar">
    <button class="btn" id="btnToggleMesh">Buildings</button>
    <button class="btn" id="btnToggleTrack">RTK track</button>
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

    function hexToCesiumColor(hex, alpha) {{
      const h = String(hex || "#aaaaaa").replace("#", "");
      const r = parseInt(h.substring(0, 2), 16) / 255;
      const g = parseInt(h.substring(2, 4), 16) / 255;
      const b = parseInt(h.substring(4, 6), 16) / 255;
      return new Cesium.Color(r, g, b, alpha == null ? 1.0 : alpha);
    }}

    const globeOffsetM = Number(cfg.globeOffsetM || 0);
    const useWorldTerrain = Boolean(cfg.useWorldTerrain);

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

    async function createTerrainProvider() {{
      if (useWorldTerrain && cfg.token) {{
        try {{
          return await Cesium.createWorldTerrainAsync();
        }} catch (err) {{
          console.warn("Cesium World Terrain unavailable, using flat ellipsoid:", err);
        }}
      }}
      return new OffsetEllipsoidTerrainProvider(globeOffsetM);
    }}

    async function buildTrackPolylines(viewer, track) {{
      const polylineWidth = Number(cfg.lineWidth || 4);
      const polylines = viewer.scene.primitives.add(new Cesium.PolylineCollection());
      const ant = Number(track.rxAntennaHeightM || 1.5);
      const segments = track.track;

      if (track.heightMode === "cesium-terrain") {{
        const flatCartos = [];
        const ranges = [];
        for (const seg of segments) {{
          const start = flatCartos.length;
          for (let i = 0; i < seg.lats.length; i++) {{
            flatCartos.push(Cesium.Cartographic.fromDegrees(seg.lons[i], seg.lats[i]));
          }}
          ranges.push([start, flatCartos.length]);
        }}
        await Cesium.sampleTerrainMostDetailed(viewer.terrainProvider, flatCartos);
        for (let si = 0; si < segments.length; si++) {{
          const seg = segments[si];
          const [a, b] = ranges[si];
          const positions = [];
          for (let j = a; j < b; j++) {{
            const c = flatCartos[j];
            positions.push(
              Cesium.Cartesian3.fromRadians(c.longitude, c.latitude, c.height + ant),
            );
          }}
          polylines.add({{
            positions,
            width: polylineWidth,
            material: Cesium.Material.fromType("Color", {{
              color: hexToCesiumColor(seg.color, 0.95),
            }}),
          }});
        }}
      }} else {{
        for (const seg of segments) {{
          const positions = [];
          for (let i = 0; i < seg.lats.length; i++) {{
            positions.push(Cesium.Cartesian3.fromDegrees(seg.lons[i], seg.lats[i], seg.alts[i]));
          }}
          polylines.add({{
            positions,
            width: polylineWidth,
            material: Cesium.Material.fromType("Color", {{
              color: hexToCesiumColor(seg.color, 0.95),
            }}),
          }});
        }}
      }}
      return polylines;
    }}

    async function initViewer() {{
      const terrainProvider = await createTerrainProvider();
      const viewer = new Cesium.Viewer("cesiumContainer", {{
        terrainProvider,
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

      const track = cfg.posTrack;
      const polylines = await buildTrackPolylines(viewer, track);

      const dem = cfg.dem || {{}};
      const posName = track.source.split(/[\\\\/]/).pop();
      const terrainLabel = useWorldTerrain && cfg.token ? "Cesium World Terrain" : "flat ellipsoid";
      document.getElementById("info").innerHTML =
        `<b>${{posName}}</b><br/>` +
        `${{track.epochs.toLocaleString()}} epochs (stride=${{track.stride}})<br/>` +
        `${{track.segments}} polyline segments<br/>` +
        `height: ${{track.heightMode}} + ${{track.rxAntennaHeightM.toFixed(1)}} m ant<br/>` +
        `globe: ${{terrainLabel}}<br/>` +
        `DEM mesh: ${{cfg.groundTriangles.toLocaleString()}} tris`;

      let legendHtml = "<b>RTK solution quality</b><br/>";
      for (const row of track.qSummary) {{
        const pct = (100.0 * row.count / track.epochs).toFixed(1);
        legendHtml +=
          `<div><span class="swatch" style="background:${{row.color}}"></span>` +
          `${{row.label}} (Q=${{row.q}}): ${{row.count.toLocaleString()}} (${{pct}}%)</div>`;
      }}
      document.getElementById("legend").innerHTML = legendHtml;

      const camLat = track.latCenter;
      const camLon = track.lonCenter;
      const camAlt = Math.max(track.altCenter + 450, (dem.max || track.altMax) + 250);
      viewer.camera.flyTo({{
        destination: Cesium.Cartesian3.fromDegrees(camLon, camLat, camAlt),
        orientation: {{ heading: 0.0, pitch: -0.6, roll: 0.0 }},
      }});

      document.getElementById("btnToggleMesh").onclick = () => {{
        if (meshModel) meshModel.show = !meshModel.show;
      }};
      document.getElementById("btnToggleTrack").onclick = () => {{
        polylines.show = !polylines.show;
      }};
      document.getElementById("btnToggleTerrain").onclick = () => {{
        if (groundModel) groundModel.show = !groundModel.show;
      }};
      document.getElementById("btnHome").onclick = () => {{
        viewer.camera.flyTo({{
          destination: Cesium.Cartesian3.fromDegrees(camLon, camLat, camAlt),
          orientation: {{ heading: 0.0, pitch: -0.6, roll: 0.0 }},
        }});
      }};
    }}

    initViewer().catch((err) => console.error("Viewer init failed:", err));
  </script>
</body>
</html>
"""


def main() -> None:
    args = _parse_args()
    base_cfg = extract_cfg_from_html(args.base_html)
    title = str(args.title).strip() or "Limassol RTK track — Cesium"
    cfg = scene_cfg_from_base(base_cfg, title=title)
    cfg["lineWidth"] = float(args.line_width)

    dem_sampler = None
    if str(args.track_height_mode) in ("auto", "dem-ground"):
        from gnss_gpu.io.dem_native import NativeDemSampler

        dem_path = resolve_dem_path(args.base_html, base_cfg, args.dem_path)
        dem_sampler = NativeDemSampler(dem_path, geoid_correction=None)

    cfg["useWorldTerrain"] = bool(not args.no_world_terrain)
    cfg["posTrack"] = build_pos_track_payload(
        args.pos_path,
        stride=int(args.pos_stride),
        dem_sampler=dem_sampler,
        height_mode=str(args.track_height_mode),
        rx_antenna_height_m=float(args.rx_antenna_height_m),
    )
    if dem_sampler is not None:
        dem_sampler.close()

    args.out_html.parent.mkdir(parents=True, exist_ok=True)
    args.out_html.write_text(build_html(cfg), encoding="utf-8")

    track = cfg["posTrack"]
    print(f"[pos] {args.pos_path.name}: epochs={track['epochs']} segments={track['segments']}", flush=True)
    print(
        f"[height] mode={track['heightMode']} ant={track['rxAntennaHeightM']:.2f} m "
        f"worldTerrain={cfg['useWorldTerrain']}",
        flush=True,
    )
    if track["heightMode"] in ("auto", "dem-ground"):
        print(
            f"[height] median(pos-dem)={track['medianPosDemDeltaM']:.2f} m "
            f"offset={track['verticalOffsetM']:.2f} m",
            flush=True,
        )
    for row in track["qSummary"]:
        print(f"  Q={row['q']} {row['label']}: {row['count']}", flush=True)
    print(f"saved: {args.out_html}", flush=True)


if __name__ == "__main__":
    main()
