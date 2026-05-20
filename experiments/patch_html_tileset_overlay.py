#!/usr/bin/env python3
"""Patch an existing Cesium LOS/NLOS HTML to overlay a local textured 3D Tiles tileset.

Also disables the yellow trajectory polyline and hides the gray LOS debug GLB by default
when the tileset is configured.

Example::

    python experiments/patch_html_tileset_overlay.py \\
        --input-html \"C:/Users/Me/Desktop/Stadio ray Tracing/luigi_ferraris_viz.html\" \\
        --tileset-url luigiFerrarisGeoLoc/tileset.json
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


STADIUM_TOGGLE_ROW = """    <div class="row" id="stadiumTilesetToggleRow" style="display:flex;">
      <label title="Original local 3D Tiles with textures (visual only; LOS uses triangles.npy)">
        <input type="checkbox" id="chkHideStadiumTileset"/> Hide textured tileset
      </label>
      <label title="Semi-transparent gray mesh exported from triangles.npy (debug overlay)">
        <input type="checkbox" id="chkShowLosMesh"/> Show LOS debug mesh
      </label>
    </div>
"""

TILESET_JS = """
const STADIUM_TILESET_URL = {tileset_url!r};
const hasTexturedTileset = !!STADIUM_TILESET_URL;
let localStadiumTileset = null;
let losMeshModel = null;

if (hasTexturedTileset) {{
  const tsRow = document.getElementById('stadiumTilesetToggleRow');
  if (tsRow) tsRow.style.display = 'flex';
  const tilesetUrl = new URL(STADIUM_TILESET_URL, window.location.href).href;
  Cesium.Cesium3DTileset.fromUrl(tilesetUrl)
    .then(tileset => {{
      localStadiumTileset = tileset;
      viewer.scene.primitives.add(tileset);
      const chkHide = document.getElementById('chkHideStadiumTileset');
      if (chkHide && chkHide.checked) tileset.show = false;
    }})
    .catch(e => console.warn('Stadium tileset load failed (serve tileset over http://):', e));
  const vs = document.getElementById('vizSources');
  if (vs) {{
    vs.innerHTML += '<br/><strong>Textured tileset:</strong> local 3D Tiles (visual). LOS/NLOS rays still use triangles.npy in Python.';
  }}
}}
"""

GLB_THEN_OLD = """    .then(model => {
      viewer.scene.primitives.add(model);
    })"""

GLB_THEN_NEW = """    .then(model => {
      losMeshModel = model;
      viewer.scene.primitives.add(model);
      const chkShow = document.getElementById('chkShowLosMesh');
      if (hasTexturedTileset) {
        model.show = !!(chkShow && chkShow.checked);
      }
    })"""

PLATEAU_VIZ_OLD = """  if (vs) {
    vs.innerHTML += '<br/><strong>PLATEAU GLB:</strong> Same mesh family as LOS/NLOS — hide OSM buildings to compare.';
  }"""

PLATEAU_VIZ_NEW = """  if (vs && !hasTexturedTileset) {
    vs.innerHTML += '<br/><strong>PLATEAU GLB:</strong> Same mesh family as LOS/NLOS — hide OSM buildings to compare.';
  }"""

EVENT_LISTENERS = """  const chkHideStadiumTileset = document.getElementById('chkHideStadiumTileset');
  if (chkHideStadiumTileset) {
    chkHideStadiumTileset.addEventListener('change', (e) => {
      if (localStadiumTileset) localStadiumTileset.show = !e.target.checked;
    });
  }
  const chkShowLosMesh = document.getElementById('chkShowLosMesh');
  if (chkShowLosMesh) {
    chkShowLosMesh.addEventListener('change', (e) => {
      if (losMeshModel) losMeshModel.show = !!e.target.checked;
    });
  }
"""


def _ensure_tileset_junction(html_path: Path, tileset_dir: Path | None) -> None:
    if tileset_dir is None:
        return
    src = tileset_dir.resolve()
    if not (src / "tileset.json").is_file():
        raise FileNotFoundError(f"tileset.json not found in {src}")
    link = html_path.parent / src.name
    if link.exists():
        return
    if os.name == "nt":
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(src)],
            check=True,
            capture_output=True,
            text=True,
        )
    else:
        link.symlink_to(src, target_is_directory=True)
    print(f"Linked tileset folder: {link} -> {src}", flush=True)


def patch_html(html: str, *, tileset_url: str, disable_yellow_trajectory: bool) -> str:
    if "STADIUM_TILESET_URL" in html:
        print("Already patched (STADIUM_TILESET_URL present).", flush=True)
        return html

    if 'id="stadiumTilesetToggleRow"' not in html:
        marker = '    <div class="row" id="osmToggleRow"'
        if marker not in html:
            raise ValueError("Could not find osmToggleRow block to insert stadium toggles.")
        insert_at = html.find("    </div>", html.find(marker))
        if insert_at < 0:
            raise ValueError("Could not find end of osmToggleRow block.")
        insert_at += len("    </div>")
        html = html[:insert_at] + "\n" + STADIUM_TOGGLE_ROW + html[insert_at:]

    marker = "const plateauSpec = datasets[0] && datasets[0].plateauModel;"
    if marker not in html:
        raise ValueError("Could not find plateauSpec block.")
    html = html.replace(marker, TILESET_JS.format(tileset_url=tileset_url) + marker, 1)

    if GLB_THEN_OLD in html:
        html = html.replace(GLB_THEN_OLD, GLB_THEN_NEW, 1)
    else:
        print("[warn] GLB .then block not found; gray mesh toggle may not work.", flush=True)

    if PLATEAU_VIZ_OLD in html:
        html = html.replace(PLATEAU_VIZ_OLD, PLATEAU_VIZ_NEW, 1)

    ev_marker = "  const chkMp = document.getElementById('chkShowMultipath');"
    if ev_marker in html and EVENT_LISTENERS.strip() not in html:
        html = html.replace(ev_marker, EVENT_LISTENERS + ev_marker, 1)

    if disable_yellow_trajectory:
        traj_marker = "function drawTrajectory(ds) {"
        if traj_marker in html and "if (!SHOW_TRAJECTORY_POLYLINE)" not in html:
            html = html.replace(
                traj_marker,
                "function drawTrajectory(ds) {\n  return;",
                1,
            )

    return html


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-html", type=Path, required=True)
    ap.add_argument("--output-html", type=Path, default=None)
    ap.add_argument(
        "--tileset-url",
        type=str,
        default="luigiFerrarisGeoLoc/tileset.json",
        help="URL/path to tileset.json relative to the HTML file (default: luigiFerrarisGeoLoc/tileset.json)",
    )
    ap.add_argument(
        "--tileset-dir",
        type=Path,
        default=None,
        help="If set, create a directory junction/symlink next to the HTML when missing.",
    )
    ap.add_argument(
        "--keep-yellow-trajectory",
        action="store_true",
        help="Do not disable the yellow trajectory polyline.",
    )
    args = ap.parse_args()

    inp = args.input_html.resolve()
    out = (args.output_html or inp).resolve()
    if not inp.is_file():
        sys.exit(f"not found: {inp}")

    if args.tileset_dir is not None:
        _ensure_tileset_junction(inp, args.tileset_dir)

    html = inp.read_text(encoding="utf-8")
    patched = patch_html(
        html,
        tileset_url=args.tileset_url.replace("\\", "/"),
        disable_yellow_trajectory=not bool(args.keep_yellow_trajectory),
    )
    out.write_text(patched, encoding="utf-8")
    print(f"Patched HTML written to {out}", flush=True)


if __name__ == "__main__":
    main()
