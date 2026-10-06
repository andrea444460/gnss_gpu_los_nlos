"""Prepare CoPre PPK tight/loose coupling project and print run steps.

Updates Rover.prj with rover GNSS/IMU/DMR/EP paths and documents how to run
LCI (loosely coupled) and TCI (tightly coupled) POS solve in CoPre GUI.

Example:
  python experiments/run_copre_ppk_tc.py --setup
  python experiments/run_copre_ppk_tc.py --setup --mode lci
"""
from __future__ import annotations

import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.dom import minidom

DEFAULT_ROVER = Path(r"C:\Users\Andrea\Desktop\Rover")
DEFAULT_BASE = DEFAULT_ROVER / "base"
DEFAULT_EP_SRC = Path(
    r"C:\Users\Andrea\Desktop\AP7-007-CAR-SCAN1-WSM-20260307-s20020002J.EP"
)
DEFAULT_WORKSPACE = DEFAULT_ROVER / "CHC_20260611164137"
SESSION = "20260324_111523"

BASE_LAT = 34.668694694
BASE_LON = 33.042387809
BASE_H = 25.6743


def _posix(p: Path) -> str:
    return p.resolve().as_posix()


def _pretty_xml(tree: ET.ElementTree) -> str:
    raw = ET.tostring(tree.getroot(), encoding="unicode")
    parsed = minidom.parseString(raw)
    return parsed.toprettyxml(indent="    ")


def find_session_files(rover_dir: Path, session: str) -> dict[str, Path]:
    mapping = {
        "hcn": rover_dir / f"{session}_T.hcn",
        "gps": rover_dir / f"{session}_T.gps",
        "imr": rover_dir / f"{session}_I.imr",
        "dmr": rover_dir / f"{session}_D.dmr",
        "cil": rover_dir / f"{session}_S.cil",
    }
    missing = [k for k, p in mapping.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing rover files in {rover_dir}: {', '.join(missing)}"
        )
    return mapping


def ensure_ep(rover_dir: Path, ep_src: Path) -> Path:
    if not ep_src.exists():
        raise FileNotFoundError(f"EP calibration not found: {ep_src}")
    dst = rover_dir / ep_src.name
    if not dst.exists() or dst.stat().st_size != ep_src.stat().st_size:
        shutil.copy2(ep_src, dst)
    return dst


def ensure_base_files(base_dir: Path) -> dict[str, Path]:
    obs = list(base_dir.glob("*.crx")) + list(base_dir.glob("*.??o"))
    nav = list(base_dir.glob("brdc*.??n"))
    if not obs:
        raise FileNotFoundError(f"No base observation file in {base_dir}")
    if not nav:
        raise FileNotFoundError(f"No broadcast nav in {base_dir}")
    return {"obs": obs[0], "nav": nav[0]}


def update_rover_prj(
    prj_path: Path,
    rover_dir: Path,
    session: str,
    files: dict[str, Path],
    ep_path: Path,
) -> None:
    tree = ET.parse(prj_path)
    root = tree.getroot()
    gps = root.find("GpsInfo")
    if gps is None:
        raise ValueError(f"No GpsInfo in {prj_path}")

    rover_posix = _posix(rover_dir)
    gps.set("path", rover_posix)

    ppk = gps.find("PPKSettings")
    if ppk is not None:
        ppk.set("navsys", "31")  # GPS+GLO+GAL+BDS bitmask used by CoPre
        ppk.set("baseType", "1")  # field base

    def set_file(tag: str, fpath: Path, *, is_need: str | None = None) -> None:
        node = gps.find(tag)
        if node is None:
            node = ET.SubElement(gps, tag)
        rel = fpath.name
        node.set("relativePath", rel)
        node.set("path", _posix(fpath))
        if is_need is not None:
            node.set("isNeed", is_need)

    hcn = files["hcn"]
    set_file("PosFile", hcn)
    pos = gps.find("PosFile")
    if pos is not None:
        pos.set("name", hcn.name)

    set_file("ImrFile", files["imr"])
    set_file("DmrFile", files["dmr"], is_need="true")

    slam = gps.find("SlamPos")
    if slam is None:
        slam = ET.SubElement(gps, "SlamPos")
    ep_node = slam.find("EPFile")
    if ep_node is None:
        ep_node = ET.SubElement(slam, "EPFile")
    ep_node.set("relativePath", ep_path.name)
    ep_node.set("path", _posix(ep_path))
    stray = gps.find("EPFile")
    if stray is not None and stray is not ep_node:
        gps.remove(stray)

    prj_path.write_text(_pretty_xml(tree), encoding="utf-8")


def print_gui_steps(
    *,
    mode: str,
    workspace: Path,
    base: dict[str, Path],
    out_lci: Path,
    out_tci: Path,
) -> None:
    cpr = workspace / f"{workspace.name}.cpr"
    print("\n=== CoPre PPK-TC (GNSS+IMU) ===")
    print(f"Workspace: {_posix(workspace)}")
    print(f"Open:      {_posix(cpr)}")
    print("\nBase station (field):")
    print(f"  obs: {_posix(base['obs'])}")
    print(f"  nav: {_posix(base['nav'])}")
    print(f"  WGS84 approx: lat={BASE_LAT:.9f} lon={BASE_LON:.9f} h={BASE_H:.4f} m")
    print("\nGUI steps:")
    print("  1. Processing -> POS Solve")
    print("  2. Field Base Station = ON")
    print("  3. Add base -> obs file above + coordinates above")
    print("  4. Coupling mode:")
    if mode in ("lci", "both"):
        print("     - Loosely Coupled  -> R_RTK_LCI")
        print(f"     - Output folder: {_posix(out_lci)}")
    if mode in ("tci", "both"):
        print("     - Tightly Coupled  -> R_RTK_TCI")
        print(f"     - Output folder: {_posix(out_tci)}")
    print("  5. OK -> Start solve -> expect *.PosT trajectory")
    print("\nNotes:")
    print("  - EP file must be linked in project (Rover.prj updated by --setup)")
    print("  - Trig file warning is OK for POS-only (no camera)")
    print("  - Programmatic API needs missing CoPreInterface.dll + Python 3.7")


def setup(args: argparse.Namespace) -> None:
    rover_dir = args.rover.resolve()
    base_dir = args.base.resolve()
    workspace = args.workspace.resolve()
    prj = rover_dir / "Rover.prj"
    if not prj.exists():
        raise SystemExit(f"Rover.prj not found: {prj}")

    files = find_session_files(rover_dir, args.session)
    ep_path = ensure_ep(rover_dir, args.ep.resolve())
    base = ensure_base_files(base_dir)

    update_rover_prj(prj, rover_dir, args.session, files, ep_path)

    out_root = workspace / "Results" / "Rover"
    out_lci = out_root / "PPK_LCI"
    out_tci = out_root / "PPK_TCI"
    out_lci.mkdir(parents=True, exist_ok=True)
    out_tci.mkdir(parents=True, exist_ok=True)

    print("Updated:", prj)
    print("EP:     ", ep_path)
    print("Rover:  ", files["hcn"].name, files["imr"].name, files["dmr"].name)
    print_gui_steps(
        mode=args.mode,
        workspace=workspace,
        base=base,
        out_lci=out_lci,
        out_tci=out_tci,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare CoPre loose/tight PPK-TC")
    p.add_argument("--setup", action="store_true", help="Update Rover.prj and print steps")
    p.add_argument("--rover", type=Path, default=DEFAULT_ROVER)
    p.add_argument("--base", type=Path, default=DEFAULT_BASE)
    p.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    p.add_argument("--ep", type=Path, default=DEFAULT_EP_SRC)
    p.add_argument("--session", default=SESSION)
    p.add_argument(
        "--mode",
        choices=["lci", "tci", "both"],
        default="both",
        help="Which coupling mode instructions to show",
    )
    args = p.parse_args()
    if not args.setup:
        p.error("Use --setup to update Rover.prj")
    setup(args)


if __name__ == "__main__":
    main()
