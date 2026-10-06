"""Convert CHCNAV proprietary GNSS files using CoPre DLLs or RTKLIB.

Supported inputs:
  - *.hcn  -> RINEX via GNSSQualityCheck.dll (SolveHcnToRinexEx)
  - *.gps  -> RINEX via RTKLIB convbin (NovAtel OEM7 binary)

Examples:
  python experiments/convert_chcnav_gnss.py hcn "C:\\path\\rover_T.hcn"
  python experiments/convert_chcnav_gnss.py gps "C:\\path\\rover_T.gps"
"""
from __future__ import annotations

import argparse
import ctypes
import os
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_COPRE = Path(r"C:\Program Files\CHCNAV\CoPre2\Release")
DEFAULT_CONVBIN = Path(
    r"C:\Program Files (x86)\RTKLIB_EX_2.5.0\RTKLIB_EX_2.5.0\convbin.EXE"
)


def convert_hcn(hcn: Path, out_dir: Path, copre: Path) -> bool:
    os.chdir(copre)
    os.add_dll_directory(str(copre))
    dll = ctypes.CDLL(str(copre / "GNSSQualityCheck.dll"))
    fn = dll["?SolveHcnToRinexEx@GNSSQualityCheck@@SA_NPEBD00NNH_N@Z"]
    fn.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.c_bool,
    ]
    fn.restype = ctypes.c_bool

    out_dir.mkdir(parents=True, exist_ok=True)
    opt = str(copre / "RinexOption.xml").encode()
    ok = fn(str(hcn).encode(), str(out_dir).encode(), opt, 0.0, 0.0, 0, False)
    return bool(ok)


def convert_gps(gps: Path, out_prefix: Path, convbin: Path) -> bool:
    if not convbin.exists():
        raise FileNotFoundError(f"convbin not found: {convbin}")
    cmd = [
        str(convbin),
        "-r",
        "nov",
        "-od",
        "-os",
        "-v",
        "3.02",
        "-o",
        str(out_prefix),
        str(gps),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
    return proc.returncode == 0


def main() -> None:
    p = argparse.ArgumentParser(description="Convert CHCNAV GNSS to RINEX")
    p.add_argument("kind", choices=["hcn", "gps"], help="Input format")
    p.add_argument("input", type=Path, help="Input file path")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output directory (hcn) or prefix (gps). Default: next to input",
    )
    p.add_argument("--copre", type=Path, default=DEFAULT_COPRE)
    p.add_argument("--convbin", type=Path, default=DEFAULT_CONVBIN)
    args = p.parse_args()

    inp = args.input.resolve()
    if not inp.exists():
        raise SystemExit(f"Input not found: {inp}")

    if args.kind == "hcn":
        out = (args.output or inp.parent).resolve()
        ok = convert_hcn(inp, out, args.copre.resolve())
        print(f"HCN -> RINEX: {'OK' if ok else 'FAILED'}")
        print(f"Look for *.??o / *.??n next to: {out}")
        raise SystemExit(0 if ok else 1)

    out_prefix = (args.output or inp.with_suffix("")).resolve()
    ok = convert_gps(inp, out_prefix, args.convbin.resolve())
    obs = Path(str(out_prefix))
    print(f"GPS -> RINEX: {'OK' if ok else 'FAILED'}")
    print(f"Observation file: {obs}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
