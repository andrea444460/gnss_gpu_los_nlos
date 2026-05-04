#!/usr/bin/env python3
"""Same as ``build_3d_visualization.py`` but defaults to ``--filter-by-obs``.

Rays are restricted to satellites that have non-zero code pseudorange (C*) in the
rover RINEX observation file at the nearest GPS TOW epoch.

Pass ``--no-filter-by-obs`` to behave like the main script (OBS filter off).
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
_PYTHON_PKG = _REPO_ROOT / "python"
if _PYTHON_PKG.is_dir() and str(_PYTHON_PKG) not in sys.path:
    sys.path.insert(0, str(_PYTHON_PKG))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from build_3d_visualization import main as _main  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    if "--no-filter-by-obs" in argv:
        argv = [a for a in argv if a != "--no-filter-by-obs"]
    else:
        argv = ["--filter-by-obs"] + argv
    _main(argv)


if __name__ == "__main__":
    main()
