"""Copia le DLL CoPre in bin/ (da aggiornare dopo upgrade CoPre)."""
from __future__ import annotations

import shutil
from pathlib import Path

COPRE = Path(r"C:\Program Files\CHCNAV\CoPre2\Release")
BIN = Path(__file__).resolve().parent / "bin"
FILES = (
    "GNSSQualityCheck.dll",
    "zlog.dll",
    "tinyxml2.dll",
    "pthreadVC2.dll",
    "RinexOption.xml",
)


def main() -> None:
    BIN.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        src = COPRE / name
        if not src.exists():
            raise SystemExit(f"Manca in CoPre: {src}")
        shutil.copy2(src, BIN / name)
        print("copied", name)


if __name__ == "__main__":
    main()
