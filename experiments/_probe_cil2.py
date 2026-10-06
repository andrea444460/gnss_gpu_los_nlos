from __future__ import annotations

import datetime
import struct
from pathlib import Path

import numpy as np

CIL = Path(r"c:\Users\Andrea\Desktop\20260324_111523_S.cil")
POS = Path(r"c:\Users\Andrea\Desktop\lidarcarLCOdo.pos")


def load_gprmc(cil: bytes) -> list[tuple[bytes, float, str]]:
    out = []
    m = b"$GPRMC"
    i = 0
    while True:
        i = cil.find(m, i)
        if i < 0:
            break
        end = cil.find(b"\n", i)
        line = cil[i:end].decode("ascii", errors="ignore")
        parts = line.split(",")
        if len(parts) >= 10 and parts[1] and parts[9]:
            hh = int(parts[1][0:2])
            mm = int(parts[1][2:4])
            ss = int(float(parts[1][4:]))
            day = int(parts[9][0:2])
            month = int(parts[9][2:4])
            year = 2000 + int(parts[9][4:6])
            dt = datetime.datetime(year, month, day, hh, mm, ss)
            sow = ((dt.weekday() + 1) % 7) * 86400 + hh * 3600 + mm * 60 + ss
            out.append((cil[max(0, i - 128) : i], float(sow), line[:80]))
        i += len(m)
    return out


def pos_time_at_hms(pos_path: Path, hh: int, mm: int, ss: int) -> float | None:
    target = f"{hh:02d}:{mm:02d}:{ss:02d}"
    with pos_path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) >= 16 and parts[1].startswith(target):
                return float(parts[15])
    return None


def main() -> None:
    cil = CIL.read_bytes()
    rec = load_gprmc(cil)
    print("gprmc count", len(rec))

    # POS time at GPRMC 12:37:12 from cil anchor
    pt = pos_time_at_hms(POS, 12, 37, 12)
    print("pos GPSTime @12:37:12", pt)

    for off in range(0, 128, 4):
        ticks = []
        times = []
        for pref, sow, _ in rec:
            if len(pref) < 128 - 4 + off + 4:
                continue
            base = len(pref) - 128
            ticks.append(struct.unpack_from("<I", pref, base + off)[0])
            times.append(sow)
        if len(ticks) < 100:
            continue
        dt = np.diff(ticks)
        dtm = np.diff(times)
        v = (dtm > 0) & (dtm < 5) & (dt > 0)
        if np.count_nonzero(v) < 50:
            continue
        med = float(np.median(dt[v] / dtm[v]))
        std = float(np.std(dt[v] / dtm[v]))
        if std / max(med, 1) < 0.01 and 1e3 < med < 5e10:
            print(f"u32@{off}: med_ratio={med:.3g} std_ratio={std:.3g}")

    # show prefix hex at 12:37:12 gprmc
    for pref, sow, line in rec:
        if line.startswith("$GPRMC,123712"):
            print("anchor line", line)
            print("prefix tail hex", pref[-64:].hex())
            for off in [0, 4, 8, 12, 16, 20, 24, 28, 32, 48, 52, 54, 56, 60]:
                print(f"  u32@{off}", struct.unpack_from("<I", pref, len(pref) - 64 + off)[0])
            break


if __name__ == "__main__":
    main()
