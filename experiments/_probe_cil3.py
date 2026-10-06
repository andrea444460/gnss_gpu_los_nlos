from __future__ import annotations

import datetime
import struct
from pathlib import Path

import numpy as np

CIL = Path(r"c:\Users\Andrea\Desktop\20260324_111523_S.cil")
POS = Path(r"c:\Users\Andrea\Desktop\lidarcarLCOdo.pos")


def gprmc_list(cil: bytes) -> list[tuple[int, str]]:
    out = []
    marker = b"$GPRMC"
    i = 0
    while True:
        i = cil.find(marker, i)
        if i < 0:
            break
        end = cil.find(b"\n", i)
        out.append((i, cil[i:end].decode("ascii", errors="ignore")))
        i += len(marker)
    return out


def pos_gps_times(pos_path: Path) -> np.ndarray:
    t = []
    with pos_path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            p = line.split()
            if len(p) >= 16 and p[0].startswith("202"):
                t.append(float(p[15]))
    return np.asarray(t)


def main() -> None:
    cil = CIL.read_bytes()
    pos_t = pos_gps_times(POS)
    rec = gprmc_list(cil)

    # find u32 offset before GPRMC matching POS time scale
    hit_counts: dict[int, int] = {}
    samples: dict[int, list[float]] = {}
    for idx, (pos, line) in enumerate(rec[:500]):
        pref = cil[max(0, pos - 128) : pos]
        parts = line.split(",")
        if len(parts) < 10:
            continue
        hh, mm = int(parts[1][0:2]), int(parts[1][2:4])
        ss = int(float(parts[1][4:]))
        day, month, year = int(parts[9][0:2]), int(parts[9][2:4]), 2000 + int(parts[9][4:6])
        dt = datetime.datetime(year, month, day, hh, mm, ss)
        # nearest pos time
        sow_nmea = ((dt.weekday() + 1) % 7) * 86400 + hh * 3600 + mm * 60 + ss
        for off in range(0, len(pref) - 3, 4):
            v = struct.unpack_from("<I", pref, off)[0]
            if 213000 < v < 219000:
                hit_counts[off] = hit_counts.get(off, 0) + 1
                samples.setdefault(off, []).append(v)

    top = sorted(hit_counts.items(), key=lambda x: -x[1])[:10]
    print("top u32 offsets matching POS scale", top)
    if top:
        off = top[0][0]
        vals = np.asarray(samples[off])
        print(f"off={off} min={vals.min()} max={vals.max()} n={len(vals)}")

    # correlate LTP tick with cil time field at best offset
    best_off = top[0][0] if top else 60
    cil_pairs = []
    for pos, line in rec:
        pref = cil[max(0, pos - 128) : pos]
        if len(pref) < best_off + 4:
            continue
        tpos = struct.unpack_from("<I", pref, best_off)[0]
        if 213000 < tpos < 219000:
            cil_pairs.append((pos, float(tpos)))

    print("cil time pairs", len(cil_pairs), cil_pairs[0], cil_pairs[-1])

    # LTP first frames
    ltp = Path(r"c:\Users\Andrea\Desktop\2026-03-24-123712_002.ltp").read_bytes()[:5_000_000]
    p = ltp.find(b"\x5a\x23\x05")
    ltp_pairs = []
    while p + 1461 <= len(ltp) and len(ltp_pairs) < 500:
        if ltp[p : p + 3] != b"\x5a\x23\x05":
            p += 1
            continue
        nxt = ltp.find(b"\x5a\x23\x05", p + 4)
        if nxt - p != 1461:
            p += 1
            continue
        tick = struct.unpack_from("<I", ltp, p + 4)[0]
        ltp_pairs.append(tick)
        p = nxt

    ticks = np.asarray(ltp_pairs, float)
    # map linear tick -> pos time using endpoints
    t0, t1 = 218232.0, 218563.0
    mapped = t0 + (ticks - ticks[0]) / (ticks[-1] - ticks[0]) * (t1 - t0)
    print("ltp mapped", mapped[0], mapped[-1], "tick step", np.median(np.diff(ticks)))


if __name__ == "__main__":
    main()
