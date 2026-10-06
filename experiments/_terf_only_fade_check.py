#!/usr/bin/env python3
"""Find PRNs with fade mainly or only at TERF vs other stations."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVENTS = ROOT / "experiments/results/gnss_stations_sat_risk/fade_time_sync/fade_time_events.csv"
OTHERS = ("GREC", "MAIA", "MAXI", "TROD")
TAG = "2026194"


def main() -> None:
    rows = list(csv.DictReader(EVENTS.open(encoding="utf-8")))
    cnt: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in rows:
        if r["tag"] != TAG:
            continue
        cnt[r["station"]][r["prn"]] += 1

    terf = cnt.get("TERF", {})
    terf_prns = sorted(terf, key=lambda k: -terf[k])

    print(f"Giorno {TAG} — confronto eventi fade (>=6 dB, el>=20°)\n")
    print("PRN   TERF  GREC  MAIA  MAXI  TROD  note")
    terf_only: list[tuple[str, int]] = []
    terf_dominant: list[tuple[str, int, dict[str, int]]] = []

    for prn in terf_prns:
        t = terf[prn]
        o = {s: cnt[s].get(prn, 0) for s in OTHERS}
        mx = max(o.values())
        sm = sum(o.values())
        note = ""
        if sm == 0:
            note = "SOLO TERF"
            terf_only.append((prn, t))
        elif t >= 3 * max(1, mx) and t > 10:
            note = "quasi solo TERF"
            terf_dominant.append((prn, t, o))
        print(
            f"{prn:4s} {t:5d} {o['GREC']:5d} {o['MAIA']:5d} {o['MAXI']:5d} {o['TROD']:5d}  {note}"
        )

    print("\n=== Fade SOLO su TERF (0 eventi sulle altre) ===")
    if terf_only:
        for prn, t in terf_only:
            print(f"  {prn}: {t} eventi")
    else:
        print("  nessuno")

    print("\n=== Fade molto più forte su TERF (>=3× le altre, TERF>10) ===")
    for prn, t, o in terf_dominant:
        print(f"  {prn}: TERF={t}, altre={o}")


if __name__ == "__main__":
    main()
