#!/usr/bin/env python3
"""Find likely obstruction directions from C/N0 drops (fixed GNSS station).

For each satellite, flags sudden C/N0 falls in the RINEX OBS (S*). Events are
joined with geometry labels (azimuth / elevation / LOS) when a labels CSV is
provided. Azimuth bins with many or deep drops suggest the sky direction of
blockage (e.g. a wind turbine near the station).

Example (Colab / local)::

    python experiments/analyze_terf_cnr_obstruction.py \\
        --obs-path /content/terf_work/data/TERF00CYP_R_20261920000_01D_30S_MO.rnx \\
        --labels-csv /content/terf_work/results/terf_los_labels_2026192_gc.csv \\
        --output-dir /content/terf_work/results \\
        --tag 2026192
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_ROOT / "python"))

from gnss_gpu.io.rinex import read_rinex_obs

GPS_EPOCH = datetime(1980, 1, 6)


def _gps_tow(dt: datetime) -> float:
    return (dt - GPS_EPOCH).total_seconds() % 604800.0


def _preferred_cnr_dbhz(obs_dict: dict) -> float | None:
    pref = ("S1C", "S1X", "S1P", "S1W", "S2W", "S2C", "S2X", "S5Q", "S5X", "S7Q", "S7X", "S8Q", "S8X")
    upper = {str(k).upper(): k for k in obs_dict}
    for code in pref:
        k = upper.get(code)
        if k is None:
            continue
        try:
            v = float(obs_dict[k])
        except (TypeError, ValueError):
            continue
        if math.isfinite(v) and v > 0.0:
            return v
    for uk, orig in sorted(upper.items()):
        if not uk.startswith("S"):
            continue
        try:
            v = float(obs_dict[orig])
        except (TypeError, ValueError):
            continue
        if math.isfinite(v) and v > 0.0:
            return v
    return None


def _normalize_prn(sat_id: str) -> str:
    s = str(sat_id).strip().upper()
    if len(s) >= 3 and s[0] in "GREJCI" and s[1:].isdigit():
        return f"{s[0]}{int(s[1:]):02d}"
    return s


def _load_geom_index(labels_csv: Path, tow_tol_s: float = 15.0) -> dict[tuple[str, int], dict]:
    """Map (prn, tow_bucket) → {az, el, is_los} from labels CSV."""
    out: dict[tuple[str, int], dict] = {}
    with open(labels_csv, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            prn = _normalize_prn(row.get("sat_id") or row.get("prn", ""))
            try:
                tow = float(row["gps_tow"])
                az = float(row["azimuth_deg"])
                el = float(row["elevation_deg"])
                is_los = int(row.get("is_los", 0))
            except (KeyError, TypeError, ValueError):
                continue
            bucket = int(round(tow / tow_tol_s))
            key = (prn, bucket)
            prev = out.get(key)
            if prev is None or abs(prev["tow"] - tow) > abs(tow - bucket * tow_tol_s):
                out[key] = {"tow": tow, "az": az, "el": el, "is_los": is_los}
    return out


@dataclass
class CnrSample:
    tow: float
    prn: str
    cnr: float


@dataclass
class DropEvent:
    tow: float
    prn: str
    cnr: float
    drop_db: float
    az: float | None
    el: float | None
    is_los: int | None


def _extract_cnr_series(obs_path: Path) -> list[CnrSample]:
    obs = read_rinex_obs(obs_path)
    out: list[CnrSample] = []
    for ep in obs.epochs:
        tow = float(_gps_tow(ep.time))
        for sat_id, odict in ep.observations.items():
            cnr = _preferred_cnr_dbhz(odict)
            if cnr is None:
                continue
            out.append(CnrSample(tow=tow, prn=_normalize_prn(str(sat_id)), cnr=float(cnr)))
    return out


def _detect_drops(
    samples: list[CnrSample],
    *,
    min_drop_db: float,
    median_margin_db: float,
    median_window: int,
    geom: dict[tuple[str, int], dict] | None,
    tow_tol_s: float,
    min_el_deg: float,
) -> list[DropEvent]:
    by_prn: dict[str, list[CnrSample]] = defaultdict(list)
    for s in samples:
        by_prn[s.prn].append(s)
    events: list[DropEvent] = []

    for prn, series in by_prn.items():
        series.sort(key=lambda x: x.tow)
        cnrs = np.array([s.cnr for s in series], dtype=np.float64)
        tows = np.array([s.tow for s in series], dtype=np.float64)
        if cnrs.size < 3:
            continue
        med = np.median(cnrs)
        for i in range(1, cnrs.size):
            step_drop = float(cnrs[i - 1] - cnrs[i])
            below_med = float(med - cnrs[i])
            if step_drop < float(min_drop_db) and below_med < float(median_margin_db):
                continue
            drop_db = max(step_drop, below_med)
            tow = float(tows[i])
            az = el = None
            is_los = None
            if geom is not None:
                g = geom.get((prn, int(round(tow / tow_tol_s))))
                if g is not None:
                    az = float(g["az"])
                    el = float(g["el"])
                    is_los = int(g["is_los"])
                    if el < float(min_el_deg):
                        continue
            events.append(
                DropEvent(
                    tow=tow,
                    prn=prn,
                    cnr=float(cnrs[i]),
                    drop_db=float(drop_db),
                    az=az,
                    el=el,
                    is_los=is_los,
                )
            )
    return events


def _azimuth_summary(events: list[DropEvent], bin_deg: float) -> list[dict]:
    with_az = [e for e in events if e.az is not None]
    if not with_az:
        return []
    bins: dict[int, list[DropEvent]] = defaultdict(list)
    n_bins = max(1, int(round(360.0 / bin_deg)))
    for e in with_az:
        b = int((e.az % 360.0) // bin_deg) % n_bins
        bins[b].append(e)
    rows = []
    for b, evs in sorted(bins.items(), key=lambda kv: (-len(kv[1]), -sum(x.drop_db for x in kv[1]))):
        az_lo = b * bin_deg
        az_hi = az_lo + bin_deg
        az_mid = (az_lo + az_hi) % 360.0
        rows.append(
            {
                "az_bin_lo_deg": az_lo,
                "az_bin_hi_deg": az_hi,
                "az_mid_deg": az_mid,
                "n_drops": len(evs),
                "mean_drop_db": float(np.mean([x.drop_db for x in evs])),
                "max_drop_db": float(np.max([x.drop_db for x in evs])),
                "example_tow_s": sorted({round(x.tow, 1) for x in evs})[:8],
                "example_prns": sorted({x.prn for x in evs})[:12],
            }
        )
    return rows


def _plot_skyplot(events: list[DropEvent], out_png: Path, title: str) -> None:
    with_az = [e for e in events if e.az is not None and e.el is not None]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
    if with_az:
        az_rad = np.radians([e.az for e in with_az])
        r = [90.0 - e.el for e in with_az]
        w = [max(4.0, min(80.0, e.drop_db * 3.0)) for e in with_az]
        colors = ["#ff6b6b" if (e.is_los == 0) else "#00d4aa" for e in with_az]
        ax.scatter(az_rad, r, s=w, c=colors, alpha=0.55, edgecolors="none")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 90)
    ax.set_yticks([0, 15, 30, 45, 60, 75, 90])
    ax.set_yticklabels(["90°", "75°", "60°", "45°", "30°", "15°", "0°"])
    ax.set_title(title + "\n(size ∝ drop dB; green=LOS label, red=NLOS)", va="bottom")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _plot_az_hist(summary: list[dict], out_png: Path, title: str) -> None:
    if not summary:
        return
    az = [r["az_mid_deg"] for r in summary]
    counts = [r["n_drops"] for r in summary]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(az, counts, width=360.0 / max(len(az), 1) * 0.85, align="center", color="#3b7ddd")
    ax.set_xlim(0, 360)
    ax.set_xlabel("Azimuth (deg from North)")
    ax.set_ylabel("# C/N0 drop events")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Detect obstruction azimuth from C/N0 drops (TERF).")
    p.add_argument("--obs-path", type=Path, required=True)
    p.add_argument("--labels-csv", type=Path, default=None, help="terf_los_labels_*_gc.csv for az/el")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--tag", type=str, default="", help="Day tag e.g. 2026192 (for output names)")
    p.add_argument("--min-drop-db", type=float, default=5.0, help="Min step drop in one OBS epoch [dB-Hz]")
    p.add_argument("--median-margin-db", type=float, default=8.0, help="Also flag if below PRN median by this")
    p.add_argument("--az-bin-deg", type=float, default=10.0)
    p.add_argument("--min-el-deg", type=float, default=8.0, help="Ignore drops below this elevation")
    p.add_argument("--tow-match-s", type=float, default=15.0)
    args = p.parse_args()

    tag = str(args.tag).strip() or args.obs_path.stem
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading OBS {args.obs_path} ...")
    samples = _extract_cnr_series(args.obs_path)
    print(f"  C/N0 samples: {len(samples)} ({len({s.prn for s in samples})} PRNs)")

    geom = None
    if args.labels_csv and args.labels_csv.is_file():
        print(f"Loading geometry {args.labels_csv} ...")
        geom = _load_geom_index(args.labels_csv, tow_tol_s=float(args.tow_match_s))
        print(f"  geometry keys: {len(geom)}")
    else:
        print("  No labels CSV — azimuth summary will be empty (pass --labels-csv).")

    events = _detect_drops(
        samples,
        min_drop_db=float(args.min_drop_db),
        median_margin_db=float(args.median_margin_db),
        median_window=21,
        geom=geom,
        tow_tol_s=float(args.tow_match_s),
        min_el_deg=float(args.min_el_deg),
    )
    print(f"  drop events: {len(events)}")

    summary = _azimuth_summary(events, bin_deg=float(args.az_bin_deg))
    top = summary[:5] if summary else []

    events_csv = args.output_dir / f"terf_cnr_drops_{tag}.csv"
    with open(events_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["gps_tow_s", "prn", "cnr_dbhz", "drop_db", "azimuth_deg", "elevation_deg", "is_los"])
        for e in sorted(events, key=lambda x: (-x.drop_db, x.tow)):
            w.writerow(
                [
                    f"{e.tow:.3f}",
                    e.prn,
                    f"{e.cnr:.2f}",
                    f"{e.drop_db:.2f}",
                    "" if e.az is None else f"{e.az:.3f}",
                    "" if e.el is None else f"{e.el:.3f}",
                    "" if e.is_los is None else e.is_los,
                ]
            )

    sky_png = args.output_dir / f"terf_cnr_drops_skyplot_{tag}.png"
    hist_png = args.output_dir / f"terf_cnr_drops_azimuth_{tag}.png"
    _plot_skyplot(events, sky_png, f"TERF C/N0 drops — {tag}")
    _plot_az_hist(summary, hist_png, f"TERF drop count by azimuth — {tag}")

    result = {
        "tag": tag,
        "obs_path": str(args.obs_path),
        "labels_csv": str(args.labels_csv) if args.labels_csv else None,
        "n_samples": len(samples),
        "n_drop_events": len(events),
        "min_drop_db": float(args.min_drop_db),
        "top_azimuth_bins": top,
        "outputs": {
            "events_csv": str(events_csv),
            "skyplot_png": str(sky_png),
            "azimuth_hist_png": str(hist_png),
        },
    }
    if top:
        best = top[0]
        result["suspected_obstruction_azimuth_deg"] = {
            "lo": best["az_bin_lo_deg"],
            "hi": best["az_bin_hi_deg"],
            "mid": best["az_mid_deg"],
            "n_drops": best["n_drops"],
            "example_gps_tow_s": best["example_tow_s"],
        }
        print(
            f"\nLikely obstruction azimuth ~{best['az_mid_deg']:.0f}° "
            f"({best['az_bin_lo_deg']:.0f}–{best['az_bin_hi_deg']:.0f}°): "
            f"{best['n_drops']} drops, max {best['max_drop_db']:.1f} dB"
        )
        print(f"  Inspect viz near GPS TOW (s): {best['example_tow_s'][:5]}")

    out_json = args.output_dir / f"terf_cnr_obstruction_{tag}.json"
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nWrote {events_csv}")
    print(f"Wrote {sky_png}")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
