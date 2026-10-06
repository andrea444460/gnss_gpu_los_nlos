#!/usr/bin/env python3
"""Find obstruction directions from *anomalous* C/N0 dips (fixed GNSS station).

Ignores the usual low-elevation C/N0 fade: only flags transient dips at
elevation >= min-el, with recovery shortly after, while the satellite is not
rapidly setting. Clusters in azimuth suggest a narrow sky blocker (e.g. wind
turbine).

Local example::

    python experiments/analyze_terf_cnr_obstruction.py \\
        --obs-path experiments/data/TERF/TERF00CYP_R_20261920000_01D_30S_MO.rnx \\
        --nav-path experiments/data/TERF/BRDC00IGS_R_20261920000_01D_MN.rnx \\
        --reference-csv experiments/data/TERF/reference_2026192.csv \\
        --output-dir experiments/results/terf_cnr_audit \\
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

from gnss_gpu.ephemeris import Ephemeris
from gnss_gpu.io.nav_rinex import read_nav_rinex_multi
from gnss_gpu.io.rinex import read_rinex_obs
from gnss_gpu.urban_signal_sim import _sat_elevation_azimuth

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


def _load_rx_ecef(reference_csv: Path) -> np.ndarray:
    with open(reference_csv, newline="", encoding="utf-8") as fh:
        row = next(csv.DictReader(fh))
    return np.array(
        [float(row["ECEF X (m)"]), float(row["ECEF Y (m)"]), float(row["ECEF Z (m)"])],
        dtype=np.float64,
    )


def _lookup_az_el_nav(eph: Ephemeris, rx: np.ndarray, tow: float, prn: str) -> tuple[float, float] | None:
    try:
        sat_ecef, _clk, used = eph.compute(float(tow), [prn])
    except Exception:
        return None
    if not used:
        return None
    el_rad, az_rad = _sat_elevation_azimuth(rx, sat_ecef[0:1])
    return float(math.degrees(float(az_rad[0]))) % 360.0, float(math.degrees(float(el_rad[0])))


def _lookup_geom(
    prn: str,
    tow: float,
    *,
    geom: dict[tuple[str, int], dict] | None,
    tow_tol_s: float,
    eph: Ephemeris | None,
    rx_ecef: np.ndarray | None,
) -> tuple[float | None, float | None, int | None]:
    az = el = None
    is_los = None
    if geom is not None:
        g = geom.get((prn, int(round(tow / tow_tol_s))))
        if g is not None:
            return float(g["az"]), float(g["el"]), int(g["is_los"])
    if eph is not None and rx_ecef is not None:
        nav_geom = _lookup_az_el_nav(eph, rx_ecef, tow, prn)
        if nav_geom is not None:
            az, el = nav_geom
    return az, el, is_los


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
    recovery_db: float
    az: float | None
    el: float | None
    is_los: int | None
    residual_db: float | None


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


def _elevation_median_model(
  pairs: list[tuple[float, float]], *, el_bin_deg: float
) -> dict[int, float]:
    """prn-agnostic elevation → median C/N0 [dB-Hz] from all samples."""
    bins: dict[int, list[float]] = defaultdict(list)
    for el, cnr in pairs:
        if el < 0.0:
            continue
        bins[int(el // el_bin_deg)].append(cnr)
    return {b: float(np.median(v)) for b, v in bins.items() if v}


def _expected_cnr(el: float, model: dict[int, float], el_bin_deg: float) -> float | None:
    b = int(el // el_bin_deg)
    if b in model:
        return model[b]
    return None


def _enrich_series(
    series: list[CnrSample],
    *,
    geom: dict[tuple[str, int], dict] | None,
    tow_tol_s: float,
    eph: Ephemeris | None,
    rx_ecef: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(series)
    tows = np.array([s.tow for s in series], dtype=np.float64)
    cnrs = np.array([s.cnr for s in series], dtype=np.float64)
    els = np.full(n, np.nan, dtype=np.float64)
    azs = np.full(n, np.nan, dtype=np.float64)
    los = np.full(n, -1, dtype=np.int8)
    prn = series[0].prn
    for i, s in enumerate(series):
        az, el, is_los = _lookup_geom(
            prn, s.tow, geom=geom, tow_tol_s=tow_tol_s, eph=eph, rx_ecef=rx_ecef
        )
        if el is not None:
            els[i] = el
        if az is not None:
            azs[i] = az
        if is_los is not None:
            los[i] = is_los
    return tows, cnrs, els, azs, los


def _detect_transient_dips(
    samples: list[CnrSample],
    *,
    min_drop_db: float,
    min_recovery_db: float,
    min_residual_db: float,
    window_epochs: int,
    min_el_deg: float,
    max_el_spread_deg: float,
    max_set_rate_deg: float,
    el_bin_deg: float,
    geom: dict[tuple[str, int], dict] | None,
    tow_tol_s: float,
    eph: Ephemeris | None,
    rx_ecef: np.ndarray | None,
) -> list[DropEvent]:
    """Transient V-shaped C/N0 dips at moderate/high elevation (not horizon fade)."""
    by_prn: dict[str, list[CnrSample]] = defaultdict(list)
    for s in samples:
        by_prn[s.prn].append(s)

    # Global elevation model from high-el samples (one NAV pass per PRN).
    el_cnr_pairs: list[tuple[float, float]] = []
    enriched: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for prn, series in by_prn.items():
        series.sort(key=lambda x: x.tow)
        pack = _enrich_series(
            series, geom=geom, tow_tol_s=tow_tol_s, eph=eph, rx_ecef=rx_ecef
        )
        enriched[prn] = pack
        _tows, cnrs, els, _azs, _los = pack
        for i in range(cnrs.size):
            if np.isfinite(els[i]) and float(els[i]) >= float(min_el_deg):
                el_cnr_pairs.append((float(els[i]), float(cnrs[i])))
    elev_model = _elevation_median_model(el_cnr_pairs, el_bin_deg=el_bin_deg)

    events: list[DropEvent] = []
    half_w = max(1, int(window_epochs))

    for prn, series in by_prn.items():
        series.sort(key=lambda x: x.tow)
        n = len(series)
        if n < 2 * half_w + 3:
            continue

        tows, cnrs, els, azs, los = enriched[prn]
        for i in range(half_w, n - half_w):
            if not np.isfinite(els[i]) or float(els[i]) < float(min_el_deg):
                continue

            lo = i - half_w
            hi = i + half_w + 1
            el_win = els[lo:hi]
            if not np.all(np.isfinite(el_win)):
                continue
            if float(np.max(el_win) - np.min(el_win)) > float(max_el_spread_deg):
                continue

            # Reject horizon setting: elevation falling fast while C/N0 drops.
            el_rate = float(els[i] - els[i - 1])
            step_drop = float(cnrs[i - 1] - cnrs[i])
            if el_rate < -float(max_set_rate_deg) and step_drop > 1.0:
                continue

            left_max = float(np.max(cnrs[lo:i]))
            right_max = float(np.max(cnrs[i + 1 : hi]))
            trough = float(cnrs[i])
            drop_db = min(left_max - trough, right_max - trough)
            recovery_db = right_max - trough
            if drop_db < float(min_drop_db) or recovery_db < float(min_recovery_db):
                continue

            expected = _expected_cnr(float(els[i]), elev_model, el_bin_deg)
            residual_db = None
            if expected is not None:
                residual_db = float(expected - trough)
                if residual_db < float(min_residual_db):
                    continue

            events.append(
                DropEvent(
                    tow=float(tows[i]),
                    prn=prn,
                    cnr=trough,
                    drop_db=drop_db,
                    recovery_db=recovery_db,
                    az=float(azs[i]) if np.isfinite(azs[i]) else None,
                    el=float(els[i]) if np.isfinite(els[i]) else None,
                    is_los=int(los[i]) if los[i] >= 0 else None,
                    residual_db=residual_db,
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
    score = lambda evs: sum(e.drop_db * (e.residual_db or e.drop_db) for e in evs)
    for b, evs in sorted(bins.items(), key=lambda kv: (-score(kv[1]), -len(kv[1]))):
        az_lo = b * bin_deg
        az_hi = az_lo + bin_deg
        az_mid = (az_lo + az_hi) / 2.0
        mean_el = float(np.mean([e.el for e in evs if e.el is not None])) if evs else 0.0
        rows.append(
            {
                "az_bin_lo_deg": az_lo,
                "az_bin_hi_deg": az_hi,
                "az_mid_deg": az_mid,
                "n_dips": len(evs),
                "mean_el_deg": mean_el,
                "mean_drop_db": float(np.mean([x.drop_db for x in evs])),
                "max_drop_db": float(np.max([x.drop_db for x in evs])),
                "mean_residual_db": float(
                    np.mean([x.residual_db for x in evs if x.residual_db is not None])
                )
                if any(x.residual_db is not None for x in evs)
                else None,
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
        w = [max(8.0, min(120.0, e.drop_db * 4.0)) for e in with_az]
        colors = [
            "#ff6b6b" if e.is_los == 0 else "#00d4aa" if e.is_los == 1 else "#e6b800"
            for e in with_az
        ]
        ax.scatter(az_rad, r, s=w, c=colors, alpha=0.7, edgecolors="white", linewidths=0.3)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 90)
    ax.set_yticks([0, 15, 30, 45, 60, 75, 90])
    ax.set_yticklabels(["90°", "75°", "60°", "45°", "30°", "15°", "0°"])
    ax.set_title(
        title + "\n(transient dips, el≥min; size∝drop; gold=unknown LOS)",
        va="bottom",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _plot_az_hist(summary: list[dict], out_png: Path, title: str) -> None:
    if not summary:
        return
    az = [r["az_mid_deg"] for r in summary]
    counts = [r["n_dips"] for r in summary]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(az, counts, width=360.0 / max(len(az), 1) * 0.85, align="center", color="#e6b800")
    ax.set_xlim(0, 360)
    ax.set_xlabel("Azimuth (deg from North)")
    ax.set_ylabel("# transient C/N0 dips")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Detect obstruction azimuth from anomalous C/N0 dips (TERF).")
    p.add_argument("--obs-path", type=Path, required=True)
    p.add_argument("--labels-csv", type=Path, default=None)
    p.add_argument("--nav-path", type=Path, default=None)
    p.add_argument("--reference-csv", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--tag", type=str, default="")
    p.add_argument("--min-drop-db", type=float, default=6.0, help="Min V-dip depth [dB-Hz]")
    p.add_argument("--min-recovery-db", type=float, default=4.0, help="Min recovery after trough [dB-Hz]")
    p.add_argument("--min-residual-db", type=float, default=5.0, help="Below elevation-median model at trough")
    p.add_argument("--min-el-deg", type=float, default=25.0, help="Ignore dips below this elevation")
    p.add_argument("--window-epochs", type=int, default=4, help="Half-window for V-dip (epochs @ 30 s)")
    p.add_argument("--max-el-spread-deg", type=float, default=8.0, help="Max el change in dip window")
    p.add_argument("--max-set-rate-deg", type=float, default=0.8, help="Reject fast setting + drop")
    p.add_argument("--el-bin-deg", type=float, default=5.0, help="Elevation bin for C/N0 median model")
    p.add_argument("--az-bin-deg", type=float, default=5.0)
    p.add_argument("--tow-match-s", type=float, default=15.0)
    args = p.parse_args()

    tag = str(args.tag).strip() or args.obs_path.stem
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading OBS {args.obs_path} ...")
    samples = _extract_cnr_series(args.obs_path)
    print(f"  C/N0 samples: {len(samples)} ({len({s.prn for s in samples})} PRNs)")

    geom = None
    eph = None
    rx_ecef = None
    if args.labels_csv and args.labels_csv.is_file():
        print(f"Loading geometry {args.labels_csv} ...")
        geom = _load_geom_index(args.labels_csv, tow_tol_s=float(args.tow_match_s))
        print(f"  geometry keys: {len(geom)}")
    elif args.nav_path and args.reference_csv:
        print(f"Loading NAV {args.nav_path} + reference {args.reference_csv} ...")
        rx_ecef = _load_rx_ecef(args.reference_csv)
        nav_messages = read_nav_rinex_multi(args.nav_path)
        eph = Ephemeris(nav_messages)
        print(f"  ephemeris PRNs: {len(eph.available_prns)}")
    else:
        raise SystemExit("Need --labels-csv or (--nav-path + --reference-csv) for azimuth/elevation.")

    print(
        f"  Detecting transient dips: el>={args.min_el_deg} deg, "
        f"drop>={args.min_drop_db} dB, recovery>={args.min_recovery_db} dB, "
        f"residual>={args.min_residual_db} dB vs elevation model"
    )
    events = _detect_transient_dips(
        samples,
        min_drop_db=float(args.min_drop_db),
        min_recovery_db=float(args.min_recovery_db),
        min_residual_db=float(args.min_residual_db),
        window_epochs=int(args.window_epochs),
        min_el_deg=float(args.min_el_deg),
        max_el_spread_deg=float(args.max_el_spread_deg),
        max_set_rate_deg=float(args.max_set_rate_deg),
        el_bin_deg=float(args.el_bin_deg),
        geom=geom,
        tow_tol_s=float(args.tow_match_s),
        eph=eph,
        rx_ecef=rx_ecef,
    )
    print(f"  transient dip events: {len(events)}")

    summary = _azimuth_summary(events, bin_deg=float(args.az_bin_deg))
    top = summary[:8] if summary else []

    events_csv = args.output_dir / f"terf_cnr_dips_{tag}.csv"
    with open(events_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "gps_tow_s",
                "prn",
                "cnr_dbhz",
                "drop_db",
                "recovery_db",
                "residual_db",
                "azimuth_deg",
                "elevation_deg",
                "is_los",
            ]
        )
        for e in sorted(events, key=lambda x: (-(x.residual_db or 0), -x.drop_db, x.tow)):
            w.writerow(
                [
                    f"{e.tow:.3f}",
                    e.prn,
                    f"{e.cnr:.2f}",
                    f"{e.drop_db:.2f}",
                    f"{e.recovery_db:.2f}",
                    "" if e.residual_db is None else f"{e.residual_db:.2f}",
                    "" if e.az is None else f"{e.az:.3f}",
                    "" if e.el is None else f"{e.el:.3f}",
                    "" if e.is_los is None else e.is_los,
                ]
            )

    sky_png = args.output_dir / f"terf_cnr_dips_skyplot_{tag}.png"
    hist_png = args.output_dir / f"terf_cnr_dips_azimuth_{tag}.png"
    _plot_skyplot(events, sky_png, f"TERF transient C/N0 dips — {tag} (el≥{args.min_el_deg:.0f}°)")
    _plot_az_hist(summary, hist_png, f"Transient dip count by azimuth — {tag}")

    result = {
        "tag": tag,
        "method": "transient_high_el_dip",
        "filters": {
            "min_el_deg": float(args.min_el_deg),
            "min_drop_db": float(args.min_drop_db),
            "min_recovery_db": float(args.min_recovery_db),
            "min_residual_db": float(args.min_residual_db),
        },
        "n_samples": len(samples),
        "n_dip_events": len(events),
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
            "n_dips": best["n_dips"],
            "mean_el_deg": best["mean_el_deg"],
            "example_gps_tow_s": best["example_tow_s"],
        }
        print(
            f"\nCandidate obstruction ~{best['az_mid_deg']:.0f}° "
            f"({best['az_bin_lo_deg']:.0f}–{best['az_bin_hi_deg']:.0f}°): "
            f"{best['n_dips']} transient dips, mean el {best['mean_el_deg']:.1f}°"
        )
        print(f"  Inspect viz near GPS TOW (s): {best['example_tow_s'][:5]}")

    out_json = args.output_dir / f"terf_cnr_obstruction_{tag}.json"
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nWrote {events_csv}")
    print(f"Wrote {sky_png}")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
