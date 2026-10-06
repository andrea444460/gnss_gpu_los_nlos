#!/usr/bin/env python3
"""Build interactive HTML: C/N0 time series for all stations and at-risk PRNs.

Example::

    python experiments/build_gnss_stations_cnr_timeseries_html.py \\
        --data-dir "experiments/data/GNSS Stations" \\
        --crd-path "experiments/data/GNSS Stations/NEW_CYCLOPS_STATIONS.CRD" \\
        --output experiments/results/gnss_stations_sat_risk/cnr_timeseries_viewer.html
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from glob import glob
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_EXPERIMENTS = Path(__file__).resolve().parent
if str(_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_ROOT / "python"))
if str(_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS))

from gnss_gpu.ephemeris import Ephemeris
from gnss_gpu.io.nav_rinex import read_nav_rinex_multi
from gnss_gpu.io.rinex import read_rinex_obs
from gnss_gpu.urban_signal_sim import _sat_elevation_azimuth

import analyze_gnss_stations_sat_risk_batch as sat
import analyze_terf_satellite_risk as risk

_STATION_RE = re.compile(r"^([A-Z0-9]{4})00CYP_R_(\d{7})\d+_01D_", re.I)
GPS_EPOCH = datetime(1980, 1, 6)
GPS_EPOCH_UTC = datetime(1980, 1, 6, tzinfo=timezone.utc)


def _load_at_risk_prns(results_dir: Path, *, max_prns: int) -> list[str]:
    sticky_csv = results_dir / "all_stations_sticky_prns.csv"
    scores: dict[str, float] = {}
    counts: dict[str, int] = {}
    if sticky_csv.is_file():
        with sticky_csv.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                prn = str(row["prn"]).strip().upper()
                fade = float(row.get("fade_pct") or 0.0)
                counts[prn] = counts.get(prn, 0) + 1
                scores[prn] = max(scores.get(prn, 0.0), fade)
    if not scores:
        for md in results_dir.glob("*/sat_risk_report.md"):
            text = md.read_text(encoding="utf-8", errors="replace")
            for m in re.finditer(
                r"^\| (G\d{2}|E\d{2}|R\d{2}|C\d{2}|J\d{2}) \| ([\d.]+)%",
                text,
                re.M,
            ):
                prn = m.group(1).upper()
                fade = float(m.group(2))
                counts[prn] = counts.get(prn, 0) + 1
                scores[prn] = max(scores.get(prn, 0.0), fade)
    if not scores:
        return ["G08", "G24", "G27", "G32", "G18", "G30"][:max_prns]
    ranked = sorted(
        scores,
        key=lambda p: (-counts.get(p, 0), -scores[p], p),
    )
    return ranked[: max(1, int(max_prns))]


def _group_obs(data_dir: Path) -> dict[str, list[Path]]:
    return sat._group_obs(data_dir)


def _merge_terf_obs(groups: dict[str, list[Path]], data_dir: Path, terf_extra_dir: Path | None) -> dict[str, list[Path]]:
    """TERF: merge extra OBS days (e.g. experiments/data/TERF) with data-dir."""
    out = dict(groups)
    terf_paths = list(out.get("TERF", []))
    if terf_extra_dir and terf_extra_dir.is_dir():
        terf_paths.extend(sorted(terf_extra_dir.glob("TERF00CYP_R_*_MO.rnx")))
    if not terf_paths:
        return out
    by_tag: dict[str, Path] = {}
    for p in terf_paths:
        m = _STATION_RE.match(p.name)
        if not m:
            continue
        tag = m.group(2)
        # Prefer file under data_dir when same day exists in both places.
        if tag not in by_tag or p.parent.resolve() == data_dir.resolve():
            by_tag[tag] = p
    out["TERF"] = [by_tag[t] for t in sorted(by_tag)]
    return out


def _utc_hour_from_tow(tow: float, tag: str) -> float:
    sec = float(tow) % 86400.0
    return sec / 3600.0


def _collect_day_series(
    obs_path: Path,
    eph: Ephemeris,
    rx: np.ndarray,
    *,
    prn_filter: set[str],
    min_el_deg: float,
    fade_db: float,
    epoch_step: int,
) -> dict[str, dict]:
    step = max(1, int(epoch_step))
    obs = read_rinex_obs(obs_path)
    raw: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)

    for ei, ep in enumerate(obs.epochs):
        if step > 1 and (ei % step) != 0:
            continue
        tow = float((ep.time - GPS_EPOCH).total_seconds() % 604800.0)
        prns: list[str] = []
        cnrs: list[float] = []
        for sat_id, odict in ep.observations.items():
            prn = sat._normalize_prn(str(sat_id))
            if prn not in prn_filter:
                continue
            cnr = sat._preferred_cnr(odict)
            if cnr is None:
                continue
            prns.append(prn)
            cnrs.append(float(cnr))
        if not prns:
            continue
        try:
            sat_ecef, _, used = eph.compute(tow, prn_list=prns)
        except Exception:
            continue
        if not used:
            continue
        used_map = {str(p).strip().upper(): i for i, p in enumerate(used)}
        sel_idx: list[int] = []
        sel_prn: list[str] = []
        sel_cnr: list[float] = []
        for prn, cnr in zip(prns, cnrs):
            j = used_map.get(prn)
            if j is None:
                continue
            sel_idx.append(j)
            sel_prn.append(prn)
            sel_cnr.append(cnr)
        if not sel_idx:
            continue
        sat_sel = np.asarray(sat_ecef[sel_idx], dtype=np.float64)
        el_rad, az_rad = _sat_elevation_azimuth(rx, sat_sel)
        el_deg = np.degrees(el_rad)
        az_deg = np.degrees(az_rad) % 360.0
        for k in range(len(sel_prn)):
            if float(el_deg[k]) < min_el_deg:
                continue
            raw[sel_prn[k]].append((tow, sel_cnr[k], float(el_deg[k]), float(az_deg[k])))

    out: dict[str, dict] = {}
    fade = float(fade_db)
    for prn, pts in raw.items():
        pts.sort(key=lambda x: x[0])
        tow = np.array([p[0] for p in pts], dtype=np.float64)
        cnr = np.array([p[1] for p in pts], dtype=np.float64)
        el = np.array([p[2] for p in pts], dtype=np.float32)
        az = np.array([p[3] for p in pts], dtype=np.float32)
        med = float(np.median(cnr))
        norm = (cnr - med).astype(np.float32)
        is_fade = (cnr < med - fade).astype(np.int8)
        out[prn] = {
            "tow": np.round(tow, 1).tolist(),
            "utc_h": np.round(np.mod(tow, 86400.0) / 3600.0, 4).tolist(),
            "cnr": np.round(cnr, 1).tolist(),
            "cnr_norm": np.round(norm, 2).tolist(),
            "el": np.round(el, 1).tolist(),
            "az": np.round(az, 1).tolist(),
            "fade": is_fade.tolist(),
            "median_db": round(med, 1),
        }
    return out


_HTML = r"""<!DOCTYPE html>
<html lang="it">
<head>
  <meta charset="utf-8"/>
  <title>C/N0 stazioni Cipro — serie temporali</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    * { box-sizing: border-box; }
    body { margin: 0; font-family: system-ui, Segoe UI, sans-serif; background: #0f1117; color: #e8ecf3; }
    header { padding: 12px 16px; background: #171b26; border-bottom: 1px solid #2a3142; }
    h1 { margin: 0 0 4px; font-size: 18px; }
    .sub { color: #9aa7bd; font-size: 12px; }
    .layout { display: grid; grid-template-columns: 300px 1fr; min-height: calc(100vh - 64px); }
    aside { padding: 12px; border-right: 1px solid #2a3142; overflow-y: auto; }
    main { padding: 8px; }
    label { display: block; font-size: 11px; color: #9aa7bd; margin: 10px 0 4px; text-transform: uppercase; letter-spacing: .04em; }
    select, button { width: 100%; background: #1a1f2b; color: #e8ecf3; border: 1px solid #3a455c; border-radius: 6px; padding: 8px; }
    button { cursor: pointer; margin-top: 6px; }
    button:hover { background: #243049; }
    .chkbox-list { max-height: 160px; overflow-y: auto; background: #1a1f2b; border: 1px solid #3a455c; border-radius: 6px; padding: 6px; }
    .chkbox-list label { display: flex; align-items: center; gap: 6px; margin: 0; padding: 3px 4px; font-size: 12px; color: #d5dceb; text-transform: none; letter-spacing: 0; cursor: pointer; border-radius: 4px; }
    .chkbox-list label:hover { background: #243049; }
    .row2 { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
    #plot { width: 100%; height: calc(100vh - 90px); min-height: 480px; }
    .pill { display: inline-block; background: #243049; padding: 2px 8px; border-radius: 999px; font-size: 11px; margin-right: 6px; }
  </style>
</head>
<body>
<header>
  <h1>C/N0 — stazioni permanenti Cipro</h1>
  <div class="sub">Sovrapponi satelliti a rischio tra stazioni · fade ≥ __FADE_DB__ dB · el ≥ __MIN_EL__° · campionamento __STEP_S__ s</div>
</header>
<div class="layout">
  <aside>
    <label>Giorno</label>
    <select id="day"></select>

    <label>Grandezza</label>
    <select id="metric">
      <option value="cnr">C/N0 raw (dB-Hz)</option>
      <option value="cnr_norm" selected>C/N0 − mediana PRN (dB)</option>
    </select>

    <label>Stazioni</label>
    <div class="row2">
      <button type="button" id="staAll">Tutte</button>
      <button type="button" id="staNone">Nessuna</button>
    </div>
    <div id="stations" class="chkbox-list"></div>

    <label>Satelliti (PRN)</label>
    <div class="row2">
      <button type="button" id="prnRisk">A rischio</button>
      <button type="button" id="prnAll">Tutti</button>
    </div>
    <div id="prns" class="chkbox-list"></div>

    <label>Opzioni</label>
    <div class="chkbox-list" style="max-height:none">
      <label><input type="checkbox" id="showFade" checked/> Evidenzia campioni in fade</label>
      <label><input type="checkbox" id="connect" checked/> Collega passaggi (gap ≤ __MAX_GAP_MIN__ min)</label>
    </div>
    <button type="button" id="redraw">Aggiorna grafico</button>
  </aside>
  <main>
    <div id="plot"></div>
  </main>
</div>
<script>
const DATA = __DATA_JSON__;
const AT_RISK = new Set(DATA.at_risk_prns);
const STATION_COLORS = { GREC:'#4ea1ff', MAIA:'#ff7b72', MAXI:'#3fb950', TROD:'#d2a8ff', TERF:'#ffa657' };

function fillSelect(id, values) {
  const el = document.getElementById(id);
  el.innerHTML = values.map(v => `<option value="${v}">${v}</option>`).join('');
}

function fillChecks(containerId, values, prefix) {
  const box = document.getElementById(containerId);
  const day = document.getElementById('day').value;
  box.innerHTML = values.map(v => {
    const id = `${prefix}_${v}`;
    const risk = AT_RISK.has(v) ? ' ★' : '';
    let extra = '';
    if (prefix === 'sta' && DATA.station_days && DATA.station_days[v]) {
      const hasDay = DATA.station_days[v].includes(day);
      extra = hasDay ? '' : ' (no dati)';
      if (!hasDay) return `<label style="opacity:.45"><input type="checkbox" id="${id}" value="${v}" disabled/> ${v}${extra}</label>`;
    }
    return `<label><input type="checkbox" id="${id}" value="${v}" checked/> ${v}${risk}${extra}</label>`;
  }).join('');
}

function checked(prefix, values) {
  return values.filter(v => document.getElementById(`${prefix}_${v}`).checked);
}

function setChecks(prefix, values, on) {
  values.forEach(v => { document.getElementById(`${prefix}_${v}`).checked = on; });
}

const MAX_CONNECT_GAP_S = (DATA.meta && DATA.meta.max_connect_gap_s) || 180;

function splitByGap(utc_h, y, tow, el, az, maxGapS) {
  const x = [], yOut = [], cd = [];
  for (let i = 0; i < utc_h.length; i++) {
    if (i > 0) {
      let dt = tow[i] - tow[i - 1];
      if (dt < 0) dt += 604800;
      if (dt > maxGapS) {
        x.push(null); yOut.push(null); cd.push([null, null]);
      }
    }
    x.push(utc_h[i]);
    yOut.push(y[i]);
    cd.push([el[i], az[i]]);
  }
  return { x, y: yOut, customdata: cd };
}

function buildTraces() {
  const day = document.getElementById('day').value;
  const metric = document.getElementById('metric').value;
  const stations = checked('sta', DATA.stations);
  const prns = checked('prn', DATA.prns);
  const showFade = document.getElementById('showFade').checked;
  const connect = document.getElementById('connect').checked;
  const traces = [];

  for (const st of stations) {
    const dayBlock = DATA.series[st] && DATA.series[st][day];
    if (!dayBlock) continue;
    for (const prn of prns) {
      const s = dayBlock[prn];
      if (!s || !s.utc_h.length) continue;
      const y = s[metric];
      const name = `${st} · ${prn}`;
      const color = STATION_COLORS[st] || '#ccc';
      const dash = prn.startsWith('G') ? 'solid' : 'dot';
      let x = s.utc_h, yPlot = y, customdata = s.el.map((el, i) => [el, s.az[i]]);
      let mode = 'markers';
      if (connect && s.tow) {
        const split = splitByGap(s.utc_h, y, s.tow, s.el, s.az, MAX_CONNECT_GAP_S);
        x = split.x;
        yPlot = split.y;
        customdata = split.customdata;
        mode = 'lines+markers';
      }
      traces.push({
        x: x,
        y: yPlot,
        type: 'scatter',
        mode: mode,
        connectgaps: false,
        name,
        legendgroup: prn,
        line: { color, width: 1.6, dash },
        marker: { size: 5, color, opacity: 0.85 },
        hovertemplate: `${name}<br>UTC %{x:.2f} h<br>${metric}: %{y}<br>el %{customdata[0]:.1f}° az %{customdata[1]:.1f}°<extra></extra>`,
        customdata: customdata,
      });
      if (showFade) {
        const fx = [], fy = [];
        s.fade.forEach((f, i) => { if (f) { fx.push(s.utc_h[i]); fy.push(y[i]); } });
        if (fx.length) {
          traces.push({
            x: fx, y: fy, type: 'scatter', mode: 'markers', name: `${name} fade`,
            marker: { size: 9, color: '#ff4d4f', symbol: 'x', line: { width: 1, color: '#fff' } },
            showlegend: false, hoverinfo: 'skip',
          });
        }
      }
    }
  }
  return traces;
}

function redraw() {
  const traces = buildTraces();
  const metric = document.getElementById('metric').value;
  const ylab = metric === 'cnr' ? 'C/N0 (dB-Hz)' : 'C/N0 − mediana PRN (dB)';
  Plotly.react('plot', traces, {
    title: { text: 'Serie temporali C/N0', font: { color: '#e8ecf3', size: 14 } },
    paper_bgcolor: '#0f1117', plot_bgcolor: '#141824',
    font: { color: '#c9d4e8' },
    xaxis: { title: 'Ora UTC (h)', gridcolor: '#2a3142', zerolinecolor: '#2a3142', range: [0, 24] },
    yaxis: { title: ylab, gridcolor: '#2a3142', zerolinecolor: '#556' },
    legend: { orientation: 'h', y: 1.08, font: { size: 10 } },
    margin: { t: 48, r: 16, b: 48, l: 56 },
    hovermode: 'closest',
  }, { responsive: true, displayModeBar: true });
}

fillSelect('day', DATA.days);
fillChecks('stations', DATA.stations, 'sta');
fillChecks('prns', DATA.prns, 'prn');
document.getElementById('staAll').onclick = () => setChecks('sta', DATA.stations, true);
document.getElementById('staNone').onclick = () => setChecks('sta', DATA.stations, false);
document.getElementById('prnRisk').onclick = () => {
  DATA.prns.forEach(v => setChecks('prn', [v], AT_RISK.has(v)));
};
document.getElementById('prnAll').onclick = () => setChecks('prn', DATA.prns, true);
document.getElementById('redraw').onclick = redraw;
function onDayChange() { fillChecks('stations', DATA.stations, 'sta'); redraw(); }
document.getElementById('day').addEventListener('change', onDayChange);
['metric','showFade','connect'].forEach(id => document.getElementById(id).addEventListener('change', redraw));
document.getElementById('stations').addEventListener('change', redraw);
document.getElementById('prns').addEventListener('change', redraw);
redraw();
</script>
</body>
</html>
"""


def main() -> None:
    p = argparse.ArgumentParser(description="Build C/N0 time-series HTML viewer.")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--crd-path", type=Path, required=True)
    p.add_argument("--terf-reference-csv", type=Path, default=Path("experiments/data/TERF/reference_2026192.csv"))
    p.add_argument("--terf-extra-dir", type=Path, default=Path("experiments/data/TERF"),
                   help="Extra TERF OBS days (191-193); 194 from --data-dir")
    p.add_argument("--results-dir", type=Path, default=Path("experiments/results/gnss_stations_sat_risk"))
    p.add_argument("--output", type=Path, default=Path("experiments/results/gnss_stations_sat_risk/cnr_timeseries_viewer.html"))
    p.add_argument("--min-el-deg", type=float, default=20.0)
    p.add_argument("--cnr-fade-db", type=float, default=6.0)
    p.add_argument("--epoch-step", type=int, default=1, help="OBS epoch stride (1=30s, 2=60s)")
    p.add_argument("--max-prns", type=int, default=6, help="Max sticky PRNs to include")
    p.add_argument("--prns", type=str, default="", help="Explicit PRN list (overrides --max-prns)")
    args = p.parse_args()

    coords = sat._load_crd(args.crd_path)
    terf_rx = risk._load_rx(args.terf_reference_csv) if args.terf_reference_csv.is_file() else None
    nav_paths = [Path(x) for x in sorted(glob(str(args.data_dir / "BRDC*.rnx")))]
    groups = _merge_terf_obs(sat._group_obs(args.data_dir), args.data_dir, args.terf_extra_dir)

    if args.prns.strip():
        prn_filter = {x.strip().upper() for x in args.prns.split(",") if x.strip()}
    else:
        prn_filter = set(_load_at_risk_prns(args.results_dir, max_prns=int(args.max_prns)))
    at_risk = sorted(prn_filter)

    print(f"PRN: {', '.join(at_risk)}")

    series: dict[str, dict[str, dict[str, dict]]] = {}
    days: set[str] = set()
    eph_cache: dict[Path, Ephemeris] = {}
    t0 = time.perf_counter()

    for station in sorted(groups):
        obs_paths = groups[station]
        if station in coords:
            rx = coords[station]
        elif station == "TERF" and terf_rx is not None:
            rx = terf_rx
        else:
            from analyze_gnss_stations_cnr_batch import _rx_from_obs_header

            rx = _rx_from_obs_header(obs_paths[0])
        series[station] = {}
        for obs_path in obs_paths:
            nav = risk._match_nav(obs_path, nav_paths)
            if nav is None:
                continue
            if nav not in eph_cache:
                eph_cache[nav] = Ephemeris(read_nav_rinex_multi(nav))
            tag = risk._tag_from_obs(obs_path)
            days.add(tag)
            day_data = _collect_day_series(
                obs_path,
                eph_cache[nav],
                rx,
                prn_filter=prn_filter,
                min_el_deg=float(args.min_el_deg),
                fade_db=float(args.cnr_fade_db),
                epoch_step=int(args.epoch_step),
            )
            series[station][tag] = day_data
            n_pts = sum(len(v["tow"]) for v in day_data.values())
            print(f"  {station} {tag}: {len(day_data)} PRN, {n_pts} punti")

    step_s = 30 * max(1, int(args.epoch_step))
    max_connect_gap_s = max(120, 4 * step_s)
    payload = {
        "stations": sorted(series),
        "days": sorted(days),
        "station_days": {st: sorted(series[st].keys()) for st in series},
        "prns": sorted(prn_filter),
        "at_risk_prns": sorted(at_risk),
        "series": series,
        "meta": {
            "fade_db": float(args.cnr_fade_db),
            "min_el_deg": float(args.min_el_deg),
            "epoch_step": int(args.epoch_step),
            "step_s": step_s,
            "max_connect_gap_s": max_connect_gap_s,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    html = (
        _HTML.replace("__DATA_JSON__", json.dumps(payload, separators=(",", ":")))
        .replace("__FADE_DB__", str(int(args.cnr_fade_db)))
        .replace("__MIN_EL__", str(int(args.min_el_deg)))
        .replace("__STEP_S__", str(step_s))
        .replace("__MAX_GAP_MIN__", str(round(max_connect_gap_s / 60.0, 1)))
    )
    args.output.write_text(html, encoding="utf-8")
    print(f"\nDone in {time.perf_counter() - t0:.1f}s -> {args.output}")
    print(f"Apri nel browser: file:///{args.output.resolve().as_posix()}")


if __name__ == "__main__":
    main()
