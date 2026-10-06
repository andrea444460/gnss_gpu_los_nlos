#!/usr/bin/env python3
"""C/N0 elevation residuals + FFT per-satellite (TERF wind-turbine fading).

1. Build elevation model (median C/N0 vs elevation, all PRNs pooled).
2. Residual = measured - model(el) -> obstructions as negative spikes.
3. FFT on residual time series per PRN (per day); report dominant frequencies.

Note: TERF OBS is 30 s -> Nyquist ~0.0167 Hz (period ~60 s). Faster blade-pass
rates alias; interpret sub-Nyquist peaks as slow fading only.

Example::

    python experiments/analyze_terf_cnr_residual_fft.py \\
        --obs-glob experiments/data/TERF/TERF00CYP_R_*_MO.rnx \\
        --nav-glob experiments/data/TERF/BRDC*.rnx \\
        --reference-csv experiments/data/TERF/reference_2026192.csv \\
        --output-dir experiments/results/terf_cnr_spectrum
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_ROOT / "python"))

from gnss_gpu.ephemeris import Ephemeris
from gnss_gpu.io.nav_rinex import read_nav_rinex_multi
from gnss_gpu.io.rinex import read_rinex_obs
from gnss_gpu.urban_signal_sim import _sat_elevation_azimuth

GPS_EPOCH = datetime(1980, 1, 6)
_CNR_PREF = ("S1C", "S1X", "S1P", "S1W", "S2W", "S2C", "S2X", "S5Q", "S5X", "S7Q", "S7X", "S8Q", "S8X")


def _gps_tow(dt: datetime) -> float:
    return (dt - GPS_EPOCH).total_seconds() % 604800.0


def _normalize_prn(sat_id: str) -> str:
    s = str(sat_id).strip().upper()
    if len(s) >= 3 and s[0] in "GREJCI" and s[1:].isdigit():
        return f"{s[0]}{int(s[1:]):02d}"
    return s


def _preferred_cnr(obs_dict: dict) -> float | None:
    upper = {str(k).upper(): k for k in obs_dict}
    for code in _CNR_PREF:
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


def _load_rx(reference_csv: Path) -> np.ndarray:
    with open(reference_csv, newline="", encoding="utf-8") as fh:
        row = next(csv.DictReader(fh))
    return np.array(
        [float(row["ECEF X (m)"]), float(row["ECEF Y (m)"]), float(row["ECEF Z (m)"])],
        dtype=np.float64,
    )


def _tag_from_obs(name: str) -> str:
    m = re.search(r"_(\d{7})\d+_01D_", name)
    return m.group(1) if m else name


def _match_nav(obs_path: Path, nav_paths: list[Path]) -> Path | None:
    tag = _tag_from_obs(obs_path.name)
    for p in nav_paths:
        if tag in p.name:
            return p
    return None


def _collect_samples(
    obs_path: Path,
    eph: Ephemeris,
    rx: np.ndarray,
    *,
    min_el: float,
    max_el: float,
    epoch_step: int,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], dict[str, int]]:
    """Return tag_idx, prn codes (int), tow, az, el, cnr as numpy arrays."""
    obs = read_rinex_obs(obs_path)
    prn_to_code: dict[str, int] = {}
    chunks_tag: list[np.ndarray] = []
    chunks_prn: list[np.ndarray] = []
    chunks_tow: list[np.ndarray] = []
    chunks_az: list[np.ndarray] = []
    chunks_el: list[np.ndarray] = []
    chunks_cnr: list[np.ndarray] = []

    step = max(1, int(epoch_step))
    for ei, ep in enumerate(obs.epochs):
        if step > 1 and (ei % step) != 0:
            continue
        tow = float(_gps_tow(ep.time))
        prns: list[str] = []
        cnrs: list[float] = []
        for sat_id, odict in ep.observations.items():
            cnr = _preferred_cnr(odict)
            if cnr is None:
                continue
            prn = _normalize_prn(str(sat_id))
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
        ok = (el_deg >= min_el) & (el_deg <= max_el)
        if not np.any(ok):
            continue
        ok_idx = np.where(ok)[0]
        n_ok = int(ok_idx.size)
        tow_chunk = np.full(n_ok, tow, dtype=np.float64)
        az_chunk = az_deg[ok].astype(np.float64)
        el_chunk = el_deg[ok].astype(np.float64)
        cnr_chunk = np.asarray([sel_cnr[int(k)] for k in ok_idx], dtype=np.float64)
        prn_chunk = np.empty(n_ok, dtype=np.int16)
        for i, k in enumerate(ok_idx):
            prn = sel_prn[int(k)]
            if prn not in prn_to_code:
                prn_to_code[prn] = len(prn_to_code)
            prn_chunk[i] = prn_to_code[prn]
        chunks_tag.append(np.zeros(n_ok, dtype=np.int16))
        chunks_prn.append(prn_chunk)
        chunks_tow.append(tow_chunk)
        chunks_az.append(az_chunk)
        chunks_el.append(el_chunk)
        chunks_cnr.append(cnr_chunk)

    if not chunks_tow:
        empty = np.empty(0, dtype=np.float64)
        return (empty, empty, empty, empty, empty, empty), {}

    return (
        np.concatenate(chunks_tag),
        np.concatenate(chunks_prn),
        np.concatenate(chunks_tow),
        np.concatenate(chunks_az),
        np.concatenate(chunks_el),
        np.concatenate(chunks_cnr),
    ), prn_to_code


def _build_el_model(el: np.ndarray, cnr: np.ndarray, *, el_bin_deg: float, min_count: int) -> tuple[np.ndarray, np.ndarray]:
    bins = np.floor(el / el_bin_deg).astype(np.int32)
    uniq = np.unique(bins)
    centers = []
    medians = []
    for b in uniq:
        m = bins == b
        if int(np.count_nonzero(m)) < min_count:
            continue
        centers.append((b + 0.5) * el_bin_deg)
        medians.append(float(np.median(cnr[m])))
    if not centers:
        return np.array([0.0]), np.array([float(np.median(cnr))])
    order = np.argsort(centers)
    return np.asarray(centers, dtype=np.float64)[order], np.asarray(medians, dtype=np.float64)[order]


def _interp_model(el: np.ndarray, centers: np.ndarray, medians: np.ndarray, fallback: float) -> np.ndarray:
    if centers.size == 0:
        return np.full(el.shape, fallback, dtype=np.float64)
    idx = np.searchsorted(centers, el, side="right")
    idx = np.clip(idx, 0, centers.size - 1)
    out = medians[idx].astype(np.float64)
    below = el < centers[0]
    out[below] = medians[0]
    return out


def _split_passes(tow: np.ndarray, az: np.ndarray, max_gap_s: float) -> list[slice]:
    n = tow.size
    if n == 0:
        return []
    breaks = [0]
    for i in range(1, n):
        dt = float(tow[i] - tow[i - 1])
        if dt < -302400.0:
            dt += 604800.0
        elif dt > 302400.0:
            dt -= 604800.0
        daz = abs(float(az[i] - az[i - 1]))
        if daz > 180.0:
            daz = 360.0 - daz
        if dt > max_gap_s or daz > 45.0:
            breaks.append(i)
    breaks.append(n)
    return [slice(breaks[k], breaks[k + 1]) for k in range(len(breaks) - 1) if breaks[k + 1] - breaks[k] >= 8]


def _n_az_equal_area(el_deg: float, n_az_base: int) -> int:
    """Azimuth bins at elevation el so cell solid angle ~ constant."""
    return max(1, int(round(n_az_base * math.cos(math.radians(el_deg)))))


def _binned_median_grid_equal_area(
    az: np.ndarray,
    el: np.ndarray,
    values: np.ndarray,
    *,
    az_bin_deg: float,
    el_bin_deg: float,
    min_el: float,
    max_el: float,
    min_count: int,
) -> tuple[list[np.ndarray], list[int], np.ndarray]:
    """Equal-area polar bins: N_az(el) ~ N_az_base * cos(el)."""
    n_az_base = max(1, int(round(360.0 / az_bin_deg)))
    nel = max(1, int(round((max_el - min_el) / el_bin_deg)))
    el_centers = min_el + (np.arange(nel, dtype=np.float64) + 0.5) * el_bin_deg
    n_az_per_row = [_n_az_equal_area(float(e), n_az_base) for e in el_centers]

    cells: dict[tuple[int, int], float] = {}
    el_i = np.floor((el - min_el) / el_bin_deg).astype(np.int32)
    ok = (el_i >= 0) & (el_i < nel)
    az_ok = az[ok]
    el_ok = el_i[ok]
    val_ok = values[ok]
    n_az_lut = np.asarray(n_az_per_row, dtype=np.int32)
    for ei in range(nel):
        m = el_ok == ei
        if not np.any(m):
            continue
        n_az = int(n_az_lut[ei])
        az_step = 360.0 / n_az
        az_shift = 0.5 * az_step  # keep cardinal dirs off bin edges (north seam)
        ai = (((az_ok[m] + az_shift) % 360.0) // az_step).astype(np.int32) % n_az
        vv = val_ok[m]
        order = np.argsort(ai, kind="stable")
        ai_s = ai[order]
        vv_s = vv[order]
        split = np.flatnonzero(np.diff(ai_s)) + 1
        starts = np.concatenate(([0], split))
        ends = np.concatenate((split, [ai_s.size]))
        for s, e in zip(starts, ends):
            if e - s < min_count:
                continue
            cells[(ei, int(ai_s[s]))] = float(np.median(vv_s[s:e]))

    rows: list[np.ndarray] = []
    for ei in range(nel):
        n_az = n_az_per_row[ei]
        row = np.full(n_az, np.nan, dtype=np.float64)
        for ai in range(n_az):
            med = cells.get((ei, ai))
            if med is not None:
                row[ai] = med
        rows.append(row)
    r_edges = np.linspace(90.0 - max_el, 90.0 - min_el, nel + 1)
    return rows, n_az_per_row, r_edges


def _plot_binned_mean_skyplot(
    az: np.ndarray,
    el: np.ndarray,
    values: np.ndarray,
    out_path: Path,
    title: str,
    *,
    az_bin_deg: float,
    el_bin_deg: float,
    min_el: float,
    max_el: float,
    min_count: int,
    vmin: float,
    vmax: float,
    draw_empty: bool = False,
) -> None:
    rows, n_az_per_row, r_edges = _binned_median_grid_equal_area(
        az,
        el,
        values,
        az_bin_deg=az_bin_deg,
        el_bin_deg=el_bin_deg,
        min_el=min_el,
        max_el=max_el,
        min_count=min_count,
    )
    nel = len(rows)
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "polar"})
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("coolwarm")
    no_data = (0.92, 0.92, 0.92, 1.0)

    for ei in range(nel):
        row = rows[ei]
        n_az = n_az_per_row[ei]
        r_lo = float(r_edges[ei])
        r_hi = float(r_edges[ei + 1])
        height = r_hi - r_lo
        width = 2.0 * np.pi / n_az
        az_shift = 0.5 * (360.0 / n_az)
        az_centers_deg = ((np.arange(n_az, dtype=np.float64) + 0.5) * (360.0 / n_az) + az_shift) % 360.0
        thetas = np.radians(az_centers_deg)
        for ai in range(n_az):
            v = row[ai]
            if not np.isfinite(v):
                if not draw_empty:
                    continue
                color = no_data
            else:
                color = cmap(norm(float(v)))
            ax.bar(
                thetas[ai],
                height,
                width=width,
                bottom=r_lo,
                color=color,
                align="center",
                linewidth=0,
                edgecolor="none",
            )

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim(90.0 - max_el, 90.0 - min_el)
    ax.set_yticks([90.0 - e for e in (75, 60, 45, 30, 20) if min_el <= e <= max_el])
    ax.set_yticklabels([f"{int(e)}" for e in (75, 60, 45, 30, 20) if min_el <= e <= max_el])
    ax.set_title(title, va="bottom", fontsize=10)
    plt.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.1, label="Median C/N0 residual [dB-Hz]")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _plot_residual_skyplot(
    mask: np.ndarray,
    az: np.ndarray,
    el: np.ndarray,
    tow: np.ndarray,
    residual: np.ndarray,
    prn_idx: np.ndarray,
    out_path: Path,
    title: str,
    *,
    vmin: float,
    vmax: float,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "polar"})
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("coolwarm")
    for code in np.unique(prn_idx[mask]):
        m = mask & (prn_idx == code)
        order = np.argsort(tow[m])
        t = tow[m][order]
        a = az[m][order]
        e = el[m][order]
        r = residual[m][order]
        for sl in _split_passes(t, a, max_gap_s=120.0):
            th = np.radians(a[sl])
            rr = 90.0 - e[sl]
            rv = r[sl]
            ax.scatter(th, rr, c=rv, cmap="coolwarm", norm=norm, s=10, linewidths=0, alpha=0.85)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 90)
    ax.set_title(title, va="bottom", fontsize=10)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    plt.colorbar(sm, ax=ax, pad=0.1, label="C/N0 residual [dB-Hz]")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _fft_peaks(
    values: np.ndarray,
    *,
    dt_nominal: float,
    min_period_s: float,
    max_period_s: float,
    top_k: int,
) -> list[dict]:
    n = values.size
    if n < 16:
        return []
    dt = float(dt_nominal)
    t = np.arange(n, dtype=np.float64) * dt
    y = np.asarray(values, dtype=np.float64)
    y = y - np.polyval(np.polyfit(t, y, 1), t)
    y = y - np.mean(y)
    yw = y * np.hanning(n)
    spec = np.fft.rfft(yw)
    freqs = np.fft.rfftfreq(n, d=dt)
    power = (np.abs(spec) ** 2) / max(n, 1)
    if freqs.size <= 1:
        return []
    mask = (freqs > 0) & (freqs <= 1.0 / max(min_period_s, dt))
    if max_period_s > 0:
        mask &= freqs >= 1.0 / max_period_s
    idx = np.where(mask)[0]
    if idx.size == 0:
        return []
    order = idx[np.argsort(power[idx])[::-1]]
    peaks = []
    for j in order[:top_k]:
        f = float(freqs[j])
        peaks.append({"freq_hz": f, "period_s": 1.0 / f if f > 0 else None, "power": float(power[j])})
    return peaks


def _analyze_fft(
    tag: np.ndarray,
    prn_idx: np.ndarray,
    tow: np.ndarray,
    az: np.ndarray,
    residual: np.ndarray,
    code_to_prn: list[str],
    tag_names: list[str],
    *,
    dt_s: float,
    min_period_s: float,
    max_period_s: float,
    top_k: int,
    az_focus_lo: float | None,
    az_focus_hi: float | None,
) -> tuple[list[dict], dict[float, list[float]]]:
    keys = tag.astype(np.int64) * 10000 + prn_idx.astype(np.int64)
    order = np.argsort(keys, kind="stable")
    keys_s = keys[order]
    spectrum_rows: list[dict] = []
    freq_pool: dict[float, list[float]] = defaultdict(list)

    if keys_s.size == 0:
        return spectrum_rows, freq_pool

    split = np.flatnonzero(np.diff(keys_s)) + 1
    starts = np.concatenate(([0], split))
    ends = np.concatenate((split, [keys_s.size]))

    for s, e in zip(starts, ends):
        if e - s < 16:
            continue
        t = tow[order][s:e]
        a = az[order][s:e]
        y = residual[order][s:e]
        ti = int(tag[order][s])
        pi = int(prn_idx[order][s])
        if az_focus_lo is not None and az_focus_hi is not None:
            if az_focus_lo <= az_focus_hi:
                m = (a >= az_focus_lo) & (a <= az_focus_hi)
            else:
                m = (a >= az_focus_lo) | (a <= az_focus_hi)
            if int(np.count_nonzero(m)) >= 16:
                t, y = t[m], y[m]
        peaks = _fft_peaks(
            y,
            dt_nominal=dt_s,
            min_period_s=min_period_s,
            max_period_s=max_period_s,
            top_k=top_k,
        )
        for rank, pk in enumerate(peaks, start=1):
            spectrum_rows.append(
                {
                    "tag": tag_names[ti],
                    "prn": code_to_prn[pi],
                    "rank": rank,
                    "freq_hz": pk["freq_hz"],
                    "period_s": pk["period_s"],
                    "power": pk["power"],
                }
            )
            freq_pool[round(pk["freq_hz"], 5)].append(pk["power"])

    return spectrum_rows, freq_pool


def _plot_el_model(centers: np.ndarray, medians: np.ndarray, out_path: Path) -> None:
    if centers.size == 0:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(centers, medians, "o-", color="#333")
    ax.set_xlabel("Elevation [deg]")
    ax.set_ylabel("Median C/N0 [dB-Hz]")
    ax.set_title("Elevation model (pooled all PRNs/days)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_spectrum_examples(
    tag: np.ndarray,
    prn_idx: np.ndarray,
    tow: np.ndarray,
    residual: np.ndarray,
    code_to_prn: list[str],
    tag_names: list[str],
    out_dir: Path,
    *,
    dt_s: float,
    n_plots: int,
) -> None:
    keys = tag.astype(np.int64) * 10000 + prn_idx.astype(np.int64)
    order = np.argsort(keys, kind="stable")
    keys_s = keys[order]
    split = np.flatnonzero(np.diff(keys_s)) + 1
    starts = np.concatenate(([0], split))
    ends = np.concatenate((split, [keys_s.size]))
    scored: list[tuple[float, int, int]] = []
    for s, e in zip(starts, ends):
        y = residual[order][s:e]
        scored.append((float(np.std(y)), s, e))
    scored.sort(reverse=True)
    for _, s, e in scored[:n_plots]:
        if e - s < 32:
            continue
        y = residual[order][s:e]
        y = y - np.mean(y)
        n = y.size
        spec = np.abs(np.fft.rfft(y * np.hanning(n))) ** 2 / n
        freqs = np.fft.rfftfreq(n, d=dt_s)
        ti = int(tag[order][s])
        pi = int(prn_idx[order][s])
        prn = code_to_prn[pi]
        tname = tag_names[ti]
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.semilogy(freqs[1:], spec[1:], color="#1565c0")
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("PSD")
        ax.set_title(f"C/N0 residual spectrum - {prn} day {tname} (dt={dt_s}s, Nyquist={0.5/dt_s:.4f} Hz)")
        ax.axvline(0.5 / dt_s, color="red", ls="--", alpha=0.5, label="Nyquist")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"spectrum_{prn}_{tname}.png", dpi=120)
        plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="C/N0 elevation residual + FFT (TERF).")
    p.add_argument("--obs-glob", type=str, required=True)
    p.add_argument("--nav-glob", type=str, required=True)
    p.add_argument("--reference-csv", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("experiments/results/terf_cnr_spectrum"))
    p.add_argument("--min-el-deg", type=float, default=15.0)
    p.add_argument("--max-el-deg", type=float, default=85.0)
    p.add_argument("--el-bin-deg", type=float, default=2.0)
    p.add_argument("--epoch-step", type=int, default=2, help="Use every Nth epoch (2=60s, 1=30s)")
    p.add_argument("--obs-interval-s", type=float, default=30.0, help="Nominal OBS interval before epoch-step")
    p.add_argument("--min-period-s", type=float, default=60.0, help="Min period for FFT peaks")
    p.add_argument("--max-period-s", type=float, default=7200.0, help="Max period (0=disable)")
    p.add_argument("--az-focus-lo", type=float, default=190.0, help="Optional turbine az window")
    p.add_argument("--az-focus-hi", type=float, default=220.0)
    p.add_argument("--residual-clip-db", type=float, default=8.0, help="Skyplot color scale +/- dB")
    p.add_argument("--az-bin-deg", type=float, default=5.0, help="Az step at horizon for equal-area bins")
    p.add_argument("--sky-az-bin-deg", type=float, default=10.0, help="Az step @ horizon for combined skyplot")
    p.add_argument("--sky-el-bin-deg", type=float, default=5.0, help="El step for combined skyplot")
    p.add_argument("--sky-max-el-deg", type=float, default=72.0, help="Max el for combined skyplot (0=use --max-el-deg)")
    p.add_argument("--sky-bin-min-count", type=int, default=5, help="Min samples per az/el cell")
    p.add_argument("--spectrum-plots", type=int, default=6, help="N example PSD plots")
    p.add_argument("--no-skyplot", action="store_true", help="Skip residual skyplot PNGs")
    p.add_argument("--no-spectrum-plots", action="store_true", help="Skip example PSD PNGs")
    args = p.parse_args()

    t0 = time.perf_counter()
    rx = _load_rx(args.reference_csv)
    obs_paths = [Path(x) for x in sorted(glob(args.obs_glob))]
    nav_paths = [Path(x) for x in sorted(glob(args.nav_glob))]
    if not obs_paths:
        raise SystemExit(f"No OBS matched {args.obs_glob}")

    eph_cache: dict[Path, Ephemeris] = {}
    global_prn: dict[str, int] = {}
    tag_names: list[str] = []
    all_tag: list[np.ndarray] = []
    all_prn: list[np.ndarray] = []
    all_tow: list[np.ndarray] = []
    all_az: list[np.ndarray] = []
    all_el: list[np.ndarray] = []
    all_cnr: list[np.ndarray] = []

    for obs_path in obs_paths:
        t_file = time.perf_counter()
        nav = _match_nav(obs_path, nav_paths)
        if nav is None:
            print(f"skip {obs_path.name}: no NAV")
            continue
        if nav not in eph_cache:
            eph_cache[nav] = Ephemeris(read_nav_rinex_multi(nav))
        eph = eph_cache[nav]
        arrays, local_prn = _collect_samples(
            obs_path,
            eph,
            rx,
            min_el=float(args.min_el_deg),
            max_el=float(args.max_el_deg),
            epoch_step=int(args.epoch_step),
        )
        if arrays[2].size == 0:
            print(f"  {obs_path.name}: 0 samples")
            continue
        tag = _tag_from_obs(obs_path.name)
        tag_id = len(tag_names)
        tag_names.append(tag)
        _, prn_idx, tow, az, el, cnr = arrays
        prn_codes = np.empty(prn_idx.size, dtype=np.int16)
        for prn, code in local_prn.items():
            if prn not in global_prn:
                global_prn[prn] = len(global_prn)
            prn_codes[prn_idx == code] = global_prn[prn]
        all_tag.append(np.full(tow.size, tag_id, dtype=np.int16))
        all_prn.append(prn_codes.astype(np.int16))
        all_tow.append(tow)
        all_az.append(az)
        all_el.append(el)
        all_cnr.append(cnr)
        dt_file = time.perf_counter() - t_file
        print(f"  {obs_path.name}: {tow.size} samples ({dt_file:.1f}s)", flush=True)

    if not all_tow:
        raise SystemExit("No samples.")

    tag_arr = np.concatenate(all_tag)
    prn_arr = np.concatenate(all_prn)
    tow_arr = np.concatenate(all_tow)
    az_arr = np.concatenate(all_az)
    el_arr = np.concatenate(all_el)
    cnr_arr = np.concatenate(all_cnr)
    code_to_prn = [""] * len(global_prn)
    for prn, code in global_prn.items():
        code_to_prn[code] = prn

    t_collect = time.perf_counter() - t0
    print(f"Collected {tow_arr.size} samples in {t_collect:.1f}s", flush=True)

    centers, medians = _build_el_model(el_arr, cnr_arr, el_bin_deg=float(args.el_bin_deg), min_count=50)
    fallback = float(np.median(cnr_arr))
    model_cnr = _interp_model(el_arr, centers, medians, fallback)
    residual = cnr_arr - model_cnr

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _plot_el_model(centers, medians, args.output_dir / "cnr_elevation_model.png")

    clip = float(args.residual_clip_db)
    if not args.no_skyplot:
        t_plot = time.perf_counter()
        for ti, tname in enumerate(tag_names):
            mask = tag_arr == ti
            _plot_residual_skyplot(
                mask,
                az_arr,
                el_arr,
                tow_arr,
                residual,
                prn_arr,
                args.output_dir / f"cnr_residual_skyplot_{tname}.png",
                f"C/N0 residual (meas - el model) - {tname}",
                vmin=-clip,
                vmax=clip,
            )
            print(f"  skyplot {tname}", flush=True)
        print(f"Skyplots in {time.perf_counter() - t_plot:.1f}s")

    n_days = len(tag_names)
    sky_max_el = float(args.max_el_deg) if float(args.sky_max_el_deg) <= 0 else float(args.sky_max_el_deg)
    sky_mask = el_arr <= sky_max_el
    _plot_binned_mean_skyplot(
        az_arr[sky_mask],
        el_arr[sky_mask],
        residual[sky_mask],
        args.output_dir / "cnr_residual_skyplot_combined.png",
        f"C/N0 residual median — all {n_days} days pooled\n"
        f"equal-area: {args.sky_az_bin_deg:g} deg az @ horizon, {args.sky_el_bin_deg:g} deg el, "
        f"el {args.min_el_deg:g}-{sky_max_el:g} deg, min {args.sky_bin_min_count} samples | "
        f"white = no satellite coverage (GPS hole verso N)",
        az_bin_deg=float(args.sky_az_bin_deg),
        el_bin_deg=float(args.sky_el_bin_deg),
        min_el=float(args.min_el_deg),
        max_el=sky_max_el,
        min_count=int(args.sky_bin_min_count),
        vmin=-clip,
        vmax=clip,
    )
    print(f"Combined skyplot -> {args.output_dir / 'cnr_residual_skyplot_combined.png'}", flush=True)

    step = max(1, int(args.epoch_step))
    dt = float(args.obs_interval_s) * step
    nyquist = 0.5 / dt
    spectrum_rows, freq_pool = _analyze_fft(
        tag_arr,
        prn_arr,
        tow_arr,
        az_arr,
        residual,
        code_to_prn,
        tag_names,
        dt_s=dt,
        min_period_s=float(args.min_period_s),
        max_period_s=float(args.max_period_s),
        top_k=5,
        az_focus_lo=float(args.az_focus_lo),
        az_focus_hi=float(args.az_focus_hi),
    )

    spec_csv = args.output_dir / "cnr_fft_peaks_by_prn.csv"
    with open(spec_csv, "w", newline="", encoding="utf-8") as fh:
        if spectrum_rows:
            w = csv.DictWriter(fh, fieldnames=list(spectrum_rows[0].keys()))
            w.writeheader()
            w.writerows(spectrum_rows)

    common = []
    for f, powers in sorted(freq_pool.items(), key=lambda kv: -len(kv[1])):
        common.append(
            {
                "freq_hz": f,
                "period_s": 1.0 / f if f > 0 else None,
                "n_prns": len(powers),
                "median_power": float(np.median(powers)),
            }
        )
    common.sort(key=lambda r: (-r["n_prns"], -r["median_power"]))

    if not args.no_spectrum_plots:
        _plot_spectrum_examples(
            tag_arr,
            prn_arr,
            tow_arr,
            residual,
            code_to_prn,
            tag_names,
            args.output_dir,
            dt_s=dt,
            n_plots=int(args.spectrum_plots),
        )

    spike_rows = []
    for code, prn in enumerate(code_to_prn):
        m = prn_arr == code
        v = residual[m]
        if v.size == 0:
            continue
        spike_rows.append(
            {
                "prn": prn,
                "mean_residual_db": float(np.mean(v)),
                "std_residual_db": float(np.std(v)),
                "min_residual_db": float(np.min(v)),
                "frac_below_minus3db": float(np.mean(v < -3.0)),
                "n_spike_samples": int(np.count_nonzero(v < -3.0)),
            }
        )
    spike_rows.sort(key=lambda r: (-r["frac_below_minus3db"], r["min_residual_db"]))
    spike_csv = args.output_dir / "cnr_residual_spike_ranking.csv"
    with open(spike_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(spike_rows[0].keys()))
        w.writeheader()
        w.writerows(spike_rows)

    report = {
        "n_samples": int(tow_arr.size),
        "epoch_step": step,
        "effective_dt_s": dt,
        "nyquist_hz": nyquist,
        "note": "Blade pass frequencies above Nyquist alias at 30 s sampling.",
        "top_spike_prns": spike_rows[:12],
        "common_fft_peaks": common[:15],
        "outputs": {
            "elevation_model_png": str(args.output_dir / "cnr_elevation_model.png"),
            "residual_skyplots": str(args.output_dir / "cnr_residual_skyplot_*.png"),
            "combined_skyplot": str(args.output_dir / "cnr_residual_skyplot_combined.png"),
            "fft_peaks_csv": str(spec_csv),
            "spike_ranking_csv": str(spike_csv),
        },
    }
    report_path = args.output_dir / "cnr_residual_fft_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nTotal {time.perf_counter() - t0:.1f}s | Nyquist={nyquist:.4f} Hz (period>{1/nyquist:.0f}s)")
    print("\nTop PRNs by negative residual fraction (< -3 dB vs el model):")
    for r in spike_rows[:8]:
        print(
            f"  {r['prn']:4s}  frac<-3dB={r['frac_below_minus3db']*100:5.1f}%  "
            f"min={r['min_residual_db']:5.1f} dB  std={r['std_residual_db']:.2f}"
        )
    print("\nMost repeated FFT peak frequencies (all PRNs):")
    for c in common[:8]:
        print(
            f"  f={c['freq_hz']:.5f} Hz  T={c['period_s']:.0f}s  "
            f"PRNs={c['n_prns']}  median_power={c['median_power']:.2e}"
        )
    print(f"\nWrote {args.output_dir}")


if __name__ == "__main__":
    main()
