#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


GPS_EPOCH_UNIX_S = 315964800  # 1980-01-06T00:00:00Z
GPS_WEEK_S = 7 * 24 * 3600
GPST_MINUS_UTC_S = 18.0


def _gpsweek_tow_to_unix_utc_s(gps_week: np.ndarray, gps_tow_s: np.ndarray) -> np.ndarray:
    gps_week = np.asarray(gps_week, dtype=np.int64)
    gps_tow_s = np.asarray(gps_tow_s, dtype=np.float64)
    # GPS time is ahead of UTC by ~18s for the KLT years; we convert to UTC unix seconds.
    return GPS_EPOCH_UNIX_S + gps_week.astype(np.float64) * GPS_WEEK_S + gps_tow_s - GPST_MINUS_UTC_S


def _norm_sat_id(x: str) -> str:
    x = str(x).strip()
    if not x:
        return x
    # expected already like "G04", "C10"
    if len(x) == 2 and x[0].isalpha() and x[1].isdigit():
        return f"{x[0].upper()}0{x[1]}"
    if len(x) == 3 and x[0].isalpha():
        return x[0].upper() + x[1:]
    return x.upper()


def _is_gc(sat_id: str) -> bool:
    return str(sat_id).startswith(("G", "C"))


@dataclass(frozen=True)
class Metrics:
    n: int
    tp: int
    tn: int
    fp: int
    fn: int

    @property
    def acc(self) -> float:
        return (self.tp + self.tn) / self.n if self.n else float("nan")

    @property
    def prec(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else float("nan")

    @property
    def rec(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else float("nan")

    @property
    def f1(self) -> float:
        p = self.prec
        r = self.rec
        return 2 * p * r / (p + r) if (p == p and r == r and (p + r) > 0) else float("nan")


def _confusion(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_true.size == 0:
        return Metrics(n=0, tp=0, tn=0, fp=0, fn=0)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return Metrics(n=int(y_true.size), tp=tp, tn=tn, fp=fp, fn=fn)


def _load_geom_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"gps_week", "gps_tow", "sat_id", "is_los"}
    missing = need - set(df.columns)
    if missing:
        raise RuntimeError(f"{path}: missing columns {sorted(missing)}")
    df = df[["gps_week", "gps_tow", "sat_id", "is_los"]].copy()
    df["sat_id"] = df["sat_id"].map(_norm_sat_id)
    df = df[df["sat_id"].map(_is_gc)]
    df["is_los"] = df["is_los"].astype(int)
    df["t_utc_s"] = _gpsweek_tow_to_unix_utc_s(df["gps_week"].to_numpy(), df["gps_tow"].to_numpy())
    return df


def _load_gt_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"obs_time_utc", "sat_id", "in_label_set", "is_los"}
    missing = need - set(df.columns)
    if missing:
        raise RuntimeError(f"{path}: missing columns {sorted(missing)}")
    df = df[["obs_time_utc", "sat_id", "in_label_set", "is_los"]].copy()
    df["sat_id"] = df["sat_id"].map(_norm_sat_id)
    df = df[df["sat_id"].map(_is_gc)]
    df = df[df["in_label_set"].astype(int) == 1]
    df["is_los"] = df["is_los"].astype(int)
    df["t_utc_s"] = df["obs_time_utc"].astype(float)
    return df


def _merge_and_score(geom: pd.DataFrame, gt: pd.DataFrame, offset_s: float) -> tuple[Metrics, dict[str, int]]:
    # Time key: round to nearest second after applying offset to GEOM.
    g = geom.copy()
    t_key_g = np.rint(g["t_utc_s"].to_numpy(dtype=np.float64) + float(offset_s)).astype(np.int64)
    g["t_key"] = t_key_g
    g = g.drop(columns=["t_utc_s"])

    h = gt.copy()
    t_key_h = np.rint(h["t_utc_s"].to_numpy(dtype=np.float64)).astype(np.int64)
    h["t_key"] = t_key_h
    h = h.drop(columns=["t_utc_s"])

    merged = g.merge(h, on=["t_key", "sat_id"], how="inner", suffixes=("_geom", "_gt"))
    m = _confusion(merged["is_los_gt"].to_numpy(), merged["is_los_geom"].to_numpy())

    counts = {
        "geom_rows_gc": int(g.shape[0]),
        "gt_rows_gc": int(h.shape[0]),
        "merged_rows_gc": int(merged.shape[0]),
        "geom_unique_epochs": int(g["t_key"].nunique()),
        "gt_unique_epochs": int(h["t_key"].nunique()),
        "merged_unique_epochs": int(merged["t_key"].nunique()),
    }
    return m, counts


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare geometric LOS labels vs KLT GT labels (GPS+BeiDou only).")
    p.add_argument("--geom-dir", type=Path, required=True, help="Directory containing *_los_labels.csv")
    p.add_argument(
        "--gt-dir",
        type=Path,
        required=True,
        help="Directory containing klt_*_features_labels*.csv",
    )
    p.add_argument("--offsets-s", type=str, default="-18,0,18", help="Comma-separated offsets to try (seconds). Applied to GEOM.")
    p.add_argument("--out-csv", type=Path, default=Path("experiments/results/geom_vs_gt_klt_metrics.csv"))
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    geom_dir = args.geom_dir.resolve()
    gt_dir = args.gt_dir.resolve()
    offsets = [float(x.strip()) for x in str(args.offsets_s).split(",") if x.strip()]
    out_csv = args.out_csv.resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    geom_files = sorted(geom_dir.glob("*_los_labels.csv"))
    if not geom_files:
        raise RuntimeError(f"No *_los_labels.csv found under {geom_dir}")

    rows = []
    agg = {off: Metrics(n=0, tp=0, tn=0, fp=0, fn=0) for off in offsets}

    for gf in geom_files:
        ds = gf.name.replace("_los_labels.csv", "")
        # Prefer labelMinus18Auto if present, else fallback to any klt_*_features_labels.csv
        cand = [
            gt_dir / f"klt_{ds}_features_labels_labelMinus18Auto.csv",
            gt_dir / f"klt_{ds}_features_labels_labelMinus18.csv",
            gt_dir / f"klt_{ds}_features_labels.csv",
        ]
        gt_path = next((p for p in cand if p.exists()), None)
        if gt_path is None:
            print(f"[skip] {ds}: no GT file found in {gt_dir}")
            continue

        geom = _load_geom_csv(gf)
        gt = _load_gt_csv(gt_path)

        for off in offsets:
            m, c = _merge_and_score(geom, gt, offset_s=off)
            rows.append(
                {
                    "dataset": ds,
                    "offset_s": off,
                    "n": m.n,
                    "acc": m.acc,
                    "precision": m.prec,
                    "recall": m.rec,
                    "f1": m.f1,
                    "tp": m.tp,
                    "tn": m.tn,
                    "fp": m.fp,
                    "fn": m.fn,
                    **c,
                    "geom_file": str(gf),
                    "gt_file": str(gt_path),
                }
            )

            a = agg[off]
            agg[off] = Metrics(
                n=a.n + m.n,
                tp=a.tp + m.tp,
                tn=a.tn + m.tn,
                fp=a.fp + m.fp,
                fn=a.fn + m.fn,
            )

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    print(f"Wrote metrics: {out_csv}")
    for off in offsets:
        a = agg[off]
        print(
            f"AGG offset={off:+g}s  n={a.n}  acc={a.acc:.4f}  f1={a.f1:.4f}  prec={a.prec:.4f}  rec={a.rec:.4f}  "
            f"tp={a.tp} tn={a.tn} fp={a.fp} fn={a.fn}"
        )


if __name__ == "__main__":
    main()

