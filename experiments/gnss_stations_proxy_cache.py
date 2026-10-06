"""NPZ cache for C/N0 fade proxy rows (shared across batch scripts)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

ProxyArrays = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def proxy_cache_path(
    cache_dir: Path,
    obs_path: Path,
    *,
    min_el_deg: float,
    fade_db: float,
    epoch_step: int,
) -> Path:
    stem = obs_path.stem
    name = f"{stem}_el{min_el_deg:.0f}_fade{fade_db:.1f}_step{max(1, int(epoch_step))}.npz"
    return cache_dir / name


def _rows_from_arrays(
    tag: str,
    prn: np.ndarray,
    tow: np.ndarray,
    is_los: np.ndarray,
    az: np.ndarray,
    el: np.ndarray,
) -> list[dict]:
    n = int(tow.size)
    if n == 0:
        return []
    rows: list[dict] = []
    append = rows.append
    prn_list = prn.tolist()
    tow_list = tow.tolist()
    los_list = is_los.tolist()
    az_list = az.tolist()
    el_list = el.tolist()
    for i in range(n):
        append(
            {
                "tag": tag,
                "prn": str(prn_list[i]),
                "tow": float(tow_list[i]),
                "is_los": int(los_list[i]),
                "is_visible": 1,
                "az": float(az_list[i]),
                "el": float(el_list[i]),
                "excess_delay_m": 0.0,
            }
        )
    return rows


def save_proxy_cache(
    path: Path,
    *,
    tag: str,
    prn: np.ndarray,
    tow: np.ndarray,
    is_los: np.ndarray,
    az: np.ndarray,
    el: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        tag=np.asarray(tag),
        prn=np.asarray(prn, dtype="U8"),
        tow=np.asarray(tow, dtype=np.float64),
        is_los=np.asarray(is_los, dtype=np.int8),
        az=np.asarray(az, dtype=np.float32),
        el=np.asarray(el, dtype=np.float32),
    )


def load_proxy_cache(path: Path, tag: str) -> list[dict] | None:
    if not path.is_file():
        return None
    try:
        data = np.load(path, allow_pickle=False)
        return _rows_from_arrays(
            tag,
            data["prn"],
            data["tow"],
            data["is_los"],
            data["az"],
            data["el"],
        )
    except (OSError, KeyError, ValueError):
        return None


def cache_is_fresh(cache_path: Path, obs_path: Path) -> bool:
    if not cache_path.is_file():
        return False
    try:
        return cache_path.stat().st_mtime >= obs_path.stat().st_mtime
    except OSError:
        return False


def load_proxy_arrays(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    if not path.is_file():
        return None
    try:
        data = np.load(path, allow_pickle=False)
        return data["prn"], data["tow"], data["is_los"], data["az"], data["el"]
    except (OSError, KeyError, ValueError):
        return None


def get_proxy_arrays(
    obs_path: Path,
    eph: Any,
    rx: np.ndarray,
    tag: str,
    *,
    min_el_deg: float,
    fade_db: float,
    epoch_step: int,
    cache_dir: Path | None,
    no_cache: bool,
    compute_fn: Callable[..., ProxyArrays],
) -> ProxyArrays:
    """Load proxy arrays from cache or compute via ``compute_fn``."""
    step = max(1, int(epoch_step))
    cp: Path | None = None
    if cache_dir is not None and not no_cache:
        cp = proxy_cache_path(
            cache_dir,
            obs_path,
            min_el_deg=min_el_deg,
            fade_db=fade_db,
            epoch_step=step,
        )
        if cache_is_fresh(cp, obs_path):
            cached = load_proxy_arrays(cp)
            if cached is not None:
                return cached

    arrays = compute_fn(
        obs_path,
        eph,
        rx,
        tag,
        min_el_deg=min_el_deg,
        fade_db=fade_db,
        epoch_step=step,
    )
    if cp is not None:
        prn, tow, is_los, az, el = arrays
        save_proxy_cache(cp, tag=tag, prn=prn, tow=tow, is_los=is_los, az=az, el=el)
    return arrays


def get_proxy_rows(
    obs_path: Path,
    eph: Any,
    rx: np.ndarray,
    tag: str,
    *,
    min_el_deg: float,
    fade_db: float,
    epoch_step: int,
    cache_dir: Path | None,
    no_cache: bool,
    compute_fn: Callable[..., ProxyArrays],
) -> list[dict]:
    """Load proxy rows from cache or compute via ``compute_fn``."""
    arrays = get_proxy_arrays(
        obs_path,
        eph,
        rx,
        tag,
        min_el_deg=min_el_deg,
        fade_db=fade_db,
        epoch_step=epoch_step,
        cache_dir=cache_dir,
        no_cache=no_cache,
        compute_fn=compute_fn,
    )
    return _rows_from_arrays(tag, *arrays)
