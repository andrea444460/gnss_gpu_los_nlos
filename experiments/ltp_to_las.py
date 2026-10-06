#!/usr/bin/env python3
"""Decode CHCNAV LC10V2 .ltp frames and georeference with CoPre .pos trajectory.

Format (reverse-engineered, LC10V2 / #LTP):
  - Text header (~512 B) then binary stream
  - Frames delimited by sync prefix 0x5A 0x23 0x05
  - Typical frame size 1461 B:
      16 B header (sync + device tick u32@4)
      479 × 3 B returns: uint16 range_mm LE (byte0–1) + uint8 mirror encoder (byte2)
      8 B footer

Timing: map device tick (u32@4) linearly to GPS seconds-of-week (.pos col 15).
  LTP filename and $GPRMC in .cil are UTC; GPSTime = UTC + leap seconds (18).
  With --cil, t0 is cross-checked against u32@124 before each $GPRMC.
Geometry (LC10 / AU20, reverse-engineered):
  - byte0–1: range mm as full uint16 LE (bits 0–15); nibbles of byte1 are
    not echo/quality flags — b1>>4 is bits 12–15 of the distance word
  - byte2: mirror encoder (~254 tick sweep/frame) → elevation in vertical fan
  Horizontal diversity along the route comes from vehicle motion between frames.

This is a best-effort decoder — with --ep, mount RPY, FOV, and lever arm
come from CoPre LTLaser.dll; still not a full Cloud Solve replacement.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import struct
from pathlib import Path
from typing import Iterator

import numpy as np

try:
    import laspy
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install laspy: pip install laspy") from exc

try:
    from pyproj import Transformer
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install pyproj: pip install pyproj") from exc

try:
    from gnss_gpu.io.ep_laser import apply_mount_rpy, lever_offset_enu, load_ep_calib
except ImportError:  # pragma: no cover
    load_ep_calib = None  # type: ignore[misc, assignment]
    apply_mount_rpy = None  # type: ignore[misc, assignment]
    lever_offset_enu = None  # type: ignore[misc, assignment]

try:
    from gnss_gpu.io.imr import imr_time_to_gpstime, read_imr
except ImportError:  # pragma: no cover
    read_imr = None  # type: ignore[misc, assignment]
    imr_time_to_gpstime = None  # type: ignore[misc, assignment]

FRAME_PREFIX = b"\x5a\x23\x05"
FRAME_HDR = 16
FRAME_FTR = 8
POINT_RECORD = 3
TYPICAL_FRAME = 1461
TICK_STEP = 16_777_216  # u32@4 increments by 2**24 per frame (observed)
CIL_GPRMC_PREFIX = rb"\$GPRMC,"
CIL_RECORD_PREFIX = 128
CIL_GPS_TIME_OFF = 124  # u32 LE GPSTime (same scale as .pos col 15)
GPS_UTC_LEAP_SECONDS = 18  # GPSTime = UTC + leap seconds (GPRMC / LTP filename are UTC)
LC10_MIRROR_TICKS = 254.0  # typical byte2 unwrap span per frame
DEFAULT_VERT_FOV_DEG = 27.89
DEFAULT_HORIZ_HALF_DEG = 12.275707  # from EP; reserved for future 2-D mirror models


def parse_ltp_start_sow(path: Path) -> float | None:
    """Parse HHMMSS (UTC) from names like 2026-03-24-123712_002.ltp."""
    m = re.search(r"-(\d{6})_", path.name)
    if not m:
        return None
    hhmmss = m.group(1)
    return utc_hhmmss_sod(hhmmss)


def utc_hhmmss_sod(hhmmss: str) -> float:
    hh, mm, ss = int(hhmmss[0:2]), int(hhmmss[2:4]), int(hhmmss[4:6])
    return hh * 3600.0 + mm * 60.0 + ss


def gpstime_from_utc_sod(
    pos_t: np.ndarray, utc_sod: float, *, leap_seconds: float = GPS_UTC_LEAP_SECONDS
) -> float:
    """Map UTC second-of-day (GPRMC / LTP filename) to POS GPSTime (week seconds)."""
    gps_sod = (utc_sod + leap_seconds) % 86400.0
    day_sows = pos_t % 86400.0
    idx = int(np.argmin(np.abs(day_sows - gps_sod)))
    return float(pos_t[idx])


def read_chcnav_pos(path: Path) -> dict[str, np.ndarray]:
    """Load CoPre .pos: GPS SOW, lat, lon, ellipsoidal height."""
    times: list[float] = []
    lats: list[float] = []
    lons: list[float] = []
    heights: list[float] = []
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 16 or not parts[0].startswith("202"):
                continue
            try:
                times.append(float(parts[15]))
                lats.append(float(parts[2]))
                lons.append(float(parts[3]))
                heights.append(float(parts[4]))
            except ValueError:
                continue
    if not times:
        raise ValueError(f"no trajectory rows in {path}")
    return {
        "t": np.asarray(times, dtype=np.float64),
        "lat": np.asarray(lats, dtype=np.float64),
        "lon": np.asarray(lons, dtype=np.float64),
        "h": np.asarray(heights, dtype=np.float64),
    }


def validate_ep_serial(ep_path: Path, ltp_path: Path) -> None:
    blob = ep_path.read_bytes()
    serial_m = re.search(rb"s\d{9}[A-Z]", blob)
    serial = serial_m.group(0).decode("ascii") if serial_m else "?"
    ltp_serial_m = re.search(rb"s\d{9}[A-Z]", ltp_path.read_bytes()[:512])
    ltp_serial = ltp_serial_m.group(0).decode("ascii") if ltp_serial_m else "?"
    if serial != "?" and ltp_serial != "?" and serial != ltp_serial:
        raise ValueError(
            f"EP serial {serial} does not match LTP scanner {ltp_serial}"
        )


def read_cil_gprmc_times(cil_path: Path) -> list[tuple[float, str]]:
    """Parse CoPre .cil: (GPSTime, GPRMC HHMMSS) for each $GPRMC epoch."""
    records: list[tuple[float, str]] = []
    with cil_path.open("rb") as fh:
        data = fh.read()
    for match in re.finditer(CIL_GPRMC_PREFIX, data):
        pos = match.start()
        if pos < CIL_RECORD_PREFIX:
            continue
        gps_t = struct.unpack_from(
            "<I", data, pos - CIL_RECORD_PREFIX + CIL_GPS_TIME_OFF
        )[0]
        line_end = data.find(b"\n", pos)
        line = data[pos : line_end if line_end > 0 else pos + 96].decode(
            "ascii", "replace"
        )
        parts = line.split(",")
        hhmmss = parts[1].split(".")[0] if len(parts) > 1 else ""
        if len(hhmmss) == 6 and hhmmss.isdigit():
            records.append((float(gps_t), hhmmss))
    return records


def resolve_laser_sow_window_cil(
    cil_path: Path,
    ltp_path: Path,
    pos_t: np.ndarray,
    *,
    leap_seconds: float = GPS_UTC_LEAP_SECONDS,
) -> tuple[float, float, str]:
    """Anchor laser on UTC $GPRMC from LTP filename, converted to GPSTime (+ leap)."""
    records = read_cil_gprmc_times(cil_path)
    if not records:
        raise ValueError(f"no $GPRMC records in {cil_path}")

    m = re.search(r"-(\d{6})_", ltp_path.name)
    if not m:
        raise ValueError(f"cannot parse HHMMSS from {ltp_path.name}")
    target = m.group(1)

    cil_gps_t: float | None = None
    for gps_t, hhmmss in records:
        if hhmmss == target:
            cil_gps_t = gps_t
            break
    if cil_gps_t is None:
        raise ValueError(f"no $GPRMC,{target} in {cil_path}")

    utc_sod = utc_hhmmss_sod(target)
    t0 = gpstime_from_utc_sod(pos_t, utc_sod, leap_seconds=leap_seconds)
    if abs(t0 - cil_gps_t) > 1.0:
        raise ValueError(
            f"CIL/POS time mismatch at $GPRMC,{target}: "
            f"UTC+{leap_seconds:.0f}s -> GPSTime {t0:.1f}, CIL u32@124 {cil_gps_t:.1f}"
        )

    times_after = [gps_t for gps_t, _ in records if gps_t >= t0]
    t1_cil = max(times_after) if times_after else t0 + 300.0

    pos_mask = pos_t >= t0 - 1.0
    if not np.any(pos_mask):
        raise ValueError(
            f"POS does not cover laser start GPSTime {t0:.1f}"
        )
    t1 = min(t1_cil, float(pos_t[pos_mask].max()))
    hh = int(target[0:2])
    mm = int(target[2:4])
    ss = int(target[4:6])
    note = (
        f"cil $GPRMC,{target} UTC {hh:02d}:{mm:02d}:{ss:02d} "
        f"+{leap_seconds:.0f}s leap -> GPSTime {t0:.1f}"
    )
    return t0, t1, note


def resolve_laser_sow_window(
    pos_t: np.ndarray,
    ltp_path: Path,
    *,
    leap_seconds: float = GPS_UTC_LEAP_SECONDS,
) -> tuple[float, float]:
    """Return GPS SOW [t0, t1] from LTP filename UTC + leap seconds vs .pos."""
    utc_sod = parse_ltp_start_sow(ltp_path)
    if utc_sod is None:
        t0 = float(np.percentile(pos_t, 5))
        t1 = float(np.percentile(pos_t, 95))
        return t0, t1

    t0 = gpstime_from_utc_sod(pos_t, utc_sod, leap_seconds=leap_seconds)

    mask = pos_t >= t0
    if not np.any(mask):
        raise ValueError("POS does not cover laser start time")
    t1 = float(pos_t[mask].max())
    return t0, t1


def scan_tick_endpoints(ltp_path: Path, sample_tail: int = 32) -> tuple[int, int, int]:
    """Return (first_tick, last_tick, n_frames) by streaming the file."""
    first_tick: int | None = None
    last_tick: int | None = None
    n_frames = 0
    tick_ring: list[int] = []

    with ltp_path.open("rb") as fh:
        buf = fh.read(8 * 1024 * 1024)
        pos = buf.find(FRAME_PREFIX)
        if pos < 0:
            raise ValueError("no LTP frame sync found")

        def consume_frame(frame: bytes) -> None:
            nonlocal first_tick, last_tick, n_frames
            if len(frame) < FRAME_HDR + FRAME_FTR + POINT_RECORD:
                return
            tick = struct.unpack_from("<I", frame, 4)[0]
            if first_tick is None:
                first_tick = tick
            last_tick = tick
            n_frames += 1
            tick_ring.append(tick)
            if len(tick_ring) > sample_tail:
                tick_ring.pop(0)

        while True:
            while pos >= 0 and pos + 3 <= len(buf) and buf[pos : pos + 3] != FRAME_PREFIX[:3]:
                pos += 1
            if pos < 0 or pos + TYPICAL_FRAME > len(buf):
                chunk = fh.read(8 * 1024 * 1024)
                if not chunk:
                    break
                buf = buf[pos:] + chunk if pos >= 0 else chunk
                pos = 0
                continue

            nxt = buf.find(FRAME_PREFIX, pos + 4)
            if nxt < 0:
                if len(buf) - pos < TYPICAL_FRAME:
                    chunk = fh.read(8 * 1024 * 1024)
                    if not chunk:
                        if len(buf) - pos >= FRAME_HDR + FRAME_FTR + POINT_RECORD:
                            consume_frame(buf[pos:])
                        break
                    buf = buf[pos:] + chunk
                    pos = 0
                    continue
                nxt = len(buf)

            gap = nxt - pos
            if 1456 <= gap <= 1476:
                consume_frame(buf[pos:nxt])
                pos = nxt
            else:
                pos += 1

    if first_tick is None or last_tick is None or n_frames == 0:
        raise ValueError("no decodable LTP frames")
    if n_frames > sample_tail and tick_ring:
        last_tick = tick_ring[-1]
    return first_tick, last_tick, n_frames


def iter_ltp_frames(
    ltp_path: Path, *, max_frames: int | None = None, stride: int = 1
) -> Iterator[tuple[int, bytes]]:
    """Yield (frame_index, frame_bytes)."""
    idx = 0
    yielded = 0
    with ltp_path.open("rb") as fh:
        buf = fh.read(8 * 1024 * 1024)
        pos = buf.find(FRAME_PREFIX)
        if pos < 0:
            return

        while True:
            while pos >= 0 and pos + 3 <= len(buf) and buf[pos : pos + 3] != FRAME_PREFIX[:3]:
                pos += 1
            if pos < 0 or pos + FRAME_HDR + FRAME_FTR + POINT_RECORD > len(buf):
                chunk = fh.read(8 * 1024 * 1024)
                if not chunk:
                    return
                buf = buf[pos:] + chunk if pos >= 0 else chunk
                pos = 0
                continue

            nxt = buf.find(FRAME_PREFIX, pos + 4)
            if nxt < 0:
                if len(buf) - pos < TYPICAL_FRAME:
                    chunk = fh.read(8 * 1024 * 1024)
                    if not chunk:
                        if len(buf) - pos >= FRAME_HDR + FRAME_FTR + POINT_RECORD:
                            gap = len(buf) - pos
                            if yielded % stride == 0:
                                yield idx, buf[pos:]
                            idx += 1
                            yielded += 1
                        return
                    buf = buf[pos:] + chunk
                    pos = 0
                    continue
                nxt = len(buf)

            gap = nxt - pos
            if 1456 <= gap <= 1476:
                if yielded % stride == 0:
                    yield idx, buf[pos:nxt]
                idx += 1
                yielded += 1
                if max_frames is not None and yielded >= max_frames:
                    return
                pos = nxt
            else:
                pos += 1


def decode_lc10_frame(
    frame: bytes,
    *,
    elev_fov_deg: float = DEFAULT_VERT_FOV_DEG,
    horiz_half_deg: float = DEFAULT_HORIZ_HALF_DEG,
    invert_elev: bool = False,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Decode one frame -> tick, range_m, elev_rad, mirror_enc (byte2)."""
    if len(frame) < FRAME_HDR + FRAME_FTR + POINT_RECORD:
        return 0, np.empty(0), np.empty(0), np.empty(0)

    tick = struct.unpack_from("<I", frame, 4)[0]
    payload = frame[FRAME_HDR : len(frame) - FRAME_FTR]
    nrec = len(payload) // POINT_RECORD
    if nrec == 0:
        return tick, np.empty(0), np.empty(0), np.empty(0)

    rec = np.frombuffer(payload[: nrec * POINT_RECORD], dtype=np.uint8).reshape(nrec, POINT_RECORD)
    # Full 16-bit range in mm (LE). Do not mask with 0x0FFF: b1>>4 is bits
    # 12–15 of the distance, required for returns > 4.095 m (~88% of shots).
    range_mm = rec[:, 0].astype(np.uint16) + (rec[:, 1].astype(np.uint16) << 8)
    mirror = rec[:, 2].astype(np.float64)

    valid = (range_mm >= 500) & (range_mm <= 200_000) & (range_mm < 65535)
    range_m = range_mm[valid].astype(np.float64) / 1000.0
    mirror = mirror[valid]

    if range_m.size == 0:
        return tick, np.empty(0), np.empty(0), np.empty(0)

    elev = np.deg2rad((mirror - 127.5) * (elev_fov_deg / LC10_MIRROR_TICKS))
    if invert_elev:
        elev = -elev

    return tick, range_m, elev, mirror.astype(np.uint16)


def body_xyz(
    range_m: np.ndarray,
    elev_rad: np.ndarray,
    *,
    fan_plane: str = "side",
    scan_side: str = "right",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Body frame: X right, Y forward, Z up.

    side    – vertical fan in X–Z (mirror sweep beside the vehicle)
    forward – vertical fan in Y–Z (ahead)
    scan_side – left flips cross-track (X) if facades appear on wrong side
    """
    horiz = range_m * np.cos(elev_rad)
    z_loc = range_m * np.sin(elev_rad)
    if fan_plane == "forward":
        x_loc = np.zeros_like(range_m)
        y_loc = horiz
    else:
        x_loc = horiz
        y_loc = np.zeros_like(range_m)
    if scan_side == "left":
        x_loc = -x_loc
    return x_loc, y_loc, z_loc


def imu_accel_to_body(
    ax: np.ndarray, ay: np.ndarray, az: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """NovAtel/AutoPOS IMU (X fwd, Y left, Z up) -> body (X right, Y fwd, Z up)."""
    return -ay, ax, az


def roll_pitch_from_accel(
    ax: np.ndarray, ay: np.ndarray, az: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Approximate roll/pitch (rad) from specific force in body frame."""
    roll = np.arctan2(ay, az)
    pitch = np.arctan2(-ax, np.hypot(ay, az))
    return roll, pitch


def load_imu_attitude(
    imr_path: Path,
    *,
    leap_seconds: float = GPS_UTC_LEAP_SECONDS,
    imu_attitude: str = "pitch",
) -> dict[str, np.ndarray]:
    if read_imr is None or imr_time_to_gpstime is None:
        raise RuntimeError(
            "gnss_gpu not importable — run with PYTHONPATH=python or pip install -e ."
        )
    imu = read_imr(imr_path)
    t = imr_time_to_gpstime(imu.header, imu.time, leap_seconds=leap_seconds)
    bx, by, bz = imu_accel_to_body(imu.ax, imu.ay, imu.az)
    roll, pitch = roll_pitch_from_accel(bx, by, bz)
    if imu_attitude == "pitch":
        roll = np.zeros_like(roll)
    elif imu_attitude == "none":
        roll = np.zeros_like(roll)
        pitch = np.zeros_like(pitch)
    return {
        "t": t,
        "roll": roll,
        "pitch": pitch,
        "header": imu.header,
        "mode": imu_attitude,
    }


def enu_from_body(
    x_loc: np.ndarray,
    y_loc: np.ndarray,
    z_loc: np.ndarray,
    east: np.ndarray | float,
    north: np.ndarray | float,
    height: np.ndarray | float,
    yaw: np.ndarray | float,
    roll: np.ndarray | float | None = None,
    pitch: np.ndarray | float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotate body (X right, Y forward, Z up) into UTM ENU."""
    if roll is None and pitch is None:
        sin_y, cos_y = np.sin(yaw), np.cos(yaw)
        xg = east + y_loc * sin_y + x_loc * cos_y
        yg = north + y_loc * cos_y - x_loc * sin_y
        zg = height + z_loc
        return xg, yg, zg

    if roll is None:
        roll = np.zeros_like(yaw, dtype=np.float64)
    if pitch is None:
        pitch = np.zeros_like(yaw, dtype=np.float64)

    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    r00 = cy * cp
    r01 = cy * sp * sr - sy * cr
    r02 = cy * sp * cr + sy * sr
    r10 = sy * cp
    r11 = sy * sp * sr + cy * cr
    r12 = sy * sp * cr - cy * sr
    r20 = -sp
    r21 = cp * sr
    r22 = cp * cr

    xg = east + r00 * x_loc + r01 * y_loc + r02 * z_loc
    yg = north + r10 * x_loc + r11 * y_loc + r12 * z_loc
    zg = height + r20 * x_loc + r21 * y_loc + r22 * z_loc
    return xg, yg, zg


def trajectory_utm_and_heading(traj: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    trans = Transformer.from_crs("EPSG:4326", "EPSG:32636", always_xy=True)
    east, north = trans.transform(traj["lon"], traj["lat"])
    de = np.gradient(east)
    dn = np.gradient(north)
    yaw = np.arctan2(de, dn)
    yaw = np.unwrap(yaw)
    return {
        "t": traj["t"],
        "east": east,
        "north": north,
        "h": traj["h"],
        "yaw": yaw,
    }


def tick_to_time(
    tick: int, tick0: int, tick1: int, t0: float, t1: float
) -> float:
    if tick1 <= tick0:
        return t0
    return t0 + (tick - tick0) * (t1 - t0) / (tick1 - tick0)


def point_times_in_frame(
    t_frame: float,
    t_prev: float | None,
    enc: np.ndarray,
) -> np.ndarray:
    """Spread points across the frame interval using the mirror encoder (byte2)."""
    enc0, enc1 = float(enc.min()), float(enc.max())
    if enc1 > enc0:
        frac = (enc - enc0) / (enc1 - enc0)
    else:
        frac = np.linspace(0.0, 1.0, enc.size, dtype=np.float64)
    if t_prev is not None and t_frame > t_prev:
        return t_prev + frac * (t_frame - t_prev)
    return np.full(enc.size, t_frame, dtype=np.float64)


def ltp_to_las(
    ltp_path: Path,
    pos_path: Path,
    las_path: Path,
    *,
    cil_path: Path | None = None,
    ep_path: Path | None = None,
    imr_path: Path | None = None,
    imu_attitude: str = "pitch",
    invert_elev: bool = False,
    leap_seconds: float = GPS_UTC_LEAP_SECONDS,
    max_frames: int | None = None,
    stride: int = 1,
    point_stride: int = 1,
    elev_fov_deg: float = DEFAULT_VERT_FOV_DEG,
    horiz_half_deg: float = DEFAULT_HORIZ_HALF_DEG,
    fan_plane: str = "side",
    scan_side: str = "right",
    subframe_timing: bool = True,
) -> dict[str, float]:
    traj = read_chcnav_pos(pos_path)
    sync_note = "pos filename"
    if cil_path is not None:
        t0, t1, sync_note = resolve_laser_sow_window_cil(
            cil_path, ltp_path, traj["t"], leap_seconds=leap_seconds
        )
    else:
        t0, t1 = resolve_laser_sow_window(
            traj["t"], ltp_path, leap_seconds=leap_seconds
        )
        sync_note = f"filename UTC +{leap_seconds:.0f}s leap -> GPSTime {t0:.1f}"
    tick0, tick1, n_frames_file = scan_tick_endpoints(ltp_path)
    pose = trajectory_utm_and_heading(traj)
    ep_calib = None
    elev_fov = elev_fov_deg
    horiz_half = horiz_half_deg
    if ep_path is not None:
        if load_ep_calib is None or apply_mount_rpy is None or lever_offset_enu is None:
            raise RuntimeError(
                "gnss_gpu not importable — run with PYTHONPATH=python"
            )
        validate_ep_serial(ep_path, ltp_path)
        ep_calib = load_ep_calib(ep_path)
        elev_fov = ep_calib.elev_fov_deg
        if elev_fov < 1.0:
            elev_fov = elev_fov_deg if elev_fov_deg >= 1.0 else DEFAULT_VERT_FOV_DEG
        horiz_half = ep_calib.horiz_half_deg
        sync_note += (
            f"; EP mount RPY=({ep_calib.mount_roll_deg:.1f},"
            f"{ep_calib.mount_pitch_deg:.1f},{ep_calib.mount_yaw_deg:.1f})"
            f" FOVv={ep_calib.elev_fov_deg:.1f} FOVh={ep_calib.horiz_half_deg:.1f}deg"
            f" lever=({ep_calib.lever_x_m:.3f},{ep_calib.lever_y_m:.3f},"
            f"{ep_calib.lever_z_m:.3f})m"
        )
    imu = (
        load_imu_attitude(imr_path, leap_seconds=leap_seconds, imu_attitude=imu_attitude)
        if imr_path
        else None
    )
    if imu is not None:
        sync_note += f"; imu {imu['header'].imu_name} attitude={imu['mode']}"

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    zs: list[np.ndarray] = []
    ts: list[np.ndarray] = []
    is_: list[np.ndarray] = []

    n_frames = 0
    n_points = 0
    t_prev: float | None = None
    for _, frame in iter_ltp_frames(ltp_path, max_frames=max_frames, stride=stride):
        tick, range_m, elev, mirror = decode_lc10_frame(
            frame,
            elev_fov_deg=elev_fov,
            horiz_half_deg=horiz_half,
            invert_elev=invert_elev,
        )
        if range_m.size == 0:
            continue

        if point_stride > 1:
            range_m = range_m[::point_stride]
            elev = elev[::point_stride]
            mirror = mirror[::point_stride]

        t_frame = tick_to_time(tick, tick0, tick1, t0, t1)
        x_loc, y_loc, z_loc = body_xyz(
            range_m, elev, fan_plane=fan_plane, scan_side=scan_side
        )
        if ep_calib is not None:
            x_loc, y_loc, z_loc = apply_mount_rpy(x_loc, y_loc, z_loc, ep_calib)

        if subframe_timing:
            t_pts = point_times_in_frame(t_frame, t_prev, mirror.astype(np.float64))
        else:
            t_pts = np.full(range_m.size, t_frame, dtype=np.float64)
        t_prev = t_frame

        e = np.interp(t_pts, pose["t"], pose["east"])
        n = np.interp(t_pts, pose["t"], pose["north"])
        h = np.interp(t_pts, pose["t"], pose["h"])
        yaw = np.interp(t_pts, pose["t"], pose["yaw"])
        roll: np.ndarray | None = None
        pitch: np.ndarray | None = None
        if imu is not None:
            roll = np.interp(t_pts, imu["t"], imu["roll"])
            pitch = np.interp(t_pts, imu["t"], imu["pitch"])
        if ep_calib is not None:
            de, dn, dh = lever_offset_enu(ep_calib, yaw, roll=roll, pitch=pitch)
            e = e + de
            n = n + dn
            h = h + dh
        if imu is not None:
            xg, yg, zg = enu_from_body(
                x_loc, y_loc, z_loc, e, n, h, yaw, roll=roll, pitch=pitch
            )
        else:
            xg, yg, zg = enu_from_body(x_loc, y_loc, z_loc, e, n, h, yaw)

        xs.append(xg.astype(np.float64))
        ys.append(yg.astype(np.float64))
        zs.append(zg.astype(np.float64))
        ts.append(t_pts.astype(np.float64))
        is_.append(mirror.astype(np.uint16))
        n_frames += 1
        n_points += range_m.size

    if n_points == 0:
        raise RuntimeError("no points decoded — check LTP/POS time overlap")

    X = np.concatenate(xs)
    Y = np.concatenate(ys)
    Z = np.concatenate(zs)
    T = np.concatenate(ts)
    I = np.concatenate(is_)

    header = laspy.LasHeader(point_format=1, version="1.2")
    header.offsets = np.floor([X.min(), Y.min(), Z.min()])
    header.scales = np.array([0.001, 0.001, 0.001])
    las = laspy.LasData(header)
    las.x = X
    las.y = Y
    las.z = Z
    las.gps_time = T
    las.intensity = I
    las_path.parent.mkdir(parents=True, exist_ok=True)
    las.write(str(las_path))

    return {
        "frames": float(n_frames),
        "points": float(n_points),
        "t0": t0,
        "t1": t1,
        "sync": sync_note,
        "tick0": float(tick0),
        "tick1": float(tick1),
        "file_frames_est": float(n_frames_file),
        "xmin": float(X.min()),
        "xmax": float(X.max()),
        "ymin": float(Y.min()),
        "ymax": float(Y.max()),
        "zmin": float(Z.min()),
        "zmax": float(Z.max()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Decode CHCNAV .ltp and georeference with .pos")
    ap.add_argument("--ltp", type=Path, required=True, help="Input .ltp file")
    ap.add_argument("--pos", type=Path, required=True, help="CoPre trajectory .pos")
    ap.add_argument(
        "--cil",
        type=Path,
        default=None,
        help="CoPre sync .cil (recommended: GPSTime anchor from $GPRMC)",
    )
    ap.add_argument(
        "--ep",
        type=Path,
        default=None,
        help="CHCNAV .EP: mount angles, vertical FOV, lever arm via CoPre LTLaser.dll",
    )
    ap.add_argument(
        "--imr",
        type=Path,
        default=None,
        help="NovAtel Waypoint IMR for vehicle tilt (use with --imu-attitude pitch)",
    )
    ap.add_argument(
        "--imu-attitude",
        choices=("none", "pitch", "full"),
        default="pitch",
        help="IMU tilt model: pitch=road grade only (default), full=roll+pitch, none=ignore",
    )
    ap.add_argument(
        "--invert-elev",
        action="store_true",
        help="flip vertical fan sign (try if facades lean wrong way)",
    )
    ap.add_argument(
        "--leap-seconds",
        type=float,
        default=GPS_UTC_LEAP_SECONDS,
        help="UTC to GPSTime offset (default 18 leap seconds)",
    )
    ap.add_argument("--out", type=Path, required=True, help="Output .las path")
    ap.add_argument("--max-frames", type=int, default=None, help="Limit frames (debug)")
    ap.add_argument("--stride", type=int, default=1, help="Use every Nth frame")
    ap.add_argument("--point-stride", type=int, default=1, help="Keep every Nth point per frame")
    ap.add_argument(
        "--elev-fov-deg",
        type=float,
        default=DEFAULT_VERT_FOV_DEG,
        help="Vertical FOV for 16 elevation channels (default LC10 ~28 deg)",
    )
    ap.add_argument(
        "--horiz-half-deg",
        type=float,
        default=DEFAULT_HORIZ_HALF_DEG,
        help="Half horizontal FOV from EP (not used in 3-byte raw decode yet)",
    )
    ap.add_argument(
        "--fan-plane",
        choices=("side", "forward"),
        default="side",
        help="side=cross-track curtain (default); forward=cone-like ahead",
    )
    ap.add_argument(
        "--scan-side",
        choices=("right", "left"),
        default="right",
        help="cross-track side (flip if facades appear on wrong side)",
    )
    ap.add_argument(
        "--no-subframe-timing",
        action="store_true",
        help="assign whole frame to one GPS epoch (faster, coarser pose)",
    )
    args = ap.parse_args()

    stats = ltp_to_las(
        args.ltp,
        args.pos,
        args.out,
        cil_path=args.cil,
        ep_path=args.ep,
        imr_path=args.imr,
        imu_attitude=args.imu_attitude,
        invert_elev=args.invert_elev,
        leap_seconds=args.leap_seconds,
        max_frames=args.max_frames,
        stride=args.stride,
        point_stride=args.point_stride,
        elev_fov_deg=args.elev_fov_deg,
        horiz_half_deg=args.horiz_half_deg,
        fan_plane=args.fan_plane,
        scan_side=args.scan_side,
        subframe_timing=not args.no_subframe_timing,
    )
    print(f"Wrote {args.out}")
    print(f"sync: {stats['sync']}")
    print(
        f"frames={int(stats['frames'])} points={int(stats['points'])} "
        f"sow=[{stats['t0']:.1f}, {stats['t1']:.1f}] "
        f"bbox X[{stats['xmin']:.1f},{stats['xmax']:.1f}] "
        f"Y[{stats['ymin']:.1f},{stats['ymax']:.1f}] "
        f"Z[{stats['zmin']:.1f},{stats['zmax']:.1f}]"
    )


if __name__ == "__main__":
    main()
