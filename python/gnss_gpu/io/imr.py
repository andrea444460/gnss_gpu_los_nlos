#!/usr/bin/env python3
"""Parse NovAtel Waypoint generic IMR (see IMR File format spec)."""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

IMR_HEADER_SIZE = 512
IMR_RECORD_SIZE = 32
IMR_MAGIC = b"$IMURAW\x00"


@dataclass(frozen=True)
class ImrHeader:
    intel_little_endian: bool
    version: float
    delta_theta: bool
    delta_velocity: bool
    data_rate_hz: float
    gyro_scale: float
    accel_scale: float
    utc_or_gps_time: int  # 0 unknown, 1 UTC SOW, 2 GPS SOW
    rcv_time_or_corr_time: int
    time_tag_bias: float
    imu_name: str
    program_name: str
    lever_arm_valid: bool
    lever_arm_mm: tuple[int, int, int]


@dataclass(frozen=True)
class ImrData:
    header: ImrHeader
    time: np.ndarray  # seconds of week (GPS or UTC per header)
    gx: np.ndarray  # deg/s or delta-theta per header flags
    gy: np.ndarray
    gz: np.ndarray
    ax: np.ndarray  # m/s^2 or delta-v per header flags
    ay: np.ndarray
    az: np.ndarray


def _decode_c_string(buf: bytes) -> str:
    end = buf.find(b"\x00")
    if end < 0:
        end = len(buf)
    return buf[:end].decode("ascii", errors="replace")


def parse_imr_header(blob: bytes) -> ImrHeader:
    if len(blob) < IMR_HEADER_SIZE:
        raise ValueError(f"IMR header must be {IMR_HEADER_SIZE} bytes")
    if not blob.startswith(IMR_MAGIC[:7]):  # allow missing final null
        raise ValueError("not an IMR file (missing $IMURAW magic)")

    # Layout per NovAtel Waypoint IMR_File.htm (Intel / little-endian default).
    intel = struct.unpack_from("<b", blob, 8)[0] == 0
    endian = "<" if intel else ">"
    version = struct.unpack_from(f"{endian}d", blob, 9)[0]
    delta_theta, delta_velocity = struct.unpack_from(f"{endian}ii", blob, 17)
    data_rate_hz, gyro_scale, accel_scale = struct.unpack_from(f"{endian}3d", blob, 25)
    utc_or_gps, rcv_corr = struct.unpack_from(f"{endian}ii", blob, 49)
    time_tag_bias = struct.unpack_from(f"{endian}d", blob, 57)[0]
    imu_name = _decode_c_string(blob[65:97])
    program_name = _decode_c_string(blob[101:133])
    lever_valid = struct.unpack_from(f"{endian}?", blob, 145)[0]
    lx, ly, lz = struct.unpack_from(f"{endian}iii", blob, 146)

    return ImrHeader(
        intel_little_endian=intel,
        version=version,
        delta_theta=bool(delta_theta),
        delta_velocity=bool(delta_velocity),
        data_rate_hz=data_rate_hz,
        gyro_scale=gyro_scale,
        accel_scale=accel_scale,
        utc_or_gps_time=int(utc_or_gps),
        rcv_time_or_corr_time=int(rcv_corr),
        time_tag_bias=time_tag_bias,
        imu_name=imu_name,
        program_name=program_name,
        lever_arm_valid=bool(lever_valid),
        lever_arm_mm=(int(lx), int(ly), int(lz)),
    )


def _to_physical(header: ImrHeader, gyro: np.ndarray, accel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert raw counts to deg/s and m/s^2."""
    g = gyro.astype(np.float64) * header.gyro_scale
    a = accel.astype(np.float64) * header.accel_scale
    if header.delta_theta:
        g = g * header.data_rate_hz
    if header.delta_velocity:
        a = a * header.data_rate_hz
    return g, a


def read_imr(path: Path, *, max_records: int | None = None) -> ImrData:
    """Load IMR header and records."""
    size = path.stat().st_size
    n_rec = (size - IMR_HEADER_SIZE) // IMR_RECORD_SIZE
    if max_records is not None:
        n_rec = min(n_rec, max_records)

    with path.open("rb") as fh:
        header_blob = fh.read(IMR_HEADER_SIZE)
        header = parse_imr_header(header_blob)
        payload = fh.read(n_rec * IMR_RECORD_SIZE)

    n_read = len(payload) // IMR_RECORD_SIZE
    rows = np.frombuffer(payload[: n_read * IMR_RECORD_SIZE], dtype=np.uint8).reshape(
        n_read, IMR_RECORD_SIZE
    )
    if header.intel_little_endian:
        times = rows[:, :8].copy().view(np.float64).reshape(-1)
        ints = rows[:, 8:].view(np.int32).reshape(n_read, 6)
    else:
        times = rows[:, :8].copy().view(">f8").reshape(-1)
        ints = rows[:, 8:].view(">i4").reshape(n_read, 6)
    gyro = ints[:, :3].astype(np.float64)
    accel = ints[:, 3:].astype(np.float64)

    g_phys, a_phys = _to_physical(header, gyro, accel)
    return ImrData(
        header=header,
        time=times.copy(),
        gx=g_phys[:, 0],
        gy=g_phys[:, 1],
        gz=g_phys[:, 2],
        ax=a_phys[:, 0],
        ay=a_phys[:, 1],
        az=a_phys[:, 2],
    )


def imr_time_to_gpstime(header: ImrHeader, t: np.ndarray, *, leap_seconds: float = 18.0) -> np.ndarray:
    """Map IMR time tags to GPSTime (.pos / CIL scale)."""
    if header.utc_or_gps_time == 2:
        return t
    if header.utc_or_gps_time == 1:
        return t + leap_seconds
    return t + leap_seconds  # unknown defaults to UTC + leap
