"""Load CHCNAV .EP laser calibration via CoPre LTLaser.dll (ltp_ep_dump.exe)."""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class EpLaserCalib:
    mount_roll_deg: float
    mount_pitch_deg: float
    mount_yaw_deg: float
    lever_x_m: float
    lever_y_m: float
    lever_z_m: float
    elev_fov_deg: float
    horiz_half_deg: float

    @classmethod
    def from_json(cls, data: dict[str, object]) -> EpLaserCalib:
        if not data.get("ok"):
            raise ValueError(f"EP load failed: {data}")
        z = float(data.get("lever_z_m", 0.0))
        if abs(z) < 1e-6:
            z = float(data.get("lever_z_alt_m", 0.0)) if "lever_z_alt_m" in data else 0.0
        return cls(
            mount_roll_deg=float(data["mount_roll_deg"]),
            mount_pitch_deg=float(data["mount_pitch_deg"]),
            mount_yaw_deg=float(data["mount_yaw_deg"]),
            lever_x_m=float(data["lever_x_m"]),
            lever_y_m=float(data["lever_y_m"]),
            lever_z_m=z,
            elev_fov_deg=float(data["vert_fov_deg"]),
            horiz_half_deg=float(data["horiz_half_deg"]),
        )


def ep_dump_exe() -> Path:
    repo = Path(__file__).resolve().parents[3]
    return repo / "tools" / "ltp_decode" / "ltp_ep_dump.exe"


def load_ep_calib(ep_path: Path, *, dump_exe: Path | None = None) -> EpLaserCalib:
    """Parse .EP through CoPre LTLaser.dll helper (Windows + CoPre install)."""
    ep_path = ep_path.resolve()
    if not ep_path.is_file():
        raise FileNotFoundError(ep_path)
    exe = (dump_exe or ep_dump_exe()).resolve()
    if not exe.is_file():
        raise FileNotFoundError(
            f"EP dump tool not found: {exe} (build tools/ltp_decode/build_ep_dump.bat)"
        )
    proc = subprocess.run(
        [str(exe), str(ep_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"ltp_ep_dump failed ({proc.returncode}): {proc.stderr or proc.stdout}"
        )
    return EpLaserCalib.from_json(json.loads(proc.stdout))


def apply_mount_rpy(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    calib: EpLaserCalib,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotate laser-frame offsets into vehicle body using EP mount angles."""
    roll, pitch, yaw = np.deg2rad(
        [calib.mount_roll_deg, calib.mount_pitch_deg, calib.mount_yaw_deg]
    )
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
    xb = r00 * x + r01 * y + r02 * z
    yb = r10 * x + r11 * y + r12 * z
    zb = r20 * x + r21 * y + r22 * z
    return xb, yb, zb


def body_vector_to_enu(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    yaw: np.ndarray,
    roll: np.ndarray | None = None,
    pitch: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map body-frame vectors (X right, Y forward, Z up) to ENU deltas."""
    sin_y, cos_y = np.sin(yaw), np.cos(yaw)
    if roll is None and pitch is None:
        de = y * sin_y + x * cos_y
        dn = y * cos_y - x * sin_y
        du = z
        return de, dn, du

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

    de = r00 * x + r01 * y + r02 * z
    dn = r10 * x + r11 * y + r12 * z
    du = r20 * x + r21 * y + r22 * z
    return de, dn, du


def lever_offset_enu(
    calib: EpLaserCalib,
    yaw: np.ndarray,
    roll: np.ndarray | None = None,
    pitch: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """EP lever arm (POS/IMU origin → laser) rotated into ENU."""
    lx = np.full_like(yaw, calib.lever_x_m, dtype=np.float64)
    ly = np.full_like(yaw, calib.lever_y_m, dtype=np.float64)
    lz = np.full_like(yaw, calib.lever_z_m, dtype=np.float64)
    return body_vector_to_enu(lx, ly, lz, yaw, roll=roll, pitch=pitch)
