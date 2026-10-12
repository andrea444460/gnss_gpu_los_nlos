"""Compact on-disk pack for per-way GNSS quality timelines.

Design goals
------------
- Small files for city-scale networks (hours–day of samples).
- Store **run-length / change-points only** after HDOP/nLOS quantization —
  not one row per (way, epoch).
- Quantize floats to bytes (HDOP×10, n_LOS×2).
- Gzip the binary payload.

File layout (``.rqz`` = gzip of raw bytes)
-----------------------------------------
::

    magic          : 8 bytes  b"RQPACK01"
    header_len     : uint32 LE
    header_json    : UTF-8 JSON (metadata)
    for each way:
        way_id     : uint32 LE
        n_runs     : uint16 LE
        runs       : n_runs × (t_idx:uint16, hdop_q:uint8, nlos_q:uint8)

``t_idx`` is ``round((t_s - t0_s) / dt_s)`` clipped to uint16.
"""

from __future__ import annotations

import gzip
import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from gnss_gpu.routing_graph import quantize_quality

MAGIC = b"RQPACK01"
_HDR_LEN = struct.Struct("<I")
_WAY_HDR = struct.Struct("<IH")  # way_id, n_runs
_RUN = struct.Struct("<HBB")  # t_idx, hdop_q, nlos_q


def encode_hdop(hdop: float) -> int:
    if hdop != hdop:  # NaN
        return 255
    return max(0, min(254, int(round(float(hdop) * 10.0))))


def decode_hdop(q: int) -> float:
    if int(q) >= 255:
        return float("nan")
    return float(q) / 10.0


def encode_n_los(n_los: float) -> int:
    if n_los != n_los:
        return 255
    return max(0, min(254, int(round(float(n_los) * 2.0))))


def decode_n_los(q: int) -> float:
    if int(q) >= 255:
        return float("nan")
    return float(q) / 2.0


@dataclass
class WayQualityRuns:
    way_id: int
    # (t_s, hdop, n_los) at run starts (decoded)
    runs: list[tuple[float, float, float]] = field(default_factory=list)


@dataclass
class RoadQualityPack:
    """In-memory compact quality pack."""

    bbox: list[float]
    t0_s: float
    horizon_s: float
    dt_s: float
    hdop_step: float
    n_los_step: float
    ways: list[WayQualityRuns]
    source: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def n_ways(self) -> int:
        return len(self.ways)

    @property
    def n_runs(self) -> int:
        return sum(len(w.runs) for w in self.ways)

    def to_samples_dict(self) -> dict[int, list[tuple[float, float, float]]]:
        """Expand to the dict form used by routing / GUI (change points only)."""
        out: dict[int, list[tuple[float, float, float]]] = {}
        for w in self.ways:
            if not w.runs:
                continue
            # Append a sentinel at horizon so last run has a visible end for scrubbers.
            seq = list(w.runs)
            t_end = float(self.t0_s + self.horizon_s)
            if seq[-1][0] < t_end - 1e-9:
                seq.append((t_end, seq[-1][1], seq[-1][2]))
            out[int(w.way_id)] = seq
        return out


def collapse_samples_to_runs(
    samples: list[tuple[float, float, float]],
    *,
    hdop_step: float,
    n_los_step: float,
) -> list[tuple[float, float, float]]:
    """Keep only times where the quantized quality key changes."""
    if not samples:
        return []
    ordered = sorted(samples, key=lambda x: x[0])
    runs: list[tuple[float, float, float]] = []
    t0, h0, n0 = ordered[0]
    key = quantize_quality(h0, n0, hdop_step, n_los_step)
    acc_h = [h0]
    acc_n = [n0]
    for t, h, n in ordered[1:]:
        k = quantize_quality(h, n, hdop_step, n_los_step)
        if k != key:
            runs.append(
                (
                    float(t0),
                    float(sum(acc_h) / len(acc_h)),
                    float(sum(acc_n) / len(acc_n)),
                )
            )
            t0, h0, n0 = t, h, n
            key = k
            acc_h = [h]
            acc_n = [n]
        else:
            acc_h.append(h)
            acc_n.append(n)
    runs.append(
        (
            float(t0),
            float(sum(acc_h) / len(acc_h)),
            float(sum(acc_n) / len(acc_n)),
        )
    )
    return runs


def pack_from_samples(
    samples_by_way: dict[int, list[tuple[float, float, float]]],
    *,
    bbox: list[float],
    t0_s: float,
    horizon_s: float,
    dt_s: float,
    hdop_step: float = 0.5,
    n_los_step: float = 1.0,
    source: str = "",
    meta: dict | None = None,
) -> RoadQualityPack:
    ways: list[WayQualityRuns] = []
    for wid in sorted(samples_by_way):
        runs = collapse_samples_to_runs(
            samples_by_way[wid], hdop_step=hdop_step, n_los_step=n_los_step
        )
        if runs:
            ways.append(WayQualityRuns(way_id=int(wid), runs=runs))
    return RoadQualityPack(
        bbox=[float(x) for x in bbox],
        t0_s=float(t0_s),
        horizon_s=float(horizon_s),
        dt_s=float(dt_s),
        hdop_step=float(hdop_step),
        n_los_step=float(n_los_step),
        ways=ways,
        source=source,
        meta=dict(meta or {}),
    )


def write_road_quality_pack(path: Path | str, pack: RoadQualityPack) -> Path:
    """Write gzip-compressed compact pack (``.rqz``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "version": 1,
        "bbox": pack.bbox,
        "t0_s": pack.t0_s,
        "horizon_s": pack.horizon_s,
        "dt_s": pack.dt_s,
        "hdop_step": pack.hdop_step,
        "n_los_step": pack.n_los_step,
        "n_ways": pack.n_ways,
        "n_runs": pack.n_runs,
        "encoding": "rle_change_q8",
        "source": pack.source,
        "meta": pack.meta,
    }
    hdr = json.dumps(header, separators=(",", ":")).encode("utf-8")
    chunks: list[bytes] = [MAGIC, _HDR_LEN.pack(len(hdr)), hdr]
    dt = max(1e-9, float(pack.dt_s))
    t0 = float(pack.t0_s)
    for w in pack.ways:
        runs = w.runs
        if len(runs) > 65535:
            raise ValueError(f"way {w.way_id}: too many runs ({len(runs)})")
        chunks.append(_WAY_HDR.pack(int(w.way_id) & 0xFFFFFFFF, len(runs)))
        for t_s, hdop, n_los in runs:
            t_idx = int(round((float(t_s) - t0) / dt))
            t_idx = max(0, min(65535, t_idx))
            chunks.append(
                _RUN.pack(t_idx, encode_hdop(hdop), encode_n_los(n_los))
            )
    raw = b"".join(chunks)
    with gzip.open(path, "wb", compresslevel=9) as f:
        f.write(raw)
    return path


def read_road_quality_pack(path: Path | str) -> RoadQualityPack:
    path = Path(path)
    with gzip.open(path, "rb") as f:
        raw = f.read()
    if len(raw) < 12 or raw[:8] != MAGIC:
        raise ValueError(f"not a road quality pack: {path}")
    (hlen,) = _HDR_LEN.unpack_from(raw, 8)
    off = 12
    header = json.loads(raw[off : off + hlen].decode("utf-8"))
    off += hlen
    dt = float(header["dt_s"])
    t0 = float(header["t0_s"])
    ways: list[WayQualityRuns] = []
    n_ways = int(header.get("n_ways", 0))
    for _ in range(n_ways):
        if off + _WAY_HDR.size > len(raw):
            break
        wid, n_runs = _WAY_HDR.unpack_from(raw, off)
        off += _WAY_HDR.size
        runs: list[tuple[float, float, float]] = []
        for _j in range(n_runs):
            t_idx, hq, nq = _RUN.unpack_from(raw, off)
            off += _RUN.size
            runs.append((t0 + t_idx * dt, decode_hdop(hq), decode_n_los(nq)))
        ways.append(WayQualityRuns(way_id=int(wid), runs=runs))
    return RoadQualityPack(
        bbox=[float(x) for x in header.get("bbox", [0, 0, 0, 0])],
        t0_s=t0,
        horizon_s=float(header.get("horizon_s", 0.0)),
        dt_s=dt,
        hdop_step=float(header.get("hdop_step", 0.5)),
        n_los_step=float(header.get("n_los_step", 1.0)),
        ways=ways,
        source=str(header.get("source", "")),
        meta=dict(header.get("meta") or {}),
    )


def naive_dense_bytes(n_ways: int, n_epochs: int) -> int:
    """Bytes of a naive CSV-like dense table (way,t,hdop,nlos as 8-byte floats)."""
    return int(n_ways) * int(n_epochs) * (8 + 8 + 8 + 8)
