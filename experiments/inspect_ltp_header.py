"""Inspect CHCNAV .ltp laser raw file header (proprietary)."""
from __future__ import annotations

import struct
import sys
from pathlib import Path


def inspect(path: Path) -> None:
    size = path.stat().st_size
    with path.open("rb") as f:
        hdr = f.read(4096)

    print(f"file: {path}")
    print(f"size: {size:,} bytes ({size / 1e9:.3f} GB)")
    print(f"magic: {hdr[0:4]!r}")
    print(f"version field @8: {hdr[8:12]!r}")
    print(f"scanner @16: {hdr[16:24]!r}")
    print(f"serial @32: {hdr[32:48]!r}")
    print(f"firmware @64: {hdr[64:80]!r}")
    print(f"software @80: {hdr[80:96]!r}")

    print("\nNon-zero u32 fields in first 512 bytes:")
    for off in range(0, 512, 4):
        val = struct.unpack_from("<I", hdr, off)[0]
        if val:
            fval = struct.unpack_from("<f", hdr, off)[0]
            print(f"  off {off:4d}: u32={val:12d}  f32={fval:14.6g}")

    # sample data after likely header sizes
    for header_size in (256, 512, 1024, 2048, 4096):
        with path.open("rb") as f:
            f.seek(header_size)
            chunk = f.read(128)
        print(f"\nFirst 32 bytes after offset {header_size}:")
        print("  hex:", chunk[:32].hex())
        if len(chunk) >= 12:
            for rec in (12, 16, 20, 24, 28, 32):
                vals = struct.unpack_from(f"<{rec // 4}f", chunk[: rec - (rec % 4)])
                print(f"  as {rec}-byte floats: {vals[:4]}")


def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else r"c:\Users\Andrea\Desktop\2026-03-24-123712_002.ltp")
    inspect(path)


if __name__ == "__main__":
    main()
