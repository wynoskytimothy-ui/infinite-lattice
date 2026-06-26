#!/usr/bin/env python3
"""Compare pre-electron lattice compression vs full stack."""

from __future__ import annotations

import sys
import time
import zlib
import bz2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lattice_retriever_v1.branch_order_codec import decode_branch_order, encode_branch_order
from lattice_retriever_v1.lattice_compressor import LatticeCompressor


def bench(data: bytes, label: str, *, rounds: int = 100) -> None:
    w, m = encode_branch_order(data)
    t0 = time.perf_counter()
    for _ in range(rounds):
        encode_branch_order(data)
    enc_ms = (time.perf_counter() - t0) / rounds * 1000

    t0 = time.perf_counter()
    for _ in range(rounds):
        decode_branch_order(w)
    dec_ms = (time.perf_counter() - t0) / rounds * 1000

    pre_ok = not m.get("branch_order_fallback")
    pre_bytes = len(w) if pre_ok else 0
    pre_mode = m.get("mode", "walker_fallback")

    c = LatticeCompressor()
    t0 = time.perf_counter()
    r = c.compress(data, promote=False)
    full_ms = (time.perf_counter() - t0) * 1000
    r2 = c.recompress(data)
    z = len(zlib.compress(data, 9))
    bz = len(bz2.compress(data, 9))

    print(f"=== {label} (raw {len(data):,} B) ===")
    if pre_ok:
        print(
            f"  PRE-ELECTRON: {pre_mode} | {pre_bytes} B | "
            f"{len(data) / pre_bytes:.0f}x | enc {enc_ms:.3f}ms dec {dec_ms:.3f}ms"
        )
    else:
        print(f"  PRE-ELECTRON: branch fallback -> walker")
    print(
        f"  FULL picker: {r.mode} | {r.wire_bytes} B | {r.ratio:.1f}x | "
        f"compress {full_ms:.0f}ms"
    )
    print(
        f"  SESSION: {r2.mode} | {r2.wire_bytes} B | "
        f"{len(data) / r2.wire_bytes:.0f}x"
    )
    print(f"  zlib {z} B ({len(data) / z:.0f}x)  bz2 {bz} B ({len(data) / bz:.0f}x)")
    print()


def main() -> None:
    bench(bytes([42]) * 100_000, "single 100KB")
    bench(bytes([1, 2]) * 50_000, "two-branch 100KB")
    bench(bytes([42]) * 10_000, "single 10KB")
    bench(bytes([0, 1, 2, 3]) * 2500, "4-symbol 10KB", rounds=10)
    bench(bytes(range(10)) * 1000, "10-digit 10KB", rounds=3)


if __name__ == "__main__":
    main()
