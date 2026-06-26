#!/usr/bin/env python3
"""Benchmark frontier lattice compression vs classical codecs."""

from __future__ import annotations

import argparse
import bz2
import lzma
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lattice_retriever_v1.lattice_compressor import LatticeCompressor, frontier_report


def _classical(data: bytes) -> dict[str, int]:
    return {
        "zlib": len(zlib.compress(data, 9)),
        "bz2": len(bz2.compress(data, 9)),
        "lzma": len(lzma.compress(data)),
    }


def bench_one(data: bytes, label: str) -> dict:
    comp = LatticeCompressor()
    r = comp.compress(data)
    r2 = comp.recompress(data)
    classical = _classical(data)
    return {
        "label": label,
        "raw": len(data),
        "lattice_mode": r.mode,
        "lattice_bytes": r.wire_bytes,
        "lattice_ratio": round(r.ratio, 2),
        "recompress_mode": r2.mode,
        "recompress_bytes": r2.wire_bytes,
        "walker_stored": r.walker_stored,
        **{f"{k}_bytes": v for k, v in classical.items()},
        **{f"{k}_ratio": round(len(data) / v, 2) for k, v in classical.items()},
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Frontier lattice compression benchmark")
    p.add_argument("--file", type=Path, help="optional file to compress")
    p.add_argument("--size", type=int, default=100_000, help="synthetic repeat size")
    args = p.parse_args()

    samples: list[tuple[str, bytes]] = []
    if args.file and args.file.exists():
        samples.append((str(args.file), args.file.read_bytes()))
    else:
        samples = [
            ("single_symbol", bytes([42]) * args.size),
            ("two_branch", bytes([1, 2]) * (args.size // 2)),
            ("digits", bytes(range(10)) * (args.size // 10)),
        ]

    print("Frontier lattice vs classical (bytes on wire)\n")
    for label, data in samples:
        row = bench_one(data, label)
        print(f"=== {label} ===")
        for k, v in row.items():
            if k != "label":
                print(f"  {k}: {v}")
        fr = frontier_report(data[: min(len(data), 64)])
        print(f"  promotions_found: {fr['promotions_found']}")
        print()


if __name__ == "__main__":
    main()
