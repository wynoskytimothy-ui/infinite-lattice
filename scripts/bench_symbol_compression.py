#!/usr/bin/env python3
"""
Pure symbol compression bench — NO RAG, no promotions, no cages.

Tests the lattice compression thesis at symbol level:
  - Unique alphabet (100–200 symbols) → tiny table
  - Order stream → bit-packed indices (lossless)
  - Primes + 3D dots recompute from formula (not stored)
  - Unique bigram index size (optional intersection layer) ∝ unique pairs, not corpus size

Example:
  python scripts/bench_symbol_compression.py --sample-mb 100 --extrapolate-gb 80
  python scripts/bench_symbol_compression.py --sample-mb 50 --alphabet 100 --random
  python scripts/bench_symbol_compression.py --input path/to/raw.bin
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lattice_retriever_v1.symbol_compression import decode_bytes, encode_bytes


def generate_random(n_bytes: int, alphabet_size: int, *, seed: int = 0) -> bytes:
    import random

    rng = random.Random(seed)
    alphabet = list(range(alphabet_size))
    return bytes(rng.choice(alphabet) for _ in range(n_bytes))


def generate_repetitive(n_bytes: int, alphabet_size: int, *, seed: int = 0) -> bytes:
    """Skewed — few symbols dominate (better for meet-based locality)."""
    import random

    rng = random.Random(seed)
    hot = list(range(min(10, alphabet_size)))
    out = bytearray()
    while len(out) < n_bytes:
        if rng.random() < 0.85:
            out.append(rng.choice(hot))
        else:
            out.append(rng.randint(0, alphabet_size - 1))
    return bytes(out[:n_bytes])


def read_or_generate(args) -> tuple[bytes, str]:
    if args.input:
        p = Path(args.input)
        if args.sample_mb and p.stat().st_size > args.sample_mb * 1_000_000:
            with open(p, "rb") as f:
                return f.read(args.sample_mb * 1_000_000), f"file:{p.name} (first {args.sample_mb}MB)"
        return p.read_bytes(), f"file:{p.name}"
    n = args.sample_mb * 1_000_000
    if args.repetitive:
        data = generate_repetitive(n, args.alphabet, seed=args.seed)
        label = f"synthetic repetitive alphabet={args.alphabet}"
    else:
        data = generate_random(n, args.alphabet, seed=args.seed)
        label = f"synthetic uniform alphabet={args.alphabet}"
    return data, label


def theoretical_packed_bits(n_bytes: int, n_symbols: int) -> int:
    bits = max(1, math.ceil(math.log2(max(n_symbols, 1))))
    return (n_bytes * bits + 7) // 8


def main() -> None:
    parser = argparse.ArgumentParser(description="Pure symbol compression — no RAG")
    parser.add_argument("--sample-mb", type=int, default=100, help="Sample size in MB (default 100)")
    parser.add_argument("--extrapolate-gb", type=float, default=80.0, help="Extrapolate results to this raw GB")
    parser.add_argument("--alphabet", type=int, default=100, help="Symbol alphabet size for synthetic data")
    parser.add_argument("--random", action="store_true", default=True, help="Uniform random over alphabet")
    parser.add_argument("--repetitive", action="store_true", help="Skewed symbol distribution")
    parser.add_argument("--input", type=str, default="", help="Optional raw input file")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json-out", type=str, default="")
    args = parser.parse_args()

    t0 = time.perf_counter()
    data, label = read_or_generate(args)
    t_read = time.perf_counter() - t0

    t0 = time.perf_counter()
    payload, ledger = encode_bytes(data)
    roundtrip = decode_bytes(payload)
    t_codec = time.perf_counter() - t0
    assert roundtrip == data, "lossless roundtrip failed"

    t0 = time.perf_counter()
    zlib_bytes = zlib.compress(data, 9)
    t_zlib = time.perf_counter() - t0

    actual_symbols = ledger.n_symbols
    theo_stream = theoretical_packed_bits(len(data), actual_symbols)
    theo_total = ledger.alphabet_bytes + ledger.header_bytes + theo_stream

    extrap = ledger.extrapolate(int(args.extrapolate_gb * 1e9))
    extrap_zlib = int(len(zlib_bytes) * (args.extrapolate_gb * 1e9 / len(data)))

    print("=" * 60)
    print("PURE SYMBOL COMPRESSION (no RAG)")
    print("=" * 60)
    print(f"  source:           {label}")
    print(f"  raw sample:       {len(data)/1e6:.2f} MB  ({len(data):,} bytes)")
    print(f"  unique symbols:   {actual_symbols}  (requested {args.alphabet})")
    print(f"  unique bigrams:   {ledger.n_unique_pairs:,}")
    print()
    print("--- STORED (symbol codec — lossless) ---")
    print(f"  alphabet table:   {ledger.alphabet_bytes:,} B")
    print(f"  order stream:     {ledger.order_stream_bytes:,} B  (bit-packed indices)")
    print(f"  header:           {ledger.header_bytes:,} B")
    print(f"  TOTAL codec:      {ledger.total_symbol_codec_bytes:,} B  ({ledger.total_symbol_codec_bytes/1e6:.2f} MB)")
    print(f"  ratio:            {ledger.ratio:.3f}x   savings {ledger.savings_pct:.1f}%")
    print()
    print("--- NOT STORED (formula recomputes) ---")
    print(f"  prime assignment: {ledger.prime_table_bytes:,} B equivalent (computed from alphabet index)")
    print(f"  3D dot placement: 0 B per dot (regenerated on read)")
    print()
    print("--- OPTIONAL intersection index (unique pairs only) ---")
    print(f"  unique pair keys: {ledger.unique_pair_index_bytes:,} B  ({ledger.n_unique_pairs:,} pairs × 8 B)")
    print(f"  NOT per-occurrence — does not grow with each dot in doc")
    print()
    print("--- BASELINE ---")
    print(f"  zlib -9:          {len(zlib_bytes):,} B  ({len(zlib_bytes)/1e6:.2f} MB)  ratio {len(data)/len(zlib_bytes):.2f}x")
    print()
    print(f"--- EXTRAPOLATED to {args.extrapolate_gb:.0f} GB raw ---")
    print(f"  symbol codec:     {extrap.total_symbol_codec_bytes/1e9:.3f} GB  ({extrap.ratio:.2f}x)")
    print(f"  order stream:     {extrap.order_stream_bytes/1e9:.3f} GB")
    print(f"  alphabet+header:  {(extrap.alphabet_bytes+extrap.header_bytes)/1e3:.1f} KB  (constant)")
    print(f"  unique pair idx:  {ledger.unique_pair_index_bytes/1e6:.2f} MB  (from sample, ~constant if alphabet fixed)")
    print(f"  zlib -9 (est):    {extrap_zlib/1e9:.3f} GB")
    print()
    print(f"  theoretical packed stream @ {actual_symbols} symbols: {theo_stream/1e6:.2f} MB")
    print(f"  timing: read/gen {t_read:.2f}s  codec {t_codec:.3f}s  zlib {t_zlib:.3f}s")
    print()
    print("NOTE: 80 GB with '100 tokens' means 100 unique SYMBOL TYPES in the alphabet,")
    print("      not 100 bytes total. Order stream still scales with corpus length,")
    print("      but as log2(alphabet) bits per symbol, not 8 bits per byte.")

    out = {
        "label": label,
        "sample_bytes": len(data),
        **ledger.explain(),
        "extrapolate_gb": args.extrapolate_gb,
        "extrapolated": extrap.explain(),
        "zlib_bytes": len(zlib_bytes),
        "extrapolated_zlib_gb": extrap_zlib / 1e9,
        "lossless_roundtrip": True,
    }
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"  json: {args.json_out}")


if __name__ == "__main__":
    main()
