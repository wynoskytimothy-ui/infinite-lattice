#!/usr/bin/env python3
"""
Lattice storage codec bench — raw data compression, NOT RAG.

  python scripts/bench_storage_codec.py --sample-mb 10 --alphabet 200
  python scripts/bench_storage_codec.py --sample-mb 1 --repetitive --extrapolate-tb 100
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lattice_retriever_v1.storage_codec import encode_storage, extrapolate_ledger


def generate(n: int, alphabet: int, *, repetitive: bool, seed: int) -> bytes:
    rng = random.Random(seed)
    if repetitive:
        hot = list(range(min(10, alphabet)))
        chunk = bytes(rng.choice(hot) for _ in range(20))
        return (chunk * (n // len(chunk) + 1))[:n]
    return bytes(rng.randint(0, alphabet - 1) for _ in range(n))


def main() -> None:
    p = argparse.ArgumentParser(description="Lattice storage codec bench")
    p.add_argument("--sample-mb", type=float, default=1.0)
    p.add_argument("--alphabet", type=int, default=200)
    p.add_argument("--repetitive", action="store_true")
    p.add_argument("--extrapolate-tb", type=float, default=0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    n = int(args.sample_mb * 1_000_000)
    data = generate(n, args.alphabet, repetitive=args.repetitive, seed=args.seed)

    t0 = time.perf_counter()
    payload, ledger, patterns = encode_storage(data, min_cohesion=0.8)
    enc_ms = (time.perf_counter() - t0) * 1000

    zlib_bytes = len(zlib.compress(data, 9))
    report = ledger.explain()
    report["encode_ms"] = round(enc_ms, 1)
    report["payload_bytes"] = len(payload)
    report["zlib_bytes"] = zlib_bytes
    report["vs_zlib_ratio_x"] = round(len(data) / zlib_bytes, 3)
    report["sample_repetitive"] = args.repetitive
    report["top_patterns"] = [pat.explain() for pat in patterns[:8]]

    if args.extrapolate_tb > 0:
        target = int(args.extrapolate_tb * 1e12)
        report["extrapolation"] = extrapolate_ledger(ledger, target_bytes=target)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
