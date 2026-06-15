#!/usr/bin/env python3
"""
Test 20 - Token alphabet at scale: does Test 16's negative flip positive?

Test 16 found the online promotion token alphabet LOSES at 64KB (4.034 vs
2.870 bits/byte): promoted tokens pay cold first-use costs and fragment
context statistics before they amortize. The stated hypothesis: like BPE
vocabularies in LLMs, the alphabet upgrade should pay at larger scale.

This test runs the UNCHANGED Test 16 token codec and the UNCHANGED Test 15
byte codec on 64KB, 256KB, and the full repo markdown corpus (~1.4MB),
tracking the gap. bz2 -9 included for reference.

Whatever the outcome, it's recorded: either the crossover exists (scaling
law confirmed) or the token approach needs more than scale (deeper fix).
"""

from __future__ import annotations

import bz2
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_lattice_context_compressor as byte_codec
import test_promotion_subword_codec as token_codec

ROOT = Path(__file__).resolve().parents[1]


def header(s: str):
    print()
    print("=" * 72)
    print(s)
    print("=" * 72)


def assertion(cond: bool, msg: str):
    print(f"  [{'PASS' if cond else 'FAIL'}]  {msg}")
    if not cond:
        sys.exit(1)


def corpus() -> bytes:
    files = sorted(ROOT.glob("derivations/*.md")) + \
            sorted(ROOT.glob("book/**/*.md")) + \
            sorted(ROOT.glob("*.md"))
    return b"".join(f.read_bytes() for f in files)


def main():
    header("Token alphabet at scale - Test 16's hypothesis on trial")

    full = corpus()
    sizes = [65536, 262144, len(full)]
    print(f"  corpus: {len(full)} bytes of repo markdown")
    print(f"  scales: {', '.join(f'{s//1024}KB' for s in sizes)}")

    print(f"\n  {'scale':>7} | {'bz2 -9':>14} | {'byte codec':>14} | "
          f"{'token codec':>14} | {'token stats':<26}")
    print(f"  {'-'*7} | {'-'*14} | {'-'*14} | {'-'*14} | {'-'*26}")

    gaps = []
    last_token_blob = None
    last_data = None
    for size in sizes:
        data = full[:size]
        n = len(data)

        bz = len(bz2.compress(data, 9))

        t0 = time.time()
        b_blob, _, _ = byte_codec.compress(data)
        t_byte = time.time() - t0

        t0 = time.time()
        t_blob, stats = token_codec.compress(data)
        t_tok = time.time() - t0
        last_token_blob, last_data = t_blob, data

        gap = len(t_blob) / len(b_blob)
        gaps.append(gap)
        print(f"  {n//1024:>5}KB | {bz:>7} ({bz*8/n:.3f}) | "
              f"{len(b_blob):>7} ({len(b_blob)*8/n:.3f}) | "
              f"{len(t_blob):>7} ({len(t_blob)*8/n:.3f}) | "
              f"vocab {stats['vocab']}, {stats['avg_token_len']:.2f} B/tok")

    # Verify the token codec still round-trips at full scale
    print()
    t0 = time.time()
    restored = token_codec.decompress(last_token_blob, len(last_data))
    assertion(restored == last_data,
              f"token codec round-trip byte-exact at {len(last_data)//1024}KB "
              f"({time.time()-t0:.1f}s)")

    # The scaling trend
    print(f"\n  token/byte size ratio by scale: " +
          " -> ".join(f"{g:.3f}" for g in gaps))
    improving = all(gaps[i] > gaps[i + 1] for i in range(len(gaps) - 1))
    crossed = gaps[-1] < 1.0

    header("RESULT")
    if crossed:
        print("  CROSSOVER CONFIRMED: the token alphabet beats the byte codec")
        print(f"  at {sizes[-1]//1024}KB (ratio {gaps[-1]:.3f}).")
    elif improving:
        print("  TREND CONFIRMED, crossover not yet reached: the token codec")
        print(f"  closes the gap monotonically ({gaps[0]:.2f} -> {gaps[-1]:.2f})")
        print(f"  but still trails the byte codec at {sizes[-1]//1024}KB.")
        print("  The scaling hypothesis holds directionally; the crossover")
        print("  (if it exists) needs more data or a larger vocab cap.")
    else:
        print("  HYPOTHESIS REJECTED at these scales: the gap does not close")
        print("  monotonically. The token approach needs a deeper fix")
        print("  (stable tokenization, pre-mined vocab) - not just scale.")
    print()
    print("  Note: Test 19's chamber mixer v2 (2.196 bits/byte at 256KB)")
    print("  remains the best result regardless - chambers over a byte")
    print("  alphabet currently dominate alphabet upgrades at these scales.")
    assertion(True, "scaling experiment recorded")


if __name__ == "__main__":
    main()
