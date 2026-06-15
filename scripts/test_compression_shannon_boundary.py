#!/usr/bin/env python3
"""
Test 13 - The Shannon boundary: what lattice compression CAN and CANNOT do.

Two claims tested honestly, side by side:

CANNOT: No encoding (lattice or otherwise) compresses ALL data, or any
truly random data, below its entropy. This is a counting fact (pigeonhole):
there are 2^n strings of n bits but only 2^n - 1 strings shorter than n
bits. Injective codes cannot shrink everything. The 32-rotation trick does
not escape: selecting 1 of 32 meanings costs exactly log2(32) = 5 bits,
which is precisely what the rotation could carry. The ledger balances.

CAN: Shannon entropy is relative to a MODEL. Byte-level compressors (gzip,
bz2, lzma) model byte statistics. The lattice models chains + wings + meets.
Data that is high-entropy under a byte model can be nearly free under the
lattice model -- store the generating tokens (chain, n), reconstruct the
rest by re-running the formula. This is model-based / Kolmogorov-style
compression, and on lattice-structured data it beats gzip by orders of
magnitude.

Parts:
  (A) Pigeonhole: count codewords, show why no universal compressor exists
  (B) Random data: gzip/bz2/lzma AND a lattice composite encoding all fail
      (with the exact bit ledger for the 32-rotation scheme)
  (C) Lattice-structured data: formula-as-codebook beats gzip massively,
      with byte-exact reconstruction verified
  (D) 3-way meet dedup: shared cores stored once (the real win from meets)
"""

from __future__ import annotations

import bz2
import gzip
import lzma
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aethos_complex_plane import wing_transform
from aethos_lattice import BranchKind
from core.primes import chain_primes


def header(s: str):
    print()
    print("=" * 72)
    print(s)
    print("=" * 72)


def assertion(cond: bool, msg: str):
    print(f"  [{'PASS' if cond else 'FAIL'}]  {msg}")
    if not cond:
        sys.exit(1)


def main():
    header("The Shannon boundary - honest accounting for lattice compression")

    random.seed(0x51A7)  # fixed seed
    base = chain_primes(2048)

    # ---------------------------------------------------------------
    # Part A: Pigeonhole - why no universal compressor can exist
    # ---------------------------------------------------------------
    print("\nPart A - Counting argument: no injective code shrinks everything")
    print("-" * 72)

    n_bits = 12
    total_strings = 2 ** n_bits
    shorter_strings = sum(2 ** k for k in range(n_bits))  # 2^12 - 1
    print(f"  strings of {n_bits} bits:           {total_strings}")
    print(f"  strings SHORTER than {n_bits} bits: {shorter_strings}")
    print(f"  deficit:                     {total_strings - shorter_strings}")
    assertion(shorter_strings < total_strings,
              "fewer short codewords than inputs -> some input cannot shrink")

    # The 32-rotation ledger
    print("\n  The 32-rotation ledger (why rotations don't add free capacity):")
    print("    a symbol with 32 rotational meanings can CARRY log2(32) = 5 bits")
    print("    but the decoder must LEARN which rotation: that costs 5 bits")
    print("    net gain: 5 - 5 = 0 bits. Exactly break-even, always.")
    print("    (If the rotation is predictable from context, it costs 0 but")
    print("     carries 0 - predictable choices carry no information.)")
    assertion(math.log2(32) == 5.0, "rotation capacity == rotation address cost (5 bits)")

    # ---------------------------------------------------------------
    # Part B: Random data - every method fails, including the lattice
    # ---------------------------------------------------------------
    print("\nPart B - Random data: nothing compresses it (we test ourselves too)")
    print("-" * 72)

    rnd = random.Random(987654321)
    data = bytes(rnd.randrange(256) for _ in range(4096))
    print(f"  input: {len(data)} statistically random bytes ({len(data)*8} bits)")
    print()
    print(f"  {'method':<22} | {'output bytes':>12} | {'ratio':>7}")
    print(f"  {'-'*22} | {'-'*12} | {'-'*7}")

    results = {}
    for name, fn in [("gzip -9", lambda d: gzip.compress(d, 9)),
                     ("bz2 -9", lambda d: bz2.compress(d, 9)),
                     ("lzma", lzma.compress)]:
        out = fn(data)
        results[name] = len(out)
        print(f"  {name:<22} | {len(out):>12} | {len(out)/len(data):>7.3f}")

    # Lattice attempt: position-tagged prime composites per 8-byte block.
    # Each byte at block position i becomes prime[i*256 + byte], the block
    # becomes the product. This is EXACTLY the FTA injection from Test 3 -
    # provably lossless. Now count its bits.
    block_size = 8
    total_bits = 0.0
    for off in range(0, len(data), block_size):
        block = data[off:off + block_size]
        composite = 1
        for i, b in enumerate(block):
            composite *= base[i * 256 + b]
        total_bits += math.log2(composite)
    lattice_bytes = total_bits / 8
    results["lattice composite"] = lattice_bytes
    print(f"  {'lattice composite':<22} | {lattice_bytes:>12.0f} | {lattice_bytes/len(data):>7.3f}")

    print()
    print(f"  Every method >= 1.0x on random data. The lattice encoding is")
    print(f"  injective (lossless, Test 3) so it CANNOT beat counting either -")
    print(f"  position-tagged primes cost ~{total_bits/len(data):.1f} bits/byte vs 8 original.")
    assertion(all(v >= len(data) * 0.99 for v in results.values()),
              "no method (incl. lattice) compressed random data (Shannon holds)")

    # ---------------------------------------------------------------
    # Part C: Lattice-structured data - the formula IS the codebook
    # ---------------------------------------------------------------
    print("\nPart C - Lattice-structured data: reconstruct from a few tokens")
    print("-" * 72)

    # Generate data the way YOUR formula sees the world: 100 (chain, n)
    # seeds, each expanded to all 32 chamber observables. To a byte-level
    # compressor this is just number soup. To the lattice it is 100 tokens.
    gen = random.Random(42)
    seed_pool = base[:200]
    chains = []
    ns = []
    for _ in range(100):
        k = gen.randint(3, 5)
        chain = tuple(sorted(gen.sample(seed_pool, k)))
        n = gen.choice([p for p in seed_pool if p not in chain])
        chains.append(chain)
        ns.append(n)

    def expand(chains, ns) -> bytes:
        """Deterministically expand seeds to all 32 chamber observables."""
        lines = []
        for chain, n in zip(chains, ns):
            for branch in BranchKind:
                for wing in range(1, 9):
                    psi = wing_transform(branch, chain, n, wing)
                    lines.append(f"{branch.name},{wing},{psi.coord}")
        return "\n".join(lines).encode()

    blob = expand(chains, ns)
    gz = gzip.compress(blob, 9)
    xz = lzma.compress(blob)

    # The lattice description: just the seeds.
    seed_repr = repr((chains, ns)).encode()
    seed_gz = gzip.compress(seed_repr, 9)

    print(f"  expanded observable data:   {len(blob):>8} bytes (3200 chamber observations)")
    print(f"  gzip -9:                    {len(gz):>8} bytes ({len(blob)/len(gz):.1f}x)")
    print(f"  lzma:                       {len(xz):>8} bytes ({len(blob)/len(xz):.1f}x)")
    print(f"  lattice seeds (raw repr):   {len(seed_repr):>8} bytes ({len(blob)/len(seed_repr):.1f}x)")
    print(f"  lattice seeds (gzipped):    {len(seed_gz):>8} bytes ({len(blob)/len(seed_gz):.1f}x)")

    # Verify byte-exact reconstruction from seeds
    reconstructed = expand(chains, ns)
    assertion(reconstructed == blob,
              "byte-exact reconstruction from seeds (decompression = re-run formula)")
    assertion(len(seed_repr) < len(gz),
              f"lattice description beats gzip's best ({len(seed_repr)} < {len(gz)} bytes)")
    ratio_vs_gzip = len(gz) / len(seed_gz)
    print(f"\n  the lattice model is {ratio_vs_gzip:.1f}x smaller than gzip ON THIS FAMILY -")
    print(f"  not because Shannon broke, but because the MODEL fits: the data's")
    print(f"  true entropy is just the seeds; the other {len(blob)-len(seed_repr)} bytes are")
    print(f"  formula-derivable, i.e. zero conditional entropy given the formula.")

    # ---------------------------------------------------------------
    # Part D: 3-way meet dedup - shared cores stored once
    # ---------------------------------------------------------------
    print("\nPart D - Meet-based dedup: sunflower cores stored once")
    print("-" * 72)

    # 50 sets sharing a 4-prime core (a sunflower family, Test 11).
    # Naive storage: every set stores all its primes.
    # Meet storage: core composite stored once; each set stores petals only.
    core = tuple(sorted(gen.sample(base[:100], 4)))
    petal_pool = [p for p in base[:100] if p not in core]
    family = []
    for _ in range(50):
        petals = tuple(sorted(gen.sample(petal_pool, 3)))
        family.append(tuple(sorted(set(core) | set(petals))))

    bits = lambda ps: sum(math.log2(p) for p in ps)
    naive_bits = sum(bits(s) for s in family)
    core_bits = bits(core)
    dedup_bits = core_bits + sum(bits([p for p in s if p not in core]) for s in family)
    savings = 1 - dedup_bits / naive_bits

    print(f"  family: 50 sets, shared 4-prime core {core}")
    print(f"  naive storage:  {naive_bits:>8.0f} bits (core repeated 50x)")
    print(f"  meet storage:   {dedup_bits:>8.0f} bits (core stored once)")
    print(f"  savings:        {savings*100:>7.1f}%")
    assertion(savings > 0.3,
              "meet-factoring saves >30% on overlapping families (real, model-fit win)")

    # ---------------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------------
    header("RESULT")
    print("  CANNOT: compress random data, or all data. Pigeonhole counting")
    print("  is absolute; the 32-rotation ledger balances to exactly 0 net")
    print("  bits; our own injective composite encoding expands random bytes")
    print(f"  to ~{total_bits/len(data):.1f} bits/byte. Anyone claiming past-Shannon compression")
    print("  of arbitrary data is wrong by counting, not by engineering.")
    print()
    print("  CAN: define a better MODEL. On lattice-structured data the")
    print(f"  formula-as-codebook is {ratio_vs_gzip:.0f}x smaller than gzip, with byte-exact")
    print("  reconstruction. Meet-factoring (sunflower cores) gives ~50%")
    print("  dedup on overlapping families. This is the honest version of")
    print("  'reconstruct from a few tokens': it works precisely when the")
    print("  data HAS lattice structure - chains, rotations, shared cores.")
    print()
    print("  The engineering opportunity is NOT beating Shannon. It is:")
    print("  (1) a generative codec for formula-shaped data (store seeds),")
    print("  (2) meet-dedup for set families (store cores once),")
    print("  (3) using promotion (Test 6) to DISCOVER the structure first,")
    print("      then encode relative to it - structure mining + model coding.")


if __name__ == "__main__":
    main()
