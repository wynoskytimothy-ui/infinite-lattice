#!/usr/bin/env python3
"""Show what is constant vs scales for fixed alphabet pair-origin codec."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lattice_retriever_v1.intersection_dot_codec import encode_pair_vectors


def generate(n_bytes: int, alphabet: int, seed: int = 0) -> bytes:
    rng = random.Random(seed)
    hot = list(range(min(10, alphabet)))
    return bytes(
        rng.choice(hot) if rng.random() < 0.85 else rng.randint(0, alphabet - 1)
        for _ in range(n_bytes)
    )


def bench(n_bytes: int, alphabet: int = 100) -> dict:
    data = generate(n_bytes, alphabet)
    _, ledger, _, _ = encode_pair_vectors(data)
    index_only = ledger.alphabet_bytes + ledger.header_bytes
    return {
        "raw_mb": n_bytes / 1_000_000,
        "index_bytes": index_only,
        "origins_max": ledger.n_pair_origins_max,
        "origins_used": ledger.n_pair_origins_used,
        "walk_bytes": ledger.walk_bytes,
        "total_bytes": ledger.total_bytes,
        "max_pair_n": ledger.max_pair_n,
    }


def main() -> None:
    alphabet = 100
    print(f"Fixed alphabet = {alphabet} symbols\n")
    print("INDEX LAYER (formula + alphabet — constant for fixed vocabulary)")
    print(f"{'raw':>8}  {'index_B':>8}  {'origins_max':>12}  {'origins_used':>13}  {'max_n':>8}")
    rows = []
    for mb in (1, 10, 100):
        r = bench(mb * 1_000_000, alphabet)
        rows.append(r)
        print(
            f"{r['raw_mb']:7.0f}MB  {r['index_bytes']:8}  "
            f"{r['origins_max']:12}  {r['origins_used']:13}  {r['max_pair_n']:8}"
        )

    print("\nFULL LOSSLESS CORPUS (walk of pair dots — scales with data size)")
    print(f"{'raw':>8}  {'codec_MB':>10}  {'walk_MB':>10}")
    for r in rows:
        print(
            f"{r['raw_mb']:7.0f}MB  {r['total_bytes']/1e6:10.2f}  {r['walk_bytes']/1e6:10.2f}"
        )

    r1 = rows[0]
    per_sym = r1["walk_bytes"] / 1_000_000
    est_80gb = r1["index_bytes"] + per_sym * 80e9
    print(f"\n80 GB extrapolation: index stays {r1['index_bytes']} B; walk ~{est_80gb/1e9:.1f} GB")
    print(f"  (~{per_sym:.3f} B per raw symbol in walk)")


if __name__ == "__main__":
    main()
