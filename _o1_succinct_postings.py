#!/usr/bin/env python3
"""SUCCINCT POSTINGS lens: can a tighter coder beat the current gamma/varint doc-id codec (the proven
~153-195 B/doc serve footprint is partly doc-ids, partly weights)? Exact bit-accounting of the doc-id
portion across ALL terms of the real 50k SPLADE index, four coders:
  - Elias gamma   (the current best per the bitpack synthesis)
  - Elias-Fano    (the succinct, WAND-friendly monotone-set coder -- near-optimal for sorted ids)
  - FOR + bitpack (frame-of-reference fixed-width on gaps)
  - Roaring       (pyroaring serialized size, if installed)
Plus decode throughput (postings/s) for gamma vs Elias-Fano. Measures bits/posting and the doc-id B/doc;
honest about whether anything beats gamma at equal-or-faster decode."""
import os, sys, time, math
os.environ["WORK"] = r"C:\Users\wynos\trng\marco_data\splade_native"   # 50k calib (fast)
os.environ.setdefault("PYTHONUTF8", "1")
import numpy as np
import marco_splade_native as m


def gamma_bits(gaps):
    """Elias gamma on gaps>=1: 2*floor(log2(g))+1 bits each. Vectorized."""
    g = gaps.astype(np.int64)
    g[g < 1] = 1
    return int((2 * np.floor(np.log2(g)).astype(np.int64) + 1).sum())


def ef_bits(sorted_ids, U):
    """Exact Elias-Fano size for a sorted int sequence in [0,U): n*l low bits + (n + U>>l) high bits."""
    n = len(sorted_ids)
    if n == 0:
        return 0
    l = max(0, int(math.floor(math.log2(U / n)))) if U > n else 0
    high = n + (U >> l)
    low = n * l
    return high + low


def for_bits(gaps):
    """Frame-of-reference fixed width on the gaps: ceil(log2(max_gap+1)) bits each."""
    if len(gaps) == 0:
        return 0
    w = max(1, int(math.ceil(math.log2(int(gaps.max()) + 1))))
    return len(gaps) * w


def pfor_bits(gaps, pct=90):
    """PForDelta-ish: bitpack to the p90 gap width, store exceptions at 32 bits."""
    if len(gaps) == 0:
        return 0
    thr = np.percentile(gaps, pct)
    w = max(1, int(math.ceil(math.log2(max(2, thr + 1)))))
    n_exc = int((gaps > (1 << w) - 1).sum())
    return len(gaps) * w + n_exc * 32


# ---- real gamma + Elias-Fano encode/decode for a decode-speed measurement ----
def gamma_encode(gaps):
    bits = []
    for g in gaps:
        g = int(g) if g >= 1 else 1
        nb = g.bit_length()
        bits.extend([0] * (nb - 1))
        for i in range(nb - 1, -1, -1):
            bits.append((g >> i) & 1)
    return np.array(bits, np.uint8)


def gamma_decode(bits, n):
    out = np.empty(n, np.int64)
    i = 0; k = 0
    L = len(bits)
    while k < n and i < L:
        z = 0
        while i < L and bits[i] == 0:
            z += 1; i += 1
        val = 1
        i += 1  # the leading 1
        for _ in range(z):
            val = (val << 1) | int(bits[i]); i += 1
        out[k] = val; k += 1
    return out


def main():
    print("=" * 80)
    print("SUCCINCT POSTINGS -- doc-id coder shootout on the real 50k SPLADE index")
    print("=" * 80)
    t0 = time.perf_counter()
    si = m.ServedIndex()
    n_docs = si.n_docs
    # tloc holds LOCAL doc-ids in [0, len(present)) -- the universe for the succinct coders is the
    # number of present docs, NOT the global max doc-id.
    U = len(si.present)
    nterms = len(si.tloc)
    print(f"  loaded: {n_docs:,} docs, {nterms:,} terms, universe U={U:,}  ({time.perf_counter()-t0:.1f}s)\n")

    tot_post = 0
    b_gamma = b_ef = b_for = b_pfor = 0
    roaring_bytes = None
    try:
        from pyroaring import BitMap
        roaring_bytes = 0
        have_roaring = True
    except Exception:
        have_roaring = False

    sample_terms = []   # (gaps, sorted_ids) for decode-speed test
    for j in range(nterms):
        loc, _w = si.tloc[j]
        ids = np.sort(loc.astype(np.int64))
        if len(ids) == 0:
            continue
        gaps = np.diff(ids, prepend=-1)   # first gap = ids[0]-(-1); all >=1
        tot_post += len(ids)
        b_gamma += gamma_bits(gaps)
        b_ef += ef_bits(ids, U)
        b_for += for_bits(gaps)
        b_pfor += pfor_bits(gaps)
        if have_roaring:
            bm = BitMap(ids.tolist())
            roaring_bytes += len(bm.serialize())
        if len(sample_terms) < 200 and 50 <= len(ids) <= 5000:
            sample_terms.append((gaps.copy(), ids.copy()))

    def bdoc(bits):
        return bits / 8 / n_docs

    print(f"  total doc-id postings: {tot_post:,}  ({tot_post/n_docs:.1f} postings/doc)\n")
    print(f"  {'coder':<16}{'bits/posting':>14}{'doc-id B/doc':>14}{'vs gamma':>12}")
    print("  " + "-" * 54)
    base = b_gamma
    for name, bits in [("Elias gamma", b_gamma), ("Elias-Fano", b_ef),
                       ("FOR+bitpack", b_for), ("PForDelta", b_pfor)]:
        print(f"  {name:<16}{bits/tot_post:>13.2f}{bdoc(bits):>13.1f} {bits/base:>11.3f}x")
    if have_roaring:
        rb = roaring_bytes * 8
        print(f"  {'Roaring (pyroaring)':<16}{rb/tot_post:>13.2f}{roaring_bytes/n_docs:>13.1f} {rb/base:>11.3f}x")
    else:
        print(f"  Roaring: pyroaring not installed (skipped)")

    # ---- decode throughput: gamma vs Elias-Fano-low-bits proxy ----
    print("\n  DECODE throughput (postings/s, sample of {} terms):".format(len(sample_terms)))
    # gamma decode
    enc = [(gamma_encode(g), len(g)) for g, _ in sample_terms]
    t = time.perf_counter(); npost = 0
    for bits, n in enc:
        gamma_decode(bits, n); npost += n
    dt = time.perf_counter() - t
    print(f"    Elias gamma : {npost/dt/1e6:.2f} M postings/s  (pure-python ref decode)")
    # EF decode proxy: low-bits are a fixed-width read (random access!) -> just the cumsum of gaps for parity
    t = time.perf_counter(); npost = 0
    for g, ids in sample_terms:
        np.cumsum(g) - 1; npost += len(g)   # gap->id reconstruction cost parity
    dt = time.perf_counter() - t
    print(f"    (gap cumsum): {npost/dt/1e6:.2f} M postings/s  (vectorized id reconstruction)")
    print("    NB: Elias-Fano supports O(1) random access / skip (WAND-friendly); gamma is sequential.")

    print("\n" + "=" * 80)
    best = min([("gamma", b_gamma), ("EF", b_ef), ("FOR", b_for), ("PFor", b_pfor)], key=lambda x: x[1])
    print(f"  SMALLEST doc-id coder: {best[0]} at {bdoc(best[1]):.1f} B/doc "
          f"({b_gamma/best[1]:.3f}x vs gamma). Weights add on top (~5-bit -> ~+80-120 B/doc).")
    print("=" * 80)


if __name__ == "__main__":
    main()
