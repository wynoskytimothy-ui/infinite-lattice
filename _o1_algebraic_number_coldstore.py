"""
_o1_algebraic_number_coldstore.py  --  LENS: algebraic-number-coldstore

Cold/archival tier where a document IS a single number: the product of its
term-primes (Fundamental Theorem of Arithmetic).  decode == factor the number.

We MEASURE, honestly, both sides of the cold-tier trade:

  FOOTPRINT  (B/doc):
     - inverted-index baseline (the algebraic-corpus FOR codec, on-disk bit
       accounting from AlgebraicCorpus.footprint()).
     - algebraic cold store = one bigint per doc.  Two variants:
          set-only   : prod(prime(w))            -> recovers the WORD SET
          tf-preserving: prod(prime(w)**tf(w))   -> recovers words AND counts
       Bytes = ceil(bit_length/8) per doc, summed; on-disk via np.save of the
       packed byte blob (real getsize).
     - plus the unavoidable shared vocab side-dict (prime<->word).

  DECODE SPEED (the cost of "cold"):
     - factor each doc number back to its words.  Two factor strategies:
          (a) known-vocab trial division  -- the corpus knows its own primes,
              so factoring is O(#words) exact divisions (the corpus's decode()).
          (b) BLIND sympy.factorint        -- no vocab hint, pretend it is a
              true cold archive with only the number.  This is the honest
              "I lost the dictionary" decode cost.
     - compare to inverted-index "decode" (there is none: the postings ARE the
       words; reading a doc's words back from an inverted index requires a full
       scan, so we also measure that as the fair other-side baseline).

Run:  python "_o1_algebraic_number_coldstore.py"
"""

from __future__ import annotations
import os, math, time, io, tempfile
from collections import defaultdict

import numpy as np

from aethos_append_index import words
from aethos_algebraic_corpus import AlgebraicCorpus
from core.primes import chain_primes


def bytelen(n: int) -> int:
    """Minimal big-endian byte length to store nonneg int n (>=1 byte)."""
    if n <= 1:
        return 1
    return (n.bit_length() + 7) // 8


def pack_bigints(nums):
    """Serialize a list of bigints to a real byte blob (varint-length + bytes),
    so we can np.save it and read getsize off disk = the true on-disk footprint."""
    out = bytearray()
    for n in nums:
        b = n.to_bytes(bytelen(n), "big")
        L = len(b)
        # LEB128 varint for the length
        while True:
            byte = L & 0x7F
            L >>= 7
            if L:
                out.append(byte | 0x80)
            else:
                out.append(byte)
                break
        out.extend(b)
    return bytes(out)


def main():
    from scripts.bench_supervised_bridges import load

    print("=" * 78)
    print("ALGEBRAIC-NUMBER COLD STORE  (doc = product of term-primes; decode = factor)")
    print("=" * 78)

    corpus, queries, qrels_train, qrels_test = load("scifact")
    doc_ids = list(corpus)
    N = len(doc_ids)
    print(f"\ncorpus: {N} docs (scifact)")

    # ---- build the algebraic corpus (gives us the inverted-index baseline) ----
    t0 = time.perf_counter()
    ac = AlgebraicCorpus()
    for did in doc_ids:
        ac.add(did, corpus[did])
    ac.build()
    t_build = time.perf_counter() - t0
    st = ac.stats()
    print(f"built in {t_build:.1f}s: {st['vocab']:,} word-primes, "
          f"{st['postings']:,} postings")

    # =====================================================================
    # FOOTPRINT  --  inverted index baseline vs algebraic cold store
    # =====================================================================
    fp = ac.footprint()
    inv_for_total = fp["total_FOR_bytes"]          # chains FOR + vocab
    inv_chamber_total = fp["total_chamber_bytes"]  # chamber-projected cold tier
    vocab_bytes = fp["vocab_bytes"]
    print("\n--- FOOTPRINT (B/doc) ---")
    print(f"inverted index (FOR codec + vocab) : {inv_for_total/N:8.1f} B/doc "
          f"({inv_for_total/1e6:.2f} MB total)")
    print(f"inverted index (chamber + vocab)   : {inv_chamber_total/N:8.1f} B/doc "
          f"({inv_chamber_total/1e6:.2f} MB total)  [cold-tier codec]")

    # --- algebraic cold store: one bigint per doc ---
    # set-only: prod(prime(w))     ;  tf-preserving: prod(prime(w)**tf)
    set_nums, tf_nums = [], []
    bits_set = bits_tf = 0
    for did in doc_ids:
        ps = ac.doc_primes[did]                 # frozenset of primes
        tfp = ac.doc_tf[did]                     # {prime: tf}
        n_set = 1
        for p in ps:
            n_set *= p
        n_tf = 1
        for p, c in tfp.items():
            n_tf *= p ** c
        set_nums.append(n_set)
        tf_nums.append(n_tf)
        bits_set += n_set.bit_length()
        bits_tf += n_tf.bit_length()

    # real on-disk blobs
    blob_set = pack_bigints(set_nums)
    blob_tf = pack_bigints(tf_nums)
    tmp = tempfile.mkdtemp()
    f_set = os.path.join(tmp, "set.npy")
    f_tf = os.path.join(tmp, "tf.npy")
    np.save(f_set, np.frombuffer(blob_set, dtype=np.uint8))
    np.save(f_tf, np.frombuffer(blob_tf, dtype=np.uint8))
    disk_set = os.path.getsize(f_set)
    disk_tf = os.path.getsize(f_tf)

    set_total = disk_set + vocab_bytes
    tf_total = disk_tf + vocab_bytes
    print(f"\nalgebraic SET-ONLY   (prod prime(w))      : {set_total/N:8.1f} B/doc "
          f"({set_total/1e6:.2f} MB; numbers {disk_set/1e6:.2f} + vocab {vocab_bytes/1e6:.2f})")
    print(f"  mean number size: {bits_set/N/8:.1f} bytes/doc ({bits_set/N:.0f} bits), "
          f"recovers WORD SET only (loses tf)")
    print(f"algebraic TF-PRESERVING (prod prime(w)^tf): {tf_total/N:8.1f} B/doc "
          f"({tf_total/1e6:.2f} MB; numbers {disk_tf/1e6:.2f} + vocab {vocab_bytes/1e6:.2f})")
    print(f"  mean number size: {bits_tf/N/8:.1f} bytes/doc ({bits_tf/N:.0f} bits), "
          f"recovers words AND counts")

    # gzip the number blob too -- a real cold archive would compress at rest
    import gzip
    gz_set = len(gzip.compress(blob_set, 9))
    gz_tf = len(gzip.compress(blob_tf, 9))
    print(f"  gzip(set numbers)={gz_set/1e6:.2f}MB  gzip(tf numbers)={gz_tf/1e6:.2f}MB "
          f"-> set {((gz_set+vocab_bytes)/N):.1f} B/doc with vocab")

    print(f"\nRATIO vs inverted-FOR : set-only {inv_for_total/set_total:.2f}x  "
          f"tf-preserving {inv_for_total/tf_total:.2f}x  (>1 = cold store SMALLER)")
    print(f"RATIO vs inverted-chamber: set-only {inv_chamber_total/set_total:.2f}x  "
          f"tf-preserving {inv_chamber_total/tf_total:.2f}x")

    # --- HONEST competing FORWARD index: store each doc's words as a sorted
    #     vocab-ID list, delta-coded + min-width bit-packed (per doc). This is
    #     the fair "doc-as-a-list-of-numbers" baseline; it holds the SAME set
    #     as the bigint but in sum(log2 gap) bits, not sum(log2 prime) bits.
    #     IDs assigned in idf-rank order (rarest=smallest id, like the primes).
    id_of = {}                                  # word -> dense idf-rank id
    # rank words by df ascending to mirror the prime ranks
    df_word = defaultdict(int)
    for did in doc_ids:
        for p in ac.doc_primes[did]:
            df_word[p] += 1
    rank = sorted(df_word, key=lambda p: (df_word[p], p))
    id_of_prime = {p: i for i, p in enumerate(rank)}
    fwd_bits = 0
    fwd_blob = bytearray()
    bitbuf = 0
    nbit = 0
    def _emit(val, width):
        nonlocal bitbuf, nbit
        bitbuf |= (val & ((1 << width) - 1)) << nbit
        nbit += width
        while nbit >= 8:
            fwd_blob.append(bitbuf & 0xFF)
            bitbuf >>= 8
            nbit -= 8
    for did in doc_ids:
        ids = sorted(id_of_prime[p] for p in ac.doc_primes[did])
        # delta code
        gaps = [ids[0]] + [ids[i] - ids[i - 1] for i in range(1, len(ids))]
        w = max(1, max(gaps).bit_length())
        # header: count (16b) + width (5b)
        _emit(len(ids), 16)
        _emit(w, 5)
        for g in gaps:
            _emit(g, w)
            fwd_bits += w
        fwd_bits += 21
    if nbit:
        fwd_blob.append(bitbuf & 0xFF)
    fwd_total = len(fwd_blob) + vocab_bytes
    print(f"\nHONEST competitor FORWARD index (delta+bitpack vocab-IDs): "
          f"{fwd_total/N:8.1f} B/doc ({len(fwd_blob)/1e6:.2f}MB ids + vocab {vocab_bytes/1e6:.2f})")
    print(f"  same WORD SET as set-only bigint, but sum(log2 gap) bits not "
          f"sum(log2 prime) bits")
    print(f"  -> set-only bigint is {set_total/fwd_total:.2f}x the size of the "
          f"plain ID-list codec (>1 = bigint LOSES)")

    # =====================================================================
    # CORRECTNESS  --  exact round trip (factor recovers the words)
    # =====================================================================
    print("\n--- CORRECTNESS (decode = factor recovers exact word set) ---")
    check = doc_ids[:400]
    ok = 0
    for did in check:
        decoded = set(ac.decode(did))
        expected = set(words(corpus[did]))
        if decoded == expected:
            ok += 1
    print(f"known-vocab factor round-trip: {ok}/{len(check)} exact word-set match")

    # =====================================================================
    # DECODE SPEED  --  the cost of "cold"
    # =====================================================================
    print("\n--- DECODE SPEED (factor the number back to words) ---")

    # (a) known-vocab trial division (the corpus knows its own primes)
    sample = doc_ids[:1000]
    t0 = time.perf_counter()
    for did in sample:
        ac.decode(did)
    t_known = (time.perf_counter() - t0) / len(sample)
    print(f"(a) known-vocab trial division : {t_known*1e6:8.2f} us/doc "
          f"({1/t_known:,.0f} docs/s)  -- corpus knows its primes")

    # (b) BLIND decode -- true cold archive: only the number survives, no prime
    #     vocabulary hint.  The realistic blind strategy for a product-of-small-
    #     primes is BOUNDED TRIAL DIVISION up to the largest possible vocab prime
    #     (sympy factorint(n, limit=L)).  We measure that.  (Unbounded Pollard
    #     rho is pathological here: the factors are medium primes up to ~vocab
    #     size, so rho is far slower than just trial-dividing the prime table --
    #     which is exactly the known-vocab path.  So blind ~= bounded trial div.)
    try:
        from sympy import factorint
        primes_chain = chain_primes(st["vocab"] + 5)
        max_vocab_prime = int(primes_chain[st["vocab"] - 1])
        L = max_vocab_prime + 1            # trial-divide up to the largest vocab prime
        sizes = sorted(range(len(doc_ids)), key=lambda i: set_nums[i].bit_length())
        idxs = [sizes[int(f * (len(sizes) - 1))] for f in
                (0.1, 0.25, 0.5, 0.75, 0.9, 0.97)]
        print(f"(b) BLIND bounded trial-division (factorint limit={L:,}, no per-doc "
              f"hint) -- the 'lost the dict' cost:")
        blind_times = []
        for i in idxs:
            n = set_nums[i]
            nbits = n.bit_length()
            nwords = len(ac.doc_primes[doc_ids[i]])
            t0 = time.perf_counter()
            fac = factorint(n, limit=L)          # bounded: divides by primes up to L
            dt = time.perf_counter() - t0
            blind_times.append(dt)
            # fully factored iff residual (any factor > L) is gone; count distinct
            recovered = sum(1 for f in fac if f <= L)
            ok = (recovered == nwords)
            print(f"      {nbits:5d}-bit ({nwords:3d} primes): {dt*1e3:9.2f} ms  "
                  f"recovered={recovered}/{nwords} ok={ok}")
        mean_blind = sum(blind_times) / len(blind_times)
        print(f"      sampled blind-decode mean: {mean_blind*1e3:.2f} ms/doc "
              f"(vs {t_known*1e6:.1f} us known-vocab = {mean_blind/t_known:,.0f}x slower)")
        print(f"      full-corpus blind decode would be ~{mean_blind*N:.0f}s "
              f"({N} docs) vs known-vocab {t_known*N:.2f}s")
    except Exception as e:
        import traceback
        print(f"  blind decode failed: {e}")
        traceback.print_exc()
        mean_blind = float("nan")

    # (c) fair other side: read a doc's words back from the INVERTED INDEX.
    #     The inverted index has no per-doc word list; recovering one doc's
    #     words = scan all postings (or keep a forward index).  Measure the scan.
    t0 = time.perf_counter()
    rebuilt = defaultdict(list)
    for p, pl in ac.postings.items():
        for d in pl:
            rebuilt[d].append(p)
    t_inv_full = time.perf_counter() - t0
    print(f"(c) inverted-index decode (full posting scan -> all docs' words): "
          f"{t_inv_full*1e3:.1f} ms total = {t_inv_full/N*1e6:.2f} us/doc amortized")
    print("    (a single doc's words need a forward index or a full scan; the")
    print("     bigint is SELF-DESCRIBING -- one number factors to its own words)")

    # =====================================================================
    # VERDICT SUMMARY
    # =====================================================================
    print("\n" + "=" * 78)
    print("COLD-TIER TRADE (measured):")
    print(f"  smallest store : algebraic set-only = {set_total/N:.1f} B/doc "
          f"(gzip {((gz_set+vocab_bytes)/N):.1f}) vs inverted-FOR {inv_for_total/N:.1f}")
    print(f"  decode cost    : known-vocab {t_known*1e6:.1f} us/doc (fast), "
          f"BLIND factor = ms/doc (the cold penalty)")
    print("=" * 78)

    return {
        "N": N,
        "inv_for_bdoc": inv_for_total / N,
        "inv_chamber_bdoc": inv_chamber_total / N,
        "set_bdoc": set_total / N,
        "tf_bdoc": tf_total / N,
        "set_gz_bdoc": (gz_set + vocab_bytes) / N,
        "known_decode_us": t_known * 1e6,
    }


if __name__ == "__main__":
    main()
