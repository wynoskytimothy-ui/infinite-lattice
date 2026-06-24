#!/usr/bin/env python3
"""
aethos_corpus_persist.py - TARGET 1: COMPRESS THE CHAINS FOR REAL.

The AlgebraicCorpus (aethos_algebraic_corpus.py) reports a *projected* chamber
footprint (n_postings * 9.24 / 8) in footprint(). This file replaces the
projection with REAL on-disk bytes:

  1. Build AlgebraicCorpus on scifact (corpus = product of word-primes).
  2. Extract the word-gear posting-GAP stream: for each prime (term), sort its
     posting doc-ids by DENSE ordinal, gap-delta them. That gap stream + tf is
     the chains - ~74% of the footprint; correlations are regenerated = 0 bytes.
  3. FOR bit-pack the gaps at per-term min bit-width (the marco_slim_for codec)
     and WRITE the blob to disk. stat() the real bytes.
  4. CHAMBER-compress the SAME gap stream (serialized to a byte buffer) with the
     v5 native context-mixer (scripts/test_chamber_mixer_v5_native) and WRITE the
     blob to disk. stat() the real bytes.
  5. ROUND-TRIP: read both blobs back, decode the gaps, re-cumsum to doc-ids,
     rebuild the postings, and confirm:
        (a) the decoded gaps are IDENTICAL to the originals (bit-exact), and
        (b) retrieval on the rebuilt corpus is IDENTICAL to the original.
  6. Compare measured-on-disk FOR vs chamber vs the 0.85MB projection vs raw.

Two-sided: report where chamber WINS and where it LOSES (decode cost, random
access). Honesty: is the projection honest?

Run:  python aethos_corpus_persist.py
"""

from __future__ import annotations

import os
import struct
import time
from pathlib import Path

import numpy as np

from aethos_algebraic_corpus import AlgebraicCorpus
from aethos_append_index import words
from scripts.bench_supervised_bridges import load
import scripts.test_chamber_mixer_v5_native as chamber

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "lattice_retriever_v1" / "corpus_persist"
OUT.mkdir(parents=True, exist_ok=True)


# ===========================================================================
# GAP STREAM EXTRACTION - the word-gear chains as dense-ordinal posting gaps.
# ===========================================================================
def extract_gap_stream(ac: AlgebraicCorpus):
    """For each term-prime, sorted dense-ordinal doc-ids -> (first, gaps, tf).

    Returns a list of per-term records in a STABLE order (sorted by prime), so
    both codecs and the round-trip see the exact same stream.
    """
    ord_of = {d: i for i, d in enumerate(ac.doc_len)}   # external id -> dense ordinal
    terms = []
    for p in sorted(ac.postings):                       # stable: ascending prime
        pl = ac.postings[p]
        pairs = sorted((ord_of[d], tf) for d, tf in pl.items())  # ordinal ascending
        ords = np.array([o for o, _ in pairs], dtype=np.int64)
        tfs = np.array([tf for _, tf in pairs], dtype=np.int64)
        first = int(ords[0])
        gaps = np.diff(ords).astype(np.int64) if len(ords) > 1 else np.zeros(0, np.int64)
        terms.append({"prime": p, "first": first, "gaps": gaps, "tf": tfs, "n": len(ords)})
    return terms, ord_of


# ===========================================================================
# FOR CODEC (the marco_slim_for bit-packer) - on OUR gap stream.
# Per term: first (uint32 raw) + per-term-min-width packed gaps + 4-bit tf.
# ===========================================================================
def for_pack(terms):
    """Bit-pack the gap stream the way marco_slim_for does. Returns a single
    bytes blob (header + packed gaps + packed tf) that fully reconstructs."""
    n_terms = len(terms)
    first = np.zeros(n_terms, np.uint32)
    nn = np.zeros(n_terms, np.uint32)
    width = np.zeros(n_terms, np.uint8)
    toff = np.zeros(n_terms + 1, np.uint64)
    chunks = []
    tf_chunks = []
    byte_cur = 0
    for j, t in enumerate(terms):
        first[j] = t["first"]
        nn[j] = t["n"]
        gaps = t["gaps"]
        if t["n"] > 1:
            w = max(1, int(int(gaps.max()).bit_length()))
            bits = ((gaps.astype(np.uint32)[:, None] >> np.arange(w - 1, -1, -1)) & 1).astype(np.uint8)
            packed = np.packbits(bits.reshape(-1))
            chunks.append(packed)
            width[j] = w
            toff[j] = byte_cur
            byte_cur += packed.nbytes
        else:
            width[j] = 1
            toff[j] = byte_cur
        tf_chunks.append(np.minimum(t["tf"], 15).astype(np.uint8))
    toff[n_terms] = byte_cur
    blob = np.concatenate(chunks) if chunks else np.zeros(0, np.uint8)
    tf_all = np.concatenate(tf_chunks) if tf_chunks else np.zeros(0, np.uint8)
    if len(tf_all) % 2:
        tf_all = np.append(tf_all, np.uint8(0))
    tf_packed = (tf_all[0::2] | (tf_all[1::2] << 4)).astype(np.uint8) if len(tf_all) else np.zeros(0, np.uint8)

    # serialize EVERYTHING needed to reconstruct, as one on-disk byte blob:
    #   [n_terms u32][first u32*][nn u32*][width u8*][toff u64*(n+1)]
    #   [len(blob) u64][blob][len(tf_packed) u64][tf_packed]
    buf = bytearray()
    buf += struct.pack("<I", n_terms)
    buf += first.tobytes()
    buf += nn.tobytes()
    buf += width.tobytes()
    buf += toff.tobytes()
    buf += struct.pack("<Q", blob.nbytes) + blob.tobytes()
    buf += struct.pack("<Q", tf_packed.nbytes) + tf_packed.tobytes()
    return bytes(buf)


def for_unpack(buf: bytes):
    """Inverse of for_pack: bytes -> per-term (first, gaps, tf)."""
    off = 0
    (n_terms,) = struct.unpack_from("<I", buf, off); off += 4
    first = np.frombuffer(buf, np.uint32, n_terms, off).copy(); off += 4 * n_terms
    nn = np.frombuffer(buf, np.uint32, n_terms, off).copy(); off += 4 * n_terms
    width = np.frombuffer(buf, np.uint8, n_terms, off).copy(); off += n_terms
    toff = np.frombuffer(buf, np.uint64, n_terms + 1, off).copy(); off += 8 * (n_terms + 1)
    (blob_n,) = struct.unpack_from("<Q", buf, off); off += 8
    blob = np.frombuffer(buf, np.uint8, blob_n, off).copy(); off += blob_n
    (tf_n,) = struct.unpack_from("<Q", buf, off); off += 8
    tf_packed = np.frombuffer(buf, np.uint8, tf_n, off).copy(); off += tf_n

    # unpack tf (4-bit) in posting order
    tf_all = np.empty(tf_packed.size * 2, np.uint8)
    tf_all[0::2] = tf_packed & 0xF
    tf_all[1::2] = tf_packed >> 4

    out = []
    pi = 0
    for j in range(n_terms):
        n = int(nn[j])
        if n > 1:
            w = int(width[j])
            o0, o1 = int(toff[j]), int(toff[j + 1])
            bits = np.unpackbits(blob[o0:o1])[:(n - 1) * w].reshape(n - 1, w)
            gaps = bits.dot((1 << np.arange(w - 1, -1, -1)).astype(np.uint32)).astype(np.int64)
        else:
            gaps = np.zeros(0, np.int64)
        tf = tf_all[pi:pi + n].astype(np.int64)
        pi += n
        out.append({"first": int(first[j]), "gaps": gaps, "tf": tf, "n": n})
    return out


# ===========================================================================
# CHAMBER CODEC - on the SAME gap stream serialized to a flat byte buffer.
# The chamber compresses arbitrary bytes; we hand it the varint gap+tf stream
# (the natural "gap stream" the algebraic corpus's footprint() refers to).
# ===========================================================================
def varint(n):
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            break
    return out


def read_varint(buf, off):
    shift = 0
    val = 0
    while True:
        b = buf[off]; off += 1
        val |= (b & 0x7F) << shift
        if not (b & 0x80):
            break
        shift += 7
    return val, off


def gap_stream_bytes(terms):
    """Serialize first/n/gaps/tf as a varint byte stream - this IS the
    posting-gap stream the chamber compresses (smallest-storage tier)."""
    buf = bytearray()
    buf += varint(len(terms))
    for t in terms:
        buf += varint(t["n"])
        buf += varint(t["first"])
        for g in t["gaps"]:
            buf += varint(int(g))
        for v in t["tf"]:
            buf += varint(int(min(v, 15)))
    return bytes(buf)


def gap_stream_unbytes(buf: bytes):
    """Inverse of gap_stream_bytes."""
    off = 0
    n_terms, off = read_varint(buf, off)
    out = []
    for _ in range(n_terms):
        n, off = read_varint(buf, off)
        first, off = read_varint(buf, off)
        gaps = np.empty(max(0, n - 1), np.int64)
        for i in range(n - 1):
            g, off = read_varint(buf, off)
            gaps[i] = g
        tf = np.empty(n, np.int64)
        for i in range(n):
            v, off = read_varint(buf, off)
            tf[i] = v
        out.append({"first": first, "gaps": gaps, "tf": tf, "n": n})
    return out


# ===========================================================================
# RECONSTRUCT postings dict from decoded per-term records (cumsum the gaps).
# ===========================================================================
def rebuild_postings(records, primes_in_order, ord_to_id):
    postings = {}
    for rec, p in zip(records, primes_in_order):
        n = rec["n"]
        ords = np.empty(n, np.int64)
        ords[0] = rec["first"]
        if n > 1:
            ords[1:] = rec["first"] + np.cumsum(rec["gaps"])
        pl = {}
        for o, tf in zip(ords.tolist(), rec["tf"].tolist()):
            pl[ord_to_id[o]] = int(tf)
        postings[p] = pl
    return postings


def main():
    print("=" * 76)
    print("CORPUS-AS-A-NUMBER: REAL on-disk compression of the chains (scifact)")
    print("=" * 76)

    # ---- BUILD the algebraic corpus ----
    corpus, queries, qrels_train, qrels_test = load("scifact")
    t0 = time.time()
    ac = AlgebraicCorpus()
    for did, text in corpus.items():
        ac.add(did, text)
    ac.build()
    t_build = time.time() - t0
    st = ac.stats()
    print(f"\nBuilt {st['docs']:,} docs, {st['vocab']:,} word-primes, "
          f"{st['postings']:,} postings in {t_build:.1f}s")

    # ---- EXTRACT the gap stream ----
    terms, ord_of = extract_gap_stream(ac)
    ord_to_id = {i: d for d, i in ord_of.items()}
    primes_in_order = [t["prime"] for t in terms]
    n_postings = sum(t["n"] for t in terms)
    print(f"Extracted gap stream: {len(terms):,} terms, {n_postings:,} postings")

    # ---- FOR pack + PERSIST ----
    t0 = time.time()
    for_blob = for_pack(terms)
    t_for_enc = time.time() - t0
    for_path = OUT / "chains_FOR.bin"
    for_path.write_bytes(for_blob)
    for_disk = os.path.getsize(for_path)

    # ---- CHAMBER compress + PERSIST ----
    raw_gap_bytes = gap_stream_bytes(terms)
    gap_raw_path = OUT / "chains_gapstream_raw.bin"
    gap_raw_path.write_bytes(raw_gap_bytes)
    print(f"\nGap-stream (varint) raw buffer: {len(raw_gap_bytes)/1e6:.3f} MB "
          f"-> feeding the chamber context-mixer ...")
    # warm-up JIT (compile) so timing is the real codec, not LLVM compile
    _w = b"warmup " * 256
    _b, _ = chamber.compress(_w)
    chamber.decompress(_b, len(_w))
    t0 = time.time()
    cham_blob, _stats = chamber.compress(raw_gap_bytes)
    t_cham_enc = time.time() - t0
    cham_path = OUT / "chains_chamber.bin"
    cham_path.write_bytes(cham_blob)
    cham_disk = os.path.getsize(cham_path)

    # ---- vocab side dict (prime<->word) persist for a complete footprint ----
    vocab_buf = bytearray()
    for w in ac.prime_of:
        wb = w.encode("utf-8")
        vocab_buf += struct.pack("<I", ac.prime_of[w]) + struct.pack("<H", len(wb)) + wb
    vocab_path = OUT / "vocab.bin"
    vocab_path.write_bytes(vocab_buf)
    vocab_disk = os.path.getsize(vocab_path)

    # ======================================================================
    # ROUND-TRIP 1: FOR -> identical gaps -> identical retrieval
    # ======================================================================
    print("\n" + "-" * 76)
    print("ROUND-TRIP VERIFICATION")
    print("-" * 76)
    for_back = for_unpack(for_path.read_bytes())
    for_gaps_ok = all(
        np.array_equal(a["gaps"], b["gaps"]) and a["first"] == b["first"]
        and np.array_equal(np.minimum(a["tf"], 15), b["tf"])
        for a, b in zip(terms, for_back))
    print(f"  FOR     decode -> gaps identical: {for_gaps_ok}")

    # ======================================================================
    # ROUND-TRIP 2: CHAMBER -> identical gaps
    # ======================================================================
    t0 = time.time()
    cham_buf, _ = chamber.decompress(cham_path.read_bytes(), len(raw_gap_bytes))
    t_cham_dec = time.time() - t0
    cham_bytes_ok = (cham_buf == raw_gap_bytes)
    cham_back = gap_stream_unbytes(cham_buf)
    cham_gaps_ok = all(
        np.array_equal(a["gaps"], b["gaps"]) and a["first"] == b["first"]
        and np.array_equal(np.minimum(a["tf"], 15), b["tf"])
        for a, b in zip(terms, cham_back))
    print(f"  chamber decode -> byte-exact stream: {cham_bytes_ok}  "
          f"gaps identical: {cham_gaps_ok}")

    # rebuild postings from FOR (note tf clamped to 4 bits in BOTH codecs)
    rebuilt = rebuild_postings(for_back, primes_in_order, ord_to_id)
    posting_ids_ok = all(
        set(rebuilt[p]) == set(ac.postings[p]) for p in primes_in_order)
    print(f"  rebuilt postings: same doc-id sets per term: {posting_ids_ok}")

    # ---- retrieval identity: original vs rebuilt (clamp tf to match codec) ----
    ac2 = AlgebraicCorpus()
    ac2.prime_of = dict(ac.prime_of)
    ac2.word_of = dict(ac.word_of)
    ac2.doc_len = dict(ac.doc_len)
    ac2._total_len = ac._total_len
    ac2.doc_primes = dict(ac.doc_primes)
    ac2.postings = rebuilt
    ac2.df = {p: len(rebuilt[p]) for p in rebuilt}
    ac2._frozen = True

    gold = {q: {d for d, s in rel.items() if s > 0} for q, rel in qrels_test.items()}
    test_qids = [q for q in gold if gold[q] and q in queries]
    same_cold = same_warm = 0
    for q in test_qids:
        if ac.query(queries[q], k=20, T=0.0) == ac2.query(queries[q], k=20, T=0.0):
            same_cold += 1
        if ac.query(queries[q], k=20, T=ac.warm_T) == ac2.query(queries[q], k=20, T=ac.warm_T):
            same_warm += 1
    nq = len(test_qids)
    print(f"  retrieval identity (orig vs rebuilt-from-disk), {nq} test queries:")
    print(f"     COLD T=0 identical top-20: {same_cold}/{nq}")
    print(f"     WARM     identical top-20: {same_warm}/{nq}")

    # ======================================================================
    # FOOTPRINT TABLE - measured-on-disk vs projection vs raw
    # ======================================================================
    fp = ac.footprint()
    proj_chamber = fp["chains_chamber_bytes"]       # the 0.85MB-class projection
    raw_chains = fp["chains_raw_bytes"]             # 5B/posting naive

    print("\n" + "=" * 76)
    print("FOOTPRINT - REAL on-disk bytes (the chains)")
    print("=" * 76)
    print(f"  postings: {n_postings:,}   terms: {len(terms):,}")
    print(f"  raw chains (5B/posting, naive)      : {raw_chains/1e6:8.3f} MB")
    print(f"  FOR  on disk (MEASURED)             : {for_disk/1e6:8.3f} MB"
          f"   ({for_disk*8/n_postings:.2f} bits/posting)")
    print(f"  chamber on disk (MEASURED)          : {cham_disk/1e6:8.3f} MB"
          f"   ({cham_disk*8/n_postings:.2f} bits/posting)")
    print(f"  chamber PROJECTION (footprint())    : {proj_chamber/1e6:8.3f} MB"
          f"   (9.24 bits/posting)")
    print(f"  gap-stream raw (varint, pre-chamber): {len(raw_gap_bytes)/1e6:8.3f} MB")
    print(f"  vocab side-dict on disk (MEASURED)  : {vocab_disk/1e6:8.3f} MB")
    print()
    print(f"  FOR     vs raw    : {raw_chains/for_disk:5.2f}x smaller")
    print(f"  chamber vs raw    : {raw_chains/cham_disk:5.2f}x smaller")
    print(f"  chamber vs FOR    : {for_disk/cham_disk:5.2f}x "
          f"({'smaller' if cham_disk < for_disk else 'LARGER'})")
    proj_err = (cham_disk - proj_chamber) / proj_chamber * 100
    print(f"  projection honesty: measured {cham_disk/1e6:.3f} MB vs projected "
          f"{proj_chamber/1e6:.3f} MB  ({proj_err:+.1f}%)")

    print("\n  TOTAL corpus-as-a-number on disk (chains + vocab; correlations=0):")
    print(f"     FOR     : {(for_disk + vocab_disk)/1e6:.3f} MB")
    print(f"     chamber : {(cham_disk + vocab_disk)/1e6:.3f} MB")

    print("\n  SPEED (two-sided): chamber encode {:.2f}s, decode {:.2f}s for the "
          "WHOLE stream".format(t_cham_enc, t_cham_dec))
    print(f"     chamber is SEQUENTIAL (decode-all, no random access); "
          f"FOR is random-access per-term.")
    print(f"     FOR encode {t_for_enc*1000:.0f}ms (vectorized, per-term decode O(n)).")

    return {
        "n_postings": n_postings,
        "for_disk": for_disk,
        "cham_disk": cham_disk,
        "proj_chamber": proj_chamber,
        "raw_chains": raw_chains,
        "vocab_disk": vocab_disk,
        "for_gaps_ok": for_gaps_ok,
        "cham_gaps_ok": cham_gaps_ok,
        "cham_bytes_ok": cham_bytes_ok,
        "posting_ids_ok": posting_ids_ok,
        "same_cold": same_cold,
        "same_warm": same_warm,
        "nq": nq,
        "t_cham_enc": t_cham_enc,
        "t_cham_dec": t_cham_dec,
    }


if __name__ == "__main__":
    main()
