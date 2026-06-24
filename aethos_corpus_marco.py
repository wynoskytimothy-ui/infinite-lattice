#!/usr/bin/env python3
"""
aethos_corpus_marco.py - TARGET 3: scale the CORPUS-IS-A-NUMBER to MS MARCO.

The math-derived AlgebraicCorpus (aethos_algebraic_corpus.py) treats a corpus as
NUMBERS: each doc = product of its content-word primes (FTA), decode = factor,
append = multiply, correlations REGENERATED from the lattice meet (0 stored).
It ties BM25 on small corpora (scifact); the wins are STRUCTURAL.

This file SCALES it toward the 8.8M MS MARCO collection WITHOUT rebuilding 8.8M
from scratch (too slow).  Instead:

  (a) Build AlgebraicCorpus on a LARGE subset (default 50,000 MARCO passages,
      pulled via marco_full_eval.FullIndex.text(docid)).  Measure:
        - build rate (docs/s),
        - footprint:  FOR bit-packed (the corpus's own footprint(), which mirrors
          marco_slim_for.py) AND the chamber-codec projection,
        - cold/warm retrieval latency on a query sample.

  (b) PROJECT the per-doc composite size + posting count + footprint to 8.8M and
      compare to the persisted 0.428 GB FOR headline (commit 9e47a88).

  (c) CONFIRM the algebraic ops still hold at this scale:
        - decode = factor  (round-trip the composite -> exact word set),
        - append = multiply (one bigint multiply, no reindex, immediately retrievable).

HONEST: the subset is a RANDOM 50k of 8.8M.  Projection assumes the subset's
per-doc posting/byte rates are representative; passage length and idf-rank prime
sizes shift slightly at full scale (more vocab -> larger primes -> bigger
composites, but the FOR codec stores doc-id GAPS, not the composite, so footprint
projects on postings/doc, which is length-driven and stable).  We report the
projection two-sided and flag what subset-vs-full can and cannot tell us.

Run:  python aethos_corpus_marco.py [n_docs] [n_queries]
        n_docs    default 50000   (MARCO passages to ingest)
        n_queries default 200     (dev queries for latency timing)
"""
from __future__ import annotations

import sys
import time
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from aethos_algebraic_corpus import AlgebraicCorpus, _digits
from aethos_append_index import words
from marco_full_eval import FullIndex, stoks, MARCO

# The persisted FOR slim headline (commit 9e47a88 / honest-headline-scorecard):
#   0.428 GB on disk (di FOR-packed), round-trip MATCH, ~6-8ms retrieval.
HEADLINE_FOR_GB = 0.428
# The slim FOR keeps only idf>=4.0 terms (marco_slim_for.KEEP_IDF) across 8.8M.
MARCO_N_FULL = 8_841_823   # canonical MS MARCO passage count


def _line(c="-", n=74):
    print(c * n)


def build_subset(idx: FullIndex, n_docs: int, seed: int = 42):
    """Ingest n_docs random MARCO passages into an AlgebraicCorpus.

    Uses idx.text(pid) for the raw passage text (same loader the FOR slim built on),
    then AlgebraicCorpus.add() stages bag-of-words and build() assigns idf-rank primes.
    Returns (corpus, doc_ids, build_seconds, ingest_seconds)."""
    rng = random.Random(seed)
    # sample distinct pids in [0, idx.N)
    pids = rng.sample(range(idx.N), n_docs)
    ac = AlgebraicCorpus()
    t0 = time.perf_counter()
    n_chars = 0
    for pid in pids:
        txt = idx.text(pid)
        n_chars += len(txt)
        ac.add(str(pid), txt)         # APPEND-ONLY stage (string doc-id)
    t_ingest = time.perf_counter() - t0
    t1 = time.perf_counter()
    ac.build()                        # idf-rank primes + materialize composites
    t_build = time.perf_counter() - t1
    return ac, [str(p) for p in pids], t_build, t_ingest, n_chars


def main():
    n_docs = int(sys.argv[1]) if len(sys.argv) > 1 else 50_000
    n_queries = int(sys.argv[2]) if len(sys.argv) > 2 else 200

    _line("=")
    print("TARGET 3 - CORPUS-IS-A-NUMBER at MARCO scale")
    print(f"  subset = {n_docs:,} random passages of the 8.8M MS MARCO collection")
    _line("=")

    print("\nloading full MARCO index (for text(pid) + dev queries)...")
    idx = FullIndex()

    # ---------------------------------------------------------------------
    # (a) BUILD on the subset; measure rate + footprint
    # ---------------------------------------------------------------------
    print(f"\n[a] BUILD AlgebraicCorpus on {n_docs:,} passages")
    ac, doc_ids, t_build, t_ingest, n_chars = build_subset(idx, n_docs)
    t_total = t_ingest + t_build
    st = ac.stats()
    fp = ac.footprint()
    n_docs_real = st["docs"]
    rate = n_docs_real / t_total

    print(f"    ingest (stage bag-of-words) : {t_ingest:6.1f}s")
    print(f"    build  (primes + composites): {t_build:6.1f}s")
    print(f"    TOTAL                       : {t_total:6.1f}s  "
          f"=> {rate:,.0f} docs/s")
    print(f"    docs={n_docs_real:,}  word-primes(vocab)={st['vocab']:,}  "
          f"postings={st['postings']:,}")
    postings_per_doc = st["postings"] / n_docs_real
    print(f"    postings/doc = {postings_per_doc:.1f}  "
          f"(avg passage content-word count)")
    print(f"    corpus_product is a {st['corpus_product_digits']:,}-digit integer "
          f"(the WHOLE subset as ONE number)")

    # per-doc composite size (bits) - the FTA bigint per document
    comp_bits = [ac.doc_number[d].bit_length() for d in doc_ids]
    comp_bits = np.array(comp_bits, dtype=np.float64)
    print(f"    per-doc composite: mean {comp_bits.mean():.0f} bits "
          f"({comp_bits.mean()/8:.0f} bytes), max {comp_bits.max():.0f} bits "
          f"({_digits(ac.doc_number[doc_ids[int(comp_bits.argmax())]]):,} digits)")
    print("    NOTE: the composite is the IN-MEMORY math object; the ON-DISK codec")
    print("          stores posting-GAPS (doc-ids), not the composite -> footprint")
    print("          below is gap-driven, not composite-driven.")

    print(f"\n    FOOTPRINT (subset, {n_docs_real:,} docs):")
    print(f"      chains raw (5B/posting)   : {fp['chains_raw_bytes']/1e6:8.2f} MB")
    print(f"      chains FOR (bit-packed)   : {fp['chains_FOR_bytes']/1e6:8.2f} MB"
          f"  ({fp['for_bits_per_posting']:.2f} bits/posting)")
    print(f"      chains chamber (projected): {fp['chains_chamber_bytes']/1e6:8.2f} MB"
          f"  (9.24 bits/posting, cold tier)")
    print(f"      correlations              : {fp['correlations_bytes']/1e6:8.2f} MB"
          f"  (REGENERATED from the meet - 0 stored)")
    print(f"      vocab side-dict           : {fp['vocab_bytes']/1e6:8.2f} MB")
    print(f"      total (FOR + vocab)       : {fp['total_FOR_bytes']/1e6:8.2f} MB")
    print(f"      total (chamber + vocab)   : {fp['total_chamber_bytes']/1e6:8.2f} MB")
    bytes_per_doc_for = fp["total_FOR_bytes"] / n_docs_real
    bytes_per_doc_cham = fp["total_chamber_bytes"] / n_docs_real
    print(f"      per-doc: FOR {bytes_per_doc_for:.1f} B/doc  |  "
          f"chamber {bytes_per_doc_cham:.1f} B/doc")

    # ---------------------------------------------------------------------
    # (a cont.) COLD / WARM retrieval latency on a query sample
    # ---------------------------------------------------------------------
    print(f"\n    COLD/WARM retrieval latency ({n_queries} dev queries):")
    queries = {}
    with open(MARCO / "queries.dev.tsv", encoding="utf-8") as f:
        for ln in f:
            a = ln.rstrip("\n").split("\t", 1)
            if len(a) == 2:
                queries[a[0]] = a[1]
    qsample = list(queries.values())
    random.Random(7).shuffle(qsample)
    qsample = qsample[:n_queries]
    # warm-up
    for q in qsample[:5]:
        ac.query(q, k=100, T=0.0)
    cold_lat, warm_lat = [], []
    for q in qsample:
        t0 = time.perf_counter(); ac.query(q, k=100, T=0.0)
        cold_lat.append((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter(); ac.query(q, k=100, T=ac.warm_T)
        warm_lat.append((time.perf_counter() - t0) * 1000)
    cold_lat = np.array(cold_lat); warm_lat = np.array(warm_lat)
    print(f"      COLD (exact meet)  : median {np.median(cold_lat):7.2f} ms  "
          f"p90 {np.percentile(cold_lat,90):7.2f} ms")
    print(f"      WARM (corridor)    : median {np.median(warm_lat):7.2f} ms  "
          f"p90 {np.percentile(warm_lat,90):7.2f} ms")
    print("      (warm regenerates the co-occurrence corridor per query = the")
    print("       free correlation; cold is the hard set-intersection)")

    # ---------------------------------------------------------------------
    # (c) ALGEBRAIC OPS survive scale:  decode=factor, append=multiply
    # ---------------------------------------------------------------------
    print("\n[c] ALGEBRAIC OPS at scale")
    # decode = factor: round-trip composites back to their word set
    n_check = min(2000, n_docs_real)
    check = doc_ids[:n_check]
    ok = 0
    t0 = time.perf_counter()
    for d in check:
        decoded = set(ac.decode(d))
        expected = set(words(idx.text(int(d))))
        if decoded == expected:
            ok += 1
    t_dec = time.perf_counter() - t0
    print(f"    decode=factor: {ok}/{n_check} composites round-trip to the EXACT")
    print(f"      word set (FTA inverse), {t_dec:.2f}s = "
          f"{n_check/max(1e-9,t_dec):,.0f} decodes/s")

    # append = multiply: add a live doc = one bigint multiply, no reindex
    cp_before = ac.corpus_product
    post_before = ac.stats()["postings"]
    new_id = "LIVE_MARCO_DOC"
    new_text = ("the transformer architecture uses self attention to model long "
                "range dependencies in sequences for machine translation")
    t0 = time.perf_counter()
    ac.add(new_id, new_text)          # frozen corpus -> _materialize: one *=
    t_app = (time.perf_counter() - t0) * 1000
    post_after = ac.stats()["postings"]
    ratio = ac.corpus_product // cp_before
    multiply_exact = (ratio == ac.doc_number[new_id])
    decode_new = ac.decode(new_id)
    hit = ac.query("transformer self attention translation", k=10, T=0.0)
    retrievable = new_id in hit
    print(f"    append=multiply: added 1 doc in {t_app:.2f} ms; "
          f"postings {post_before:,} -> {post_after:,}")
    print(f"      corpus_product *= doc_number EXACT? {multiply_exact}  "
          f"(no existing posting rewritten)")
    print(f"      decode(new)[:6] = {decode_new[:6]}")
    print(f"      immediately retrievable (no rebuild)? {retrievable}")

    # ---------------------------------------------------------------------
    # (b) PROJECT to 8.8M and compare to the 0.428 GB FOR headline
    # ---------------------------------------------------------------------
    print("\n[b] PROJECT to the full 8.8M collection")
    scale = MARCO_N_FULL / n_docs_real
    proj_postings = st["postings"] * scale
    proj_for_gb = fp["total_FOR_bytes"] * scale / 1e9
    proj_cham_gb = fp["total_chamber_bytes"] * scale / 1e9
    # vocab does NOT scale linearly (Heaps' law: V ~ k * n^beta, beta~0.5).
    # Project vocab via Heaps from the subset to flag the side-dict separately.
    beta = 0.5
    proj_vocab_terms = st["vocab"] * (scale ** beta)
    avg_word_len = (fp["vocab_bytes"] / max(1, st["vocab"])) - 4  # subtract the 4B prime
    proj_vocab_bytes = proj_vocab_terms * (4 + max(1.0, avg_word_len))
    # re-project FOR chains alone (without subset vocab) + Heaps vocab
    proj_for_chains_gb = fp["chains_FOR_bytes"] * scale / 1e9
    proj_for_gb_heaps = (fp["chains_FOR_bytes"] * scale + proj_vocab_bytes) / 1e9
    proj_cham_gb_heaps = (fp["chains_chamber_bytes"] * scale + proj_vocab_bytes) / 1e9

    print(f"    linear scale factor = 8.8M / {n_docs_real:,} = {scale:,.1f}x")
    print(f"    projected postings  = {proj_postings/1e9:.2f} B "
          f"({postings_per_doc:.1f}/doc x 8.8M)")
    print(f"    projected FOR (linear vocab)  : {proj_for_gb:.3f} GB")
    print(f"    projected FOR (Heaps vocab)   : {proj_for_gb_heaps:.3f} GB   "
          f"<- chains {proj_for_chains_gb:.3f} + vocab {proj_vocab_bytes/1e9:.3f}")
    print(f"    projected chamber (Heaps vocab): {proj_cham_gb_heaps:.3f} GB  "
          f"(cold/archival tier)")
    print(f"\n    HEADLINE persisted FOR slim   : {HEADLINE_FOR_GB:.3f} GB "
          f"(idf>=4 terms, 8.8M, on disk)")
    delta = proj_for_gb_heaps - HEADLINE_FOR_GB
    print(f"    delta (this projection - headline): {delta:+.3f} GB")
    print(f"    NOTE: the headline FOR keeps ONLY idf>=4.0 terms (rare-term slim);")
    print(f"          this AlgebraicCorpus keeps ALL content words (no idf prune),")
    print(f"          so MORE postings -> a LARGER projection is EXPECTED + honest.")
    # also project a slim variant: how many postings survive an idf>=4 prune here?
    # use the corpus's OWN idf (BM25 idf over the subset df), matching marco_slim_for.KEEP_IDF=4.0
    rare_postings = sum(len(pl) for p, pl in ac.postings.items()
                        if ac._idf(p) >= 4.0)
    rare_frac = rare_postings / max(1, st["postings"])
    proj_for_slim_gb = proj_for_chains_gb * rare_frac + proj_vocab_bytes / 1e9
    print(f"    if we slim to idf>=4 like the headline: "
          f"{rare_postings:,}/{st['postings']:,} postings survive "
          f"({100*rare_frac:.1f}%) -> projected FOR ~{proj_for_slim_gb:.3f} GB "
          f"(apples-to-apples vs the 0.428 headline)")

    # ---------------------------------------------------------------------
    # SUMMARY (the structured-output numbers)
    # ---------------------------------------------------------------------
    print("\n" + "=" * 74)
    print("SUMMARY")
    print(f"  build rate           : {rate:,.0f} docs/s ({n_docs_real:,} docs / {t_total:.1f}s)")
    print(f"  per-doc composite    : {comp_bits.mean()/8:.0f} bytes (mean), in-memory math object")
    print(f"  per-doc footprint    : FOR {bytes_per_doc_for:.1f} B/doc  chamber {bytes_per_doc_cham:.1f} B/doc")
    print(f"  projected full (FOR) : {proj_for_gb_heaps:.3f} GB (all-terms) vs headline {HEADLINE_FOR_GB:.3f} GB (idf>=4 slim)")
    print(f"  projected idf>=4 slim: {proj_for_slim_gb:.3f} GB (apples-to-apples with the 0.428 headline)")
    print(f"  decode=factor        : {ok}/{n_check} exact round-trips at scale")
    print(f"  append=multiply      : exact={multiply_exact}, live-retrievable={retrievable}, {t_app:.2f}ms")
    print("=" * 74)

    return dict(rate=rate, n_docs=n_docs_real, postings=st["postings"],
                postings_per_doc=postings_per_doc,
                comp_bytes_mean=comp_bits.mean() / 8,
                bytes_per_doc_for=bytes_per_doc_for,
                bytes_per_doc_cham=bytes_per_doc_cham,
                proj_for_gb=proj_for_gb_heaps, proj_cham_gb=proj_cham_gb_heaps,
                proj_for_slim_gb=proj_for_slim_gb, headline_gb=HEADLINE_FOR_GB,
                decode_ok=ok, decode_n=n_check,
                multiply_exact=multiply_exact, retrievable=retrievable,
                cold_med=float(np.median(cold_lat)), warm_med=float(np.median(warm_lat)))


if __name__ == "__main__":
    main()
