#!/usr/bin/env python3
"""INGEST PROFILE — where does ingest time actually go? Split tokenize vs placement, current baseline.
We optimized query/footprint but never ingest. Measure the real bottleneck before building the fast path."""
import os, sys, time
os.environ.setdefault("PYTHONUTF8", "1")
import numpy as np
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).resolve().parent))
from aethos_append_index import AppendOnlyLatticeIndex, words, _v_word
from scripts.bench_supervised_bridges import load


def profile(name):
    corpus, *_ = load(name)
    docs = list(corpus.items())
    n = len(docs)
    total_chars = sum(len(t) for _, t in docs)

    # --- (1) full current add() path, word-only (the edge champion's ingest) ---
    t0 = time.perf_counter()
    idx = AppendOnlyLatticeIndex(index_mode="kappa_primary")
    for d, t in docs: idx.add(d, t)
    t_add = time.perf_counter() - t0
    n_post = sum(len(v) for v in idx.postings.values())

    # --- (2) tokenization only (words() over all docs) ---
    t0 = time.perf_counter()
    toks = [words(t) for _, t in docs]
    t_tok = time.perf_counter() - t0
    n_tok = sum(len(x) for x in toks)

    # --- (3) placement only: bag -> dict-of-dicts postings (no tokenize) ---
    t0 = time.perf_counter()
    tp = {}; postings = defaultdict(dict); df = defaultdict(int); nxt = 0
    for (d, _), tl in zip(docs, toks):
        bag = {}
        for w in tl: bag[w] = bag.get(w, 0.0) + 1.0
        for w, wt in bag.items():
            p = tp.get(w)
            if p is None: p = nxt; tp[w] = p; nxt += 1
            postings[p][d] = wt; df[p] += 1
    t_place = time.perf_counter() - t0

    # --- (4) full current add() path, FULL multiview (word+trigram+prefix) for contrast ---
    t0 = time.perf_counter()
    idxf = AppendOnlyLatticeIndex()
    for d, t in docs: idxf.add(d, t)
    t_addf = time.perf_counter() - t0
    n_postf = sum(len(v) for v in idxf.postings.values())

    print(f"\n  {name}: {n:,} docs, {total_chars/1e6:.1f} MB text, {n_tok:,} word-tokens")
    print(f"    FULL multiview add():  {t_addf:6.2f}s  ({n/t_addf:8.0f} docs/s, {n_postf:,} postings)")
    print(f"    WORD-ONLY    add():    {t_add:6.2f}s  ({n/t_add:8.0f} docs/s, {n_post:,} postings)  <- edge champion")
    print(f"      |- tokenize (words):  {t_tok:6.2f}s  ({n_tok/t_tok/1e6:5.2f}M tok/s)  = {100*t_tok/t_add:.0f}% of word-only")
    print(f"      |- placement (dicts): {t_place:6.2f}s  ({n_post/t_place/1e6:5.2f}M post/s) = {100*t_place/t_add:.0f}% of word-only")
    return dict(n=n, t_add=t_add, t_tok=t_tok, t_place=t_place, n_post=n_post)


def main():
    print("=" * 84)
    print("INGEST PROFILE — current baseline + tokenize/placement split (the unleveraged axis)")
    print("=" * 84)
    for name in ["scifact", "fiqa"]:
        profile(name)
    print("\n  READ: the dominant cost is the target. Placement -> numba columnar (the monitor's 230M/s route).")
    print("  Tokenization -> batched/parallel. Word-only already skips the trigram gear (~5x less placement work).")


if __name__ == "__main__":
    main()
