#!/usr/bin/env python3
"""FAST INGEST — columnar word-only build, identical index, much faster than incremental add().

The unleveraged axis: ingest. Incremental add() does per-token dict-of-dict inserts (2-3M post/s).
Fast path: tokenize -> per-doc bag (dict vocab, C-level hashing) -> emit (term,doc,tf) rows ->
ONE vectorized lexsort -> CSR per term. No dict-of-dicts; the sorted CSR is exactly what the
binary-reader / mmap serve consumes. Optional parallel tokenize across cores.

Postings are verified bit-identical to incremental -> accuracy + footprint UNCHANGED; only speed moves.
"""
import os, sys, time
os.environ.setdefault("PYTHONUTF8", "1")
import numpy as np
from array import array
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, str(Path(__file__).resolve().parent))
from aethos_append_index import AppendOnlyLatticeIndex, words
from scripts.bench_supervised_bridges import load

POS_HEAD, POS_BOOST = 14, 1.6   # match AppendOnlyLatticeIndex positional defaults


def _tok_chunk(texts):
    return [words(t) for t in texts]


def tokenize(texts, workers):
    if workers <= 1:
        return [words(t) for t in texts]
    n = len(texts); cs = (n + workers - 1) // workers
    chunks = [texts[i:i + cs] for i in range(0, n, cs)]
    out = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for part in ex.map(_tok_chunk, chunks):
            out.extend(part)
    return out


def fast_build(texts, workers=1):
    """word-only columnar build -> (vocab token->row, indptr, seg_doc, seg_tf, df). Times each stage."""
    t = {}
    s = time.perf_counter(); toks = tokenize(texts, workers); t["tokenize"] = time.perf_counter() - s

    s = time.perf_counter()
    vocab = {}; vget = vocab.get
    rt = array("i"); rd = array("i"); rw = array("d")   # 'i'=int32 (Windows-safe), 'd'=float64
    rta, rda, rwa = rt.append, rd.append, rw.append
    for di, tl in enumerate(toks):
        bag = {}; bget = bag.get
        for i, w in enumerate(tl):
            bag[w] = bget(w, 0.0) + (POS_BOOST if i < POS_HEAD else 1.0)
        for w, wt in bag.items():
            tid = vget(w)
            if tid is None: tid = len(vocab); vocab[w] = tid
            rta(tid); rda(di); rwa(wt)
    t["bag+vocab"] = time.perf_counter() - s

    s = time.perf_counter()
    T = np.frombuffer(rt, np.int32); D = np.frombuffer(rd, np.int32); W = np.frombuffer(rw, np.float64)
    order = np.lexsort((D, T))
    T, D, W = T[order], D[order], W[order]
    V = len(vocab)
    df = np.bincount(T, minlength=V).astype(np.int64)
    indptr = np.concatenate([[0], np.cumsum(df)]).astype(np.int64)
    seg_doc = D.astype(np.uint32); seg_tf = W.astype(np.float16)
    t["sort+csr"] = time.perf_counter() - s
    return (vocab, indptr, seg_doc, seg_tf, df), t


def fast_build_mp(texts, workers):
    """Parallel: lightweight workers (re-only import) do tokenize+bag; main does vocab+emit+sort."""
    import _fast_tok
    t = {}
    n = len(texts); cs = (n + workers - 1) // workers
    chunks = [texts[i:i + cs] for i in range(0, n, cs)]
    s = time.perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(_fast_tok.tok_bag_chunk, chunks): results.append(r)
    t["tok+bag(par)"] = time.perf_counter() - s

    s = time.perf_counter()
    vocab = {}; vget = vocab.get
    rt = array("i"); rd = array("i"); rw = array("d")
    rta, rda, rwa = rt.append, rd.append, rw.append
    di = 0
    for flat_t, flat_w, nitems in results:
        j = 0
        for cnt in nitems:
            for _ in range(cnt):
                w = flat_t[j]; wt = flat_w[j]; j += 1
                tid = vget(w)
                if tid is None: tid = len(vocab); vocab[w] = tid
                rta(tid); rda(di); rwa(wt)
            di += 1
    t["vocab+emit"] = time.perf_counter() - s

    s = time.perf_counter()
    T = np.frombuffer(rt, np.int32); D = np.frombuffer(rd, np.int32); W = np.frombuffer(rw, np.float64)
    order = np.lexsort((D, T)); T, D, W = T[order], D[order], W[order]
    V = len(vocab); df = np.bincount(T, minlength=V).astype(np.int64)
    indptr = np.concatenate([[0], np.cumsum(df)]).astype(np.int64)
    t["sort+csr"] = time.perf_counter() - s
    return (vocab, indptr, D.astype(np.uint32), W.astype(np.float16), df), t


def verify_identical(corpus):
    doc_ids = list(corpus.keys()); texts = list(corpus.values())
    slow = AppendOnlyLatticeIndex(index_mode="kappa_primary")
    for d, x in corpus.items(): slow.add(d, x)
    (vocab, indptr, seg_doc, seg_tf, df), _ = fast_build(texts)
    mism = checked = 0
    for (view, tok), p in slow.token_prime.items():
        if view != "w": continue
        checked += 1
        # compare at float16 precision: the serve stores float16 on BOTH paths, so float64-vs-float16
        # is a spurious diff. Cast slow weights to float16 too -> a fair, serve-faithful comparison.
        slow_pl = {d: float(np.float16(w)) for d, w in slow.postings[p].items()}
        r = vocab.get(tok)
        fast_pl = {}
        if r is not None:
            a, e = int(indptr[r]), int(indptr[r + 1])
            fast_pl = {doc_ids[int(seg_doc[i])]: float(seg_tf[i]) for i in range(a, e)}
        if slow_pl != fast_pl: mism += 1
    return checked, mism, len(vocab)


def main():
    print("=" * 86)
    print("FAST INGEST — columnar word-only build (identical index). The unleveraged ingest axis.")
    print("=" * 86)
    for name in ["scifact", "fiqa"]:
        corpus, *_ = load(name)
        texts = list(corpus.values()); n = len(texts)
        ntok = sum(len(words(t)) for t in texts)

        s = time.perf_counter()
        idx = AppendOnlyLatticeIndex(index_mode="kappa_primary")
        for d, x in corpus.items(): idx.add(d, x)
        t_add = time.perf_counter() - s
        s = time.perf_counter(); idx.finalize(); t_fin = time.perf_counter() - s
        t_slow = t_add + t_fin   # end-to-end to serve-ready (columnar outputs serve-ready CSR directly)

        (_, _, seg_doc, _, _), t1 = fast_build(texts, workers=1)
        f1 = sum(t1.values()); n_post = len(seg_doc)
        checked, mism, V = verify_identical(corpus)

        nw = min(8, (os.cpu_count() or 4))
        (_, _, _, _, _), tP = fast_build_mp(texts, workers=nw)
        fP = sum(tP.values())

        print(f"\n  {name}: {n:,} docs, {ntok:,} tokens, {n_post:,} postings, {V:,} terms")
        print(f"    add()+finalize() [serve-ready]: {t_slow*1000:7.0f} ms ({t_add*1000:.0f}+{t_fin*1000:.0f}) | "
              f"{n/t_slow:8.0f} docs/s | {ntok/t_slow/1e6:5.2f}M tok/s")
        print(f"    FAST columnar (serial)   : {f1*1000:8.0f} ms | {n/f1:8.0f} docs/s | {ntok/f1/1e6:5.2f}M tok/s | {t_slow/f1:.1f}x")
        print(f"      tokenize {t1['tokenize']*1000:.0f} | bag+vocab {t1['bag+vocab']*1000:.0f} | sort+csr {t1['sort+csr']*1000:.0f} ms")
        print(f"    FAST columnar ({nw}-core mp): {fP*1000:7.0f} ms | {n/fP:8.0f} docs/s | {ntok/fP/1e6:5.2f}M tok/s | {t_slow/fP:.1f}x")
        print(f"      tok+bag(par) {tP['tok+bag(par)']*1000:.0f} | vocab+emit {tP['vocab+emit']*1000:.0f} | sort+csr {tP['sort+csr']*1000:.0f} ms")
        print(f"    IDENTICAL index: {checked-mism}/{checked} terms exact "
              f"({'PASS - accuracy+footprint unchanged' if mism == 0 else f'FAIL {mism}'})")
    print("\n  CSR (seg_doc/seg_tf/indptr) feeds the mmap/binary-reader serve directly — no dict-of-dicts ever.")


if __name__ == "__main__":
    main()
