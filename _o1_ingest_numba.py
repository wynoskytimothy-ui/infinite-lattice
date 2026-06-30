#!/usr/bin/env python3
"""NUMBA INGEST — the big swing: tokenize + vocab leave Python entirely.

A numba byte-scanner walks the lowercased ASCII buffer, finds [a-z][a-z0-9]{2,} tokens, computes a 64-bit
FNV-1a hash per token INLINE (no Python str objects), filters stopwords by hash, and emits
(doc, hash, positional-weight) as native arrays. Vocab = unique uint64 hashes (np.unique). The whole
tokenize+vocab stage becomes native scan + integer sort. Identical-index gate verifies bit-for-bit equality
(hash collisions among ~70k terms in 2^64 are ~1e-10, so the index is functionally identical).
"""
import os, sys, time
os.environ.setdefault("PYTHONUTF8", "1")
import numpy as np
from numba import njit
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from aethos_append_index import AppendOnlyLatticeIndex
from scripts.bench_supervised_bridges import load
from _fast_tok import _STOP, words

FNV_OFF = np.uint64(1469598103934665603)
FNV_PRM = np.uint64(1099511628211)
POS_HEAD, POS_BOOST = 14, 1.6


_MASK = 0xFFFFFFFFFFFFFFFF


def fnv(bs):
    """Pure-int 64-bit FNV-1a (masked) — matches numba's np.uint64 wraparound exactly, no overflow warning."""
    h = 1469598103934665603
    for c in bs:
        h = ((h ^ c) * 1099511628211) & _MASK
    return h


@njit(cache=True)
def _scan(buf, doc_starts, stop_sorted, out_doc, out_hash, out_w):
    cnt = 0
    ndoc = doc_starts.size - 1
    for d in range(ndoc):
        e = doc_starts[d + 1]; i = doc_starts[d]; tokpos = 0
        while i < e:
            c = buf[i]
            if 97 <= c <= 122:                       # token must start [a-z]
                j = i + 1
                h = (FNV_OFF ^ np.uint64(c)) * FNV_PRM
                while j < e:
                    cj = buf[j]
                    if (97 <= cj <= 122) or (48 <= cj <= 57):
                        h = (h ^ np.uint64(cj)) * FNV_PRM
                        j += 1
                    else:
                        break
                if j - i > 2:                        # len>2 (matches words())
                    lo = 0; hi = stop_sorted.size; isstop = False   # binary search stopword hashes
                    while lo < hi:
                        mid = (lo + hi) >> 1
                        if stop_sorted[mid] == h: isstop = True; break
                        elif stop_sorted[mid] < h: lo = mid + 1
                        else: hi = mid
                    if not isstop:
                        out_doc[cnt] = d; out_hash[cnt] = h
                        out_w[cnt] = POS_BOOST if tokpos < POS_HEAD else 1.0
                        cnt += 1; tokpos += 1
                i = j
            else:
                i += 1
    return cnt


def numba_build(texts):
    t = {}
    s = time.perf_counter()
    # 'replace' (non-ascii -> '?', a delimiter) NOT 'ignore' (which would drop the char and MERGE the
    # surrounding letters into one token -> diverges from the regex, which splits on non-[a-z0-9]).
    db = [x.lower().encode("ascii", "replace") for x in texts]
    lens = np.fromiter((len(b) for b in db), np.int64, len(db))
    buf = np.frombuffer(b"".join(db), np.uint8)
    doc_starts = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    stop_sorted = np.array(sorted(int(fnv(w.encode())) for w in _STOP), np.uint64)
    t["prep_buffer"] = time.perf_counter() - s

    s = time.perf_counter()
    cap = buf.size + 1
    out_doc = np.empty(cap, np.int32); out_hash = np.empty(cap, np.uint64); out_w = np.empty(cap, np.float64)
    cnt = _scan(buf, doc_starts, stop_sorted, out_doc, out_hash, out_w)
    od, oh, ow = out_doc[:cnt], out_hash[:cnt], out_w[:cnt]
    t["scan+hash"] = time.perf_counter() - s

    s = time.perf_counter()
    order = np.lexsort((od, oh))                     # group by (hash, doc)
    oh_s, od_s, ow_s = oh[order], od[order], ow[order]
    pair_change = np.ones(cnt, bool)
    np.logical_or(oh_s[1:] != oh_s[:-1], od_s[1:] != od_s[:-1], out=pair_change[1:])
    starts = np.nonzero(pair_change)[0]
    tf = np.add.reduceat(ow_s, starts).astype(np.float16)   # tf per (hash,doc)
    u_hash = oh_s[starts]; u_doc = od_s[starts].astype(np.uint32)
    # term ids = dense rank of unique hash (u_hash already sorted by the lexsort major key)
    term_change = np.ones(u_hash.size, bool); term_change[1:] = u_hash[1:] != u_hash[:-1]
    term_id = np.cumsum(term_change) - 1
    V = int(term_id[-1]) + 1 if u_hash.size else 0
    df = np.bincount(term_id, minlength=V).astype(np.int64)
    indptr = np.concatenate([[0], np.cumsum(df)]).astype(np.int64)
    uniq_hash = u_hash[term_change]                  # hash per term_id (for query/verify)
    t["group_csr"] = time.perf_counter() - s
    return dict(uniq_hash=uniq_hash, indptr=indptr, seg_doc=u_doc, seg_tf=tf, V=V), t


def verify(corpus):
    doc_ids = list(corpus.keys()); texts = list(corpus.values())
    slow = AppendOnlyLatticeIndex(index_mode="kappa_primary")
    for d, x in corpus.items(): slow.add(d, x)
    B, _ = numba_build(texts)
    h2row = {int(h): i for i, h in enumerate(B["uniq_hash"])}
    mism = checked = 0
    for (view, tok), p in slow.token_prime.items():
        if view != "w": continue
        checked += 1
        slow_pl = {d: float(np.float16(w)) for d, w in slow.postings[p].items()}
        r = h2row.get(int(fnv(tok.encode())))
        fast_pl = {}
        if r is not None:
            a, e = int(B["indptr"][r]), int(B["indptr"][r + 1])
            fast_pl = {doc_ids[int(B["seg_doc"][i])]: float(B["seg_tf"][i]) for i in range(a, e)}
        if slow_pl != fast_pl: mism += 1
    return checked, mism, B["V"]


def main():
    print("=" * 86)
    print("NUMBA INGEST — native byte-scan tokenize+hash (identical index). The big ingest lever.")
    print("=" * 86)
    # warm the JIT
    numba_build(["warm up the jit compiler now please"])
    for name in ["scifact", "fiqa"]:
        corpus, *_ = load(name)
        texts = list(corpus.values()); n = len(texts)
        ntok = sum(len(words(x)) for x in texts)

        s = time.perf_counter()
        idx = AppendOnlyLatticeIndex(index_mode="kappa_primary")
        for d, x in corpus.items(): idx.add(d, x)
        idx.finalize()
        t_slow = time.perf_counter() - s

        B, t = numba_build(texts); f = sum(t.values())
        checked, mism, V = verify(corpus)
        print(f"\n  {name}: {n:,} docs, {ntok:,} tokens, {len(B['seg_doc']):,} postings, {V:,} terms")
        print(f"    add()+finalize() [serve-ready]: {t_slow*1000:7.0f} ms | {n/t_slow:8.0f} docs/s | {ntok/t_slow/1e6:5.2f}M tok/s")
        print(f"    NUMBA build:                  {f*1000:7.0f} ms | {n/f:8.0f} docs/s | {ntok/f/1e6:6.1f}M tok/s | {t_slow/f:.1f}x")
        print(f"      prep_buf {t['prep_buffer']*1000:.0f} | scan+hash {t['scan+hash']*1000:.0f} | group_csr {t['group_csr']*1000:.0f} ms")
        print(f"    IDENTICAL index: {checked-mism}/{checked} terms exact "
              f"({'PASS' if mism == 0 else f'FAIL {mism}'})")
    print("\n  tokenize+vocab now native uint64 (scan+hash). To project: M tok/s * your corpus = ingest time.")


if __name__ == "__main__":
    main()
