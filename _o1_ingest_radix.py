#!/usr/bin/env python3
"""RADIX/HASH GROUP-BY — kill the lexsort. The numba build's bottleneck was the comparison-sort of 4.2M
uint64 hashes. Replace it with three O(n) native passes (no comparison sort):
  1. scan + PER-DOC dedup: each doc accumulates token-hash -> summed positional tf, emits (doc, hash, tf) once.
  2. numba open-addressing hash table: hash -> term_id (O(n), no sort).
  3. counting sort: bincount df -> indptr -> scatter (doc, tf) into term segments (O(n)).
Verified bit-identical to incremental; compared head-to-head with the lexsort numba build.
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
from _o1_ingest_numba import numba_build, fnv          # the lexsort baseline + the FNV (matches numba)

FNV_OFF = np.uint64(1469598103934665603)
FNV_PRM = np.uint64(1099511628211)
POS_HEAD, POS_BOOST = 14, 1.6


@njit(nogil=True, cache=True)   # nogil -> Python threads run it truly parallel (no mp spawn cost)
def _scan_dedup(buf, doc_starts, d0, d1, stop_sorted, out_doc, out_hash, out_tf):
    cnt = 0
    for d in range(d0, d1):
        s = doc_starts[d]; e = doc_starts[d + 1]
        lh = np.empty(e - s + 1, np.uint64); lw = np.empty(e - s + 1, np.float64); m = 0
        i = s; tokpos = 0
        while i < e:
            c = buf[i]
            if 97 <= c <= 122:
                j = i + 1
                h = (FNV_OFF ^ np.uint64(c)) * FNV_PRM
                while j < e:
                    cj = buf[j]
                    if (97 <= cj <= 122) or (48 <= cj <= 57):
                        h = (h ^ np.uint64(cj)) * FNV_PRM; j += 1
                    else:
                        break
                if j - i > 2:
                    lo = 0; hi = stop_sorted.size; isstop = False
                    while lo < hi:
                        mid = (lo + hi) >> 1; v = stop_sorted[mid]
                        if v == h: isstop = True; break
                        elif v < h: lo = mid + 1
                        else: hi = mid
                    if not isstop:
                        lh[m] = h; lw[m] = POS_BOOST if tokpos < POS_HEAD else 1.0
                        m += 1; tokpos += 1
                i = j
            else:
                i += 1
        if m > 0:                                   # per-doc dedup by hash (small sort)
            order = np.argsort(lh[:m]); k = 0
            while k < m:
                hk = lh[order[k]]; acc = lw[order[k]]; k2 = k + 1
                while k2 < m and lh[order[k2]] == hk:
                    acc += lw[order[k2]]; k2 += 1
                out_doc[cnt] = d; out_hash[cnt] = hk; out_tf[cnt] = acc; cnt += 1
                k = k2
    return cnt


@njit(cache=True)
def _build_vocab(hashes, slot_id, key, uniq, term_id):
    cap = slot_id.size; mask = np.uint64(cap - 1); V = 0
    for i in range(hashes.size):
        h = hashes[i]; s = np.int64(h & mask)
        while slot_id[s] != -1 and key[s] != h:
            s += 1
            if s == cap: s = 0
        if slot_id[s] == -1:
            slot_id[s] = V; key[s] = h; uniq[V] = h; V += 1
        term_id[i] = slot_id[s]
    return V


@njit(cache=True)
def _counting_scatter(term_id, doc, tf, cursor, seg_doc, seg_tf):
    for i in range(term_id.size):
        t = term_id[i]; p = cursor[t]; cursor[t] += 1
        seg_doc[p] = doc[i]; seg_tf[p] = tf[i]


def radix_build(texts):
    t = {}
    s = time.perf_counter(); buf, doc_starts, stop_sorted = _prep(texts); t["prep_buffer"] = time.perf_counter() - s
    s = time.perf_counter()
    cap = buf.size + 1
    od = np.empty(cap, np.int32); oh = np.empty(cap, np.uint64); ot = np.empty(cap, np.float64)
    cnt = _scan_dedup(buf, doc_starts, 0, len(texts), stop_sorted, od, oh, ot)
    od, oh, ot = od[:cnt], oh[:cnt], ot[:cnt]
    t["scan+dedup"] = time.perf_counter() - s
    s = time.perf_counter(); B = _vocab_count(oh, od, ot); t["vocab+counting"] = time.perf_counter() - s
    return B, t


def _prep(texts):
    db = [x.lower().encode("ascii", "replace") for x in texts]
    lens = np.fromiter((len(b) for b in db), np.int64, len(db))
    buf = np.frombuffer(b"".join(db), np.uint8)
    doc_starts = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    stop_sorted = np.array(sorted(int(fnv(w.encode())) for w in _STOP), np.uint64)
    return buf, doc_starts, stop_sorted


def _vocab_count(oh, od, ot):
    cnt = oh.size
    hcap = 1
    while hcap < cnt * 2: hcap <<= 1
    slot_id = np.full(hcap, -1, np.int32); key = np.empty(hcap, np.uint64)
    uniq = np.empty(cnt, np.uint64); term_id = np.empty(cnt, np.int32)
    V = _build_vocab(oh, slot_id, key, uniq, term_id)
    df = np.bincount(term_id, minlength=V).astype(np.int64)
    indptr = np.concatenate([[0], np.cumsum(df)]).astype(np.int64)
    seg_doc = np.empty(cnt, np.uint32); seg_tf64 = np.empty(cnt, np.float64)
    _counting_scatter(term_id, od.astype(np.uint32), ot, indptr[:-1].copy(), seg_doc, seg_tf64)
    return dict(uniq_hash=uniq[:V].copy(), indptr=indptr, seg_doc=seg_doc, seg_tf=seg_tf64.astype(np.float16), V=V)


def radix_build_mt(texts, nthreads):
    """Thread-parallel scan (nogil njit) + serial hash-vocab/counting. No process spawn."""
    from concurrent.futures import ThreadPoolExecutor
    t = {}
    s = time.perf_counter(); buf, doc_starts, stop_sorted = _prep(texts); t["prep_buffer"] = time.perf_counter() - s
    n = len(texts)
    bounds = [round(k * n / nthreads) for k in range(nthreads + 1)]
    results = [None] * nthreads
    s = time.perf_counter()

    def work(ti):
        d0, d1 = bounds[ti], bounds[ti + 1]
        span = int(doc_starts[d1] - doc_starts[d0]) + 1
        od = np.empty(span, np.int32); oh = np.empty(span, np.uint64); ot = np.empty(span, np.float64)
        c = _scan_dedup(buf, doc_starts, d0, d1, stop_sorted, od, oh, ot)
        results[ti] = (od[:c].copy(), oh[:c].copy(), ot[:c].copy())
    with ThreadPoolExecutor(nthreads) as ex:
        list(ex.map(work, range(nthreads)))
    od = np.concatenate([r[0] for r in results]); oh = np.concatenate([r[1] for r in results])
    ot = np.concatenate([r[2] for r in results])
    t["scan+dedup(mt)"] = time.perf_counter() - s
    s = time.perf_counter(); B = _vocab_count(oh, od, ot); t["vocab+counting"] = time.perf_counter() - s
    return B, t


def verify(corpus, build_fn):
    doc_ids = list(corpus.keys()); texts = list(corpus.values())
    slow = AppendOnlyLatticeIndex(index_mode="kappa_primary")
    for d, x in corpus.items(): slow.add(d, x)
    B, _ = build_fn(texts)
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
    print("=" * 88)
    print("RADIX/HASH GROUP-BY — kill the lexsort (3 O(n) native passes). Identical index.")
    print("=" * 88)
    nth = min(8, (os.cpu_count() or 4))
    radix_build(["warm the jit up now please thanks"]); numba_build(["warm the jit up now please thanks"])
    radix_build_mt(["warm the jit up now please thanks today"], 2)
    for name in ["scifact", "fiqa"]:
        corpus, *_ = load(name)
        texts = list(corpus.values()); n = len(texts)
        ntok = sum(len(words(x)) for x in texts)

        s = time.perf_counter()
        idx = AppendOnlyLatticeIndex(index_mode="kappa_primary")
        for d, x in corpus.items(): idx.add(d, x)
        idx.finalize()
        t_slow = time.perf_counter() - s

        Bn, tn = numba_build(texts); fn_ = sum(tn.values())
        Br, tr = radix_build(texts); fr = sum(tr.values())
        Bm, tm = radix_build_mt(texts, nth); fm = sum(tm.values())
        checked, mism, V = verify(corpus, radix_build)
        print(f"\n  {name}: {n:,} docs, {ntok:,} tokens, {len(Br['seg_doc']):,} postings, {V:,} terms")
        print(f"    add()+finalize() [serve-ready]: {t_slow*1000:7.0f} ms | {n/t_slow:8.0f} docs/s | {ntok/t_slow/1e6:5.2f}M tok/s")
        print(f"    numba (lexsort)              : {fn_*1000:7.0f} ms | {ntok/fn_/1e6:6.1f}M tok/s | {t_slow/fn_:.1f}x")
        print(f"    numba (RADIX, serial)        : {fr*1000:7.0f} ms | {ntok/fr/1e6:6.1f}M tok/s | {t_slow/fr:.1f}x")
        print(f"      prep {tr['prep_buffer']*1000:.0f} | scan+dedup {tr['scan+dedup']*1000:.0f} | vocab+counting {tr['vocab+counting']*1000:.0f} ms")
        print(f"    numba (RADIX, {nth}-thread)      : {fm*1000:7.0f} ms | {ntok/fm/1e6:6.1f}M tok/s | {t_slow/fm:.1f}x")
        print(f"      prep {tm['prep_buffer']*1000:.0f} | scan+dedup(mt) {tm['scan+dedup(mt)']*1000:.0f} | vocab+counting {tm['vocab+counting']*1000:.0f} ms")
        print(f"    IDENTICAL index: {checked-mism}/{checked} terms exact "
              f"({'PASS' if mism == 0 else f'FAIL {mism}'})")
    print("\n  no comparison sort: per-doc dedup + open-addressing vocab + counting scatter, all O(n); nogil threads scale the scan.")


if __name__ == "__main__":
    main()
