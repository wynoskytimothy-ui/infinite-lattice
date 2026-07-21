"""
aethos_bmw.py -- exact Block-Max WAND serve, extracted from the verified prototype (_splade_bmw.py) as an importable
kernel for EdgeRAG/EdgeRAG-2. BMW is an EXACT top-k algorithm: it only skips docs whose block-max upper bound provably
cannot reach the current k-th score, so it is recall-LOSSLESS by construction while pruning most postings. Its home turf
is few-term queries over doc-sorted, non-negative-impact postings -- i.e. the lexical BM25 first stage (the MARCO serve).

Correctness crux (kept from the prototype): the block-max UB at a pivot sums over ALL query terms with cursor <= pivot_doc
(rmax, not just the pivot index). UBs are valid because per-posting impacts are >= 0.
"""
from __future__ import annotations
import numpy as np
from numba import njit


@njit(cache=True)
def _build_blocks(indptr, seg_doc, seg_w, B, blk_indptr, blk_maxdoc, blk_maxw, term_maxw):
    V = indptr.size - 1
    for t in range(V):
        a = indptr[t]; e = indptr[t + 1]; bi = blk_indptr[t]; tm = 0.0; bs = a
        while bs < e:
            be = bs + B
            if be > e: be = e
            mx = 0.0
            for p in range(bs, be):
                w = seg_w[p]
                if w > mx: mx = w
            blk_maxdoc[bi] = seg_doc[be - 1]                 # docs ascending -> last is the max doc-id in block
            blk_maxw[bi] = mx
            if mx > tm: tm = mx
            bi += 1; bs = be
        term_maxw[t] = tm


@njit(cache=True, inline="always")
def _heap_push(hs, hd, n, s, d):
    hs[n] = s; hd[n] = d; i = n
    while i > 0:
        p = (i - 1) // 2
        if hs[p] > hs[i]:
            hs[p], hs[i] = hs[i], hs[p]; hd[p], hd[i] = hd[i], hd[p]; i = p
        else: break
    return n + 1


@njit(cache=True, inline="always")
def _heap_replace(hs, hd, k, s, d):
    hs[0] = s; hd[0] = d; i = 0
    while True:
        l = 2 * i + 1; r = 2 * i + 2; sm = i
        if l < k and hs[l] < hs[sm]: sm = l
        if r < k and hs[r] < hs[sm]: sm = r
        if sm != i:
            hs[i], hs[sm] = hs[sm], hs[i]; hd[i], hd[sm] = hd[sm], hd[i]; i = sm
        else: break


@njit(cache=True, inline="always")
def _lb(a, lo, hi, target):
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] < target: lo = mid + 1
        else: hi = mid
    return lo


@njit(cache=True, inline="always")
def _block_of(blk_maxdoc, blo, bhi, pivot):
    lo = blo; hi = bhi - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if blk_maxdoc[mid] < pivot: lo = mid + 1
        else: hi = mid
    return lo


@njit(cache=True)
def _bmw(qt, qw, k, indptr, seg_doc, seg_w, tmaxw, blk_indptr, blk_maxdoc, blk_maxw):
    m = qt.size
    beg = np.empty(m, np.int64); end = np.empty(m, np.int64); cur = np.empty(m, np.int64)
    wv = np.empty(m, np.float64); umax = np.empty(m, np.float64)
    bb = np.empty(m, np.int64); be = np.empty(m, np.int64)
    for i in range(m):
        t = qt[i]
        beg[i] = indptr[t]; end[i] = indptr[t + 1]; cur[i] = beg[i]
        wv[i] = qw[i]; umax[i] = qw[i] * tmaxw[t]
        bb[i] = blk_indptr[t]; be[i] = blk_indptr[t + 1]
    order = np.arange(m)
    hs = np.empty(k, np.float64); hd = np.empty(k, np.int64); hn = 0
    theta = 0.0
    BIG = np.int64(1) << 62
    while True:
        for a in range(1, m):                               # adaptive insertion sort (order stays near-sorted)
            key = order[a]
            kd = seg_doc[cur[key]] if cur[key] < end[key] else BIG
            b = a - 1
            while b >= 0:
                ob = order[b]
                od = seg_doc[cur[ob]] if cur[ob] < end[ob] else BIG
                if od > kd:
                    order[b + 1] = order[b]; b -= 1
                else: break
            order[b + 1] = key
        i0 = order[0]
        d0 = seg_doc[cur[i0]] if cur[i0] < end[i0] else BIG
        if d0 == BIG: break
        acc = 0.0; piv = -1
        for r in range(m):
            i = order[r]
            cd = seg_doc[cur[i]] if cur[i] < end[i] else BIG
            if cd == BIG: break
            acc += umax[i]
            if acc >= theta: piv = r; break
        if piv == -1: break
        ip = order[piv]; pivot_doc = seg_doc[cur[ip]]
        rmax = piv
        for r in range(piv + 1, m):
            i = order[r]
            cd = seg_doc[cur[i]] if cur[i] < end[i] else BIG
            if cd <= pivot_doc: rmax = r
            else: break
        bub = 0.0
        for r in range(rmax + 1):
            i = order[r]
            blk = _block_of(blk_maxdoc, bb[i], be[i], pivot_doc)
            bub += wv[i] * blk_maxw[blk]
        if bub < theta:
            nd = BIG
            for r in range(rmax + 1):
                i = order[r]
                blk = _block_of(blk_maxdoc, bb[i], be[i], pivot_doc)
                cand = blk_maxdoc[blk] + 1
                if cand < nd: nd = cand
            if rmax + 1 < m:
                inx = order[rmax + 1]
                nxt = seg_doc[cur[inx]] if cur[inx] < end[inx] else BIG
                if nxt < nd: nd = nxt
            if nd <= pivot_doc: nd = pivot_doc + 1
            best = -1; bmx = -1.0
            for r in range(rmax + 1):
                i = order[r]
                cd = seg_doc[cur[i]] if cur[i] < end[i] else BIG
                if cd < nd and umax[i] > bmx: bmx = umax[i]; best = i
            if best == -1: best = ip
            cur[best] = _lb(seg_doc, cur[best], end[best], nd)
            continue
        if d0 == pivot_doc:
            s = 0.0
            for r in range(m):
                i = order[r]
                cd = seg_doc[cur[i]] if cur[i] < end[i] else BIG
                if cd != pivot_doc: break
                s += wv[i] * np.float64(seg_w[cur[i]]); cur[i] += 1
            if hn < k:
                hn = _heap_push(hs, hd, hn, s, pivot_doc)
                if hn == k: theta = hs[0]
            elif s > theta:
                _heap_replace(hs, hd, k, s, pivot_doc); theta = hs[0]
        else:
            best = -1; bmx = -1.0
            for r in range(piv):
                i = order[r]
                cd = seg_doc[cur[i]] if cur[i] < end[i] else BIG
                if cd < pivot_doc and umax[i] > bmx: bmx = umax[i]; best = i
            if best == -1: best = i0
            cur[best] = _lb(seg_doc, cur[best], end[best], pivot_doc)
    rd = np.empty(hn, np.int64); rs = np.empty(hn, np.float64)
    for i in range(hn): rd[i] = hd[i]; rs[i] = hs[i]
    return rd, rs


def build_blockmax(indptr, seg_doc, seg_w, B=128):
    """Build the block-max sidecar for a doc-sorted, non-negative-impact CSR. Returns a dict of arrays for _bmw."""
    indptr = np.ascontiguousarray(indptr, np.int64)
    seg_doc = np.ascontiguousarray(seg_doc, np.int64)
    seg_w = np.ascontiguousarray(seg_w, np.float32)
    V = indptr.size - 1; df = np.diff(indptr)
    nblk = (df + B - 1) // B
    blk_indptr = np.concatenate([[0], np.cumsum(nblk)]).astype(np.int64)
    total = int(blk_indptr[-1])
    blk_maxdoc = np.zeros(total, np.int64); blk_maxw = np.zeros(total, np.float32); term_maxw = np.zeros(V, np.float32)
    _build_blocks(indptr, seg_doc, seg_w, B, blk_indptr, blk_maxdoc, blk_maxw, term_maxw)
    return dict(indptr=indptr, seg_doc=seg_doc, seg_w=seg_w, term_maxw=term_maxw,
                blk_indptr=blk_indptr, blk_maxdoc=blk_maxdoc, blk_maxw=blk_maxw, B=B)


def bmw_search(qt, qw, blk, k=10):
    """Exact top-k over the block-max index. qt=int64 term ids, qw=float64 query weights. Returns (doc_idx, scores)."""
    if len(qt) == 0:
        return np.empty(0, np.int64), np.empty(0)
    qt = np.ascontiguousarray(qt, np.int64); qw = np.ascontiguousarray(qw, np.float64)
    rd, rs = _bmw(qt, qw, k, blk["indptr"], blk["seg_doc"], blk["seg_w"], blk["term_maxw"],
                  blk["blk_indptr"], blk["blk_maxdoc"], blk["blk_maxw"])
    o = np.argsort(rs)[::-1]
    return rd[o], rs[o]
