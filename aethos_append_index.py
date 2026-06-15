"""
aethos_append_index.py - append-only multi-view lattice retrieval index.

The training paradigm (Tests 54-56) turned into a retrieval engine:

  - APPEND-ONLY: each new document is ingested as new prime addresses; adding
    a document only appends to posting lists - old entries are never touched,
    so there is no reindex and no retrain (Test 54 continual learning).
  - MULTI-VIEW TOKENS: every term flows through several tokenization "gears"
    (word, char-trigram, prefix), each on its own prime namespace, so a term
    is findable through any view - robust to typos and variants (Test 56).
  - STABLE ADDRESSING: a (view, token) gets a prime on first sight and keeps
    it forever; vocabulary grows monotonically (Test 55 - the counting set is
    a knob; here we use primes for the multiplicative composite).

Scoring is multi-view BM25 with per-view weights; idf is read from the live
document frequency at query time, so deletions and additions are reflected
without rebuilding anything.

Query speed/footprint (all lossless, measured in scripts/):
  - tri_df_frac caps the high-df, ~0-idf char-trigrams (the longest lists) at
    score time - 2.6-3.6x faster, recall holds (bench_fast_query.py).
  - finalize() builds a compact numpy dense fast path (uint16 doc-ids, float16
    tf, high-df trigrams pruned): ~15x faster, ~4x smaller, metric-lossless
    (identical nDCG/Recall; 99.9-100% top-10 overlap, dense_tf_dtype trades
    bytes vs the tiny boundary tie-swap). search() uses it automatically;
    add()/remove() invalidate it (append stays O(1); re-finalize in a batch).
    bm25_delta / containment_bonus are dict-path only - finalize() stays on the
    dict path if either is set.

Supervised relevance bridges (learn query->gold-doc term links by counting
qrels, deterministic/append-only) live in scripts/bench_supervised_bridges.py
and rerank/expand on top of this index - the accuracy lever.
"""

from __future__ import annotations

import heapq
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field

try:
    import numpy as np            # optional: enables the finalize() dense fast path
except ImportError:               # pragma: no cover
    np = None

from core.primes import chain_primes

_TOK = re.compile(r"[a-z][a-z0-9]+")
_STOP = set("a an and are as at be by for from has he in is it its of on that the "
            "to was were will with this these those which who we our you your they "
            "their them not no can may also been being have had but or if than then "
            "so such into over under more most some any all".split())


def words(text):
    return [w for w in _TOK.findall(text.lower()) if w not in _STOP and len(w) > 2]


# tokenization gears: name -> (weight, fn(word) -> iterable of tokens)
def _v_word(w):
    return (("w", w),)


def _v_tri(w):
    p = f"^{w}$"
    return (("3", p[i:i + 3]) for i in range(len(p) - 2))


def _v_prefix(w):
    return (("p", w[:4]),)


GEARS = {"word": (1.0, _v_word), "tri": (0.30, _v_tri), "prefix": (0.20, _v_prefix)}
# weights pulled out so the optimized _multiview stays in sync with GEARS
_GW, _GT, _GP = GEARS["word"][0], GEARS["tri"][0], GEARS["prefix"][0]


@dataclass
class AppendOnlyLatticeIndex:
    k1: float = 1.2
    b: float = 0.75
    # levers studied from the v10/UltraFast version (SciFact 0.78). Ablation on
    # THIS multi-view index (scripts/bench_append_index.py) showed only the
    # positional lever transfers (+0.007 nDCG); BM25+ delta and containment were
    # tuned for the pure-word pipeline and HURT the char-gram gears, so they
    # default off here. The remaining gap to 0.78 is the geodesic/formula rerank.
    bm25_delta: float = 0.0          # off: hurt the multi-view index in ablation
    positional: bool = True          # title / lead words weighted higher (HELPS)
    pos_head: int = 14               # leading words counted as "title/lead"
    pos_boost: float = 1.6           # multiplier for head words
    containment_bonus: float = 0.0   # off: hurt the multi-view index in ablation
    tri_df_frac: float = 0.5         # skip query trigrams with df > frac*N at score
    #   (high-df trigrams = ~0 idf but longest posting lists; capping is near-
    #   lossless and 2.6-3.6x faster - scripts/bench_fast_query.py. <1.0 enables.)
    dense_tf_dtype: str = "f16"      # finalize() tf storage: f16 (2B, metric-lossless,
    #   default) | f32 (4B, ~1e-7) | f64 (8B). doc-ids are uint16/uint32 by N.
    token_prime: dict = field(default_factory=dict)     # (view,token) -> prime
    postings: dict = field(default_factory=lambda: defaultdict(dict))  # prime -> {doc: tf}
    df: dict = field(default_factory=lambda: defaultdict(int))         # prime -> live df
    doc_len: dict = field(default_factory=dict)
    doc_words: dict = field(default_factory=dict)        # doc -> set of word prims (containment)
    alive: set = field(default_factory=set)
    _primes: tuple = field(default_factory=lambda: chain_primes(200000))
    _next: int = 0
    _total_len: int = 0
    _n_removed: int = 0              # 0 => no tombstones => skip alive-check in hot loop
    _gc: dict = field(default_factory=dict)   # word -> cached gear keys (ingest speed)
    _dense_ready: bool = False       # True after finalize(); invalidated by add/remove

    # ---- stable, append-only prime allocation ----
    def _grow_primes(self, need):
        """Extend the prime pool when a large vocabulary exhausts it (cached sieve;
        deterministic, so any two pool sizes agree on prime[i] - shard-safe)."""
        if need >= len(self._primes):
            self._primes = chain_primes(max(need + 1, len(self._primes) * 2))
        return self._primes

    def _prime_for(self, tok, create=True):
        p = self.token_prime.get(tok)
        if p is None and create:
            i = len(self.token_prime)
            primes = self._primes if i < len(self._primes) else self._grow_primes(i)
            p = primes[i]                              # == _next; shared-vocab safe
            self.token_prime[tok] = p
            self._next = i + 1
        return p

    def _multiview(self, text, positional=False):
        """word -> all gear tokens, per-gear weight + positional on the WORD gear.

        Optimized inline of the GEARS pipeline (must match _v_word/_v_tri/
        _v_prefix): each distinct word's trigram+prefix KEYS are cached in _gc
        and built once, not per occurrence (trigrams are ~80% of the tokens).
        Output is identical to the gear loop."""
        cache = self._gc
        bag = {}
        bget = bag.get
        pos_head, pos_boost = self.pos_head, self.pos_boost
        for i, w in enumerate(words(text)):
            pos_w = pos_boost if (positional and i < pos_head) else 1.0
            kw = ("w", w)
            bag[kw] = bget(kw, 0.0) + _GW * pos_w                # word gear
            g = cache.get(w)
            if g is None:
                p = "^" + w + "$"
                g = (tuple(("3", p[j:j + 3]) for j in range(len(p) - 2)), ("p", w[:4]))
                cache[w] = g
            tris, pk = g
            for k in tris:
                bag[k] = bget(k, 0.0) + _GT                      # trigram gear
            bag[pk] = bget(pk, 0.0) + _GP                        # prefix gear
        return bag

    # ---- ingest: APPEND only ----
    def add(self, doc_id, text):
        if doc_id in self.alive:
            return
        bag = self._multiview(text, positional=self.positional)
        tp, postings, df, primes = self.token_prime, self.postings, self.df, self._primes
        dl = 0.0
        word_prims = set()
        wadd = word_prims.add
        for tok, wt in bag.items():
            p = tp.get(tok)
            if p is None:                          # inline _prime_for (shared-vocab safe)
                i = len(tp)
                if i >= len(primes):
                    primes = self._grow_primes(i)  # extend pool for large vocab
                p = primes[i]
                tp[tok] = p
            postings[p][doc_id] = wt               # append a posting
            df[p] += 1
            dl += wt
            if tok[0] == "w":
                wadd(p)
        self._next = len(tp)
        self.doc_len[doc_id] = dl
        self.doc_words[doc_id] = word_prims
        self.alive.add(doc_id)
        self._total_len += dl
        self._dense_ready = False              # stale: re-finalize() to re-enable fast path

    # ---- delete: tombstone (postings stay; df/alive updated) ----
    def remove(self, doc_id):
        if doc_id not in self.alive:
            return
        self.alive.discard(doc_id)
        self._n_removed += 1
        self._dense_ready = False
        self._total_len -= self.doc_len.get(doc_id, 0.0)
        for p, plist in self.postings.items():
            if doc_id in plist:
                self.df[p] -= 1

    def _idf(self, p, N):
        return math.log(1 + (N - self.df[p] + 0.5) / (self.df[p] + 0.5))

    # ---- query: multi-view BM25+ over live docs, with containment ----
    def _score(self, query):
        N = max(1, len(self.alive))
        avgdl = self._total_len / N
        qbag = self._multiview(query)                  # query: no positional boost
        q_word_prims = set()
        if self.containment_bonus:                     # only needed for containment
            q_word_prims = {self._prime_for(t, create=False)
                            for t in qbag if t[0] == "w"}
            q_word_prims.discard(None)
        scores = defaultdict(float)
        # hoist the BM25 length-norm into per-doc constants: denom = tf + A + B*dl
        k1, b = self.k1, self.b
        A, Bc, k1p1 = k1 * (1 - b), k1 * b / avgdl, k1 + 1
        tri_cap = self.tri_df_frac * N if self.tri_df_frac < 1.0 else None
        no_removals = (self._n_removed == 0)
        alive, doc_len, df, postings = self.alive, self.doc_len, self.df, self.postings
        for tok, qwt in qbag.items():
            p = self.token_prime.get(tok)
            if p is None:
                continue
            dfp = df[p]
            if dfp == 0:
                continue
            if tri_cap is not None and tok[0] == "3" and dfp > tri_cap:
                continue                                  # drop low-idf, longest lists
            idf = math.log(1 + (N - dfp + 0.5) / (dfp + 0.5))
            delta = self.bm25_delta if tok[0] == "w" else 0.0   # BM25+ floor on words
            pl = postings[p]
            if delta:
                cf = qwt * idf
                for doc, tf in pl.items():
                    if no_removals or doc in alive:
                        scores[doc] += cf * (tf * k1p1 / (tf + A + Bc * doc_len[doc]) + delta)
            else:
                c = qwt * idf * k1p1
                if no_removals:
                    for doc, tf in pl.items():
                        scores[doc] += c * tf / (tf + A + Bc * doc_len[doc])
                else:
                    for doc, tf in pl.items():
                        if doc in alive:
                            scores[doc] += c * tf / (tf + A + Bc * doc_len[doc])
        if self.containment_bonus and q_word_prims:
            for doc in list(scores):
                frac = len(q_word_prims & self.doc_words.get(doc, ())) / len(q_word_prims)
                scores[doc] *= (1.0 + self.containment_bonus * frac)
        return scores

    # ---- dense fast path: vectorize the SAME scoring (lossless, ~15x) ----
    def finalize(self, champion_m=None, global_avgdl=None):
        """Build the numpy query accelerator (call after a batch of add()).

        champion_m: if set, keep only each term's top-M docs by impact (champion
        lists). Query work becomes O(query_terms * M) - INDEPENDENT of corpus
        size, so sub-ms at ANY N - and the dense arrays shrink. Approximate: it
        can miss docs strong only on the score SUM (scifact M=500 ~0.695 nDCG vs
        0.702 full; M=2000 ~lossless). None (default) = exact lossless full scan,
        sub-ms only up to ~50k docs. scripts/bench_scaling.py.

        Same arithmetic as _score, but each term's posting traversal becomes a
        vectorized scatter-add, so the Python loop runs once per query TERM, not
        per posting. METRIC-LOSSLESS: nDCG/Recall identical; top-10 set overlap
        99.9-100% (boundary tie swaps from float ordering, ~1e-15 for f64 tf,
        ~1e-6 for the default f16 tf - scripts/bench_compress_dense.py). Appending
        or removing invalidates it (append stays O(1); re-vectorize in a batch).
        No-op without numpy. scripts/bench_numpy_scorer.py."""
        if np is None:
            return self
        if self.bm25_delta or self.containment_bonus:
            # the dense path scores only the core BM25 term; these levers live in
            # _score, so stay on the dict path rather than silently diverge.
            self._dense_ready = False
            return self
        N = max(1, len(self.alive))
        k1, b = self.k1, self.b
        avgdl = global_avgdl if global_avgdl is not None else self._total_len / N
        A, Bc = k1 * (1 - b), k1 * b / avgdl
        docs = list(self.alive)
        d2i = {d: i for i, d in enumerate(docs)}
        self._d_docs = docs
        self._d_denom = np.array([A + Bc * self.doc_len[d] for d in docs], dtype=np.float64)
        # compressed dense (lossless, ~3x smaller - scripts/bench_compress_dense.py):
        #   doc-id uint16 when <65536 docs (else uint32); tf float16; and skip the
        #   df>cap trigrams the scorer never traverses anyway (index-time prune).
        didx_dtype = np.uint16 if N < 65536 else np.uint32
        tf_dtype = {"f16": np.float16, "f32": np.float32, "f64": np.float64}.get(
            self.dense_tf_dtype, np.float16)
        tri_cap = self.tri_df_frac * N if self.tri_df_frac < 1.0 else None
        pdoc, ptf = {}, {}
        for (view, _tok), p in self.token_prime.items():
            pl = self.postings.get(p)
            if not pl:
                continue
            if tri_cap is not None and view == "3" and self.df[p] > tri_cap:
                continue                                   # never traversed -> don't store
            di = np.fromiter((d2i[d] for d in pl if d in d2i), dtype=didx_dtype)
            if not di.size:
                continue
            pdoc[p] = di
            ptf[p] = np.fromiter((pl[d] for d in pl if d in d2i),
                                 dtype=np.float64, count=di.size).astype(tf_dtype)
        if champion_m:                                  # bounded-work query for scale
            denom = self._d_denom
            for p, di in list(pdoc.items()):
                if di.size > champion_m:
                    tf = ptf[p].astype(np.float64)
                    top = np.argpartition(tf / (tf + denom[di]), -champion_m)[-champion_m:]
                    pdoc[p], ptf[p] = di[top], ptf[p][top]
        self._d_pdoc, self._d_ptf = pdoc, ptf
        self._gc = {}                                  # free the ingest gear cache
        self._dense_ready = True
        return self

    def _dense_score_array(self, query, gN=None, gdf=None):
        """Full dense lexical score vector (length = #alive docs), SAME math as
        _score. Requires finalize(). Powers search() and dense rerankers.

        gN / gdf: GLOBAL corpus size and df (prime->count) for shard scoring -
        a shard scores its docs with global stats so a fan-out + merge equals a
        single index exactly (see aethos_sharded_index.py). Default = local."""
        N = gN if gN is not None else max(1, len(self.alive))
        df = gdf if gdf is not None else self.df
        k1p1 = self.k1 + 1
        cap = self.tri_df_frac * N if self.tri_df_frac < 1.0 else None
        denom, pdoc, ptf = self._d_denom, self._d_pdoc, self._d_ptf
        scores = np.zeros(len(self._d_docs), dtype=np.float64)
        for tok, qwt in self._multiview(query).items():
            p = self.token_prime.get(tok)
            if p is None:
                continue
            dfp = df.get(p, 0)
            if dfp == 0:
                continue
            if cap is not None and tok[0] == "3" and dfp > cap:
                continue
            di = pdoc.get(p)
            if di is None:
                continue
            idf = math.log(1 + (N - dfp + 0.5) / (dfp + 0.5))
            tfa = ptf[p].astype(np.float64)            # float16 -> float64 for the math
            scores[di] += (qwt * idf * k1p1) * tfa / (tfa + denom[di])
        return scores

    def dense_scores(self, query):
        """(scores_array, docs) on the dense fast path - for rerankers that need
        the full lexical vector aligned to dense indices (e.g. supervised
        bridges). Requires finalize(); doc i in `docs` <-> scores_array[i]."""
        return self._dense_score_array(query), self._d_docs

    def dense_posting(self, prime):
        """(dense_doc_indices, tf_float64) for a prime, or None - for dense
        expansion (e.g. bridge target words). Requires finalize()."""
        di = self._d_pdoc.get(prime)
        if di is None:
            return None
        return di, self._d_ptf[prime].astype(np.float64)

    def _search_dense(self, query, k):
        scores = self._dense_score_array(query)
        kk = min(k, len(self._d_docs))
        part = np.argpartition(scores, -kk)[-kk:]
        part = part[np.argsort(scores[part])[::-1]]
        docs = self._d_docs
        return [docs[i] for i in part if scores[i] > 0.0]

    def search(self, query, k=10):
        if self._dense_ready:
            return self._search_dense(query, k)            # vectorized, ~15x, lossless
        scores = self._score(query)
        return heapq.nlargest(k, scores, key=scores.get)   # O(n) top-k, not O(n log n)

    # ---- manifold rerank: geodesic idea, native to the lattice ----
    def search_manifold(self, query, k=10, pool=60, beta=0.25, knn=20):
        """Conservative cluster-centrality boost on top of BM25 (geodesic idea,
        native to the lattice). Build a meet-overlap graph over the top BM25
        candidates (edges = shared idf-weighted word primes, a meet) and give
        each doc a small boost for being adjacent to OTHER high-BM25 docs - the
        relevant cluster lifts, BM25 stays dominant. No 24-D encoder, no index
        change (graph built at query time)."""
        base = self._score(query)
        if len(base) < 4:
            return sorted(base, key=lambda d: base[d], reverse=True)[:k]
        N = max(1, len(self.alive))
        cand = sorted(base, key=lambda d: base[d], reverse=True)[:pool]
        s0 = [base[d] for d in cand]
        mx = max(s0) or 1.0
        sn = [v / mx for v in s0]
        wp = [self.doc_words.get(d, set()) for d in cand]
        # each doc's connectivity to the high-scoring cluster (one hop)
        boost = [0.0] * len(cand)
        for i in range(len(cand)):
            sims = []
            for j in range(len(cand)):
                if i == j:
                    continue
                shared = wp[i] & wp[j]
                if shared:
                    sim = sum(self._idf(p, N) for p in shared)
                    sims.append((sim, j))
            sims.sort(reverse=True)
            tot = sum(w for w, _ in sims[:knn]) or 1.0
            boost[i] = sum(w * sn[j] for w, j in sims[:knn]) / tot   # weighted by BM25
        bmax = max(boost) or 1.0
        final = [s0[i] * (1.0 + beta * boost[i] / bmax) for i in range(len(cand))]
        order = sorted(range(len(cand)), key=lambda i: final[i], reverse=True)
        return [cand[i] for i in order[:k]]

    # ---- compact, still-appendable persistence ----
    def save(self, path):
        """Write a compact snapshot (CSR: per-prime doc-index segments, delta-
        encoded for zlib, + float16 tf; savez_compressed). Compacts tombstones
        (only alive docs written). Reload with load() - the result is fully
        appendable. Requires numpy. scripts/bench_persist.py."""
        if np is None:
            raise RuntimeError("save() requires numpy")
        docs = list(self.alive)
        d2i = {d: i for i, d in enumerate(docs)}
        idt = np.uint16 if len(docs) < 65536 else np.uint32
        primes, views, toks, indptr = [], [], [], [0]
        ind_parts, data_parts = [], []
        for (view, tok), p in self.token_prime.items():
            pl = self.postings.get(p)
            if not pl:
                continue
            seg = sorted((d2i[d], tf) for d, tf in pl.items() if d in d2i)
            if not seg:
                continue
            di = np.fromiter((i for i, _ in seg), dtype=idt, count=len(seg))
            primes.append(p); views.append(ord(view)); toks.append(tok)
            ind_parts.append(np.diff(di, prepend=idt(0)).astype(idt))   # delta -> zlib
            data_parts.append(np.fromiter((tf for _, tf in seg),
                                          dtype=np.float16, count=len(seg)))
            indptr.append(indptr[-1] + len(seg))
        cfg = {k: getattr(self, k) for k in
               ("k1", "b", "bm25_delta", "positional", "pos_head", "pos_boost",
                "containment_bonus", "tri_df_frac", "dense_tf_dtype")}
        np.savez_compressed(
            path,
            config=np.frombuffer(json.dumps(cfg).encode(), dtype=np.uint8),
            doc_ids=np.frombuffer("\n".join(docs).encode(), dtype=np.uint8),
            doc_len=np.array([self.doc_len[d] for d in docs], dtype=np.float32),
            tokens=np.frombuffer("\n".join(toks).encode(), dtype=np.uint8),
            views=np.array(views, dtype=np.uint8),
            primes=np.array(primes, dtype=np.uint32),
            indptr=np.array(indptr, dtype=np.int64),
            indices=np.concatenate(ind_parts) if ind_parts else np.zeros(0, idt),
            data=np.concatenate(data_parts) if data_parts else np.zeros(0, np.float16),
            scalars=np.array([self._next, self._total_len], dtype=np.float64),
        )
        return self

    @classmethod
    def load(cls, path, finalize=True):
        """Reconstruct an index written by save(): rebuilds the appendable dict
        structures and (by default) the dense fast path. Requires numpy."""
        if np is None:
            raise RuntimeError("load() requires numpy")
        fn = str(path) if str(path).endswith(".npz") else str(path) + ".npz"
        z = np.load(fn, allow_pickle=False)
        idx = cls(**json.loads(bytes(z["config"]).decode()))
        docs = bytes(z["doc_ids"]).decode().split("\n")
        toks = bytes(z["tokens"]).decode().split("\n")
        views, primes = z["views"], z["primes"]
        indptr, indices, data = z["indptr"], z["indices"], z["data"]
        dlen, scalars = z["doc_len"], z["scalars"]
        idx.doc_len = {d: float(dlen[i]) for i, d in enumerate(docs)}
        idx.alive = set(docs)
        idx._next, idx._total_len, idx._n_removed = int(scalars[0]), float(scalars[1]), 0
        tp, postings = {}, defaultdict(dict)
        df, doc_words = defaultdict(int), defaultdict(set)
        for k in range(len(primes)):
            p = int(primes[k]); view = chr(int(views[k]))
            tp[(view, toks[k])] = p
            a, e = int(indptr[k]), int(indptr[k + 1])
            seg = np.cumsum(indices[a:e].astype(np.int64))          # undo delta
            seg_docs = [docs[i] for i in seg.tolist()]
            postings[p] = dict(zip(seg_docs, data[a:e].astype(np.float64).tolist()))
            df[p] = e - a
            if view == "w":
                for d in seg_docs:
                    doc_words[d].add(p)
        idx.token_prime, idx.postings, idx.df = tp, postings, df
        idx.doc_words = dict(doc_words)
        if finalize:
            idx.finalize()
        return idx

    def stats(self):
        return {
            "live_docs": len(self.alive),
            "vocab": len(self.token_prime),
            "postings": sum(len(p) for p in self.postings.values()),
        }
