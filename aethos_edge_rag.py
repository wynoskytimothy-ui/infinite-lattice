#!/usr/bin/env python3
"""aethos_edge_rag — the coherent edge engine: fast numba ingest -> mmap CSR serve -> counting-bridges.

One class wires the three proven pieces:
  INGEST  : radix/hash numba build (nogil-threaded) -> serve-ready CSR (24.5M tok/s, _o1_ingest_radix).
  SERVE   : mmap-able CSR (seg_doc/seg_tf/indptr), scatter-add BM25 over the query's term segments;
            query tokens -> FNV hash -> term_id (the same hashing ingest used). RAM = working set.
  ACCURACY: counting-bridges (learned from train qrels) rerank the top-100 -- neural-free.

word-only = the edge champion (198 B/doc). Verified to reproduce the edge champion nDCG exactly:
scifact lexical 0.6712 -> +bridges 0.7112 (same numbers the slow pipeline produced -- accuracy untouched).
"""
import os, sys, json, math, time
os.environ.setdefault("PYTHONUTF8", "1")
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _fast_tok import words
from _o1_ingest_radix import radix_build_mt, fnv
from aethos_posting_codec import encode_docids_best, decode_doc_gaps, fixed_gap_decode, bic_unpack

K1, B = 1.2, 0.75


def _trng_root():
    env = os.environ.get("TRNG_ROOT", "").strip()
    if env and Path(env).is_dir():
        return Path(env)
    cand = Path(r"c:\Users\wynos\trng")
    if cand.is_dir():
        return cand
    here = Path(__file__).resolve().parent
    for p in (here.parent / "trng", here / "trng"):
        if p.is_dir():
            return p
    return cand


def _ensure_trng_path():
    root = _trng_root()
    s = str(root)
    if root.is_dir() and s not in sys.path:
        sys.path.insert(0, s)
    return root


class EdgeRAG:
    # Below this corpus size the dense scan is already sub-0.1ms and beats BMW's block/heap overhead
    # (measured: scifact 5.2K -> BMW only ~1.3x median, p90 wash). BMW's win grows with N (38x @ MARCO 8.8M),
    # so it auto-activates only at scale. Both paths are exact; this only picks the faster one.
    BMW_MIN_DOCS = 50_000

    def __init__(self, nthreads=None):
        self.nthreads = nthreads or min(8, (os.cpu_count() or 4))
        self._stem = False

    # ---------------- stem (the one UNIVERSAL accuracy lever: idempotent prefix stem) ----------------
    @staticmethod
    def _stem_word(w, k=6):
        return w[:k] if len(w) > k else w

    def _qwords(self, text):
        """tokenize (and stem if enabled). Idempotent: stem(stem(w))==stem(w), so safe to call at every leaf."""
        ws = words(text)
        return [self._stem_word(w) for w in ws] if self._stem else list(ws)

    def _sq(self, query):
        return " ".join(self._qwords(query)) if self._stem else query

    def _stem_corpus(self, corpus):
        return {d: self._sq(t) for d, t in corpus.items()} if self._stem else corpus

    # ---------------- ingest ----------------
    def build(self, corpus, stem=False):
        self._stem = stem
        self.doc_ids = list(corpus.keys())
        self._d2i = {d: i for i, d in enumerate(self.doc_ids)}
        texts = [self._sq(t) for t in corpus.values()] if stem else list(corpus.values())
        t0 = time.perf_counter()
        Bd, _ = radix_build_mt(texts, self.nthreads)
        self.ingest_s = time.perf_counter() - t0
        self.uniq_hash = Bd["uniq_hash"]; self.indptr = Bd["indptr"]
        self.seg_doc = Bd["seg_doc"]; self.seg_tf = Bd["seg_tf"]
        self.V = Bd["V"]; self.N = len(self.doc_ids)
        self.hash2term = {int(h): i for i, h in enumerate(self.uniq_hash)}
        self.df = np.diff(self.indptr).astype(np.int64)
        self.doc_len = np.bincount(self.seg_doc, weights=self.seg_tf.astype(np.float64),
                                   minlength=self.N)
        self._finish()
        return self

    def build_serving(self, corpus, stem=True, doc2query=None):
        """PRODUCTION build: stems the index by DEFAULT (priority #3, the one UNIVERSAL lever -- scifact +0.030 R@100
        /+0.025 nDCG, nfcorpus +0.017; stem the index DIRECTLY, RRF-fusing dilutes). Optionally bakes LABEL-FREE
        DOC2QUERY question words into each doc at ingest (Finding 11d: +0.0188 nDCG, zero qrels). Keeps a corpus ref
        for the 3-way meet expansion tier. The bare build(stem=False) stays the experiment baseline (130 call sites)."""
        self._corpus = corpus
        src = {d: (t + " " + " ".join(doc2query.get(d, []))) for d, t in corpus.items()} if doc2query else corpus
        return self.build(src, stem=stem)

    # ---------------- label-free expansion tiers (3-way meet + doc2query), governor-gated ----------------
    def _expand_3way(self, query, cap=30, topT=8, gate=2.5):
        """LABEL-FREE 3-way meet expansion (Finding 11c: +0.0263 nDCG on mismatch corpora). Rarest 2/3 query terms
        whose doc-MEET is a tiny pool -> expand with that pool's shared high-idf words. Precise because the 3-way
        pool is small + relevant (gold in it ~70%). Returns an expansion string ("" if no real small meet exists)."""
        import itertools
        def _idf(w):
            t = self._term_id(self._stem_word(w)); return self._idf(int(self.df[t])) if t is not None else 0.0
        def _post(w):
            t = self._term_id(self._stem_word(w))
            if t is None: return None
            a, e = int(self.indptr[t]), int(self.indptr[t + 1]); return set(self.seg_doc[a:e].tolist())
        ws = [w for w, s in sorted(((w, _idf(w)) for w in dict.fromkeys(self._stem_word(x) for x in words(query))),
                                   key=lambda x: -x[1]) if _idf(w) >= gate][:6]
        best, bspec = None, -1.0
        for k in (3, 2):
            for c in itertools.combinations(ws, k):
                ps = [_post(w) for w in c]
                if any(p is None for p in ps): continue
                pool = set.intersection(*ps)
                if 1 <= len(pool) <= cap:
                    spec = sum(_idf(w) for w in c) * k
                    if spec > bspec: bspec, best = spec, pool
        if not best or not getattr(self, "_corpus", None): return ""
        df = Counter()
        for di in best:
            for w in set(self._stem_word(x) for x in words(self._corpus[self.doc_ids[di]])):
                if _idf(w) >= 1.5: df[w] += 1
        exp = [w for w, c in sorted(df.items(), key=lambda x: -(x[1] / len(best)) * _idf(x[0]))
               if c >= max(2, len(best) // 2)][:topT]
        return " ".join(exp)

    def search_expanded(self, query, k=10, w_exp=0.4):
        """Production serve: BM25 fused with the 3-way meet expansion (label-free). Gated by _use_meet3 (default on;
        set by fit_tier_governor so saturated corpora hold it). doc2query, if used, is already baked into the index."""
        base = self.score(query); base = base / (base.max() or 1.0)
        if getattr(self, "_use_meet3", True):
            exp = self._expand_3way(query)
            if exp:
                e = self.score(exp); e = e / (e.max() or 1.0); base = base + w_exp * e
        return self._top(base, k)

    def fit_tier_governor(self, train_queries, train_qrels):
        """NEVER-REGRESS gate for the 3-way expansion tier: enable it only if it does NOT drop nDCG@10 on the train
        qrels -> mismatch corpora (nfcorpus) fire it, saturated corpora (scifact) hold it. Label-free at serve."""
        import math
        def nd(ranked, rels):
            dcg = sum(rels.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked[:10]))
            ideal = sorted(rels.values(), reverse=True)[:10]; idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
            return dcg / idcg if idcg else 0.0
        ids = [q for q in train_qrels if q in train_queries][:400]
        def mean_nd(use):
            self._use_meet3 = use
            return sum(nd(self.search_expanded(train_queries[q], 10), train_qrels[q]) for q in ids) / max(1, len(ids))
        base, exp = mean_nd(False), mean_nd(True)
        self._use_meet3 = exp >= base - 1e-6
        return {"meet3_enabled": self._use_meet3, "train_nDCG_base": round(base, 4), "train_nDCG_meet3": round(exp, 4)}

    def _finish(self):
        avgdl = self.doc_len.sum() / max(1, self.N)
        self._A = K1 * (1 - B); self._Bc = K1 * B / avgdl; self._k1p1 = K1 + 1
        self._denom = self._A + self._Bc * self.doc_len

    def _term_id(self, token):
        return self.hash2term.get(fnv(token.encode()))

    def _idf(self, dfp):
        return math.log(1 + (self.N - dfp + 0.5) / (dfp + 0.5))

    # ---------------- serve ----------------
    def score(self, query):
        scores = np.zeros(self.N)
        ip, sd, st, den = self.indptr, self.seg_doc, self.seg_tf, self._denom
        for w, qwt in Counter(self._qwords(query)).items():
            tid = self._term_id(w)
            if tid is None: continue
            a, e = int(ip[tid]), int(ip[tid + 1])
            if e == a: continue
            di = sd[a:e]; tf = st[a:e].astype(np.float64)
            idf = self._idf(e - a)
            scores[di] += (qwt * idf * self._k1p1) * tf / (tf + den[di])
        return scores

    def _top(self, scores, k):
        kk = min(k, self.N)
        part = np.argpartition(scores, -kk)[-kk:]
        part = part[np.argsort(scores[part])[::-1]]
        return [self.doc_ids[i] for i in part if scores[i] > 0.0]

    # ---- sublinear fusion primitives: kill the O(N) score()+argsort drag in the fusion tiers ----
    def _lex_topk_idx(self, query, depth):
        """Top-`depth` doc INDICES by exact lexical BM25. Sublinear via BMW at scale (BMW top-k is bit-identical to
        score(), verify_bmw_lossless); dense-scan fallback for small corpora / no-BMW. Replaces the
        `np.argsort(self.score(query))[::-1][:depth]` pattern -- same result, no O(N) alloc/sort at scale."""
        if getattr(self, "_use_meet3", False):
            return self._lex_topk_idx_expanded(query, depth)
        if self.N >= self.BMW_MIN_DOCS:
            try:
                return [self._d2i[d] for d in self.search_bmw(query, depth)]
            except Exception:
                pass
        sc = self.score(query); m = min(depth, self.N)
        idx = np.argpartition(sc, -m)[-m:]; idx = idx[np.argsort(sc[idx])[::-1]]
        return [int(i) for i in idx if sc[i] > 0.0]

    def _lex_topk_idx_expanded(self, query, depth, w_exp=0.4):
        """Top-k by BM25 fused with label-free 3-way meet expansion (governor-gated via _use_meet3)."""
        base = self.score(query)
        base = base / (base.max() or 1.0)
        exp = self._expand_3way(query)
        if exp:
            e = self.score(exp); e = e / (e.max() or 1.0)
            base = base + w_exp * e
        m = min(depth, self.N)
        if m <= 0:
            return []
        idx = np.argpartition(base, -m)[-m:]
        idx = idx[np.argsort(base[idx])[::-1]]
        return [int(i) for i in idx if base[i] > 0.0]

    def _lex_scores(self, query, cand_idx):
        """Exact BM25 for a BOUNDED candidate set -- no O(N) allocation. Per query term, searchsorted the sorted
        posting segment against the (sorted) candidates: O(sum_q |cand| log df). Equals score()[cand_idx] exactly."""
        cand = np.asarray(cand_idx, np.int64)
        if cand.size == 0: return np.zeros(0)
        if not getattr(self, "_sorted", False): self.sort_segments()
        order = np.argsort(cand); cs = cand[order]
        den = self._denom[cs]; out = np.zeros(cand.size)
        ip, sd, st = self.indptr, self.seg_doc, self.seg_tf
        for w, qwt in Counter(self._qwords(query)).items():
            tid = self._term_id(w)
            if tid is None: continue
            a, e = int(ip[tid]), int(ip[tid + 1])
            if e == a: continue
            seg = sd[a:e]; j = np.searchsorted(seg, cs); j = np.minimum(j, seg.size - 1)
            hit = seg[j] == cs
            tf = np.where(hit, st[a:e][j].astype(np.float64), 0.0)
            out[order] += np.where(hit, (qwt * self._idf(e - a) * self._k1p1) * tf / (tf + den), 0.0)
        return out

    def search(self, query, k=10, exact_scan=False):
        """Default lexical top-k serve == exact Block-Max WAND (priority #1). Corpus-size-independent posting
        pruning; top-k is BIT-IDENTICAL to the dense scan (per-posting impact folds idf in). BMW is the ONE
        production region routing -- k-means / chamber IVF cluster-routing is RETIRED (measured 6/6 negative,
        7-28pp recall loss; BMW's working-set touch is recall-lossless by construction). Falls back to the exact
        dense scan for tiny corpora or if the numba BMW kernel is unavailable -- the result is identical either
        way, since the scan is the oracle BMW is verified against (see verify_bmw_lossless)."""
        if exact_scan or self.N < self.BMW_MIN_DOCS:
            self._bmw_active = False
            return self._search_scan(query, k)
        try:
            out = self.search_bmw(query, k); self._bmw_active = True; return out
        except Exception:
            self._bmw_active = False
            return self._search_scan(query, k)

    def _search_scan(self, query, k=10):
        """Dense-accumulator top-k oracle (scores every doc). BMW is verified bit-identical against this; also the
        fallback for tiny corpora / no-numba environments. Fusion tiers (bridged/pooled/stack) still call score()
        directly -- they need the full score vector, which BMW's top-k pruning intentionally does not produce."""
        return self._top(self.score(query), k)

    def verify_bmw_lossless(self, queries, k=10, n=200):
        """Prove the production BMW serve == the dense oracle on real queries. Returns (match_rate, n_checked,
        sample_mismatches). Priority #1 ships only when match_rate == 1.0 (recall-lossless by construction)."""
        qs = list(queries.values()) if isinstance(queries, dict) else list(queries)
        if n: qs = qs[:n]
        ok, mism = 0, []
        for q in qs:
            a = self.search_bmw(q, k); b = self._search_scan(q, k)
            if a == b: ok += 1
            elif len(mism) < 5: mism.append((str(q)[:50], a[:4], b[:4]))
        return ok / max(len(qs), 1), len(qs), mism

    # ---------------- exact block-max WAND serve (fast EXACT top-k, posting-pruned, corpus-size-independent) ----------
    def build_blockmax(self, B=128):
        """Build the block-max WAND sidecar so few-term queries get EXACT top-k with heavy posting pruning -- the fast
        lexical serve at MARCO scale. Per-posting BM25 impact folds idf in (query weight = qwt), so BMW top-k is
        bit-identical to score()/search(). Chunked impact build bounds peak memory at 8.8M scale."""
        import aethos_bmw as _bmw
        if not getattr(self, "_sorted", False): self.sort_segments()
        df = np.diff(self.indptr).astype(np.int64)
        idf_terms = np.where(df > 0, np.log1p((self.N - df + 0.5) / (df + 0.5)), 0.0).astype(np.float32)
        idf_pp = np.repeat(idf_terms, df)
        sd = np.asarray(self.seg_doc, np.int64); impact = np.empty(sd.size, np.float32); CH = 20_000_000
        for s in range(0, sd.size, CH):
            e = min(s + CH, sd.size); tf = np.asarray(self.seg_tf[s:e], np.float64); den = self._denom[sd[s:e]]
            impact[s:e] = (idf_pp[s:e] * self._k1p1 * tf / (tf + den)).astype(np.float32)
        self._blk = _bmw.build_blockmax(self.indptr, self.seg_doc, impact, B=B)
        return self

    def search_bmw(self, query, k=10):
        """Exact top-k via block-max WAND (same result as search(), faster by pruning). Auto-builds the sidecar once."""
        import aethos_bmw as _bmw
        if getattr(self, "_blk", None) is None: self.build_blockmax()
        qt = []; qw = []
        for w, c in Counter(self._qwords(query)).items():
            tid = self._term_id(w)
            if tid is not None and self.indptr[tid + 1] > self.indptr[tid]:
                qt.append(int(tid)); qw.append(float(c))
        di, _ = _bmw.bmw_search(np.array(qt, np.int64), np.array(qw, np.float64), self._blk, k)
        return [self.doc_ids[int(i)] for i in di]

    def exact_keys(self, query):
        """The df==1 fast-path: any query term unique to ONE doc is a 100%-faithful O(1) content-address (its single
        posting). Returns the exact-key doc-ids in query order (the certain hits), no scoring. This is the exact-key
        regime of the unbounded address space -- pays off on identifier-heavy data (codes/IDs/names)."""
        out = []
        for w in dict.fromkeys(self._qwords(query)):
            tid = self._term_id(w)
            if tid is not None and int(self.df[tid]) == 1:
                out.append(self.doc_ids[int(self.seg_doc[int(self.indptr[tid])])])
        return list(dict.fromkeys(out))

    def search_exactkey(self, query, k=10):
        """Exact-key hits pinned on top (certain), then normal scoring fills the rest. Returns (results, n_exact)."""
        ex = self.exact_keys(query)
        if len(ex) >= k:
            return ex[:k], len(ex)
        seen = set(ex)
        rest = [d for d in self.search(query, k + len(ex)) if d not in seen]
        return (ex + rest)[:k], len(ex)

    # ---------------- bridges (counting, neural-free) ----------------
    def learn_bridges(self, queries, train_qrels, corpus, min_pairs=2, top_per=12, idf_gate=1.5):
        def idf_tok(w):
            tid = self._term_id(w)
            return self._idf(int(self.df[tid])) if tid is not None else 0.0
        cooc = defaultdict(Counter); qtp = Counter(); qc, dc = {}, {}
        for qid, rels in train_qrels.items():
            if qid not in queries: continue
            qts = qc.get(qid)
            if qts is None:
                qts = tuple(w for w in set(self._qwords(queries[qid])) if idf_tok(w) >= idf_gate); qc[qid] = qts
            for cid, sc in rels.items():
                if sc <= 0 or cid not in corpus: continue
                dts = dc.get(cid)
                if dts is None:
                    dts = frozenset(w for w in set(self._qwords(corpus[cid])) if idf_tok(w) >= idf_gate); dc[cid] = dts
                for qt in qts:
                    qtp[qt] += 1
                    if dts: cooc[qt].update(dts)
        self.bridge = {}
        for qt, partners in cooc.items():
            np_ = qtp[qt]
            scored = [(dt, (c / np_) * idf_tok(dt)) for dt, c in partners.items() if dt != qt and c >= min_pairs]
            scored.sort(key=lambda x: (-x[1], x[0]))
            if scored: self.bridge[qt] = scored[:top_per]
        return self

    def _bridge_scores_vec(self, query, cand_idx):
        """Bridge score over a BOUNDED candidate set -- no O(N) allocation. Per bridge-target term, searchsorted the
        sorted segment against the (sorted) candidates. Equals the old acc[cand_idx] exactly (each candidate that
        contains a target term accrues its summed bridge weight)."""
        cand = np.asarray(cand_idx, np.int64); out = np.zeros(cand.size)
        targets = {}                                       # term_id -> summed bridge weight
        for qt in set(self._qwords(query)):
            for dt, w in self.bridge.get(qt, ()):
                tid = self._term_id(dt)
                if tid is None: continue
                targets[tid] = targets.get(tid, 0.0) + w
        if not targets or cand.size == 0: return out
        if not getattr(self, "_sorted", False): self.sort_segments()
        order = np.argsort(cand); cs = cand[order]
        ip, sd = self.indptr, self.seg_doc
        for tid, w in targets.items():
            a, e = int(ip[tid]), int(ip[tid + 1])
            if e == a: continue
            seg = sd[a:e]; j = np.searchsorted(seg, cs); j = np.minimum(j, seg.size - 1)
            out[order] += np.where(seg[j] == cs, w, 0.0)
        return out

    # ---------------- trng memory-plane pool enrich (opt-in sidecar; postings untouched) ----------------
    def attach_lattice_chambers(self, corpus, *, use_lattice_enrich=False, max_pool=200, sidecar_path=None):
        """Build token→prime (AethosPureIndexer) + LazyChamberIndex sidecar for pool enrichment at serve.
        Set use_lattice_enrich=True to union lattice recall into the top-100 BM25 pool before bridge rerank."""
        _ensure_trng_path()
        from aethos_api import AethosPureIndexer, tokenize
        from memory_plane.lazy_chambers import LazyChamberIndex, build_chamber_index_from_corpus

        idx = AethosPureIndexer()
        for doc_id, text in corpus.items():
            if tokenize(text):
                try:
                    idx.index_document(text, doc_id=doc_id)
                except Exception:
                    pass
        idx.lattice._recompute_rarity()
        idx.finalize_lattice(precompute_u16=False)
        self._word_prime = dict(idx.lattice.word_prime)
        if sidecar_path and Path(sidecar_path).exists():
            pf = set(self._word_prime.values())
            self._lattice_chamber = LazyChamberIndex.load_from(sidecar_path, prime_filter=pf)
        else:
            self._lattice_chamber = build_chamber_index_from_corpus(corpus, self._word_prime)
            if sidecar_path:
                self._lattice_chamber.persist_to(sidecar_path)
        self._use_lattice_enrich = bool(use_lattice_enrich)
        self._lattice_max_pool = int(max_pool)
        return self

    def attach_plane_walk(self, *, use_plane_mask=False, k=12):
        """Optional unified_complex_plane walk to mask lazy-chamber pair lookup by active 32-chamber coords."""
        _ensure_trng_path()
        from unified_complex_plane import UnifiedComplexPlanePipeline
        self._plane_pipe = UnifiedComplexPlanePipeline()
        self._plane_k = int(k)
        self._use_plane_mask = bool(use_plane_mask)
        return self

    def _lattice_query_primes(self, query):
        from aethos_api import tokenize
        wp = getattr(self, "_word_prime", None)
        if not wp:
            return set()
        qs = set(self._qwords(query)) if getattr(self, "_stem", False) else set(tokenize(query))
        return {wp[w] for w in qs if wp.get(w, 0) > 2}

    def _plane_pair_bucket_filter(self, query):
        if not getattr(self, "_use_plane_mask", False):
            return None
        chamber = getattr(self, "_lattice_chamber", None)
        if chamber is None:
            return None
        qps = sorted(self._lattice_query_primes(query))
        if len(qps) < 2:
            return None
        from unified_complex_plane import walk
        from prime_hotel.premonition import prime_octant
        n = sum(qps) % (1 << 20)
        st = walk(n, k=getattr(self, "_plane_k", 12), primes=tuple(qps[:8]))
        active = {f.index % 32 for f in st.lattice_32.frames}

        def _ch(p):
            return (prime_octant(p) * 4 + (p % 4)) % 32

        bf = {key for key in chamber.lazy_correlations if _ch(key[0]) in active or _ch(key[1]) in active}
        return bf or None

    def _lattice_enrich_pool(self, query, cand_idx):
        if not getattr(self, "_use_lattice_enrich", False):
            return cand_idx
        chamber = getattr(self, "_lattice_chamber", None)
        if chamber is None:
            return cand_idx
        from memory_plane.pool_enrich import enrich_candidate_pool
        qps = self._lattice_query_primes(query)
        if not qps:
            return cand_idx
        extra = chamber.candidates_for_query_primes(qps, top_pairs=20, bucket_filter=self._plane_pair_bucket_filter(query))
        cand = np.asarray(cand_idx, np.int64)
        if cand.size == 0:
            return cand
        lex = self._lex_scores(query, cand)
        ranked = sorted(((self.doc_ids[int(i)], float(lex[j])) for j, i in enumerate(cand)), key=lambda x: -x[1])
        enriched = enrich_candidate_pool(ranked, extra, max_pool=getattr(self, "_lattice_max_pool", 200))
        out = [self._d2i[d] for d, _ in enriched if d in self._d2i]
        return np.asarray(out, np.int64) if out else cand

    def search_bridged(self, query, k=10, lam=0.15, append_pool_union=None, append_pool_k=None):
        """Bridged rerank: BM25 top-100 (+ optional append-pool union widen) + lattice enrich + bridge fusion."""
        base_depth = min(100, self.N)
        cand_list = self._lex_topk_idx(query, base_depth)
        use_union = (
            append_pool_union
            if append_pool_union is not None
            else getattr(self, "_append_pool_union", False)
        )
        pool_k = (
            append_pool_k
            if append_pool_k is not None
            else getattr(self, "_append_pool_k", 200)
        )
        if use_union and pool_k > base_depth:
            union = self._lex_topk_idx(query, min(pool_k, self.N))
            cand_list = sorted(set(cand_list) | set(union))
        cand = np.asarray(cand_list, np.int64)
        cand = self._lattice_enrich_pool(query, cand)
        if cand.size == 0: return []
        lex = self._lex_scores(query, cand); lmax = lex.max() or 1.0               # exact BM25 on the candidates
        bs = self._bridge_scores_vec(query, cand)
        bmax = bs.max() if bs.size and bs.max() > 0 else 1.0
        final = lex / lmax + lam * bs / bmax
        fin = cand[np.argsort(final)[::-1]]
        return [self.doc_ids[int(i)] for i in fin[:k]]

    def configure_append_pool_union(self, enabled: bool = True, k: int = 200) -> "EdgeRAG":
        """Union BM25 top-k into bridged candidate pool (hybrid append_pool_union lever for EdgeRAG)."""
        self._append_pool_union = enabled
        self._append_pool_k = k
        return self

    # ---------------- trigger pools (champion lists): precise, pre-sorted, corpus-size-independent --------
    def build_pools(self, M=256):
        """Per term, the pool of docs it TRIGGERS most strongly: top-M by BM25 impact, PRE-SORTED by impact.
        Query work becomes O(#query_terms * M) -- independent of corpus size -- and candidates arrive already
        ranked, so pulling is precise and trivial to sort. (The lattice knows each term's triggered docs +
        their impact; this just materializes + caps + orders them.)"""
        self.M = M
        ip, sd, st, den, k1p1 = self.indptr, self.seg_doc, self.seg_tf, self._denom, self._k1p1
        pd, pi, ptr = [], [], [0]
        for t in range(self.V):
            a, e = int(ip[t]), int(ip[t + 1])
            di = sd[a:e]; tf = st[a:e].astype(np.float64)
            imp = (self._idf(e - a) * k1p1) * tf / (tf + den[di])      # BM25 contribution of this term per doc
            if di.size > M:
                keep = np.argpartition(imp, -M)[-M:]; di = di[keep]; imp = imp[keep]
            order = np.argsort(imp)[::-1]                               # PRE-SORT by impact (desc)
            pd.append(di[order]); pi.append(imp[order]); ptr.append(ptr[-1] + order.size)
        self.pool_doc = np.concatenate(pd).astype(np.uint32)
        self.pool_imp = np.concatenate(pi).astype(np.float64)
        self.pool_ptr = np.asarray(ptr, np.int64)
        return self

    def sort_segments(self):
        """Sort each term's CSR segment by doc-id (enables binary-search lookup for rarest-anchor scoring).
        Vectorized: one argsort by (term, doc). Postings reorder within term; indptr unchanged; serve unaffected."""
        tpp = np.repeat(np.arange(self.V), self.df)                    # term-id per posting
        key = tpp.astype(np.int64) * (self.N + 1) + self.seg_doc.astype(np.int64)
        o = np.argsort(key, kind="stable")
        self.seg_doc = np.ascontiguousarray(self.seg_doc[o]); self.seg_tf = np.ascontiguousarray(self.seg_tf[o])
        self._sorted = True
        return self

    def search_anchored(self, query, k=10):
        """Rarest triggered word = smallest pool; score those candidates EXACTLY over all query terms (binary
        search in sorted segments). Bounded by the rare term's df (corpus-size-independent); near-lossless."""
        if not getattr(self, "_sorted", False): self.sort_segments()
        ip, sd, st, den, k1p1 = self.indptr, self.seg_doc, self.seg_tf, self._denom, self._k1p1
        terms = []
        for w, qwt in Counter(self._qwords(query)).items():
            tid = self._term_id(w)
            if tid is not None and ip[tid + 1] > ip[tid]: terms.append((int(self.df[tid]), tid, qwt))
        if not terms: return []
        terms.sort()                                                   # by df asc -> rarest first = the anchor
        atid = terms[0][1]; a, e = int(ip[atid]), int(ip[atid + 1])
        cands = sd[a:e].astype(np.int64)                               # candidate doc-indices (sorted)
        scores = np.zeros(cands.size); dc = den[cands]
        for dfp, tid, qwt in terms:
            sa, se = int(ip[tid]), int(ip[tid + 1])
            seg = sd[sa:se]; idx = np.searchsorted(seg, cands); idx = np.minimum(idx, seg.size - 1)
            hit = seg[idx] == cands
            tf = np.where(hit, st[sa:se][idx].astype(np.float64), 0.0)
            scores += np.where(hit, (qwt * self._idf(dfp) * k1p1) * tf / (tf + dc), 0.0)
        top = np.argpartition(scores, -min(k, scores.size))[-min(k, scores.size):]
        top = top[np.argsort(scores[top])[::-1]]
        return [self.doc_ids[int(cands[i])] for i in top if scores[i] > 0]

    def search_pooled(self, query, k=10):
        """Pull from the triggered pools only (pre-sorted by impact), accumulate, top-k. Bounded by query*M."""
        acc = {}
        for w, qwt in Counter(self._qwords(query)).items():
            tid = self._term_id(w)
            if tid is None: continue
            a, e = int(self.pool_ptr[tid]), int(self.pool_ptr[tid + 1])
            docs = self.pool_doc[a:e]; imp = self.pool_imp[a:e]
            for i in range(docs.size):
                d = int(docs[i]); acc[d] = acc.get(d, 0.0) + qwt * imp[i]
        return [self.doc_ids[d] for d in sorted(acc, key=acc.get, reverse=True)[:k]]

    # ---------------- the dial: one API across all tiers + weighted RRF fusion ----------------
    def attach_splade(self, splade_index):
        """Attach a DistilledSpladeIndex (encoder-free SPLADE tier) so retrieve() can dial to it."""
        self._splade = splade_index; return self

    # ---- routed complement (#7): dense RANKER + BM25/SPLADE fired only on dense's LOW-CONFIDENCE queries ----
    def attach_dense(self, doc_emb, doc_ids=None):
        """Attach dense doc embeddings (N x D) for the routed-complement tier. Dense is the RANKER; the structural
        signals fire only where dense is unsure (verified +0.0144 nDCG scifact, +0.0034/+0.0110 nf). Needs a query
        embedding at serve -- the one model touch, breaking encoder-free purity (the cost of the .74 dense tier)."""
        E = np.asarray(doc_emb, np.float32); E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
        if doc_ids is not None:
            pos = {str(d): i for i, d in enumerate(doc_ids)}
            E = E[[pos[str(d)] for d in self.doc_ids]]        # reorder to engine doc order
        self._dense_E = np.ascontiguousarray(E, np.float32)
        return self

    def _dense_scores(self, qvec):
        v = np.asarray(qvec, np.float32); v = v / (np.linalg.norm(v) + 1e-9)
        return self._dense_E @ v

    @staticmethod
    def _dconf(de):
        top = np.sort(de)[::-1][:10]
        return float(top[0] - top[1:].mean()) if top.size > 1 else 0.0

    def _routed_signals(self, query, qvec, pool_depth=300):
        """Bounded routed complement signals: dense(ranker) + BM25(+SPLADE) on a dense u BMW (u SPLADE) pool."""
        de = self._dense_scores(qvec)
        cand = set(np.argsort(de)[::-1][:pool_depth].tolist()) | set(self._lex_topk_idx(query, pool_depth))
        sp_full = None
        if getattr(self, "_splade", None) is not None:
            sp_full = self._splade.score_vec(query, tau=getattr(self._splade, "tau", None))
            cand |= set(np.argsort(sp_full)[::-1][:pool_depth].tolist())
        cand = np.array(sorted(cand), np.int64)
        def zn(a): m = a.max() if a.size else 0.0; return a / m if m > 0 else a
        return dict(de=de, conf=self._dconf(de), cand=cand, zde=zn(de[cand]),
                    zbm=zn(self._lex_scores(query, cand)), zsp=(zn(sp_full[cand]) if sp_full is not None else None))

    def _routed_rank(self, sig, k, thr, w_bm, w_sp):
        if sig["conf"] >= thr:                                # confident -> dense alone, no complement
            top = np.argsort(sig["de"])[::-1][:k]
            return [self.doc_ids[int(i)] for i in top]
        sc = sig["zde"] + w_bm * sig["zbm"] + (w_sp * sig["zsp"] if sig["zsp"] is not None else 0.0)
        cand = sig["cand"]; order = np.argsort(sc)[::-1][:k]
        return [self.doc_ids[int(cand[j])] for j in order]

    def fit_complement_governor(self, train_q, train_qrels, train_qv, k=10):
        """Fit the gate (dense-confidence threshold + BM25/SPLADE weights) on TRAIN qrels. LEAKAGE-CLEAN: the gate
        never sees eval qrels (fixes the audit's test-CV flag). Config is chosen by INTERNAL validation (a disjoint
        slice of train) and DEFAULTS to dense-alone -- so a complement that doesn't generalize is never shipped
        (measured: scifact keeps its lift, nfcorpus correctly falls back to dense-alone)."""
        ids = [q for q in train_qrels if q in train_q and q in train_qv
               and any(train_qrels[q].get(d, 0) > 0 and d in self._d2i for d in train_qrels[q])]
        def nd(ranked, rels):
            g = {d for d in rels if rels.get(d, 0) > 0}
            dcg = sum(1.0 / np.log2(i + 2) for i, d in enumerate(ranked[:k]) if d in g)
            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(g))))
            return dcg / idcg if idcg else 0.0
        cache = {q: self._routed_signals(train_q[q], train_qv[q]) for q in ids}
        has_sp = any(cache[q]["zsp"] is not None for q in ids)
        confs = np.array([cache[q]["conf"] for q in ids])
        grid = [(float(np.percentile(confs, p)), wb, ws) for p in (30, 40, 50, 60, 70)
                for wb in (0.2, 0.3, 0.4) for ws in ((0.0, 0.2, 0.4) if has_sp else (0.0,))]
        def cfg_nd(q, cfg): return nd(self._routed_rank(cache[q], k, *cfg), train_qrels[q])
        def dalone_nd(q): return nd([self.doc_ids[int(i)] for i in np.argsort(cache[q]["de"])[::-1][:k]], train_qrels[q])
        def best_on(subset):
            bc, bm = None, -1.0
            for cfg in grid:
                m = float(np.mean([cfg_nd(q, cfg) for q in subset]))
                if m > bm: bm, bc = m, cfg
            return bc
        # nested k-fold CV: select config on fold-train, score on fold-test -> honest GO/NO-GO for the complement
        folds = min(5, len(ids)); rng = np.random.default_rng(0); o = list(ids); rng.shuffle(o)
        fid = {q: i % folds for i, q in enumerate(o)}
        cv_r, cv_d = [], []
        for f in range(folds):
            tr = [q for q in ids if fid[q] != f]; te = [q for q in ids if fid[q] == f]
            if not te or not tr: continue
            cfg = best_on(tr)
            cv_r += [cfg_nd(q, cfg) for q in te]; cv_d += [dalone_nd(q) for q in te]
        lift = float(np.mean(cv_r) - np.mean(cv_d)) if cv_r else 0.0
        self._cg_cv_lift = lift
        if lift > 0.0:                                        # complement generalizes -> ship best config (refit on all)
            bc = best_on(ids); self._cg = {"thr": bc[0], "w_bm": bc[1], "w_sp": bc[2]}
        else:                                                 # does NOT generalize -> dense alone (no harm)
            self._cg = {"thr": float(confs.min() - 1.0), "w_bm": 0.0, "w_sp": 0.0}
        return self

    def retrieve_dense_routed(self, query, qvec, k=10):
        """Serve the routed complement: dense ranker + governed low-confidence BM25/SPLADE complement."""
        cg = getattr(self, "_cg", None)
        sig = self._routed_signals(query, qvec)
        if cg is None: return [self.doc_ids[int(i)] for i in np.argsort(sig["de"])[::-1][:k]]
        return self._routed_rank(sig, k, cg["thr"], cg["w_bm"], cg["w_sp"])

    def _tier(self, query, tier, n):
        if tier == "lexical": return self.search(query, n)
        if tier == "bridged": return self.search_bridged(query, n)
        if tier == "wand":
            if not getattr(self, "_sorted", False): self.sort_segments()
            return self.search_anchored(query, n)
        if tier == "distilled":
            if not getattr(self, "_splade", None): raise ValueError("no SPLADE index attached (attach_splade)")
            return self._splade.search(query, n)
        if tier == "gbdt":
            if getattr(self, "_gbdt", None) is None: raise ValueError("no GBDT reranker attached (attach_gbdt)")
            return self._search_gbdt(query, n)
        if tier == "drift":
            if getattr(self, "_drift_tid", None) is None: raise ValueError("no index drift built (build_drift_index)")
            return self._search_drift_native(query, n)
        if tier == "convbridge":
            if getattr(self, "_cbgraph", None) is None: raise ValueError("no convbridge (attach_convbridge)")
            return [self.doc_ids[i] for i in self._convbridge_pool(query, n)]
        if tier == "doclattice":
            if getattr(self, "_docnbrs", None) is None: raise ValueError("no doclattice (attach_doclattice)")
            return [self.doc_ids[i] for i in self._doclattice_pool(query, n)]
        if tier == "recall_fuse":
            return [self.doc_ids[i] for i in self._recall_pool(query, n)]
        if tier == "auto":
            return self._tier(query, getattr(self, "_auto_tier", "bridged"), n)
        if tier == "cascade":
            return self._search_cascade(query, n)
        raise ValueError(f"unknown tier '{tier}'")

    # ---- the proven campaign stack (fuse 4 complementary sources -> deeper pool -> rerank) ----
    def attach_drift(self, corpus, seeds=None):
        """Build the zero-shot co-occurrence DRIFT graph (Timothy's higher-D idea) for the stack tier."""
        from _o1_drift import build_drift_fast
        corpus = self._stem_corpus(corpus)                 # drift graph must live in the same (stemmed) token space
        if seeds is None: seeds = self._all_seed_words(corpus)
        self._drift, _, _ = build_drift_fast(corpus, seeds)  # sparse-matmul co-occurrence (scales like ingest)
        return self

    def _all_seed_words(self, corpus):
        from _fast_tok import words as _w
        return {x for t in corpus.values() for x in _w(t)}

    def attach_reranker(self, ce, corpus):
        """Attach a cross-encoder reranker + the doc texts it needs (for the stack tier)."""
        self._ce = ce; self._doc_text = corpus; return self

    def _drift_idx(self, query, depth):
        from _o1_drift import drift_bag, score_bag
        sc = score_bag(self, drift_bag(self._sq(query), self._drift, alpha=0.4))
        return list(np.argsort(sc)[::-1][:depth])

    def _stack(self, query, k=10, depth=300, rerank=True):
        """The campaign stack: fuse lexical+bridges+drift+distilled-SPLADE top-`depth`, union, CE-rerank, top-k."""
        pools = [self._lex_topk_idx(query, depth),                                  # sublinear lexical pool (was O(N) score+sort)
                 [self._d2i[d] for d in self.search_bridged(query, depth)]]
        if getattr(self, "_drift", None) is not None: pools.append(self._drift_idx(query, depth))
        if getattr(self, "_splade", None) is not None:
            pools.append([self._d2i[d] for d in self._splade.search(query, depth)])
        if getattr(self, "_cbgraph", None) is not None: pools.append(self._convbridge_pool(query, depth))   # word door
        if getattr(self, "_docnbrs", None) is not None: pools.append(self._doclattice_pool(query, depth))   # doc door
        u = list(set().union(*[set(p) for p in pools]))
        if not (rerank and getattr(self, "_ce", None) is not None and u):
            return [self.doc_ids[i] for i in u[:k]]                    # fused pool, unranked (no reranker)
        sc = self._ce.predict([(query, self._doc_text[self.doc_ids[i]][:512]) for i in u],
                              batch_size=256, show_progress_bar=False)
        order = [u[j] for j in np.argsort(sc)[::-1]]
        return [self.doc_ids[i] for i in order[:k]]

    # ---- Timothy's two VALIDATED recall doors: word convergent bridge + doc-side lattice (complementary, fuse into stack) ----
    def attach_convbridge(self, corpus):
        """WORD CONVERGENT BRIDGE: all-vocab co-occurrence graph -> 2-hop diffusion with MULTI-SOURCE CONVERGENCE
        (targets reached by 2+ query terms reinforce) + rare-target weighting. A RECALL door: pulls TERM-bridged gold
        into the pool (scifact Q132 346->9, Q54 10->1), fused into the stack + reranked. Small/mid corpora (O(V^2) graph)."""
        from _o1_drift import build_drift_fast
        scorp = self._stem_corpus(corpus)
        self._cbgraph, _, _ = build_drift_fast(scorp, self._all_seed_words(scorp))
        return self

    def _cb_idf(self, w):
        t = self._term_id(w)
        return self._idf(int(self.df[t])) if t is not None else 1.0

    def _conv_bag(self, query, hops=2, alpha=0.3):
        from collections import defaultdict
        g = self._cbgraph
        qt = list(dict.fromkeys(words(self._sq(query)))); qset = set(qt)
        bag = {w: 1.0 for w in qt}
        reached = defaultdict(float); srcs = defaultdict(set)
        frontier = [(w, self._cb_idf(w), w) for w in qt]
        for h in range(hops):
            nxt = []
            for node, m, src in frontier:
                for nb, pmi in g.get(node, ())[:16]:
                    if nb in qset: continue
                    c = m * pmi * (0.5 ** h); reached[nb] += c; srcs[nb].add(src); nxt.append((nb, c, src))
            frontier = nxt
        if reached:
            for t in list(reached): reached[t] *= len(srcs[t]) * self._cb_idf(t)   # convergence x rare-target
            mx = max(reached.values())
            for t, m in sorted(reached.items(), key=lambda x: -x[1])[:40]:
                bag[t] = bag.get(t, 0.0) + alpha * (m / mx)
        return bag

    def _convbridge_pool(self, query, depth):
        from _o1_drift import score_bag
        return list(np.argsort(score_bag(self, self._conv_bag(query)))[::-1][:depth])

    def attach_doclattice(self, corpus, knn=20, df_hi=0.15):
        """DOC-SIDE LATTICE (Timothy's 'second lattice'): doc-doc tf-idf kNN graph -- docs INTERSECT where they share
        discriminative words. A COMPLEMENTARY recall door: pulls DOCUMENT-similar gold that shares NO query word
        (scifact Q577 >1000->53). Built once at ingest; serve diffuses BM25 seeds -> similar docs."""
        import scipy.sparse as sp
        from sklearn.preprocessing import normalize
        from collections import defaultdict
        scorp = self._stem_corpus(corpus); N = self.N; cap = df_hi * N
        vi = {}; rows = []; cols = []; vals = []; self._dl_terms = [None] * N
        for i, d in enumerate(self.doc_ids):
            tf = defaultdict(int)
            for w in words(scorp[d]): tf[w] += 1
            disc = []
            for w, c in tf.items():
                t = self._term_id(w)
                if t is None: continue
                dfw = int(self.df[t]); iw = self._idf(dfw)
                if 2 <= dfw <= cap:
                    j = vi.setdefault(w, len(vi)); rows.append(i); cols.append(j); vals.append((1.0 + np.log(c)) * iw)
                if iw >= 2.0: disc.append((w, iw))                       # discriminative terms -> the meet-order pools
            self._dl_terms[i] = sorted(disc, key=lambda x: -x[1])[:30]
        M = normalize(sp.csr_matrix((vals, (rows, cols)), shape=(N, max(1, len(vi)))))
        S = (M @ M.T).tocsr(); nbrs = [[] for _ in range(N)]
        for i in range(N):
            a, e = S.indptr[i], S.indptr[i + 1]; idx = S.indices[a:e]; dat = S.data[a:e]
            nbrs[i] = [(int(idx[k]), float(dat[k])) for k in np.argsort(dat)[::-1] if idx[k] != i][:knn]
        self._docnbrs = nbrs
        return self

    def _meet_pool(self, query, seedk=30, rar=30, kmin=2):
        """Meet-order pool (Timothy's 2/3/4-way meets): docs sharing >= kmin DISCRIMINATIVE terms with a retrieved seed.
        Reaches SCATTERED gold (shares a few rare terms but not a cosine cluster) that the flat cosine buries."""
        from collections import defaultdict
        seeds = self._lex_topk_idx(query, seedk)           # sublinear top-seedk (already score>0, desc)
        bestk = defaultdict(int); mass = defaultdict(float)
        for s in seeds:
            si = int(s)
            cnt = defaultdict(int); sco = defaultdict(float)
            for w, iw in self._dl_terms[si][:rar]:
                t = self._term_id(w)
                if t is None: continue
                for d in self.seg_doc[int(self.indptr[t]):int(self.indptr[t + 1])]:
                    di = int(d)
                    if di != si: cnt[di] += 1; sco[di] += iw
            for d, c in cnt.items():
                if c > bestk[d]: bestk[d] = c
                mass[d] += sco[d]
        cand = [d for d, c in bestk.items() if c >= kmin]
        cand.sort(key=lambda d: -mass[d])                              # rank by rare-weighted meet mass (rarest-first)
        return cand

    def _doclattice_pool(self, query, depth, beta=1.0, seedk=50):
        # sublinear: seeds + base ranking bounded to (lexical top-depth UNION seed-neighbors); prop is nonzero only
        # on seed-neighbors, so no doc outside this candidate set can enter the top-depth of (bm + beta*prop).
        seeds = self._lex_topk_idx(query, seedk); ss = self._lex_scores(query, np.asarray(seeds, np.int64))
        prop = {}
        for s, sc in zip(seeds, ss):
            if sc <= 0: continue
            for nb, w in self._docnbrs[int(s)]: prop[nb] = prop.get(nb, 0.0) + sc * w
        top = self._lex_topk_idx(query, depth)
        cand = list(dict.fromkeys(list(top) + list(prop.keys())))
        lex = self._lex_scores(query, np.asarray(cand, np.int64))
        bmmax = float(ss.max()) if ss.size else 1.0; pmax = max(prop.values()) if prop else 0.0
        scale = (bmmax / pmax) if pmax > 0 else 0.0
        sc_cand = lex + beta * np.array([prop.get(c, 0.0) * scale for c in cand])
        cos = [int(cand[j]) for j in np.argsort(sc_cand)[::-1][:depth]]  # cosine door (clustered gold)
        if getattr(self, "_dl_terms", None) is not None:                # + meet-order door (scattered gold), unioned
            seen = set(cos)
            for d in self._meet_pool(query):
                if d not in seen: cos.append(d); seen.add(d)
        return cos

    def _recall_pool(self, query, depth=300):
        """Union of lexical + both complementary recall doors (candidate generation for a reranker)."""
        pools = [self._lex_topk_idx(query, depth)]                                  # sublinear lexical pool
        if getattr(self, "_cbgraph", None) is not None: pools.append(self._convbridge_pool(query, depth))
        if getattr(self, "_docnbrs", None) is not None: pools.append(self._doclattice_pool(query, depth))
        return list(set().union(*[set(p) for p in pools]))

    def _cheap_order(self, query, pool):
        """No-GPU rarest-first ordering of a candidate pool: BM25 + doc-lattice diffusion (best cheap order, measured).
        Bounded: BM25 scored only on the pool; diffusion seeds from the sublinear lexical top-k."""
        pool = list(pool)
        cheap = dict(zip(pool, self._lex_scores(query, np.asarray(pool, np.int64))))   # exact BM25 on pool only
        if getattr(self, "_docnbrs", None) is not None:
            seeds = self._lex_topk_idx(query, 60); ss = self._lex_scores(query, np.asarray(seeds, np.int64))
            prop = {}
            for s, sc in zip(seeds, ss):
                if sc <= 0: continue
                for nb, w in self._docnbrs[int(s)]: prop[nb] = prop.get(nb, 0.0) + sc * w
            if prop:
                bmmax = float(ss.max()) if ss.size else 1.0; pmax = max(prop.values())
                scale = (bmmax / pmax) if pmax > 0 else 0.0
                for d in pool: cheap[d] = cheap.get(d, 0.0) + prop.get(d, 0.0) * scale
        return sorted(pool, key=lambda d: -cheap.get(d, 0.0))

    def _cascade_stack(self, query, k=10, depth=1000, rerank_m=200):
        """Rarest-first CASCADE: fuse the deep recall pool -> cheap rarest-first order -> CE reranks ONLY the top-M.
        Measured: reranking ~50-200 (cheap-ordered) == reranking the whole pool, at a fraction of the CE cost."""
        pool = self._recall_pool(query, depth)
        if not pool: return []
        order = self._cheap_order(query, pool); cut = order[:rerank_m]
        if getattr(self, "_ce", None) is None or not cut:
            return [self.doc_ids[i] for i in cut[:k]]                   # no reranker -> the cheap order itself
        sc = self._ce.predict([(query, self._doc_text[self.doc_ids[i]][:512]) for i in cut],
                              batch_size=256, show_progress_bar=False)
        ranked = [cut[j] for j in np.argsort(sc)[::-1]]
        return [self.doc_ids[i] for i in ranked[:k]]

    def retrieve(self, query, k=10, tier="bridged", fuse=None, weights=None, K0=60, depth=300, rerank_m=200):
        """ONE API for the whole dial. tier in {lexical, bridged, distilled, wand, stack}; or fuse=[tiers...]
        for weighted reciprocal-rank fusion. 'stack' = the proven campaign stack (fuse 4 sources -> pool depth
        -> CE rerank). Encoder-free unless tier/fuse includes a SPLADE path that needs an encoder at serve."""
        if tier == "stack" and not fuse:
            return self._stack(query, k, depth)
        if tier == "poolcascade" and not fuse:
            return self._cascade_stack(query, k, max(depth, 1000), rerank_m)   # deep pool -> cheap order -> CE top-M
        if not fuse:
            return self._tier(query, tier, k)
        w = weights or {}
        agg = {}
        for t in fuse:
            wt = w.get(t, 1.0)
            for r, d in enumerate(self._tier(query, t, 100)):
                agg[d] = agg.get(d, 0.0) + wt / (K0 + r)
        return sorted(agg, key=agg.get, reverse=True)[:k]

    # ---------------- CPU GBDT reranker (no-GPU precision tier, glass-box provenance features) ----------------
    def attach_gbdt(self, queries, train_qrels, corpus, depth=200, ranker="lambdarank"):
        """Train the no-GPU reranker: build NPMI-drift + 2nd-order expansion graphs on the (stemmed) corpus, then fit
        a learned ranker on 10 glass-box features [bm25, drift, 2nd, matched-idf, coverage, doc-len, n_sources, 3
        ranks] over the train-qrels candidate pools. Serves via the 'gbdt' tier. Leakage-guarded: graphs are
        unsupervised (corpus co-occurrence); only the labels use train qrels.
        ranker='lambdarank' (default): listwise LGBMRanker -- the MEASURED winner over pointwise (+0.021/+0.011/
        +0.005 nDCG@10 scifact/nf/fiqa, multi-seed, _lambdamart_test.py). Falls back to pointwise HistGBDT if
        lightgbm is unavailable or ranker='pointwise'."""
        from _surv_edgeweight import build_drift_w
        from _arch_2nd_order import build_2nd_order
        scorp = self._stem_corpus(corpus)
        ids = [q for q in train_qrels if q in queries
               and any(train_qrels[q].get(d, 0) > 0 and d in self._d2i for d in train_qrels[q])]
        seeds = {w for q in ids for w in self._qwords(queries[q])}
        self._g_d1, _, _ = build_drift_w(scorp, seeds, mode="npmi")
        self._g_d2, _, _ = build_2nd_order(scorp, seeds)
        self._g_depth = depth; self._g_scorp = scorp; self._g_dwset = {}
        self._g_idf = {w: self._idf(int(self.df[t])) for w in seeds if (t := self._term_id(w)) is not None}
        self._g_dl = np.log1p(np.array([len(words(scorp[d])) for d in self.doc_ids], float))
        X, Y, G = [], [], []
        for qid in ids:
            fx, fy, _ = self._gbdt_feats(queries[qid], train_qrels[qid])
            if len(fy): X.append(fx); Y.append(fy); G.append(len(fy))
        if ranker == "lambdarank":
            try:
                import lightgbm as lgb
                keep = [i for i, y in enumerate(Y) if y.sum() > 0]      # lambdarank needs a positive per group
                Xl = np.vstack([X[i] for i in keep]); Yl = np.concatenate([Y[i] for i in keep])
                clf = lgb.LGBMRanker(objective="lambdarank", n_estimators=200, num_leaves=15, max_depth=4,
                                     learning_rate=0.1, min_child_samples=5, random_state=0, verbose=-1)
                clf.fit(Xl, Yl, group=[G[i] for i in keep])
                self._gbdt = clf; self._gbdt_kind = "lambdarank"
                return self
            except ImportError:
                pass                                                     # fall through to pointwise
        from sklearn.ensemble import HistGradientBoostingClassifier
        clf = HistGradientBoostingClassifier(max_iter=200, max_depth=4, class_weight="balanced", l2_regularization=1.0)
        clf.fit(np.vstack(X), np.concatenate(Y)); self._gbdt = clf; self._gbdt_kind = "pointwise"
        return self

    def _gbdt_dws(self, i):
        if i not in self._g_dwset: self._g_dwset[i] = set(words(self._g_scorp[self.doc_ids[i]]))
        return self._g_dwset[i]

    def _gbdt_feats(self, query, qrels=None):
        from _o1_drift import drift_bag, score_bag
        q = self._sq(query); qw = list(dict.fromkeys(self._qwords(query)))
        lex = self.score(query)
        dr = score_bag(self, drift_bag(q, self._g_d1, alpha=0.5))
        c2 = score_bag(self, drift_bag(q, self._g_d2, alpha=0.5))
        D = self._g_depth; srcs = []; rk = []
        for sc in (lex, dr, c2):
            o = np.argsort(sc)[::-1][:D]; srcs.append(set(o.tolist())); rk.append({int(dd): r for r, dd in enumerate(o)})
        u = list(set().union(*srcs))
        gold = {self._d2i[d] for d in (qrels or {}) if qrels.get(d, 0) > 0 and d in self._d2i} if qrels else set()
        qidf = {w: self._g_idf.get(w, 0.0) for w in qw}
        X, Y = [], []
        for i in u:
            m = [w for w in qw if w in self._gbdt_dws(i)]
            X.append([lex[i], dr[i], c2[i], sum(qidf[w] for w in m), len(m) / max(1, len(qw)), self._g_dl[i],
                      sum(1 for s in srcs if i in s), rk[0].get(i, D), rk[1].get(i, D), rk[2].get(i, D)])
            Y.append(1 if i in gold else 0)
        return np.array(X, float), np.array(Y), u

    def _search_gbdt(self, query, k=10):
        fx, _, u = self._gbdt_feats(query)
        if not len(u): return self.search(query, k)
        if getattr(self, "_gbdt_kind", "pointwise") == "lambdarank":
            p = self._gbdt.predict(fx)                         # ranker scores directly
        else:
            p = self._gbdt.predict_proba(fx)[:, 1]
        return [self.doc_ids[u[j]] for j in np.argsort(p)[::-1][:k]]

    # ---------------- autotune: the corpus-adaptive, never-regress gate ----------------
    def _tier_available(self, t):
        if t == "bridged": return hasattr(self, "bridge")
        if t == "gbdt": return getattr(self, "_gbdt", None) is not None
        if t == "distilled": return getattr(self, "_splade", None) is not None
        if t == "stack": return getattr(self, "_ce", None) is not None
        if t == "convbridge": return getattr(self, "_cbgraph", None) is not None
        if t == "doclattice": return getattr(self, "_docnbrs", None) is not None
        return t in ("lexical", "wand")

    def autotune(self, train_queries, train_qrels, candidates=("lexical", "wand", "bridged", "gbdt"), k=10, metric=None):
        """Corpus-adaptive GATE: evaluate available candidate tiers on TRAIN qrels (nDCG@10), freeze the argmax
        (ties -> cheapest tier), with a never-regress guarantee vs lexical. Sets the default used by retrieve('auto').
        STEM stays always-on (it's a build flag). One index feeds every tier, so the gate is ~0 serve cost."""
        if metric is None:
            from scripts.bench_supervised_bridges import ndcg10 as metric
        ids = [q for q in train_qrels if q in train_queries
               and any(train_qrels[q].get(d, 0) > 0 and d in self._d2i for d in train_qrels[q])]
        cost = {"lexical": 0, "wand": 1, "bridged": 2, "gbdt": 3, "distilled": 3, "stack": 4}
        avail = [t for t in candidates if self._tier_available(t)]
        sc = {}
        for t in avail:
            s = 0.0
            for qid in ids: s += metric(self._tier(train_queries[qid], t, k), train_qrels[qid])
            sc[t] = s / max(1, len(ids))
        best = max(avail, key=lambda t: (round(sc[t], 4), -cost.get(t, 9)))    # max nDCG, tie -> cheapest
        if "lexical" in sc and sc[best] < sc["lexical"] - 1e-9: best = "lexical"  # never-regress guarantee
        self._auto_tier = best; self._auto_scores = sc
        return self

    # ---------------- index-native drift: co-occurrence straight from the ingested CSR (no re-tokenization) ----------
    def build_drift_index(self, seeds, topk=24, df_hi=0.25, mode="raw", min_c=2, seed_cap=50000):
        """Build the drift co-occurrence graph DIRECTLY from the numba-ingested index -- the inverted index IS the
        doc x term presence matrix M (wraps indptr/seg_doc), so cooc = M_seed^T @ M_band is one sparse matmul with NO
        re-tokenization. Keyed by TERM-ID (index-native). Bounded by |seeds| (the query vocab) -> scales like ingest.
        This is the giant-corpus drift path (MARCO-ready)."""
        import scipy.sparse as sp
        N = self.N; cap = df_hi * N
        band = (self.df >= 2) & (self.df <= cap)                      # mid-freq mask (drop hapax + super-common)
        M = sp.csc_matrix((np.ones(len(self.seg_doc), np.float32), np.asarray(self.seg_doc), self.indptr), shape=(N, self.V))
        scap = min(cap, seed_cap)                                     # giant-corpus safety: seeds must be DISCRIMINATIVE
        seed_tids = np.array(sorted({t for w in seeds                 # (common terms co-occur with everything = pure cost,
                                     if (t := self._term_id(w)) is not None and band[t] and self.df[t] <= scap}), np.int64)  # no signal)
        if seed_tids.size == 0: self._drift_tid = {}; return self
        C = (M[:, seed_tids].T @ M).tocsr()                          # (n_seeds x V); full M = zero-copy wrapper of the index
        drift = {}
        for r in range(seed_tids.size):
            s = int(seed_tids[r]); row = C.getrow(r); idx = row.indices; cnt = row.data; ds = self.df[s]; scored = []
            for k in range(idx.size):
                j = int(idx[k]); c = float(cnt[k])
                if c < min_c or j == s or not band[j]: continue      # band-filter partners here (no M copy)
                pmi = math.log((c * N) / (ds * self.df[j]))
                if pmi <= 0: continue
                scored.append((j, pmi / (-math.log(c / N)) if mode == "npmi" else pmi))
            scored.sort(key=lambda x: -x[1])
            if scored: drift[s] = scored[:topk]
        self._drift_tid = drift
        return self

    def _score_bag_tid(self, bag):
        s = np.zeros(self.N); ip, sd, st, den, k1p1 = self.indptr, self.seg_doc, self.seg_tf, self._denom, self._k1p1
        for tid, qwt in bag.items():
            a, e = int(ip[tid]), int(ip[tid + 1]); dfp = e - a
            if dfp == 0: continue
            di = sd[a:e]; tf = st[a:e].astype(np.float64)
            s[di] += (qwt * self._idf(dfp) * k1p1) * tf / (tf + den[di])
        return s

    def _search_drift_native(self, query, k=10, alpha=0.4, max_exp=24):
        qtids = [t for w in dict.fromkeys(self._qwords(query)) if (t := self._term_id(w)) is not None]
        bag = {t: 1.0 for t in qtids}; exp = Counter()
        for t in qtids:
            for pt, pmi in self._drift_tid.get(t, ()):
                if pt not in bag: exp[pt] += pmi
        if exp:
            mx = max(exp.values())
            for pt, m in exp.most_common(max_exp): bag[pt] = bag.get(pt, 0.0) + alpha * (m / mx)
        sc = self._score_bag_tid(bag); top = np.argsort(sc)[::-1][:k]
        return [self.doc_ids[int(i)] for i in top if sc[i] > 0]

    # ---------------- coarse-to-fine cascade: per-query escalation gate ----------------
    def _cheap_margin(self, query):
        """confidence signal: normalized gap between the top-1 and top-2 lexical scores. High = the cheap tier is
        already confident (skip the heavy tier); low = ambiguous (escalate). ~free (one lexical scatter-add)."""
        sc = self.score(query)
        if sc.size < 2: return 1.0
        top = np.partition(sc, -2)[-2:]
        s1, s2 = float(max(top)), float(min(top))
        return (s1 - s2) / s1 if s1 > 0 else 0.0

    def tune_cascade(self, train_queries, train_qrels, cheap="wand", heavy=None, k=10, max_drop=0.003, metric=None):
        """Per-QUERY gate (complements autotune's per-CORPUS gate): run CHEAP; escalate to HEAVY only when the cheap
        top-1 margin < threshold. Tune the threshold on train to MAXIMIZE the skip rate while holding nDCG within
        max_drop of all-heavy. Sets the 'cascade' tier."""
        if metric is None:
            from scripts.bench_supervised_bridges import ndcg10 as metric
        self._casc_cheap = cheap; self._casc_heavy = heavy or getattr(self, "_auto_tier", "bridged")
        ids = [q for q in train_qrels if q in train_queries
               and any(train_qrels[q].get(d, 0) > 0 and d in self._d2i for d in train_qrels[q])]
        rows = []                                          # (margin, cheap_nDCG, heavy_nDCG) per train query
        for qid in ids:
            mg = self._cheap_margin(train_queries[qid])
            nc = metric(self._tier(train_queries[qid], self._casc_cheap, k), train_qrels[qid])
            nh = metric(self._tier(train_queries[qid], self._casc_heavy, k), train_qrels[qid])
            rows.append((mg, nc, nh))
        allheavy = float(np.mean([r[2] for r in rows])) if rows else 0.0
        best_thr, best_skip = 0.0, -1.0
        for thr in np.linspace(0, 1, 101):
            nd = float(np.mean([nc if mg >= thr else nh for mg, nc, nh in rows]))   # keep cheap iff margin>=thr
            skip = float(np.mean([1.0 if mg >= thr else 0.0 for mg, _, _ in rows]))
            if nd >= allheavy - max_drop and skip > best_skip: best_skip, best_thr = skip, thr
        self._casc_thr = best_thr; self._casc_train_skip = best_skip
        return self

    def _search_cascade(self, query, k=10):
        heavy = getattr(self, "_casc_heavy", "bridged"); cheap = getattr(self, "_casc_cheap", "wand")
        if self._cheap_margin(query) >= getattr(self, "_casc_thr", 1.0):
            return self._tier(query, cheap, k)             # confident -> cheap tier
        return self._tier(query, heavy, k)                 # ambiguous -> escalate

    # ---------------- mmap persistence (RAM = working set) ----------------
    def save_mmap(self, path):
        os.makedirs(path, exist_ok=True)
        np.save(f"{path}/seg_doc.npy", self.seg_doc); np.save(f"{path}/seg_tf.npy", self.seg_tf)
        np.save(f"{path}/indptr.npy", self.indptr); np.save(f"{path}/uniq_hash.npy", self.uniq_hash)
        np.save(f"{path}/doc_len.npy", self.doc_len)
        json.dump({"doc_ids": self.doc_ids, "V": self.V, "N": self.N, "stem": self._stem,
                   "bridge": getattr(self, "bridge", {})}, open(f"{path}/meta.json", "w"))
        return self

    @classmethod
    def load_mmap(cls, path):
        self = cls()
        self.seg_doc = np.load(f"{path}/seg_doc.npy", mmap_mode="r")
        self.seg_tf = np.load(f"{path}/seg_tf.npy", mmap_mode="r")
        self.indptr = np.load(f"{path}/indptr.npy"); self.uniq_hash = np.load(f"{path}/uniq_hash.npy")
        self.doc_len = np.load(f"{path}/doc_len.npy")
        m = json.load(open(f"{path}/meta.json"))
        self.doc_ids = m["doc_ids"]; self.V = m["V"]; self.N = m["N"]; self._stem = m.get("stem", False)
        self.bridge = {k: [tuple(x) for x in v] for k, v in m.get("bridge", {}).items()}
        self.hash2term = {int(h): i for i, h in enumerate(self.uniq_hash)}
        self.df = np.diff(self.indptr).astype(np.int64)
        self._finish()
        return self

    # ---------------- footprint codec: delta-gap doc-ids + 3-bit weights (smaller on-disk B/doc) ----------------
    def save_codec(self, path, weight_bits=3):
        """Compress the CSR: sort segments, encode doc-ids with the SMALLEST of gamma-gaps / fixed-width-gaps / BIC
        (never worse than fixed-width), and `weight_bits`-quantize the term weights. Doc-ids are LOSSLESS; weights
        near-lossless. Footprint trims (#5, near-lossless): doc_len f64->f16 on disk (upcast on load), doc_ids ->
        zlib blob out of meta.json (dominant metadata at scale), weight_bits=2 (tf 1..4) where BM25 saturates."""
        os.makedirs(path, exist_ok=True)
        if not getattr(self, "_sorted", False): self.sort_segments()      # doc-ids sorted within each term
        sd = np.asarray(self.seg_doc).astype(np.int64); ip = np.asarray(self.indptr, np.int64)
        method, payload = encode_docids_best(sd, ip, N=self.N)             # 'gamma'/'bic' -> (buf,) ; 'fixed' -> (buf, gw)
        gw = 0
        if method in ("gamma", "bic"):
            np.save(f"{path}/gaps.npy", payload[0])
        else:
            np.save(f"{path}/gaps.npy", payload[0]); gw = payload[1]
        wb = int(weight_bits); cap = 1 << wb                              # tf 1..cap; 2-bit=1..4, 3-bit=1..8
        tfq = (np.minimum(np.asarray(self.seg_tf), cap).astype(np.uint8) - 1)
        shifts = np.arange(wb - 1, -1, -1, dtype=np.uint8)
        bits = ((tfq[:, None] >> shifts) & 1).astype(np.uint8).ravel()
        np.save(f"{path}/tfq.npy", np.packbits(bits))
        np.save(f"{path}/indptr.npy", self.indptr); np.save(f"{path}/uniq_hash.npy", self.uniq_hash)
        dl = np.asarray(self.doc_len, np.float64)                         # #5: f64 -> f16 on disk (near-lossless)
        dl_ok = float(dl.max()) < 65000.0 if dl.size else True
        np.save(f"{path}/doc_len.npy", dl.astype(np.float16) if dl_ok else dl.astype(np.float32))
        np.savez_compressed(f"{path}/doc_ids.npz", doc_ids=np.array(self.doc_ids, dtype=object))  # #5: zlib blob
        json.dump({"V": self.V, "N": self.N, "stem": self._stem, "codec": True, "di_codec": method, "gw": gw,
                   "npost": int(sd.size), "wbits": wb, "doc_ids_blob": True, "bridge": getattr(self, "bridge", {})},
                  open(f"{path}/meta.json", "w"))
        return self

    @classmethod
    def load_codec(cls, path):
        self = cls()
        m = json.load(open(f"{path}/meta.json"))
        self.indptr = np.load(f"{path}/indptr.npy"); self.uniq_hash = np.load(f"{path}/uniq_hash.npy")
        self.doc_len = np.load(f"{path}/doc_len.npy").astype(np.float64)   # upcast (stored f16) for exact BM25 sums
        V = m["V"]; ip = np.asarray(self.indptr, np.int64)
        df = np.diff(ip).astype(np.int64); npost = m["npost"]; N = m["N"]
        gaps = np.load(f"{path}/gaps.npy")
        di_codec = m.get("di_codec", "fixed")
        if di_codec == "gamma":                                           # M1 gamma path (variable-length, lossless)
            self.seg_doc = decode_doc_gaps(gaps, ip, npost)
        elif di_codec == "bic":                                           # BIC path (variable-length, lossless)
            self.seg_doc = bic_unpack(gaps, ip, npost, N)
        else:                                                             # legacy fixed-width path
            self.seg_doc = fixed_gap_decode(gaps, ip, npost)
        wb = int(m.get("wbits", 3))                                       # weight bits (2-bit trim or legacy 3-bit)
        tfp = f"{path}/tfq.npy" if os.path.exists(f"{path}/tfq.npy") else f"{path}/tf3.npy"
        tb = np.unpackbits(np.load(tfp))[:npost * wb].reshape(-1, wb)
        self.seg_tf = (tb @ (1 << np.arange(wb - 1, -1, -1)) + 1).astype(np.uint16)
        if "doc_ids" in m:                                                # backward-compat: doc_ids may be inline
            self.doc_ids = m["doc_ids"]
        else:
            self.doc_ids = np.load(f"{path}/doc_ids.npz", allow_pickle=True)["doc_ids"].tolist()
        self.V = V; self.N = N; self._stem = m.get("stem", False)
        self.bridge = {k: [tuple(x) for x in v] for k, v in m.get("bridge", {}).items()}
        self.hash2term = {int(h): i for i, h in enumerate(self.uniq_hash)}
        self.df = df; self._sorted = True; self._finish()
        return self


def _selftest():
    from scripts.bench_supervised_bridges import load, ndcg10, recall10
    print("=" * 84)
    print("EDGE RAG ENGINE — fast ingest -> mmap serve -> bridges, end-to-end (reproduce edge champion)")
    print("=" * 84)
    corpus, queries, train_q, test_q = load("scifact")
    test_ids = [q for q in test_q if q in queries]

    EdgeRAG().build({"_w": "warm the numba jit so the ingest time is steady-state not cold"})  # warm JIT
    eng = EdgeRAG().build(corpus)
    eng.learn_bridges(queries, train_q, corpus)

    def evl(fn):
        nd = rc = 0.0; lat = []
        for qid in test_ids:
            t0 = time.perf_counter(); r = fn(queries[qid]); lat.append((time.perf_counter() - t0) * 1000)
            nd += ndcg10(r, test_q[qid]); rc += recall10(r, test_q[qid])
        n = len(test_ids); return nd / n, rc / n, float(np.median(lat))

    nd_lex, rc_lex, ms_lex = evl(lambda q: eng.search(q, 10))
    nd_br, rc_br, ms_br = evl(lambda q: eng.search_bridged(q, 10))
    print(f"\n  ingest: {eng.N:,} docs in {eng.ingest_s*1000:.0f} ms ({eng.N/eng.ingest_s:,.0f} docs/s), "
          f"{eng.V:,} terms, {len(eng.seg_doc):,} postings")
    print(f"  lexical (word-only) : nDCG {nd_lex:.4f}  Recall {rc_lex:.4f}  {ms_lex:.2f} ms/q   (edge champ C=0.6712)")
    print(f"  + counting-bridges  : nDCG {nd_br:.4f}  Recall {rc_br:.4f}  {ms_br:.2f} ms/q   (edge champ D=0.7112)")
    ok_lex = abs(nd_lex - 0.6712) < 0.004; ok_br = abs(nd_br - 0.7112) < 0.006
    print(f"  reproduces edge champion: lexical {'OK' if ok_lex else 'DRIFT'} | bridged {'OK' if ok_br else 'DRIFT'}")

    # optional trng lazy_chambers pool enrich (default off — must not move baseline bridged)
    eng_lat = EdgeRAG().build(corpus).learn_bridges(queries, train_q, corpus)
    eng_lat.attach_lattice_chambers(corpus, use_lattice_enrich=True)
    nd_lat, rc_lat, ms_lat = evl(lambda q: eng_lat.search_bridged(q, 10))
    print(f"  + lattice enrich    : nDCG {nd_lat:.4f}  Recall {rc_lat:.4f}  {ms_lat:.2f} ms/q   (dNDCG {nd_lat-nd_br:+.4f} vs bridges)")

    # mmap round-trip
    import tempfile
    p = os.path.join(tempfile.gettempdir(), "edge_rag_idx")
    eng.save_mmap(p); eng2 = EdgeRAG.load_mmap(p)
    nd2, _, _ = evl(lambda q: eng2.search_bridged(q, 10))
    print(f"  mmap round-trip serve: nDCG {nd2:.4f}  ({'MATCH' if abs(nd2-nd_br) < 1e-9 else 'DIFF'})")
    print("\n  one engine: numba radix ingest + mmap CSR serve + counting-bridges. small + fast + accurate.")


if __name__ == "__main__":
    _selftest()
