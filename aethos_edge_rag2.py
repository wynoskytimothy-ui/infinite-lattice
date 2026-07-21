"""
aethos_edge_rag2.py -- EdgeRAG-2 / AETHOS-Meet Core: the ground-up assembly from the 7-repo audit (_GROUNDUP_BUILD_SPEC.md).

It is the proven EdgeRAG spine + the four measured net-new wins - one deleted headline layer:
  M1  gamma-di + fixed best-of codec (save_codec)         -- doc-ids LOSSLESS, ~2x smaller than fixed-width.
  M2  query-shape serve router (route())                  -- WAND few-term / scatter many-term (both exact on BEIR;
                                                              the exact block-max-WAND kernel is the MARCO port, M6).
  M3  doc-side bake at ingest (bake_M)                     -- inject each doc's top-M NPMI correlates as phantom
                                                              postings; +0.041 R@100 / +0.025 nDCG at fast-path speed.
  M4  supervised tier: counting-bridges (nDCG) + GBDT      -- GBDT is a RECALL@100 tier, NOT an nDCG lever (measured).
  M5  governor: margin autotune + elimination ledger       -- never-regress with a MARGIN (won't collapse to champion
                                                              under qrel-density mismatch); refuses the 7-month graveyard.
DELETED: recursive/z_depth lattice (+0.0000 at every shippable budget). Formulas that survive by measurement: the
invertible min-plus MEET (storage/integrity + tropical-WAND), never as a scoring layer.
"""
from __future__ import annotations
import os, time
from collections import Counter
from typing import Any, Dict, List, Optional
import numpy as np
from aethos_edge_rag import EdgeRAG
from _fast_tok import words

# The elimination ledger: constructs MEASURED to fail/reduce across 7 months. The governor refuses to enable these,
# so the graveyard can never silently creep back into a build (each tag traces to a captured negative run).
ELIMINATION_LEDGER = frozenset({
    "recursive_lattice",        # +0.0000 at every shippable storage budget (_o1_recursive_lattice_scifact.txt)
    "chamber_routing",          # 6/6 negative, 7-28pp recall loss
    "complex_rotation_rank",    # 2x negative (nf +0.0001, sf +0.000)
    "meet_conjunction_fusion",  # RRF-fusing the meet drags nDCG -0.13
    "docdoc_bare_view",         # bare RRF doc-doc view crashes scifact 0.699->0.50 (reranker-only)
    "query_time_drift_rank",    # craters nDCG 0.696->0.582 + 3-8x slower (bake at ingest instead)
    "proximity_order_rerank",   # hurt all 3 corpora
    "algebraic_number_store",   # 290.7 B/doc, loses 1.22x to a plain delta list
    "exact_key_hardpin",        # unique term only ~50-61% gold; idf already handles it
    "bmw_on_splade",            # block-max WAND 14-60x slower on many-term SPLADE
    "chamber_codec",            # 483 B/doc, worst in the sweep
})


def bake_corpus(corpus: Dict[Any, str], M: int = 16, mode: str = "npmi", drift=None):
    """M3 doc-side bake: append each doc's top-M co-occurrence correlates (NPMI) as phantom postings (tf=1 words the
    tokenizer will index). Co-locates a doc's OWN correlates -> lifts near-lexical recall AND nDCG at fast serve; the
    length-norm keeps the low-weight phantoms from polluting precision (query-time drift, by contrast, craters nDCG).
    Returns (enriched_corpus, drift). Small-corpus path (NPMI is O(V^2)); >=100k uses the SPLADE-distill bake (M6)."""
    if M <= 0:
        return dict(corpus), None
    if mode == "npmi" and len(corpus) > 150_000:                   # NPMI co-occurrence is O(V^2) -> small-corpus only;
        return dict(corpus), None                                  # >=100k uses the SPLADE-distill bake (M6 scale path)
    from _o1_drift import build_drift_fast
    if drift is None:
        drift, _, _ = build_drift_fast(corpus, {w for d in corpus.values() for w in words(d)}, mode=mode)
    out = {}
    for d, txt in corpus.items():
        dw = set(words(txt)); acc = Counter()
        for w in dw:
            for cw, pmi in drift.get(w, ()):
                if cw not in dw:
                    acc[cw] += pmi
        out[d] = txt + ((" " + " ".join(cw for cw, _ in acc.most_common(M))) if acc else "")
    return out, drift


class EdgeRAG2:
    def __init__(self, stem: bool = True, bake_M: int = 16, bake_mode: str = "npmi",
                 margin: float = 0.002, few_term: int = 4, continuum: bool = False,
                 append_pool_union: bool = False, append_pool_k: int = 200):
        self.stem = stem; self.bake_M = bake_M; self.bake_mode = bake_mode
        self.margin = margin; self.few_term = few_term
        self.append_pool_union = append_pool_union
        self.append_pool_k = append_pool_k
        self.lex: Optional[EdgeRAG] = None
        self._enriched: Dict[Any, str] = {}
        self._drift = None
        self._auto_tier = "lexical"; self._gov_scores: Dict[str, float] = {}; self._gov_log = ""
        self._corpus_raw: Dict[Any, str] = {}
        self._ce = None; self._ce_depth = 100; self._ce_margin = 0.05; self._ce_fired = 0; self._ce_seen = 0
        self._apex_ok = False; self._apex_log = ""
        self._continuum = None; self._continuum_on = continuum

    # ---- ingest (M3 bake -> M1 gamma-codec-capable lexical lattice) ----
    def build(self, corpus: Dict[Any, str]) -> "EdgeRAG2":
        self._corpus_raw = dict(corpus)
        self._enriched, self._drift = bake_corpus(corpus, self.bake_M, self.bake_mode)
        self.lex = EdgeRAG().build(self._enriched, stem=self.stem)
        self.lex._corpus = self._corpus_raw  # label-free 3-way meet expansion needs raw doc text
        if self.append_pool_union:
            self.lex.configure_append_pool_union(True, self.append_pool_k)
        if self._continuum_on:
            from aethos_edge_continuum import attach_continuum
            self._continuum = attach_continuum(self.lex, corpus)
        return self

    # ---- M4 supervised tiers + M5 governor ----
    def learn(self, queries: Dict[Any, str], train_qrels: Dict[Any, Dict[Any, int]],
              tiers=("bridged", "gbdt")) -> "EdgeRAG2":
        if "bridged" in tiers:
            self.lex.learn_bridges(queries, train_qrels, self._enriched)
        if "gbdt" in tiers:
            self.lex.attach_gbdt(queries, train_qrels, self._enriched)
        self._meet3_log = self.lex.fit_tier_governor(queries, train_qrels)
        self._govern(queries, train_qrels)
        return self

    def attach_lattice_chambers(self, corpus=None, **kw) -> "EdgeRAG2":
        """Delegate to underlying EdgeRAG trng lazy_chambers sidecar (opt-in pool enrich)."""
        src = corpus or self._corpus_raw or self._enriched
        self.lex.attach_lattice_chambers(src, **kw)
        return self

    def attach_plane_walk(self, **kw) -> "EdgeRAG2":
        self.lex.attach_plane_walk(**kw)
        return self

    def enable_append_pool_union(self, k: int = 200) -> "EdgeRAG2":
        """Union BM25 top-k into bridged pool (hybrid append_pool_union lever, +recall on corridor misses)."""
        self.append_pool_union = True
        self.append_pool_k = k
        if self.lex is not None:
            self.lex.configure_append_pool_union(True, k)
        return self

    def _govern(self, queries, train_qrels, k: int = 10) -> None:
        """M5: evaluate candidate tiers on TRAIN nDCG@10; promote past lexical ONLY if it beats lexical by >= margin
        (prevents the 'always champion' collapse under train/test qrel-density mismatch). Ledger-blocked tiers can never
        be chosen. GBDT is scored but flagged recall-tier (kept out of the nDCG argmax unless it also wins nDCG)."""
        from scripts.bench_supervised_bridges import ndcg10
        ids = [q for q in train_qrels if q in queries
               and any(train_qrels[q].get(d, 0) > 0 and d in self.lex._d2i for d in train_qrels[q])]
        cand = [t for t in ("lexical", "bridged", "continuum", "gbdt")
                if t not in ELIMINATION_LEDGER
                and (t == "continuum" and self._continuum is not None
                     or t != "continuum" and self.lex._tier_available(t))]
        sc = {}
        for t in cand:
            s = 0.0
            for qid in ids:
                if t == "continuum":
                    r = self._continuum.search(queries[qid], k)
                else:
                    r = self.lex._tier(queries[qid], t, k)
                s += ndcg10(r, train_qrels[qid])
            sc[t] = s / max(1, len(ids))
        base = sc.get("lexical", 0.0)
        promoted = [t for t in cand if t != "lexical" and sc[t] >= base + self.margin]
        cost = {"lexical": 0, "bridged": 2, "continuum": 2, "gbdt": 3}
        best = max(promoted, key=lambda t: (round(sc[t], 4), -cost[t])) if promoted else "lexical"
        self._auto_tier = best; self._gov_scores = sc
        self._gov_log = (f"governor: base(lexical)={base:.4f}; " +
                         ", ".join(f"{t}={sc[t]:.4f}" for t in cand if t != "lexical") +
                         f"; margin={self.margin} -> tier='{best}'")

    # ---- M2 query-shape serve router ----
    def route(self, query: str) -> str:
        """Pick the serve kernel by query shape: few-term -> WAND (rarest-anchor, exact block-max is the MARCO port);
        many-term/expanded -> scatter-add (SIMD, the right primitive for many non-neg terms). Both exact on BEIR."""
        n = len({w for w in words(query)})
        return "wand" if n <= self.few_term else "scatter"

    def build_blockmax(self, B: int = 128) -> "EdgeRAG2":
        """Build the exact block-max WAND sidecar so the router serves few-term queries with pruned EXACT top-k -- the
        fast lexical serve at MARCO scale (6.5ms exact / 37x vs dense-accumulator on 8.8M)."""
        self.lex.build_blockmax(B); return self

    def _serve_lexical(self, query: str, k: int) -> List[Any]:
        # M2 router: few-term -> exact block-max WAND (if built); many-term/no-sidecar -> exact scatter-add. Both exact.
        if self.route(query) == "wand" and getattr(self.lex, "_blk", None) is not None:
            return self.lex.search_bmw(query, k)
        return self.lex.search(query, k)

    def _base(self, query: str, k: int) -> List[Any]:
        t = self._auto_tier
        return self._serve_lexical(query, k) if t == "lexical" else self.lex.retrieve(query, k, tier=t)

    # ---- M7 optional CE apex (off by default): fire the CE only on UNCERTAIN queries (cheap margin gate) ----
    def attach_ce(self, ce, depth: int = 100, fire_margin: float = 0.05) -> "EdgeRAG2":
        """Attach a cross-encoder as the optional text-reading apex. It fires ONLY when the cheap ranker is uncertain
        (top-1/top-2 BM25 margin < fire_margin) -- so mean latency stays low. Off unless retrieve(tier='apex')."""
        self._ce = ce; self._ce_depth = depth; self._ce_margin = fire_margin; self._ce_fired = self._ce_seen = 0
        return self

    def enable_apex(self, train_queries, train_qrels, sample: int = 120) -> "EdgeRAG2":
        """Governor decision for the CE apex: measure apex vs the no-GPU base on a TRAIN sample and enable it for 'auto'
        ONLY if it beats the base by >= margin. Out-of-domain CE HURTS aligned corpora (scifact -0.036) but helps
        semantic-bound ones (nfcorpus +0.013) -- so the apex is per-corpus governed, never blanket-on."""
        from scripts.bench_supervised_bridges import ndcg10
        ids = [q for q in train_qrels if q in train_queries
               and any(train_qrels[q].get(d, 0) > 0 for d in train_qrels[q])][:sample]
        base = ce = 0.0
        for qid in ids:
            base += ndcg10(self._base(train_queries[qid], 10), train_qrels[qid])
            ce += ndcg10(self._serve_apex(train_queries[qid], 10), train_qrels[qid])
        n = max(1, len(ids)); self._apex_ok = (ce / n) >= (base / n) + self.margin
        self._apex_log = f"apex governor: train base={base/n:.4f} apex={ce/n:.4f} -> {'ENABLED' if self._apex_ok else 'kept OFF (base wins)'}"
        return self

    def _serve_apex(self, query: str, k: int) -> List[Any]:
        self._ce_seen += 1
        pool = self._base(query, self._ce_depth)
        sc = self.lex.score(query); ids = [self.lex._d2i[d] for d in pool if d in self.lex._d2i]
        if self._ce is None or len(ids) < 2:
            return pool[:k]
        s = np.sort(sc[ids])[::-1]; margin = (s[0] - s[1]) / (abs(s[0]) + 1e-9)
        if margin >= self._ce_margin:                          # confident -> skip the CE (the gate's whole point)
            return pool[:k]
        self._ce_fired += 1
        cs = self._ce.predict([(query, self._corpus_raw[d][:512]) for d in pool], show_progress_bar=False)
        return [pool[i] for i in np.argsort(-np.asarray(cs))][:k]

    # ---- serve ----
    def retrieve(self, query: str, k: int = 10, tier: str = "auto") -> List[Any]:
        if tier in ELIMINATION_LEDGER:
            raise ValueError(f"tier '{tier}' is in the elimination ledger (measured-negative)")
        if tier == "apex":
            return self._serve_apex(query, k)
        if tier == "auto":
            if self._apex_ok and self._ce is not None:         # governor enabled the CE apex for this corpus
                return self._serve_apex(query, k)
            tier = self._auto_tier
        if tier == "lexical":
            return self._serve_lexical(query, k)
        if tier == "continuum":
            if self._continuum is None:
                raise ValueError("continuum tier requires build(continuum=True) or attach_continuum()")
            return self._continuum.search(query, k)
        return self.lex.retrieve(query, k, tier=tier)

    # ---- M1 footprint codec ----
    def save_codec(self, path: str) -> "EdgeRAG2":
        self.lex.save_codec(path); return self

    def bytes_per_doc(self, path: Optional[str] = None) -> float:
        import tempfile
        p = path or os.path.join(tempfile.gettempdir(), "edge_rag2_codec")
        self.lex.save_codec(p)
        files = ("gaps.npy", "tf3.npy", "indptr.npy", "uniq_hash.npy", "doc_len.npy")
        return sum(os.path.getsize(os.path.join(p, f)) for f in files if os.path.exists(os.path.join(p, f))) / self.lex.N


def _proof(name: str = "scifact", *, append_pool_union: bool = False, max_queries: int = 0):
    import sys
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass
    from scripts.bench_supervised_bridges import load, ndcg10, recall10
    corpus, queries, train_q, test_q = load(name)
    test_ids = [q for q in test_q if q in queries]
    if max_queries > 0:
        test_ids = test_ids[:max_queries]

    base = EdgeRAG().build(corpus)                                     # plain lexical baseline (no bake, no codec)
    eng = EdgeRAG2(stem=True, bake_M=16, append_pool_union=append_pool_union).build(corpus)
    eng.learn(queries, train_q)

    def evl(fn):
        nd = rc = 0.0; lat = []
        for qid in test_ids:
            t = time.perf_counter(); r = fn(qid); lat.append((time.perf_counter() - t) * 1000)
            nd += ndcg10(r, test_q[qid]); rc += recall10(r, test_q[qid])
        n = len(test_ids); return nd / n, rc / n, float(np.median(lat))

    print("=" * 92); print(f"EdgeRAG-2 (AETHOS-Meet Core) end-to-end proof -- {name}: {len(corpus):,} docs"); print("=" * 92)
    b_nd, b_rc, b_ms = evl(lambda q: base.search(queries[q], 10))
    print(f"  {'baseline EdgeRAG (no bake)':<34}nDCG {b_nd:.4f}  R@10 {b_rc:.4f}  {b_ms:.2f}ms")
    eng_cont = EdgeRAG2(stem=True, bake_M=16, continuum=True).build(corpus)
    eng_cont.learn(queries, train_q)
    for label, tier in [("M3 baked lexical", "lexical"), ("M4 baked+bridged (nDCG tier)", "bridged"),
                        ("M8 continuum+lazy meets", "continuum"),
                        ("M4 baked+GBDT (recall tier)", "gbdt"), ("M5 governor 'auto'", "auto")]:
        runner = eng_cont if tier == "continuum" else eng
        nd, rc, ms = evl(lambda q, t=tier, r=runner: r.retrieve(queries[q], 10, tier=t))
        rc100 = np.mean([len(set(runner.retrieve(queries[q], 100, tier=tier)) & set(test_q[q])) /
                         max(1, len([d for d in test_q[q] if test_q[q][d] > 0])) for q in test_ids])
        suffix = " [append_pool_union]" if append_pool_union and tier == "bridged" else ""
        print(f"  {label + suffix:<34}nDCG {nd:.4f}  R@10 {rc:.4f}  R@100 {rc100:.4f}  {ms:.2f}ms")
    print(f"\n  M1 footprint: {eng.bytes_per_doc():.1f} B/doc (gamma-di codec, doc-ids lossless)")
    print(f"  M2 router: '{queries[test_ids[0]][:40]}...' -> {eng.route(queries[test_ids[0]])}")
    print(f"  M5 {eng._gov_log}")
    print(f"  meet3 governor: {getattr(eng, '_meet3_log', {})}")
    print(f"  elimination ledger active: {len(ELIMINATION_LEDGER)} measured-negative constructs blocked")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="EdgeRAG-2 end-to-end proof")
    p.add_argument("corpus", nargs="?", default="scifact")
    p.add_argument(
        "--append-pool-union",
        action="store_true",
        help="Union BM25 top-200 into bridged candidate pool",
    )
    p.add_argument("--max-queries", type=int, default=0)
    args = p.parse_args()
    _proof(args.corpus, append_pool_union=args.append_pool_union, max_queries=args.max_queries)
