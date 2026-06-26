"""
aethos_bridges.py - supervised relevance bridges (the accuracy layer).

Learns query-term -> doc-term links by COUNTING relevant (query, gold-doc) pairs
from qrels - deterministic, append-only, verifiable, no gradient descent. At
query time the bridges rerank and EXPAND the lexical candidate pool (a doc
reachable through learned partners of the query's words enters even with no
query word in it), then a conservative lex + lam*bridge fusion ranks the pool.

The lift tracks how much training data + exploitable structure a corpus has:
big on lexically-clean scifact (+0.06 nDCG), modest on diverse corpora (+0.01).
See scripts/run_beir.py for the measured numbers and scripts/bench_*bridge*.py
for the method studies.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations

from aethos_append_index import words

try:
    import numpy as np
except ImportError:                       # pragma: no cover
    np = None


class RelevanceBridges:
    """query-term -> doc-term bridges learned by counting relevant pairs.

    Deterministic (counting), append-only (counts only grow), verifiable (every
    bridge traces to named train queries). P(dt|qt) is the fraction of qt's
    relevant pairs whose gold doc contains dt; only bridges seen in >= min_pairs
    distinct relevant pairs are kept (generalisation gate), weighted by the
    doc-term's corpus idf (specificity gate; suppresses generic words)."""

    def __init__(self, idx, N, min_pairs=2, top_per_term=12, idf_gate=1.5):
        self.idx = idx
        self.N = N
        self.min_pairs = min_pairs
        self.top_per_term = top_per_term
        self.idf_gate = idf_gate
        self.cooc = defaultdict(Counter)
        self.qt_pairs = Counter()
        self.bridge = {}
        # (r1, r2) -> [(doc_term, weight)] when gold has r2 but not literal r1
        self.corridor_bridge: dict[tuple[str, str], list[tuple[str, float]]] = {}

    def _idf(self, w):
        p = self.idx.token_prime.get(("w", w))
        return self.idx._idf(p, self.N) if p else 0.0

    def learn(self, queries, train_qrels, corpus):
        """Count co-occurrence over relevant pairs (cached + C-level update);
        the bridges are deterministic and order-independent."""
        idf_cache, gate = {}, self.idf_gate

        def idf(w):
            v = idf_cache.get(w)
            if v is None:
                p = self.idx.token_prime.get(("w", w))
                v = self.idx._idf(p, self.N) if p else 0.0
                idf_cache[w] = v
            return v

        qcache, dcache = {}, {}
        cooc, qtp = self.cooc, self.qt_pairs
        for qid, rels in train_qrels.items():
            if qid not in queries:
                continue
            qt_set = qcache.get(qid)
            if qt_set is None:
                qt_set = tuple(w for w in set(words(queries[qid])) if idf(w) >= gate)
                qcache[qid] = qt_set
            if not qt_set:
                continue
            for cid, sc in rels.items():
                if sc <= 0 or cid not in corpus:
                    continue
                dt_set = dcache.get(cid)
                if dt_set is None:
                    dt_set = frozenset(w for w in set(words(corpus[cid]))
                                       if idf(w) >= gate)
                    dcache[cid] = dt_set
                for qt in qt_set:
                    qtp[qt] += 1
                    if dt_set:
                        cooc[qt].update(dt_set)
        for qt, partners in cooc.items():
            np_ = qtp[qt]
            scored = [(dt, (c / np_) * idf(dt)) for dt, c in partners.items()
                      if dt != qt and c >= self.min_pairs]
            scored.sort(key=lambda x: (-x[1], x[0]))
            if scored:
                self.bridge[qt] = scored[:self.top_per_term]
        return self

    def learn_rarest_corridors(
        self,
        queries,
        train_qrels,
        corpus,
        *,
        min_pairs: int = 1,
        companion_idf_gate: float = 2.5,
        top_per_pair: int | None = None,
    ) -> RelevanceBridges:
        """Learn r1->doc-term bridges from gold where r2 is present but r1 is absent.

        Captures the 'company they keep' — rarest-2 anchors the doc; r1 reaches
        it through learned doc-term partners co-occurring with r2 in train gold.
        """
        idf_cache: dict[str, float] = {}
        top = top_per_pair or self.top_per_term

        def idf(w: str) -> float:
            v = idf_cache.get(w)
            if v is None:
                p = self.idx.token_prime.get(("w", w))
                v = self.idx._idf(p, self.N) if p else 0.0
                idf_cache[w] = v
            return v

        cooc: dict[tuple[str, str], Counter] = defaultdict(Counter)
        pair_hits: Counter[tuple[str, str]] = Counter()
        qcache: dict[str, tuple[str, str] | None] = {}

        for qid, rels in train_qrels.items():
            if qid not in queries:
                continue
            pair = qcache.get(qid)
            if pair is None:
                uniq = list(dict.fromkeys(
                    w for w in words(queries[qid]) if w.isalpha() and len(w) >= 3
                ))
                if len(uniq) < 2:
                    qcache[qid] = None
                    continue
                ranked = sorted(uniq, key=lambda w: (-idf(w), w))
                pair = (ranked[0], ranked[1])
                qcache[qid] = pair
            if pair is None:
                continue
            r1, r2 = pair
            for cid, sc in rels.items():
                if sc <= 0 or cid not in corpus:
                    continue
                dtoks = set(words(corpus[cid]))
                if r1 in dtoks or r2 not in dtoks:
                    continue
                pair_hits[(r1, r2)] += 1
                for dt in dtoks:
                    if dt == r1 or idf(dt) < companion_idf_gate:
                        continue
                    cooc[(r1, r2)][dt] += 1

        self.corridor_bridge.clear()
        for key, partners in cooc.items():
            np_ = pair_hits[key]
            if np_ < min_pairs:
                continue
            r1, r2 = key
            scored = [
                (dt, (c / np_) * idf(dt))
                for dt, c in partners.items()
                if c >= min_pairs and dt != r1
            ]
            scored.sort(key=lambda x: (-x[1], x[0]))
            if scored:
                self.corridor_bridge[key] = scored[:top]
                # lift r1 solo bridges from corridor evidence
                merged = dict(self.bridge.get(r1, ()))
                for dt, w in scored[:top]:
                    merged[dt] = max(merged.get(dt, 0.0), w * 0.85)
                if merged:
                    self.bridge[r1] = sorted(
                        merged.items(), key=lambda x: (-x[1], x[0]),
                    )[: self.top_per_term]
        return self

    def stats(self):
        n_cor = sum(len(v) for v in self.corridor_bridge.values())
        return len(self.bridge), sum(len(v) for v in self.bridge.values()) + n_cor


def rarest_query_pair(query: str, idf) -> tuple[str, str] | None:
    """Two rarest content words in query (by corpus idf)."""
    uniq = list(dict.fromkeys(
        w for w in words(query) if w.isalpha() and len(w) >= 3
    ))
    if len(uniq) < 2:
        return None
    ranked = sorted(uniq, key=lambda w: (-idf(w), w))
    return ranked[0], ranked[1]


def corridor_bridge_expansion(
    idx,
    br: RelevanceBridges,
    query: str,
    *,
    idf,
    companion_boost: float = 1.35,
) -> dict[str, float]:
    """Expand via (r1, r2) corridor when query has both rarest terms."""
    if not br.corridor_bridge or idf is None:
        return {}
    pair = rarest_query_pair(query, idf)
    if pair is None:
        return {}
    r1, r2 = pair
    partners = br.corridor_bridge.get(pair)
    if not partners:
        return {}
    exp: dict[str, float] = defaultdict(float)
    r2_prime = idx.token_prime.get(("w", r2))
    r2_post = idx.postings.get(r2_prime, {}) if r2_prime else {}
    for dt, w in partners:
        p = idx.token_prime.get(("w", dt))
        if p is None:
            continue
        for d, tf in idx.postings.get(p, {}).items():
            if d not in idx.alive:
                continue
            boost = companion_boost if d in r2_post else 1.0
            exp[d] += w * boost * tf / (tf + 1.0)
    return exp


def bridge_expansion(idx, br, query, *, idf=None, hub_idf_gate: float = 0.0, hub_blocklist=None,
                     use_corridors: bool = True):
    """Doc weights from supervised / knowledge term bridges.

    hub_idf_gate: skip bridge fan from ultra-common query terms (0 or >=50 = off).
    hub_blocklist: extra terms to skip (cross-corpus learned hub diluters).
    """
    exp = defaultdict(float)
    gate = hub_idf_gate if 0 < hub_idf_gate < 50 else 0.0
    block = set(hub_blocklist or ())
    for qt in set(words(query)):
        if qt in block:
            continue
        if idf is not None and gate > 0 and idf(qt) < gate:
            continue
        for dt, w in br.bridge.get(qt, ()):
            p = idx.token_prime.get(("w", dt))
            if p is None:
                continue
            for d, tf in idx.postings.get(p, {}).items():
                if d in idx.alive:
                    exp[d] += w * tf / (tf + 1.0)
    if use_corridors and idf is not None and getattr(br, "corridor_bridge", None):
        for d, s in corridor_bridge_expansion(idx, br, query, idf=idf).items():
            exp[d] += s
    return exp


def bridge_search(idx, br, query, lam=0.25, n_expand=20, k=10):
    """Rerank + pool-expand on the DICT path (reflects appended docs immediately,
    no finalize needed). Lexical candidates plus bridge-reachable docs, fused."""
    lex = idx._score(query)
    cand = sorted(lex, key=lambda d: lex[d], reverse=True)[:100]
    exp = bridge_expansion(idx, br, query)
    cset = set(cand)
    extra = [d for d in sorted(exp, key=lambda d: exp[d], reverse=True)
             if d not in cset][:n_expand]
    pool = cand + extra
    if not pool:
        return []
    lmax = max((lex.get(d, 0.0) for d in pool), default=1.0) or 1.0
    emax = max(exp.values()) if exp else 1.0
    final = {d: lex.get(d, 0.0) / lmax + lam * exp.get(d, 0.0) / emax for d in pool}
    return sorted(final, key=lambda d: final[d], reverse=True)[:k]


def bridge_search_dense(idx, br, query, lam=0.25, n_expand=20, k=10):
    """Rerank + pool-expand on the DENSE fast path (~9x faster; requires
    finalize()). Same ranking as bridge_search up to float16 tie-ordering."""
    if np is None:
        raise RuntimeError("bridge_search_dense requires numpy")
    lex, docs = idx.dense_scores(query)
    nz = np.nonzero(lex)[0]
    if nz.size == 0:
        return []
    cand = nz[np.argpartition(lex[nz], -100)[-100:]] if nz.size > 100 else nz
    exp = np.zeros(len(docs), dtype=np.float64)
    for qt in set(words(query)):
        for dt, w in br.bridge.get(qt, ()):
            p = idx.token_prime.get(("w", dt))
            if p is None:
                continue
            post = idx.dense_posting(p)
            if post is None:
                continue
            di, tfa = post
            exp[di] += w * tfa / (tfa + 1.0)
    cset = set(int(i) for i in cand)
    enz = np.nonzero(exp)[0]
    extra = ([int(i) for i in enz[np.argsort(exp[enz])[::-1]] if int(i) not in cset][:n_expand]
             if enz.size else [])
    pool = [int(i) for i in cand] + extra
    lmax = max(lex[i] for i in pool) or 1.0
    emax = exp.max() or 1.0
    final = {i: lex[i] / lmax + lam * exp[i] / emax for i in pool}
    return [docs[i] for i in sorted(final, key=final.get, reverse=True)[:k]]


class GoldDocBridges:
    """Gold-doc triangulation (the champion bridge from trng / aethos_master).

    Per-pair RelevanceBridges links one query-term to one doc-term. This instead
    works PER GOLD DOC: a gold doc becomes reachable by the union of the rare /
    high-idf words from ALL training queries that hit it, PLUS the doc's own rare
    words -- kept only if globally DISCRIMINATIVE (appears in <= df_frac of gold
    docs), plus rare-word bigram bridges. That cross-query union is the extra
    generalisation: a held-out query reaches a doc because it shares a rare word
    with a DIFFERENT training query that hit the same doc.

    HIGH-VARIANCE SPECIALIST (held-out, verify_golddoc.py, 3 corpora): SciFact
    +6.5pp (beats per-pair's +3.5pp) BUT NFCorpus +0.7pp (per-pair +1.4pp wins)
    and FiQA -6.8pp (HARMFUL -- 0% gold-doc reuse, so it boosts wrong docs and
    demotes correct ones). Big win only when gold docs are a SELECTIVE subset WITH
    reuse. Use `choose_bridge()` to route safely; per-pair is the robust default.
    Do NOT stack with per-pair bridges -- they dilute it."""

    def __init__(self, idx, N, rare_gate=3.0, df_frac=0.15, top_per_doc=25, ng_top=8):
        self.idx = idx
        self.N = N
        self.rare_gate = rare_gate
        self.df_frac = df_frac
        self.top_per_doc = top_per_doc
        self.ng_top = ng_top
        self.word_docs = defaultdict(list)        # BRIDGE:w     -> [gold docs]
        self.ng_docs = defaultdict(list)          # BRIDGE_NG:ab -> [gold docs]
        self._idf = {}

    def idf(self, w):
        v = self._idf.get(w)
        if v is None:
            p = self.idx.token_prime.get(("w", w))
            v = self.idx._idf(p, self.N) if p else 0.0
            self._idf[w] = v
        return v

    def _rare(self, text):
        return {w for w in set(words(text)) if self.idf(w) >= self.rare_gate}

    def learn(self, queries, train_qrels, corpus):
        gold_to_qids = defaultdict(set)
        for qid, rels in train_qrels.items():
            for d, sc in rels.items():
                if sc > 0:
                    gold_to_qids[d].add(qid)
        cands, dfc = {}, Counter()
        for d, qids in gold_to_qids.items():
            if d not in corpus:
                continue
            aqr = set()
            for qid in qids:
                if qid in queries:
                    aqr |= self._rare(queries[qid])
            c = aqr | self._rare(corpus[d])
            cands[d] = (c, aqr)
            for w in c:
                dfc[w] += 1
        thr = max(len(gold_to_qids) * self.df_frac, 3)
        for d, (c, aqr) in cands.items():
            disc = sorted((w for w in c if dfc[w] <= thr),
                          key=lambda w: -self.idf(w))[:self.top_per_doc]
            for w in disc:
                self.word_docs[w].append(d)
            qr = sorted(aqr, key=lambda w: -self.idf(w))[:self.ng_top]
            for a, b in combinations(sorted(qr), 2):
                self.ng_docs[(a, b)].append(d)
        return self

    def stats(self):
        return len(self.word_docs), sum(len(v) for v in self.word_docs.values())

    def expand(self, query):
        """{doc: weight} -- gold docs reachable from the query's rare words."""
        exp = defaultdict(float)
        rw = self._rare(query)
        for w in rw:
            iw = self.idf(w)
            for d in self.word_docs.get(w, ()):
                exp[d] += iw
        for a, b in combinations(sorted(rw, key=lambda w: -self.idf(w))[:self.ng_top], 2):
            for d in self.ng_docs.get((a, b), ()):
                exp[d] += self.idf(a) + self.idf(b)
        return exp


def golddoc_search(idx, gb, query, lam=0.25, n_expand=20, k=10):
    """Lexical candidates + gold-doc-triangulation pool expansion, fused.
    The production semantic-correlation search (replaces bridge_search)."""
    lex = idx._score(query)
    cand = sorted(lex, key=lex.get, reverse=True)[:100]
    exp = gb.expand(query)
    cset = set(cand)
    extra = [d for d in sorted(exp, key=exp.get, reverse=True) if d not in cset][:n_expand]
    pool = cand + extra
    if not pool:
        return []
    lmax = max((lex.get(d, 0.0) for d in pool), default=1.0) or 1.0
    emax = max(exp.values()) if exp else 1.0
    final = {d: lex.get(d, 0.0) / lmax + lam * exp.get(d, 0.0) / emax for d in pool}
    return sorted(final, key=final.get, reverse=True)[:k]


def choose_bridge(idx, N, queries, train_qrels, corpus):
    """Route to the SAFE semantic-correlation bridge for THIS corpus.

    Three-corpus audit (scifact/nfcorpus/fiqa): gold-doc triangulation is a big win
    ONLY when gold docs are a selective subset WITH reuse (scifact +6.5pp); it is
    weaker when nearly all docs are gold (nfcorpus +0.7 vs per-pair +1.4), and
    HARMFUL with no reuse (fiqa -6.8pp). Per-pair bridges are positive on all three
    and never harmful, so they are the default; gold-doc is enabled only when the
    structure favours it (gold_frac < 0.5 AND reuse_frac >= 0.2 -- routes correctly
    on all three). For production prefer validating both on a held-out slice.

    Returns (bridge_obj, search_fn, name, info)."""
    gold = defaultdict(set)
    for qid, rels in train_qrels.items():
        for d, sc in rels.items():
            if sc > 0 and d in corpus:
                gold[d].add(qid)
    n_gold = len(gold)
    gold_frac = n_gold / max(len(corpus), 1)
    reuse_frac = (sum(1 for d in gold if len(gold[d]) > 1) / n_gold) if n_gold else 0.0
    info = {"gold_frac": round(gold_frac, 3), "reuse_frac": round(reuse_frac, 3)}
    if gold_frac < 0.5 and reuse_frac >= 0.2:
        gb = GoldDocBridges(idx, N).learn(queries, train_qrels, corpus)
        return gb, golddoc_search, "gold-doc", info
    br = RelevanceBridges(idx, N).learn(queries, train_qrels, corpus)
    return br, bridge_search, "per-pair", info
