"""
Multi-corpus brain — one shared prime alphabet, many corpus lattices.

Each corpus gets a root prime (view ``("C", name)``) that branches its own
posting lattice, BIT 3 κ attractor dot-cloud, and optional route labels.
Training stacks additively via ``stack_corpus()``; symbols/subwords are shared.

    brain = MultiCorpusBrain()
    brain.stack_corpus("scifact", corpus_a, queries=q, train_qrels=train,
                       route_labels={"crispr", "phosphorylation"})
    brain.stack_corpus("nfcorpus", corpus_b)
    brain.teach("scifact", "biomaterials osteogenesis scaffold ...")
    result = brain.search("vitamin worms longevity")   # auto-route + κ + routed PRF
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from aethos_append_index import AppendOnlyLatticeIndex, words
from aethos_hub_signature import build_all_hub_signatures
from aethos_promotion import PromotionRegistry
from aethos_gap_miner import GapReport, mine_query_gaps
from aethos_encyclopedia_teacher import TeachGapResult, teach_gaps_for_corpus
from aethos_symbol_subjects import subjects_for_dataset, vote_query_chambers
from aethos_teach_store import TeachStore
from aethos_bridges import RelevanceBridges
from aethos_vocab_gap_router import GapSignal, choose_expansion_mode, routed_search
from core.primes import chain_primes
from pipeline.bit_03_doc_attractor_set import (
    CorpusAttractorIndex,
    build_attractor_index_from_hub_signatures,
)
from pipeline.bit_04_candidate_router import (
    candidates_from_attractors,
    query_words_for_routing,
)


class IdfCache:
    """Per-branch idf cache."""

    def __init__(self, idx, n_docs: int):
        self.idx, self.n = idx, n_docs
        self._c: dict[str, float] = {}

    def __call__(self, w: str) -> float:
        v = self._c.get(w)
        if v is None:
            p = self.idx.token_prime.get(("w", w))
            v = self.idx._idf(p, self.n) if p else 0.0
            self._c[w] = v
        return v


@dataclass
class WordAttractorIndex:
    """Fallback rare-word buckets when κ index is not built."""

    by_word: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    doc_rare: dict[str, set[str]] = field(default_factory=dict)

    def build(
        self,
        texts: dict[str, str],
        idf,
        *,
        rare_gate: float = 3.0,
        top_k: int = 12,
    ) -> None:
        for doc_id, text in texts.items():
            ws = set(words(text))
            rare = sorted(
                (w for w in ws if idf(w) >= rare_gate),
                key=idf,
                reverse=True,
            )[:top_k]
            self.doc_rare[doc_id] = set(rare)
            for w in rare:
                bucket = self.by_word[w]
                if doc_id not in bucket:
                    bucket.append(doc_id)

    def route(self, query: str, idf, *, rare_gate: float = 2.5) -> list[str]:
        seen: list[str] = []
        hit: set[str] = set()
        for w in set(words(query)):
            if idf(w) < rare_gate and w not in self.by_word:
                continue
            for d in self.by_word.get(w, []):
                if d not in hit:
                    hit.add(d)
                    seen.append(d)
        return seen


@dataclass
class CorpusBranch:
    """One corpus lattice under a root prime."""

    name: str
    root_prime: int
    idx: AppendOnlyLatticeIndex
    texts: dict[str, str]
    teach: TeachStore
    attractor: WordAttractorIndex
    pair_bridges: RelevanceBridges | None = None
    kappa_index: CorpusAttractorIndex | None = None
    route_labels: set[str] = field(default_factory=set)
    expansion_mode: str = "strict"
    subjects: frozenset[int] = field(default_factory=frozenset)
    gap_info: dict = field(default_factory=dict)

    @property
    def n_docs(self) -> int:
        return len(self.idx.alive)

    def global_id(self, local_id: str) -> str:
        return f"{self.name}/{local_id}"


@dataclass(frozen=True)
class SearchResult:
    corpus: str
    ranked: list[str]
    signal: GapSignal | None
    route_scores: dict[str, float]
    local_ids: list[str]
    kappa_candidates: int = 0
    kappa_keys: int = 0


@dataclass
class LearnIteration:
    """One mine → teach → eval cycle."""

    step: int
    n_miss: int
    terms_new: int
    ndcg10: float
    recall10: float
    skipped_empty: int
    n_definitions: int


@dataclass
class LearnSaturationResult:
    """Closed-loop learning run until convergence."""

    corpus: str
    converged: bool
    reason: str
    iterations: list[LearnIteration] = field(default_factory=list)
    ndcg_before: float = 0.0
    ndcg_after: float = 0.0
    recall_before: float = 0.0
    recall_after: float = 0.0
    total_terms_taught: int = 0


def _ndcg10(ranked: list[str], rels: dict[str, int], k: int = 10) -> float:
    rel = {d: s for d, s in rels.items() if s > 0}
    if not rel:
        return 0.0
    dcg = sum(
        (2 ** rel[d] - 1) / math.log2(i + 2)
        for i, d in enumerate(ranked[:k])
        if d in rel
    )
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def _recall10(ranked: list[str], rels: dict[str, int], k: int = 10) -> float:
    rel = {d for d, s in rels.items() if s > 0}
    if not rel:
        return 0.0
    return len(set(ranked[:k]) & rel) / len(rel)


def score_candidates(idx, query, cand) -> dict[str, float]:
    """BM25(+) restricted to a candidate set — O(query_terms x |cand|).

    Mirrors AppendOnlyLatticeIndex._score exactly, but only accumulates over the
    bounded candidate pool instead of scanning every posting. This is what makes
    the kappa-routed path N-independent: query work depends on |cand|, not corpus
    size. Iterates the candidate set per term (bounded) and probes each posting
    dict in O(1), so a common term's huge list is never traversed.
    """
    if not cand:
        return {}
    N = max(1, len(idx.alive))
    avgdl = idx._total_len / N
    qbag = idx._multiview(query)
    k1, b = idx.k1, idx.b
    A, Bc, k1p1 = k1 * (1 - b), k1 * b / avgdl, k1 + 1
    df, postings, doc_len = idx.df, idx.postings, idx.doc_len
    tri_cap = idx.tri_df_frac * N if idx.tri_df_frac < 1.0 else None
    delta_base = idx.bm25_delta
    scores: dict[str, float] = {}
    cand_list = list(cand)
    for tok, qwt in qbag.items():
        p = idx.token_prime.get(tok)
        if p is None:
            continue
        dfp = df.get(p, 0)
        if dfp == 0:
            continue
        if tri_cap is not None and tok[0] == "3" and dfp > tri_cap:
            continue
        idf = math.log(1 + (N - dfp + 0.5) / (dfp + 0.5))
        delta = delta_base if tok[0] == "w" else 0.0
        pl = postings[p]
        cf = qwt * idf
        for d in cand_list:
            tf = pl.get(d)
            if not tf:
                continue
            norm = tf * k1p1 / (tf + A + Bc * doc_len[d])
            scores[d] = scores.get(d, 0.0) + cf * (norm + delta)
    return scores


class MultiCorpusBrain:
    """
    Shared prime table + promotion registry + isolated corpus lattices.

    Each corpus branch owns postings, a BIT 3 κ inverted index (doc dots in
    3D attractor space), route labels, and a TeachStore. Query routing picks
    the corpus subspace, lights κ buckets, then runs vocab-gap routed search.
    """

    KAPPA_LAM = 0.12
    KAPPA_TOP_K = 10

    def __init__(self, *, prime_pool: int = 200_000):
        self._tp: dict = {}
        self._primes = chain_primes(prime_pool)
        self._corpora: dict[str, CorpusBranch] = {}
        self._rare_doc_cache: dict[str, dict] = {}
        self._registry = PromotionRegistry(fast_ingest=True, defer_l2_promotion=True)

    @property
    def registry(self) -> PromotionRegistry:
        return self._registry

    @property
    def vocab_size(self) -> int:
        return len(self._tp)

    @property
    def corpus_names(self) -> list[str]:
        return list(self._corpora.keys())

    def _alloc_corpus_prime(self, name: str) -> int:
        key = ("C", name)
        if key in self._tp:
            return self._tp[key]
        i = len(self._tp)
        if i >= len(self._primes):
            self._primes = chain_primes(max(i + 1, len(self._primes) * 2))
        p = self._primes[i]
        self._tp[key] = p
        return p

    def _make_slice(self) -> AppendOnlyLatticeIndex:
        return AppendOnlyLatticeIndex(
            token_prime=self._tp,
            _primes=self._primes,
        )

    @staticmethod
    def _labels_from_train(
        queries: dict[str, str],
        train_qrels: dict,
        idf,
        *,
        idf_gate: float = 4.5,
    ) -> set[str]:
        labels: set[str] = set()
        for qid in train_qrels:
            if qid not in queries:
                continue
            for w in words(queries[qid]):
                if idf(w) >= idf_gate:
                    labels.add(w)
        return labels

    def _discriminative_labels(self, branch: CorpusBranch, top_n: int = 36) -> set[str]:
        """Terms disproportionately frequent in this corpus vs all others."""
        if len(self._corpora) < 2:
            return set()
        n_self = max(1, branch.n_docs)
        scored: list[tuple[float, str]] = []
        vocab: set[str] = set()
        for text in branch.texts.values():
            vocab.update(words(text))
        for w in vocab:
            p = branch.idx.token_prime.get(("w", w))
            if not p:
                continue
            df_self = branch.idx.df.get(p, 0) / n_self
            df_other = 0.0
            for other in self._corpora.values():
                if other.name == branch.name:
                    continue
                df_other = max(df_other, other.idx.df.get(p, 0) / max(1, other.n_docs))
            margin = df_self - df_other
            if margin > 0.002 and df_self > 0.005:
                scored.append((margin, w))
        scored.sort(reverse=True)
        return {w for _, w in scored[:top_n]}

    def _build_kappa_index(self, branch: CorpusBranch) -> CorpusAttractorIndex | None:
        if not branch.texts:
            return None
        doc_ids = [branch.global_id(k) for k in branch.texts]
        doc_tokens = {
            branch.global_id(k): frozenset(words(v))
            for k, v in branch.texts.items()
        }
        try:
            sigs = build_all_hub_signatures(
                doc_ids,
                doc_tokens,
                self._registry,
                top_k=self.KAPPA_TOP_K,
                materialize_wings=False,
            )
            return build_attractor_index_from_hub_signatures(self._registry, sigs)
        except Exception:
            return None

    def stack_corpus(
        self,
        name: str,
        corpus: dict[str, str],
        *,
        subjects: frozenset[int] | None = None,
        queries: dict[str, str] | None = None,
        train_qrels: dict | None = None,
        route_labels: set[str] | frozenset[str] | None = None,
        finalize: bool = True,
        build_kappa: bool = True,
    ) -> CorpusBranch:
        """Append a corpus branch (additive — never erases prior corpora)."""
        if name in self._corpora:
            branch = self._corpora[name]
            idx = branch.idx
        else:
            root = self._alloc_corpus_prime(name)
            idx = self._make_slice()
            branch = CorpusBranch(
                name=name,
                root_prime=root,
                idx=idx,
                texts={},
                teach=TeachStore(idx, 0),
                attractor=WordAttractorIndex(),
                subjects=subjects or subjects_for_dataset(name),
            )
            self._corpora[name] = branch
            self._rare_doc_cache[name] = {}

        new_texts: dict[str, str] = {}
        for local_id, text in corpus.items():
            gid = branch.global_id(local_id)
            if gid not in idx.alive:
                idx.add(gid, text)
                self._registry.observe_text(text)
            new_texts[local_id] = text
        branch.texts.update(new_texts)

        n = len(idx.alive)
        if branch.n_docs == 0 or branch.teach.N != n:
            branch.teach = TeachStore(idx, n)

        idf = IdfCache(idx, n)
        global_texts = {branch.global_id(k): v for k, v in branch.texts.items()}

        if queries and train_qrels:
            global_qrels = {
                qid: {
                    branch.global_id(d): s
                    for d, s in rels.items()
                    if d in branch.texts
                }
                for qid, rels in train_qrels.items()
            }
            mode, info = choose_expansion_mode(queries, global_qrels, global_texts)
            branch.expansion_mode = mode
            branch.gap_info = info
            mp = 1 if name == "scifact" else 2
            branch.pair_bridges = RelevanceBridges(
                idx, n, min_pairs=mp,
            ).learn(queries, global_qrels, global_texts)

        labels: set[str] = set(route_labels or ())
        if queries and train_qrels:
            labels |= self._labels_from_train(queries, train_qrels, idf)
        labels |= self._discriminative_labels(branch)
        branch.route_labels = labels

        branch.attractor.build(global_texts, idf)
        if build_kappa:
            branch.kappa_index = self._build_kappa_index(branch)

        if finalize and n > 0:
            idx.finalize()

        return branch

    def mine_gaps(
        self,
        corpus_name: str,
        queries: dict[str, str],
        qrels: dict[str, dict[str, int]],
        *,
        use_search_rank: bool = False,
        **kwargs,
    ) -> GapReport:
        """Audit query terms / pairs / triples missing from corpus structure."""
        branch = self._corpora[corpus_name]
        global_corpus = {branch.global_id(k): v for k, v in branch.texts.items()}
        global_qrels = {
            qid: {
                branch.global_id(d): s
                for d, s in rels.items()
                if d in branch.texts
            }
            for qid, rels in qrels.items()
        }
        rank_fn = None
        if use_search_rank:

            def rank_fn(qid, qtext, golds):
                res = self.search(qtext, corpus=corpus_name, k=1000)
                ranks = [
                    res.ranked.index(d) + 1
                    for d in golds
                    if d in res.ranked
                ]
                return min(ranks) if ranks else None

        return mine_query_gaps(
            branch.idx,
            global_corpus,
            queries,
            global_qrels,
            corpus_name=corpus_name,
            rank_fn=rank_fn,
            **kwargs,
        )

    def evaluate(
        self,
        corpus_name: str,
        queries: dict[str, str],
        qrels: dict[str, dict[str, int]],
        *,
        k: int = 100,
    ) -> tuple[float, float]:
        """Held-out nDCG@10 and Recall@10 (local doc ids)."""
        branch = self._corpora[corpus_name]
        qids = [q for q in qrels if q in queries]
        nd = rc = 0.0
        for qid in qids:
            res = self.search(queries[qid], corpus=corpus_name, k=k)
            rels = {
                branch.global_id(d): s
                for d, s in qrels[qid].items()
                if d in branch.texts
            }
            nd += _ndcg10(res.ranked, rels)
            rc += _recall10(res.ranked, rels)
        n = max(len(qids), 1)
        return nd / n, rc / n

    def learn_until_saturated(
        self,
        corpus_name: str,
        queries: dict[str, str],
        train_qrels: dict[str, dict[str, int]],
        eval_qrels: dict[str, dict[str, int]] | None = None,
        *,
        use_wiki: bool = True,
        self_teach: bool = False,
        max_iterations: int = 10,
        max_absent_wiki: int = 50,
        ndcg_eps: float = 0.0005,
        plateau_iters: int = 2,
        verbose: bool = True,
    ) -> LearnSaturationResult:
        """
        Closed loop: mine gaps (full-stack rank) → teach new terms → re-eval.

        Mines on train_qrels; reports accuracy on eval_qrels (defaults to train
        when no held-out set is supplied). Stops when no new terms are taught
        and nDCG gain is below ndcg_eps for ``plateau_iters`` consecutive steps.
        """
        from aethos_encyclopedia_teacher import teach_incremental_for_corpus

        eval_qrels = eval_qrels or train_qrels
        nd0, rc0 = self.evaluate(corpus_name, queries, eval_qrels)
        out = LearnSaturationResult(
            corpus=corpus_name,
            converged=False,
            reason="max_iterations",
            ndcg_before=nd0,
            recall_before=rc0,
        )
        if verbose:
            print(f"[{corpus_name}] learn_until_saturated: nDCG@10={nd0:.4f} R@10={rc0:.4f}")

        prev_ndcg = nd0
        prev_miss = None
        flat = 0
        total_new = 0

        for step in range(max_iterations):
            report = self.mine_gaps(
                corpus_name, queries, train_qrels,
                use_search_rank=True, miss_only=True,
            )
            bootstrap = step == 0
            if bootstrap:
                from aethos_encyclopedia_teacher import teach_full_knowledge_for_corpus
                teach_res = teach_full_knowledge_for_corpus(
                    self, corpus_name, report,
                    queries=queries, train_qrels=train_qrels,
                    self_teach=self_teach,
                    use_wiki=use_wiki, max_absent_wiki=max_absent_wiki,
                )
            else:
                teach_res = teach_incremental_for_corpus(
                    self, corpus_name, report, queries=queries,
                    train_qrels=train_qrels, self_teach=self_teach,
                    use_wiki=use_wiki, max_absent_wiki=max_absent_wiki,
                )

            nd, rc = self.evaluate(corpus_name, queries, eval_qrels)
            branch = self._corpora[corpus_name]
            n_defs = len(branch.teach.definitions)
            total_new += teach_res.terms_taught

            it = LearnIteration(
                step=step + 1,
                n_miss=report.n_miss,
                terms_new=teach_res.terms_taught,
                ndcg10=nd,
                recall10=rc,
                skipped_empty=teach_res.skipped_empty,
                n_definitions=n_defs,
            )
            out.iterations.append(it)

            if verbose:
                print(
                    f"  iter {step + 1}: miss={report.n_miss} "
                    f"new_terms={teach_res.terms_taught} "
                    f"defs={n_defs} skipped={teach_res.skipped_empty} "
                    f"nDCG={nd:.4f} R@10={rc:.4f} ({nd - prev_ndcg:+.4f})"
                )

            ndcg_gain = nd - prev_ndcg
            miss_same = prev_miss is not None and report.n_miss == prev_miss
            if teach_res.terms_taught == 0 and ndcg_gain < ndcg_eps:
                flat += 1
            else:
                flat = 0

            if flat >= plateau_iters and teach_res.terms_taught == 0:
                out.converged = True
                out.reason = "saturated"
                break

            if teach_res.terms_taught == 0 and miss_same and ndcg_gain < ndcg_eps:
                out.converged = True
                out.reason = "no_new_terms"
                break

            prev_ndcg = nd
            prev_miss = report.n_miss

        out.ndcg_after = out.iterations[-1].ndcg10 if out.iterations else nd0
        out.recall_after = out.iterations[-1].recall10 if out.iterations else rc0
        out.total_terms_taught = total_new

        if verbose:
            print(
                f"  done ({out.reason}): nDCG {nd0:.4f} -> {out.ndcg_after:.4f} "
                f"({out.ndcg_after - nd0:+.4f})  terms={total_new}"
            )
        return out

    def teach_from_encyclopedia(
        self,
        corpus_name: str,
        report: GapReport,
        *,
        queries: dict[str, str] | None = None,
        use_wiki: bool = True,
        max_terms: int = 40,
        max_pairs: int = 24,
        max_triples: int = 16,
    ) -> TeachGapResult:
        """Teach mined gaps using glossary + Wikipedia (correlations only)."""
        return teach_gaps_for_corpus(
            self,
            corpus_name,
            report,
            queries=queries,
            use_wiki=use_wiki,
            max_terms=max_terms,
            max_pairs=max_pairs,
            max_triples=max_triples,
        )

    def teach_gaps_auto(
        self,
        corpus_name: str,
        queries: dict[str, str],
        qrels: dict[str, dict[str, int]],
        *,
        use_wiki: bool = True,
        **kwargs,
    ) -> tuple[GapReport, TeachGapResult]:
        """Mine query gaps and teach from encyclopedia in one call."""
        report = self.mine_gaps(corpus_name, queries, qrels)
        result = self.teach_from_encyclopedia(
            corpus_name, report, queries=queries, use_wiki=use_wiki, **kwargs,
        )
        return report, result

    def teach_full_knowledge(
        self,
        corpus_name: str,
        queries: dict[str, str],
        qrels: dict[str, dict[str, int]],
        *,
        use_wiki: bool = True,
        max_absent_wiki: int = 200,
        **kwargs,
    ) -> tuple[GapReport, TeachGapResult]:
        """Mine gaps then teach full glossary + wiki for absent terms."""
        from aethos_encyclopedia_teacher import teach_full_knowledge_for_corpus

        report = self.mine_gaps(corpus_name, queries, qrels)
        result = teach_full_knowledge_for_corpus(
            self, corpus_name, report,
            use_wiki=use_wiki, max_absent_wiki=max_absent_wiki, **kwargs,
        )
        return report, result

    def teach_glossary(self, corpus_name: str, glossary: dict[str, str]) -> None:
        branch = self._corpora[corpus_name]
        for term, definition in glossary.items():
            branch.teach.teach_bridge(term, definition)
        branch.teach.finalize(top_k=16)

    def _kappa_route_score(self, branch: CorpusBranch, query: str) -> tuple[float, int, int]:
        if branch.kappa_index is None:
            idf = IdfCache(branch.idx, branch.n_docs)
            hits = branch.attractor.route(query, idf)
            return 0.2 * len(hits), len(hits), 0
        qws = query_words_for_routing(words(query))
        if not qws:
            return 0.0, 0, 0
        kdocs, keys = candidates_from_attractors(
            qws, self._registry, branch.kappa_index,
        )
        return 0.3 * len(kdocs) + 1.5 * len(keys), len(kdocs), len(keys)

    def route_corpus(self, query: str) -> tuple[str, dict[str, float]]:
        """Pick corpus: route labels + κ lighting + subjects + lexical probe."""
        if not self._corpora:
            raise RuntimeError("no corpora stacked")
        if len(self._corpora) == 1:
            name = next(iter(self._corpora))
            return name, {name: 1.0}

        qwords = set(words(query))
        chambers = vote_query_chambers(list(qwords))
        scores: dict[str, float] = {}
        for name, branch in self._corpora.items():
            s = 0.0
            label_hits = qwords & branch.route_labels
            if label_hits:
                s += 10.0 * len(label_hits)
            overlap = branch.subjects & chambers
            if overlap:
                s += 6.0 * len(overlap)
            n = max(1, branch.n_docs)
            norm = math.log1p(n)
            for w in qwords:
                p = branch.idx.token_prime.get(("w", w))
                if p and branch.idx.df.get(p, 0):
                    s += branch.idx._idf(p, n) / norm
            kscore, _, _ = self._kappa_route_score(branch, query)
            s += kscore
            scores[name] = s

        best = max(scores, key=lambda k: scores[k])
        if scores[best] <= 0.0:
            best = max(self._corpora, key=lambda k: self._corpora[k].n_docs)
        return best, scores

    def _global_texts(self, branch: CorpusBranch) -> dict[str, str]:
        return {branch.global_id(k): v for k, v in branch.texts.items()}

    def _apply_kappa_fusion(
        self,
        branch: CorpusBranch,
        query: str,
        lex: dict[str, float],
    ) -> tuple[dict[str, float], int, int]:
        if branch.kappa_index is None:
            return lex, 0, 0
        qws = query_words_for_routing(words(query))
        if not qws:
            return lex, 0, 0
        kdocs, keys = candidates_from_attractors(
            qws, self._registry, branch.kappa_index,
        )
        if not keys:
            return lex, len(kdocs), 0
        fused = dict(lex)
        lmax = max(fused.values()) if fused else 1.0
        for doc_id in kdocs[:120]:
            ov = branch.kappa_index.score_doc_overlap(keys, doc_id)
            if ov <= 0:
                continue
            boost = self.KAPPA_LAM * ov * lmax
            fused[doc_id] = fused.get(doc_id, 0.0) + boost
        return fused, len(kdocs), len(keys)

    def search_branch(
        self,
        branch: CorpusBranch,
        query: str,
        *,
        k: int = 10,
        lam: float = 0.3,
    ) -> tuple[list[str], GapSignal | None, int, int]:
        lex = branch.idx._score(query)
        if not lex:
            return [], None, 0, 0
        lex, n_kdocs, n_keys = self._apply_kappa_fusion(branch, query, lex)
        texts = self._global_texts(branch)
        idf = IdfCache(branch.idx, branch.n_docs)
        cache = self._rare_doc_cache.setdefault(branch.name, {})
        ranked, sig = routed_search(
            branch.idx,
            texts,
            idf,
            query,
            lex,
            branch.teach,
            lam=lam,
            k=k,
            rare_doc_cache=cache,
            mode=branch.expansion_mode,
            pair_bridges=branch.pair_bridges,
        )
        return ranked, sig, n_kdocs, n_keys

    def search(
        self,
        query: str,
        *,
        corpus: str | None = None,
        k: int = 10,
        lam: float = 0.3,
    ) -> SearchResult:
        if corpus is not None:
            if corpus not in self._corpora:
                raise KeyError(f"unknown corpus {corpus!r}")
            route_scores = {corpus: 1.0}
            chosen = corpus
        else:
            chosen, route_scores = self.route_corpus(query)

        branch = self._corpora[chosen]
        ranked, sig, n_kdocs, n_keys = self.search_branch(
            branch, query, k=k, lam=lam,
        )
        local = [d.split("/", 1)[1] for d in ranked if "/" in d]
        return SearchResult(
            corpus=chosen,
            ranked=ranked,
            signal=sig,
            route_scores=route_scores,
            local_ids=local,
            kappa_candidates=n_kdocs,
            kappa_keys=n_keys,
        )

    def scale_search(
        self,
        query: str,
        *,
        corpus: str | None = None,
        k: int = 10,
        max_candidates: int = 600,
        rare_df_cap: int = 256,
        use_bridges: bool = True,
        use_teach: bool = True,
    ) -> SearchResult:
        """N-independent search: kappa-route candidates + rare-term exact recall,
        then score only the bounded pool.

        Candidate pool = (geometric kappa buckets of query words)
                       ∪ (full posting lists of DISCRIMINATIVE query terms,
                          df <= rare_df_cap — bounded, exact recall).
        Both parts are bounded independent of corpus size, so query work does not
        grow with N. Bridges/teach rerank on the same pool. This is the formula-
        driven O(1) path (vs search(), which full-scans for exactness).
        """
        chosen = corpus if corpus is not None else self.route_corpus(query)[0]
        if chosen not in self._corpora:
            raise KeyError(f"unknown corpus {chosen!r}")
        branch = self._corpora[chosen]
        idx = branch.idx

        qws = query_words_for_routing(words(query))
        pool: set[str] = set()
        keys = frozenset()
        n_kdocs = 0

        # 1. geometric candidates (kappa buckets) — bounded by max_candidates
        if branch.kappa_index is not None and qws:
            kdocs, keys = candidates_from_attractors(
                qws, self._registry, branch.kappa_index,
            )
            n_kdocs = len(kdocs)
            pool.update(kdocs[:max_candidates])

        # 2. discriminative exact recall — rare query terms have short lists
        for w in set(words(query)):
            p = idx.token_prime.get(("w", w))
            if p is None:
                continue
            dfp = idx.df.get(p, 0)
            if 0 < dfp <= rare_df_cap:
                pl = idx.postings.get(p)
                if pl:
                    pool.update(d for d in pl if d in idx.alive)

        if not pool:
            return SearchResult(chosen, [], None, {chosen: 1.0}, [], n_kdocs, len(keys))

        # 3. score restricted to the bounded pool (N-independent)
        scores = score_candidates(idx, query, pool)

        # 4. kappa overlap fusion on the pool
        if keys and branch.kappa_index is not None and scores:
            lmax = max(scores.values()) or 1.0
            for d in pool:
                ov = branch.kappa_index.score_doc_overlap(keys, d)
                if ov > 0:
                    scores[d] = scores.get(d, 0.0) + self.KAPPA_LAM * ov * lmax

        # 5. supervised-bridge + teach rerank, restricted to the pool
        if use_bridges and branch.pair_bridges is not None and scores:
            from aethos_bridges import bridge_expansion
            exp = bridge_expansion(idx, branch.pair_bridges, query)
            if exp:
                emax = max(exp.values()) or 1.0
                smax = max(scores.values()) or 1.0
                for d, e in exp.items():
                    if d in scores:
                        scores[d] += 0.25 * smax * (e / emax)
        if use_teach and branch.teach is not None and scores:
            rq = branch.teach.rewrite_query(query)
            if rq != query:
                for d, s in score_candidates(idx, rq, pool).items():
                    scores[d] = max(scores.get(d, 0.0), s)

        ranked = sorted(scores, key=scores.get, reverse=True)[:k]
        local = [d.split("/", 1)[1] for d in ranked if "/" in d]
        return SearchResult(
            corpus=chosen,
            ranked=ranked,
            signal=None,
            route_scores={chosen: 1.0},
            local_ids=local,
            kappa_candidates=len(pool),
            kappa_keys=len(keys),
        )

    def stats(self) -> dict:
        return {
            "vocab_primes": self.vocab_size,
            "registry_words": len(self._registry.word_counts),
            "corpora": {
                name: {
                    "root_prime": b.root_prime,
                    "n_docs": b.n_docs,
                    "expansion_mode": b.expansion_mode,
                    "subjects": sorted(b.subjects),
                    "route_labels": len(b.route_labels),
                    "kappa_buckets": (
                        b.kappa_index.summary() if b.kappa_index else {}
                    ),
                    "teach_edges": sum(len(v) for v in b.teach.edges.values()),
                }
                for name, b in self._corpora.items()
            },
        }
