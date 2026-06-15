"""
Query-vs-corpus gap miner — find missing words, subwords, pairs, triples, compounds.

Scans queries (especially gold@10 misses) for signal that never landed in the
corpus: rare terms, morph pieces, 2-way and 3-way bundles with no co-occurrence
in any document. These are the targets for encyclopedia / glossary teaching.
"""

from __future__ import annotations

import itertools
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from aethos_append_index import AppendOnlyLatticeIndex, words
from aethos_promotion import _chunk_subwords


@dataclass
class GapReport:
    """Glass-box audit of vocabulary / correlation holes."""

    corpus_name: str
    n_queries: int
    n_miss: int
    miss_qids: list[str] = field(default_factory=list)
    missing_words: Counter = field(default_factory=Counter)
    missing_subwords: Counter = field(default_factory=Counter)
    missing_pairs: Counter = field(default_factory=Counter)
    missing_triples: Counter = field(default_factory=Counter)
    absent_from_corpus: Counter = field(default_factory=Counter)

    _SKIP_PRIORITY = frozenset({
        "does", "pro", "biological", "terminal", "using", "used", "based",
        "show", "shows", "associated", "related", "study", "studies",
    })

    @property
    def priority_terms(self) -> list[str]:
        """Ranked teach list: absent-from-corpus first, then miss-weighted rare terms."""
        merged = Counter()
        for w, c in self.absent_from_corpus.items():
            if len(w) >= 4 and w not in self._SKIP_PRIORITY:
                merged[w] += c * 10
        for w, c in self.missing_words.items():
            if len(w) >= 5 and w not in self._SKIP_PRIORITY:
                merged[w] += c * 2
        return [w for w, _ in merged.most_common()]

    def glossary_targets(self, glossary: dict[str, str], queries: dict[str, str]) -> list[str]:
        """Glossary terms appearing in miss queries (highest-value teach set)."""
        out: list[str] = []
        seen: set[str] = set()
        for qid in self.miss_qids:
            if qid not in queries:
                continue
            for w in words(queries[qid]):
                if w in glossary and w not in seen:
                    seen.add(w)
                    out.append(w)
        return out

    def summary(self) -> dict:
        return {
            "n_queries": self.n_queries,
            "n_miss": self.n_miss,
            "missing_words": len(self.missing_words),
            "missing_subwords": len(self.missing_subwords),
            "missing_pairs": len(self.missing_pairs),
            "missing_triples": len(self.missing_triples),
            "absent_from_corpus": len(self.absent_from_corpus),
            "priority_top12": self.priority_terms[:12],
        }


def _corpus_vocab(corpus: dict[str, str]) -> set[str]:
    vocab: set[str] = set()
    for text in corpus.values():
        vocab.update(words(text))
    return vocab


def _doc_term_sets(corpus: dict[str, str]) -> dict[str, set[str]]:
    return {d: set(words(t)) for d, t in corpus.items()}


def _cooccur_in_corpus(a: str, b: str, doc_terms: dict[str, set[str]]) -> bool:
    return any(a in ts and b in ts for ts in doc_terms.values())


def _triple_in_corpus(a: str, b: str, c: str, doc_terms: dict[str, set[str]]) -> bool:
    return any(a in ts and b in ts and c in ts for ts in doc_terms.values())


def _gold_rank(idx, query: str, golds: set[str]) -> int | None:
    lex = idx._score(query)
    ranked = sorted(lex, key=lex.get, reverse=True)
    for i, d in enumerate(ranked):
        if d in golds:
            return i + 1
    return None


def mine_query_gaps(
    idx: AppendOnlyLatticeIndex,
    corpus: dict[str, str],
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    *,
    corpus_name: str = "",
    idf_gate: float = 3.5,
    rare_gate: float = 3.0,
    miss_only: bool = True,
    min_subword_len: int = 3,
    rank_fn=None,
) -> GapReport:
    """
    Find query signal missing from corpus co-occurrence structure.

    miss_only: if True, weight gaps from queries whose gold is outside top-10.
    rank_fn: optional (qid, query, golds) -> rank|None; defaults to lexical _score.
    """
    n = max(1, len(idx.alive))
    doc_terms = _doc_term_sets(corpus)
    cvocab = _corpus_vocab(corpus)

    def idf(w: str) -> float:
        p = idx.token_prime.get(("w", w))
        return idx._idf(p, n) if p else 99.0

    def in_corpus(w: str) -> bool:
        p = idx.token_prime.get(("w", w))
        return p is not None and idx.df.get(p, 0) > 0

    report = GapReport(corpus_name=corpus_name, n_queries=0, n_miss=0)

    qids = [q for q in qrels if q in queries]
    report.n_queries = len(qids)

    for qid in qids:
        qtext = queries[qid]
        qterms = list(dict.fromkeys(words(qtext)))
        golds = {d for d, s in qrels[qid].items() if s > 0 and d in corpus}
        if not golds:
            continue

        if rank_fn is not None:
            rank = rank_fn(qid, qtext, golds)
        else:
            rank = _gold_rank(idx, qtext, golds)
        is_miss = rank is None or rank > 10
        if is_miss:
            report.n_miss += 1
            report.miss_qids.append(qid)
        if miss_only and not is_miss:
            continue

        rare = [w for w in qterms if idf(w) >= idf_gate]
        weight = 2 if is_miss else 1

        for w in qterms:
            if not in_corpus(w):
                report.absent_from_corpus[w] += weight
            elif idf(w) >= rare_gate:
                report.missing_words[w] += weight

        for w in qterms:
            for piece in _chunk_subwords(w):
                if len(piece) < min_subword_len:
                    continue
                if piece not in cvocab and piece != w:
                    report.missing_subwords[piece] += weight

        rare_sorted = sorted(set(rare), key=idf, reverse=True)[:8]
        for a, b in itertools.combinations(rare_sorted, 2):
            if not _cooccur_in_corpus(a, b, doc_terms):
                report.missing_pairs[(a, b)] += weight

        for a, b, c in itertools.combinations(rare_sorted, 3):
            if not _triple_in_corpus(a, b, c, doc_terms):
                report.missing_triples[(a, b, c)] += weight

    return report


def format_pair(key: tuple[str, ...]) -> str:
    return " + ".join(key)
