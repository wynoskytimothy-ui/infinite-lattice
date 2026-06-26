"""
Corpus-scoped lattice — global 3-way meets across documents.

Skeleton builder: corpus spine prime + per-doc rare triple witnesses.
Retrieval routing via ``global_3way`` is optional (``enable_corpus_lattice``).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations

from lattice_retriever_v1.doc_lattice_codec import (
    DocPrimePool,
    encode_doc,
    select_rare_in_doc,
)
from lattice_retriever_v1.k_meet import velocity_meet
from lattice_retriever_v1.stage04_promote import Stage04Registry
from lattice_retriever_v1.stage06_composites import meet_composite_k
from lattice_retriever_v1.stage07_semantic_light import SemanticLightIndex


@dataclass(frozen=True)
class DocEntry:
    doc_id: str
    doc_prime: int
    rare_terms: tuple[str, ...]


@dataclass
class TripleMeetRecord:
    doc_ids: set[str] = field(default_factory=set)
    correlated_terms: set[str] = field(default_factory=set)


@dataclass
class CorpusLattice:
    corpus_prime: int
    doc_registry: dict[str, DocEntry]
    global_3way: dict[int, TripleMeetRecord]
    term_to_docs: dict[str, frozenset[str]] = field(default_factory=dict)
    term_to_meet_keys: dict[str, frozenset[int]] = field(default_factory=dict)
    pair_meet_postings: dict[int, frozenset[str]] = field(default_factory=dict)

    def lookup_pair_meet(self, pa: int, pb: int) -> frozenset[str]:
        """Docs indexed under corpus-spine pair meet key for two term primes."""
        chain = tuple(sorted({self.corpus_prime, pa, pb}))
        if len(chain) < 3:
            return frozenset()
        vel = velocity_meet(*chain)
        if vel is None or not vel.unified:
            return frozenset()
        try:
            key = meet_composite_k(*chain)
        except ValueError:
            return frozenset()
        return self.pair_meet_postings.get(key, frozenset())

    def lookup_pair_meet_terms(self, a: str, b: str) -> frozenset[str]:
        """Cross-doc global_3way docs for meet witnesses touching either query term."""
        al, bl = a.lower(), b.lower()
        keys = self.term_to_meet_keys.get(al, frozenset()) | self.term_to_meet_keys.get(
            bl, frozenset()
        )
        out: set[str] = set()
        for mk in keys:
            rec = self.global_3way.get(mk)
            if rec is not None:
                out |= rec.doc_ids
        return frozenset(out)

    def route_pool(
        self,
        query_terms: list[str],
        *,
        semantic: SemanticLightIndex,
    ) -> tuple[set[str], list[dict]]:
        """Route via rare-term postings + indexed global_3way meets (no mod heuristic)."""
        pool: set[str] = set()
        steps: list[dict] = []
        for term in query_terms:
            tl = term.lower()
            if not semantic.is_rare(term):
                continue
            term_docs = set(self.term_to_docs.get(tl, ()))
            meet_docs: set[str] = set()
            for meet_key in self.term_to_meet_keys.get(tl, ()):
                rec = self.global_3way.get(meet_key)
                if rec is not None:
                    meet_docs |= rec.doc_ids
            added = (term_docs | meet_docs) - pool
            pool |= term_docs | meet_docs
            if added:
                steps.append(
                    {
                        "step": "corpus_lattice_term",
                        "term": tl,
                        "term_postings": len(term_docs),
                        "meet_postings": len(meet_docs),
                        "added_docs": len(added),
                        "pool_size": len(pool),
                    }
                )
        return pool, steps


class CorpusLatticeBuilder:
    """Ingest docs under a corpus spine; accumulate cross-doc 3-way meets."""

    def __init__(
        self,
        corpus_prime: int,
        registry: Stage04Registry,
        semantic: SemanticLightIndex,
        pool: DocPrimePool,
        *,
        k_rare: int = 8,
        max_df_frac: float = 0.05,
    ) -> None:
        self.corpus_prime = corpus_prime
        self.registry = registry
        self.semantic = semantic
        self.pool = pool
        self.k_rare = k_rare
        self.max_df_frac = max_df_frac
        self._doc_registry: dict[str, DocEntry] = {}
        self._global_3way: dict[int, TripleMeetRecord] = {}
        self._term_to_docs: dict[str, set[str]] = defaultdict(set)
        self._term_to_meet_keys: dict[str, set[int]] = defaultdict(set)
        self._pair_meet_postings: dict[int, set[str]] = defaultdict(set)

    def _index_pair_meet(self, doc_id: str, pa: int, pb: int) -> None:
        chain = tuple(sorted({self.corpus_prime, pa, pb}))
        if len(chain) < 3:
            return
        vel = velocity_meet(*chain)
        if vel is None or not vel.unified:
            return
        try:
            key = meet_composite_k(*chain)
        except ValueError:
            return
        self._pair_meet_postings[key].add(doc_id)

    def observe_doc(self, doc_id: str, text: str) -> DocEntry:
        placement = encode_doc(
            doc_id,
            text,
            self.registry,
            self.pool,
            semantic=self.semantic,
            corpus_prime=self.corpus_prime,
        )
        rare = select_rare_in_doc(
            placement.words,
            self.semantic,
            k=self.k_rare,
            max_df_frac=self.max_df_frac,
        )
        entry = DocEntry(
            doc_id=doc_id,
            doc_prime=placement.doc_prime,
            rare_terms=rare,
        )
        self._doc_registry[doc_id] = entry
        for t in rare:
            self._term_to_docs[t].add(doc_id)
        primes = [self.semantic._prime_for_term(t) for t in rare]
        for pa, pb in combinations(primes, 2):
            self._index_pair_meet(doc_id, pa, pb)
        return entry

    def finalize(self) -> CorpusLattice:
        for term, docset in self._term_to_docs.items():
            if len(docset) < 2:
                continue
            tp = self.semantic._prime_for_term(term)
            for d1, d2 in combinations(sorted(docset), 2):
                raw_d1 = self.pool.doc_id_to_prime.get(d1)
                if raw_d1 is None:
                    continue
                chain = tuple(sorted({self.corpus_prime, tp, raw_d1}))
                if len(chain) < 3:
                    continue
                vel = velocity_meet(*chain)
                if vel is None or not vel.unified:
                    continue
                try:
                    meet_key = meet_composite_k(*chain)
                except ValueError:
                    continue
                rec = self._global_3way.setdefault(meet_key, TripleMeetRecord())
                rec.doc_ids.update([d1, d2])
                rec.correlated_terms.add(term)
                self._term_to_meet_keys[term].add(meet_key)
        term_to_docs = {t: frozenset(ds) for t, ds in self._term_to_docs.items()}
        term_to_meet_keys = {
            t: frozenset(keys) for t, keys in self._term_to_meet_keys.items()
        }
        pair_meet_postings = {
            k: frozenset(ds) for k, ds in self._pair_meet_postings.items()
        }
        return CorpusLattice(
            corpus_prime=self.corpus_prime,
            doc_registry=dict(self._doc_registry),
            global_3way=dict(self._global_3way),
            term_to_docs=term_to_docs,
            term_to_meet_keys=term_to_meet_keys,
            pair_meet_postings=pair_meet_postings,
        )
