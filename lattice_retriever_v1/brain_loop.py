"""
AETHOS brain loop — minimal Stage 08 retrieve + self-teach wrapper.

Wraps LatticeRetriever + SemanticLightIndex + Stage04Registry without
MultiCorpusBrain, BM25, or embedding backends.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lattice_retriever_v1.corpus_prime import tag_doc_id
from lattice_retriever_v1.stage04_promote import Stage04Registry
from lattice_retriever_v1.stage07_semantic_light import SemanticLightIndex
from lattice_retriever_v1.neuron_room import open_room, seed_from_primes
from lattice_retriever_v1.stage08_retrieve import (
    LatticeRetriever,
    RetrieveHit,
    RetrieveTrace,
)


@dataclass
class BrainLoop:
    """Retrieve → explain → teach-from-miss on the Stage 08 stack."""

    retriever: LatticeRetriever = field(default_factory=LatticeRetriever)
    enable_neuron_room: bool = True
    _last_trace: RetrieveTrace | None = field(default=None, init=False, repr=False)
    _neuron_room_trace: dict[str, Any] | None = field(default=None, init=False, repr=False)
    _electron_ingest_profiles: dict[str, dict[str, Any]] = field(
        default_factory=dict, init=False, repr=False
    )

    @property
    def semantic(self) -> SemanticLightIndex:
        return self.retriever.semantic

    @property
    def registry(self) -> Stage04Registry:
        return self.retriever.semantic.registry

    def index_corpus(
        self,
        corpus: dict[str, str],
        *,
        corpus_name: str | None = None,
        byte_corpus: dict[str, bytes] | None = None,
    ) -> None:
        if corpus:
            if corpus_name is None:
                self.retriever.index_corpus(corpus)
            else:
                tagged = {
                    tag_doc_id(corpus_name, doc_id): text for doc_id, text in corpus.items()
                }
                self.retriever.index_corpus(tagged)
        if byte_corpus:
            from lattice_retriever_v1.electron_ingest import electron_ingest_profile

            self._electron_ingest_profiles = {
                doc_id: electron_ingest_profile(data) for doc_id, data in byte_corpus.items()
            }

    def electron_ingest_profiles(self) -> dict[str, dict[str, Any]]:
        """Profiles from the most recent byte_corpus passed to index_corpus."""
        return dict(self._electron_ingest_profiles)

    def retrieve(self, query: str, *, limit: int = 10) -> list[RetrieveHit]:
        """Run retrieve and store the glass-box trace for explain_last()."""
        self._last_trace = self.retriever.retrieve_with_trace(query, limit=limit)
        self._neuron_room_trace = (
            self._open_query_neuron_room(query) if self.enable_neuron_room else None
        )
        return list(self._last_trace.hits)

    def _open_query_neuron_room(self, query: str) -> dict[str, Any] | None:
        """Open neuron room for query's rarest pin chain (glass-box)."""
        terms = [
            t
            for t in self.retriever._query_terms(query)
            if self.retriever.semantic.is_rare(t)
        ]
        if len(terms) < 2:
            return None
        terms.sort(key=lambda t: (self.retriever.semantic.doc_freq.get(t, 0), t))
        primes = tuple(self.retriever._identity_for(t) for t in terms[:4])
        if len(primes) < 2:
            return None
        chain = primes[:3] if len(primes) >= 3 else primes
        seed = seed_from_primes(*chain)
        room = open_room(seed)
        l01 = room.wings[0].coord if room.wings else None
        from lattice_retriever_v1.origin_corridor import corridor_key_with_origin

        origin_corridor = corridor_key_with_origin(
            chain,
            seed.transgressor_n,
            quadrant=seed.quadrant,
            invoke_order=tuple(seed.invoke_order),
        )
        return {
            "status": room.status.value,
            "n_wings": len(room.wings) if room.wings else 0,
            "L01_coord": list(l01) if l01 else None,
            "seed_primes": list(seed.primes),
            "k": seed.k,
            "origin_corridor": origin_corridor,
        }

    def explain_last(self) -> dict[str, Any]:
        """Glass-box dict from the most recent retrieve trace."""
        if self._last_trace is None:
            return {"error": "no retrieve yet"}
        out = self._last_trace.explain()
        if self._neuron_room_trace is not None:
            out["neuron_room"] = self._neuron_room_trace
        return out

    def teach_from_miss(
        self,
        query: str,
        gold_doc_id: str,
        bridge_terms: tuple[str, ...] | list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Teach a bridge after a miss: append query terms + bridge terms to the
        gold doc and strengthen wing-cage correlations for re-retrieval.
        """
        gold = self.retriever.docs.get(gold_doc_id)
        if gold is None:
            return {"taught": False, "reason": "unknown gold doc", "gold_doc_id": gold_doc_id}

        qterms = self.retriever._query_terms(query)
        if bridge_terms is None:
            bridge_terms = tuple(
                t for t in gold.words if self.retriever.semantic.is_rare(t)
            )[:4]
        else:
            bridge_terms = tuple(bridge_terms)

        supplement = " ".join(dict.fromkeys([*qterms, *bridge_terms]))
        new_text = f"{gold.text} {supplement}".strip()
        self._reindex_doc(gold_doc_id, new_text)
        self._teach_correlations(qterms, bridge_terms)

        return {
            "taught": True,
            "gold_doc_id": gold_doc_id,
            "query": query,
            "bridge_terms": list(bridge_terms),
            "supplement": supplement,
        }

    def _reindex_doc(self, doc_id: str, text: str) -> None:
        """Remove old postings for doc_id and index fresh text."""
        old = self.retriever.docs.pop(doc_id, None)
        if old is not None:
            for pin in old.corridor_pins:
                bucket = self.retriever.postings.get(pin)
                if bucket is not None:
                    bucket.discard(doc_id)
                prev = self.retriever.pin_doc_freq.get(pin, 0)
                if prev <= 1:
                    self.retriever.pin_doc_freq.pop(pin, None)
                else:
                    self.retriever.pin_doc_freq[pin] = prev - 1
            for w in old.words:
                for lp in set(self.retriever._letter_primes(w)):
                    lp_bucket = self.retriever.letter_postings.get(lp)
                    if lp_bucket is not None:
                        lp_bucket.discard(doc_id)
        self.retriever.index_doc(doc_id, text)

    def _teach_correlations(
        self,
        query_terms: list[str],
        bridge_terms: tuple[str, ...],
    ) -> None:
        """Add cage correlations linking query terms to bridge terms."""
        window = list(dict.fromkeys([*query_terms, *bridge_terms]))[:6]
        if len(window) < 3:
            return
        for i in range(len(window) - 2):
            triple = window[i : i + 3]
            cage = self.retriever.semantic._cage_for_triple(*triple)
            anchor_p = cage.anchor_primes[0] if cage.anchor_primes else 3
            for t in bridge_terms:
                p = self.retriever.semantic._prime_for_term(t)
                cage.add_correlation(t, p, source_prime=anchor_p, strength=2)
