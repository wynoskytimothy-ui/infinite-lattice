"""
Rare-shell lattice index — ingest sidecar stacked on SemanticLight df.

Per doc: rare-term subset → correlation shells only.
Global invert: anchor_composite → doc postings + optional neighbor co-doc counts.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable

from lattice_retriever_v1.lattice2_correlation import TermCorrelationShell
from lattice_retriever_v1.stage07_semantic_light import SemanticLightIndex


@dataclass
class RareShellLatticeIndex:
    """Global rare-shell invert — Phase A anchor postings + per-doc shells."""

    anchor_postings: dict[int, set[str]] = field(default_factory=dict)
    doc_shells: dict[str, tuple[TermCorrelationShell, ...]] = field(default_factory=dict)
    neighbor_global: dict[int, dict[str, int]] = field(default_factory=dict)
    term_anchors: dict[str, set[int]] = field(default_factory=lambda: defaultdict(set))
    n_docs: int = 0

    def observe_doc(
        self,
        doc_id: str,
        shells: tuple[TermCorrelationShell, ...],
    ) -> None:
        self.n_docs += 1
        self.doc_shells[doc_id] = shells
        for shell in shells:
            comp = shell.anchor_composite
            self.anchor_postings.setdefault(comp, set()).add(doc_id)
            ng = self.neighbor_global.setdefault(comp, {})
            if shell.key_kind == "term":
                self.term_anchors[shell.key].add(comp)
            elif shell.key_kind == "anchor":
                for part in shell.key.split("|"):
                    self.term_anchors[part].add(comp)
            for term, neighbor in shell.neighbors.items():
                self.term_anchors[term].add(comp)
                ng[term] = ng.get(term, 0) + 1

    def route_pool(
        self,
        query_terms: list[str],
        *,
        semantic: SemanticLightIndex,
        term_prime_fn: Callable[[str], int] | None = None,
    ) -> tuple[set[str], list[dict]]:
        """Phase A — rarest query term first, intersect anchor postings."""
        _ = term_prime_fn  # reserved for parity with SharedTermIndex signature
        if not query_terms:
            all_docs: set[str] = set()
            for bucket in self.anchor_postings.values():
                all_docs |= bucket
            return all_docs, [{"step": "empty_query", "pool_size": len(all_docs)}]

        ordered = sorted(
            query_terms,
            key=lambda t: (semantic.doc_freq.get(t.lower(), 10**9), t),
        )
        steps: list[dict] = []
        pool: set[str] | None = None
        for term in ordered:
            anchors = self.term_anchors.get(term.lower(), set())
            bucket: set[str] = set()
            for comp in anchors:
                bucket |= self.anchor_postings.get(comp, set())
            steps.append(
                {
                    "step": "rare_anchor_light",
                    "term": term,
                    "anchors": len(anchors),
                    "doc_freq": semantic.doc_freq.get(term.lower(), 0),
                    "pool_size": len(bucket if pool is None else (pool & bucket)),
                }
            )
            pool = bucket if pool is None else pool & bucket
            if not pool:
                break

        if not pool and ordered:
            rarest = ordered[0]
            anchors = self.term_anchors.get(rarest.lower(), set())
            pool = set()
            for comp in anchors:
                pool |= self.anchor_postings.get(comp, set())
            steps.append(
                {
                    "step": "widen_rarest_anchor",
                    "term": rarest,
                    "anchors": len(anchors),
                    "pool_size": len(pool),
                }
            )

        return pool or set(), steps

    def route_pool_widen(
        self,
        query_terms: list[str],
        *,
        semantic: SemanticLightIndex,
        term_prime_fn: Callable[[str], int] | None = None,
    ) -> tuple[set[str], list[dict]]:
        """Phase A widen — union anchor postings for rarest query term only (no intersect)."""
        _ = term_prime_fn
        if not query_terms:
            all_docs: set[str] = set()
            for bucket in self.anchor_postings.values():
                all_docs |= bucket
            return all_docs, [{"step": "empty_query", "pool_size": len(all_docs)}]

        ordered = sorted(
            query_terms,
            key=lambda t: (semantic.doc_freq.get(t.lower(), 10**9), t),
        )
        rarest = ordered[0]
        anchors = self.term_anchors.get(rarest.lower(), set())
        pool: set[str] = set()
        for comp in anchors:
            pool |= self.anchor_postings.get(comp, set())
        steps = [
            {
                "step": "rare_anchor_widen",
                "term": rarest,
                "anchors": len(anchors),
                "doc_freq": semantic.doc_freq.get(rarest.lower(), 0),
                "pool_size": len(pool),
            }
        ]
        return pool, steps

    def explain(self) -> dict:
        return {
            "n_docs": self.n_docs,
            "n_anchors": len(self.anchor_postings),
            "n_term_anchors": len(self.term_anchors),
            "anchor_posting_sizes": {
                str(comp): len(docs)
                for comp, docs in sorted(
                    self.anchor_postings.items(), key=lambda x: -len(x[1])
                )[:12]
            },
        }
