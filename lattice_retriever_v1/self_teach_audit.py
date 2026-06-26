"""
Synthetic Q→doc self-teach audit — zero-shot continual learning loop.

For each corpus doc:
  1. Build a synthetic query from its rarest terms (no external labels).
  2. Run hybrid retrieve; if the source doc misses pool or top-10, teach a bridge
     from the rarest query term → company snippet extracted from that doc.

Bridges live in TeachStore (correlation-only, append-only on existing primes).
"""

from __future__ import annotations

from dataclasses import dataclass

from aethos_append_index import words
from aethos_glass_box_search import rarest_terms, word_idf
from aethos_miss_r1_teach import gold_company_snippet, _idf_fn
from aethos_teach_store import TeachStore

from lattice_retriever_v1.hybrid_retriever import HybridZeroShotRetriever


def synthetic_query_from_doc(
    text: str,
    idx,
    N: int,
    *,
    n_terms: int = 4,
    idf_gate: float = 2.0,
) -> str:
    """Rare-term query synthesized from one document."""
    idf = _idf_fn(idx, N)
    uniq = list(dict.fromkeys(words(text)))
    ranked = sorted(
        (w for w in uniq if idf(w) >= idf_gate),
        key=lambda w: (-idf(w), w),
    )
    if not ranked:
        ranked = sorted(uniq, key=lambda w: (-idf(w), w))[:n_terms]
    return " ".join(ranked[:n_terms])


@dataclass
class SelfTeachAuditStats:
    docs_audited: int = 0
    pool_miss: int = 0
    rank_miss: int = 0
    bridges_taught: int = 0
    terms_touched: int = 0

    def explain(self) -> dict:
        return {
            "docs_audited": self.docs_audited,
            "pool_miss": self.pool_miss,
            "rank_miss": self.rank_miss,
            "bridges_taught": self.bridges_taught,
            "terms_touched": self.terms_touched,
        }


def audit_and_self_teach(
    retriever: HybridZeroShotRetriever,
    corpus: dict[str, str],
    *,
    max_docs: int | None = None,
    rare_gate: float = 2.5,
    top_k: int = 16,
    progress_every: int = 500,
) -> tuple[TeachStore, SelfTeachAuditStats]:
    """
    Offline audit loop — teach correlation bridges for synthetic probe failures.

    Does not add documents to the index; only TeachStore edges on existing primes.
    """
    idx = retriever.append_idx
    N = len(idx.alive)
    teach = TeachStore(idx, N)
    idf = _idf_fn(idx, N)
    stats = SelfTeachAuditStats()
    items = list(corpus.items())
    if max_docs is not None:
        items = items[:max_docs]
    total = len(items)

    for i, (doc_id, text) in enumerate(items, start=1):
        stats.docs_audited += 1
        query = synthetic_query_from_doc(text, idx, N)
        if not query.strip():
            continue

        trace = retriever.retrieve_with_trace(query, limit=10)
        in_pool = doc_id in trace.pool_docs
        in_top = any(h.doc_id == doc_id for h in trace.hits)
        if in_pool and in_top:
            if progress_every and i % progress_every == 0:
                print(
                    f"  self-teach {i}/{total} | taught {stats.bridges_taught} | "
                    f"pool_miss {stats.pool_miss} rank_miss {stats.rank_miss}",
                    flush=True,
                )
            continue

        if not in_pool:
            stats.pool_miss += 1
        else:
            stats.rank_miss += 1

        qterms = words(query)
        if not qterms:
            continue
        rare = rarest_terms(qterms, idx, N)
        r1 = rare[0]
        r2 = rare[1] if len(rare) > 1 else r1
        snippet = gold_company_snippet(text, r1, r2, idf, rare_gate=rare_gate)
        n = teach.teach_edges(r1, snippet, bridge_gate=rare_gate)
        if n:
            stats.bridges_taught += 1
            stats.terms_touched += 1

        if progress_every and i % progress_every == 0:
            print(
                f"  self-teach {i}/{total} | taught {stats.bridges_taught} | "
                f"pool_miss {stats.pool_miss} rank_miss {stats.rank_miss}",
                flush=True,
            )

    teach.finalize(top_k=top_k)
    return teach, stats
