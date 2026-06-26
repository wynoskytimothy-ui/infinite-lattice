"""
Per-doc promotion ingest profiler — localize O(corpus) eager work.

Flat per-doc time = lazy as designed (Stage 04/05 spec).
Rising per-doc time = eager re-walk bug blocking promotion-on bench.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable

from aethos_promotion import LatticeTier, PromotionRegistry
from lattice_retriever_v1.stage04_promote import Stage04Registry
from lattice_retriever_v1.stage07_semantic_light import SemanticLightIndex
from lattice_retriever_v1.stage08_retrieve import FIXTURE_CORPUS, DocRecord, LatticeRetriever


# Last-quartile / first-quartile median ratio above this → likely eager scaling.
DEFAULT_MAX_QUARTILE_RATIO = 4.0


@dataclass(frozen=True)
class DocTiming:
    doc_id: str
    promotion_ms: float
    flush_ms: float
    flush_candidates: int
    cooccur_ms: float
    semantic_ms: float
    postings_ms: float
    total_ms: float
    n_words: int
    n_l2_promotions: int


@dataclass
class IngestProfile:
    timings: list[DocTiming] = field(default_factory=list)
    promotion_on: bool = True

    @property
    def totals_ms(self) -> list[float]:
        return [t.total_ms for t in self.timings]

    def explain(self) -> dict:
        flat = analyze_flatness(self.totals_ms)
        promo = [t.promotion_ms for t in self.timings]
        flush = [t.flush_ms for t in self.timings]
        return {
            "n_docs": len(self.timings),
            "promotion_on": self.promotion_on,
            "flatness": flat,
            "phase_medians_ms": {
                "promotion_observe_text": _median(promo),
                "l2_flush": _median(flush),
                "semantic_observe": _median([t.semantic_ms for t in self.timings]),
                "postings_index": _median([t.postings_ms for t in self.timings]),
                "total": _median(self.totals_ms),
            },
            "final_l2_promotions": self.timings[-1].n_l2_promotions if self.timings else 0,
            "per_doc": [
                {
                    "doc_id": t.doc_id,
                    "total_ms": round(t.total_ms, 3),
                    "promotion_ms": round(t.promotion_ms, 3),
                    "flush_ms": round(t.flush_ms, 3),
                    "flush_candidates": t.flush_candidates,
                    "n_words": t.n_words,
                }
                for t in self.timings
            ],
        }


def _median(xs: list[float]) -> float:
    return statistics.median(xs) if xs else 0.0


def analyze_flatness(timings_ms: list[float], *, max_ratio: float = DEFAULT_MAX_QUARTILE_RATIO) -> dict:
    """Compare first vs last quartile medians — rising ratio signals corpus-scaling cost."""
    n = len(timings_ms)
    if n < 4:
        med = _median(timings_ms)
        return {
            "n": n,
            "median_ms": med,
            "median_first_quartile_ms": med,
            "median_last_quartile_ms": med,
            "quartile_ratio": 1.0,
            "flat": True,
            "max_ratio": max_ratio,
        }
    q = max(1, n // 4)
    first = timings_ms[:q]
    last = timings_ms[-q:]
    med_first = _median(first)
    med_last = _median(last)
    ratio = med_last / med_first if med_first > 1e-9 else (float("inf") if med_last > 0 else 1.0)
    return {
        "n": n,
        "median_ms": _median(timings_ms),
        "median_first_quartile_ms": round(med_first, 4),
        "median_last_quartile_ms": round(med_last, 4),
        "quartile_ratio": round(ratio, 4),
        "flat": ratio <= max_ratio,
        "max_ratio": max_ratio,
    }


def _instrument_promotion_registry(registry: PromotionRegistry) -> dict[str, list]:
    """Hook flush + cooccurrence for phase breakdown (profiler only)."""
    stats: dict[str, list] = {"flush_ms": [], "flush_candidates": [], "cooccur_ms": []}
    orig_flush = registry._flush_l2_candidates
    orig_cooc = registry.observe_cooccurrence

    def wrapped_flush() -> None:
        t0 = time.perf_counter()
        n = len(registry._l2_candidates)
        orig_flush()
        stats["flush_ms"].append((time.perf_counter() - t0) * 1000.0)
        stats["flush_candidates"].append(n)

    def wrapped_cooc(words: Iterable[str]) -> None:
        t0 = time.perf_counter()
        orig_cooc(words)
        stats["cooccur_ms"].append((time.perf_counter() - t0) * 1000.0)

    registry._flush_l2_candidates = wrapped_flush  # type: ignore[method-assign]
    registry.observe_cooccurrence = wrapped_cooc  # type: ignore[method-assign]
    return stats


def make_promotion_on_stack(*, fast_ingest: bool = False) -> tuple[Stage04Registry, SemanticLightIndex, LatticeRetriever, dict]:
    inner = PromotionRegistry(fast_ingest=fast_ingest, defer_l2_promotion=True)
    hooks = _instrument_promotion_registry(inner)
    stage = Stage04Registry(registry=inner)
    semantic = SemanticLightIndex(registry=stage)
    retriever = LatticeRetriever(semantic=semantic)
    return stage, semantic, retriever, hooks


def fixture_corpus_slice(n_docs: int) -> dict[str, str]:
    """Cycle synthetic FIXTURE_CORPUS to n_docs — deterministic, no BEIR."""
    texts = list(FIXTURE_CORPUS.values())
    out: dict[str, str] = {}
    for i in range(n_docs):
        out[f"f{i:03d}"] = texts[i % len(texts)]
    return out


def profile_ingest(
    corpus: dict[str, str],
    *,
    fast_ingest: bool = False,
    on_doc: Callable[[int, DocTiming], None] | None = None,
) -> IngestProfile:
    """
    Mirror bench_lattice_retriever_v1 ingest path with per-doc timing.

    Phases match production bench:
      stage.observe_text → postings → semantic.observe_doc
    """
    stage, semantic, retriever, hooks = make_promotion_on_stack(fast_ingest=fast_ingest)
    profile = IngestProfile(promotion_on=not fast_ingest)

    for i, (doc_id, text) in enumerate(corpus.items()):
        t0 = time.perf_counter()

        t_promo0 = time.perf_counter()
        stage.observe_text(text)
        promotion_ms = (time.perf_counter() - t_promo0) * 1000.0

        t_post0 = time.perf_counter()
        retriever.index_doc(doc_id, text)
        postings_ms = (time.perf_counter() - t_post0) * 1000.0
        words = retriever.docs[doc_id].words

        t_sem0 = time.perf_counter()
        semantic.observe_doc(text)
        semantic_ms = (time.perf_counter() - t_sem0) * 1000.0

        total_ms = (time.perf_counter() - t0) * 1000.0
        n_l2 = sum(1 for k in stage.registry.promoted if k[0] == LatticeTier.L2_SUBWORD)

        timing = DocTiming(
            doc_id=doc_id,
            promotion_ms=promotion_ms,
            flush_ms=hooks["flush_ms"][-1] if hooks["flush_ms"] else 0.0,
            flush_candidates=hooks["flush_candidates"][-1] if hooks["flush_candidates"] else 0,
            cooccur_ms=hooks["cooccur_ms"][-1] if hooks["cooccur_ms"] else 0.0,
            semantic_ms=semantic_ms,
            postings_ms=postings_ms,
            total_ms=total_ms,
            n_words=len(words),
            n_l2_promotions=n_l2,
        )
        profile.timings.append(timing)
        if on_doc:
            on_doc(i, timing)

    return profile


def assert_ingest_flat(
    profile: IngestProfile,
    *,
    max_quartile_ratio: float = DEFAULT_MAX_QUARTILE_RATIO,
    max_total_ms: float | None = None,
) -> None:
    flat = analyze_flatness(profile.totals_ms, max_ratio=max_quartile_ratio)
    assert flat["flat"], (
        f"per-doc ingest cost rising: quartile_ratio={flat['quartile_ratio']:.2f} "
        f"(first={flat['median_first_quartile_ms']}ms last={flat['median_last_quartile_ms']}ms)"
    )
    if max_total_ms is not None:
        assert sum(profile.totals_ms) < max_total_ms, f"ingest too slow: {sum(profile.totals_ms):.0f}ms"
