"""
Markov correlation brain — predict next word via cascading node transitions.

Each word sits at a **node** (prime chain + imaginary intersection).  Reading text
is a walk: node_t → node_{t+1}.  The brain predicts the next token from:

  1. **Direct Markov** — observed bigram transitions P(w₂|w₁)
  2. **Correlation cascade** — cross_links from SymbolKnowledgeIndex
  3. **Prime intersection** — shared chain primes between nodes

When prediction **misses**, the transition is reinforced (stronger correlation).

No global sequence model is required — each node only needs local transition
stats + correlation neighbors (lazy Markov on the 12-bit plane).
"""

from __future__ import annotations

import math
import pickle
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from aethos_symbol_cellular import _DEFAULT_MEMBRANE
from aethos_symbol_map import text_icn_chain, text_intersection

if TYPE_CHECKING:
    from aethos_symbol_knowledge import SymbolKnowledgeIndex

_TOKEN_RE = re.compile(r"[a-z]+")


@dataclass(frozen=True)
class TransitionNode:
    """One word at its lattice address — local Markov state."""

    word: str
    chain: tuple[int, ...]
    imag: int

    @property
    def chain_key(self) -> frozenset[int]:
        return frozenset(self.chain)


@dataclass(frozen=True)
class Prediction:
    word: str
    score: float
    source: str  # "markov" | "correlation" | "prime_overlap"


@dataclass
class ObserveResult:
    """One step: predict → compare → maybe strengthen."""

    prev: str
    actual: str
    predicted: tuple[Prediction, ...]
    hit: bool
    rank: int  # 0 = miss, 1 = top-1 hit
    strengthened: bool


@dataclass
class MarkovCorrelationBrain:
    """
    Lazy Markov layer on symbol knowledge.

    Transitions monitored at nodes; mismatches deepen correlations.
    """

    knowledge: SymbolKnowledgeIndex
    bigram: dict[tuple[str, str], int] = field(default_factory=dict)
    unigram: dict[str, int] = field(default_factory=dict)
    node_cache: dict[str, TransitionNode] = field(default_factory=dict)
    mismatch_strengthen: int = 0
    total_steps: int = 0
    hit_top1: int = 0
    hit_top5: int = 0

    # scoring weights
    w_markov: float = 1.0
    w_correlation: float = 0.6
    w_prime: float = 0.35

    def node(self, word: str) -> TransitionNode:
        w = word.lower()
        if w in self.node_cache:
            return self.node_cache[w]
        from pipeline.bit_12_symbol_plane_index import symbol_word_chain, symbol_word_imag

        chain = symbol_word_chain(self.knowledge, w)
        if not chain:
            chain = tuple(text_icn_chain(w))
        imag = symbol_word_imag(self.knowledge, w)
        if imag <= 0:
            imag = text_intersection(w)
        nd = TransitionNode(word=w, chain=chain, imag=imag)
        self.node_cache[w] = nd
        return nd

    def ingest_text(self, text: str) -> int:
        """Record token transitions from one document; returns bigram count added."""
        tokens = [
            t for t in _TOKEN_RE.findall(text.lower())
            if t not in _DEFAULT_MEMBRANE and len(t) >= 2
        ]
        added = 0
        for t in tokens:
            self.unigram[t] = self.unigram.get(t, 0) + 1
        for i in range(len(tokens) - 1):
            key = (tokens[i], tokens[i + 1])
            self.bigram[key] = self.bigram.get(key, 0) + 1
            added += 1
        return added

    def ingest_corpus(self, corpus: dict[str, str] | None = None) -> int:
        corpus = corpus or self.knowledge.corpus
        return sum(self.ingest_text(t) for t in corpus.values())

    def _markov_scores(self, prev: str) -> dict[str, float]:
        p = prev.lower()
        row: dict[str, float] = {}
        total = sum(c for (a, _), c in self.bigram.items() if a == p)
        if total <= 0:
            return row
        for (a, b), c in self.bigram.items():
            if a == p:
                row[b] = self.w_markov * (c / total)
        return row

    def _correlation_scores(self, prev: str) -> dict[str, float]:
        p = prev.lower()
        out: dict[str, float] = {}
        adj = getattr(self.knowledge, "cross_links", {})
        # fast adjacency if plane index attached to knowledge — else scan neighbors
        try:
            from pipeline.bit_12_symbol_plane_index import SymbolPlaneIndex

            plane = getattr(self, "_plane", None)
            if isinstance(plane, SymbolPlaneIndex):
                for other, strength, kind in plane.word_adjacency.get(p, [])[:24]:
                    boost = 1.0 if kind == "direct" else 0.5
                    out[other] = max(out.get(other, 0.0), self.w_correlation * strength * boost)
                return out
        except ImportError:
            pass

        for key, link in adj.items():
            if p not in key:
                continue
            other = key[1] if key[0] == p else key[0]
            boost = 1.0 if link.kind == "direct" else 0.55 if link.kind == "morph" else 0.35
            out[other] = max(out.get(other, 0.0), self.w_correlation * link.strength * boost)
        return out

    def _prime_overlap_scores(self, prev: str, candidates: set[str]) -> dict[str, float]:
        na = self.node(prev)
        out: dict[str, float] = {}
        if not na.chain:
            return out
        sa = na.chain_key
        for w in candidates:
            nb = self.node(w)
            if not nb.chain:
                continue
            inter = len(sa & nb.chain_key)
            if inter > 0:
                union = len(sa | nb.chain_key)
                out[w] = self.w_prime * (inter / union)
        return out

    def _blend_scores(self, prev: str) -> dict[str, float]:
        """Full cascade score distribution (not truncated)."""
        scores: dict[str, float] = {}
        for w, s in self._markov_scores(prev).items():
            scores[w] = scores.get(w, 0.0) + s
        corr = self._correlation_scores(prev)
        candidates = set(scores) | set(corr)
        for w, s in corr.items():
            scores[w] = scores.get(w, 0.0) + s
        for w, s in self._prime_overlap_scores(prev, candidates).items():
            scores[w] = scores.get(w, 0.0) + s
        return scores

    def predict_next(
        self,
        prev: str,
        *,
        top_k: int = 5,
    ) -> list[Prediction]:
        """Cascade: Markov row + correlation neighbors + prime overlap."""
        scores = self._blend_scores(prev)
        ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
        return [
            Prediction(word=w, score=s, source="blend")
            for w, s in ranked[:top_k]
        ]

    def log_prob_next(self, prev: str, actual: str, *, eps: float = 1e-9) -> float:
        """
        log P(actual | prev) from blended scores + smoothing over vocab floor.
        """
        scores = self._blend_scores(prev)
        actual_l = actual.lower()
        if scores:
            total = sum(scores.values())
            if actual_l in scores:
                p = scores[actual_l] / total
            else:
                # unseen in cascade support — smoothed mass
                p = eps / (total + eps * max(len(self.unigram), 1))
        else:
            # backoff to unigram
            u = self.unigram.get(actual_l, 0)
            tot = sum(self.unigram.values()) or 1
            p = max(u / tot, eps)
        return math.log(max(p, eps))

    def observe_step(
        self,
        prev: str,
        actual: str,
        *,
        top_k: int = 5,
        strengthen_on_miss: bool = True,
    ) -> ObserveResult:
        """
        Monitor one transition: predict next, compare, strengthen if miss.
        """
        self.total_steps += 1
        preds = self.predict_next(prev, top_k=top_k)
        pred_words = [p.word for p in preds]
        actual_l = actual.lower()

        if actual_l in pred_words:
            rank = pred_words.index(actual_l) + 1
            self.hit_top1 += 1 if rank == 1 else 0
            self.hit_top5 += 1
            hit = True
            strengthened = False
        else:
            rank = 0
            hit = False
            strengthened = False
            if strengthen_on_miss:
                strengthened = self._strengthen_transition(prev, actual_l)

        # always record observed transition
        key = (prev.lower(), actual_l)
        self.bigram[key] = self.bigram.get(key, 0) + 1
        self.unigram[actual_l] = self.unigram.get(actual_l, 0) + 1

        return ObserveResult(
            prev=prev.lower(),
            actual=actual_l,
            predicted=tuple(preds),
            hit=hit,
            rank=rank,
            strengthened=strengthened,
        )

    def _strengthen_transition(self, prev: str, actual: str) -> bool:
        """Miss → deepen correlation in knowledge brain."""
        prev_l, act = prev.lower(), actual.lower()
        self.mismatch_strengthen += 1

        # bump co-occurrence pair count (lazy deepen)
        key = tuple(sorted((prev_l, act)))
        pairs = self.knowledge._cooccur_pairs
        pairs[key] = pairs.get(key, 0) + 2

        self.knowledge._add_link(prev_l, act, kind="direct", strength=float(pairs[key]))
        return True

    def walk_text(
        self,
        text: str,
        *,
        top_k: int = 5,
        strengthen_on_miss: bool = True,
    ) -> list[ObserveResult]:
        """Predict through full text; strengthen on each miss."""
        tokens = [
            t for t in _TOKEN_RE.findall(text.lower())
            if t not in _DEFAULT_MEMBRANE and len(t) >= 2
        ]
        results: list[ObserveResult] = []
        for i in range(len(tokens) - 1):
            results.append(
                self.observe_step(
                    tokens[i],
                    tokens[i + 1],
                    top_k=top_k,
                    strengthen_on_miss=strengthen_on_miss,
                )
            )
        return results

    def eval_text_perplexity(
        self,
        text: str,
        *,
        strengthen_on_miss: bool = False,
    ) -> dict[str, float]:
        """Perplexity on one text; optional strengthen (training mode)."""
        tokens = [
            t for t in _TOKEN_RE.findall(text.lower())
            if t not in _DEFAULT_MEMBRANE and len(t) >= 2
        ]
        if len(tokens) < 2:
            return {"steps": 0, "perplexity": float("inf"), "avg_log_prob": 0.0}
        log_sum = 0.0
        steps = 0
        for i in range(len(tokens) - 1):
            prev, actual = tokens[i], tokens[i + 1]
            log_sum += self.log_prob_next(prev, actual)
            steps += 1
            if strengthen_on_miss:
                self.observe_step(prev, actual, strengthen_on_miss=True)
            else:
                # eval only — count hits without mutating bigram
                preds = self.predict_next(prev, top_k=5)
                pred_words = [p.word for p in preds]
                self.total_steps += 1
                if actual in pred_words:
                    self.hit_top5 += 1
                    if pred_words[0] == actual:
                        self.hit_top1 += 1
        avg = log_sum / steps
        return {
            "steps": steps,
            "perplexity": math.exp(-avg),
            "avg_log_prob": avg,
        }

    def eval_corpus_perplexity(
        self,
        corpus: dict[str, str],
        *,
        strengthen_on_miss: bool = False,
        max_docs: int | None = None,
    ) -> dict[str, float]:
        """Aggregate perplexity over corpus documents."""
        doc_ids = sorted(corpus.keys())
        if max_docs is not None:
            doc_ids = doc_ids[:max_docs]
        total_steps = 0
        log_sum = 0.0
        for did in doc_ids:
            r = self.eval_text_perplexity(
                corpus[did], strengthen_on_miss=strengthen_on_miss,
            )
            n = int(r["steps"])
            if n > 0:
                total_steps += n
                log_sum += r["avg_log_prob"] * n
        if total_steps <= 0:
            return {"docs": 0, "steps": 0, "perplexity": float("inf")}
        avg = log_sum / total_steps
        return {
            "docs": len(doc_ids),
            "steps": total_steps,
            "perplexity": math.exp(-avg),
            "avg_log_prob": avg,
            "top1": self.hit_top1 / max(self.total_steps, 1),
            "top5": self.hit_top5 / max(self.total_steps, 1),
        }

    def training_passes(
        self,
        corpus: dict[str, str],
        *,
        n_passes: int = 3,
        strengthen_on_miss: bool = True,
        max_docs: int | None = None,
    ) -> list[dict[str, float]]:
        """Run N strengthen passes; return perplexity after each pass."""
        doc_ids = sorted(corpus.keys())
        if max_docs is not None:
            doc_ids = doc_ids[:max_docs]
        history: list[dict[str, float]] = []
        for p in range(n_passes):
            self.hit_top1 = 0
            self.hit_top5 = 0
            self.total_steps = 0
            for did in doc_ids:
                self.eval_text_perplexity(
                    corpus[did], strengthen_on_miss=strengthen_on_miss,
                )
            history.append({
                "pass": p + 1,
                "perplexity": self.eval_corpus_perplexity(
                    {k: corpus[k] for k in doc_ids},
                    strengthen_on_miss=False,
                )["perplexity"],
                "mismatch_strengthen": self.mismatch_strengthen,
                "bigram_edges": len(self.bigram),
                "top1": self.hit_top1 / max(self.total_steps, 1),
                "top5": self.hit_top5 / max(self.total_steps, 1),
            })
        return history

    def accuracy(self) -> dict[str, float]:
        if self.total_steps <= 0:
            return {"top1": 0.0, "top5": 0.0, "steps": 0}
        return {
            "top1": self.hit_top1 / self.total_steps,
            "top5": self.hit_top5 / self.total_steps,
            "steps": self.total_steps,
            "mismatch_strengthen": self.mismatch_strengthen,
        }

    def attach_plane(self, plane) -> None:
        """Optional BIT 12 index for fast correlation cascade."""
        self._plane = plane

    def summary(self) -> dict[str, int | float]:
        acc = self.accuracy()
        return {
            "bigram_edges": len(self.bigram),
            "vocab": len(self.unigram),
            "nodes_cached": len(self.node_cache),
            **acc,
        }

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        return out

    @classmethod
    def load(cls, path: str | Path) -> MarkovCorrelationBrain:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"expected MarkovCorrelationBrain, got {type(obj)}")
        return obj


def build_markov_brain(
    knowledge: SymbolKnowledgeIndex,
    *,
    attach_plane_index: bool = True,
) -> MarkovCorrelationBrain:
    """Build Markov transitions from knowledge corpus + attach plane for cascade."""
    t0 = time.perf_counter()
    brain = MarkovCorrelationBrain(knowledge=knowledge)
    brain.ingest_corpus()

    if attach_plane_index:
        try:
            from pipeline.bit_12_symbol_plane_index import build_symbol_plane_index

            plane = build_symbol_plane_index(knowledge, pair_key_limit=50_000)
            brain.attach_plane(plane)
        except Exception:
            pass

    brain._build_ms = (time.perf_counter() - t0) * 1000.0
    return brain


def markov_path(dataset: str) -> Path:
    root = Path(__file__).resolve().parent / "brains" / "symbol_knowledge"
    return root / f"{dataset}_markov.pkl"


def demo() -> None:
    from aethos_symbol_knowledge import PRETRAIN_QUANTUM_GOLD, SymbolKnowledgeIndex

    knowledge = SymbolKnowledgeIndex.build_from_corpus(
        PRETRAIN_QUANTUM_GOLD, dataset="markov_demo",
    )
    brain = build_markov_brain(knowledge, attach_plane_index=False)
    brain.ingest_corpus()

    print("=" * 60)
    print("MARKOV CORRELATION BRAIN — cascade prediction")
    print("=" * 60)
    print(f"  summary: {brain.summary()}")

    prev = "quantum"
    preds = brain.predict_next(prev, top_k=6)
    print(f"\n  predict after {prev!r}:")
    for p in preds:
        print(f"    {p.word:14} score={p.score:.3f}  source={p.source}")

    text = PRETRAIN_QUANTUM_GOLD["gold_quantum_biometrics"]
    results = brain.walk_text(text, strengthen_on_miss=True)
    misses = [r for r in results if not r.hit]
    print(f"\n  walk {len(results)} steps  top1 acc={brain.accuracy()['top1']:.2f}")
    print(f"  misses={len(misses)}  strengthened={brain.mismatch_strengthen}")
    if misses[:3]:
        r = misses[0]
        print(f"  example miss: {r.prev!r} -> {r.actual!r}  predicted={[p.word for p in r.predicted[:3]]}")


if __name__ == "__main__":
    demo()
