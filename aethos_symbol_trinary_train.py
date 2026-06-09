"""
Query → gold doc trinary training — promote rarest 3-way correlations.

Training flow
-------------
1. Take query + gold doc(s) from qrels
2. Extract word triples in gold text (sliding window, membrane excluded)
3. Score rarity: low corpus frequency + query overlap + not yet linked
4. Promote top rare 3-ways → strengthen all 3 pair edges + register triple meet
5. Multiple gold docs: find triples for the **rarest** gold doc and shared
   triples across all golds — predict that doc's correlation bundle together

Uses symbol prime meets for the 3-way witness (imag sum) on the complex plane.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from aethos_symbol_cellular import _DEFAULT_MEMBRANE
from aethos_symbol_map import text_intersection

if TYPE_CHECKING:
    from aethos_symbol_knowledge import SymbolKnowledgeIndex

_TOKEN_RE = re.compile(r"[a-z]+")


@dataclass(frozen=True)
class WordTriple:
    """Three content words forming a rare correlation in a gold doc."""

    words: tuple[str, str, str]
    gold_doc_id: str
    rarity: float
    query_overlap: int
    meet_imag: int
    in_all_golds: bool = False


@dataclass
class PromotedTriple:
    """Promoted 3-way correlation saved in the brain."""

    words: tuple[str, str, str]
    gold_doc_ids: frozenset[str]
    rarity: float
    meet_imag: int
    promotion_strength: float = 5.0


@dataclass
class TrinaryTrainReport:
    query_id: str
    query_text: str
    gold_doc_ids: tuple[str, ...]
    rarest_gold_doc: str
    triples_found: int
    triples_promoted: int
    promoted: tuple[PromotedTriple, ...]


@dataclass
class TrinaryTrainer:
    """
    Promote rare 3-way correlations from query/gold supervision.
    """

    knowledge: SymbolKnowledgeIndex
    promoted_triples: dict[tuple[str, str, str], PromotedTriple] = field(
        default_factory=dict,
    )
    window: int = 15
    max_triples_per_query: int = 12
    min_word_len: int = 3

    def _tokens(self, text: str) -> list[str]:
        return [
            t for t in _TOKEN_RE.findall(text.lower())
            if t not in _DEFAULT_MEMBRANE and len(t) >= self.min_word_len
        ]

    def _word_doc_freq(self, word: str) -> int:
        w = word.lower()
        count = 0
        for text in self.knowledge.corpus.values():
            if w in self._tokens(text):
                count += 1
        return count

    def _rarity_score(
        self,
        triple: tuple[str, str, str],
        query_words: set[str],
    ) -> tuple[float, int]:
        """Higher = rarer. Returns (score, query_overlap)."""
        overlap = sum(1 for w in triple if w in query_words)
        if overlap == 0:
            return 0.0, 0
        freq_scores: list[float] = []
        n_docs = max(len(self.knowledge.corpus), 1)
        for w in triple:
            df = self._word_doc_freq(w)
            freq_scores.append(math.log((n_docs + 1) / (df + 1)) + 1.0)
        pair_penalty = 0.0
        for a, b in combinations(triple, 2):
            if self.knowledge.correlates(a, b):
                lk = self.knowledge.correlates(a, b)
                if lk and lk.strength > 3:
                    pair_penalty += 0.5
        # Favor triples that bind query terms together (2+ query words = gold signal)
        query_boost = 1.0 + overlap * 0.75
        if overlap >= 2:
            query_boost += 2.0
        if overlap >= 3:
            query_boost += 3.0
        base = min(freq_scores) * query_boost
        return max(base - pair_penalty, 0.01), overlap

    def _meet_imag(self, triple: tuple[str, str, str]) -> int:
        from pipeline.bit_12_symbol_plane_index import symbol_word_imag

        return sum(symbol_word_imag(self.knowledge, w) for w in triple)

    def _extract_triples_from_text(
        self,
        text: str,
        gold_doc_id: str,
        query_words: set[str],
    ) -> list[WordTriple]:
        tokens = self._tokens(text)
        seen: set[tuple[str, str, str]] = set()
        out: list[WordTriple] = []
        for i in range(len(tokens)):
            window = tokens[i : i + self.window]
            if len(window) < 3:
                continue
            for combo in combinations(dict.fromkeys(window), 3):
                key = tuple(sorted(combo))
                if key in seen:
                    continue
                rarity, overlap = self._rarity_score(key, query_words)
                if overlap == 0 or rarity <= 0:
                    continue
                seen.add(key)
                out.append(
                    WordTriple(
                        words=key,
                        gold_doc_id=gold_doc_id,
                        rarity=rarity,
                        query_overlap=overlap,
                        meet_imag=self._meet_imag(key),
                    )
                )
        out.sort(key=lambda t: (-t.rarity, -t.query_overlap, t.words))
        return out

    def _gold_doc_rarity(self, doc_id: str, query_words: set[str]) -> float:
        """How rare is this gold doc's vocabulary (lower freq = rarer doc)."""
        text = self.knowledge.corpus.get(doc_id, "")
        toks = set(self._tokens(text))
        if not toks:
            return 0.0
        n_docs = max(len(self.knowledge.corpus), 1)
        scores = [
            math.log((n_docs + 1) / (self._word_doc_freq(w) + 1))
            for w in toks
            if w not in query_words
        ]
        return sum(scores) / max(len(scores), 1)

    def find_rare_triples(
        self,
        query_text: str,
        gold_doc_ids: Sequence[str],
    ) -> tuple[str, list[WordTriple]]:
        """
        Find rare 3-way correlations in gold doc(s).

        Returns (rarest_gold_doc_id, ranked triples).
        """
        query_words = set(self._tokens(query_text))
        if not query_words or not gold_doc_ids:
            return "", []

        gold_rarity = {
            gid: self._gold_doc_rarity(gid, query_words)
            for gid in gold_doc_ids
            if gid in self.knowledge.corpus
        }
        if not gold_rarity:
            return "", []

        rarest_gold = max(gold_rarity, key=lambda g: gold_rarity[g])

        per_doc: dict[str, list[WordTriple]] = {}
        for gid in gold_doc_ids:
            if gid not in self.knowledge.corpus:
                continue
            per_doc[gid] = self._extract_triples_from_text(
                self.knowledge.corpus[gid], gid, query_words,
            )

        all_gold_set = set(gold_doc_ids)
        triple_keys_in_all: set[tuple[str, str, str]] = set()
        if len(all_gold_set) > 1:
            key_sets = [
                {t.words for t in per_doc.get(gid, [])}
                for gid in gold_doc_ids
                if gid in per_doc
            ]
            if key_sets:
                triple_keys_in_all = set.intersection(*key_sets) if key_sets else set()

        merged: dict[tuple[str, str, str], WordTriple] = {}
        for gid, triples in per_doc.items():
            for t in triples:
                key = t.words
                in_all = key in triple_keys_in_all
                boost = 1.5 if gid == rarest_gold else 1.0
                if in_all:
                    boost *= 1.25
                adj = WordTriple(
                    words=t.words,
                    gold_doc_id=gid,
                    rarity=t.rarity * boost,
                    query_overlap=t.query_overlap,
                    meet_imag=t.meet_imag,
                    in_all_golds=in_all,
                )
                if key not in merged or merged[key].rarity < adj.rarity:
                    merged[key] = adj

        ranked = sorted(
            merged.values(),
            key=lambda t: (-t.query_overlap, -t.rarity, t.words),
        )
        return rarest_gold, ranked[: self.max_triples_per_query]

    def promote_triple(
        self,
        triple: WordTriple,
        *,
        gold_doc_ids: frozenset[str] | None = None,
        strength: float = 5.0,
    ) -> PromotedTriple:
        """Strengthen all 3 pair edges + register 3-way meet in brain."""
        a, b, c = triple.words
        gids = gold_doc_ids or frozenset({triple.gold_doc_id})

        for pair in ((a, b), (a, c), (b, c)):
            key = tuple(sorted(pair))
            self.knowledge._cooccur_pairs[key] = (
                self.knowledge._cooccur_pairs.get(key, 0) + strength
            )
            master = self.knowledge._chamber_cooccur.setdefault(0, {})
            master[key] = master.get(key, 0) + int(strength)
            self.knowledge._add_link(
                pair[0], pair[1], kind="direct", strength=float(
                    self.knowledge._cooccur_pairs[key],
                ),
            )

        prom = PromotedTriple(
            words=triple.words,
            gold_doc_ids=gids,
            rarity=triple.rarity,
            meet_imag=triple.meet_imag,
            promotion_strength=strength,
        )
        self.promoted_triples[triple.words] = prom
        return prom

    def train_query(
        self,
        query_id: str,
        query_text: str,
        gold_doc_ids: Sequence[str],
    ) -> TrinaryTrainReport:
        """
        Query + gold doc(s) → find rarest 3-ways → promote for joint prediction.
        """
        rarest, triples = self.find_rare_triples(query_text, gold_doc_ids)
        gset = frozenset(gold_doc_ids)
        promoted: list[PromotedTriple] = []
        for t in triples:
            promoted.append(
                self.promote_triple(t, gold_doc_ids=gset),
            )

        return TrinaryTrainReport(
            query_id=query_id,
            query_text=query_text,
            gold_doc_ids=tuple(gold_doc_ids),
            rarest_gold_doc=rarest,
            triples_found=len(triples),
            triples_promoted=len(promoted),
            promoted=tuple(promoted),
        )

    def predict_triple_completion(
        self,
        query_text: str,
        known_words: Sequence[str],
        *,
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """
        Given query + 2 known words, predict 3rd from promoted triples.
        """
        query_words = set(self._tokens(query_text))
        known = {w.lower() for w in known_words}
        scores: dict[str, float] = {}
        for triple, prom in self.promoted_triples.items():
            tset = set(triple)
            if not known.issubset(tset):
                continue
            missing = (tset - known).pop() if len(tset - known) == 1 else None
            if not missing:
                continue
            overlap = len(tset & query_words)
            scores[missing] = max(
                scores.get(missing, 0.0),
                prom.rarity * (1 + overlap),
            )
        ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
        return ranked[:top_k]

    def train_from_qrels(
        self,
        queries: dict[str, str],
        qrels: dict[str, dict[str, int]],
        *,
        max_queries: int | None = 50,
    ) -> list[TrinaryTrainReport]:
        """Batch train rare 3-ways from BEIR-style qrels."""
        reports: list[TrinaryTrainReport] = []
        qids = [q for q in qrels if q in queries and qrels[q]]
        if max_queries:
            qids = qids[:max_queries]
        for qid in qids:
            gold_ids = [d for d, rel in qrels[qid].items() if rel > 0]
            reports.append(
                self.train_query(qid, queries[qid], gold_ids),
            )
        return reports


def load_beir_qrels_train(dataset: str = "scifact") -> tuple[dict[str, str], dict[str, dict[str, int]]]:
    """Load BEIR queries + train qrels."""
    from eval_beir import load_paths, load_queries, load_qrels
    from beir_data_root import resolve_beir_root

    paths = load_paths(Path(resolve_beir_root()), dataset)
    queries = load_queries(paths.queries)
    qrels = load_qrels(paths.qrels_train)
    return queries, qrels


def demo() -> None:
    from aethos_symbol_knowledge import PRETRAIN_QUANTUM_GOLD, SymbolKnowledgeIndex

    corpus = dict(PRETRAIN_QUANTUM_GOLD)
    corpus["d_sparse"] = "protein gene expression only"
    knowledge = SymbolKnowledgeIndex.build_from_corpus(corpus, dataset="trinary_demo")
    trainer = TrinaryTrainer(knowledge=knowledge)

    query = "quantum zero dimension inductive biometrics"
    report = trainer.train_query("q1", query, ["gold_quantum_biometrics"])

    print("=" * 60)
    print("TRINARY TRAIN — rarest 3-way in gold doc")
    print("=" * 60)
    print(f"  query: {query!r}")
    print(f"  rarest gold: {report.rarest_gold_doc}")
    print(f"  promoted: {report.triples_promoted}")
    for p in report.promoted[:6]:
        print(f"    {p.words}  rarity={p.rarity:.2f}  imag={p.meet_imag}")

    pred = trainer.predict_triple_completion(
        query, ["quantum", "zero"],
    )
    print(f"\n  predict 3rd given quantum+zero: {pred[:5]}")


if __name__ == "__main__":
    demo()
