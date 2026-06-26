"""
Stage 07 — Semantic light-up at 3-way wing cages (L4–L6).

Promotion ladder (free tokens from formula — no extra vocab rows):
  L1 symbol → letter prime
  L2 subword → 2-way meet promoted (th, ing) — Stage 04
  L2/L3 subword → 3-way meet promoted (hing, ing) — same pool machinery
  L3 compound → product of constituent pool primes (th×ing ≠ t×hing ≠ thing letters)

Each promoted 3-way node may branch a wing cage:
  - Anchor = meet_composite (Section 5 / Stage 06 identity)
  - Correlated terms live on L4/L5/L6 (dim4/dim5/dim6) + rotation quadrant 1–4
  - 2-way edges carry strength (how often co-seen) → drift vs anchor

Up to 6 tokens per observation window; every sliding 3-way triple can light a cage.
Query lights rare shared dots — hub meets contribute near zero.
"""

from __future__ import annotations

import math
import unicodedata
from collections import defaultdict
from functools import reduce
from operator import mul
from dataclasses import dataclass, field
from typing import Iterable, Literal

CageIngestMode = Literal["positional", "rare_combo"]

from aethos_promotion import CorrelationLink, LatticeTier, PromotedToken
from aethos_words import letter_to_prime

from lattice_retriever_v1.stage04_promote import Stage04Registry, promote_from_stream
from lattice_retriever_v1.stage05_free_token import meet_composite
from lattice_retriever_v1.stage06_composites import (
    meet_composite_k,
    three_way_address,
)
from lattice_retriever_v1.stage06_composites import decompose_word as decompose_word_base


def _distinct_letter_primes(word: str) -> list[int]:
    """Letter primes for a token; NFKC ligatures (e.g. ﬁ) expand to multiple letters."""
    seen: set[int] = set()
    for c in word:
        if not c.isalpha():
            continue
        for ch in unicodedata.normalize("NFKC", c):
            if ch.isalpha():
                seen.add(letter_to_prime(ch))
    return sorted(seen)

HUB_WORDS = frozenset({"the", "is", "a", "an", "and", "or", "in", "on", "at", "to", "of"})


def anchor_composite(*token_ids: int) -> int:
    """
    Wing-cage inverted-index key = product of token identities.

    Token ids may be pool primes OR letter-composite products (free tokens stack).
    """
    if len(token_ids) < 2:
        raise ValueError("anchor needs at least 2 token identities")
    if len(set(token_ids)) != len(token_ids):
        raise ValueError(f"duplicate token identity in anchor: {token_ids}")
    return reduce(mul, sorted(token_ids), 1)


def correlation_dims(source_prime: int, target_prime: int, strength: int = 1) -> tuple[float, float, float]:
    """Deterministic L4–L6 placement from prime identities (same as PromotionRegistry)."""
    a = PromotedToken(
        text=str(source_prime),
        tier=LatticeTier.L2_SUBWORD,
        prime=source_prime,
        parent_primes=(source_prime,),
    )
    b = PromotedToken(
        text=str(target_prime),
        tier=LatticeTier.L2_SUBWORD,
        prime=target_prime,
        parent_primes=(target_prime,),
    )
    link = CorrelationLink.from_pair(a, b, strength=strength)
    return link.dim4, link.dim5, link.dim6


def rotation_quadrant_l4(source_prime: int, target_prime: int) -> int:
    """Map correlation pair → rotation slot 1–4 (VA branch family on L4–L6 plane)."""
    return ((source_prime ^ target_prime) % 4) + 1


@dataclass(frozen=True)
class CorrelationDot:
    """One correlated term on the L4–L6 wing cage."""

    term: str
    prime: int
    strength: int
    dim4: float
    dim5: float
    dim6: float
    rotation_quadrant: int

    @property
    def drift_weight(self) -> float:
        """High co-frequency → stays close (strong); used for scoring."""
        return float(self.strength) * (self.dim4 + self.dim6 + 1.0)

    def explain(self) -> dict:
        return {
            "term": self.term,
            "prime": self.prime,
            "strength": self.strength,
            "dim4": round(self.dim4, 4),
            "dim5": round(self.dim5, 4),
            "dim6": round(self.dim6, 4),
            "rotation_quadrant": self.rotation_quadrant,
            "drift_weight": round(self.drift_weight, 4),
        }


@dataclass
class WingCage:
    """
    3-way anchor node + lazy L4–L6 correlations.

    Base FTA address unchanged; cage materialized only when queried or observed.
    """

    anchor_label: str
    anchor_composite: int
    anchor_primes: tuple[int, ...]
    correlations: dict[str, CorrelationDot] = field(default_factory=dict)

    def add_correlation(
        self,
        term: str,
        prime: int,
        *,
        source_prime: int,
        strength: int = 1,
    ) -> None:
        d4, d5, d6 = correlation_dims(source_prime, prime, strength)
        rot = rotation_quadrant_l4(source_prime, prime)
        existing = self.correlations.get(term)
        if existing:
            strength = existing.strength + strength
            d4, d5, d6 = correlation_dims(source_prime, prime, strength)
        self.correlations[term] = CorrelationDot(
            term=term,
            prime=prime,
            strength=strength,
            dim4=d4,
            dim5=d5,
            dim6=d6,
            rotation_quadrant=rot,
        )

    def explain(self) -> dict:
        return {
            "anchor_label": self.anchor_label,
            "anchor_composite": self.anchor_composite,
            "anchor_primes": list(self.anchor_primes),
            "n_correlations": len(self.correlations),
            "correlations": [c.explain() for c in sorted(self.correlations.values(), key=lambda x: x.term)],
        }


@dataclass
class SemanticLightIndex:
    """
    Small inverted index: anchor_composite → WingCage.
    Doc observation fills 3-way cages from sliding windows (max 6 tokens).
    """

    registry: Stage04Registry = field(default_factory=lambda: promote_from_stream([]))
    cages: dict[int, WingCage] = field(default_factory=dict)
    doc_freq: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    n_docs: int = 0

    def idf(self, term: str) -> float:
        df = self.doc_freq.get(term.lower(), 0)
        return math.log((self.n_docs + 1) / (df + 1)) + 1.0

    def is_rare(self, term: str, *, max_df: int = 2) -> bool:
        t = term.lower()
        return t not in HUB_WORDS and self.doc_freq.get(t, 0) <= max_df

    def _prime_for_term(self, term: str) -> int:
        """One identity prime per word — pool promotion or Stage 06 decompose path."""
        t = term.lower()
        tok = self.registry.promoted_subword(t)
        if tok is not None:
            return tok.prime
        if len(t) == 1:
            return letter_to_prime(t)
        try:
            dec = decompose_word_base(t, self.registry)
            mc = dec.get("meet_composite")
            if mc is not None:
                return mc
        except ValueError:
            pass
        distinct = _distinct_letter_primes(t)
        if len(distinct) >= 2:
            return meet_composite_k(*distinct)
        return distinct[0]

    def corridor_pins_for_term(self, term: str) -> frozenset[int]:
        """
        Corridor posting keys: whole-word identity plus promoted L2 boundary
        atoms (prefix/suffix) so run/running share stem pins without indexing
        every internal 2-gram in the corpus.
        """
        t = term.lower()
        pins: set[int] = {self._prime_for_term(t)}
        n = len(t)
        for ln in range(2, min(5, n + 1)):
            for sw in (t[:ln], t[-ln:]):
                tok = self.registry.promoted_subword(sw)
                if tok is not None:
                    pins.add(tok.prime)
        return frozenset(pins)

    def _cage_for_triple(self, w1: str, w2: str, w3: str) -> WingCage:
        label = f"{w1}|{w2}|{w3}"
        p1, p2, p3 = self._prime_for_term(w1), self._prime_for_term(w2), self._prime_for_term(w3)
        primes = (p1, p2, p3)
        if len(set(primes)) == 3:
            comp = anchor_composite(*primes)
        elif len(set(primes)) == 2:
            sp = sorted(set(primes))
            comp = sp[0] * sp[1]
        else:
            comp = p1
        cage = self.cages.get(comp)
        if cage is None:
            cage = WingCage(anchor_label=label, anchor_composite=comp, anchor_primes=primes)
            self.cages[comp] = cage
        return cage

    def _observe_triple_cage(
        self,
        triple: tuple[str, str, str],
        *,
        context: tuple[str, ...],
    ) -> None:
        cage = self._cage_for_triple(*triple)
        anchor_p = cage.anchor_primes[0] if cage.anchor_primes else 3
        for t in triple:
            p = self._prime_for_term(t)
            cage.add_correlation(t, p, source_prime=anchor_p, strength=1)
        for t in context:
            if t in triple:
                continue
            p = self._prime_for_term(t)
            cage.add_correlation(t, p, source_prime=anchor_p, strength=1)

    def observe_doc(
        self,
        text: str,
        *,
        max_window: int = 6,
        mode: CageIngestMode = "positional",
        k_rare: int = 8,
        max_df_frac: float = 0.05,
    ) -> None:
        words = [w.lower() for w in text.split() if w.isalpha()]
        if not words:
            return
        self.n_docs += 1
        seen: set[str] = set()
        for w in words:
            if w not in seen:
                self.doc_freq[w] += 1
                seen.add(w)
        if mode == "rare_combo":
            from lattice_retriever_v1.doc_lattice_codec import build_rare_combo_cages

            triples = build_rare_combo_cages(
                words,
                self,
                self.registry,
                k_rare=k_rare,
                max_df_frac=max_df_frac,
            )
            rare_ctx = tuple(
                dict.fromkeys(t for triple in triples for t in triple)
            )
            for triple in triples:
                self._observe_triple_cage(triple, context=rare_ctx)
            return
        window = words[:max_window]
        for i in range(len(window) - 2):
            triple = tuple(window[i : i + 3])
            self._observe_triple_cage(triple, context=tuple(window))

    def observe_corpus(self, texts: Iterable[str]) -> None:
        for text in texts:
            self.observe_doc(text)

    def touch_weight(self, query_terms: Iterable[str], cage: WingCage) -> float:
        """Rare-weighted shared correlation touch — hub terms contribute ~0."""
        anchor_words = frozenset(cage.anchor_label.split("|"))
        total = 0.0
        hub = 0.0
        for qt in query_terms:
            q = qt.lower()
            w = 0.0
            dot = cage.correlations.get(q)
            if dot is not None:
                w = dot.drift_weight * self.idf(q)
            elif q in anchor_words:
                w = 2.0 * self.idf(q)
            if w == 0.0:
                continue
            total += w
            if q in HUB_WORDS:
                hub += w
        return total - hub

    def lift_score(self, query_terms: Iterable[str], anchor_composite: int) -> dict:
        cage = self.cages.get(anchor_composite)
        if cage is None:
            return {"score": 0.0, "cage": None}
        score = self.touch_weight(query_terms, cage)
        rare_hits = [t for t in query_terms if self.is_rare(t) and t.lower() in cage.correlations]
        return {
            "score": score,
            "anchor_composite": anchor_composite,
            "rare_hits": rare_hits,
            "cage": cage.explain(),
        }


def word_path_identities(word: str, registry: Stage04Registry) -> dict:
    """
    Glass-box: th+ing vs t+hing vs full letter product — all different free-token paths.

    Promoted subwords are pool primes; compounds = products (Stage 05/06).
    """
    w = word.lower()
    paths: dict[str, dict] = {}

    dec = decompose_word_base(w, registry)
    paths["promoted_greedy"] = dec

    letter_primes = tuple(letter_to_prime(c) for c in w if c.isalpha())
    if len(set(letter_primes)) == len(letter_primes):
        paths["letter_product"] = {
            "pieces": list(w),
            "meet_composite": meet_composite_k(*letter_primes),
        }

    th = registry.promoted_subword("th")
    ing = registry.promoted_subword("ing")
    hing = registry.promoted_subword("hing")
    if th and ing and w == "thing":
        paths["th_plus_ing"] = {
            "pieces": ["th", "ing"],
            "meet_composite": meet_composite(th.prime, ing.prime),
        }
    if hing and w == "thing":
        t = letter_to_prime("t")
        paths["t_plus_hing"] = {
            "pieces": ["t", "hing"],
            "meet_composite": meet_composite(t, hing.prime),
        }

    composites = {k: v["meet_composite"] for k, v in paths.items() if isinstance(v, dict) and "meet_composite" in v}
    key_paths = ("letter_product", "th_plus_ing", "t_plus_hing")
    key_comps = [composites[k] for k in key_paths if k in composites]
    paths["structural_paths_distinct"] = len(set(key_comps)) == len(key_comps)
    paths["composites"] = composites
    return paths


def build_demo_registry() -> Stage04Registry:
    """Corpus promoting th, ing, hing as separate subword meets."""
    corpus = [
        "running quickly",
        "walking slowly",
        "thinking deeply",
        "building houses",
        "math path",
        "bath myth",
        "nothing else",
        "everything counts",
        "door hinge",
        "with hinges",
        "the thing",
        "that thought",
    ]
    return promote_from_stream(corpus)
