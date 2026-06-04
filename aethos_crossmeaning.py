"""
L7-L9 cross-meaning lattice — category vectors full of correlating primes + Markov weights.

Example: "apple phone" with category "technical"
  -> technical vector accumulates primes for phone, chip, software, ...
  -> Markov P(word|technical), P(technical|word) score belonging
  -> query technical vector -> all known technical things ranked by weight
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Iterable

from aethos_promotion import LatticeTier, MultiLatticeStack, PromotionRegistry, PromotedToken


class CrossDim(IntEnum):
    L7 = 7
    L8 = 8
    L9 = 9


@dataclass
class CategoryVector:
    """
    Cross-meaning vector for a theme (e.g. 'technical').
    Holds weighted promoted primes — the lattice address of everything
    that correlates with this meaning.
    """

    name: str
    category_prime: int
    prime_weights: dict[int, float] = field(default_factory=dict)  # prime -> weight
    word_weights: dict[str, float] = field(default_factory=dict)   # token -> weight
    dim7: float = 0.0
    dim8: float = 0.0
    dim9: float = 0.0
    total_observations: int = 0

    def add_token(self, word: str, prime: int, weight: float = 1.0) -> None:
        self.prime_weights[prime] = self.prime_weights.get(prime, 0.0) + weight
        self.word_weights[word] = self.word_weights.get(word, 0.0) + weight
        self.total_observations += 1
        self._refresh_dims()

    def _refresh_dims(self) -> None:
        h = hash(self.name)
        s = sum(self.prime_weights.values()) or 1.0
        self.dim7 = ((h % 10_000) / 1000.0) + s * 0.001
        self.dim8 = (sum(self.prime_weights.keys()) % 10_000) / 1000.0
        self.dim9 = len(self.prime_weights) * 0.01

    def top_words(self, limit: int = 10) -> list[tuple[str, float]]:
        return sorted(self.word_weights.items(), key=lambda x: -x[1])[:limit]

    def top_primes(self, limit: int = 10) -> list[tuple[int, float]]:
        return sorted(self.prime_weights.items(), key=lambda x: -x[1])[:limit]


@dataclass
class MarkovCrossLattice:
    """
    Markovian cross-meaning layer on L7-L9.

    - P(word | category)  from co-occurrence in tagged sentences
    - P(category | word)  Bayes-style belonging
    - P(w2 | w1, category) bigram transitions within a theme
    """

    registry: PromotionRegistry
    categories: dict[str, CategoryVector] = field(default_factory=dict)
    _cat_word: dict[tuple[str, str], int] = field(default_factory=dict)
    _cat_total: dict[str, int] = field(default_factory=dict)
    _word_cat: dict[tuple[str, str], int] = field(default_factory=dict)
    _word_total: dict[str, int] = field(default_factory=dict)
    _bigram: dict[tuple[str, str, str], int] = field(default_factory=dict)  # cat,w1,w2
    _next_cat_prime_idx: int = 0

    def _cat_prime(self, name: str) -> int:
        from aethos_promotion import PROMOTION_POOL

        if name in self.categories:
            return self.categories[name].category_prime
        p = PROMOTION_POOL[400 + self._next_cat_prime_idx]
        self._next_cat_prime_idx += 1
        return p

    def ensure_category(self, name: str) -> CategoryVector:
        key = name.lower()
        if key not in self.categories:
            self.categories[key] = CategoryVector(name=key, category_prime=self._cat_prime(key))
        return self.categories[key]

    def prune_active(self, active: set[str]) -> None:
        """Drop stale category vectors and Markov counts not in current cluster_hubs."""
        active_l = {c.lower() for c in active}
        for cid in list(self.categories.keys()):
            if cid not in active_l:
                del self.categories[cid]
        for key in list(self._cat_word.keys()):
            if key[0] not in active_l:
                del self._cat_word[key]
        for key in list(self._cat_total.keys()):
            if key not in active_l:
                del self._cat_total[key]
        for key in list(self._word_cat.keys()):
            if key[1] not in active_l:
                del self._word_cat[key]
        for key in list(self._bigram.keys()):
            if key[0] not in active_l:
                del self._bigram[key]

    def _word_token(self, registry: PromotionRegistry, word: str) -> PromotedToken | None:
        w = word.lower().replace(" ", "")
        if not w.isalpha():
            return None
        return registry.resolve_token(w)

    def observe(
        self,
        category: str,
        words: Iterable[str],
        phrase_weights: dict[str, float] | None = None,
    ) -> None:
        """
        Tag a sentence/window with cross-meaning category.
        e.g. observe("technical", ["apple", "phone", "chip", "software"])
        """
        cat = self.ensure_category(category)
        word_list = [w.lower().replace(" ", "") for w in words if w.strip()]
        weights = phrase_weights or {}

        for w in word_list:
            tok = self._word_token(self.registry, w)
            if not tok:
                continue
            wt = weights.get(w, 1.0)
            cat.add_token(w, tok.prime, wt)

            self._cat_word[(cat.name, w)] = self._cat_word.get((cat.name, w), 0) + 1
            self._cat_total[cat.name] = self._cat_total.get(cat.name, 0) + 1
            self._word_cat[(w, cat.name)] = self._word_cat.get((w, cat.name), 0) + 1
            self._word_total[w] = self._word_total.get(w, 0) + 1

        for i in range(len(word_list) - 1):
            w1, w2 = word_list[i], word_list[i + 1]
            key = (cat.name, w1, w2)
            self._bigram[key] = self._bigram.get(key, 0) + 1

    def p_word_given_category(self, word: str, category: str) -> float:
        w, c = word.lower(), category.lower()
        num = self._cat_word.get((c, w), 0)
        den = self._cat_total.get(c, 0)
        return num / den if den else 0.0

    def p_category_given_word(self, word: str, category: str) -> float:
        w, c = word.lower(), category.lower()
        num = self._word_cat.get((w, c), 0)
        den = self._word_total.get(w, 0)
        return num / den if den else 0.0

    def p_bigram_given_category(self, w1: str, w2: str, category: str) -> float:
        c = category.lower()
        num = self._bigram.get((c, w1.lower(), w2.lower()), 0)
        den = self._cat_total.get(c, 0)
        return num / den if den else 0.0

    def prime_overlap_weight(self, word: str, category: str) -> float:
        """Cosine-like overlap: word's primes vs category vector prime weights."""
        w = word.lower().replace(" ", "")
        tok = self._word_token(self.registry, w)
        if not tok or category.lower() not in self.categories:
            return 0.0
        cat = self.categories[category.lower()]
        word_primes = set(tok.parent_primes + (tok.prime,))
        num = sum(cat.prime_weights.get(p, 0.0) for p in word_primes)
        den = sum(cat.prime_weights.values()) or 1.0
        return num / den

    def belonging_score(self, word: str, category: str) -> float:
        """
        Markovian belonging: combine P(cat|word), P(word|cat), prime-vector overlap.
        """
        p_c_w = self.p_category_given_word(word, category)
        p_w_c = self.p_word_given_category(word, category)
        overlap = self.prime_overlap_weight(word, category)
        if p_c_w == 0 and p_w_c == 0 and overlap == 0:
            return 0.0
        return (p_c_w + p_w_c + overlap) / 3.0

    def phrase_belonging(self, words: Iterable[str], category: str) -> float:
        """Multi-word phrase e.g. ('apple','phone') -> average + bigram Markov boost."""
        wl = [w.lower().replace(" ", "") for w in words]
        if not wl:
            return 0.0
        scores = [self.belonging_score(w, category) for w in wl]
        base = sum(scores) / len(scores)
        if len(wl) >= 2:
            bg = self.p_bigram_given_category(wl[0], wl[1], category)
            base = 0.7 * base + 0.3 * bg
        return base

    def all_in_category(self, category: str, min_weight: float = 0.01) -> list[tuple[str, float]]:
        """Find everything whose vector weight / Markov score ties to this category."""
        cat = self.categories.get(category.lower())
        if not cat:
            return []
        seen: dict[str, float] = {}
        for w, wt in cat.word_weights.items():
            score = max(wt / max(cat.total_observations, 1), self.belonging_score(w, category))
            if score >= min_weight:
                seen[w] = score
        return sorted(seen.items(), key=lambda x: -x[1])

    def cross_point_9d(
        self, word: str, category: str
    ) -> tuple[float, float, float, float, float, float, float, float, float]:
        """L1-L3 dot + L4-L6 from registry + L7-L9 category vector."""
        w = word.lower().replace(" ", "")
        reg = self.registry
        base = reg.lattice_address(w, LatticeTier.L3_WORD) if (LatticeTier.L3_WORD, w) in reg.promoted else (0, 0, 0)
        cat = self.categories.get(category.lower())
        d7, d8, d9 = (cat.dim7, cat.dim8, cat.dim9) if cat else (0, 0, 0)
        bw = self.belonging_score(w, category)
        return (base[0], base[1], base[2], bw, bw * 0.5, bw * 0.25, d7, d8, d9)


@dataclass
class SemanticStack(MultiLatticeStack):
    """Full stack: L1-L6 promotion + L7-L9 cross-meaning Markov layer."""

    cross: MarkovCrossLattice = field(default_factory=lambda: MarkovCrossLattice(PromotionRegistry()))

    def __post_init__(self) -> None:
        self.cross.registry = self.registry

    def train_tagged(self, category: str, *documents: str) -> None:
        for doc in documents:
            words = ["".join(c for c in t.lower() if c.isalpha()) for t in doc.split()]
            words = [w for w in words if w]
            self.registry.observe_text(doc)
            self.registry.observe_cooccurrence(words)
            self.cross.observe(category, words)

    def explain_cross(self, word: str, category: str) -> str:
        w = word.lower()
        lines = [
            f"Cross-meaning {word!r} x {category!r}:",
            f"  belonging score:     {self.cross.belonging_score(w, category):.4f}",
            f"  P({w}|{category}):   {self.cross.p_word_given_category(w, category):.4f}",
            f"  P({category}|{w}):   {self.cross.p_category_given_word(w, category):.4f}",
            f"  prime overlap:       {self.cross.prime_overlap_weight(w, category):.4f}",
        ]
        cat = self.cross.categories.get(category.lower())
        if cat:
            lines.append(f"  L7-L9 vector dims:   ({cat.dim7:.3f}, {cat.dim8:.3f}, {cat.dim9:.3f})")
            lines.append(f"  technical primes:    {cat.top_primes(5)}")
        pt = self.cross.cross_point_9d(w, category)
        lines.append(f"  9D point L1-L9:      {pt}")
        return "\n".join(lines)


def demo() -> None:
    stack = SemanticStack()

    tech_docs = [
        "apple phone chip processor technical",
        "apple phone software technical update",
        "phone technical support hardware",
        "samsung phone technical chip",
        "computer technical software hardware",
        "technical engineering phone tablet",
    ]
    food_docs = [
        "apple fruit pie recipe",
        "apple orchard fruit fresh",
        "banana fruit salad",
    ]

    for doc in tech_docs:
        stack.train_tagged("technical", doc)
    for doc in food_docs:
        stack.train_tagged("food", doc)

    print("=" * 60)
    print("L7-L9 CROSS-MEANING — Markov category vectors")
    print("=" * 60)

    print("\n--- 'technical' vector: all correlated things ---")
    for w, score in stack.cross.all_in_category("technical")[:12]:
        print(f"  {w:15s}  weight={score:.4f}")

    print("\n--- apple phone x technical (cross-meaning) ---")
    print(stack.explain_cross("phone", "technical"))
    phrase_score = stack.cross.phrase_belonging(["apple", "phone"], "technical")
    print(f"\n  phrase ('apple','phone') belonging to technical: {phrase_score:.4f}")

    print("\n--- apple: technical vs food (disambiguation) ---")
    t = stack.cross.belonging_score("apple", "technical")
    f = stack.cross.belonging_score("apple", "food")
    print(f"  apple -> technical: {t:.4f}")
    print(f"  apple -> food:      {f:.4f}")
    print(f"  stronger category:  {'technical' if t > f else 'food'}")

    print("\n--- technical vector prime weights (top) ---")
    cat = stack.cross.categories["technical"]
    for p, w in cat.top_primes(8):
        print(f"  prime {p}  weight={w:.1f}")


if __name__ == "__main__":
    demo()
