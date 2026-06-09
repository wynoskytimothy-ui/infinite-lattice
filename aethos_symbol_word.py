"""
L4 word primes — 2-way / 3-way correlations on ≤9-symbol anchors → length 10–27.

After L3 composites cap at **9 symbols**, promoted word-prime nodes can **correlate**
again on the imaginary line:

  2-way correlation (each parent span ≤ 9)
    9 + 1  →  10 symbols  →  new word prime
    9 + 9  →  18 symbols  →  new word prime
    span ∈ [10, 18]

  3-way correlation (each parent span ≤ 9)
    9 + 9 + 9  →  27 symbols  →  new word prime
    span ∈ [10, 27]

Each promoted word prime is a solo anchor (distinct Ψ) that records parent word
primes and the meet witness — same pattern as L2→L3 branching.

Standalone ladder
-----------------
  L1   symbol primes
  L2   1–3-gram ordered subwords
  L3   4–9 symbol composites   (``aethos_symbol_composite``)
  L4   10–27 symbol word primes (this module)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

from aethos_complex_plane import ComplexPlane3D, wing_transform
from aethos_intersection_nodes import MeetWitness
from aethos_lattice import BranchKind
from aethos_sequences import SequenceKind, make_chain
from aethos_symbol_composite import (
    CompositePromotionRegistry,
    MeetNode,
    probe_pair_meet,
    probe_triple_meet,
)

MAX_WORD_PARENT_SPAN = 9
MIN_WORD_SPAN_2WAY = 10
MAX_WORD_SPAN_2WAY = 18
MIN_WORD_SPAN_3WAY = 10
MAX_WORD_SPAN_3WAY = 27

_L4_WORD_PRIMES: tuple[int, ...] = make_chain(SequenceKind.PRIMES, 40_000)[15_000:30_000]


@dataclass(frozen=True)
class PromotedWord:
    """L4 word prime from 2-way or 3-way correlation of ≤9-symbol anchors."""

    text: str
    prime: int
    symbol_span: int
    correlation_arity: int  # 2 or 3
    span_label: str  # e.g. "10-span", "18-span", "27-span"
    parent_primes: tuple[int, ...]
    parent_texts: tuple[str, ...]
    witness: MeetWitness


@dataclass
class WordPromotionRegistry:
    """L4 pool primes for word-length correlations."""

    max_parent_span: int = MAX_WORD_PARENT_SPAN
    _cursor: int = 0
    words: dict[str, PromotedWord] = field(default_factory=dict)
    nodes: dict[int, MeetNode] = field(default_factory=dict)

    def _alloc_prime(self) -> int:
        if self._cursor >= len(_L4_WORD_PRIMES):
            raise RuntimeError("standalone L4 word pool exhausted")
        p = _L4_WORD_PRIMES[self._cursor]
        self._cursor += 1
        return p

    def _key(self, arity: int, texts: tuple[str, ...]) -> str:
        return f"w{arity}:{'+'.join(texts)}"

    def promote_pair(
        self,
        left: MeetNode,
        right: MeetNode,
        witness: MeetWitness,
    ) -> PromotedWord | None:
        if left.symbol_span > self.max_parent_span or right.symbol_span > self.max_parent_span:
            return None
        span = left.symbol_span + right.symbol_span
        if span < MIN_WORD_SPAN_2WAY or span > MAX_WORD_SPAN_2WAY:
            return None
        texts = (left.text, right.text)
        key = self._key(2, texts)
        if key in self.words:
            return self.words[key]
        prime = self._alloc_prime()
        word = PromotedWord(
            text=left.text + right.text,
            prime=prime,
            symbol_span=span,
            correlation_arity=2,
            span_label=f"{span}-span",
            parent_primes=(left.prime, right.prime),
            parent_texts=texts,
            witness=witness,
        )
        self.words[key] = word
        self.nodes[prime] = MeetNode(
            text=word.text,
            prime=prime,
            symbol_span=span,
            tier=4,
            parent_primes=word.parent_primes,
        )
        return word

    def promote_triple(
        self,
        a: MeetNode,
        b: MeetNode,
        c: MeetNode,
        witness: MeetWitness,
    ) -> PromotedWord | None:
        for node in (a, b, c):
            if node.symbol_span > self.max_parent_span:
                return None
        span = a.symbol_span + b.symbol_span + c.symbol_span
        if span < MIN_WORD_SPAN_3WAY or span > MAX_WORD_SPAN_3WAY:
            return None
        texts = (a.text, b.text, c.text)
        key = self._key(3, texts)
        if key in self.words:
            return self.words[key]
        prime = self._alloc_prime()
        word = PromotedWord(
            text=a.text + b.text + c.text,
            prime=prime,
            symbol_span=span,
            correlation_arity=3,
            span_label=f"{span}-span",
            parent_primes=(a.prime, b.prime, c.prime),
            parent_texts=texts,
            witness=witness,
        )
        self.words[key] = word
        self.nodes[prime] = MeetNode(
            text=word.text,
            prime=prime,
            symbol_span=span,
            tier=4,
            parent_primes=word.parent_primes,
        )
        return word

    def by_span(self, span: int) -> tuple[PromotedWord, ...]:
        return tuple(w for w in self.words.values() if w.symbol_span == span)

    def word_node(
        self,
        text: str,
        *,
        n: int = 7,
        branch: BranchKind = BranchKind.VA1,
        wing: int = 1,
    ) -> tuple[PromotedWord, ComplexPlane3D] | None:
        for w in self.words.values():
            if w.text == text:
                psi = wing_transform(branch, (w.prime,), n=n, wing=wing)
                return w, psi
        return None


def word_seed_nodes(
    comp_reg: CompositePromotionRegistry,
    *,
    max_parent_span: int = MAX_WORD_PARENT_SPAN,
) -> list[MeetNode]:
    """All ≤9-symbol anchors (L2 + L3 composites) eligible for L4 correlation."""
    return [
        n for n in comp_reg.nodes.values()
        if n.symbol_span <= max_parent_span
    ]


def discover_word_pairs(
    seeds: list[MeetNode],
    word_reg: WordPromotionRegistry,
    *,
    max_new: int | None = None,
) -> list[PromotedWord]:
    found: list[PromotedWord] = []
    for left, right in combinations(seeds, 2):
        span = left.symbol_span + right.symbol_span
        if span < MIN_WORD_SPAN_2WAY or span > MAX_WORD_SPAN_2WAY:
            continue
        witness = probe_pair_meet(left, right)
        if witness is None:
            continue
        hit = word_reg.promote_pair(left, right, witness)
        if hit:
            found.append(hit)
            if max_new is not None and len(found) >= max_new:
                break
    return found


def discover_word_triples(
    seeds: list[MeetNode],
    word_reg: WordPromotionRegistry,
    *,
    max_new: int | None = None,
) -> list[PromotedWord]:
    found: list[PromotedWord] = []
    for trio in combinations(seeds, 3):
        a, b, c = trio
        span = a.symbol_span + b.symbol_span + c.symbol_span
        if span < MIN_WORD_SPAN_3WAY or span > MAX_WORD_SPAN_3WAY:
            continue
        witness = probe_triple_meet(a, b, c)
        if witness is None:
            continue
        hit = word_reg.promote_triple(a, b, c, witness)
        if hit:
            found.append(hit)
            if max_new is not None and len(found) >= max_new:
                break
    return found


def correlate_words(
    comp_reg: CompositePromotionRegistry,
    *,
    max_new_pairs: int = 5000,
    max_new_triples: int = 5000,
    max_parent_span: int = MAX_WORD_PARENT_SPAN,
    prefer_max_span_seeds: bool = True,
) -> WordPromotionRegistry:
    """
    Build L4 word primes from 2-way (10–18) and 3-way (10–27) correlations.

    ``prefer_max_span_seeds``: when True, prioritize 9-span composite seeds
    first so 9+1=10 and 9+9=18 fire before smaller parents exhaust the cap.
    """
    word_reg = WordPromotionRegistry(max_parent_span=max_parent_span)
    seeds = word_seed_nodes(comp_reg, max_parent_span=max_parent_span)

    if prefer_max_span_seeds:
        seeds = sorted(seeds, key=lambda n: (-n.symbol_span, n.prime))

    discover_word_pairs(seeds, word_reg, max_new=max_new_pairs)
    discover_word_triples(seeds, word_reg, max_new=max_new_triples)
    return word_reg


def span_range_table(
    comp_reg: CompositePromotionRegistry,
) -> dict[str, object]:
    """Demonstrate 9+1=10, 9+9=18, 9+9+9=27 with synthetic 9-span anchors."""
    from aethos_symbol_promotion import SymbolPromotionRegistry

    l2 = SymbolPromotionRegistry()
    # Build three distinct 9-span composites from promoted trigrams
    for sw in ("the", "cat", "sat"):
        l2.promote(sw)
    from aethos_symbol_composite import CompositePromotionRegistry as CR, MeetNode as MN

    cr = CR()
    for t in l2.promoted.values():
        cr.register_l2(t)

    nine_nodes: list[MeetNode] = []
    for label, parts in (
        ("nine_a", ("the", "cat", "sat")),
        ("nine_b", ("the", "the", "cat")),
        ("nine_c", ("cat", "sat", "the")),
    ):
        texts = parts
        primes = tuple(l2.promoted[p].prime for p in texts)
        node = MN("".join(texts), primes[0], 9, 3, primes)
        nine_nodes.append(node)

    # Use real 9-span composites from triple meet when possible
    seeds = [n for n in comp_reg.nodes.values() if n.symbol_span == 9]
    if len(seeds) < 2:
        seeds = word_seed_nodes(comp_reg)
    nine = [n for n in seeds if n.symbol_span == 9]
    one = [n for n in seeds if n.symbol_span == 1]

    word_reg = WordPromotionRegistry()
    rows: list[dict[str, object]] = []

    if nine and one:
        w = probe_pair_meet(nine[0], one[0])
        hit = word_reg.promote_pair(nine[0], one[0], w) if w else None
        rows.append({
            "kind": "2-way",
            "parents": (nine[0].symbol_span, one[0].symbol_span),
            "span": nine[0].symbol_span + one[0].symbol_span,
            "prime": hit.prime if hit else None,
        })

    if len(nine) >= 2:
        w = probe_pair_meet(nine[0], nine[1])
        hit = word_reg.promote_pair(nine[0], nine[1], w) if w else None
        rows.append({
            "kind": "2-way",
            "parents": (9, 9),
            "span": 18,
            "prime": hit.prime if hit else None,
        })

    if len(nine) >= 3:
        w = probe_triple_meet(nine[0], nine[1], nine[2])
        hit = word_reg.promote_triple(nine[0], nine[1], nine[2], w) if w else None
        rows.append({
            "kind": "3-way",
            "parents": (9, 9, 9),
            "span": 27,
            "prime": hit.prime if hit else None,
        })

    return {"correlations": rows, "nine_span_seeds": len(nine)}


def demo() -> None:
    from aethos_symbol_corpus import CorpusSubwordIndex

    print("=" * 60)
    print("L4 WORD PRIMES — 2-way 10–18, 3-way 10–27")
    print("=" * 60)

    idx = CorpusSubwordIndex()
    idx.ingest_text("d1", "the cat sat on the mat hypothesis ether")
    idx.promote_all()
    comp = idx.branch_composites(max_rounds=1, min_frequency=1, max_composites_per_round=800)
    words = correlate_words(comp, max_new_pairs=2000, max_new_triples=1000)

    seeds = word_seed_nodes(comp)
    print(f"  <=9-symbol seeds: {len(seeds)}")
    print(f"  L4 word primes: {len(words.words)}")
    for lo, hi, label in ((10, 18, "2-way"), (10, 27, "3-way+")):
        bucket = [w for w in words.words.values() if lo <= w.symbol_span <= hi]
        print(f"  span {lo}–{hi} ({label}): {len(bucket)}")
        if bucket:
            w = bucket[0]
            print(f"    e.g. span={w.symbol_span} arity={w.correlation_arity} prime={w.prime}")

    table = span_range_table(comp)
    print("\n  anchor table:")
    for row in table["correlations"]:
        print(f"    {row['kind']}  parents={row['parents']}  span={row['span']}  prime={row['prime']}")


if __name__ == "__main__":
    demo()
