"""
Branching meets on promoted subword primes — 4/5/6/…/9 symbol composites.

When an L2 prime represents an ordered 1–3 symbol subword, it can **meet** another
promoted prime (2-way solo swap) or two others (3-way triple lock).  Each meet
promotes to a **new** pool prime whose symbol span is the sum of parents:

  2-way meet
    3 + 1 symbols  →  4-way span  →  new prime P₄
    3 + 2 symbols  →  5-way span  →  new prime P₅
    3 + 3 symbols  →  6-way span  →  new prime P₆

  3-way meet (max span 9)
    3 + 3 + 3      →  9-way span  →  new prime P₉

Each new composite prime is a solo anchor that can **branch again** — pair/triple
meets with other L2 or L3+ nodes discover deeper intersections until span caps at 9.

Standalone ladder
-----------------
  L1  symbol primes
  L2  ordered 1–3-gram primes   (``aethos_symbol_promotion``)
  L3+ composite meet primes     (this module)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

from aethos_complex_plane import ComplexPlane3D, wing_transform
from aethos_intersection_nodes import IntersectionNetwork, MeetKind, MeetWitness
from aethos_lattice import BranchKind
from aethos_sequences import SequenceKind, make_chain
from aethos_symbol_map import text_symbol_chain
from aethos_symbol_promotion import MAX_SUBWORD_LEN, OrderedSubword, SymbolPromotionRegistry

MAX_COMPOSITE_SPAN = 9
MIN_COMPOSITE_SPAN = 4

# L3+ composite band — above standalone L2 pool start (576); large enough for branching.
_L3_COMPOSITE_PRIMES: tuple[int, ...] = make_chain(SequenceKind.PRIMES, 20_000)[3000:15_000]


@dataclass(frozen=True)
class MeetNode:
    """Any promotable solo anchor — L2 subword or L3+ composite."""

    text: str
    prime: int
    symbol_span: int
    tier: int  # 2 = L2, 3+ = composite generation
    parent_primes: tuple[int, ...] = ()


@dataclass(frozen=True)
class PromotedComposite:
    """Meet-promoted composite — new prime for combined symbol span."""

    text: str
    prime: int
    symbol_span: int
    meet_arity: int  # 2 = pair meet, 3 = triple meet
    way_label: str  # e.g. "4-way", "9-way"
    parent_primes: tuple[int, ...]
    parent_texts: tuple[str, ...]
    witness: MeetWitness


@dataclass
class CompositePromotionRegistry:
    """Allocates L3+ primes for meet-derived composites; supports re-branching."""

    max_span: int = MAX_COMPOSITE_SPAN
    _cursor: int = 0
    composites: dict[str, PromotedComposite] = field(default_factory=dict)
    nodes: dict[int, MeetNode] = field(default_factory=dict)  # prime → node

    def register_l2(self, tok: OrderedSubword) -> MeetNode:
        node = MeetNode(
            text=tok.text,
            prime=tok.prime,
            symbol_span=tok.length,
            tier=2,
            parent_primes=(tok.prime,),
        )
        self.nodes[tok.prime] = node
        return node

    def register_composite(self, comp: PromotedComposite) -> MeetNode:
        node = MeetNode(
            text=comp.text,
            prime=comp.prime,
            symbol_span=comp.symbol_span,
            tier=max(3, comp.meet_arity + 1),
            parent_primes=comp.parent_primes,
        )
        self.nodes[comp.prime] = node
        return node

    def _alloc_prime(self) -> int:
        if self._cursor >= len(_L3_COMPOSITE_PRIMES):
            raise RuntimeError("standalone L3 composite pool exhausted")
        p = _L3_COMPOSITE_PRIMES[self._cursor]
        self._cursor += 1
        return p

    def _meet_key(self, arity: int, texts: tuple[str, ...]) -> str:
        return f"{arity}:{'+'.join(texts)}"

    def promote_pair_meet(
        self,
        left: MeetNode,
        right: MeetNode,
        witness: MeetWitness,
    ) -> PromotedComposite | None:
        span = left.symbol_span + right.symbol_span
        if span < MIN_COMPOSITE_SPAN or span > self.max_span:
            return None
        texts = (left.text, right.text)
        key = self._meet_key(2, texts)
        if key in self.composites:
            return self.composites[key]
        prime = self._alloc_prime()
        comp = PromotedComposite(
            text=left.text + right.text,
            prime=prime,
            symbol_span=span,
            meet_arity=2,
            way_label=f"{span}-way",
            parent_primes=(left.prime, right.prime),
            parent_texts=texts,
            witness=witness,
        )
        self.composites[key] = comp
        self.register_composite(comp)
        return comp

    def promote_triple_meet(
        self,
        a: MeetNode,
        b: MeetNode,
        c: MeetNode,
        witness: MeetWitness,
    ) -> PromotedComposite | None:
        span = a.symbol_span + b.symbol_span + c.symbol_span
        if span < MIN_COMPOSITE_SPAN or span > self.max_span:
            return None
        texts = (a.text, b.text, c.text)
        key = self._meet_key(3, texts)
        if key in self.composites:
            return self.composites[key]
        prime = self._alloc_prime()
        comp = PromotedComposite(
            text=a.text + b.text + c.text,
            prime=prime,
            symbol_span=span,
            meet_arity=3,
            way_label=f"{span}-way",
            parent_primes=(a.prime, b.prime, c.prime),
            parent_texts=texts,
            witness=witness,
        )
        self.composites[key] = comp
        self.register_composite(comp)
        return comp

    def composite_node(
        self,
        text: str,
        *,
        n: int = 7,
        branch: BranchKind = BranchKind.VA1,
        wing: int = 1,
    ) -> tuple[PromotedComposite, ComplexPlane3D] | None:
        for comp in self.composites.values():
            if comp.text == text:
                psi = wing_transform(branch, (comp.prime,), n=n, wing=wing)
                return comp, psi
        return None

    def by_span(self, span: int) -> tuple[PromotedComposite, ...]:
        return tuple(c for c in self.composites.values() if c.symbol_span == span)


def _as_nodes(registry: SymbolPromotionRegistry) -> list[MeetNode]:
    return [
        MeetNode(
            text=t.text,
            prime=t.prime,
            symbol_span=t.length,
            tier=2,
            parent_primes=(t.prime,),
        )
        for t in registry.promoted.values()
    ]


def probe_pair_meet(left: MeetNode, right: MeetNode, *, wing: int = 1) -> MeetWitness | None:
    net = IntersectionNetwork()
    p, q = sorted((left.prime, right.prime))
    return net.probe_solo_swap(p, q, wing=wing, branch=BranchKind.VA1)


def probe_triple_meet(a: MeetNode, b: MeetNode, c: MeetNode, *, wing: int = 1) -> MeetWitness | None:
    primes = sorted((a.prime, b.prime, c.prime))
    if len(set(primes)) < 3:
        return None
    net = IntersectionNetwork()
    return net.probe_triple(primes[0], primes[1], primes[2], wing=wing, branch=BranchKind.VA1)


def discover_pair_meets(
    nodes: list[MeetNode],
    comp_reg: CompositePromotionRegistry,
    *,
    max_new: int | None = None,
) -> list[PromotedComposite]:
    found: list[PromotedComposite] = []
    for left, right in combinations(nodes, 2):
        span = left.symbol_span + right.symbol_span
        if span < MIN_COMPOSITE_SPAN or span > comp_reg.max_span:
            continue
        witness = probe_pair_meet(left, right)
        if witness is None:
            continue
        comp = comp_reg.promote_pair_meet(left, right, witness)
        if comp:
            found.append(comp)
            if max_new is not None and len(found) >= max_new:
                break
    return found


def discover_triple_meets(
    nodes: list[MeetNode],
    comp_reg: CompositePromotionRegistry,
    *,
    max_new: int | None = None,
) -> list[PromotedComposite]:
    found: list[PromotedComposite] = []
    for trio in combinations(nodes, 3):
        a, b, c = trio
        span = a.symbol_span + b.symbol_span + c.symbol_span
        if span < MIN_COMPOSITE_SPAN or span > comp_reg.max_span:
            continue
        witness = probe_triple_meet(a, b, c)
        if witness is None:
            continue
        comp = comp_reg.promote_triple_meet(a, b, c, witness)
        if comp:
            found.append(comp)
            if max_new is not None and len(found) >= max_new:
                break
    return found


def _l2_nodes_for_branching(
    l2_registry: SymbolPromotionRegistry,
    *,
    min_frequency: int = 1,
) -> list[MeetNode]:
    """Corpus-observed L2 nodes only — skip zero-frequency path siblings for branching."""
    return [
        MeetNode(
            text=t.text,
            prime=t.prime,
            symbol_span=t.length,
            tier=2,
            parent_primes=(t.prime,),
        )
        for t in l2_registry.promoted.values()
        if t.frequency >= min_frequency
    ]


def _try_pair(
    left: MeetNode,
    right: MeetNode,
    comp_reg: CompositePromotionRegistry,
) -> PromotedComposite | None:
    span = left.symbol_span + right.symbol_span
    if span < MIN_COMPOSITE_SPAN or span > comp_reg.max_span:
        return None
    witness = probe_pair_meet(left, right)
    if witness is None:
        return None
    return comp_reg.promote_pair_meet(left, right, witness)


def branch_meets(
    l2_registry: SymbolPromotionRegistry,
    *,
    max_rounds: int = 3,
    max_span: int = MAX_COMPOSITE_SPAN,
    min_frequency: int = 0,
    max_composites_per_round: int = 2000,
) -> CompositePromotionRegistry:
    """
    Branch 2-way / 3-way meets into 4–9 symbol composite primes.

    Round 0 — among L2 seeds (frequency ≥ ``min_frequency``):
      pair meets span 4–9, triple meets span 4–9 (3+3+3 → 9-way).

    Later rounds — each composite from the prior frontier re-meets L2 seeds
    and sibling composites (controlled growth).

    Use ``min_frequency=1`` after corpus ingest to skip zero-freq path siblings.
    """
    comp_reg = CompositePromotionRegistry(max_span=max_span)
    for tok in l2_registry.promoted.values():
        comp_reg.register_l2(tok)

    l2_seeds = _l2_nodes_for_branching(l2_registry, min_frequency=min_frequency)
    frontier: list[MeetNode] = []

    discover_pair_meets(l2_seeds, comp_reg, max_new=max_composites_per_round)
    discover_triple_meets(l2_seeds, comp_reg, max_new=max_composites_per_round)
    frontier = [
        comp_reg.nodes[c.prime] for c in comp_reg.composites.values()
    ]

    for _ in range(max_rounds - 1):
        if not frontier:
            break
        new_frontier: list[MeetNode] = []
        added = 0

        for comp_node in frontier:
            for l2 in l2_seeds:
                if added >= max_composites_per_round:
                    break
                hit = _try_pair(comp_node, l2, comp_reg)
                if hit:
                    new_frontier.append(comp_reg.nodes[hit.prime])
                    added += 1

            for other in frontier:
                if other.prime >= comp_node.prime:
                    continue
                if added >= max_composites_per_round:
                    break
                hit = _try_pair(comp_node, other, comp_reg)
                if hit:
                    new_frontier.append(comp_reg.nodes[hit.prime])
                    added += 1

        if not new_frontier:
            break
        frontier = new_frontier

    return comp_reg


def span_table_for_trigram_base(
    l2_registry: SymbolPromotionRegistry,
    trigram: str = "the",
) -> dict[str, object]:
    """
    Show 3-symbol base meeting 1/2/3-symbol partners → 4/5/6-way composites.
    """
    l2_registry.promote_with_siblings(trigram, frequency=1)
    base = l2_registry.promoted[trigram]
    base_node = MeetNode(base.text, base.prime, base.length, 2, (base.prime,))

    partners = {
        "1-symbol": l2_registry.promote("e"),
        "2-symbol": l2_registry.promote("th"),
        "3-symbol": l2_registry.promote("cat"),
    }
    comp_reg = CompositePromotionRegistry()
    comp_reg.register_l2(base)
    for p in partners.values():
        comp_reg.register_l2(p)

    rows: list[dict[str, object]] = []
    for label, tok in partners.items():
        node = MeetNode(tok.text, tok.prime, tok.length, 2, (tok.prime,))
        witness = probe_pair_meet(base_node, node)
        comp = (
            comp_reg.promote_pair_meet(base_node, node, witness)
            if witness
            else None
        )
        rows.append({
            "partner": label,
            "partner_text": tok.text,
            "span": base.length + tok.length,
            "meet_ok": witness is not None,
            "composite_prime": comp.prime if comp else None,
            "way": comp.way_label if comp else None,
        })

    # 9-way: three 3-symbol trigrams
    for sw in ("the", "cat", "sat"):
        l2_registry.promote(sw)
    three = [
        MeetNode(l2_registry.promoted[sw].text, l2_registry.promoted[sw].prime, 3, 2)
        for sw in ("the", "cat", "sat")
    ]
    w9 = probe_triple_meet(*three)
    c9 = comp_reg.promote_triple_meet(*three, w9) if w9 else None

    return {
        "base": trigram,
        "pair_meets": rows,
        "nine_way": {
            "texts": ("the", "cat", "sat"),
            "span": 9,
            "meet_ok": w9 is not None,
            "composite_prime": c9.prime if c9 else None,
            "composite_text": c9.text if c9 else None,
        },
    }


def demo() -> None:
    from aethos_symbol_corpus import CorpusSubwordIndex

    print("=" * 60)
    print("BRANCHING MEETS — 4/5/6/9-way composite promotion")
    print("=" * 60)

    idx = CorpusSubwordIndex()
    idx.ingest_text("d1", "the cat sat on the mat")
    idx.promote_all()
    comp_reg = idx.branch_composites(max_rounds=2, min_frequency=1)

    print(f"  L2 promoted: {len(idx.registry.promoted)}")
    print(f"  L3+ composites: {len(comp_reg.composites)}")
    for span in range(4, 10):
        n = len(comp_reg.by_span(span))
        if n:
            sample = comp_reg.by_span(span)[0]
            print(f"    {span}-way: {n} composites  e.g. {sample.text!r} prime={sample.prime}")

    print("\n  trigram base meet table:")
    reg = SymbolPromotionRegistry()
    table = span_table_for_trigram_base(reg, "the")
    for row in table["pair_meets"]:
        print(f"    3 + {row['partner']:10}  span={row['span']}  {row['way']}  prime={row['composite_prime']}")
    n9 = table["nine_way"]
    print(f"    3+3+3 triple  span={n9['span']}  text={n9['composite_text']!r}  prime={n9['composite_prime']}")


if __name__ == "__main__":
    demo()
