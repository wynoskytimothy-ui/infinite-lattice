"""L4 word primes — 10–18 two-way, 10–27 three-way correlations."""

from __future__ import annotations

import unittest

from aethos_symbol_composite import (
    CompositePromotionRegistry,
    MeetNode,
    branch_meets,
    probe_pair_meet,
    probe_triple_meet,
)
from aethos_symbol_promotion import SymbolPromotionRegistry
from aethos_symbol_word import (
    MAX_WORD_SPAN_2WAY,
    MAX_WORD_SPAN_3WAY,
    MIN_WORD_SPAN_2WAY,
    WordPromotionRegistry,
    correlate_words,
    discover_word_pairs,
    word_seed_nodes,
)


def _nine_span_node(prime: int, text: str = "nine_span") -> MeetNode:
    return MeetNode(text=text, prime=prime, symbol_span=9, tier=3, parent_primes=(prime,))


def _one_span_node(prime: int, text: str = "a") -> MeetNode:
    return MeetNode(text=text, prime=prime, symbol_span=1, tier=2, parent_primes=(prime,))


class TestWordSpans(unittest.TestCase):
    def test_nine_plus_one_is_ten(self) -> None:
        reg = SymbolPromotionRegistry()
        l2_nine = reg.promote("the")
        l2_one = reg.promote("a")
        # Fake 9-span nodes using distinct high primes for meet geometry
        cr = CompositePromotionRegistry()
        n9 = _nine_span_node(l2_nine.prime + 10_000, "thecatsat")
        n1 = _one_span_node(l2_one.prime)
        cr.nodes[n9.prime] = n9
        cr.nodes[n1.prime] = n1

        word_reg = WordPromotionRegistry()
        w = probe_pair_meet(n9, n1)
        self.assertIsNotNone(w)
        hit = word_reg.promote_pair(n9, n1, w)  # type: ignore[arg-type]
        self.assertIsNotNone(hit)
        assert hit is not None
        self.assertEqual(hit.symbol_span, 10)

    def test_nine_plus_nine_is_eighteen(self) -> None:
        reg = SymbolPromotionRegistry()
        p1 = reg.promote("the").prime
        p2 = reg.promote("cat").prime
        n1 = _nine_span_node(p1 + 10_000, "aaa")
        n2 = _nine_span_node(p2 + 10_000, "bbb")
        word_reg = WordPromotionRegistry()
        w = probe_pair_meet(n1, n2)
        hit = word_reg.promote_pair(n1, n2, w) if w else None
        self.assertIsNotNone(hit)
        assert hit is not None
        self.assertEqual(hit.symbol_span, 18)

    def test_nine_triple_is_twenty_seven(self) -> None:
        reg = SymbolPromotionRegistry()
        primes = [reg.promote(x).prime for x in ("the", "cat", "sat")]
        nodes = [_nine_span_node(p + 10_000, x) for p, x in zip(primes, ("aaa", "bbb", "ccc"))]
        word_reg = WordPromotionRegistry()
        w = probe_triple_meet(*nodes)
        hit = word_reg.promote_triple(*nodes, w) if w else None
        self.assertIsNotNone(hit)
        assert hit is not None
        self.assertEqual(hit.symbol_span, 27)


class TestSpanBounds(unittest.TestCase):
    def test_pair_rejects_below_ten(self) -> None:
        reg = SymbolPromotionRegistry()
        a = _one_span_node(reg.promote("a").prime)
        b = _one_span_node(reg.promote("b").prime)
        word_reg = WordPromotionRegistry()
        self.assertIsNone(word_reg.promote_pair(a, b, probe_pair_meet(a, b)))  # type: ignore[arg-type]

    def test_pair_rejects_above_eighteen(self) -> None:
        n9a = _nine_span_node(50000, "a")
        n9b = _nine_span_node(50002, "b")
        n2 = MeetNode("xy", 50004, 2, 2)
        word_reg = WordPromotionRegistry()
        # 9+2=11 ok; 9+9=18 ok; 9+9+? - test 9+9+1 invalid for pair 19
        fake_w = probe_pair_meet(n9a, MeetNode("big", 60000, 10, 4))
        if fake_w:
            self.assertIsNone(word_reg.promote_pair(n9a, MeetNode("big", 60000, 10, 4), fake_w))


class TestCorrelateWords(unittest.TestCase):
    def test_correlate_from_branch(self) -> None:
        l2 = SymbolPromotionRegistry()
        for sw in ("t", "h", "e", "the", "cat", "sat", "on"):
            l2.record_frequency(sw, 1)
            l2.promote(sw)
        comp = branch_meets(l2, max_rounds=1, min_frequency=1, max_composites_per_round=200)
        words = correlate_words(comp, max_new_pairs=100, max_new_triples=50)
        spans = {w.symbol_span for w in words.words.values()}
        self.assertTrue(any(MIN_WORD_SPAN_2WAY <= s <= MAX_WORD_SPAN_2WAY for s in spans))

    def test_word_seeds_cap_at_nine(self) -> None:
        l2 = SymbolPromotionRegistry()
        l2.promote("the")
        comp = branch_meets(l2, min_frequency=0, max_composites_per_round=50)
        seeds = word_seed_nodes(comp)
        self.assertTrue(all(n.symbol_span <= 9 for n in seeds))


if __name__ == "__main__":
    unittest.main()
