"""Branching meets — 4/5/6/9-way composite promotion."""

from __future__ import annotations

import unittest

from aethos_symbol_composite import (
    CompositePromotionRegistry,
    branch_meets,
    probe_pair_meet,
    probe_triple_meet,
    span_table_for_trigram_base,
    MeetNode,
)
from aethos_symbol_promotion import SymbolPromotionRegistry


class TestPairSpans(unittest.TestCase):
    def test_three_plus_one_is_four_way(self) -> None:
        reg = SymbolPromotionRegistry()
        reg.promote("the")
        reg.promote("e")
        table = span_table_for_trigram_base(reg, "the")
        row = table["pair_meets"][0]
        self.assertEqual(row["span"], 4)
        self.assertEqual(row["way"], "4-way")
        self.assertTrue(row["meet_ok"])

    def test_three_plus_two_is_five_way(self) -> None:
        reg = SymbolPromotionRegistry()
        reg.promote("the")
        reg.promote("th")
        table = span_table_for_trigram_base(reg, "the")
        self.assertEqual(table["pair_meets"][1]["span"], 5)
        self.assertEqual(table["pair_meets"][1]["way"], "5-way")

    def test_three_plus_three_is_six_way(self) -> None:
        reg = SymbolPromotionRegistry()
        reg.promote("the")
        reg.promote("cat")
        table = span_table_for_trigram_base(reg, "the")
        self.assertEqual(table["pair_meets"][2]["span"], 6)
        self.assertEqual(table["pair_meets"][2]["way"], "6-way")


class TestNineWay(unittest.TestCase):
    def test_triple_trigrams_nine_span(self) -> None:
        reg = SymbolPromotionRegistry()
        table = span_table_for_trigram_base(reg, "the")
        n9 = table["nine_way"]
        self.assertEqual(n9["span"], 9)
        self.assertTrue(n9["meet_ok"])
        self.assertEqual(n9["composite_text"], "thecatsat")
        self.assertIsNotNone(n9["composite_prime"])


class TestBranching(unittest.TestCase):
    def test_branch_finds_composites(self) -> None:
        reg = SymbolPromotionRegistry()
        for sw in ("t", "h", "e", "the", "cat"):
            reg.promote(sw)
        comp = branch_meets(reg, max_rounds=2)
        self.assertGreater(len(comp.composites), 0)
        spans = {c.symbol_span for c in comp.composites.values()}
        self.assertTrue(any(4 <= s <= 9 for s in spans))

    def test_composite_can_rebranch(self) -> None:
        reg = SymbolPromotionRegistry()
        reg.promote("the")
        reg.promote("cat")
        reg.promote("e")
        comp_reg = branch_meets(reg, max_rounds=3)
        primes_round1 = len(comp_reg.composites)
        self.assertGreaterEqual(primes_round1, 1)


class TestMeetProbes(unittest.TestCase):
    def test_distinct_l2_primes_solo_swap(self) -> None:
        reg = SymbolPromotionRegistry()
        a = reg.promote("the")
        b = reg.promote("cat")
        na = MeetNode(a.text, a.prime, a.length, 2)
        nb = MeetNode(b.text, b.prime, b.length, 2)
        w = probe_pair_meet(na, nb)
        self.assertIsNotNone(w)

    def test_triple_meet_three_promoted(self) -> None:
        reg = SymbolPromotionRegistry()
        nodes = [
            MeetNode(reg.promote("the").text, reg.promote("the").prime, 3, 2),
            MeetNode(reg.promote("cat").text, reg.promote("cat").prime, 3, 2),
            MeetNode(reg.promote("sat").text, reg.promote("sat").prime, 3, 2),
        ]
        w = probe_triple_meet(*nodes)
        self.assertIsNotNone(w)


if __name__ == "__main__":
    unittest.main()
