"""Symbol prime 2-way swaps and 3-way triple equalization."""

from __future__ import annotations

import unittest

from aethos_intersection_nodes import MeetKind
from aethos_symbol_map import symbol_to_prime, text_icn_chain
from aethos_symbol_meets import (
    discover_pair_meet,
    discover_text_meets,
    discover_triple_meet,
)


class TestTwoWayMeets(unittest.TestCase):
    def test_th_solo_swap(self) -> None:
        chain = text_icn_chain("th")
        self.assertEqual(len(chain), 2)
        meet = discover_pair_meet(chain[0], chain[1])
        self.assertIsNotNone(meet)
        assert meet is not None
        self.assertEqual(meet.witness.kind, MeetKind.SOLO_SWAP)
        self.assertEqual(meet.left_symbol, "h")
        self.assertEqual(meet.right_symbol, "t")

    def test_text_discovery_two_way(self) -> None:
        d = discover_text_meets("th", grow_network=False)
        self.assertEqual(len(d.two_way), 1)
        self.assertEqual(len(d.three_way), 0)
        self.assertIsNotNone(d.locked_coord)


class TestThreeWayMeets(unittest.TestCase):
    def test_ing_triple_lock(self) -> None:
        chain = text_icn_chain("ing")
        self.assertEqual(len(chain), 3)
        meet = discover_triple_meet(*chain)
        self.assertIsNotNone(meet)
        assert meet is not None
        self.assertEqual(meet.witness.kind, MeetKind.TRIPLE)
        coords = {psi.coord for _, psi in meet.witnesses.values()}
        self.assertEqual(len(coords), 1)

    def test_the_triple_lock(self) -> None:
        d = discover_text_meets("the", grow_network=False)
        self.assertGreaterEqual(len(d.two_way), 3)  # C(3,2)
        self.assertEqual(len(d.three_way), 1)  # C(3,3)
        self.assertIsNotNone(d.composite)
        t, h, e = symbol_to_prime("t"), symbol_to_prime("h"), symbol_to_prime("e")
        triple = d.three_way[0]
        self.assertEqual(triple.primes, tuple(sorted((e, h, t))))

    def test_four_letter_word_multiple_triples(self) -> None:
        d = discover_text_meets("cats", grow_network=False)
        self.assertEqual(len(d.chain), 4)
        self.assertEqual(len(d.two_way), 6)  # C(4,2)
        self.assertEqual(len(d.three_way), 4)  # C(4,3)


if __name__ == "__main__":
    unittest.main()
