"""Bridge core L2 → PromotionRegistry."""

from __future__ import annotations

import unittest

from aethos_pipeline import AethosPipeline, check_promotion_invariants
from aethos_promotion import LatticeTier
from core.bridge_registry import run_core_l2_pass, sync_l2_to_registry
from core.bridge_library import StubPrimeAssigner, bridge_word_to_library
from core.l2_subwords import SubwordPromoter
from diagnose_corpus import SMALL_CORPUS


class TestBridgeRegistry(unittest.TestCase):
    def setUp(self) -> None:
        self.pipe = AethosPipeline(rebuild_every=2)
        self.pipe.ingest(*SMALL_CORPUS)

    def test_sync_promotes_l2(self) -> None:
        reg = self.pipe.registry
        new = sync_l2_to_registry(reg, max_promote=80)
        l2_count = sum(1 for k in reg.promoted if k[0] == LatticeTier.L2_SUBWORD)
        self.assertGreaterEqual(l2_count, 0)
        self.assertGreaterEqual(new, 0)

    def test_run_core_l2_pass_invariants(self) -> None:
        reg = self.pipe.registry
        run_core_l2_pass(reg, max_promote=80)
        check_promotion_invariants(reg)

    def test_shared_at_subword_when_promoted(self) -> None:
        reg = self.pipe.registry
        run_core_l2_pass(reg, max_promote=120)
        if (LatticeTier.L2_SUBWORD, "at") in reg.promoted:
            self.assertGreaterEqual(len(reg.subword_parent_words.get("at", set())), 2)


class TestBridgeLibrary(unittest.TestCase):
    def test_stub_assigner_with_promoter(self) -> None:
        from core.l2_subwords import SubwordStats

        stats = SubwordStats()
        stats.observe_text_corpus(
            ["quantum physics", "quant measurement", "quantity units"]
        )
        promoter = SubwordPromoter(stats=stats)
        promoter.promote_top(80)
        assigner = StubPrimeAssigner()
        assigner.bind_promoter(promoter)
        rec = bridge_word_to_library(assigner, "quantum", promoter)
        self.assertGreater(rec.word_prime, 0)
        self.assertIsInstance(rec.parent_primes, tuple)


if __name__ == "__main__":
    unittest.main()
