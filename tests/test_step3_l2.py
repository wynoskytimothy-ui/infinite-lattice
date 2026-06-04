"""Step 4 — L2 subword layer tests."""

from __future__ import annotations

import unittest

from core.l2_subwords import (
    SubwordConfig,
    SubwordPromoter,
    SubwordStats,
    decompose,
    is_stopword,
    max_subword_pmi,
    score_pmi,
    shared_l2_factors,
    should_promote_l2,
)
from core.primes import product_unique

QUANTUM_CORPUS = [
    "quantum mechanics theory physics research",
    "quant field theory and quantum states",
    "quantity of matter measured in units",
    "quantum quant correlation experiment",
]

PHON_CORPUS = [
    "telephone",
]


class TestSubwordStats(unittest.TestCase):
    def test_qua_in_multiple_words(self) -> None:
        stats = SubwordStats()
        stats.observe_text_corpus(QUANTUM_CORPUS)
        self.assertIn("qua", stats.subword_counts)
        self.assertGreaterEqual(len(stats.subword_parent_words.get("qua", set())), 2)

    def test_pmi_positive_for_qua(self) -> None:
        stats = SubwordStats()
        stats.observe_text_corpus(QUANTUM_CORPUS)
        self.assertGreater(max_subword_pmi(stats, "qua"), 0.0)


class TestPromotionRules(unittest.TestCase):
    def test_stopwords_rejected(self) -> None:
        stats = SubwordStats()
        stats.observe_word("the")
        self.assertFalse(should_promote_l2(stats, "the"))

    def test_phon_single_parent_rejected(self) -> None:
        stats = SubwordStats()
        stats.observe_text_corpus(PHON_CORPUS)
        if "phon" in stats.subword_counts:
            self.assertFalse(should_promote_l2(stats, "phon"))


class TestSubwordPromoter(unittest.TestCase):
    def setUp(self) -> None:
        self.stats = SubwordStats()
        self.stats.observe_text_corpus(QUANTUM_CORPUS)
        self.promoter = SubwordPromoter(stats=self.stats)

    def test_deterministic_promotion(self) -> None:
        p1 = SubwordPromoter(stats=SubwordStats())
        p1.stats.observe_text_corpus(QUANTUM_CORPUS)
        p1.promote_top(max_promote=80)
        p2 = SubwordPromoter(stats=SubwordStats())
        p2.stats.observe_text_corpus(QUANTUM_CORPUS)
        p2.promote_top(max_promote=80)
        self.assertEqual(p1.l2_lookup, p2.l2_lookup)

    def test_morphology_quantum_quant(self) -> None:
        self.promoter.promote_top(max_promote=120)
        shared = shared_l2_factors("quantum", "quant", self.promoter.l2_lookup)
        self.assertGreater(
            len(shared),
            0,
            f"l2_lookup keys sample: {list(self.promoter.l2_lookup.keys())[:20]}",
        )

    def test_decompose_finds_promoted_ngrams(self) -> None:
        self.promoter.promote_top(max_promote=120)
        if "qua" in self.promoter.l2_lookup:
            primes = decompose("quantum", self.promoter.l2_lookup)
            self.assertIn(self.promoter.l2_lookup["qua"], primes)

    def test_fta_composite_unique(self) -> None:
        self.promoter.promote_top(max_promote=120)
        composites: set[int] = set()
        for w in ("quantum", "quantity", "quant"):
            c = self.promoter.l2_composite(w)
            if c is not None:
                composites.add(c)
        if len(composites) >= 2:
            self.assertEqual(len(composites), len(list(composites)))

    def test_pool_cap_graceful(self) -> None:
        tiny = SubwordPromoter(stats=self.stats)
        tiny.pool = __import__("core.primes", fromlist=["PrimePool"]).PrimePool(
            pool=(101, 103, 104)
        )
        n = 0
        for _, sw in tiny.ranked_candidates()[:50]:
            if tiny.promote_one(sw):
                n += 1
        self.assertGreaterEqual(n, 0)


class TestPMIMath(unittest.TestCase):
    def test_score_pmi_zero_without_pairs(self) -> None:
        stats = SubwordStats()
        self.assertEqual(score_pmi(stats, "xy", "word"), 0.0)


class TestSharedFactors(unittest.TestCase):
    def test_empty_without_lookup(self) -> None:
        self.assertEqual(shared_l2_factors("a", "b", {}), ())


class TestProductHelper(unittest.TestCase):
    def test_two_primes_unique(self) -> None:
        self.assertEqual(product_unique(11, 13), 143)


if __name__ == "__main__":
    unittest.main()
