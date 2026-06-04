"""Regression tests for small-corpus semantic fixes."""

from __future__ import annotations

import unittest

from aethos_pipeline import AethosPipeline
from aethos_promotion import is_stopword
from aethos_words import encode_word_at_site, word_to_order
from diagnose_corpus import SMALL_CORPUS


class TestCorpusFixes(unittest.TestCase):
    def setUp(self) -> None:
        self.pipe = AethosPipeline(rebuild_every=2)
        self.pipe.ingest(*SMALL_CORPUS)

    def test_stopwords_stay_intersection_only(self) -> None:
        for w in ("the", "a", "and", "with", "from"):
            self.assertTrue(is_stopword(w))
            self.assertTrue(self.pipe.registry.is_intersection_only(w), w)

    def test_apple_disambiguates_on_small_corpus(self) -> None:
        tech = self.pipe.resolve("apple", ["phone", "chip"])
        food = self.pipe.resolve("apple", ["fruit", "pie"])
        self.assertNotEqual(tech["cluster_id"], food["cluster_id"])

    def test_phone_ignores_unrelated_food_context(self) -> None:
        r = self.pipe.resolve("phone", ["fruit", "pie"])
        self.assertEqual(r["cluster_id"], "theme_phone")

    def test_apple_overlay_with_duplicate_letters(self) -> None:
        ov = self.pipe.semantic_overlay("apple")
        self.assertTrue(ov.registry_equals_codec_local)
        self.assertEqual(ov.tier, "dedicated_l3")

    def test_apple_word_dot_roundtrip(self) -> None:
        from aethos_words import decode_word

        dot = encode_word_at_site("apple")
        self.assertEqual(decode_word(dot), "apple")
        self.assertEqual(len(word_to_order("apple")), 5)
        self.assertEqual(len(set(word_to_order("apple"))), 4)

    def test_cat_routes_to_own_cluster_without_context(self) -> None:
        r = self.pipe.resolve("cat")
        self.assertEqual(r["cluster_id"], "theme_cat")
        self.assertGreater(r["cluster_score"], 0.1)

    def test_oov_word_returns_no_cluster(self) -> None:
        r = self.pipe.resolve("zebra")
        self.assertEqual(r["cluster_id"], "")
        self.assertEqual(r["cluster_score"], 0.0)

    def test_oov_with_unknown_context_stays_unknown(self) -> None:
        r = self.pipe.resolve("zebra", ["animal"])
        self.assertEqual(r["cluster_id"], "")
        self.assertEqual(r["cluster_score"], 0.0)

    def test_stale_cluster_ids_pruned_after_discover(self) -> None:
        active = set(self.pipe.reader.cluster_hubs.keys())
        self.assertEqual(set(self.pipe.reader.cross.categories.keys()), active)


if __name__ == "__main__":
    unittest.main()
