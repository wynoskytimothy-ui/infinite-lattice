"""Corpus 1/2/3-gram discovery, frequency, promotion."""

from __future__ import annotations

import unittest

from aethos_symbol_corpus import CorpusSubwordIndex, extract_ordered_subwords


class TestExtract(unittest.TestCase):
    def test_bigrams_from_word(self) -> None:
        sws = extract_ordered_subwords("cat", min_len=1, max_len=3)
        self.assertIn("ca", sws)
        self.assertIn("at", sws)
        self.assertIn("cat", sws)

    def test_len1_symbols(self) -> None:
        sws = extract_ordered_subwords("ab", min_len=1, max_len=1)
        self.assertEqual(sws, ["a", "b"])


class TestCorpusIndex(unittest.TestCase):
    def test_frequency_and_doc_inventory(self) -> None:
        idx = CorpusSubwordIndex()
        idx.ingest_corpus({
            "d1": "the cat",
            "d2": "the ether",
        })
        self.assertGreater(idx.counts.get("the", 0), 1)
        self.assertIn("the", idx.subwords_in_doc("d1"))
        self.assertIn("cat", idx.subwords_in_doc("d1"))
        by_len = idx.subwords_by_length_in_doc("d1")
        self.assertIn("t", by_len[1])
        self.assertIn("th", by_len[2])

    def test_promote_all_includes_trigram_siblings(self) -> None:
        idx = CorpusSubwordIndex()
        idx.ingest_text("d1", "the")
        idx.promote_all()
        self.assertEqual(len(idx.registry.by_length(3)), 6)
        self.assertIn("het", idx.registry.promoted)
        self.assertIn("eth", idx.registry.promoted)

    def test_promoted_for_doc(self) -> None:
        idx = CorpusSubwordIndex()
        idx.ingest_text("d1", "cat")
        promoted = idx.promoted_for_doc("d1")
        texts = {t.text for t in promoted}
        self.assertIn("cat", texts)
        self.assertIn("ca", texts)


if __name__ == "__main__":
    unittest.main()
