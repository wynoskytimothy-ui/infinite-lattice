"""Concrete Plane P1 — canonical morph chain decay."""

from __future__ import annotations

import unittest

from aethos_symbol_knowledge import SymbolKnowledgeIndex
from aethos_symbol_morph import (
    MorphRegistry,
    build_morph_registry,
    canonical_morph_chain_and_imag,
)
from pipeline.bit_12_symbol_plane_index import (
    symbol_word_chain,
    symbol_word_chain_query,
    symbol_word_imag,
    symbol_word_imag_query,
)


class TestCanonicalMorphChain(unittest.TestCase):
    def test_scifact_cell_family_shared_query_chain(self) -> None:
        knowledge = SymbolKnowledgeIndex.load("scifact")
        root = symbol_word_chain_query(knowledge, "cell")
        self.assertEqual(symbol_word_chain_query(knowledge, "cellular"), root)
        self.assertEqual(symbol_word_chain_query(knowledge, "cells"), root)
        self.assertNotEqual(
            symbol_word_chain_query(knowledge, "show"),
            symbol_word_chain_query(knowledge, "shows"),
        )
        self.assertNotEqual(symbol_word_chain(knowledge, "cellular"), root)

    def test_symbol_word_chain_uses_canonical_on_corpus(self) -> None:
        corpus = {
            "d1": "cellular cells express protein in the cell membrane",
        }
        knowledge = SymbolKnowledgeIndex.build_from_corpus(corpus, dataset="canon_test")
        if "cell" not in knowledge.morph.subwords:
            knowledge.morph.promote_morph_piece(
                "cell",
                parents=frozenset({"cell", "cells", "cellular"}),
            )
        root_chain = symbol_word_chain(knowledge, "cell")
        self.assertGreater(len(root_chain), 0)
        self.assertEqual(
            symbol_word_chain_query(knowledge, "cellular"),
            root_chain,
            "cellular query should decay to cell chain",
        )
        self.assertEqual(
            symbol_word_imag_query(knowledge, "cellular"),
            symbol_word_imag_query(knowledge, "cell"),
        )


if __name__ == "__main__":
    unittest.main()
