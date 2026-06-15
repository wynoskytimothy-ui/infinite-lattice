"""Concrete Plane P2 — unified morph piece extraction."""

from __future__ import annotations

import unittest

from aethos_symbol_entangle import find_morph_pieces
from aethos_symbol_knowledge import SymbolKnowledgeIndex
from aethos_symbol_morph_pieces import morph_pieces, morph_pieces_for_token
from aethos_query_oov import morph_subword_pieces
from aethos_rare_rank import morph_trigger_pieces


class TestMorphPiecesUnified(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.knowledge = SymbolKnowledgeIndex.load("scifact")

    def test_query_paths_agree(self) -> None:
        tokens = ["cellular", "cells", "diminished", "expression", "biomaterials"]
        for tok in tokens:
            a = set(morph_trigger_pieces(self.knowledge, tok))
            b = set(morph_subword_pieces(self.knowledge, tok))
            c = set(morph_pieces(self.knowledge, tok, mode="query"))
            self.assertEqual(a, b, tok)
            self.assertEqual(a, c, tok)

    def test_cellular_finds_cell_in_query_mode(self) -> None:
        pieces = morph_pieces(self.knowledge, "cellular", mode="query")
        self.assertIn("cell", pieces)

    def test_ingest_finds_catalog_tokens(self) -> None:
        morph = self.knowledge.morph
        if not morph.subwords and not morph.composites:
            self.skipTest("empty morph registry")
        sample = next(iter(morph.subwords or morph.composites))
        text = f"the {sample} pathway"
        found = find_morph_pieces(text, morph)
        self.assertIn(sample, found)

    def test_ingest_skips_embedded_only_hits(self) -> None:
        morph = self.knowledge.morph
        if "cell" not in morph.subwords:
            self.skipTest("cell subword missing")
        # cellular not necessarily in catalog — ingest should not substring-scan
        ingest = morph_pieces_for_token(morph, "cellular", mode="ingest")
        query = morph_pieces_for_token(
            morph, "cellular", vocab=self.knowledge.vocab, mode="query",
        )
        self.assertNotIn("cell", ingest)
        self.assertIn("cell", query)


if __name__ == "__main__":
    unittest.main()
