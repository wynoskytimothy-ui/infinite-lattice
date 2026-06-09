"""OOV query lattice — structural subword routing."""

from __future__ import annotations

import unittest

from aethos_query_oov import (
    build_query_lattice_node,
    morph_subword_pieces,
    structural_anchor_words,
    word_needs_oov_build,
)
from aethos_symbol_knowledge import SymbolKnowledgeIndex
from pipeline.bit_12_symbol_plane_index import (
    build_symbol_plane_index,
    query_symbol_plane_keys,
    route_symbol_plane_candidates,
)


class TestQueryOOV(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.corpus = {
            "d1": "biomaterial scaffold inductive properties dimensional matrix",
            "d2": "protein kinase study cells treatment cancer patients",
            "d3": "material science polymer composite tensile strength",
        }
        cls.knowledge = SymbolKnowledgeIndex.build_from_corpus(
            cls.corpus, dataset="oov_test",
        )
        cls.plane = build_symbol_plane_index(cls.knowledge)

    def test_oov_token_detected(self) -> None:
        self.assertTrue(word_needs_oov_build(self.knowledge, self.plane, "biomaterials"))
        self.assertFalse(word_needs_oov_build(self.knowledge, self.plane, "material"))

    def test_morph_pieces_find_material(self) -> None:
        pieces = morph_subword_pieces(self.knowledge, "biomaterials")
        anchors = structural_anchor_words(self.knowledge, "biomaterials", self.plane)
        self.assertIn("material", anchors)

    def test_lattice_node_cached(self) -> None:
        node = self.knowledge.ensure_query_lattice("biomaterials", self.plane)
        self.assertGreater(len(node.icn_chain), 0)
        self.assertIn("material", node.anchors)
        self.assertIs(self.knowledge.ensure_query_lattice("biomaterials", self.plane), node)

    def test_oov_query_routes_to_material_doc(self) -> None:
        keys = query_symbol_plane_keys(
            self.knowledge,
            self.plane,
            ["biomaterials", "inductive"],
            max_keys=256,
        )
        self.assertGreater(len(keys), 0)
        route = route_symbol_plane_candidates(
            self.knowledge,
            self.plane,
            ["biomaterials", "inductive"],
            max_candidates=20,
        )
        self.assertIn("d1", route.doc_ids)


if __name__ == "__main__":
    unittest.main()
