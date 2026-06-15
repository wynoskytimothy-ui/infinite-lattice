"""Tests for SpacetimeCell wiring: collapse, entanglement, attractor index."""
import unittest

from aethos_attractor_index import (
    CorpusAttractorIndex,
    cell_attractor_key,
    spring_attractor_key,
)
from aethos_intersection_nodes import IntersectionNetwork, find_entangled_meet_pairs
from aethos_lattice import BranchKind
from aethos_physics import (
    MeasurementCollapse,
    SpacetimeCell,
    apply_sg_collapse,
    compress_spacetime_cell,
)


class TestMeasurementCollapse(unittest.TestCase):
    def test_im_suppressed_at_strong_lambda(self):
        cell = SpacetimeCell.at((3, 5, 7), 5, BranchKind.VA1)
        col = compress_spacetime_cell(cell, lambda_n=10.0)
        self.assertIsInstance(col, MeasurementCollapse)
        self.assertEqual(col.regime, "hard")
        self.assertAlmostEqual(col.post.im, 0.0, places=3)
        self.assertAlmostEqual(col.post.re, 30.0, places=3)

    def test_zeta_pinned_on_hard(self):
        cell = SpacetimeCell.at((3, 5, 7), 5, BranchKind.VA1)
        col = compress_spacetime_cell(cell, lambda_n=10.0)
        self.assertEqual(col.regime, "hard")
        self.assertEqual(col.post.zeta, cell.zeta)

    def test_sg_collapse_runs(self):
        cell = SpacetimeCell.at((5,), 5, BranchKind.VA1)
        col = apply_sg_collapse(cell)
        self.assertGreater(col.lambda_n, 0.0)


class TestEntanglementMeets(unittest.TestCase):
    def test_triple_and_swap_share_spring(self):
        net = IntersectionNetwork()
        net.follow_and_branch([3, 5, 7, 15, 30, 210], max_nodes=64)
        pairs = net.entangled_pairs()
        self.assertGreater(len(pairs), 0)
        p = pairs[0]
        self.assertTrue(p.spring_match)
        self.assertTrue(p.path_distinct)

    def test_witness_spacetime_cell(self):
        net = IntersectionNetwork()
        w = net.probe_triple(3, 5, 7)
        self.assertIsNotNone(w)
        cell = w.spacetime_cell(chain=(3, 5, 7))
        self.assertEqual(cell.z, complex(12, 5))


class TestAttractorIndex(unittest.TestCase):
    def test_same_zeta_bucket(self):
        cell = SpacetimeCell.promote_witness((3, 5, 7), (3, 5))
        key = cell_attractor_key(cell)
        self.assertEqual(key, spring_attractor_key(complex(12, 5), 15.0))

    def test_query_finds_docs(self):
        cell = SpacetimeCell.promote_witness((3, 5, 7), (3, 5))
        key = cell_attractor_key(cell)
        idx = CorpusAttractorIndex()
        idx.add("doc1", key, "alpha")
        idx.add("doc2", (0, 0, 0), "beta")
        hits = idx.query_by_cell(cell)
        self.assertEqual(hits, ["doc1"])

    def test_jaccard_rank(self):
        idx = CorpusAttractorIndex()
        k1 = (12, 5, 15)
        k2 = (8, 3, 8)
        idx.add("a", k1)
        idx.add("a", k2)
        idx.add("b", k2)
        ranked = idx.rank_docs_by_overlap({k1, k2})
        self.assertEqual(ranked[0][1], "a")
        self.assertGreater(ranked[0][0], ranked[1][0])


if __name__ == "__main__":
    unittest.main()
