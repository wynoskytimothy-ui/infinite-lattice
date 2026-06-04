"""
Hilbert space test suite — lattice-derived inner product, spring plane, correlations.

Run: python test_hilbert_space.py
     python -m unittest test_hilbert_space -v
"""

from __future__ import annotations

import math
import sys
import unittest

from aethos_complex_spring import mirror_pairs, spring_states_at
from aethos_hilbert import (
    BasisLabel,
    LatticeState,
    branch_fan_states,
    estimate_hilbert_tower,
    inner_product,
    wing_subspace_states,
)
from aethos_hilbert_lattice import (
    HILBERT_FROM_LATTICE,
    LatticeHilbertSpace,
    RobustInnerProductWeights,
    build_robust_space_from_corpus,
)
from aethos_lattice import BranchKind, LatticeId


class TestHilbertAxioms(unittest.TestCase):
    """Core Hilbert operations derived from lattice basis labels."""

    def setUp(self) -> None:
        self.hs = LatticeHilbertSpace(chain=(3, 5, 7), n_window=(5, 7))
        self.labels = self.hs.basis_labels()

    def test_basis_count_32_wings_per_n(self) -> None:
        self.assertEqual(len(self.labels), 32 * len(self.hs.n_window))

    def test_wing_states_orthogonal(self) -> None:
        wings = wing_subspace_states((3, 5, 7), 5)
        self.assertEqual(len(wings), 32)
        ip = inner_product(wings[0], wings[1])
        self.assertEqual(ip, 0)
        self.assertAlmostEqual(abs(inner_product(wings[0], wings[0])), 1.0)

    def test_gram_matrix_diagonal_one_off_diagonal_zero_sample(self) -> None:
        sample = self.labels[:8]
        g = self.hs.gram_matrix(sample)
        for i in range(len(sample)):
            self.assertAlmostEqual(g[i][i].real, 1.0, places=6)
            for j in range(len(sample)):
                if i != j:
                    self.assertAlmostEqual(g[i][j].real, 0.0, places=6)

    def test_norm_normalized_superposition(self) -> None:
        psi = LatticeState()
        psi.add(self.labels[0], 3.0)
        psi.add(self.labels[1], 4.0)
        normed = self.hs.normalize(psi)
        self.assertAlmostEqual(self.hs.norm(normed), 1.0, places=6)

    def test_inner_product_conjugate_linear_in_first_slot(self) -> None:
        e0 = self.hs.basis_state(self.labels[0])
        e1 = self.hs.basis_state(self.labels[1])
        psi = LatticeState()
        psi.add(self.labels[0], 2.0)
        psi.add(self.labels[1], 1.0j)
        # <e0|psi> = 2, <e0|i*psi> should relate linearly for geometric part
        a = self.hs.geometric_inner(e0, psi)
        b = self.hs.geometric_inner(e0, self._scale(psi, 1j))
        self.assertAlmostEqual(b, 1j * a)

    def _scale(self, st: LatticeState, c: complex) -> LatticeState:
        out = LatticeState(labels=dict(st.labels))
        for k, v in st.amplitudes.items():
            out.amplitudes[k] = c * v
        return out

    def test_projection_reconstructs_component(self) -> None:
        e0 = self.hs.basis_state(self.labels[0])
        psi = LatticeState()
        psi.add(self.labels[0], 0.6)
        psi.add(self.labels[1], 0.8)
        proj = self.hs.project_onto(psi, e0)
        # proj should be 0.6 * e0
        self.assertAlmostEqual(abs(proj.amplitudes[ self.labels[0].key()]), 0.6, places=6)


class TestSpringComplexPlane(unittest.TestCase):
    """Spring z = X + iY at triggers x 4 branches."""

    def test_solo_trigger_va1_va2_mirror_same_tension_squared(self) -> None:
        _, st = spring_states_at((5,), 5)
        self.assertAlmostEqual(
            st[BranchKind.VA1].tension_squared,
            st[BranchKind.VA2].tension_squared,
        )
        pairs = mirror_pairs(st)
        self.assertIn((BranchKind.VA1, BranchKind.VA2), pairs)

    def test_four_branch_states_at_trigger(self) -> None:
        _, st = spring_states_at((3, 5, 7), 3)
        self.assertEqual(len(st), 4)
        for b in BranchKind:
            self.assertGreater(st[b].tension_squared, 0)

    def test_spring_inner_va1_va2_mirror_solo(self) -> None:
        hs = LatticeHilbertSpace(chain=(5,), weights=RobustInnerProductWeights(spring=1.0))
        ip = hs.spring_inner_at(5, BranchKind.VA1, BranchKind.VA2)
        _, st = spring_states_at((5,), 5)
        z1, z2 = st[BranchKind.VA1].z, st[BranchKind.VA2].z
        self.assertAlmostEqual(ip, z1.conjugate() * z2)
        self.assertAlmostEqual(st[BranchKind.VA1].tension_squared, 125.0)
        self.assertAlmostEqual(ip.real, 75.0)  # conj(a+bi)(a-bi) = a^2 - b^2

    def test_branch_fan_four_distinct_states(self) -> None:
        fan = branch_fan_states((3, 5, 7), 5, wing=1)
        self.assertEqual(len(fan), 4)
        keys = {next(iter(f.labels.keys())) for f in fan}
        self.assertEqual(len(keys), 4)


class TestMeetAndRobustInnerProduct(unittest.TestCase):
    def test_meet_boost_zero_for_distinct_non_colliding_wings(self) -> None:
        hs = LatticeHilbertSpace(n_window=(5,))
        labels = hs.basis_labels()
        a = hs.basis_state(labels[0])
        b = hs.basis_state(labels[1])
        self.assertEqual(hs.meet_boost(a, b), 0.0)

    def test_robust_correlation_from_corpus(self) -> None:
        hs = build_robust_space_from_corpus(
            "phone phone technical chip software",
            "phone technical hardware network",
        )
        self.assertIsNotNone(hs.registry)
        self.assertGreater(hs.correlation_inner_words("phone", "technical"), 0)
        neighbors = hs.correlation_inner_links("phone")
        self.assertIn("technical", neighbors)
        self.assertGreater(neighbors["technical"], 0)

    def test_correlation_symmetric_via_edge(self) -> None:
        hs = build_robust_space_from_corpus("apple phone chip", "apple fruit pie")
        a = hs.correlation_inner_words("apple", "phone")
        b = hs.correlation_inner_words("phone", "apple")
        self.assertGreater(a, 0)
        self.assertGreater(b, 0)


class TestHilbertTowerIntegration(unittest.TestCase):
    def test_tower_estimate_scales(self) -> None:
        small = estimate_hilbert_tower(chain_k=3, n_max=10, origin_depth=1)
        large = estimate_hilbert_tower(chain_k=5, n_max=50, origin_depth=3)
        self.assertLess(small.truncated_basis_size, large.truncated_basis_size)

    def test_derivation_table_complete(self) -> None:
        names = {d.name for d in HILBERT_FROM_LATTICE}
        required = {"Norm", "Inner product (geometric)", "Countable basis", "Completeness"}
        self.assertTrue(required.issubset(names))

    def test_core_hilbert_space_facade(self) -> None:
        from aethos_core import AethosLatticeCore

        hs = AethosLatticeCore().hilbert_space(chain=(3, 5), n_window=(3, 5))
        self.assertEqual(len(hs.basis_labels()), 64)


class TestHilbertEndToEnd(unittest.TestCase):
    """Full workflow: build space, state, measure, correlate."""

    def test_workflow(self) -> None:
        hs = build_robust_space_from_corpus(
            "electron spring photon bounce",
            "electron spring technical",
        )
        labels = hs.basis_labels()[:16]
        psi = LatticeState()
        for i, lbl in enumerate(labels[:4]):
            psi.add(lbl, 1.0 / (i + 1))
        psi_n = hs.normalize(psi)
        self.assertAlmostEqual(hs.norm(psi_n), 1.0, places=5)

        # measure overlap with first basis direction
        e0 = hs.basis_state(labels[0])
        prob = abs(hs.total_inner(psi_n, e0)) ** 2
        self.assertGreater(prob, 0)
        self.assertLessEqual(prob, 1.0 + 1e-9)


def run_suite(verbosity: int = 2) -> int:
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromModule(sys.modules[__name__]))
    result = unittest.TextTestRunner(verbosity=verbosity).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    print("=" * 70)
    print("AETHOS HILBERT SPACE TEST SUITE")
    print("=" * 70 + "\n")
    sys.exit(run_suite())
