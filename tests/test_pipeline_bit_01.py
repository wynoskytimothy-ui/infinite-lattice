"""BIT 1 gate — word → SpacetimeCell matches hub formula coords."""
import unittest

from aethos_lattice import BranchKind, LatticeId
from aethos_physics import SpacetimeCell
from aethos_promotion import LatticeTier
from aethos_token_processor import TokenProcessor
from diagnose_corpus import SMALL_CORPUS
from pipeline.bit_01_word_cell import (
    DEFAULT_ANCHOR_N,
    branch_kind_to_k,
    chain_for_lattice_cell,
    four_branch_cells,
    hub_formula_coord,
    spacetime_cell_at_branch,
    verify_bit01_gate,
    verify_bit01_rotation_gate,
    word_to_spacetime_cell,
)


class TestBit01WordCell(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pipe = TokenProcessor()
        cls.pipe.ingest(*SMALL_CORPUS)
        cls.registry = cls.pipe.registry

    def test_chain_matches_lattice_address(self):
        for w in ("apple", "phone", "technical", "fruit"):
            chain = chain_for_lattice_cell(self.registry, w)
            tok = self.registry.resolve_token(w)
            if tok.intersection_only:
                expected = tuple(sorted(set(tok.parent_primes)))
            else:
                expected = tuple(sorted(set(tok.parent_primes + (tok.prime,))))
            self.assertEqual(chain, expected)

    def test_triple_witness_cell(self):
        cell = SpacetimeCell.at((3, 5, 7), 5, BranchKind.VA1, 1)
        self.assertEqual(cell.z, complex(12, 5))
        self.assertEqual(cell.zeta, 15.0)

    def test_word_cell_z_matches_hub(self):
        for w in sorted(self.registry.word_counts.keys())[:50]:
            cell = word_to_spacetime_cell(self.registry, w, n=DEFAULT_ANCHOR_N)
            x, y, z = hub_formula_coord(self.registry, w, n=DEFAULT_ANCHOR_N)
            self.assertAlmostEqual(cell.z.real, x, places=9)
            self.assertAlmostEqual(cell.z.imag, y, places=9)
            self.assertAlmostEqual(cell.zeta, z, places=9)

    def test_gate_passes_on_corpus_vocab(self):
        passed, total, failures = verify_bit01_gate(
            self.registry,
            max_words=100,
        )
        self.assertEqual(failures, [], failures)
        self.assertEqual(passed, total)
        self.assertGreater(total, 10)

    def test_default_anchor_n_is_seven(self):
        self.assertEqual(DEFAULT_ANCHOR_N, 7)

    def test_cell_carries_chain_metadata(self):
        w = "apple"
        cell = word_to_spacetime_cell(self.registry, w)
        self.assertEqual(cell.chain, chain_for_lattice_cell(self.registry, w))
        self.assertEqual(cell.branch, BranchKind.VA1)
        self.assertEqual(cell.wing, 1)
        self.assertEqual(cell.n, float(DEFAULT_ANCHOR_N))

    def test_l01_wing_matches_formula_coord(self):
        w = "phone"
        chain = chain_for_lattice_cell(self.registry, w)
        cell = word_to_spacetime_cell(self.registry, w)
        coord = self.registry.lattice_address(
            w, LatticeTier.L3_WORD, DEFAULT_ANCHOR_N, LatticeId.L01
        )
        self.assertEqual((cell.z.real, cell.z.imag, cell.zeta), coord)

    def test_rotation_gate_passes(self):
        ok, failures = verify_bit01_rotation_gate()
        self.assertTrue(ok, failures)

    def test_four_branch_cells_distinct(self):
        cells = four_branch_cells((3, 5, 7), 5.0)
        self.assertEqual(len(cells), 4)
        self.assertEqual(len({(c.z, c.zeta) for c in cells}), 4)

    def test_va1_branch_rotation_matches_at(self):
        chain = (3, 5, 7)
        legacy = SpacetimeCell.at(chain, 5, BranchKind.VA1, 1)
        rotated = spacetime_cell_at_branch(chain, 5, BranchKind.VA1, 1)
        self.assertEqual(rotated.z, legacy.z)
        self.assertEqual(rotated.zeta, legacy.zeta)

    def test_non_va1_uses_rotation_index(self):
        cell = spacetime_cell_at_branch((3, 5, 7), 5, BranchKind.VA3, 1)
        self.assertEqual(branch_kind_to_k(cell.branch), 2)


if __name__ == "__main__":
    unittest.main()
