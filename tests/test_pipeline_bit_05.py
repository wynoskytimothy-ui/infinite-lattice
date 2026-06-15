"""BIT 5 gate — |z| band template and VA1+VA2 observable."""
import unittest

from aethos_lattice import BranchKind
from aethos_physics import SpacetimeCell
from aethos_token_processor import TokenProcessor
from aethos_hub_signature import build_hub_signature
from diagnose_corpus import SMALL_CORPUS
from pipeline.bit_05_z_band import (
    NUM_BANDS,
    band_profile_for_cell,
    band_profile_for_word,
    branch_moduli,
    verify_bit05_gate,
    wing_band_map,
    z_obs_va1_va2,
)


class TestBit05ZBand(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pipe = TokenProcessor()
        cls.pipe.ingest(*SMALL_CORPUS)
        cls.registry = cls.pipe.registry

    def test_gate_triple_witness(self):
        ok, failures = verify_bit05_gate(chain=(3, 5, 7), n=5)
        self.assertTrue(ok, failures)

    def test_four_distinct_branch_moduli(self):
        moduli = branch_moduli((3, 5, 7), 5, wing=1)
        self.assertEqual(len(moduli), NUM_BANDS)
        self.assertEqual(len(set(moduli)), NUM_BANDS)

    def test_wing_band_partition(self):
        bands = wing_band_map((3, 5, 7), 5)
        self.assertEqual(len(bands), 32)
        counts = {}
        for bid in bands.values():
            counts[bid] = counts.get(bid, 0) + 1
        self.assertEqual(counts, {0: 8, 1: 8, 2: 8, 3: 8})

    def test_z_obs_is_real(self):
        z_obs = z_obs_va1_va2((3, 5, 7), 5)
        self.assertAlmostEqual(z_obs, 30.0)
        va1 = SpacetimeCell.at((3, 5, 7), 5, BranchKind.VA1, 1)
        va2 = SpacetimeCell.at((3, 5, 7), 5, BranchKind.VA2, 1)
        self.assertAlmostEqual((va1.z + va2.z).imag, 0.0)

    def test_word_band_in_range(self):
        for w in ("phone", "apple", "technical"):
            prof = band_profile_for_word(self.registry, w)
            self.assertIn(prof.band_id, range(NUM_BANDS))
            self.assertGreater(prof.z_modulus, 0.0)

    def test_hub_entry_carries_band(self):
        sig = build_hub_signature(
            "d0",
            SMALL_CORPUS[0],
            self.registry,
            top_k=8,
        )
        for entry in sig.hubs.values():
            self.assertIn(entry.band_id, range(NUM_BANDS))

    def test_va1_primary_cell_is_band_zero_at_triple(self):
        cell = SpacetimeCell.at((3, 5, 7), 5, BranchKind.VA1, 1)
        prof = band_profile_for_cell(cell)
        self.assertEqual(prof.band_id, 0)


if __name__ == "__main__":
    unittest.main()
