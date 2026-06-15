"""Critical-line pin wire — regen coords match lattice hub path."""
import unittest

from aethos_hub_signature import (
    build_hub_signature,
    build_query_profile,
    materialize_lazy_hub_wings,
    score_document,
)
from aethos_hub_wire import CriticalLinePin, hub_coord_from_word, leg_sum_im_led
from aethos_tokenize import tokenize_words
from aethos_token_processor import TokenProcessor
from diagnose_corpus import SMALL_CORPUS


class TestHubWire(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pipe = TokenProcessor()
        cls.pipe.ingest(*SMALL_CORPUS)
        cls.registry = cls.pipe.registry

    def test_pin_regenerates_l01_coord(self):
        sig_legacy = build_hub_signature(
            "d0", SMALL_CORPUS[0], self.registry, top_k=8, use_pin_wire=False
        )
        sig_pin = build_hub_signature(
            "d0", SMALL_CORPUS[0], self.registry, top_k=8, use_pin_wire=True
        )
        self.assertEqual(set(sig_legacy.hubs), set(sig_pin.hubs))
        for word, leg in sig_legacy.hubs.items():
            pin_entry = sig_pin.hubs[word]
            self.assertIsNotNone(pin_entry.pin)
            self.assertTrue(pin_entry.lazy_wings)
            regen = hub_coord_from_word(self.registry, word, pin_entry.pin)
            self.assertEqual(regen, leg.coord)
            wc = pin_entry.resolve_wing_coords(self.registry)
            self.assertGreaterEqual(len(wc), 1)

    def test_pin_smaller_than_legacy_wire(self):
        sig_pin = build_hub_signature(
            "d0", SMALL_CORPUS[0], self.registry, top_k=12, use_pin_wire=True
        )
        sig_legacy = build_hub_signature(
            "d0", SMALL_CORPUS[0], self.registry, top_k=12, use_pin_wire=False
        )
        self.assertLess(sig_pin.encoded_size(), sig_legacy.encoded_size())

    def test_materialize_wings_match_legacy(self):
        sig_legacy = build_hub_signature(
            "d0", SMALL_CORPUS[0], self.registry, top_k=8, use_pin_wire=False
        )
        sig_pin = build_hub_signature(
            "d0", SMALL_CORPUS[0], self.registry, top_k=8, use_pin_wire=True
        )
        materialize_lazy_hub_wings({"d0": sig_pin}, self.registry)
        for word, leg in sig_legacy.hubs.items():
            pin_entry = sig_pin.hubs[word]
            self.assertEqual(
                pin_entry.wing_coords(),
                leg.wing_coords(),
                msg=word,
            )

    def test_pin_meet_score_matches_legacy(self):
        text = SMALL_CORPUS[0]
        sig_legacy = build_hub_signature(
            "d0", text, self.registry, top_k=8, use_pin_wire=False
        )
        sig_pin = build_hub_signature(
            "d0", text, self.registry, top_k=8, use_pin_wire=True
        )
        materialize_lazy_hub_wings({"d0": sig_pin}, self.registry)
        query = " ".join(list(sig_legacy.hub_words())[:3])
        profile = build_query_profile(
            query,
            self.registry,
            neighbor_map={},
            doc_freq={w: 1 for w in tokenize_words(text)},
            n_docs=4,
        )
        doc_tokens = frozenset(w for w in text.lower().split() if w.isalpha())
        score_leg = score_document(
            profile, doc_tokens, sig_legacy, registry=self.registry
        )
        score_pin = score_document(
            profile, doc_tokens, sig_pin, registry=self.registry
        )
        self.assertAlmostEqual(score_pin, score_leg, places=5)

    def test_critical_line_leg_sum(self):
        pin = CriticalLinePin.from_coord((5.0, 3.0, 12.0), 101)
        leg, side = leg_sum_im_led((5.0, 3.0, 12.0))
        self.assertEqual(pin.leg_sum, leg)
        self.assertEqual(pin.band_side, side)


if __name__ == "__main__":
    unittest.main()
