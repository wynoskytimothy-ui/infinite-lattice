"""Scale path tests — latency budget helpers + accuracy preserved."""

from __future__ import annotations

import unittest

from aethos_pipeline import AethosPipeline
from aethos_scale import ScaleConfig, ScaleMetrics, fingerprint_document
from diagnose_corpus import SMALL_CORPUS


class TestScalePath(unittest.TestCase):
    def test_fingerprint_under_100_bytes(self) -> None:
        fp = fingerprint_document(0, ["phone", "technical", "chip"], top_hub="phone")
        self.assertLessEqual(fp.encoded_size(), 100)

    def test_lazy_ingest_preserves_apple_routing(self) -> None:
        cfg = ScaleConfig(rebuild_every=64, lazy_clusters=True)
        pipe = AethosPipeline(rebuild_every=cfg.rebuild_every)
        pipe.apply_scale_config(cfg)
        for doc in SMALL_CORPUS:
            pipe.ingest_one(doc, finalize=False)
        pipe.flush()
        tech = pipe.resolve("apple", ["phone", "chip"])
        food = pipe.resolve("apple", ["fruit", "pie"])
        self.assertNotEqual(tech["cluster_id"], food["cluster_id"])

    def test_single_doc_cooccurrence_strength(self) -> None:
        pipe = AethosPipeline(rebuild_every=2)
        pipe.ingest(SMALL_CORPUS[0])
        strength_sum = sum(c.strength for c in pipe.registry.correlations.values())
        self.assertGreater(strength_sum, 0)
        self.assertLess(strength_sum, 500)

    def test_scale_metrics_empty(self) -> None:
        m = ScaleMetrics()
        self.assertTrue(m.pass_latency(50.0))
        self.assertTrue(m.pass_fingerprint(100))


if __name__ == "__main__":
    unittest.main()
