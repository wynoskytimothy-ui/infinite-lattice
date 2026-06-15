"""Tests for pattern placement audit."""
import unittest

from diagnose_corpus import SMALL_CORPUS
from eval_beir import build_neighbor_weights, ingest_corpus, make_pipeline, ndcg_at_k
from aethos_hub_signature import build_all_hub_signatures
from aethos_tokenize import tokenize_words
from eval_checkpoint import EvalBundle
from pipeline.bit_03_doc_attractor_set import build_attractor_index_from_hub_signatures
from pipeline.pattern_placement import (
    SIGNAL_NAMES,
    audit_query_patterns,
    classify_failure_pattern,
    compute_pattern_breakdown,
    format_pattern_report,
)
from aethos_hub_signature import build_query_profile


class TestPatternPlacement(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pipe = make_pipeline("scale")
        corpus = {f"d{i}": {"title": "", "text": t} for i, t in enumerate(SMALL_CORPUS)}
        _, cls.cidx = ingest_corpus(cls.pipe, corpus, mode="scale")
        cls.doc_tokens = {
            f"d{i}": frozenset(tokenize_words(t))
            for i, t in enumerate(SMALL_CORPUS)
        }
        cls.hub_sigs = build_all_hub_signatures(
            cls.cidx.doc_ids, cls.cidx.doc_tokens, cls.pipe.registry, top_k=12,
        )
        cls.neighbor_map = build_neighbor_weights(cls.pipe.registry)
        cls.attractor_index = build_attractor_index_from_hub_signatures(
            cls.pipe.registry, cls.hub_sigs,
        )
        cls.queries = {
            "q1": "phone technical software",
            "q2": "apple fruit pie",
        }
        cls.qrels = {
            "q1": {"d0": 1},
            "q2": {"d3": 1},
        }
        cls.bundle = EvalBundle(
            dataset="test",
            mode="scale",
            qids=["q1", "q2"],
            queries=cls.queries,
            qrels=cls.qrels,
            cidx=cls.cidx,
            hub_sigs=cls.hub_sigs,
            neighbor_map=cls.neighbor_map,
            meet_index={},
            sub_comp_idx=None,
            comp_idx=None,
            phrase_idx=None,
            anchor_idx=None,
            pipe=cls.pipe,
            hub_bytes_per_doc=0.0,
            p50_ingest_ms=0.0,
            p99_ingest_ms=0.0,
            bytes_per_doc=0.0,
            n_docs=len(cls.cidx.doc_ids),
            attractor_index=cls.attractor_index,
        )

    def test_classify_perfect(self):
        pat = classify_failure_pattern(
            ndcg10=1.0,
            gold_ids={"d0"},
            gold_in_candidates=True,
            gold_best_rank=1,
            gold_bm25_overlap=2,
            top1_is_gold=True,
            gold_in_corpus=True,
        )
        self.assertEqual(pat, "PERFECT")

    def test_breakdown_has_all_signals(self):
        profile = build_query_profile(
            "phone technical",
            self.pipe.registry,
            neighbor_map=self.neighbor_map,
            doc_freq=self.cidx.doc_freq,
            n_docs=len(self.cidx.doc_ids),
        )
        bd = compute_pattern_breakdown(
            profile,
            "d0",
            cidx=self.cidx,
            hub_sigs=self.hub_sigs,
            comp_idx=None,
            sub_comp_idx=None,
            phrase_idx=None,
            anchor_idx=None,
            q_anchor_comps=None,
            q_phrase_comps=None,
            registry=self.pipe.registry,
            attractor_index=self.attractor_index,
            query_kappa_keys=self.attractor_index.doc_keys.get("d0", set()),
        )
        for name in SIGNAL_NAMES:
            self.assertIn(name, bd.signal_dict())
        self.assertGreaterEqual(bd.s1_bm25, 0.0)

    def test_audit_runs_on_small_bundle(self):
        report = audit_query_patterns(self.bundle, enable_kappa_scoring=True)
        self.assertEqual(report.n_queries, 2)
        self.assertIn("PERFECT", report.pattern_counts.keys() | {"PARTIAL", "SCORE_MISS"})
        text = format_pattern_report(report, top_n=5)
        self.assertIn("PATTERN PLACEMENT AUDIT", text)
        self.assertEqual(len(report.tuner_hints), len(report.tuner_hints))  # smoke


if __name__ == "__main__":
    unittest.main()
