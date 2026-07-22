#!/usr/bin/env python3
"""CHAMPION -- the composed, governed, no-GPU-at-serve RAG (the "best version", assembled).

One serve path that composes the durable, adversarially-measured wins with the per-corpus governors:

    FLOOR   : full-depth BM25 owns the ranking (encoder-free).
    APEX    : rerank the floor pool with a teacher -- GOVERNED: only when the teacher beats BM25 on this
              corpus (apex_on). Pluggable teacher(query, doc_id)->score. Teacher is GPU-once at ingest
              (query-side encoding); if no teacher is supplied the tier is inert and the serve stays the
              proven no-GPU floor+shadow. (In THIS environment no teacher model is available, so apex is
              exercised only through the hook, never fabricated -- see run_champion_holdout.py.)
    SHADOW  : auto-arm shadow rescue -- append-only recall net, fires per-query on pool exhaustion
              (fill<1.0). P@1 relative to the head is untouched and recall cannot fall, BY CONSTRUCTION.
    EXPAND  : LLM query expansion -- GOVERNED: only when the vocabulary gap is large (expand_on). Also a
              GPU-once/ingest-or-LLM tier; represented as a governor decision here, not executed no-GPU.

The governors are NOT tuned on the test set. They are ESTIMATED from a small labelled TRAIN slice at
onboarding (see estimate_governors) and then applied unchanged. run_champion_holdout.py proves the
estimate from ~20 train queries makes the same decision the full test set would -- the honest validation
that separates "the best version" from a number fit to the scoreboard.

Composes the proven faces (does not reimplement them): EdgeRAGServer (floor+gate+shadow) from
edgerag_serve, shadow_retrieve + content from the same. No new retrieval math.
"""
from __future__ import annotations
import sys, os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

from edgerag_serve import EdgeRAGServer, content, shadow_retrieve


@dataclass
class Governors:
    """Per-corpus switches, estimated from a labelled TRAIN slice at onboarding (never the test set)."""
    apex_on: bool = False           # teacher rerank helps iff the teacher beats BM25 on this corpus
    expand_on: bool = False         # LLM expansion helps iff the vocabulary gap is large (>~25%)
    vocab_gap: float = 0.0          # the estimate the expand decision was made from
    expand_margin: float = 0.0      # gap - threshold: distance from the decision boundary
    expand_confident: bool = True   # |margin| >= band: a small onboarding slice is decisive here
    apex_gain_est: Optional[float] = None   # train-estimated apex nDCG gain (None if no teacher)
    source: str = "default"         # "train:N" once estimated

    def as_dict(self) -> dict:
        return {"apex_on": self.apex_on, "expand_on": self.expand_on,
                "vocab_gap": round(self.vocab_gap, 4),
                "expand_margin": round(self.expand_margin, 4),
                "expand_confident": self.expand_confident,
                "apex_gain_est": None if self.apex_gain_est is None else round(self.apex_gain_est, 4),
                "source": self.source}


# teacher(query_text, doc_id) -> float score. Higher = more relevant. GPU-once at ingest (query encoding).
Teacher = Callable[[str, str], float]


class Champion:
    """The composed governed serve, built once per corpus."""
    def __init__(self, corpus: Mapping[str, str], *, depth: int = 200, shadow_budget: int = 100,
                 governors: Optional[Governors] = None, teacher: Optional[Teacher] = None):
        self.base = EdgeRAGServer(dict(corpus), depth=depth, shadow_budget=shadow_budget)
        self.depth = depth
        self.budget = shadow_budget
        self.gov = governors or Governors()
        self.teacher = teacher

    # ---- the ONE governed serve --------------------------------------------------------------
    def serve(self, query: str) -> dict:
        scored = self.base.bm.search_scored(query, self.depth)     # FLOOR: BM25 owns the ranking
        base_ids = [d for d, _ in scored]
        qterms = content(query)

        # APEX (governed): rerank the floor pool head iff the governor is on AND a teacher is supplied.
        head = base_ids
        apex_used = False
        if self.gov.apex_on and self.teacher is not None and base_ids:
            head = [d for d, _ in sorted(((d, self.teacher(query, d)) for d in base_ids), key=lambda z: -z[1])]
            apex_used = True

        # SHADOW (always available; gate decides per-query). Append-only BELOW the head.
        arm, sig = self.base.gate(scored, qterms)
        if arm:
            shadow = shadow_retrieve(self.base.idx, qterms, set(base_ids), self.budget)
            ranked = head + shadow
        else:
            ranked = head
        return {"ranked": ranked, "base_ids": base_ids, "armed": arm, "apex_used": apex_used,
                "expand_on": self.gov.expand_on, "gate": sig, "query_terms": qterms,
                "path": ("apex+" if apex_used else "") + ("shadow" if arm else "floor")}

    # ---- governor estimation (onboarding; TRAIN slice only) ----------------------------------
    def vocab_gap(self, queries: Mapping[str, str], qrels: Mapping[str, Mapping[str, object]]) -> float:
        """Share of relevant docs that contain NONE of the query's content terms (the expand signal)."""
        miss = tot = 0
        SET = self.base.idx.SET
        for qid, rel in qrels.items():
            qset = set(content(queries.get(qid, "")))
            if not qset:
                continue
            for did in rel:
                dt = SET.get(did)
                if dt is None:
                    continue
                tot += 1
                if not (dt & qset):
                    miss += 1
        return (miss / tot) if tot else 0.0

    def estimate_governors(self, queries: Mapping[str, str], train_qrels: Mapping[str, Mapping[str, object]],
                           *, teacher: Optional[Teacher] = None, expand_gap_thresh: float = 0.25,
                           band: float = 0.10) -> Governors:
        """Set the governors from a labelled TRAIN slice. No test-set contact.

        expand_on : vocab_gap(train) > threshold.
        expand_confident : |gap - threshold| >= band. A HYSTERESIS band around the boundary: a corpus whose
                    estimated gap lands inside it (e.g. fiqa ~0.11, only ~0.14 below 0.25) is NEAR-BOUNDARY --
                    an adversarial slice-robustness check measured ~2.6% of random 20-query slices flip its
                    decision. For those, collect more labels before trusting the switch; far-from-boundary
                    corpora (scifact, nfcorpus) are decisive from ~20 queries.
        apex_on   : if a teacher is supplied, whether teacher-rerank beats BM25 nDCG@10 on the train slice;
                    else left False with apex_gain_est=None (honest -- no teacher, no claim).
        """
        gap = self.vocab_gap(queries, train_qrels)
        margin = gap - expand_gap_thresh
        apex_gain = None
        apex_on = False
        if teacher is not None:
            apex_gain = self._apex_train_gain(queries, train_qrels, teacher)
            apex_on = apex_gain > 0.0
        n = len(train_qrels)
        return Governors(apex_on=apex_on, expand_on=gap > expand_gap_thresh, vocab_gap=gap,
                         expand_margin=margin, expand_confident=abs(margin) >= band,
                         apex_gain_est=apex_gain, source=f"train:{n}")

    def _apex_train_gain(self, queries, train_qrels, teacher: Teacher) -> float:
        """nDCG@10(apex) - nDCG@10(BM25) on the train slice -- the apex governor's decision signal."""
        from eval_beir import ndcg_at_k
        db = da = n = 0.0
        for qid, rel in train_qrels.items():
            q = queries.get(qid)
            if not q:
                continue
            scored = self.base.bm.search_scored(q, self.depth)
            base_ids = [d for d, _ in scored]
            if not base_ids:
                continue
            apex_ids = [d for d, _ in sorted(((d, teacher(q, d)) for d in base_ids), key=lambda z: -z[1])]
            db += ndcg_at_k(base_ids, rel, 10)
            da += ndcg_at_k(apex_ids, rel, 10)
            n += 1
        return (da - db) / n if n else 0.0
