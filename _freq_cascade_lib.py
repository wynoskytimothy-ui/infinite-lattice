#!/usr/bin/env python3
"""Shared, tested library for a BEIR frequency-cascade retrieval experiment.

Everything here is deterministic, network-free, and reads the local BEIR
corpora (nfcorpus, scifact) via the repo harness.  It exposes:

  - load_eval(dataset)          -> dict(corpus, queries, qrels, qids, text_of)
  - class Bm25Base              -> thin wrapper over marco_baseline.BM25
  - mrr_at_k(ranked, rel, k=10)
  - class FreqPrimitives        -> df/coverage/postings/meet + anti-correlation
  - eval_ranker(rank_fn, ev)    -> mean nDCG@10/R@10/R@100/MRR@10 over ALL qids

Tokenization for the frequency primitives and the BM25 wrapper uses
``marco_baseline.tok`` (the ``[a-z0-9]+`` lowercaser), exactly as the harness
BM25 does, so DF/postings and BM25 share one vocabulary.

Metric code (ndcg_at_k, recall_at_k) is imported from eval_beir so baseline and
any variant score with identical metric definitions.  qrels are used ONLY for
scoring -- never fed into retrieval.
"""

from __future__ import annotations

import sys

sys.stdout.reconfigure(errors="replace")

import math
from collections import defaultdict
from pathlib import Path

from beir_data_root import resolve_beir_root
from eval_beir import (
    load_paths,
    load_corpus,
    load_queries,
    load_qrels,
    merge_qrels,
    doc_text,
    ndcg_at_k,
    recall_at_k,
)
from marco_baseline import BM25, tok


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_eval(dataset: str) -> dict:
    """Load a BEIR dataset for full-corpus, full-query evaluation.

    Returns a dict with:
      corpus   {doc_id: {title, text}}         -- the FULL small corpus
      queries  {qid: text}
      qrels    {qid: {doc_id: score>0}}        -- TEST qrels only (scoring only)
      qids     [qid, ...]                       -- all TEST qids present in queries
      text_of  {doc_id: "title text"}          -- title+text per doc

    Only TEST qrels are used for qids/scoring so the numbers line up with the
    standard BEIR test-set references (scifact ~0.64, nfcorpus ~0.32).  Train
    qrels are intentionally NOT merged in -- merging would swap in the (much
    larger, differently-distributed) train query set and move the target.
    """
    root = Path(resolve_beir_root())
    paths = load_paths(root, dataset)
    corpus = load_corpus(paths.corpus, max_docs=None)
    queries = load_queries(paths.queries)
    qrels = load_qrels(paths.qrels_test)                # scoring only
    qids = [q for q in qrels if q in queries]           # all test qids in queries
    text_of = {did: doc_text(corpus[did]) for did in corpus}
    return {
        "dataset": dataset,
        "corpus": corpus,
        "queries": queries,
        "qrels": qrels,
        "qids": qids,
        "text_of": text_of,
    }


# ---------------------------------------------------------------------------
# BM25 baseline wrapper
# ---------------------------------------------------------------------------

class Bm25Base:
    """Thin wrapper over marco_baseline.BM25 with standard BEIR hyperparameters.

    Build once per corpus, reuse for every query.  ``rank`` returns a plain list
    of doc_ids (top-k), which is exactly what eval_ranker consumes.
    """

    def __init__(self, ev: dict, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.bm = BM25(k1=k1, b=b)
        # index over the FULL corpus (title+text per doc)
        text_of = ev["text_of"]
        self.bm.index([(did, text_of[did]) for did in ev["corpus"]])

    def rank(self, qtext: str, k: int = 100) -> list:
        return self.bm.search(qtext, k=k)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def mrr_at_k(ranked: list, rel: dict, k: int = 10) -> float:
    for i, d in enumerate(ranked[:k]):
        if d in rel:
            return 1.0 / (i + 1)
    return 0.0


def eval_ranker(rank_fn, ev: dict) -> dict:
    """Score ``rank_fn`` over ALL qids in ``ev``.

    rank_fn(qtext) -> ranked list of doc_ids.  We take the top 100, then compute
    mean nDCG@10, Recall@10, Recall@100, MRR@10 across every qid.  Uses
    eval_beir.ndcg_at_k / recall_at_k so metric definitions match the harness.
    """
    qids = ev["qids"]
    queries = ev["queries"]
    qrels = ev["qrels"]
    n = 0
    s_ndcg = s_r10 = s_r100 = s_mrr = 0.0
    for qid in qids:
        rel = qrels[qid]
        ranked = rank_fn(queries[qid])[:100]
        s_ndcg += ndcg_at_k(ranked, rel, 10)
        s_r10 += recall_at_k(ranked, rel, 10)
        s_r100 += recall_at_k(ranked, rel, 100)
        s_mrr += mrr_at_k(ranked, rel, 10)
        n += 1
    denom = max(n, 1)
    return {
        "n_queries": n,
        "ndcg@10": s_ndcg / denom,
        "recall@10": s_r10 / denom,
        "recall@100": s_r100 / denom,
        "mrr@10": s_mrr / denom,
    }


# ---------------------------------------------------------------------------
# Frequency primitives (computed once over a corpus)
# ---------------------------------------------------------------------------

class FreqPrimitives:
    """Corpus-level frequency structure over ``tok``-ized documents.

    Built once from ev['text_of'].  Provides document frequencies, coverage,
    postings (df>=2), the meet / meet3 intersections, complement
    anti-correlation, and never-co-occur mutual-exclusion axes.

    All of this is derived purely from the corpus text -- no qrels, no queries.
    """

    def __init__(self, ev: dict, min_df_postings: int = 2):
        self.dataset = ev.get("dataset", "?")
        text_of = ev["text_of"]
        self.N = len(text_of)
        self.doc_ids = list(text_of.keys())

        # per-doc token SETS (presence, not TF) -- df is a document count
        self._doc_terms: dict = {}
        df: dict = defaultdict(int)
        postings_all: dict = defaultdict(set)
        for did in self.doc_ids:
            terms = set(tok(text_of[did]))
            self._doc_terms[did] = terms
            for w in terms:
                df[w] += 1
                postings_all[w].add(did)

        self._df = dict(df)
        # postings: drop terms with df < min_df_postings (default 2)
        self._postings = {
            w: docs for w, docs in postings_all.items()
            if len(docs) >= min_df_postings
        }

    # --- basic frequency stats ------------------------------------------
    def df(self, term: str) -> int:
        """Document frequency of ``term`` (0 if unseen)."""
        return self._df.get(term, 0)

    def coverage(self, term: str) -> float:
        """df(term) / N -- fraction of docs containing ``term``."""
        return self._df.get(term, 0) / self.N if self.N else 0.0

    def vocab_by_df(self) -> list:
        """Vocabulary sorted by descending df, then term (stable ties)."""
        return sorted(self._df, key=lambda w: (-self._df[w], w))

    # --- postings / meets -----------------------------------------------
    def postings(self, term: str) -> set:
        """Set of doc_ids containing ``term`` (empty if df<2 or unseen)."""
        return self._postings.get(term, set())

    def meet(self, a: str, b: str) -> set:
        """2-way meet: doc_ids containing BOTH a and b."""
        return self._postings.get(a, set()) & self._postings.get(b, set())

    def meet3(self, a: str, b: str, c: str) -> set:
        """3-way meet: doc_ids containing all of a, b, c."""
        return (
            self._postings.get(a, set())
            & self._postings.get(b, set())
            & self._postings.get(c, set())
        )

    # --- complement anti-correlation ------------------------------------
    def complement_anticorrelate(self, term: str, topk: int = 5) -> list:
        """Words most ENRICHED among docs that do NOT contain ``term``.

        For the complement C = docs without ``term``, each candidate word w is
        scored by its lift in the absence of ``term``:

            lift(w) = ( df_in_complement(w) / |C| ) / ( df(w) / N )

        A lift > 1 means w is over-represented where ``term`` is absent (an
        anti-correlation / mutual-exclusion signal).  Filters: df(w) >= 5,
        w != term, w must appear in the complement at least once.

        Returns up to ``topk`` (word, lift, df_in_complement) tuples, highest
        lift first.
        """
        term_docs = self._postings.get(term, set())
        complement = [d for d in self.doc_ids if d not in term_docs]
        nc = len(complement)
        if nc == 0:
            return []
        # count df of every word within the complement
        comp_df: dict = defaultdict(int)
        comp_set = set(complement)
        for w, docs in self._postings.items():
            c = len(docs & comp_set)
            if c:
                comp_df[w] = c
        scored = []
        for w, dfc in comp_df.items():
            if w == term:
                continue
            dfw = self._df.get(w, 0)
            if dfw < 5:
                continue
            base = dfw / self.N
            if base <= 0.0:
                continue
            lift = (dfc / nc) / base
            scored.append((w, lift, dfc))
        scored.sort(key=lambda t: (-t[1], -t[2], t[0]))
        return scored[:topk]

    # --- never-co-occur mutual-exclusion axes ---------------------------
    def never_cooccur_pairs(self, top_df: int = 200, max_pairs: int = 100) -> list:
        """Mutual-exclusion axes among the highest-df terms.

        Among the ``top_df`` highest-df terms, find pairs (a, b) whose observed
        co-occurrence is lowest relative to what independence predicts.  For a
        pair, under independence the expected shared-doc count is:

            expected(a,b) = df(a) * df(b) / N

        We score each pair by the ratio  observed / expected  (lower = more
        mutually exclusive; 0.0 means they never share a doc).  Ties broken by
        higher expected (bigger surprise) then term order.

        Returns up to ``max_pairs`` (a, b, observed, expected, ratio) tuples,
        lowest ratio first -- the strongest mutual-exclusion axes.
        """
        top_terms = [w for w in self.vocab_by_df()[:top_df] if w in self._postings]
        results = []
        for i in range(len(top_terms)):
            a = top_terms[i]
            pa = self._postings[a]
            dfa = len(pa)
            for j in range(i + 1, len(top_terms)):
                b = top_terms[j]
                pb = self._postings[b]
                observed = len(pa & pb)
                expected = dfa * len(pb) / self.N
                if expected <= 0.0:
                    continue
                ratio = observed / expected
                results.append((a, b, observed, expected, ratio))
        results.sort(key=lambda t: (t[4], -t[3], t[0], t[1]))
        return results[:max_pairs]


# ---------------------------------------------------------------------------
# __main__ smoke test
# ---------------------------------------------------------------------------

def _fmt(v: float) -> str:
    return f"{v:.4f}"


def _smoke() -> None:
    refs = {"scifact": 0.643, "nfcorpus": 0.321}
    print("=" * 72)
    print("BM25 smoke test (k1=1.5, b=0.75) -- full corpus, ALL test qids")
    print("=" * 72)
    rows = []
    for ds in ("nfcorpus", "scifact"):
        print(f"\n--- {ds} ---", flush=True)
        ev = load_eval(ds)
        print(
            f"  loaded corpus={len(ev['corpus'])} docs  "
            f"queries={len(ev['queries'])}  test-qids-in-queries={len(ev['qids'])}",
            flush=True,
        )
        bm = Bm25Base(ev)
        res = eval_ranker(lambda q: bm.rank(q, 100), ev)
        res["dataset"] = ds
        res["bm25_ref"] = refs[ds]
        res["delta"] = res["ndcg@10"] - refs[ds]
        rows.append(res)
        print(
            f"  BM25  n={res['n_queries']}  "
            f"ndcg@10={_fmt(res['ndcg@10'])}  "
            f"recall@10={_fmt(res['recall@10'])}  "
            f"recall@100={_fmt(res['recall@100'])}  "
            f"mrr@10={_fmt(res['mrr@10'])}",
            flush=True,
        )
        print(
            f"  ref ndcg@10={refs[ds]:.3f}  delta={res['delta']:+.4f}  "
            f"{'OK (near ref)' if abs(res['delta']) <= 0.03 else 'OFF FROM REF'}",
            flush=True,
        )

    print()
    print("=" * 88)
    print(
        f"{'corpus':<10} {'n_q':>5} {'ndcg@10':>9} {'recall@10':>10} "
        f"{'recall@100':>11} {'mrr@10':>8} {'bm25_ref':>9} {'delta':>8}"
    )
    print("-" * 88)
    for r in rows:
        print(
            f"{r['dataset']:<10} {r['n_queries']:>5} "
            f"{r['ndcg@10']:>9.4f} {r['recall@10']:>10.4f} "
            f"{r['recall@100']:>11.4f} {r['mrr@10']:>8.4f} "
            f"{r['bm25_ref']:>9.3f} {r['delta']:>+8.4f}"
        )
    print("=" * 88)

    # quick primitive sanity on scifact (deterministic, prints a few facts)
    print("\n--- FreqPrimitives sanity (scifact) ---", flush=True)
    ev = load_eval("scifact")
    fp = FreqPrimitives(ev)
    vb = fp.vocab_by_df()
    print(f"  N docs={fp.N}  vocab(df>=1)={len(vb)}  postings(df>=2)={len(fp._postings)}")
    top5 = [(w, fp.df(w), round(fp.coverage(w), 3)) for w in vb[:5]]
    print(f"  top-5 by df: {top5}")
    # a meet example on two common content-ish words if present
    if len(vb) >= 2:
        a, b = vb[0], vb[1]
        print(f"  meet({a!r},{b!r}) -> {len(fp.meet(a, b))} docs")
    anti = fp.complement_anticorrelate(vb[0], topk=3)
    print(f"  complement_anticorrelate({vb[0]!r}) top3: {[(w, round(l, 2)) for w, l, _ in anti]}")
    nc = fp.never_cooccur_pairs(top_df=100, max_pairs=3)
    print(f"  never_cooccur_pairs top3 (ratio): {[(a, b, round(r, 3)) for a, b, _, _, r in nc]}")


if __name__ == "__main__":
    _smoke()
