#!/usr/bin/env python3
"""
audit_lattice_failures.py - forensic audit of WHY queries fail.

For the worst-scoring queries on SciFact:
  - dump query word composites + their letter prime factors
  - show gold docs: where they ranked, what query terms they share
  - show top-10 retrieved with full score breakdown
  - diagnose: why is the false positive at rank 1 above the gold?

Output points at the specific mathematical noise: shared common letters,
inflated IDF on partial matches, missing intersection structure, etc.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aethos_lattice_retrieval import LatticeRetriever, tokenize
from beir_data_root import resolve_beir_root
from eval_beir import (
    doc_text,
    load_corpus,
    load_paths,
    load_qrels,
    load_queries,
    ndcg_at_k,
    recall_at_k,
)


def show_word_composite(retriever, word: str) -> str:
    """Format a word's composite + letter factors for display."""
    composite = retriever.token_to_prime.get(word, None)
    if composite is None:
        return f"  {word!r:>20} -- NOT IN VOCAB"
    factors = retriever._word_factors.get(composite, frozenset())
    df = len(retriever.lattice.resolve(composite).parents) if composite else 0
    n_docs = max(len(retriever.doc_id_to_prime), 1)
    import math
    idf = math.log((n_docs + 1) / (df + 1)) + 1
    return (
        f"  {word!r:>20}  comp={composite:<25}  "
        f"factors={sorted(factors)}  df={df}  idf={idf:.2f}"
    )


def score_doc_breakdown(retriever, query_primes_set: set[int],
                        doc_id: str, query_words: list[str]) -> dict:
    """Break down the score for one doc against one query."""
    import math
    doc_prime = retriever.doc_id_to_prime[doc_id]
    doc_chain = retriever.lattice.resolve(doc_prime).sub_chain or ()
    doc_set = set(doc_chain)
    shared = query_primes_set & doc_set

    n_docs = max(len(retriever.doc_id_to_prime), 1)
    contributions: list[dict] = []
    bm25_term = 0.0
    doc_len = len(doc_chain)
    avg = retriever._avg_doc_len
    length_norm = 1.0 - 0.75 + 0.75 * (doc_len / max(avg, 1.0))
    doc_counts = retriever.doc_token_counts.get(doc_id)
    for anchor in shared:
        word = retriever.prime_to_token.get(anchor, f"<{anchor}>")
        df = max(len(retriever.lattice.resolve(anchor).parents), 1)
        idf = math.log((n_docs + 1) / (df + 1)) + 1
        tf = 1
        if doc_counts is not None:
            tf = doc_counts.get(word, 1)
        tf_sat = (tf * 2.5) / (tf + 1.5 * length_norm)
        contrib = idf * tf_sat
        bm25_term += contrib
        contributions.append({
            "word": word, "df": df, "idf": idf, "tf": tf,
            "tf_sat": tf_sat, "contrib": contrib,
        })
    contributions.sort(key=lambda c: -c["contrib"])
    return {
        "doc_id": doc_id,
        "doc_len": doc_len,
        "length_norm": length_norm,
        "shared_count": len(shared),
        "shared_words": [c["word"] for c in contributions],
        "bm25_term": bm25_term,
        "contributions": contributions,
    }


def audit_query(retriever, qid, query_text, gold, corpus, k=10):
    """Run one query, return diagnostic block."""
    results = retriever.query(query_text, k=k)
    ranked_ids = [d for d, _ in results]
    scores_by_id = {d: s for d, s in results}
    ndcg = ndcg_at_k(ranked_ids, gold, k=k)
    recall = recall_at_k(ranked_ids, gold, k=k)

    # Query word composites
    query_words = tokenize(query_text)
    q_primes_set: set[int] = set()
    for w in query_words:
        p = retriever.token_to_prime.get(w)
        if p is not None:
            q_primes_set.add(p)

    print("=" * 78)
    print(f"QUERY {qid}: {query_text!r}")
    print(f"  nDCG@{k}={ndcg:.3f}   Recall@{k}={recall:.3f}   gold_docs={list(gold.keys())}")
    print("=" * 78)

    print("\n-- Query word composites (letter prime factors) --")
    for w in query_words:
        print(show_word_composite(retriever, w))

    # Gold doc analysis
    print("\n-- GOLD docs --")
    for gold_id in gold:
        if gold_id not in retriever.doc_id_to_prime:
            print(f"  {gold_id!r}: NOT IN INDEX")
            continue
        rank = ranked_ids.index(gold_id) + 1 if gold_id in ranked_ids else None
        rank_str = f"rank {rank}/{k}" if rank else f"NOT IN TOP {k}"
        breakdown = score_doc_breakdown(retriever, q_primes_set, gold_id, query_words)
        gold_text = corpus.get(gold_id, {})
        title = gold_text.get("title", "")[:80] if isinstance(gold_text, dict) else ""
        body = gold_text.get("text", "")[:200] if isinstance(gold_text, dict) else ""
        score = scores_by_id.get(gold_id, None)
        print(f"\n  GOLD {gold_id} [{rank_str}, score={score}]")
        print(f"    title: {title!r}")
        print(f"    body:  {body!r}")
        print(f"    doc_len={breakdown['doc_len']}, length_norm={breakdown['length_norm']:.2f}")
        print(f"    shared with query: {breakdown['shared_count']} words -> {breakdown['shared_words']}")
        print(f"    BM25 term breakdown (top 5):")
        for c in breakdown["contributions"][:5]:
            print(f"      {c['word']!r:>20}  df={c['df']:>5}  idf={c['idf']:>5.2f}  "
                  f"tf={c['tf']}  tf_sat={c['tf_sat']:.2f}  contrib={c['contrib']:>6.2f}")
        print(f"    TOTAL bm25_term: {breakdown['bm25_term']:.2f}")

    # Top-K retrieved
    print(f"\n-- TOP {k} RETRIEVED --")
    for i, (d_id, score) in enumerate(results, 1):
        mark = "GOLD" if d_id in gold else "    "
        breakdown = score_doc_breakdown(retriever, q_primes_set, d_id, query_words)
        doc = corpus.get(d_id, {})
        title = doc.get("title", "")[:80] if isinstance(doc, dict) else ""
        print(f"\n  [{i}] {mark} {d_id}  score={score:.2f}  shared={breakdown['shared_count']}  "
              f"len={breakdown['doc_len']}")
        print(f"      title: {title!r}")
        top_contribs = breakdown["contributions"][:3]
        ctop = ", ".join(
            f"{c['word']!r}(idf={c['idf']:.1f},c={c['contrib']:.1f})"
            for c in top_contribs
        )
        print(f"      top contribs: {ctop}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="scifact")
    ap.add_argument("--n-failures", type=int, default=5)
    ap.add_argument("--ndcg-threshold", type=float, default=0.30)
    args = ap.parse_args()

    root = Path(resolve_beir_root())
    paths = load_paths(root, args.dataset)

    print(f"loading {args.dataset}...")
    corpus = load_corpus(paths.corpus, max_docs=None)
    queries_all = load_queries(paths.queries)
    qrels = load_qrels(paths.qrels_test)
    if not qrels:
        qrels = load_qrels(paths.qrels_train)

    print(f"building LatticeRetriever...")
    retriever = LatticeRetriever(token_pool_size=50000, doc_pool_size=20000)
    corpus_text = {did: doc_text(d) for did, d in corpus.items()}
    retriever.build_from_corpus(corpus_text)
    print(f"  built: {retriever.estimated_footprint()['n_tokens']} unique tokens, "
          f"{retriever.estimated_footprint()['n_docs']} docs\n")

    # Score all queries with qrels, find worst
    qids = [qid for qid in queries_all if qid in qrels]
    scored: list[tuple[str, float]] = []
    for qid in qids:
        results = retriever.query(queries_all[qid], k=10)
        ranked = [d for d, _ in results]
        nd = ndcg_at_k(ranked, qrels[qid], k=10)
        scored.append((qid, nd))
    scored.sort(key=lambda x: x[1])

    failing = [q for q in scored if q[1] < args.ndcg_threshold][:args.n_failures]
    print(f"=== AUDITING {len(failing)} WORST FAILING QUERIES (nDCG<{args.ndcg_threshold}) ===\n")
    for qid, nd in failing:
        audit_query(retriever, qid, queries_all[qid], qrels[qid], corpus, k=10)
        print()


if __name__ == "__main__":
    main()
