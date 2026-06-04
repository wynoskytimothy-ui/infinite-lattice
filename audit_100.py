#!/usr/bin/env python3
"""
Audit the first 100 test queries in detail.
Shows per-query NDCG, what signals fired, and WHY each query succeeded or failed.
"""
import sys
from pathlib import Path
from collections import defaultdict, Counter
sys.stdout.reconfigure(encoding="utf-8")

from beir_data_root import resolve_beir_root
from eval_beir import (
    load_corpus, load_queries, load_qrels, merge_qrels, load_paths,
    build_corpus_index, make_pipeline, ingest_corpus, build_neighbor_weights,
    candidate_ids, ndcg_at_k, recall_at_k, doc_text,
)
from aethos_hub_signature import (
    build_all_hub_signatures, build_query_profile,
    rank_with_hub_signatures, score_document, CONSENSUS_WINGS,
)
from aethos_subword_composite import build_subword_composite_index
from aethos_composite import build_composite_index
from aethos_phrase_composite import (
    build_phrase_composite_index, query_phrase_composites as _qpc
)
from aethos_discriminative import (
    build_heavy_anchor_index, train_on_qrels, query_anchor_composites,
    score_with_heavy_anchors, train_convergence_loop,
)
from aethos_persist import brain_path_for_dataset, load_brain
from aethos_iterative import build_multi_pass
from aethos_scale import timed_ingest_one
from aethos_tokenize import tokenize_words
import aethos_hub_signature as _hs
import time

N_QUERIES = 100
DATASET = "scifact"

root = Path(resolve_beir_root())
paths = load_paths(root, DATASET)

print(f"Loading {DATASET}...", flush=True)
corpus = load_corpus(paths.corpus, max_docs=None)
queries = load_queries(paths.queries)
qrels = merge_qrels(load_qrels(paths.qrels_test), load_qrels(paths.qrels_train))
qrels_train = load_qrels(paths.qrels_train)
qrels_test = load_qrels(paths.qrels_test)

# Build pipeline
pipe = make_pipeline("scale")
metrics, cidx = ingest_corpus(pipe, corpus, mode="scale")
try:
    pipe.flush()
except Exception:
    pass

print(f"  {len(cidx.doc_ids)} docs", flush=True)

corpus_texts = [doc_text(doc) for doc in corpus.values()]
mp = build_multi_pass(pipe, corpus_texts, cidx.doc_tokens, n_passes=1, verbose=True)

hub_sigs = build_all_hub_signatures(cidx.doc_ids, cidx.doc_tokens, pipe.registry, top_k=12)
sub_comp_idx = build_subword_composite_index(pipe.registry, cidx.doc_tokens, max_composites=500)
comp_idx = build_composite_index(hub_sigs)
phrase_idx = mp.phrase_idx or build_phrase_composite_index(
    pipe.registry, cidx.doc_tokens, min_word_len=4, min_pair_count=3,
    max_pairs_per_doc=32, use_pool_primes_only=True
)
anchor_idx = build_heavy_anchor_index(
    pipe.registry, cidx.doc_tokens, cidx.doc_freq,
    max_doc_count=5, rarity_threshold=0.018
)

b_path = brain_path_for_dataset(DATASET, "scale")
init_lc, init_ln = load_brain(b_path, anchor_idx, verbose=True)
_hs.LAMBDA_COORD = init_lc
_hs.LAMBDA_NEIGHBOR = init_ln

train_on_qrels(anchor_idx, queries, qrels_train, cidx.doc_ids, cidx.doc_tokens,
               pipe.registry, cidx.doc_freq, len(cidx.doc_ids))

neighbor_map = build_neighbor_weights(pipe.registry)

qids_test = [q for q in qrels_test if q in queries][:N_QUERIES]
n_docs = len(cidx.doc_ids)

print(f"\n{'='*80}")
print(f"AUDIT: first {len(qids_test)} test queries")
print(f"{'='*80}\n")

failure_modes = Counter()
ndcg_by_type = defaultdict(list)

for qi, qid in enumerate(qids_test):
    profile = build_query_profile(
        queries[qid], pipe.registry,
        neighbor_map=neighbor_map,
        doc_freq=cidx.doc_freq, n_docs=n_docs,
    )
    q_anchor = query_anchor_composites(
        list(profile.word_set), anchor_idx, pipe.registry, idf=profile.idf
    )
    q_phrase = _qpc(profile.words, phrase_idx, pipe.registry, profile.idf)

    cands = candidate_ids(profile.words, cidx.inv, neighbor_map, cidx.doc_ids)
    ranked = rank_with_hub_signatures(
        profile, cands, hub_sigs, cidx.doc_ids,
        doc_tokens=cidx.doc_tokens, doc_tf=cidx.doc_tf,
        doc_len=cidx.doc_len, avg_dl=cidx.avg_dl,
        sub_comp_idx=sub_comp_idx, registry=pipe.registry,
        phrase_idx=phrase_idx,
        query_anchor_comps=q_anchor,
        query_phrase_comps=q_phrase,
        anchor_idx=anchor_idx,
        top_k=100,
    )

    rel = qrels_test[qid]
    ndcg = ndcg_at_k(ranked, rel, 10)
    r10 = recall_at_k(ranked, rel, 10)
    top1 = ranked[0] if ranked else ""

    # --- determine failure mode ---
    gold_ids = set(rel.keys())
    top10 = set(ranked[:10])
    cand_set = set(cands)

    if ndcg == 1.0:
        mode = "PERFECT"
    elif ndcg > 0:
        mode = "PARTIAL"
    elif not gold_ids & cand_set:
        mode = "MISSED_CANDIDATE"   # correct doc never reached top-100 candidates
    else:
        # correct doc was a candidate but scored too low
        # check if BM25 was the issue
        gold_in_cands = gold_ids & cand_set
        gold_doc = next(iter(gold_in_cands))
        gold_tokens = cidx.doc_tokens.get(gold_doc, frozenset())
        bm25_overlap = profile.word_set & gold_tokens
        if not bm25_overlap:
            mode = "ZERO_BM25"      # no shared vocabulary at all
        else:
            mode = "SCORE_MISS"     # has BM25 overlap but ranked too low

    failure_modes[mode] += 1
    ndcg_by_type[mode].append(ndcg)

    # Print summary for first 100
    gold_ids_str = ",".join(list(gold_ids)[:2])
    query_short = queries[qid][:50]
    print(
        f"Q{qi+1:3d} [{mode:18s}] NDCG={ndcg:.3f} R@10={r10:.2f} "
        f"| {query_short!r:.50}",
        flush=True
    )

    if mode in ("ZERO_BM25", "MISSED_CANDIDATE", "SCORE_MISS") and qi < 20:
        # Deep dive for first 20 failures
        gold_doc = next(iter(gold_ids), None)
        if gold_doc:
            gold_rank = next((i+1 for i, d in enumerate(ranked) if d in gold_ids), ">100")
            gold_tokens = cidx.doc_tokens.get(gold_doc, frozenset())
            overlap = sorted(profile.word_set & gold_tokens)
            missing = sorted(profile.word_set - gold_tokens)
            print(f"       gold={gold_doc}  rank={gold_rank}")
            print(f"       BM25 overlap: {overlap}")
            print(f"       query words NOT in gold: {missing}")
            if mode == "ZERO_BM25":
                # Show top signal scores for gold doc
                gold_sig = hub_sigs.get(gold_doc)
                gold_tf = cidx.doc_tf.get(gold_doc)
                gold_dl = cidx.doc_len.get(gold_doc, 0)
                s_full = score_document(
                    profile, gold_tokens, gold_sig,
                    doc_tf=gold_tf, doc_len=gold_dl, avg_dl=cidx.avg_dl,
                )
                print(f"       gold total score: {s_full:.3f}  (BM25=0, lattice-only)")
                if ranked:
                    top1_tokens = cidx.doc_tokens.get(top1, frozenset())
                    top1_sig = hub_sigs.get(top1)
                    top1_tf = cidx.doc_tf.get(top1)
                    top1_dl = cidx.doc_len.get(top1, 0)
                    s_top1 = score_document(
                        profile, top1_tokens, top1_sig,
                        doc_tf=top1_tf, doc_len=top1_dl, avg_dl=cidx.avg_dl,
                    )
                    print(f"       #1 wrong score: {s_top1:.3f}")

print(f"\n{'='*80}")
print(f"FAILURE MODE BREAKDOWN ({len(qids_test)} queries):")
print(f"{'='*80}")
for mode, count in sorted(failure_modes.items(), key=lambda x: -x[1]):
    avg = sum(ndcg_by_type[mode]) / max(len(ndcg_by_type[mode]), 1)
    print(f"  {mode:20s}: {count:4d} queries  avg NDCG={avg:.4f}")

total_ndcg = sum(v for vlist in ndcg_by_type.values() for v in vlist) / max(len(qids_test), 1)
print(f"\n  TOTAL avg NDCG@10 = {total_ndcg:.4f}")
print(f"\n  KEY INSIGHT:")
print(f"  PERFECT queries drive the early high score (first 50 have more easy queries).")
print(f"  ZERO_BM25 / MISSED_CANDIDATE queries are the floor — need lattice signals.")
print(f"  SCORE_MISS queries have BM25 overlap but lose to higher-scored wrong docs.")
