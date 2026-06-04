#!/usr/bin/env python3
"""Quick diagnostic for BEIR ID matching and scoring pipeline."""
import json, csv, sys
from pathlib import Path
from collections import defaultdict
sys.stdout.reconfigure(encoding='utf-8')

from aethos_pipeline import AethosPipeline
from aethos_tokenize import tokenize_words
from eval_beir import (
    build_corpus_index, build_neighbor_weights, candidate_ids,
    doc_text, load_corpus, load_qrels, merge_qrels, load_queries,
    load_paths, make_pipeline, ndcg_at_k, BM25_K1, BM25_B
)
from aethos_hub_signature import (
    build_all_hub_signatures, build_query_profile,
    rank_with_hub_signatures, score_document
)
from aethos_scale import timed_ingest_one
from beir_data_root import resolve_beir_root

root = Path(resolve_beir_root())
paths = load_paths(root, "scifact")

# --- load a small slice ---
MAX_DOCS = None  # full corpus
corpus = load_corpus(paths.corpus, max_docs=MAX_DOCS)
queries = load_queries(paths.queries)
qrels = merge_qrels(load_qrels(paths.qrels_test), load_qrels(paths.qrels_train))

doc_ids_in_corpus = set(corpus.keys())

# use known query 148 + find a few more
good_qids = []
target_qids = ['148', '3', '5', '21']
for qid in target_qids:
    if qid in qrels and qid in queries:
        good_qids.append((qid, list(qrels[qid].keys())))

print(f"Corpus docs loaded:  {len(corpus)}")
print(f"Total queries:       {len(queries)}")
print(f"Qrel queries:        {len(qrels)}")
print(f"Queries with rel in slice: {len(good_qids)}")

if not good_qids:
    print("\nNO queries have relevant docs in the first", MAX_DOCS, "docs.")
    print("Sample corpus IDs:", list(corpus.keys())[:5])
    q0 = list(qrels.keys())[0]
    print(f"Sample qrel qid={q0!r}  rel={list(qrels[q0].keys())[:3]}")
    sys.exit(0)

if not good_qids:
    print("no target queries found"); sys.exit(1)
qid, rel_ids = good_qids[0]
print(f"\nSample query qid={qid!r}")
print(f"  text: {queries[qid][:80]}")
print(f"  relevant docs in slice: {rel_ids}")

# --- ingest ---
print("\nIngesting...")
pipe = make_pipeline("scale")
from collections import Counter as _Counter
doc_tokens = {}
doc_tf = {}
doc_len = {}
doc_id_list = []
for i, (did, doc) in enumerate(corpus.items()):
    text = doc_text(doc)
    doc_id_list.append(did)
    words = tokenize_words(text)
    tf = _Counter(words)
    doc_tokens[did] = frozenset(tf.keys())
    doc_tf[did] = dict(tf)
    doc_len[did] = len(words)
    pipe.ingest_one(text)
try:
    pipe.flush()
except Exception:
    pass

cidx = build_corpus_index(doc_id_list, doc_tokens, doc_tf, doc_len)
print(f"  word_counts size: {len(pipe.registry.word_counts)}")
print(f"  doc_freq size:    {len(cidx.doc_freq)}")

# check a few query words
qwords = tokenize_words(queries[qid])
print(f"\nQuery words: {qwords}")
for w in qwords[:4]:
    print(f"  {w!r}  doc_freq={cidx.doc_freq.get(w,0)}  in_inv={w in cidx.inv}")
    if w in cidx.inv:
        print(f"    -> {len(cidx.inv[w])} docs contain it")
        print(f"    -> rel doc in inv: {any(d in cidx.inv[w] for d in rel_ids)}")

# check doc tokens for relevant doc
rel_doc = rel_ids[0]
print(f"\nRelevant doc {rel_doc!r} tokens sample: {list(cidx.doc_tokens.get(rel_doc, set()))[:8]}")

# --- build profile and score ---
neighbor_map = build_neighbor_weights(pipe.registry)
hub_sigs = build_all_hub_signatures(doc_id_list, cidx.doc_tokens, pipe.registry, top_k=12)

profile = build_query_profile(
    queries[qid], pipe.registry,
    neighbor_map=neighbor_map, doc_freq=cidx.doc_freq, n_docs=len(corpus)
)
print(f"\nProfile word_set: {profile.word_set}")
print(f"Profile idf (first 3): {dict(list(profile.idf.items())[:3])}")

# score relevant doc directly
rel_tokens = cidx.doc_tokens.get(rel_doc, frozenset())
rel_sig = hub_sigs.get(rel_doc)
s = score_document(profile, rel_tokens, rel_sig)
print(f"\nDirect score of relevant doc {rel_doc!r}: {s:.4f}")

# rank
cands = candidate_ids(profile.words, cidx.inv, neighbor_map, doc_id_list)
print(f"Candidates: {len(cands)} (relevant in cands: {any(d in cands for d in rel_ids)})")

ranked = rank_with_hub_signatures(
    profile, cands, hub_sigs, doc_id_list,
    doc_tokens=cidx.doc_tokens,
    doc_tf=cidx.doc_tf,
    doc_len=cidx.doc_len,
    avg_dl=cidx.avg_dl,
    top_k=20,
)
print(f"Rank of relevant doc: {next((i+1 for i,d in enumerate(ranked) if d in rel_ids), 'not in top 20')}")
print(f"NDCG@10: {ndcg_at_k(ranked, qrels[qid], 10):.4f}")
print(f"Top-5 ranked: {ranked[:5]}")
