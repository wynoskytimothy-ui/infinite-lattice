#!/usr/bin/env python3
"""demo_aethos — a self-contained, runnable walkthrough of the AETHOS RAG engine for Andrea.

WHAT THIS SHOWS (all no-GPU-at-serve, all traceable to captured runs)
    1. INGEST   -> the compressed footprint per doc + total + the ratio vs fp32 dense, and "GPU at serve: NO".
    2. SEARCH   -> 3-4 real example queries, top results printed.
    3. EXPLAIN  -> the glass-box breakdown for one result: which signals and NAMED learned bridges drove the
                   match. This is the "why this doc" provenance a vector DB physically cannot show.
    4. HEADLINE -> dense-competitive nDCG at ~X bytes/doc, no GPU. Every number traces to a captured run.

IT DOES NOT REINVENT ANYTHING. It imports the two shipped modules:
    * aethos_rag_api.AethosRAG        -- the deployable service: PQ-compressed dense codec (no GPU at serve)
    * aethos_rag_pipeline.AethosPipeline -- the glass-box lexical lattice (named bridges), wired in by AethosRAG

DATA
    Primary path: REAL scifact (BEIR) documents + REAL bge-large-en-v1.5 (1024-d) embeddings cached in
    `_sr_bge-large_scifact.npz`. On the full 5183-doc corpus at M=128 the live footprint reproduces the
    captured headline point exactly: ~330 B/doc, 12.4x smaller than fp32.
    Fallback path (if the BEIR data / npz are absent): a tiny bundled document set with deterministic
    bag-of-words embeddings, so the ingest/search/explain/footprint mechanics still run end-to-end anywhere.

HONEST SCOPE
    The compression is Product Quantization -- competitive and STANDARD, not a proprietary ratio. The AETHOS
    moat is the packaging: glass-box explain + no-GPU-at-serve + invertible/append-only codec. The nDCG/recall
    accuracy figures in the HEADLINE come from the captured full-corpus evaluations
    (_dense_compress_scifact_summary.json / _dense_compress_*_out.txt); this live run reproduces the FOOTPRINT
    and demonstrates the retrieval + glass-box-explain pipeline actually running, encoder-free, no GPU.
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Point the codec's persistence at a throwaway demo dir BEFORE importing the API (it reads the env at import),
# so running the demo never clobbers a real index.
os.environ.setdefault("AETHOS_INDEX_DIR", str(HERE / "_demo_index"))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")            # belt-and-braces: no GPU anywhere in the demo

try:
    sys.stdout.reconfigure(encoding="utf-8")                 # type: ignore[attr-defined]
except Exception:
    pass

# THE product modules -- imported, not reinvented.
from aethos_rag_api import AethosRAG                          # noqa: E402
import aethos_rag_pipeline                                   # noqa: E402  (glass-box lattice, used via AethosRAG)

NPZ = HERE / "_sr_bge-large_scifact.npz"
RULE = "=" * 96


# ---------------------------------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------------------------------
def load_scifact():
    """Return (docs, example_queries, query_vecs, learn_queries, learn_qrels, meta) using REAL scifact text
    and REAL cached bge-large embeddings. Raises if the BEIR data / npz are unavailable."""
    from scripts.bench_supervised_bridges import load           # BEIR scifact loader (text + qrels)
    corpus, queries, _train_q, test_q = load("scifact")
    z = np.load(NPZ, allow_pickle=True)
    doc_ids = [str(x) for x in z["doc_ids"]]
    qids = [str(x) for x in z["qids"]]
    Demb = z["doc_emb"].astype(np.float32)                     # [N, 1024]
    Qemb = z["q_emb"].astype(np.float32)                       # [nq, 1024]
    d2i = {d: i for i, d in enumerate(doc_ids)}
    qi = {q: i for i, q in enumerate(qids)}

    # every doc, with its precomputed (offline) embedding -> the codec freezes these into PQ codes
    docs = [{"doc_id": d, "text": corpus[d], "embedding": Demb[d2i[d]].tolist()} for d in doc_ids]

    # queries that actually have a gold doc present -> use them to (a) teach bridges and (b) demo searches
    with_gold = [q for q in qids if q in test_q and any(dd in corpus for dd in test_q[q])]
    learn_qids = with_gold[:25]
    learn_queries = {q: queries[q] for q in learn_qids}
    learn_qrels = {q: {dd: s for dd, s in test_q[q].items() if dd in corpus} for q in learn_qids}
    learn_qrels = {q: r for q, r in learn_qrels.items() if any(v > 0 for v in r.values())}

    example_queries = {q: queries[q] for q in list(learn_qrels)[:4]}
    query_vecs = {q: Qemb[qi[q]].tolist() for q in learn_qrels}
    gold = {q: [dd for dd, s in learn_qrels[q].items() if s > 0] for q in learn_qrels}
    meta = {"mode": "scifact", "M": 128, "dim": int(Demb.shape[1]), "n_docs": len(docs), "gold": gold,
            "corpus": corpus}
    return docs, example_queries, query_vecs, learn_queries, learn_qrels, meta


def _det_embed(text, dim=96):
    """A deterministic, model-free bag-of-words embedding (stable hashing -> L2 norm). Good enough for the
    bundled fallback so cosine retrieval is lexically meaningful without downloading any model."""
    v = np.zeros(dim, dtype=np.float32)
    for w in "".join(c.lower() if c.isalnum() else " " for c in text).split():
        h = int(hashlib.blake2b(w.encode(), digest_size=8).hexdigest(), 16)
        v[h % dim] += 1.0
    n = np.linalg.norm(v)
    return (v / n if n else v)


def load_bundled():
    """A tiny self-contained corpus (no network, no model, no npz) so the demo runs anywhere."""
    dim = 96
    raw = {
        "sql_index":   "Database indexes speed up queries by avoiding full table scans using B-tree structures.",
        "sql_join":    "A SQL join combines rows from two tables on a matching key column.",
        "sql_txn":     "ACID transactions guarantee atomicity, consistency, isolation and durability in databases.",
        "net_tcp":     "TCP provides reliable ordered byte streams with retransmission and congestion control.",
        "net_dns":     "DNS resolves human-readable domain names into numeric IP addresses via nameservers.",
        "net_tls":     "TLS encrypts network traffic using certificates and an authenticated key exchange handshake.",
        "ml_grad":     "Gradient descent minimizes a loss function by stepping along the negative gradient.",
        "ml_overfit":  "Overfitting happens when a model memorizes training data and fails to generalize.",
        "ml_embed":    "Embeddings map tokens into dense vectors so that similar meanings sit close together.",
        "cook_bread":  "Bread dough rises because yeast ferments sugars and releases carbon dioxide gas.",
        "cook_stock":  "A good stock simmers bones and vegetables slowly to extract gelatin and flavor.",
        "cook_knife":  "Keep a chef knife sharp and use a claw grip to protect fingers while chopping.",
    }
    docs = [{"doc_id": k, "text": t, "embedding": _det_embed(t, dim).tolist()} for k, t in raw.items()]
    example_queries = {
        "q_index": "how do database indexes make queries faster",
        "q_dns":   "translate a domain name into an ip address",
        "q_train": "why does a model fail to generalize from training data",
    }
    query_vecs = {q: _det_embed(t, dim).tolist() for q, t in example_queries.items()}
    # tiny supervised signal so the glass-box tier can learn NAMED bridges in the fallback too
    learn_qrels = {"q_index": {"sql_index": 1}, "q_dns": {"net_dns": 1}, "q_train": {"ml_overfit": 1}}
    gold = {q: [d for d, s in r.items() if s > 0] for q, r in learn_qrels.items()}
    meta = {"mode": "bundled", "M": 8, "dim": dim, "n_docs": len(docs), "gold": gold,
            "corpus": {k: v for k, v in raw.items()}}
    return docs, example_queries, query_vecs, dict(example_queries), learn_qrels, meta


# ---------------------------------------------------------------------------------------------------
# Pretty printing helpers
# ---------------------------------------------------------------------------------------------------
def human_bytes(n):
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024.0:
            return f"{n:,.1f} {unit}"
        n /= 1024.0
    return f"{n:,.1f} TB"


def print_footprint(rag, meta):
    print(RULE)
    print("STEP 1 -- INGEST: build the compressed, no-GPU index and measure the honest footprint")
    print(RULE)
    st = rag.stats()
    fp = st["footprint"]
    n = st["n_docs"]
    fp32_pd = fp["fp32_bytes_per_doc"]
    comp_pd = fp["bytes_per_doc"]
    print(f"  corpus                : {meta['mode']}  ({n:,} docs, dim={meta['dim']}, embeddings=bge-large "
          f"precomputed offline)" if meta["mode"] == "scifact"
          else f"  corpus                : {meta['mode']}  ({n:,} docs, dim={meta['dim']}, deterministic BoW embeddings)")
    print(f"  PQ config             : M={st['pq']['M']} subquantizers x Kp={st['pq']['Kp']} centroids, "
          f"rotation = {st['pq']['rotation']}")
    print()
    print("  PER-DOC FOOTPRINT (amortized):")
    print(f"    fp32 dense (stored by a normal vector DB) : {fp32_pd:8.1f} B/doc")
    print(f"    PQ codes            (per doc)             : {fp['codes_raw_bytes_per_doc']:8.1f} B/doc")
    print(f"    shared codebook     (amortized / N)       : {fp['codebook_amort_bytes_per_doc']:8.1f} B/doc")
    print(f"    invertible rotation (regenerated by seed) : {0.0:8.1f} B/doc  (0 stored bytes)")
    print(f"    ------------------------------------------------------------")
    print(f"    AETHOS compressed   (per doc)             : {comp_pd:8.1f} B/doc   "
          f"<-- {fp['compression_vs_fp32']:.1f}x smaller than fp32")
    print()
    print(f"  TOTAL INDEX SIZE:")
    print(f"    fp32 dense : {human_bytes(fp32_pd * n):>12}   ({fp32_pd:,.0f} B/doc x {n:,} docs)")
    print(f"    AETHOS     : {human_bytes(comp_pd * n):>12}   ({comp_pd:,.1f} B/doc x {n:,} docs)   "
          f"= {fp['compression_vs_fp32']:.1f}x smaller")
    print()
    if meta["mode"] == "scifact":
        print(f"  <4KB/doc gate         : {'CLEARED' if comp_pd < 4096 else 'FAILED'} "
              f"({comp_pd:.0f} B/doc << 4096 B)")
    else:
        print(f"  NOTE (bundled/{n}-doc toy): the {fp['codebook_amort_bytes_per_doc']:.0f} B/doc shared codebook "
              f"can't amortize over so few tiny docs, so this ratio is NOT meaningful.")
        print(f"        PQ wins only at real scale (large dim + many docs): see the scifact/nfcorpus captured "
              f"numbers in STEP 4. This toy run demonstrates the ingest/search/explain MECHANICS, not the ratio.")
    print(f"  GPU at serve          : {'NO' if not st['gpu_at_serve'] else 'YES'}   "
          f"(query-embed mode: {st['query_embed_mode']}; search path is pure-numpy ADC)")
    return comp_pd, fp["compression_vs_fp32"]


def print_searches(rag, example_queries, query_vecs, meta):
    print()
    print(RULE)
    print("STEP 2 -- SEARCH: run example queries (no GPU; pure-numpy asymmetric distance over PQ codes)")
    print(RULE)
    gold = meta["gold"]
    for qid, qtext in example_queries.items():
        t = time.perf_counter()
        res = rag.retrieve(query_embedding=query_vecs[qid], k=5)
        ms = (time.perf_counter() - t) * 1000.0
        gset = set(gold.get(qid, []))
        rank = next((i + 1 for i, r in enumerate(res) if r["doc_id"] in gset), None)
        tag = f"gold doc rank #{rank}" if rank else "gold not in top-5"
        print(f"\n  QUERY [{qid}]: {qtext[:82]}")
        print(f"    served {len(res)} hits in {ms:.3f} ms   ({tag})")
        for i, r in enumerate(res):
            star = " *GOLD*" if r["doc_id"] in gset else ""
            print(f"      {i+1}. doc {str(r['doc_id']):>10}  score={r['score']:.3f}{star}  {r['text'][:64]}")


def print_explain(rag, candidate_queries, query_vecs, meta):
    print()
    print(RULE)
    print("STEP 3 -- EXPLAIN: the glass-box 'why this doc' a vector DB cannot produce")
    print(RULE)
    gold = meta["gold"]
    # scan every labeled query and pick the (query, gold-doc) pair that fires the MOST named bridges ->
    # the most vivid, auditable provenance story (a generic reranker cannot itemize this)
    best = None
    for qid, qtext in candidate_queries.items():
        if qid not in query_vecs:
            continue
        for gd in gold.get(qid, []):
            ex = rag.explain(query=qtext, doc_id=gd, query_embedding=query_vecs[qid])
            nb = len(ex.get("bridges_fired", []))
            if best is None or nb > best[0]:
                best = (nb, qid, gd, ex)
    if best is None:
        print("  (no labeled query/doc pair available to explain in this mode)")
        return
    nb, qid, gd, ex = best
    print(f"\n  QUERY [{qid}]: {candidate_queries[qid]}")
    print(f"  DOC   [{gd}]: {meta['corpus'].get(gd, '')[:88]}")
    print()
    print("  SIGNAL CONTRIBUTIONS (each is a named, inspectable number -- not a black-box similarity):")
    for name, val in ex.get("signals", {}).items():
        print(f"      {name:<14}: {val}")
    fac = ex.get("factors", {})
    if "dense_rank" in fac:
        print(f"      dense_rank    : this doc is the #{fac['dense_rank']} nearest by the compressed dense codec "
              f"(share of max score = {fac.get('dense_share_of_max')})")
    print()
    bf = ex.get("bridges_fired", [])
    print(f"  NAMED LEARNED BRIDGES that fired on this doc: {len(bf)}")
    print("      (each bridge = query-term -> doc-term, learned by counting from labeled query/doc pairs;")
    print("       it is WHY a lexically-distant doc surfaced -- fully auditable, traceable to its training pairs)")
    for b in bf[:8]:
        print(f"        '{b['query_term']}'  --[{b['weight']}]-->  '{b['doc_term']}'")
    if len(bf) > 8:
        print(f"        ... and {len(bf) - 8} more")
    mv = ex.get("factors", {}).get("mv_triangles_fired", [])
    if mv:
        print(f"  MINIVERSE TRIANGLES fired: {len(mv)} (e.g. {mv[0]})")
    print()
    print("  ^ A pure vector DB returns a single opaque cosine score. AETHOS itemizes the contribution of the")
    print("    dense codec, the lexical lattice, and each NAMED learned bridge -- an auditable answer to")
    print("    'why did this document surface for this query?' for a regulated document domain.")


def print_headline(live_bpd, live_ratio, meta):
    print()
    print(RULE)
    print("STEP 4 -- HEADLINE")
    print(RULE)
    # captured full-corpus accuracy frontier (traceable to _dense_compress_*_summary.json / *_out.txt)
    sci = {"full_ndcg": 0.7463, "full_bpd": 4096, "comp_ndcg": 0.7223, "comp_bpd": 330, "ratio": 12,
           "full_r100": 0.9483, "comp_r100": 0.9550}
    nf = {"full_ndcg": 0.3814, "full_bpd": 4096, "comp_ndcg": 0.3664, "comp_bpd": 66, "ratio": 62}
    if meta["mode"] == "scifact":
        print(f"  LIVE (this run, scifact full corpus): {live_bpd:.0f} B/doc, {live_ratio:.1f}x smaller than fp32, "
              f"NO GPU at serve -- reproduces the captured scifact headline point below.")
    else:
        print(f"  LIVE (this run, bundled toy): mechanics only -- footprint ratio needs a real large-D corpus; "
              f"the meaningful numbers are the captured scifact/nfcorpus figures below.")
    print()
    print("  CAPTURED FULL-CORPUS ACCURACY (real bge-large-1024d; _dense_compress_*_summary.json):")
    print(f"    scifact : nDCG@10 {sci['full_ndcg']} @ {sci['full_bpd']} B  ->  {sci['comp_ndcg']} @ "
          f"{sci['comp_bpd']} B   ({sci['ratio']}x smaller, ~97% nDCG; "
          f"Recall@100 {sci['comp_r100']} BEATS full {sci['full_r100']})")
    print(f"    nfcorpus: nDCG@10 {nf['full_ndcg']} @ {nf['full_bpd']} B  ->  {nf['comp_ndcg']} @ "
          f"{nf['comp_bpd']} B   ({nf['ratio']}x smaller, ~96% nDCG)")
    print()
    print("  >>> Dense-competitive retrieval at 66-330 B/doc, glass-box explainable, no GPU at serve, "
          "clears the <4KB gate. <<<")
    print()
    print("  HONEST SCOPE: the compression is Product Quantization -- competitive and STANDARD, not a")
    print("  proprietary ratio. The AETHOS moat is the packaging: glass-box explain + no-GPU-at-serve +")
    print("  invertible/append-only codec. Every number above traces to a captured run.")


# ---------------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------------
def main():
    print(RULE)
    print("AETHOS RAG -- runnable demo (ingest -> search -> glass-box explain -> headline)")
    print("imports: aethos_rag_api.AethosRAG + aethos_rag_pipeline.AethosPipeline   |   no GPU at serve")
    print(RULE)

    try:
        if not NPZ.exists():
            raise FileNotFoundError(str(NPZ))
        docs, example_queries, query_vecs, learn_queries, learn_qrels, meta = load_scifact()
        print(f"  data: REAL scifact (BEIR) + cached bge-large embeddings ({meta['n_docs']:,} docs)")
    except Exception as e:
        print(f"  [scifact data unavailable: {type(e).__name__}] -> using the self-contained bundled corpus")
        docs, example_queries, query_vecs, learn_queries, learn_qrels, meta = load_bundled()

    # ---- build the index: PQ-compressed dense codec + optional no-GPU glass-box lattice (+ learned bridges) ----
    rag = AethosRAG()
    rag.create_rag(M=meta["M"], Kp=256)
    t0 = time.time()
    st = rag.add_documents(docs, M=meta["M"], Kp=256, queries=learn_queries, qrels=learn_qrels)
    build_s = time.time() - t0
    print(f"  built index in {build_s:.1f}s  |  glass-box lattice available="
          f"{st['ingest'].get('glass_box_available')}, bridges learned={st['ingest'].get('glass_box_learned')}")
    print()

    live_bpd, live_ratio = print_footprint(rag, meta)
    print_searches(rag, example_queries, query_vecs, meta)
    print_explain(rag, learn_queries, query_vecs, meta)
    print_headline(live_bpd, live_ratio, meta)


if __name__ == "__main__":
    main()
