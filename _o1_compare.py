#!/usr/bin/env python3
"""EDGE RAG vs COMPETITORS — same corpora, same eval harness, measured locally (fair head-to-head).
Measures every step (ingest, query), footprint, accuracy for:
  * EdgeRAG  (this engine: numba ingest + mmap CSR + bridges, CPU, no GPU)
  * BM25     (rank_bm25 Okapi, CPU)            -- the lexical baseline (Lucene/Elasticsearch family)
  * Dense    (sentence-transformers + faiss)   -- the vector-DB family (ChromaDB/FAISS), needs an encoder
Published BEIR numbers for SPLADE/ColBERT are added in the writeup (cited, not measured here)."""
import os, sys, time, pickle, tempfile
os.environ.setdefault("PYTHONUTF8", "1")
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from aethos_edge_rag import EdgeRAG
from _fast_tok import words
from scripts.bench_supervised_bridges import load, ndcg10


def dir_bytes(p):
    return sum(os.path.getsize(os.path.join(p, f)) for f in os.listdir(p))


def measure_edgerag(corpus, queries, train_q, test_q, test_ids):
    EdgeRAG().build({"_w": "warm jit"})                       # warm
    eng = EdgeRAG().build(corpus)
    ing = eng.ingest_s
    eng.learn_bridges(queries, train_q, corpus)
    p = os.path.join(tempfile.gettempdir(), "cmp_edge");
    import shutil; shutil.rmtree(p, ignore_errors=True)
    eng.save_mmap(p); bdoc = dir_bytes(p) / len(corpus)
    def ev(fn):
        nd = 0.0; lat = []
        for q in test_ids:
            t0 = time.perf_counter(); r = fn(queries[q]); lat.append((time.perf_counter()-t0)*1000)
            nd += ndcg10(r, test_q[q])
        return nd/len(test_ids), float(np.median(lat))
    nl, ml = ev(lambda q: eng.search(q, 10))
    nb, mb = ev(lambda q: eng.search_bridged(q, 10))
    return dict(ingest=ing, bdoc=bdoc, q_ms=mb, ndcg=nb, ndcg_lex=nl, q_ms_lex=ml)


def measure_bm25(corpus, queries, test_q, test_ids):
    from rank_bm25 import BM25Okapi
    doc_ids = list(corpus.keys())
    toks = [words(corpus[d]) for d in doc_ids]
    t0 = time.perf_counter(); bm = BM25Okapi(toks); ing = time.perf_counter()-t0
    bdoc = len(pickle.dumps(bm)) / len(corpus)
    lat = []; nd = 0.0
    for q in test_ids:
        qt = words(queries[q])
        t0 = time.perf_counter()
        sc = bm.get_scores(qt); top = np.argpartition(sc, -10)[-10:]; top = top[np.argsort(sc[top])[::-1]]
        ranked = [doc_ids[i] for i in top if sc[i] > 0]
        lat.append((time.perf_counter()-t0)*1000)
        nd += ndcg10(ranked, test_q[q])
    return dict(ingest=ing, bdoc=bdoc, q_ms=float(np.median(lat)), ndcg=nd/len(test_ids))


def measure_dense(corpus, queries, test_q, test_ids, model="all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer
    import faiss
    m = SentenceTransformer(model)
    dim = m.get_sentence_embedding_dimension()
    doc_ids = list(corpus.keys()); texts = [corpus[d] for d in doc_ids]
    t0 = time.perf_counter()
    emb = m.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True).astype(np.float32)
    idx = faiss.IndexFlatIP(dim); idx.add(emb)
    ing = time.perf_counter() - t0
    bdoc = emb.nbytes / len(corpus)                          # fp32 dense vectors (the index)
    lat = []; nd = 0.0
    for q in test_ids:
        t0 = time.perf_counter()
        qv = m.encode([queries[q]], normalize_embeddings=True).astype(np.float32)
        D, I = idx.search(qv, 10)
        ranked = [doc_ids[i] for i in I[0]]
        lat.append((time.perf_counter()-t0)*1000)
        nd += ndcg10(ranked, test_q[q])
    return dict(ingest=ing, bdoc=bdoc, q_ms=float(np.median(lat)), ndcg=nd/len(test_ids), dim=dim)


def main():
    do_dense = "--no-dense" not in sys.argv
    corpora = sys.argv[1].split(",") if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else ["scifact", "nfcorpus", "fiqa"]
    print("=" * 100)
    print("EDGE RAG vs COMPETITORS — same corpora + same eval, measured locally (CPU). nDCG@10, held-out test.")
    print("=" * 100)
    for name in corpora:
        corpus, queries, train_q, test_q = load(name)
        test_ids = [q for q in test_q if q in queries]
        n = len(corpus)
        print(f"\n  {name}: {n:,} docs, {len(test_ids)} test queries")
        print(f"    {'system':<26}{'ingest':>12}{'B/doc':>9}{'query':>10}{'nDCG@10':>10}{'GPU?':>7}")
        e = measure_edgerag(corpus, queries, train_q, test_q, test_ids)
        print(f"    {'EdgeRAG (word+bridges)':<26}{e['ingest']*1000:>9.0f}ms{e['bdoc']:>9.0f}{e['q_ms']:>8.2f}ms{e['ndcg']:>10.4f}{'no':>7}")
        print(f"    {'  EdgeRAG lexical only':<26}{e['ingest']*1000:>9.0f}ms{e['bdoc']:>9.0f}{e['q_ms_lex']:>8.2f}ms{e['ndcg_lex']:>10.4f}{'no':>7}")
        try:
            b = measure_bm25(corpus, queries, test_q, test_ids)
            print(f"    {'BM25 (rank_bm25 Okapi)':<26}{b['ingest']*1000:>9.0f}ms{b['bdoc']:>9.0f}{b['q_ms']:>8.2f}ms{b['ndcg']:>10.4f}{'no':>7}")
        except Exception as ex:
            print(f"    BM25 failed: {ex}")
        if do_dense and (name != "fiqa"):                    # dense encode too slow on CPU for 57k docs
            try:
                d = measure_dense(corpus, queries, test_q, test_ids)
                print(f"    {'Dense MiniLM-L6 + FAISS':<26}{d['ingest']*1000:>9.0f}ms{d['bdoc']:>9.0f}{d['q_ms']:>8.2f}ms{d['ndcg']:>10.4f}{'enc':>7}")
            except Exception as ex:
                print(f"    Dense failed/skipped: {str(ex)[:60]}")
        elif name == "fiqa":
            print(f"    {'Dense MiniLM-L6 + FAISS':<26}{'(skip: CPU encode of 57k docs too slow; needs GPU)':>0}")
    print("\n  ingest = build index from raw text (EdgeRAG warmed JIT). query = median/q. B/doc = serialized index.")
    print("  EdgeRAG + Dense are no-CE; SPLADE/ColBERT (cited in writeup) need a GPU encoder.")


if __name__ == "__main__":
    main()
