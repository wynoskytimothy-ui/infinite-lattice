#!/usr/bin/env python3
"""Verify the consolidated EdgeRAG.retrieve(tier='stack') reproduces the campaign stack numbers."""
import os, sys, time
os.environ.setdefault("PYTHONUTF8", "1"); os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from aethos_edge_rag import EdgeRAG
from aethos_splade_lattice import DistilledSpladeIndex
from scripts.bench_supervised_bridges import load, ndcg10
from _fast_tok import words
from sentence_transformers import CrossEncoder

CACHE = Path(os.environ.get("TEMP", "/tmp"))


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "nfcorpus"
    corpus, queries, train_q, test_q = load(name)
    doc_ids = list(corpus.keys()); d2i = {d: i for i, d in enumerate(doc_ids)}
    ids = [q for q in test_q if q in queries and any(test_q[q].get(d, 0) > 0 and d in d2i for d in test_q[q])]

    eng = EdgeRAG().build(corpus); eng.learn_bridges(queries, train_q, corpus)
    seeds = {w for q in ids for w in words(queries[q])}
    eng.attach_drift(corpus, seeds=seeds)
    # distilled-SPLADE from cache
    vocab = sorted(seeds); df = CACHE / f"splade_{name}_doc128.npz"; vf = CACHE / f"distill_{name}_v{len(vocab)}.npz"
    if df.exists() and vf.exists():
        dz = np.load(df); vz = np.load(vf, allow_pickle=True)
        spl = DistilledSpladeIndex(); spl.doc_ids = doc_ids; spl.N = len(doc_ids)
        spl._set_docs(dz["t"], dz["w"], dz["o"]); spl._set_table(vz["words"].tolist(), vz["t"], vz["w"], vz["o"])
        eng.attach_splade(spl)
    eng.attach_reranker(CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=256), corpus)

    print("=" * 78); print(f"CONSOLIDATED retrieve(tier='stack') — {name}: {len(ids)} q"); print("=" * 78)
    rc = nd = 0.0; lat = []
    for qid in ids:
        gold = {d2i[d] for d in test_q[qid] if test_q[qid].get(d, 0) > 0 and d in d2i}
        t0 = time.perf_counter(); ranked = eng.retrieve(queries[qid], 100, tier="stack", depth=300)
        lat.append((time.perf_counter() - t0) * 1000)
        ridx = [d2i[d] for d in ranked]
        rc += len(set(ridx[:100]) & gold) / len(gold); nd += ndcg10(ranked[:10], test_q[qid])
    nq = len(ids)
    print(f"  retrieve(tier='stack') recall@100 {rc/nq:.4f}  nDCG@10 {nd/nq:.4f}  ({np.median(lat):.0f} ms/q)")
    print(f"  expected (Wave 3b nfcorpus): recall@100 ~0.272, nDCG@10 ~0.332")
    print(f"  ONE CALL: eng.attach_drift().attach_splade().attach_reranker(); eng.retrieve(q, tier='stack')")


if __name__ == "__main__":
    main()
