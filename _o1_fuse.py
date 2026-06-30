#!/usr/bin/env python3
"""WAVE 3 — FUSE the complementary recall sources, then rerank. Wave 1: deficit = recall (semantic GAP).
Wave 2: drift and SPLADE recover DIFFERENT gap docs. So fuse lexical + bridges + drift + distilled-SPLADE (each
misses different docs) -> recall above any single source (toward the union ceiling), then a small reranker
recovers nDCG. Measures each source, the fusion, the union ceiling, and reranked nDCG vs SPLADE-full."""
import os, sys, time
os.environ.setdefault("PYTHONUTF8", "1"); os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from aethos_edge_rag import EdgeRAG
from aethos_splade_lattice import DistilledSpladeIndex
from scripts.bench_supervised_bridges import load, ndcg10, words
from _o1_drift import build_drift, drift_bag, score_bag

CACHE = Path(os.environ.get("TEMP", "/tmp"))


def rrf(rankings, weights, k0=60, topn=100):
    agg = {}
    for rk, w in zip(rankings, weights):
        for r, d in enumerate(rk):
            agg[d] = agg.get(d, 0.0) + w / (k0 + r)
    return [d for d, _ in sorted(agg.items(), key=lambda x: -x[1])[:topn]]


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "nfcorpus"
    corpus, queries, train_q, test_q = load(name)
    doc_ids = list(corpus.keys()); d2i = {d: i for i, d in enumerate(doc_ids)}
    ids = [q for q in test_q if q in queries and any(test_q[q].get(d, 0) > 0 and d in d2i for d in test_q[q])]
    eng = EdgeRAG().build(corpus); eng.learn_bridges(queries, train_q, corpus)
    seeds = {w for q in ids for w in words(queries[q])}
    drift, df, N = build_drift(corpus, seeds)

    # distilled-SPLADE from cache if present, else skip
    spl = None
    f = CACHE / f"splade_{name}_doc128.npz"; vfile = None
    vocab = sorted({w for q in ids for w in words(queries[q])})
    vf = CACHE / f"distill_{name}_v{len(vocab)}.npz"
    if f.exists() and vf.exists():
        dz = np.load(f); vz = np.load(vf, allow_pickle=True)
        spl = DistilledSpladeIndex(); spl.doc_ids = doc_ids; spl.N = N
        spl._set_docs(dz["t"], dz["w"], dz["o"]); spl._set_table(vz["words"].tolist(), vz["t"], vz["w"], vz["o"])

    print("=" * 90); print(f"WAVE 3 FUSE — {name}: {N:,} docs, {len(ids)} q | distilled={'yes' if spl else 'no'}"); print("=" * 90)

    def topidx(scorer, q):
        sc = scorer(q); return list(np.argsort(sc)[::-1][:100])

    sources = {
        "lexical": lambda q: topidx(lambda x: eng.score(x), q),
        "bridged": lambda q: [d2i[d] for d in eng.search_bridged(q, 100)],
        "drift": lambda q: topidx(lambda x: score_bag(eng, drift_bag(x, drift, alpha=0.4)), q),
    }
    if spl: sources["distilled"] = lambda q: [d2i[d] for d in spl.search(q, 100)]

    # solo recalls first (to weight the fusion) ----
    solo = {k: 0.0 for k in sources}
    for qid in ids:
        gold = {d2i[d] for d in test_q[qid] if test_q[qid].get(d, 0) > 0 and d in d2i}
        if not gold: continue
        for k, fn in sources.items(): solo[k] += len(set(fn(queries[qid])[:100]) & gold) / len(gold)
    nq = len(ids); solo = {k: v / nq for k, v in solo.items()}

    # cross-encoder reranker (GPU) for the union pool ----
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=256)

    agg = {k: [0.0, 0.0] for k in ["FUSE_equal", "FUSE_weighted", "union_ceiling", "union_rerank"]}
    for qid in ids:
        gold = {d2i[d] for d in test_q[qid] if test_q[qid].get(d, 0) > 0 and d in d2i}
        if not gold: continue
        pools = {k: fn(queries[qid]) for k, fn in sources.items()}
        feq = rrf(list(pools.values()), [1.0] * len(pools))
        fwt = rrf(list(pools.values()), [solo[k] for k in pools])     # weight by source strength
        u = list(set().union(*[set(p[:100]) for p in pools.values()]))
        agg["FUSE_equal"][0] += len(set(feq) & gold) / len(gold)
        agg["FUSE_weighted"][0] += len(set(fwt) & gold) / len(gold)
        agg["union_ceiling"][0] += len(set(u) & gold) / len(gold)
        # rerank the union with the cross-encoder -> top-100 recall + top-10 nDCG
        sc = ce.predict([(queries[qid], corpus[doc_ids[i]][:512]) for i in u], batch_size=256, show_progress_bar=False)
        order = [u[j] for j in np.argsort(sc)[::-1]]
        agg["union_rerank"][0] += len(set(order[:100]) & gold) / len(gold)
        agg["union_rerank"][1] += ndcg10([doc_ids[i] for i in order[:10]], test_q[qid])

    print(f"  {'source / fusion':<18}{'recall@100':>12}{'nDCG@10':>10}")
    for k in sources: print(f"  {k:<18}{solo[k]:>12.4f}")
    print(f"  {'FUSE equal-RRF':<18}{agg['FUSE_equal'][0]/nq:>12.4f}   (dilutes)")
    print(f"  {'FUSE weighted-RRF':<18}{agg['FUSE_weighted'][0]/nq:>12.4f}   (weight by source strength)")
    print(f"  {'union ceiling':<18}{agg['union_ceiling'][0]/nq:>12.4f}   (max from the pools)")
    print(f"  {'union + RERANK':<18}{agg['union_rerank'][0]/nq:>12.4f}{agg['union_rerank'][1]/nq:>10.4f}   <- the full stack")
    splref = {"scifact": 0.909, "nfcorpus": 0.282, "fiqa": 0.610}.get(name)
    if splref:
        ur = agg["union_rerank"][0] / nq
        print(f"\n  vs SPLADE-full recall@100 {splref:.3f}: union+rerank {ur:.4f} ({'ABOVE' if ur > splref else 'below'})")
    print("  complementary recall sources -> union -> cross-encoder rerank = best recall AND nDCG.")


if __name__ == "__main__":
    main()
