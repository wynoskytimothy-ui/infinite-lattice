#!/usr/bin/env python3
"""WAVE 3b — CAPTURE THE UNION HEADROOM. Wave 3 found the 4 recall sources are complementary (union ceiling
0.352) but the top-100 rerank captured only 0.285. The gold is in the pool, ranked past 100. Two levers:
DEEPER source pools (more gold into the union) + measuring recall@k (the gold is just past the cutoff). Take
top-300 from each source, union, cross-encoder rerank, report recall@{100,200,500} + nDCG@10 vs the ceiling."""
import os, sys
os.environ.setdefault("PYTHONUTF8", "1"); os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from aethos_edge_rag import EdgeRAG
from aethos_splade_lattice import DistilledSpladeIndex
from scripts.bench_supervised_bridges import load, ndcg10, words
from _o1_drift import build_drift, drift_bag, score_bag

CACHE = Path(os.environ.get("TEMP", "/tmp"))
DEPTH = 300


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "nfcorpus"
    corpus, queries, train_q, test_q = load(name)
    doc_ids = list(corpus.keys()); d2i = {d: i for i, d in enumerate(doc_ids)}
    ids = [q for q in test_q if q in queries and any(test_q[q].get(d, 0) > 0 and d in d2i for d in test_q[q])]
    eng = EdgeRAG().build(corpus); eng.learn_bridges(queries, train_q, corpus)
    drift, df, N = build_drift(corpus, {w for q in ids for w in words(queries[q])})

    spl = None
    f = CACHE / f"splade_{name}_doc128.npz"; vocab = sorted({w for q in ids for w in words(queries[q])})
    vf = CACHE / f"distill_{name}_v{len(vocab)}.npz"
    if f.exists() and vf.exists():
        dz = np.load(f); vz = np.load(vf, allow_pickle=True)
        spl = DistilledSpladeIndex(); spl.doc_ids = doc_ids; spl.N = N
        spl._set_docs(dz["t"], dz["w"], dz["o"]); spl._set_table(vz["words"].tolist(), vz["t"], vz["w"], vz["o"])

    from sentence_transformers import CrossEncoder
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=256)
    print("=" * 90); print(f"WAVE 3b HEADROOM — {name}: {N:,} docs, {len(ids)} q | depth={DEPTH}, distilled={'y' if spl else 'n'}"); print("=" * 90)

    def srcs(q):
        out = [list(np.argsort(eng.score(q))[::-1][:DEPTH]),
               [d2i[d] for d in eng.search_bridged(q, DEPTH)],
               list(np.argsort(score_bag(eng, drift_bag(q, drift, alpha=0.4)))[::-1][:DEPTH])]
        if spl: out.append([d2i[d] for d in spl.search(q, DEPTH)])
        return out

    ks = [100, 200, 500]
    R = {f"rerank@{k}": 0.0 for k in ks}; R["ceiling@300pool"] = 0.0; R["lexical@100"] = 0.0; ndcg = 0.0
    for qid in ids:
        gold = {d2i[d] for d in test_q[qid] if test_q[qid].get(d, 0) > 0 and d in d2i}
        if not gold: continue
        pools = srcs(queries[qid])
        u = list(set().union(*[set(p) for p in pools]))
        R["ceiling@300pool"] += len(set(u) & gold) / len(gold)
        R["lexical@100"] += len(set(pools[0][:100]) & gold) / len(gold)
        sc = ce.predict([(queries[qid], corpus[doc_ids[i]][:512]) for i in u], batch_size=256, show_progress_bar=False)
        order = [u[j] for j in np.argsort(sc)[::-1]]
        for k in ks: R[f"rerank@{k}"] += len(set(order[:k]) & gold) / len(gold)
        ndcg += ndcg10([doc_ids[i] for i in order[:10]], test_q[qid])
    nq = len(ids)
    print(f"  {'metric':<22}{'recall':>10}")
    print(f"  {'lexical recall@100':<22}{R['lexical@100']/nq:>10.4f}   (baseline)")
    for k in ks: print(f"  {'fused+rerank recall@'+str(k):<22}{R[f'rerank@{k}']/nq:>10.4f}")
    print(f"  {'union ceiling@300':<22}{R['ceiling@300pool']/nq:>10.4f}   (max from depth-300 pools)")
    print(f"  reranked nDCG@10: {ndcg/nq:.4f}")
    base = R['lexical@100']/nq
    print(f"\n  CAPTURED: recall@100 {base:.3f} -> {R['rerank@100']/nq:.3f}; @200 {R['rerank@200']/nq:.3f}; "
          f"@500 {R['rerank@500']/nq:.3f} (ceiling {R['ceiling@300pool']/nq:.3f}).")
    print("  deeper pools raise the ceiling; recall@k shows how much gold sits just past 100.")


if __name__ == "__main__":
    main()
