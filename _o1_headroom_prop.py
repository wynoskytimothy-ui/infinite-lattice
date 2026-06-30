#!/usr/bin/env python3
"""Does adding the PROPAGATION (hop-2) source to the full deep-pool+rerank stack capture the 60% hop-2-reachable
gold the glass-box found? Compare the 4-source stack (lexical+bridges+drift+distilled) vs 5-source (+propagation),
both deep-pooled (300) + cross-encoder reranked, at recall@{100,200,500} + nDCG@10. The glass-box says the hop-2
gold lands in the deep pool -> propagation should raise the RERANKED recall, not just the ceiling."""
import os, sys
os.environ.setdefault("PYTHONUTF8", "1"); os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from aethos_edge_rag import EdgeRAG
from aethos_splade_lattice import DistilledSpladeIndex
from scripts.bench_supervised_bridges import load, ndcg10, words
from _o1_drift import build_drift, drift_bag, score_bag
from _o1_propagate import propagate, exp_bag

CACHE = Path(os.environ.get("TEMP", "/tmp")); DEPTH = 300


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "nfcorpus"
    corpus, queries, train_q, test_q = load(name)
    doc_ids = list(corpus.keys()); d2i = {d: i for i, d in enumerate(doc_ids)}
    ids = [q for q in test_q if q in queries and any(test_q[q].get(d, 0) > 0 and d in d2i for d in test_q[q])]
    eng = EdgeRAG().build(corpus); eng.learn_bridges(queries, train_q, corpus)
    seeds = {w for q in ids for w in words(queries[q])}
    drift, _, _ = build_drift(corpus, seeds)
    hop1 = {cw for w in seeds for cw, _ in drift.get(w, ())}
    drift2, _, _ = build_drift(corpus, seeds | hop1); prop = propagate(drift2)
    spl = None; vocab = sorted(seeds)
    df = CACHE / f"splade_{name}_doc128.npz"; vf = CACHE / f"distill_{name}_v{len(vocab)}.npz"
    if df.exists() and vf.exists():
        dz = np.load(df); vz = np.load(vf, allow_pickle=True)
        spl = DistilledSpladeIndex(); spl.doc_ids = doc_ids; spl.N = len(doc_ids)
        spl._set_docs(dz["t"], dz["w"], dz["o"]); spl._set_table(vz["words"].tolist(), vz["t"], vz["w"], vz["o"])
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=256)
    print("=" * 88); print(f"PROP IN THE RERANKED STACK — {name}: {len(ids)} q, depth {DEPTH}"); print("=" * 88)

    def pools(qid):
        q = queries[qid]
        p = {"lexical": list(np.argsort(eng.score(q))[::-1][:DEPTH]),
             "bridged": [d2i[d] for d in eng.search_bridged(q, DEPTH)],
             "drift": list(np.argsort(score_bag(eng, drift_bag(q, drift2, alpha=0.4)))[::-1][:DEPTH]),
             "prop": list(np.argsort(score_bag(eng, exp_bag(q, drift2, prop, a1=0.4, a2=0.4)))[::-1][:DEPTH])}
        if spl: p["distilled"] = [d2i[d] for d in spl.search(q, DEPTH)]
        return p

    s4 = ["lexical", "bridged", "drift"] + (["distilled"] if spl else [])
    ks = [100, 200, 500]
    R = {f"{m}@{k}": 0.0 for m in ("no_prop", "with_prop") for k in ks}; nd4 = nd5 = nq = 0
    for qid in ids:
        gold = {d2i[d] for d in test_q[qid] if test_q[qid].get(d, 0) > 0 and d in d2i}
        if not gold: continue
        nq += 1
        p = pools(qid)
        for tag, srcs in [("no_prop", s4), ("with_prop", s4 + ["prop"])]:
            u = list(set().union(*[set(p[k]) for k in srcs]))
            sc = ce.predict([(queries[qid], corpus[doc_ids[i]][:512]) for i in u], batch_size=256, show_progress_bar=False)
            order = [u[j] for j in np.argsort(sc)[::-1]]
            for k in ks: R[f"{tag}@{k}"] += len(set(order[:k]) & gold) / len(gold)
            if tag == "no_prop": nd4 += ndcg10([doc_ids[i] for i in order[:10]], test_q[qid])
            else: nd5 += ndcg10([doc_ids[i] for i in order[:10]], test_q[qid])
    print(f"  {'k':>6}{'4-source':>12}{'5-source(+prop)':>18}{'gain':>10}")
    for k in ks:
        a, b = R[f"no_prop@{k}"] / nq, R[f"with_prop@{k}"] / nq
        print(f"  {k:>6}{a:>12.4f}{b:>18.4f}{b-a:>+10.4f}")
    print(f"  nDCG@10: 4-source {nd4/nq:.4f} | 5-source {nd5/nq:.4f} ({(nd5-nd4)/nq:+.4f})")
    print("  -> does the hop-2 propagation source raise the RERANKED recall (capture the glass-box's 60%)?")


if __name__ == "__main__":
    main()
