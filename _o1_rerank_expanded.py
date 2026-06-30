#!/usr/bin/env python3
"""Glass-box-targeted at the reranker bottleneck: the hop-2 gold is in the deep pool but the CE ranks it low
because the doc shares few LITERAL query words. Test feeding the CE the EXPANDED query (raw query + its drift
bridge terms) so it can match the doc's hop-2 terms -> does it surface the deep hop-2 gold to top-100?"""
import os, sys
os.environ.setdefault("PYTHONUTF8", "1"); os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
import numpy as np
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from aethos_edge_rag import EdgeRAG
from aethos_splade_lattice import DistilledSpladeIndex
from scripts.bench_supervised_bridges import load, ndcg10, words
from _o1_drift import build_drift, drift_bag, score_bag

CACHE = Path(os.environ.get("TEMP", "/tmp")); DEPTH = 300


def expand_text(query, drift, n=12):
    qw = list(dict.fromkeys(words(query))); acc = Counter()
    for w in qw:
        for cw, pmi in drift.get(w, ()):
            if cw not in qw: acc[cw] += pmi
    return query + " " + " ".join(cw for cw, _ in acc.most_common(n))


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "nfcorpus"
    corpus, queries, train_q, test_q = load(name)
    doc_ids = list(corpus.keys()); d2i = {d: i for i, d in enumerate(doc_ids)}
    ids = [q for q in test_q if q in queries and any(test_q[q].get(d, 0) > 0 and d in d2i for d in test_q[q])]
    eng = EdgeRAG().build(corpus); eng.learn_bridges(queries, train_q, corpus)
    seeds = {w for q in ids for w in words(queries[q])}; drift, _, _ = build_drift(corpus, seeds)
    spl = None; vocab = sorted(seeds)
    df = CACHE / f"splade_{name}_doc128.npz"; vf = CACHE / f"distill_{name}_v{len(vocab)}.npz"
    if df.exists() and vf.exists():
        dz = np.load(df); vz = np.load(vf, allow_pickle=True)
        spl = DistilledSpladeIndex(); spl.doc_ids = doc_ids; spl.N = len(doc_ids)
        spl._set_docs(dz["t"], dz["w"], dz["o"]); spl._set_table(vz["words"].tolist(), vz["t"], vz["w"], vz["o"])
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=256)
    print("=" * 84); print(f"RERANK WITH EXPANDED QUERY — {name}: {len(ids)} q"); print("=" * 84)

    R = {"raw": [0.0, 0.0], "expanded": [0.0, 0.0], "avg": [0.0, 0.0]}; nq = 0
    for qid in ids:
        gold = {d2i[d] for d in test_q[qid] if test_q[qid].get(d, 0) > 0 and d in d2i}
        if not gold: continue
        nq += 1
        q = queries[qid]
        pools = [list(np.argsort(eng.score(q))[::-1][:DEPTH]), [d2i[d] for d in eng.search_bridged(q, DEPTH)],
                 list(np.argsort(score_bag(eng, drift_bag(q, drift, alpha=0.4)))[::-1][:DEPTH])]
        if spl: pools.append([d2i[d] for d in spl.search(q, DEPTH)])
        u = list(set().union(*[set(p) for p in pools]))
        texts = [corpus[doc_ids[i]][:512] for i in u]
        qx = expand_text(q, drift)
        sc_r = ce.predict([(q, t) for t in texts], batch_size=256, show_progress_bar=False)
        sc_x = ce.predict([(qx, t) for t in texts], batch_size=256, show_progress_bar=False)
        sc_a = 0.5 * np.array(sc_r) + 0.5 * np.array(sc_x)              # average the two rerank views
        for tag, sc in [("raw", sc_r), ("expanded", sc_x), ("avg", sc_a)]:
            order = [u[j] for j in np.argsort(sc)[::-1]]
            R[tag][0] += len(set(order[:100]) & gold) / len(gold)
            R[tag][1] += ndcg10([doc_ids[i] for i in order[:10]], test_q[qid])
    print(f"  {'rerank query':<22}{'recall@100':>12}{'nDCG@10':>10}")
    for tag in ["raw", "expanded", "avg"]:
        print(f"  {tag:<22}{R[tag][0]/nq:>12.4f}{R[tag][1]/nq:>10.4f}")
    print("  does giving the cross-encoder the bridge terms surface the deep hop-2 gold (the reranker bottleneck)?")


if __name__ == "__main__":
    main()
