#!/usr/bin/env python3
"""THE DIAL — one EdgeRAG.retrieve() API across tiers + weighted RRF fusion. Measures the encoder-free tiers
(lexical, bridged, distilled-SPLADE) and their fusion on BEIR, so you see which point of the dial wins per
corpus. All encoder-free at serve (GPU only for the one-time distilled-index build)."""
import os, sys, time
os.environ.setdefault("PYTHONUTF8", "1")
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from aethos_edge_rag import EdgeRAG
from aethos_splade_lattice import SpladeEncoder, DistilledSpladeIndex
from scripts.bench_supervised_bridges import load, ndcg10
from _fast_tok import words


def main():
    corpora = sys.argv[1].split(",") if len(sys.argv) > 1 else ["scifact", "nfcorpus", "fiqa"]
    enc = SpladeEncoder(); enc.encode(["warm"], topk=128)
    print("=" * 100)
    print(f"THE DIAL — EdgeRAG.retrieve(tier=/fuse=), encoder-free tiers (device={enc.device} for 1-time build)")
    print("=" * 100)
    tiers = ["lexical", "bridged", "distilled"]
    print(f"  {'corpus':<10}" + "".join(f"{t:>22}" for t in tiers) + f"{'fuse(brdg+dist)':>22}")
    print(f"  {'':<10}" + "".join(f"{'nDCG/R@100':>22}" for _ in tiers) + f"{'nDCG/R@100':>22}")
    for name in corpora:
        corpus, queries, train_q, test_q = load(name)
        doc_ids = list(corpus.keys()); d2i = {d: i for i, d in enumerate(doc_ids)}
        test_ids = [q for q in test_q if q in queries]
        eng = EdgeRAG().build(corpus); eng.learn_bridges(queries, train_q, corpus)
        idx = DistilledSpladeIndex().build_docs(corpus, enc, topk=128)
        idx.distill(sorted({w for q in test_ids for w in words(queries[q])}), enc, topk=64)
        eng.attach_splade(idx)

        def evl(fn):
            nd = rc = 0.0
            for qid in test_ids:
                gold = {d for d in test_q[qid] if test_q[qid].get(d, 0) > 0 and d in d2i}
                r = fn(queries[qid])
                nd += ndcg10(r[:10], test_q[qid]); rc += (len(set(r[:100]) & gold) / len(gold) if gold else 0)
            return nd / len(test_ids), rc / len(test_ids)

        cells = []
        for t in tiers:
            nd, rc = evl(lambda q, t=t: eng.retrieve(q, 100, tier=t)); cells.append(f"{nd:.4f}/{rc:.4f}")
        ndf, rcf = evl(lambda q: eng.retrieve(q, 100, fuse=["bridged", "distilled"]))
        cells.append(f"{ndf:.4f}/{rcf:.4f}")
        print(f"  {name:<10}" + "".join(f"{c:>22}" for c in cells))
    print("\n  one API: retrieve(tier='lexical'|'bridged'|'distilled'|'wand') or fuse=[...] (weighted RRF).")
    print("  all rows encoder-free at serve; pick the dial point per corpus (fusion is usually the safe default).")


if __name__ == "__main__":
    main()
