#!/usr/bin/env python3
"""MINI-VERSE per node = within-document passage retrieval (Timothy's recursive lattice aimed at long docs).
Each doc node seeds a sub-lattice over its PASSAGES; the doc's score = the MEET (max) over its passage scores,
so the best-matching passage represents the doc instead of the diluted whole-doc bag. Tests whether two-level
(doc->passage) beats flat whole-doc retrieval, and where (long-doc corpora)."""
import os, sys, time
os.environ.setdefault("PYTHONUTF8", "1")
import numpy as np
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from aethos_edge_rag import EdgeRAG
from scripts.bench_supervised_bridges import load, ndcg10
from _fast_tok import words

CHUNK = 50   # words per passage


def passage_split(corpus):
    passages, parent = {}, {}
    for d, txt in corpus.items():
        toks = txt.split()
        n = max(1, (len(toks) + CHUNK - 1) // CHUNK)
        for i in range(n):
            pid = f"{d}#p{i}"; passages[pid] = " ".join(toks[i * CHUNK:(i + 1) * CHUNK]); parent[pid] = d
    return passages, parent


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "scifact"
    corpus, queries, train_q, test_q = load(name)
    doc_ids = list(corpus.keys()); d2i = {d: i for i, d in enumerate(doc_ids)}
    ids = [q for q in test_q if q in queries and any(test_q[q].get(d, 0) > 0 and d in d2i for d in test_q[q])]
    avg_len = np.mean([len(corpus[d].split()) for d in corpus])

    eng_d = EdgeRAG().build(corpus)                                 # flat whole-doc
    passages, parent = passage_split(corpus)
    eng_p = EdgeRAG().build(passages)                               # passage mini-lattices
    pass_ids = eng_p.doc_ids
    # map: doc index -> list of passage indices (its mini-verse)
    doc_pass = defaultdict(list)
    for pi, pid in enumerate(pass_ids): doc_pass[d2i[parent[pid]]].append(pi)

    print("=" * 80); print(f"MINI-VERSE (passage) — {name}: {len(corpus):,} docs, {len(passages):,} passages, "
                            f"avg {avg_len:.0f} words/doc"); print("=" * 80)

    def evl_doc(scorer):
        rc = nd = 0.0
        for qid in ids:
            gold = {d2i[d] for d in test_q[qid] if test_q[qid].get(d, 0) > 0 and d in d2i}
            sc = scorer(queries[qid]); top = np.argsort(sc)[::-1][:100]
            rc += len(set(top) & gold) / len(gold); nd += ndcg10([doc_ids[i] for i in top[:10]], test_q[qid])
        n = len(ids); return rc / n, nd / n

    def maxp(query):                                               # doc score = max over its passages (the meet)
        ps = eng_p.score(query)
        out = np.zeros(len(doc_ids))
        for di, plist in doc_pass.items():
            if plist: out[di] = ps[plist].max()
        return out

    r_d, n_d = evl_doc(lambda q: eng_d.score(q))
    r_p, n_p = evl_doc(maxp)
    print(f"  {'retrieval':<22}{'recall@100':>12}{'nDCG@10':>10}")
    print(f"  {'flat whole-doc':<22}{r_d:>12.4f}{n_d:>10.4f}")
    print(f"  {'mini-verse (max-passage)':<22}{r_p:>12.4f}{n_p:>10.4f}")
    print(f"  delta: recall {r_p-r_d:+.4f}, nDCG {n_p-n_d:+.4f} "
          f"({'mini-verse helps' if n_p > n_d + 0.002 else 'neutral/whole-doc better'})")
    print(f"  (structure: each doc node seeds a passage sub-lattice; doc score = meet/max over passages.)")


if __name__ == "__main__":
    main()
