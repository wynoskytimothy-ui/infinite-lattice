#!/usr/bin/env python3
"""Does INGEST PROPAGATION add to the stack? Test if the 2-hop transitive source reaches gold the other 4
sources (lexical, bridges, drift, distilled-SPLADE) miss -> does it RAISE THE UNION CEILING (the campaign's
proven test of whether a new source is genuinely additive, not just redundant)."""
import os, sys
os.environ.setdefault("PYTHONUTF8", "1"); os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from aethos_edge_rag import EdgeRAG
from aethos_splade_lattice import DistilledSpladeIndex
from scripts.bench_supervised_bridges import load, words
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
    spl = None
    vocab = sorted(seeds); df = CACHE / f"splade_{name}_doc128.npz"; vf = CACHE / f"distill_{name}_v{len(vocab)}.npz"
    if df.exists() and vf.exists():
        dz = np.load(df); vz = np.load(vf, allow_pickle=True)
        spl = DistilledSpladeIndex(); spl.doc_ids = doc_ids; spl.N = len(doc_ids)
        spl._set_docs(dz["t"], dz["w"], dz["o"]); spl._set_table(vz["words"].tolist(), vz["t"], vz["w"], vz["o"])
    print("=" * 86); print(f"PROP IN THE STACK — {name}: {len(corpus):,} docs, {len(ids)} q, distilled={'y' if spl else 'n'}"); print("=" * 86)

    def src(qid):
        q = queries[qid]
        s = {"lexical": list(np.argsort(eng.score(q))[::-1][:DEPTH]),
             "bridged": [d2i[d] for d in eng.search_bridged(q, DEPTH)],
             "drift": list(np.argsort(score_bag(eng, drift_bag(q, drift2, alpha=0.4)))[::-1][:DEPTH]),
             "prop": list(np.argsort(score_bag(eng, exp_bag(q, drift2, prop, a1=0.4, a2=0.4)))[::-1][:DEPTH])}
        if spl: s["distilled"] = [d2i[d] for d in spl.search(q, DEPTH)]
        return s

    base4 = ["lexical", "bridged", "drift"] + (["distilled"] if spl else [])
    c4 = c5 = uniq = solo_prop = nq = 0
    for qid in ids:
        gold = {d2i[d] for d in test_q[qid] if test_q[qid].get(d, 0) > 0 and d in d2i}
        if not gold: continue
        nq += 1
        s = src(qid)
        u4 = set().union(*[set(s[k][:DEPTH]) for k in base4])
        u5 = u4 | set(s["prop"][:DEPTH])
        c4 += len(u4 & gold) / len(gold); c5 += len(u5 & gold) / len(gold)
        uniq += len((set(s["prop"][:DEPTH]) - u4) & gold) / len(gold)   # gold ONLY prop reaches
        solo_prop += len(set(s["prop"][:100]) & gold) / len(gold)
    print(f"  union ceiling@{DEPTH}  (4 sources, no prop): {c4/nq:.4f}")
    print(f"  union ceiling@{DEPTH}  (5 sources, + prop) : {c5/nq:.4f}   ({(c5-c4)/nq:+.4f})")
    print(f"  gold ONLY propagation reaches (not in the other 4): {uniq/nq:.4f}")
    print(f"  -> propagation is {'COMPLEMENTARY (raises the ceiling)' if (c5-c4)/nq > 0.002 else 'mostly redundant'} "
          f"with the existing sources.")


if __name__ == "__main__":
    main()
