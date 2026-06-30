#!/usr/bin/env python3
"""DISTILL SPLADE INTO THE LATTICE (Timothy's idea) — teach the lattice what SPLADE knows about each word,
SAVE it as a static expansion table, then serve queries ENCODER-FREE. Tests how much of SPLADE's recall/nDCG
survives WITHOUT running the query encoder at serve time (keep lattice speed, gain semantic recall).

  lexical            : raw words, no expansion (EdgeRAG)
  SPLADE full        : full-query encoder both sides (the target; needs encoder at serve)
  DISTILLED          : SPLADE docs + per-word distilled expansion table (NO query encoder at serve)
  DISTILLED + lex RRF: fuse distilled with lexical
Doc-side expansion is always baked in at ingest ('every doc is a question, the model fills its correlations')."""
import os, sys, time
os.environ.setdefault("PYTHONUTF8", "1"); os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scripts.bench_supervised_bridges import load, ndcg10, words
from aethos_edge_rag import EdgeRAG
from _o1_splade_roles import Splade, cached_encode, build_csr, CACHE


def distill_table(sp, vocab_words, key, topk=64):
    """For each word, SAVE what SPLADE expands it to (single-word encode) -> static term->expansion table."""
    f = CACHE / f"distill_{key}.npz"
    if f.exists():
        z = np.load(f, allow_pickle=True); return z["words"].tolist(), z["t"], z["w"], z["o"]
    t, w, o = sp.encode(vocab_words, bs=64, topk=topk)                  # one forward per word, ONCE (offline)
    np.savez(f, words=np.array(vocab_words, object), t=t, w=w, o=o)
    return vocab_words, t, w, o


def scores_from_sparse(indptr, seg_doc, seg_w, terms, wts, N):
    s = np.zeros(N)
    for tid, qw in zip(terms, wts):
        a, e = int(indptr[tid]), int(indptr[tid + 1])
        if e > a: s[seg_doc[a:e]] += float(qw) * seg_w[a:e].astype(np.float32)
    return s


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "nfcorpus"
    topk = 128
    corpus, queries, train_q, test_q = load(name)
    doc_ids = list(corpus.keys()); d2i = {d: i for i, d in enumerate(doc_ids)}
    test_ids = [q for q in test_q if q in queries]; N = len(corpus)
    print("=" * 92); print(f"DISTILL SPLADE -> LATTICE — {name}: {N:,} docs (encoder-free serve test)"); print("=" * 92)

    sp = Splade()
    dt, dw, do = cached_encode(sp, f"{name}_doc{topk}", [corpus[d] for d in doc_ids], topk)
    qt, qw, qo = cached_encode(sp, f"{name}_qry", [queries[q] for q in test_ids], None)   # full-query (target)
    indptr, seg_doc, order = build_csr(dt, do); seg_w = dw[order]

    # distilled per-word table over the query vocabulary (one-time offline; production distills full vocab)
    vocab = sorted({w for q in test_ids for w in words(queries[q])})
    vw, tt, tw, to = distill_table(sp, vocab, f"{name}_v{len(vocab)}")
    w2i = {w: i for i, w in enumerate(vw)}

    eng = EdgeRAG().build(corpus)

    def metrics(top, gold): return ndcg10([doc_ids[i] for i in top[:10]], test_q[qid]), \
        (len(set(top[:100]) & gold) / len(gold) if gold else 0.0)

    agg = {m: [0.0, 0.0] for m in ["lexical", "splade_full", "distilled", "distilled_rrf"]}; nq = 0
    for qi, qid in enumerate(test_ids):
        gold = {d2i[d] for d in test_q[qid] if test_q[qid].get(d, 0) > 0 and d in d2i}
        if not gold: continue
        nq += 1
        lex = eng.score(queries[qid]); lex100 = np.argsort(lex)[::-1][:100]
        # splade full (uses the query encoder at serve)
        sf = scores_from_sparse(indptr, seg_doc, seg_w, qt[qo[qi]:qo[qi+1]], qw[qo[qi]:qo[qi+1]], N)
        # DISTILLED: expand raw query words via the saved table -- NO encoder at serve
        dterms, dwts = [], []
        for word in words(queries[qid]):
            j = w2i.get(word)
            if j is None: continue
            a, e = int(to[j]), int(to[j + 1]); dterms.append(tt[a:e]); dwts.append(tw[a:e])
        if dterms:
            dd = scores_from_sparse(indptr, seg_doc, seg_w, np.concatenate(dterms), np.concatenate(dwts), N)
        else:
            dd = np.zeros(N)
        d100 = np.argsort(dd)[::-1][:100]
        # distilled RRF with lexical
        lr = {int(d): r for r, d in enumerate(lex100)}; dr = {int(d): r for r, d in enumerate(d100)}
        cand = set(lr) | set(dr)
        rrf = np.array(sorted(cand, key=lambda d: -(1/(60+lr.get(d, 1e6)) + 1/(60+dr.get(d, 1e6)))))
        for m, top in [("lexical", lex100), ("splade_full", np.argsort(sf)[::-1][:100]),
                       ("distilled", d100), ("distilled_rrf", rrf)]:
            nd, rc = metrics(top, gold); agg[m][0] += nd; agg[m][1] += rc

    print(f"\n  {'mode':<22}{'nDCG@10':>10}{'Recall@100':>12}{'encoder@serve?':>16}")
    enc = {"lexical": "no", "splade_full": "YES", "distilled": "NO", "distilled_rrf": "NO"}
    for m in ["lexical", "splade_full", "distilled", "distilled_rrf"]:
        print(f"  {m:<22}{agg[m][0]/nq:>10.4f}{agg[m][1]/nq:>12.4f}{enc[m]:>16}")
    L, F, D, R = (agg[m][0]/nq for m in ["lexical", "splade_full", "distilled", "distilled_rrf"])
    LR, FR, DR, RR = (agg[m][1]/nq for m in ["lexical", "splade_full", "distilled", "distilled_rrf"])
    rec = (DR - LR) / (FR - LR) * 100 if FR > LR else 0
    print(f"\n  RECALL@100: lexical {LR:.3f} -> distilled(no encoder) {DR:.3f} -> SPLADE-full {FR:.3f} "
          f"(distilled recovers {rec:.0f}% of SPLADE's recall gain, NO query encoder)")
    print(f"  nDCG@10:    lexical {L:.3f} -> distilled {D:.3f} -> distilled+lex {R:.3f} -> SPLADE-full {F:.3f}")


if __name__ == "__main__":
    main()
