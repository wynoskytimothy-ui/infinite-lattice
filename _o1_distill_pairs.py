#!/usr/bin/env python3
"""WORD-PAIR DISTILLATION — close the cross-term-context gap. Per-word distillation (84% recall recovery)
loses the context SPLADE sees when words are TOGETHER. So distill co-occurring word-PAIRS: encode "w1 w2"
jointly, SAVE its expansion, compose query vectors from pair expansions (max-pool). Encoder-free at serve.
Tests how much of the remaining 16% gap pair-context recovers vs single-word.

Reuses the cached SPLADE doc index; only the small pair table needs encoding."""
import os, sys, time, itertools
os.environ.setdefault("PYTHONUTF8", "1"); os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scripts.bench_supervised_bridges import load, ndcg10, words
from aethos_edge_rag import EdgeRAG
from _o1_splade_roles import Splade, cached_encode, build_csr, CACHE
from _o1_distill import distill_table, scores_from_sparse


def pairs_of(ws, window=4):
    """unordered co-occurring pairs within a window (caps long queries)."""
    u = list(dict.fromkeys(ws))
    out = []
    for i in range(len(u)):
        for j in range(i + 1, min(i + window + 1, len(u))):
            out.append((u[i], u[j]))
    return out


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "nfcorpus"
    topk = 128
    corpus, queries, train_q, test_q = load(name)
    doc_ids = list(corpus.keys()); d2i = {d: i for i, d in enumerate(doc_ids)}
    test_ids = [q for q in test_q if q in queries]; N = len(corpus)
    print("=" * 92); print(f"WORD-PAIR DISTILLATION — {name}: {N:,} docs (encoder-free; closing the context gap)"); print("=" * 92)

    sp = Splade()
    dt, dw, do = cached_encode(sp, f"{name}_doc{topk}", [corpus[d] for d in doc_ids], topk)
    qt, qw, qo = cached_encode(sp, f"{name}_qry", [queries[q] for q in test_ids], None)
    indptr, seg_doc, order = build_csr(dt, do); seg_w = dw[order]

    # single-word table (the 84% baseline)
    vocab = sorted({w for q in test_ids for w in words(queries[q])})
    vw, wt, ww, wo = distill_table(sp, vocab, f"{name}_v{len(vocab)}")
    w2i = {w: i for i, w in enumerate(vw)}

    # PAIR table: encode "w1 w2" jointly for every co-occurring pair, ONCE (offline distillation)
    allpairs = sorted({p for q in test_ids for p in pairs_of(words(queries[q]))})
    pstr = [f"{a} {b}" for a, b in allpairs]
    pkey = f"{name}_pairs{len(allpairs)}"
    t0 = time.perf_counter(); _, pt, pw, po = distill_table(sp, pstr, pkey, topk=64); enc = time.perf_counter() - t0
    p2i = {p: i for i, p in enumerate(allpairs)}
    print(f"  distilled {len(vocab)} words + {len(allpairs)} pairs (pairs encoded/cached in {enc:.0f}s)")

    eng = EdgeRAG().build(corpus)

    def maxpool(keys, table, idx):
        acc = {}
        for k in keys:
            j = idx.get(k)
            if j is None: continue
            a, e = int(table[2][j]), int(table[2][j + 1])
            for t, w in zip(table[0][a:e], table[1][a:e]):
                t = int(t)
                if w > acc.get(t, 0.0): acc[t] = float(w)
        if not acc: return np.zeros(0, np.int32), np.zeros(0, np.float32)
        return np.fromiter(acc.keys(), np.int32), np.fromiter(acc.values(), np.float32)

    WT = (wt, ww, wo); PT = (pt, pw, po)
    agg = {m: [0.0, 0.0] for m in ["lexical", "splade_full", "distilled_word", "distilled_pair", "distilled_wp"]}
    nq = 0
    for qi, qid in enumerate(test_ids):
        gold = {d2i[d] for d in test_q[qid] if test_q[qid].get(d, 0) > 0 and d in d2i}
        if not gold: continue
        nq += 1
        ws = words(queries[qid]); prs = pairs_of(ws)
        lex = eng.score(queries[qid]); l100 = np.argsort(lex)[::-1][:100]
        sf = scores_from_sparse(indptr, seg_doc, seg_w, qt[qo[qi]:qo[qi+1]], qw[qo[qi]:qo[qi+1]], N)
        wterms, wwts = maxpool(ws, WT, w2i)
        pterms, pwts = maxpool(prs, PT, p2i)
        # word+pair: max-pool the union
        comb = {}
        for t, w in zip(wterms, wwts): comb[int(t)] = max(comb.get(int(t), 0.0), float(w))
        for t, w in zip(pterms, pwts): comb[int(t)] = max(comb.get(int(t), 0.0), float(w))
        ct = np.fromiter(comb.keys(), np.int32); cw = np.fromiter(comb.values(), np.float32)
        sd_word = scores_from_sparse(indptr, seg_doc, seg_w, wterms, wwts, N)
        sd_pair = scores_from_sparse(indptr, seg_doc, seg_w, pterms, pwts, N) if pterms.size else np.zeros(N)
        sd_wp = scores_from_sparse(indptr, seg_doc, seg_w, ct, cw, N)
        for m, sc in [("lexical", lex), ("splade_full", sf), ("distilled_word", sd_word),
                      ("distilled_pair", sd_pair), ("distilled_wp", sd_wp)]:
            top = np.argsort(sc)[::-1][:100]
            agg[m][0] += ndcg10([doc_ids[i] for i in top[:10]], test_q[qid])
            agg[m][1] += len(set(top[:100]) & gold) / len(gold)

    print(f"\n  {'mode':<22}{'nDCG@10':>10}{'Recall@100':>12}{'encoder@serve?':>16}")
    enc_m = {"lexical": "no", "splade_full": "YES", "distilled_word": "NO", "distilled_pair": "NO", "distilled_wp": "NO"}
    for m in ["lexical", "splade_full", "distilled_word", "distilled_pair", "distilled_wp"]:
        print(f"  {m:<22}{agg[m][0]/nq:>10.4f}{agg[m][1]/nq:>12.4f}{enc_m[m]:>16}")
    L = agg["lexical"][1]/nq; F = agg["splade_full"][1]/nq
    for m in ["distilled_word", "distilled_pair", "distilled_wp"]:
        rec = (agg[m][1]/nq - L) / (F - L) * 100 if F > L else 0
        print(f"  recall recovery {m:<16}: {rec:>5.0f}% of SPLADE's gain (encoder-free)")


if __name__ == "__main__":
    main()
