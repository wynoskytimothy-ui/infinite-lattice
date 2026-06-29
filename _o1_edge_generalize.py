#!/usr/bin/env python3
"""Generalize the edge champion across corpora. For one corpus (argv[1]):
build full-multiview + word-only, learn bridges from TRAIN qrels, eval all 4 configs on TEST.
Self-certifies no train/test query leakage. Emits one JSON line (last line) for clean capture.

Configs:  A full multiview | B full+bridges | C word-only | D word-only+bridges (the edge champion)
"""
import os, sys, time, json, tempfile
os.environ.setdefault("PYTHONUTF8", "1")
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from aethos_append_index import AppendOnlyLatticeIndex
from aethos_bridges import RelevanceBridges
from scripts.bench_supervised_bridges import load, ndcg10, recall10, words


def bridge_score(br, idx, query, cand):
    cand_set = set(cand); bs = {d: 0.0 for d in cand_set}
    for qt in set(words(query)):
        for dt, w in br.bridge.get(qt, ()):
            p = idx.token_prime.get(("w", dt))
            if p is None: continue
            pl = idx.postings.get(p)
            if not pl: continue
            for d in cand_set:
                if d in pl: bs[d] += w
    return bs


def build(corpus, **cfg):
    idx = AppendOnlyLatticeIndex(**cfg)
    for d, t in corpus.items(): idx.add(d, t)
    return idx


def disk_bdoc(idx, n):
    fd, path = tempfile.mkstemp(suffix=".npz"); os.close(fd)
    idx.save(path[:-4]); nb = os.path.getsize(path); os.remove(path)
    return nb / n


def eval_idx(idx, queries, test_q, test_ids, br=None, lam=0.15):
    nd = rc = 0.0; lat = []
    for qid in test_ids:
        q = queries[qid]; t0 = time.perf_counter()
        if br is None:
            ranked = idx.search(q, 10)
        else:
            lex = idx._score(q)
            cand = sorted(lex, key=lambda d: lex[d], reverse=True)[:100]
            if cand:
                lmax = max(lex[d] for d in cand) or 1.0
                bs = bridge_score(br, idx, q, cand); bmax = (max(bs.values()) if bs else 0.0) or 1.0
                final = {d: lex[d]/lmax + lam*bs.get(d, 0.0)/bmax for d in cand}
                ranked = sorted(final, key=lambda d: final[d], reverse=True)[:10]
            else: ranked = []
        lat.append((time.perf_counter()-t0)*1000)
        nd += ndcg10(ranked, test_q[qid]); rc += recall10(ranked, test_q[qid])
    n = len(test_ids)
    return round(nd/n, 4), round(rc/n, 4), round(float(np.median(lat)), 3)


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "scifact"
    corpus, queries, train_q, test_q = load(name)
    test_ids = [q for q in test_q if q in queries]
    # ---- leakage self-certification: train and test query IDs must be disjoint ----
    leak = set(train_q) & set(test_q)
    n = len(corpus)

    full = build(corpus); bdoc_f = disk_bdoc(full, n); full.finalize()
    A = eval_idx(full, queries, test_q, test_ids)
    br_full = RelevanceBridges(full, len(full.alive)).learn(queries, train_q, corpus)
    B = eval_idx(full, queries, test_q, test_ids, br=br_full)

    wo = build(corpus, index_mode="kappa_primary"); bdoc_w = disk_bdoc(wo, n); wo.finalize()
    C = eval_idx(wo, queries, test_q, test_ids)
    br_wo = RelevanceBridges(wo, len(wo.alive)).learn(queries, train_q, corpus)
    D = eval_idx(wo, queries, test_q, test_ids, br=br_wo)

    res = {
        "corpus": name, "n_docs": n, "n_test_q": len(test_ids),
        "n_train_qrels": sum(len(v) for v in train_q.values()),
        "train_test_query_leak": len(leak),
        "bdoc_full": round(bdoc_f, 1), "bdoc_word_only": round(bdoc_w, 1),
        "A_full":        {"ndcg": A[0], "recall": A[1], "ms": A[2]},
        "B_full_bridge": {"ndcg": B[0], "recall": B[1], "ms": B[2]},
        "C_word_only":   {"ndcg": C[0], "recall": C[1], "ms": C[2]},
        "D_edge_champ":  {"ndcg": D[0], "recall": D[1], "ms": D[2]},
        "D_beats_A": D[0] >= A[0] - 0.002,
        "bridge_gain_on_word_only": round(D[0] - C[0], 4),
        "shrink_vs_full": round(bdoc_f / bdoc_w, 2),
    }
    print(f"\n{name}: docs={n} test_q={len(test_ids)} leak={len(leak)}")
    print(f"  A full         {bdoc_f:6.0f} B/doc  nDCG {A[0]:.4f}")
    print(f"  B full+bridge  {bdoc_f:6.0f} B/doc  nDCG {B[0]:.4f}")
    print(f"  C word-only    {bdoc_w:6.0f} B/doc  nDCG {C[0]:.4f}")
    print(f"  D edge champ   {bdoc_w:6.0f} B/doc  nDCG {D[0]:.4f}  (bridge {D[0]-C[0]:+.4f}, vs full {D[0]-A[0]:+.4f})")
    print("JSON " + json.dumps(res))


if __name__ == "__main__":
    main()
