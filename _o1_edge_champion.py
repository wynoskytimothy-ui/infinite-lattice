#!/usr/bin/env python3
"""THE EDGE CHAMPION — does the neural-free bridge lever recover word-only's accuracy at 4x smaller size?
Head-to-head, REAL save() bytes + held-out test nDCG:
  A. full multiview            (fat, accuracy ref)
  B. full multiview + bridges  (the server champion)
  C. word-only                 (small)
  D. word-only + bridges       <-- the edge candidate: small AND accurate, neural-free?
If D >= A, the edge index gets full-fat accuracy at word-only footprint + 7x speed."""
import os, sys, time, gc, tempfile
os.environ.setdefault("PYTHONUTF8", "1")
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from aethos_append_index import AppendOnlyLatticeIndex
from aethos_bridges import RelevanceBridges
from scripts.bench_supervised_bridges import load, ndcg10, recall10, words


def bridge_score(br, idx, query, cand):
    """Reconstruct the bridge score from the learned `bridge` dict (qt -> [(doc_term, weight)]):
    each candidate gets sum of bridge weights for the learned doc-terms it actually contains."""
    cand_set = set(cand)
    bs = {d: 0.0 for d in cand_set}
    for qt in set(words(query)):
        for dt, w in br.bridge.get(qt, ()):  # learned partners for this query term
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


def disk_bytes(idx, corpus):
    fd, path = tempfile.mkstemp(suffix=".npz"); os.close(fd)
    idx.save(path[:-4]); nb = os.path.getsize(path); os.remove(path)
    return nb, nb / len(corpus)


def eval_idx(idx, queries, test_q, test_ids, br=None, lam=0.15):
    nd = rc = 0.0; lat = []
    for qid in test_ids:
        q = queries[qid]
        t0 = time.perf_counter()
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
    return nd/n, rc/n, float(np.median(lat))


def main():
    corpus, queries, train_q, test_q = load("scifact")
    test_ids = [q for q in test_q if q in queries]
    print("=" * 96)
    print(f"EDGE CHAMPION — scifact {len(corpus):,} docs | BM25 ref ~0.665 | held-out test nDCG@10")
    print("=" * 96)
    print(f"  {'config':<30}{'B/doc':>7}{'total':>8}{'nDCG':>9}{'Recall':>8}{'speed':>8}")

    rows = []
    # A/B full multiview
    full = build(corpus); _, bdoc_f = disk_bytes(full, corpus)
    full.finalize()
    nd, rc, ms = eval_idx(full, queries, test_q, test_ids)
    rows.append(("A. full multiview", bdoc_f, nd, rc, ms))
    br_full = RelevanceBridges(full, len(full.alive)).learn(queries, train_q, corpus)
    nd, rc, ms = eval_idx(full, queries, test_q, test_ids, br=br_full)
    rows.append(("B. full + bridges", bdoc_f, nd, rc, ms))

    # C/D word-only
    wo = build(corpus, index_mode="kappa_primary"); _, bdoc_w = disk_bytes(wo, corpus)
    wo.finalize()
    nd, rc, ms = eval_idx(wo, queries, test_q, test_ids)
    rows.append(("C. word-only", bdoc_w, nd, rc, ms))
    br_wo = RelevanceBridges(wo, len(wo.alive)).learn(queries, train_q, corpus)
    nd, rc, ms = eval_idx(wo, queries, test_q, test_ids, br=br_wo)
    rows.append(("D. word-only + bridges", bdoc_w, nd, rc, ms))

    for name, bdoc, nd, rc, ms in rows:
        print(f"  {name:<30}{bdoc:>7.0f}{bdoc*len(corpus)/1e6:>7.1f}M{nd:>9.4f}{rc:>8.4f}{ms:>6.2f}ms")

    A, B_, C, D = rows
    print(f"\n  VERDICT:")
    print(f"   * dropping trigrams: {A[1]:.0f}->{C[1]:.0f} B/doc ({A[1]/C[1]:.1f}x smaller), nDCG {A[2]:.4f}->{C[2]:.4f}")
    print(f"   * bridges on word-only: {C[2]:.4f}->{D[2]:.4f} ({D[2]-C[2]:+.4f}), neural-free, learned from train qrels")
    print(f"   * EDGE CHAMPION (D) vs fat full (A): {D[2]:.4f} vs {A[2]:.4f} at {A[1]/D[1]:.1f}x smaller "
          f"-> {'WINS: full accuracy at edge size' if D[2] >= A[2]-0.002 else 'recovers most of the gap'}")


if __name__ == "__main__":
    main()
