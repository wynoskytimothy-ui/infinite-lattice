#!/usr/bin/env python3
"""
Decisive experiment: zero-shot rare-node dual-corridor INTERSECTION retrieval.

Timothy's hypothesis, made testable: each rare query term lights up its
co-occurrence corridor; the rare terms that BUNCH together across the top
pseudo-gold docs (the "company they keep") triangulate the answer doc -- with
NO qrels. We test whether that label-free triangulation moves ranking, against
the lexical floor and the SUPERVISED bridge ceiling.

Methods compared (held-out test queries, FULL corpus -- never a 500-doc subset):
  LEX        multi-view prime-mass BM25 (aethos_append_index)        [floor]
  ZS-PRF     zero-shot rare-term corridor intersection (this work)   [new]
             top-K lexical docs -> rare terms recurring across >=R of
             them = the bunch -> expand to docs sharing the bunch ->
             fuse lex + lam*bunch  (pure pseudo-relevance, no labels)
  ZS-GOLD    zero-shot pseudo-gold GoldDocBridges (this work)         [new]
             lexical top-K as pseudo-gold -> GoldDocBridges.learn ->
             golddoc_search  (cross-query rare-word reuse, no labels)
  SUP        supervised bridges from TRAIN qrels (choose_bridge)      [ceiling]

Metrics: nDCG@10, Recall@10, Recall@100.
Bars (from the forensic plan):
  nfcorpus (primary, high vocab-mismatch): ZS Recall@100 >= +2pp over LEX
    AND nDCG@10 >= BM25 ref 0.325, with NO labels.
  scifact (control): ZS must be >= neutral on nDCG (no drift); stretch =
    close the gap toward SUP.
Kill criterion: if ZS is neutral-to-negative on nfcorpus nDCG (replicating
first-order-PPMI drift), the bunch needs supervision -> productionize SUP.
"""

from __future__ import annotations

import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aethos_append_index import AppendOnlyLatticeIndex, words
from aethos_bridges import choose_bridge, golddoc_search, GoldDocBridges
from scripts.bench_supervised_bridges import load, ndcg10, recall10


def recall_at(ranked, rels, k):
    rel = {d for d, s in rels.items() if s > 0}
    return len(set(ranked[:k]) & rel) / len(rel) if rel else 0.0


def build_index(corpus):
    idx = AppendOnlyLatticeIndex()
    for d, txt in corpus.items():
        idx.add(d, txt)
    return idx, len(idx.alive)


class IdfCache:
    def __init__(self, idx, N):
        self.idx, self.N, self._c = idx, N, {}

    def __call__(self, w):
        v = self._c.get(w)
        if v is None:
            p = self.idx.token_prime.get(("w", w))
            v = self.idx._idf(p, self.N) if p else 0.0
            self._c[w] = v
        return v


def rare_terms(text, idf, gate):
    return {w for w in set(words(text)) if idf(w) >= gate}


def zs_prf_search(idx, corpus, idf, query, lex,
                  k_prf=10, min_docs=2, rare_gate=3.0,
                  lam=0.3, n_expand=30, top_terms=40, k=100,
                  rare_doc_cache=None, return_bunch=False):
    """Zero-shot rare-term corridor intersection (pseudo-relevance).

    lex: precomputed idx._score(query) dict. No qrels used.
    """
    cand = sorted(lex, key=lex.get, reverse=True)[:100]
    if not cand:
        return ([], []) if return_bunch else []
    pseudo = cand[:k_prf]
    # the BUNCH: rare terms recurring across >= min_docs of the top pseudo docs
    bunch_docs = Counter()
    bunch_idf = {}
    for d in pseudo:
        if rare_doc_cache is not None:
            rt = rare_doc_cache.get(d)
            if rt is None:
                rt = rare_terms(corpus[d], idf, rare_gate)
                rare_doc_cache[d] = rt
        else:
            rt = rare_terms(corpus[d], idf, rare_gate)
        for w in rt:
            bunch_docs[w] += 1
            bunch_idf[w] = idf(w)
    qwords = set(words(query))
    bunch = [(w, c) for w, c in bunch_docs.items()
             if c >= min_docs and w not in qwords]
    # rank bunch terms by (recurrence across pseudo docs) x idf  -> rarest, most shared first
    bunch.sort(key=lambda x: (-(x[1] * bunch_idf[x[0]]), x[0]))
    bunch = bunch[:top_terms]
    exp = defaultdict(float)
    for w, c in bunch:
        p = idx.token_prime.get(("w", w))
        if p is None:
            continue
        wt = bunch_idf[w] * c
        for d, tf in idx.postings.get(p, {}).items():
            if d in idx.alive:
                exp[d] += wt * tf / (tf + 1.0)
    cset = set(cand)
    extra = [d for d in sorted(exp, key=exp.get, reverse=True)
             if d not in cset][:n_expand]
    pool = cand + extra
    lmax = max((lex.get(d, 0.0) for d in pool), default=1.0) or 1.0
    emax = max(exp.values()) if exp else 1.0
    final = {d: lex.get(d, 0.0) / lmax + lam * exp.get(d, 0.0) / emax for d in pool}
    ranked = sorted(final, key=final.get, reverse=True)[:k]
    if return_bunch:
        return ranked, bunch
    return ranked


def evaluate(name, fn, queries, test_q, test_ids):
    nd = r10 = r100 = 0.0
    t0 = time.time()
    for qid in test_ids:
        ranked = fn(queries[qid])
        rels = test_q[qid]
        nd += ndcg10(ranked, rels)
        r10 += recall10(ranked, rels)
        r100 += recall_at(ranked, rels, 100)
    n = len(test_ids)
    dt = (time.time() - t0) / n * 1000
    return {"name": name, "ndcg10": nd / n, "recall10": r10 / n,
            "recall100": r100 / n, "ms": dt}


def run_dataset(name):
    corpus, queries, train_q, test_q = load(name)
    test_ids = [q for q in test_q if q in queries]
    print(f"\n{'='*72}\n{name}: {len(corpus)} docs | train {len(train_q)} q | "
          f"test {len(test_ids)} q", flush=True)

    idx, N = build_index(corpus)
    idf = IdfCache(idx, N)

    # precompute lexical scores once (reused by LEX + ZS-PRF)
    lex_cache = {qid: idx._score(queries[qid]) for qid in test_ids}
    rare_doc_cache = {}

    def lex_fn(q, qid=None):
        return None  # placeholder; we use cached scores below

    # LEX
    def f_lex(qid):
        lex = lex_cache[qid]
        return sorted(lex, key=lex.get, reverse=True)[:100]

    # ZS-PRF
    def f_prf(qid):
        return zs_prf_search(idx, corpus, idf, queries[qid], lex_cache[qid],
                             rare_doc_cache=rare_doc_cache)

    # evaluate by qid (so we reuse cached lexical scores)
    def eval_by_qid(label, fn):
        nd = r10 = r100 = 0.0
        t0 = time.time()
        for qid in test_ids:
            ranked = fn(qid)
            rels = test_q[qid]
            nd += ndcg10(ranked, rels)
            r10 += recall10(ranked, rels)
            r100 += recall_at(ranked, rels, 100)
        n = len(test_ids)
        return {"name": label, "ndcg10": nd / n, "recall10": r10 / n,
                "recall100": r100 / n, "ms": (time.time() - t0) / n * 1000}

    res = []
    res.append(eval_by_qid("LEX (floor)", f_lex))
    res.append(eval_by_qid("ZS-PRF (new)", f_prf))

    # ZS-GOLD: zero-shot pseudo-gold GoldDocBridges (label-free)
    pseudo_qrels = {}
    for qid in test_ids:
        lex = lex_cache[qid]
        top = sorted(lex, key=lex.get, reverse=True)[:10]
        pseudo_qrels[qid] = {d: 1 for d in top}
    t0 = time.time()
    gb_zs = GoldDocBridges(idx, N).learn(queries, pseudo_qrels, corpus)
    zs_learn_s = time.time() - t0

    def f_zsgold(qid):
        return golddoc_search(idx, gb_zs, queries[qid], lam=0.25, n_expand=30, k=100)

    res.append(eval_by_qid("ZS-GOLD (new)", f_zsgold))

    # SUP: supervised ceiling (choose_bridge from TRAIN qrels)
    t0 = time.time()
    bridge_obj, search_fn, bname, binfo = choose_bridge(idx, N, queries, train_q, corpus)
    sup_learn_s = time.time() - t0

    def f_sup(qid):
        return search_fn(idx, bridge_obj, queries[qid], lam=0.25, n_expand=30, k=100)

    res.append(eval_by_qid(f"SUP ceiling [{bname}]", f_sup))

    # report
    print(f"  (ZS pseudo-gold learn {zs_learn_s:.1f}s | SUP learn {sup_learn_s:.1f}s | "
          f"bridge route={bname} {binfo})", flush=True)
    print(f"  {'method':<22}{'nDCG@10':>9}{'R@10':>8}{'R@100':>8}{'ms/q':>8}", flush=True)
    base = res[0]
    for r in res:
        dnd = r["ndcg10"] - base["ndcg10"]
        dr100 = r["recall100"] - base["recall100"]
        tag = "" if r is base else f"  (nDCG {dnd:+.4f}, R@100 {dr100:+.4f})"
        print(f"  {r['name']:<22}{r['ndcg10']:>9.4f}{r['recall10']:>8.4f}"
              f"{r['recall100']:>8.4f}{r['ms']:>8.1f}{tag}", flush=True)

    return {"dataset": name, "N": N, "results": res,
            "bridge_route": bname, "bridge_info": binfo}


def probes(name, probe_terms):
    """Show the corridor 'bunch' a probe query lights up + gold recovery."""
    corpus, queries, train_q, test_q = load(name)
    idx, N = build_index(corpus)
    idf = IdfCache(idx, N)
    test_ids = [q for q in test_q if q in queries]
    print(f"\n--- probes on {name}: do rare-term corridors light up the bunch? ---",
          flush=True)
    rare_doc_cache = {}
    shown = 0
    for qid in test_ids:
        qtext = queries[qid].lower()
        if not any(t in qtext for t in probe_terms):
            continue
        lex = idx._score(queries[qid])
        lex_ranked = sorted(lex, key=lex.get, reverse=True)
        gold = [d for d, s in test_q[qid].items() if s > 0]
        prf_ranked, bunch = zs_prf_search(
            idx, corpus, idf, queries[qid], lex, rare_doc_cache=rare_doc_cache,
            return_bunch=True)
        g_lex = next((i + 1 for i, d in enumerate(lex_ranked) if d in set(gold)), None)
        g_prf = next((i + 1 for i, d in enumerate(prf_ranked) if d in set(gold)), None)
        bunch_str = ", ".join(f"{w}x{c}" for w, c in bunch[:8])
        print(f"  Q{qid}: {queries[qid][:60]!r}", flush=True)
        print(f"     gold rank  LEX={g_lex}  ZS-PRF={g_prf}", flush=True)
        print(f"     corridor bunch: {bunch_str}", flush=True)
        shown += 1
        if shown >= 4:
            break
    if shown == 0:
        print("  (no test query contained the probe terms)", flush=True)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    datasets = args or ["scifact", "nfcorpus"]
    print("ZERO-SHOT rare-node dual-corridor intersection vs lexical floor / "
          "supervised ceiling", flush=True)
    summary = []
    for ds in datasets:
        summary.append(run_dataset(ds))
    # probes using Timothy's own examples
    probes("scifact", ["arthritis", "biomaterial", "inductive", "p53"])
    probes("nfcorpus", ["arthritis", "vitamin", "cancer", "diabetes"])

    print(f"\n{'='*72}\nVERDICT", flush=True)
    for s in summary:
        r = {x["name"]: x for x in s["results"]}
        lex = r["LEX (floor)"]
        prf = r["ZS-PRF (new)"]
        print(f"  {s['dataset']}: ZS-PRF nDCG {prf['ndcg10']-lex['ndcg10']:+.4f}, "
              f"R@100 {prf['recall100']-lex['recall100']:+.4f}  "
              f"(route {s['bridge_route']})", flush=True)


if __name__ == "__main__":
    main()
