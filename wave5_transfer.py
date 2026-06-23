#!/usr/bin/env python3
"""Wave 5: CROSS-CORPUS SEMANTIC TRANSFER. Corridors (qt->dt links learned from qrels) are general facts.
Can corridors from a SOURCE corpus build semantics in ANOTHER -- esp. the test-only BEIR corpora that have
NO train qrels (currently lexical-only)? Transfer = source LINK P(dt|qt) x TARGET idf(dt) (specificity from
the target), applied as pool-expansion. Sources: MARCO (500k qrels) + BEIR-union (scifact+nfcorpus+fiqa).
Targets: scidocs/trec-covid/arguana (no qrels) + scifact (transfer vs own).
"""
import sys, pickle
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scripts.bench_supervised_bridges import load, ndcg10
from tune import build, search, toks, corridors as own_corridors, IDF_Q, IDF_D, MINP, TOP

MARCO_CORR = Path(r"C:\Users\wynos\trng\marco_data\full_idx_corridors.pkl")
SOURCES_TRAIN = ["scifact", "nfcorpus", "fiqa"]
TARGETS = ["scifact", "scidocs", "trec-covid", "arguana"]


def learn_links(corpus, queries, train_q, widf):
    cooc = defaultdict(Counter); npairs = Counter()
    for qid, rels in train_q.items():
        qts = [w for w in set(toks(queries.get(qid, ""))) if widf(w) >= IDF_Q]
        if not qts:
            continue
        for cid, rel in rels.items():
            if rel <= 0 or cid not in corpus:
                continue
            dts = set(w for w in toks(corpus[cid]) if widf(w) >= IDF_D)
            for qt in qts:
                npairs[qt] += 1; cooc[qt].update(dts)
    links = defaultdict(dict)
    for qt, cnt in cooc.items():
        n = npairs[qt]
        for dt, c in cnt.items():
            if c >= MINP and dt != qt:
                links[qt][dt] = max(links[qt].get(dt, 0.0), c / n)
    return links


def target_corr(src, eng, is_marco):
    wtid, idf = eng["wtid"], eng["idf"]
    def tidf(w):
        i = wtid.get(w); return float(idf[i]) if i is not None else 0.0
    corr = {}
    for qt, parts in src.items():
        pairs = parts if is_marco else parts.items()                  # marco: [(dt,w)]; union: {dt:P}
        sc = ([(dt, w) for dt, w in pairs if tidf(dt) > 0] if is_marco
              else [(dt, p * tidf(dt)) for dt, p in pairs if tidf(dt) > 0])
        if sc:
            sc.sort(key=lambda x: -x[1]); corr[qt] = sc[:TOP]
    return corr


def ev(eng, corr, queries, test_q, ids):
    return sum(ndcg10(search(eng, corr, queries[q], {}, 10), test_q[q]) for q in ids) / len(ids)


def main():
    marco_corr = pickle.load(open(MARCO_CORR, "rb"))
    print(f"  loaded MARCO corridors: {len(marco_corr):,} query-terms (500k-qrels source)", flush=True)

    union = defaultdict(dict)
    for nm in SOURCES_TRAIN:
        corpus, queries, train_q, _ = load(nm)
        eng = build(corpus)
        def widf(w, e=eng):
            i = e["wtid"].get(w); return float(e["idf"][i]) if i is not None else 0.0
        for qt, d in learn_links(corpus, queries, train_q, widf).items():
            for dt, p in d.items():
                union[qt][dt] = max(union[qt].get(dt, 0.0), p)
        del eng, corpus
    print(f"  BEIR-union links: {len(union):,} query-terms (scifact+nfcorpus+fiqa)\n", flush=True)

    print(f"  {'target':<12}{'lexical':>9}{'own':>9}{'MARCO-xfer':>14}{'BEIR-xfer':>14}")
    for nm in TARGETS:
        corpus, queries, train_q, test_q = load(nm)
        eng = build(corpus); ids = [q for q in test_q if q in queries]
        lex = ev(eng, {}, queries, test_q, ids)
        own = ev(eng, own_corridors(eng, corpus, queries, train_q), queries, test_q, ids) if train_q else None
        mx = ev(eng, target_corr(marco_corr, eng, True), queries, test_q, ids)
        bx = ev(eng, target_corr(union, eng, False), queries, test_q, ids)
        owns = f"{own:.4f}" if own is not None else "   -  "
        print(f"  {nm:<12}{lex:>9.4f}{owns:>9}{mx:>10.4f} {mx-lex:>+.4f}{bx:>10.4f} {bx-lex:>+.4f}", flush=True)
        del eng, corpus
    print(f"\n  +delta vs lexical = the semantic reach TRANSFERRED from a labeled source to an unlabeled target.")


if __name__ == "__main__":
    main()
