#!/usr/bin/env python3
"""ZERO-SHOT RAG serve optimization — faster + accuracy-held-or-better, on the full 8.8M index.
Profiles search_corr (where do the 123 ms go?) then tests:
  - corr            : current lazy composite-meet (baseline, 123 ms, MRR 0.3986)
  - corr_capped     : cap candidate set by meet-multiplicity before scoring (SPEED)
  - corr_3way       : add lazy 3-way meets to the pool (ACCURACY, zero-shot multi-hop)
  - corr_capped_3way: both
Same 250 cached queries (no GPU). Reports MRR@10 / recall@100 / median+p90 ms + a profile breakdown."""
import os, sys, time, pickle
os.environ["WORK"] = r"C:\Users\wynos\trng\marco_data\splade_native_full"
os.environ.setdefault("PYTHONUTF8", "1")
import numpy as np
from pathlib import Path
from collections import defaultdict
import marco_splade_native as m

NQ = int(sys.argv[1]) if len(sys.argv) > 1 else 250


def main():
    si = m.ServedIndex()
    col, tloc, present, acc = si.col, si.tloc, si.present, si.acc
    print(f"  loaded {si.n_docs:,} docs\n", flush=True)
    MARCO = m.MARCO
    qrels = defaultdict(set)
    with open(MARCO / "qrels.dev.small.tsv", encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) >= 4 and int(p[3]) > 0: qrels[p[0]].add(int(p[2]))
    raw = pickle.load(open(Path(os.environ["WORK"].replace("_full", "")) / "_dd_qenc_cache.pkl", "rb"))
    queries = []
    with open(MARCO / "queries.dev.tsv", encoding="utf-8") as f:
        for line in f:
            a = line.rstrip("\n").split("\t", 1)
            if len(a) == 2 and a[0] in qrels and a[0] in raw: queries.append(a[0])
    queries = queries[:NQ]
    venc = {}
    for qid in queries:
        ids, w = raw[qid]
        venc[qid] = (np.array(ids.tolist(), np.int64), np.array([x * m.QSCALE for x in w.tolist()], np.float32))
    print(f"  {len(queries)} queries (cached)\n", flush=True)

    def anchors_terms(qids, qw, topq, n_anchor):
        qw = np.asarray(qw, np.float32); top = np.argsort(-qw)[:topq]
        terms = []
        for i in top:
            j = col.get(int(qids[i]))
            if j is None: continue
            loc, w = tloc[j]; terms.append((loc, w, float(qw[i])))
        terms.sort(key=lambda t: len(t[0]))
        return terms, terms[:n_anchor]

    def meet(x, y):
        if len(x) > len(y): x, y = y, x
        pos = np.searchsorted(y, x); pc = np.minimum(pos, len(y) - 1)
        return x[y[pc] == x]

    def score_candidates(terms, C):
        sc = np.zeros(len(C), np.float32)
        for loc, w, qweight in terms:
            pos = np.searchsorted(loc, C); pc = np.minimum(pos, len(loc) - 1)
            hit = loc[pc] == C; sc[hit] += qweight * w[pc[hit]]
        return sc

    def serve(qid, mode, k=100, topq=30, n_anchor=6, cap=4000):
        qids, qw = venc[qid]
        terms, anchors = anchors_terms(qids, qw, topq, n_anchor)
        if not terms: return np.zeros(0, np.uint32)
        parts = []
        mult = defaultdict(int)
        for a in range(len(anchors)):
            for b in range(a + 1, len(anchors)):
                ab = meet(anchors[a][0], anchors[b][0])
                if len(ab):
                    parts.append(ab)
        if "3way" in mode:
            for a in range(len(anchors)):
                for b in range(a + 1, len(anchors)):
                    for c in range(b + 1, len(anchors)):
                        abc = meet(meet(anchors[a][0], anchors[b][0]), anchors[c][0])
                        if len(abc): parts.append(abc)
        parts.append(anchors[0][0])
        C = np.unique(np.concatenate(parts))
        if "capped" in mode and len(C) > cap:
            # rank candidates by meet-multiplicity (how many composite lists they appear in = correlation)
            allcat = np.concatenate(parts); vals, counts = np.unique(allcat, return_counts=True)
            order = np.argsort(-counts)[:cap]
            C = np.sort(vals[order])
        sc = score_candidates(terms, C)
        sel = np.argpartition(-sc, k)[:k] if len(C) > k else np.arange(len(C))
        order = sel[np.argsort(-sc[sel])]
        return present[C[order]]

    # ---- profile baseline ----
    print("  PROFILE search_corr (250 q): meet-phase vs score-phase", flush=True)
    t_meet = t_score = 0.0; csizes = []
    for qid in queries:
        qids, qw = venc[qid]; terms, anchors = anchors_terms(qids, qw, 30, 6)
        if not terms: continue
        t0 = time.perf_counter(); parts = []
        for a in range(len(anchors)):
            for b in range(a + 1, len(anchors)):
                ab = meet(anchors[a][0], anchors[b][0]);  parts.append(ab) if len(ab) else None
        parts.append(anchors[0][0]); C = np.unique(np.concatenate(parts)); t_meet += time.perf_counter() - t0
        csizes.append(len(C))
        t0 = time.perf_counter(); _ = score_candidates(terms, C); t_score += time.perf_counter() - t0
    n = len(queries)
    print(f"    meet-phase {t_meet/n*1000:.1f} ms/q | score-phase {t_score/n*1000:.1f} ms/q | "
          f"median |C|={int(np.median(csizes)):,} (max {max(csizes):,})\n", flush=True)

    def bench(mode, **kw):
        for qid in queries[:5]: serve(qid, mode, **kw)
        mrr = rec = 0.0; lat = []
        for qid in queries:
            t = time.perf_counter(); ids = serve(qid, mode, **kw); lat.append((time.perf_counter()-t)*1000)
            top = [int(d) for d in ids[:100]]; gold = qrels[qid]
            if any(d in gold for d in top): rec += 1
            for r, d in enumerate(top[:10]):
                if d in gold: mrr += 1.0/(r+1); break
        lat = np.array(lat); n = len(queries)
        print(f"  {mode:<20}{mrr/n:>9.4f}{rec/n*100:>11.2f}%{np.median(lat):>9.1f}{np.percentile(lat,90):>9.1f}", flush=True)

    print(f"  {'serve':<20}{'MRR@10':>9}{'recall@100':>12}{'med ms':>9}{'p90 ms':>9}", flush=True)
    bench("corr")
    bench("corr_capped", cap=4000)
    bench("corr_capped", cap=2000)
    bench("corr_3way")
    bench("corr_capped_3way", cap=4000)
    print(f"\n  footprint orthogonal: base 168 B/doc (3-bit wt) / EF doc-ids ~165 / 2-bit ~153.", flush=True)


if __name__ == "__main__":
    main()
