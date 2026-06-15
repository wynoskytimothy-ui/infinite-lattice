#!/usr/bin/env python3
"""Run the diagnostic-tuned ladder on the FULL 8.8M MS MARCO collection.

Loads the cached full_idx_* (marco_full_build.py), trains the gold-doc corridors on
qrels.train (cached), then for a sample of dev queries: BM25 top-100 over the full
collection (numpy, dense accumulator) -> rerank with the ladder (entity-gate + rare
crossover + gold company + diversity), fetching the top-100 passage texts on demand via
collection.offsets.npy. Reports BM25 vs ladder MRR@10 + R@100 -- does +11.3% hold at 30x
distractors?

  python marco_full_eval.py [n_queries]      (default 3000, matches the pool eval sample)
"""
import sys, re, time, math, pickle, random
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
from stem_safe import safe

MARCO = Path(r"C:\Users\wynos\trng\marco_data")
WORD = re.compile(r"[a-z0-9]+")
K1, B = 0.9, 0.4
RARE, ENTITY_IDF, QGATE = 4.0, 5.5, 0.3
SCORE_FLOOR, CAND_FLOOR = 1.0, 2.0   # skip stopword-ish (df>37%) from scan; nominate cands from idf>=2
A_XO, G_XO, B_COMP, P_DIV = 0.5, 1.5, 0.3, 0.25   # ladder constants (= marco_distinct.py)

_st = {}
def st(w):
    r = _st.get(w)
    if r is None:
        r = safe(w); _st[w] = r
    return r

def stoks(s):
    return [st(w) for w in WORD.findall(s.lower())]


class FullIndex:
    def __init__(self):
        t0 = time.perf_counter()
        self.di = np.load(MARCO / "full_idx_di.npy")
        self.tf = np.load(MARCO / "full_idx_tf.npy")
        self.ptr = np.load(MARCO / "full_idx_ptr.npy")
        self.doclen = np.load(MARCO / "full_idx_doclen.npy").astype(np.float32)
        self.idfa = np.load(MARCO / "full_idx_idf.npy")
        with open(MARCO / "full_idx_meta.pkl", "rb") as f:
            m = pickle.load(f)
        self.vocab = m["vocab"]; self.N = m["N"]; self.avgdl = m["avgdl"]
        self.tid = {t: i for i, t in enumerate(self.vocab)}
        self.offsets = np.load(MARCO / "collection.offsets.npy", mmap_mode="r")
        self.cf = open(MARCO / "collection.tsv", "r", encoding="utf-8", errors="replace")
        self.acc = np.zeros(self.N, dtype=np.float32)
        print(f"  loaded full index: {self.N:,} docs, {len(self.vocab):,} terms, "
              f"{len(self.di):,} postings ({time.perf_counter()-t0:.0f}s)", flush=True)

    def idf_of(self, term):
        i = self.tid.get(term)
        return float(self.idfa[i]) if i is not None else 0.0

    def text(self, pid):
        self.cf.seek(int(self.offsets[pid]))
        line = self.cf.readline()
        tab = line.find("\t")
        return line[tab + 1:] if tab >= 0 else ""

    def bm25_top(self, qterms, k=100):
        acc = self.acc
        touched = []          # every scored term's dis (for reset)
        cand_src = []         # only discriminative terms' dis (for candidate nomination)
        best = None           # (idf, dis) rarest scored term, fallback when no idf>=CAND_FLOOR
        for w in qterms:
            i = self.tid.get(w)
            if i is None:
                continue
            wi = self.idfa[i]
            if wi < SCORE_FLOOR:            # stopword-ish: huge list, ~uniform contribution -> skip
                continue
            s, e = int(self.ptr[i]), int(self.ptr[i + 1])
            dis = self.di[s:e]
            tfs = self.tf[s:e].astype(np.float32)
            dl = self.doclen[dis]
            contrib = wi * tfs * (K1 + 1.0) / (tfs + K1 * (1.0 - B + B * dl / self.avgdl))
            acc[dis] += contrib
            touched.append(dis)
            if wi >= CAND_FLOOR:
                cand_src.append(dis)
            if best is None or wi > best[0]:
                best = (wi, dis)
        if not touched:
            return np.empty(0, dtype=np.uint32), {}
        if not cand_src:                    # all terms common -> nominate from the single rarest
            cand_src = [best[1]]
        cand = np.unique(np.concatenate(cand_src))
        sc = acc[cand]
        if len(cand) > k:
            top = cand[np.argpartition(-sc, k)[:k]]
        else:
            top = cand
        order = top[np.argsort(-acc[top])]
        anchor = {int(d): float(acc[d]) for d in order}
        acc[np.concatenate(touched)] = 0.0  # reset all touched (dups idempotent; no unique needed)
        return order, anchor


def train_corridors(idx, n_train=120_000, seed=42):
    cache = MARCO / "full_idx_corridors.pkl"
    if cache.exists():
        with open(cache, "rb") as f:
            g = pickle.load(f)
        print(f"  loaded cached corridors: {len(g):,} query-term corridors", flush=True)
        return g
    t0 = time.perf_counter()
    qrels_tr = defaultdict(set)
    with open(MARCO / "qrels.train.tsv", encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) >= 4 and int(p[3]) > 0:
                qrels_tr[p[0]].add(p[2])
    sel = list(qrels_tr); random.Random(seed).shuffle(sel); sel = sel[:n_train]
    sel_set = set(sel)
    qtexts = {}
    with open(MARCO / "queries.train.tsv", encoding="utf-8") as f:
        for line in f:
            a = line.rstrip("\n").split("\t", 1)
            if len(a) == 2 and a[0] in sel_set:
                qtexts[a[0]] = a[1]
    cooc = defaultdict(Counter); npairs = Counter()
    for qi, q in enumerate(sel):
        qts = [w for w in set(stoks(qtexts.get(q, ""))) if idx.idf_of(w) >= 2.0]
        if not qts:
            continue
        for pid in qrels_tr[q]:
            dts = [w for w in set(stoks(idx.text(int(pid)))) if idx.idf_of(w) >= 3.0]
            for qt in qts:
                npairs[qt] += 1; cooc[qt].update(dts)
        if (qi + 1) % 20000 == 0:
            print(f"    corridors: {qi+1:,}/{len(sel):,} train q ({time.perf_counter()-t0:.0f}s)", flush=True)
    gold = {}
    for qt, cnt in cooc.items():
        n = npairs[qt]
        sc = sorted(((dt, (c / n) * idx.idf_of(dt)) for dt, c in cnt.items()
                     if dt != qt and idx.idf_of(dt) > 0 and c >= 2), key=lambda x: -x[1])
        if sc:
            gold[qt] = sc[:12]
    with open(cache, "wb") as f:
        pickle.dump(gold, f)
    print(f"  trained {len(gold):,} corridors from {len(sel):,} train q ({time.perf_counter()-t0:.0f}s)", flush=True)
    return gold


def main():
    nq = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    idx = FullIndex()
    gold = train_corridors(idx)

    qrels = defaultdict(set)
    with open(MARCO / "qrels.dev.tsv", encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) >= 4 and int(p[3]) > 0:
                qrels[p[0]].add(p[2])
    queries = {}
    with open(MARCO / "queries.dev.tsv", encoding="utf-8") as f:
        for line in f:
            a = line.rstrip("\n").split("\t", 1)
            if len(a) == 2:
                queries[a[0]] = a[1]
    qids = [q for q in qrels if q in queries]
    random.Random(42).shuffle(qids)
    qids = qids[:nq]

    bm_mrr = bm_r100 = la_mrr = la_r100 = 0.0
    hit_bm = hit_la = miss_bm = miss_la = 0.0
    nhit = nmiss = 0
    t0 = time.perf_counter()
    for n, q in enumerate(qids):
        rel = qrels[q]
        qs = [w for w in set(stoks(queries[q])) if idx.idf_of(w) >= QGATE]
        if not qs:
            nmiss += 1; continue
        qrare = set(w for w in qs if idx.idf_of(w) >= RARE)
        entity = max(qs, key=lambda w: idx.idf_of(w))
        if idx.idf_of(entity) < ENTITY_IDF:
            entity = None
        order, anchor = idx.bm25_top(qs, k=100)
        cands = [int(d) for d in order]
        # ladder rerank on the top-100
        final = {}
        for di in cands:
            dl = stoks(idx.text(di))
            ds = set(dl); tot = max(1, len(dl))
            k = sum(1 for w in qrare if w in ds)
            xo = sum(idx.idf_of(w) for w in qrare if w in ds) * (k ** G_XO if k else 0.0)
            comp = sum(w for qt in qs for dt, w in gold.get(qt, []) if dt in ds)
            if entity is not None and entity not in ds:
                comp = 0.0
            base = anchor[di] + A_XO * xo + B_COMP * comp
            distinct = len(ds) / tot
            final[di] = base * (distinct ** P_DIV)
        bm_order = cands
        la_order = sorted(final, key=final.get, reverse=True)
        rb = next((1.0 / i for i, d in enumerate(bm_order[:10], 1) if str(d) in rel), 0.0)
        rl = next((1.0 / i for i, d in enumerate(la_order[:10], 1) if str(d) in rel), 0.0)
        rbset = set(str(d) for d in bm_order[:100])
        rlset = set(str(d) for d in la_order[:100])
        bm_mrr += rb; la_mrr += rl
        bm_r100 += len(rel & rbset) / len(rel)
        la_r100 += len(rel & rlset) / len(rel)
        if rb > 0:
            nhit += 1; hit_bm += rb; hit_la += rl
        else:
            nmiss += 1; miss_bm += rb; miss_la += rl
        if (n + 1) % 250 == 0:
            el = time.perf_counter() - t0
            print(f"    {n+1}/{len(qids)} q | BM25 {bm_mrr/(n+1):.4f} | ladder {la_mrr/(n+1):.4f} "
                  f"| {el:.0f}s ({el/(n+1)*1000:.0f}ms/q)", flush=True)
    m = len(qids)
    print(f"\nFULL 8.8M COLLECTION -- {m} dev queries, top-100 rerank\n")
    print(f"   {'':14}{'MRR@10':>9}{'R@100':>9}")
    print(f"   {'BM25 (stem)':14}{bm_mrr/m:>9.4f}{bm_r100/m:>9.4f}")
    print(f"   {'+ ladder':14}{la_mrr/m:>9.4f}{la_r100/m:>9.4f}")
    d = la_mrr/m - bm_mrr/m
    print(f"   {'delta':14}{d:>+9.4f}  ({d/(bm_mrr/m)*100:+.1f}% over BM25-stem)")
    print(f"\n   hit-set {nhit}: BM25 {hit_bm/max(1,nhit):.4f} -> ladder {hit_la/max(1,nhit):.4f}")
    print(f"   miss-set {nmiss}: BM25 {miss_bm/max(1,nmiss):.4f} -> ladder {miss_la/max(1,nmiss):.4f}")
    print(f"   (pool reference: BM25-stem 0.5777 -> ladder 0.6030, +11.3% over raw BM25 0.5419)")


if __name__ == "__main__":
    main()
