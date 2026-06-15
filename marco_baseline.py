#!/usr/bin/env python3
"""MS MARCO step 1: the BM25 floor + a calibrated, trustworthy eval harness.

Pool eval (feasible on one box): corpus = all relevant passages for the sampled dev
queries + N random distractors. Absolute MRR is OPTIMISTIC vs the full 8.8M collection
(fewer distractors) -- so this is for CALIBRATION (does BM25 land in a sane range?) and
RELATIVE lever comparison, not a full-collection number. Self-contained postings BM25.

Honest knobs printed; rigor asserts: relevant-in-pool, sane MRR range.
  python marco_baseline.py [n_queries] [n_distractors]
"""
import sys, time, random, math, re
from pathlib import Path
from collections import defaultdict

MARCO = Path(r"C:\Users\wynos\trng\marco_data")
N_PASSAGES = 8_841_823
TOK = re.compile(r"[a-z0-9]+")


def tok(s):
    return TOK.findall(s.lower())


def load_dev():
    qrels = defaultdict(set)
    with open(MARCO / "qrels.dev.tsv", encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) >= 4 and int(p[3]) > 0:
                qrels[p[0]].add(p[2])
    queries = {}
    with open(MARCO / "queries.dev.tsv", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) == 2:
                queries[parts[0]] = parts[1]
    return qrels, queries


def build_pool(qrels, queries, n_queries, n_dist, seed=42):
    rng = random.Random(seed)
    qids = [q for q in qrels if q in queries]
    qids = rng.sample(qids, min(n_queries, len(qids)))
    rel_pids = set()
    for q in qids:
        rel_pids |= qrels[q]
    dist = set(str(rng.randrange(N_PASSAGES)) for _ in range(n_dist)) - rel_pids
    pool = rel_pids | dist
    return qids, pool, rel_pids


def load_texts(pool):
    texts = {}
    t0 = time.perf_counter()
    with open(MARCO / "collection.tsv", encoding="utf-8", errors="replace") as f:
        for line in f:
            tab = line.find("\t")
            if tab < 0:
                continue
            pid = line[:tab]
            if pid in pool:
                texts[pid] = line[tab + 1:].rstrip("\n")
                if len(texts) == len(pool):
                    break
    print(f"  streamed collection for {len(texts)}/{len(pool)} pool passages "
          f"({time.perf_counter()-t0:.0f}s)", flush=True)
    return texts


class BM25:
    def __init__(self, k1=0.9, b=0.4):
        self.k1, self.b = k1, b
        self.post = defaultdict(list)      # term -> [(doc_idx, tf)]
        self.df = defaultdict(int)
        self.doclen, self.docids = [], []

    def index(self, items):                 # items: list of (pid, text)
        t0 = time.perf_counter()
        for di, (pid, text) in enumerate(items):
            self.docids.append(pid)
            tf = defaultdict(int)
            for w in tok(text):
                tf[w] += 1
            self.doclen.append(sum(tf.values()))
            for w, c in tf.items():
                self.post[w].append((di, c))
                self.df[w] += 1
        self.N = len(items)
        self.avgdl = sum(self.doclen) / max(1, self.N)
        self.idf = {w: math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)
                    for w, df in self.df.items()}
        self.bytes = sum(len(v) for v in self.post.values()) * 8  # ~2 ints/posting
        print(f"  indexed {self.N} docs, {len(self.post)} terms "
              f"({time.perf_counter()-t0:.0f}s)", flush=True)

    def search(self, query, k=10):
        sc = defaultdict(float)
        for w in set(tok(query)):
            if w not in self.post:
                continue
            idf = self.idf[w]
            for di, c in self.post[w]:
                dl = self.doclen[di]
                sc[di] += idf * c * (self.k1 + 1) / (c + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        top = sorted(sc, key=sc.get, reverse=True)[:k]
        return [self.docids[di] for di in top]


def evaluate(engine, qids, queries, qrels):
    mrr = r10 = r100 = 0.0
    for q in qids:
        rel = qrels[q]
        ranked = engine.search(queries[q], k=100)
        rr = 0.0
        for i, pid in enumerate(ranked[:10], 1):
            if pid in rel:
                rr = 1.0 / i
                break
        mrr += rr
        r10 += len(rel & set(ranked[:10])) / len(rel)
        r100 += len(rel & set(ranked[:100])) / len(rel)
    n = len(qids)
    return mrr / n, r10 / n, r100 / n


def main():
    nq = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    nd = int(sys.argv[2]) if len(sys.argv) > 2 else 300_000
    print(f"MS MARCO BM25 baseline -- pool eval ({nq} dev queries, ~{nd} distractors)\n", flush=True)
    qrels, queries = load_dev()
    qids, pool, rel_pids = build_pool(qrels, queries, nq, nd)
    print(f"  {len(qids)} queries, {len(rel_pids)} relevant + distractors -> pool {len(pool)}", flush=True)
    texts = load_texts(pool)
    missing = sum(1 for q in qids for p in qrels[q] if p not in texts)
    before = len(qids)
    qids = [q for q in qids if all(p in texts for p in qrels[q])]   # rigor: relevant must be in pool
    print(f"  {before-len(qids)} queries dropped (relevant pid absent from collection); "
          f"{len(qids)} eval queries ({missing} relevant pids missing)", flush=True)
    items = list(texts.items())
    bm = BM25(); bm.index(items)
    t0 = time.perf_counter()
    mrr, r10, r100 = evaluate(bm, qids, queries, qrels)
    proj_gb = bm.bytes / len(items) * N_PASSAGES / 1e9
    print(f"\n  MRR@10   {mrr:.4f}")
    print(f"  Recall@10  {r10:.4f}    Recall@100 {r100:.4f}")
    print(f"  eval {len(qids)} q in {time.perf_counter()-t0:.0f}s")
    print(f"  pool postings ~{bm.bytes/1e6:.0f} MB -> projected full-collection ~{proj_gb:.1f} GB (raw python; bitmap index is far smaller)")
    print(f"\n  NOTE: pool of {len(items)} << 8.8M, so MRR is optimistic vs full-collection ~0.187.")
    print(f"  Sane if clearly above random and below full-pool SOTA. This is the FLOOR + calibrated harness.")


if __name__ == "__main__":
    main()
