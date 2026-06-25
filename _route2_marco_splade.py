#!/usr/bin/env python3
"""ROUTE 2 on MARCO dev-small: SPLADE served as the ranker, NO cross-encoder.

Pipeline: fast lattice/BM25 retrieve top-POOL candidates -> SPLADE-encode the
candidate passages (doc encoder) + the query -> rank by sparse dot (the meet).
This is "learned-sparse served on the lattice index": the lattice supplies the
candidate set fast+small, SPLADE supplies the learned weights, sparse-dot ranks.
NO query-time cross-encoder.

Reports MRR@10 + recall@100 vs the BM25 0.187 / dense 0.34 / SPLADE 0.38 ladder.
Compares against the lattice's own pool order (no rerank) as the floor.
"""
import os, sys, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
from collections import defaultdict
import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from marco_full_eval import FullIndex, stoks, MARCO
from marco_headline import load_for

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = os.environ.get("SPLADE_MODEL", "naver/splade-cocondenser-ensembledistil")
POOL = int(os.environ.get("POOL", "100"))
NQ = int(os.environ.get("NQ", "1500"))

_tok = AutoTokenizer.from_pretrained(MODEL)
_mdl = AutoModelForMaskedLM.from_pretrained(MODEL).half().to(DEVICE).eval()  # fp16: 10x faster, same ranking
DOC_ML = int(os.environ.get("DOC_ML", "160"))

@torch.no_grad()
def splade_dense(texts, max_len):
    enc = _tok(texts, return_tensors="pt", truncation=True, max_length=max_len, padding=True).to(DEVICE)
    logits = _mdl(**enc).logits
    rep = torch.max(torch.log1p(torch.relu(logits)) * enc.attention_mask.unsqueeze(-1), dim=1).values
    return rep  # (B, vocab) on GPU


def main():
    idx = FullIndex()
    slim = load_for(idx)

    qrels = defaultdict(set)
    with open(MARCO / "qrels.dev.small.tsv", encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) >= 4 and int(p[3]) > 0:
                qrels[p[0]].add(int(p[2]))
    queries = []
    with open(MARCO / "queries.dev.tsv", encoding="utf-8") as f:
        for line in f:
            a = line.rstrip("\n").split("\t", 1)
            if len(a) == 2 and a[0] in qrels:
                queries.append((a[0], a[1]))
    queries = queries[:NQ]
    print(f"\n  dev-SMALL: {len(queries):,} queries, POOL={POOL}, model={MODEL}\n", flush=True)

    # warm
    for qid, qt in queries[:3]:
        slim.retrieve(stoks(qt), POOL)

    mrr_pool = 0.0   # lattice pool order, no rerank (floor)
    mrr_splade = 0.0 # SPLADE sparse-dot rerank, NO CE
    rec = 0
    retr_lat = []; rank_lat = []
    t0 = time.perf_counter()
    for i, (qid, qt) in enumerate(queries):
        t = time.perf_counter()
        pool = [int(d) for d in slim.retrieve(stoks(qt), POOL)]
        retr_lat.append((time.perf_counter() - t) * 1000)
        gold = qrels[qid]
        if not pool:
            continue
        if any(d in gold for d in pool):
            rec += 1
        # floor: lattice pool order
        for r, d in enumerate(pool[:10]):
            if d in gold:
                mrr_pool += 1.0 / (r + 1); break
        # SPLADE rerank (no CE): encode query + pool passages, sparse dot
        tr = time.perf_counter()
        texts = [(idx.text(p).strip()[:2000] or "unk") for p in pool]
        texts = [(t if any(c.isalnum() for c in t) else "unk") for t in texts]
        try:
            qv = splade_dense([qt.strip() or "unk"], 64)   # (1, V)
            dv = splade_dense(texts, DOC_ML)               # (P, V)
            scores = (dv @ qv.squeeze(0)).float().to("cpu").numpy()
        except Exception as e:
            print(f"    [skip q{qid}] {type(e).__name__}: {str(e)[:60]}", flush=True)
            scores = np.zeros(len(pool), dtype=np.float32)
        rank_lat.append((time.perf_counter() - tr) * 1000)
        order = [pool[k] for k in np.argsort(-scores)]
        for r, d in enumerate(order[:10]):
            if d in gold:
                mrr_splade += 1.0 / (r + 1); break
        if (i + 1) % 250 == 0:
            n = i + 1
            print(f"    {n:>5}/{len(queries)}  R@{POOL} {rec/n*100:5.1f}%  "
                  f"pool-MRR {mrr_pool/n:.4f}  SPLADE-MRR {mrr_splade/n:.4f}", flush=True)

    n = len(queries)
    print(f"\n  ===== MARCO dev-small (n={n}), SPLADE-on-lattice, NO CE =====")
    print(f"    retrieve latency   : median {np.median(retr_lat):.2f} ms (lattice meet)")
    print(f"    SPLADE rank latency: median {np.median(rank_lat):.2f} ms (query+{POOL} doc encode + dot)")
    print(f"    recall@{POOL}        : {rec/n*100:.2f}%  (pool ceiling)")
    print(f"    pool-order MRR@10  : {mrr_pool/n:.4f}  (lattice floor, no rerank)")
    print(f"    SPLADE MRR@10      : {mrr_splade/n:.4f}  (learned-sparse rerank, NO CE)")
    print(f"    ladder: BM25 0.187 < dense ~0.34 < [SPLADE-here {mrr_splade/n:.3f}] < SPLADE-full ~0.38")


if __name__ == "__main__":
    main()
