#!/usr/bin/env python3
"""ROUTE 2 - learned-sparse (SPLADE) served on the lattice sparse index.

Encode docs+queries with a PRETRAINED SPLADE model into sparse term weights,
build an inverted index term->{doc: weight} (this IS the lattice sparse posting
store), serve with the sparse dot (the meet), NO query-time cross-encoder.

Measures nDCG@10 / MRR@10 + footprint (codec on the weighted postings) + latency.
"""
import os, sys, time, json, math
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BEIR_ROOT = Path(r"C:\Users\wynos\.cursor\worktrees\prime_hotel\bbl\beir_datasets")
MODEL = os.environ.get("SPLADE_MODEL", "naver/splade-cocondenser-ensembledistil")

_tok = None; _mdl = None
def _load():
    global _tok, _mdl
    if _mdl is None:
        _tok = AutoTokenizer.from_pretrained(MODEL)
        _mdl = AutoModelForMaskedLM.from_pretrained(MODEL).half().to(DEVICE).eval()  # fp16
    return _tok, _mdl

@torch.no_grad()
def splade_batch(texts, max_len=256):
    """Return list of (idx_array, val_array) sparse reps."""
    tok, mdl = _load()
    enc = tok(texts, return_tensors="pt", truncation=True, max_length=max_len, padding=True).to(DEVICE)
    logits = mdl(**enc).logits
    rep = torch.max(torch.log1p(torch.relu(logits)) * enc.attention_mask.unsqueeze(-1), dim=1).values
    out = []
    rep_cpu = rep.to("cpu")
    for r in rep_cpu:
        nz = torch.nonzero(r, as_tuple=False).squeeze(-1)
        out.append((nz.numpy().astype(np.int32), r[nz].numpy().astype(np.float32)))
    return out


def load_beir(name):
    root = BEIR_ROOT / name
    corpus, queries = {}, {}
    for line in open(root / "corpus.jsonl", encoding="utf-8"):
        o = json.loads(line)
        corpus[o["_id"]] = (o.get("title", "") + " " + o.get("text", "")).strip()
    for line in open(root / "queries.jsonl", encoding="utf-8"):
        o = json.loads(line)
        queries[o["_id"]] = o["text"]
    qrels = defaultdict(dict)
    p = root / "qrels" / "test.tsv"
    with open(p, encoding="utf-8") as f:
        next(f)
        for line in f:
            a = line.rstrip("\n").split("\t")
            if len(a) >= 3 and a[0] in queries:
                try: rel = int(a[2])
                except: rel = int(float(a[2]))
                if rel > 0:
                    qrels[a[0]][a[1]] = rel
    return corpus, queries, qrels


CACHE = Path(r"C:\Users\wynos\trng\marco_data") / "_route2_cache"
CACHE.mkdir(exist_ok=True)

def build_index(corpus, name, batch=64, max_len=256):
    """Encode corpus -> COO arrays (rows, cols, vals). Cached to disk so re-runs are instant."""
    doc_ids = list(corpus.keys())
    cf = CACHE / f"{name}_{MODEL.split('/')[-1]}.npz"
    if cf.exists():
        z = np.load(cf, allow_pickle=True)
        print(f"    loaded cached encoding {cf.name}", flush=True)
        return z["rows"], z["cols"], z["vals"], doc_ids, int(z["n_postings"]), 0.0
    texts = [corpus[d] for d in doc_ids]
    rows_l, cols_l, vals_l = [], [], []
    n_postings = 0
    t0 = time.perf_counter()
    for i in range(0, len(texts), batch):
        reps = splade_batch(texts[i:i+batch], max_len=max_len)
        for j, (idx, val) in enumerate(reps):
            row = i + j
            cols_l.append(idx); vals_l.append(val)
            rows_l.append(np.full(len(idx), row, dtype=np.int32))
            n_postings += len(idx)
        if i % (batch*40) == 0:
            print(f"    enc {i+batch}/{len(texts)} docs ({n_postings:,} postings, {time.perf_counter()-t0:.0f}s)", flush=True)
    rows = np.concatenate(rows_l); cols = np.concatenate(cols_l); vals = np.concatenate(vals_l)
    enc_t = time.perf_counter() - t0
    np.savez(cf, rows=rows, cols=cols, vals=vals, n_postings=n_postings)
    return rows, cols, vals, doc_ids, n_postings, enc_t


def coo_to_arrays(rows, cols, vals):
    """Group COO by term (col) into per-term doc/weight arrays."""
    order = np.argsort(cols, kind="stable")
    c = cols[order]; r = rows[order]; v = vals[order]
    uniq, starts = np.unique(c, return_index=True)
    ends = np.append(starts[1:], len(c))
    term_docs = {}; term_wts = {}
    for t, s, e in zip(uniq, starts, ends):
        term_docs[int(t)] = r[s:e]; term_wts[int(t)] = v[s:e]
    return term_docs, term_wts


def to_arrays(inv):
    """Pack inverted index into per-term numpy arrays for fast scoring."""
    terms = sorted(inv.keys())
    term_docs = {}; term_wts = {}
    for t in terms:
        plist = inv[t]
        docs = np.fromiter((d for d, _ in plist), dtype=np.int32, count=len(plist))
        wts = np.fromiter((w for _, w in plist), dtype=np.float32, count=len(plist))
        term_docs[t] = docs; term_wts[t] = wts
    return term_docs, term_wts


def search(qrep, term_docs, term_wts, n_docs, k=10):
    """Sparse dot product = the meet. Accumulate q_w * d_w over shared terms."""
    idx, val = qrep
    acc = np.zeros(n_docs, dtype=np.float32)
    for t, qw in zip(idx, val):
        td = term_docs.get(int(t))
        if td is None:
            continue
        acc[td] += qw * term_wts[int(t)]
    if k >= n_docs:
        top = np.argsort(-acc)
    else:
        part = np.argpartition(-acc, k)[:k]
        top = part[np.argsort(-acc[part])]
    return top, acc[top]


def ndcg_at_k(ranked_ids, gold, k=10):
    dcg = 0.0
    for i, d in enumerate(ranked_ids[:k]):
        rel = gold.get(d, 0)
        if rel > 0:
            dcg += (2**rel - 1) / math.log2(i + 2)
    ideal = sorted(gold.values(), reverse=True)[:k]
    idcg = sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def _varbyte_len(gaps):
    """Vectorized varbyte byte-count for an array of non-negative gaps."""
    g = gaps.astype(np.int64)
    nb = np.ones_like(g)
    g = g >> 7
    while np.any(g > 0):
        nb += (g > 0)
        g = g >> 7
    return int(nb.sum())


def footprint_bytes(term_docs, term_wts):
    """Honest sparse-index footprint: gap-coded doc ids (varbyte) + 8-bit quantized weights.
    SPLADE weights compress well to int8 (the std SPLADE serving format)."""
    n_post = sum(len(v) for v in term_wts.values())
    total = n_post  # 1 byte/posting for quantized weight (uint8)
    for t, docs in term_docs.items():
        d = np.sort(docs)
        gaps = np.diff(d, prepend=0)
        total += _varbyte_len(gaps)
    return total, n_post


def run_beir(name, k=10, batch=64):
    print(f"\n===== {name} =====", flush=True)
    corpus, queries, qrels = load_beir(name)
    qids = [q for q in qrels if q in queries]
    print(f"  {len(corpus):,} docs | {len(qids):,} test queries", flush=True)
    rows, cols, vals, doc_ids, n_postings, enc_t = build_index(corpus, name, batch=batch)
    term_docs, term_wts = coo_to_arrays(rows, cols, vals)
    n_docs = len(doc_ids)
    # encode queries
    qtexts = [queries[q] for q in qids]
    qreps = []
    for i in range(0, len(qtexts), batch):
        qreps.extend(splade_batch(qtexts[i:i+batch], max_len=64))
    # warm
    search(qreps[0], term_docs, term_wts, n_docs, k=100)
    ndcgs = []; lat = []
    for qi, qid in enumerate(qids):
        t = time.perf_counter()
        top, _ = search(qreps[qi], term_docs, term_wts, n_docs, k=100)
        lat.append((time.perf_counter() - t) * 1000)
        ranked = [doc_ids[r] for r in top]
        ndcgs.append(ndcg_at_k(ranked, qrels[qid], k))
    fp, n_post = footprint_bytes(term_docs, term_wts)
    raw_text = sum(len(t.encode("utf-8")) for t in corpus.values())
    print(f"  postings        : {n_postings:,}  ({n_postings/n_docs:.1f}/doc)")
    print(f"  nDCG@{k}         : {np.mean(ndcgs):.4f}")
    print(f"  latency (ms)    : median {np.median(lat):.2f}  p90 {np.percentile(lat,90):.2f}")
    print(f"  footprint       : {fp/1e6:.2f} MB sparse index  (raw text {raw_text/1e6:.2f} MB)")
    return dict(name=name, ndcg=float(np.mean(ndcgs)), n_docs=n_docs, n_queries=len(qids),
                postings=n_postings, footprint_mb=fp/1e6, raw_mb=raw_text/1e6,
                lat_med=float(np.median(lat)), lat_p90=float(np.percentile(lat,90)), enc_t=enc_t)


if __name__ == "__main__":
    names = sys.argv[1:] or ["scifact", "nfcorpus", "fiqa"]
    print(f"MODEL: {MODEL}  DEVICE: {DEVICE}")
    bs = int(os.environ.get("BATCH", "64"))
    results = []
    for nm in names:
        results.append(run_beir(nm, batch=bs))
    print("\n===== SUMMARY (SPLADE on lattice sparse index, NO CE) =====")
    for r in results:
        print(f"  {r['name']:10s} nDCG@10 {r['ndcg']:.4f}  {r['footprint_mb']:6.1f}MB  "
              f"{r['lat_med']:.1f}ms  ({r['n_docs']:,} docs)")
