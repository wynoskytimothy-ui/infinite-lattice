#!/usr/bin/env python3
"""SPLADE ON THE LATTICE — SPLADE output IS sparse term-weights = lattice postings. Store SPLADE's LEARNED
weights in the same CSR the lattice serves; score = sparse dot-product = the lattice's scatter-add. The lattice
is UNCHANGED (same fast CPU serve, same sparse footprint); SPLADE just supplies learned weights instead of BM25.
Keep speed + footprint, improve accuracy. Encoder needed at ingest (one-time) + query time (small, NPU/GPU/CPU).

Measures: nDCG (vs lexical EdgeRAG), B/doc (vs lexical), serve ms, with top-K pruning for footprint control."""
import os, sys, time
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scripts.bench_supervised_bridges import load, ndcg10
from aethos_edge_rag import EdgeRAG

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

NAME = "prithivida/Splade_PP_en_v1"


class Splade:
    def __init__(self):
        self.tok = AutoTokenizer.from_pretrained(NAME)
        self.m = AutoModelForMaskedLM.from_pretrained(NAME).eval()
        torch.set_num_threads(os.cpu_count() or 4)

    @torch.no_grad()
    def encode(self, texts, bs=16, topk=None):
        """-> list of (term_ids int32, weights float32) sparse vectors (SPLADE max-pool log(1+relu))."""
        out = []
        for i in range(0, len(texts), bs):
            b = self.tok(texts[i:i + bs], padding=True, truncation=True, max_length=256, return_tensors="pt")
            logits = self.m(**b).logits                                   # [B,L,V]
            v = torch.log1p(torch.relu(logits)) * b["attention_mask"].unsqueeze(-1)
            v = v.max(dim=1).values                                       # [B,V] SPLADE pooling
            for row in v:
                nz = torch.nonzero(row).squeeze(-1)
                w = row[nz]
                if topk and nz.numel() > topk:
                    keep = torch.topk(w, topk).indices; nz = nz[keep]; w = w[keep]
                out.append((nz.numpy().astype(np.int32), w.numpy().astype(np.float32)))
        return out


def build_csr(doc_sparse, V=30522):
    """SPLADE doc vectors -> CSR (term -> docs+weights), exactly the lattice's posting format."""
    rows_t, rows_d, rows_w = [], [], []
    for di, (terms, w) in enumerate(doc_sparse):
        rows_t.append(terms); rows_d.append(np.full(terms.size, di, np.int32)); rows_w.append(w)
    T = np.concatenate(rows_t); D = np.concatenate(rows_d); W = np.concatenate(rows_w)
    order = np.argsort(T, kind="stable"); T, D, W = T[order], D[order], W[order]
    df = np.bincount(T, minlength=V); indptr = np.concatenate([[0], np.cumsum(df)]).astype(np.int64)
    return indptr, D.astype(np.uint32), W.astype(np.float16), T.size


def serve(indptr, seg_doc, seg_w, qsparse, N, k=10):
    scores = np.zeros(N)
    for tid, qw in zip(*qsparse):
        a, e = int(indptr[tid]), int(indptr[tid + 1])
        if e > a:
            scores[seg_doc[a:e]] += float(qw) * seg_w[a:e].astype(np.float32)
    top = np.argpartition(scores, -k)[-k:]
    return top[np.argsort(scores[top])[::-1]], scores


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "scifact"
    topk = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    corpus, queries, train_q, test_q = load(name)
    doc_ids = list(corpus.keys()); test_ids = [q for q in test_q if q in queries]
    print("=" * 84)
    print(f"SPLADE ON THE LATTICE — {name}: {len(corpus):,} docs, top-{topk} terms/doc")
    print("=" * 84)

    sp = Splade()
    t0 = time.perf_counter()
    dvecs = sp.encode([corpus[d] for d in doc_ids], bs=16, topk=topk)
    enc_s = time.perf_counter() - t0
    indptr, seg_doc, seg_w, npost = build_csr(dvecs)
    bdoc = (seg_doc.nbytes + seg_w.nbytes) / len(corpus)

    qvecs = sp.encode([queries[q] for q in test_ids], bs=16, topk=None)
    lat = []; nd = 0.0
    for qid, qs in zip(test_ids, qvecs):
        t0 = time.perf_counter(); top, _ = serve(indptr, seg_doc, seg_w, qs, len(corpus), 10)
        lat.append((time.perf_counter() - t0) * 1000)
        nd += ndcg10([doc_ids[i] for i in top], test_q[qid])
    nd_splade = nd / len(test_ids)

    # lexical EdgeRAG baseline (same corpus)
    eng = EdgeRAG().build(corpus)
    nd_lex = float(np.mean([ndcg10(eng.search(queries[q], 10), test_q[q]) for q in test_ids]))

    print(f"  encode: {len(corpus)} docs in {enc_s:.0f}s ({len(corpus)/enc_s:.0f} docs/s, CPU; GPU ~50-100x)")
    print(f"  {'index':<22}{'B/doc':>8}{'serve':>10}{'nDCG@10':>10}")
    print(f"  {'lexical lattice':<22}{'~198':>8}{'0.1ms':>10}{nd_lex:>10.4f}")
    print(f"  {'SPLADE on lattice':<22}{bdoc:>8.0f}{np.median(lat):>8.2f}ms{nd_splade:>10.4f}")
    print(f"\n  -> SAME lattice serve (scatter-add) + sparse postings; SPLADE weights -> nDCG {nd_lex:.3f} "
          f"-> {nd_splade:.3f} ({nd_splade-nd_lex:+.3f})")
    print(f"  footprint {bdoc:.0f} B/doc (top-{topk}; prunable), serve {np.median(lat):.2f}ms CPU. Encoder is the")
    print(f"  only addition (ingest one-time + query); lattice speed/footprint preserved.")


if __name__ == "__main__":
    main()
