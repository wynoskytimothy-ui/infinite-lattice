#!/usr/bin/env python3
"""aethos_splade_lattice — the distilled, ENCODER-FREE SPLADE tier on the lattice.

SPLADE's learned weights ARE lattice postings. This packages the validated pipeline:
  BUILD   : SPLADE-encode docs once -> store learned weights as a CSR (term=WordPiece id, weight float16).
  DISTILL : SPLADE-encode each vocab WORD once -> save its expansion (the static per-word knowledge table).
  SERVE   : NO model. Query word -> look up its distilled expansion -> MAX-POOL -> scatter-add over the SPLADE
            CSR (the meet). Recovers ~92% of SPLADE's recall gain at pure-lattice speed, no GPU at query time.

The encoder (torch/transformers) is imported lazily and used ONLY at build/distill. save()/load() persist the
index + table; a loaded index serves with numpy alone (mmap-able CSR -> RAM = working set)."""
from __future__ import annotations
import os, json
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _fast_tok import words

V_WORDPIECE = 30522


class SpladeEncoder:
    """Only needed at ingest/distill (not serve). Lazy torch import."""
    def __init__(self, name="prithivida/Splade_PP_en_v1"):
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(name)
        self.m = AutoModelForMaskedLM.from_pretrained(name).eval()
        torch.set_num_threads(os.cpu_count() or 4)

    def encode(self, texts, bs=16, topk=None):
        t = self.torch
        terms, wts, off = [], [], [0]
        with t.no_grad():
            for i in range(0, len(texts), bs):
                b = self.tok(texts[i:i + bs], padding=True, truncation=True, max_length=256, return_tensors="pt")
                v = (t.log1p(t.relu(self.m(**b).logits)) * b["attention_mask"].unsqueeze(-1)).max(1).values
                for row in v:
                    nz = t.nonzero(row).squeeze(-1); w = row[nz]
                    if topk and nz.numel() > topk:
                        keep = t.topk(w, topk).indices; nz, w = nz[keep], w[keep]
                    terms.append(nz.numpy().astype(np.int32)); wts.append(w.numpy().astype(np.float32))
                    off.append(off[-1] + nz.numel())
        return np.concatenate(terms) if terms else np.zeros(0, np.int32), \
            np.concatenate(wts).astype(np.float16) if wts else np.zeros(0, np.float16), np.asarray(off, np.int64)


def _csr_by_term(terms, offs, V=V_WORDPIECE):
    di = np.repeat(np.arange(offs.size - 1), np.diff(offs)).astype(np.int32)
    order = np.argsort(terms, kind="stable"); T = terms[order]
    df = np.bincount(T, minlength=V); indptr = np.concatenate([[0], np.cumsum(df)]).astype(np.int64)
    return indptr, di[order].astype(np.uint32), order


class DistilledSpladeIndex:
    def __init__(self):
        self.doc_ids = None

    # ---- build (needs encoder) ----
    def build_docs(self, corpus, encoder, topk=128):
        self.doc_ids = list(corpus.keys()); self.N = len(self.doc_ids)
        t, w, o = encoder.encode([corpus[d] for d in self.doc_ids], topk=topk)
        self._set_docs(t, w, o)
        return self

    def _set_docs(self, terms, weights, offs):           # also used to build from cached vectors
        self.indptr, self.seg_doc, order = _csr_by_term(terms, offs)
        self.seg_w = weights[order]
        return self

    def distill(self, vocab_words, encoder, topk=64):
        self.vocab = list(vocab_words); self.w2i = {w: i for i, w in enumerate(self.vocab)}
        t, w, o = encoder.encode(self.vocab, bs=64, topk=topk)
        self.wt_terms, self.wt_w, self.wt_off = t, w, o
        return self

    def _set_table(self, vocab_words, terms, weights, offs):
        self.vocab = list(vocab_words); self.w2i = {w: i for i, w in enumerate(self.vocab)}
        self.wt_terms, self.wt_w, self.wt_off = terms, weights, offs
        return self

    # ---- serve (NO model) ----
    def _query_expansion(self, query):
        acc = {}                                          # term -> MAX weight (matches SPLADE pooling)
        for w in words(query):
            j = self.w2i.get(w)
            if j is None: continue
            a, e = int(self.wt_off[j]), int(self.wt_off[j + 1])
            for t, wt in zip(self.wt_terms[a:e], self.wt_w[a:e]):
                t = int(t); wt = float(wt)
                if wt > acc.get(t, 0.0): acc[t] = wt
        return acc

    def search(self, query, k=10):
        scores = np.zeros(self.N)
        for tid, qw in self._query_expansion(query).items():
            a, e = int(self.indptr[tid]), int(self.indptr[tid + 1])
            if e > a: scores[self.seg_doc[a:e]] += qw * self.seg_w[a:e].astype(np.float32)
        kk = min(k, self.N); top = np.argpartition(scores, -kk)[-kk:]
        top = top[np.argsort(scores[top])[::-1]]
        return [self.doc_ids[i] for i in top if scores[i] > 0]

    # ---- persistence (serve needs no model) ----
    def save(self, path):
        os.makedirs(path, exist_ok=True)
        np.save(f"{path}/indptr.npy", self.indptr); np.save(f"{path}/seg_doc.npy", self.seg_doc)
        np.save(f"{path}/seg_w.npy", self.seg_w)
        np.save(f"{path}/wt_terms.npy", self.wt_terms); np.save(f"{path}/wt_w.npy", self.wt_w)
        np.save(f"{path}/wt_off.npy", self.wt_off)
        json.dump({"doc_ids": self.doc_ids, "vocab": self.vocab, "N": self.N}, open(f"{path}/meta.json", "w"))
        return self

    @classmethod
    def load(cls, path, mmap=True):
        self = cls(); mm = "r" if mmap else None
        self.indptr = np.load(f"{path}/indptr.npy")
        self.seg_doc = np.load(f"{path}/seg_doc.npy", mmap_mode=mm)
        self.seg_w = np.load(f"{path}/seg_w.npy", mmap_mode=mm)
        self.wt_terms = np.load(f"{path}/wt_terms.npy"); self.wt_w = np.load(f"{path}/wt_w.npy")
        self.wt_off = np.load(f"{path}/wt_off.npy")
        m = json.load(open(f"{path}/meta.json"))
        self.doc_ids = m["doc_ids"]; self.vocab = m["vocab"]; self.N = m["N"]
        self.w2i = {w: i for i, w in enumerate(self.vocab)}
        return self


def _selftest():
    """Reuse the cached nfcorpus SPLADE vectors -> build, serve, verify ~92% recall recovery + save/load."""
    import time
    from scripts.bench_supervised_bridges import load, ndcg10
    from aethos_edge_rag import EdgeRAG
    CACHE = Path(os.environ.get("TEMP", "/tmp"))
    name = "nfcorpus"
    corpus, queries, train_q, test_q = load(name)
    doc_ids = list(corpus.keys()); d2i = {d: i for i, d in enumerate(doc_ids)}
    test_ids = [q for q in test_q if q in queries]
    print("=" * 84); print("DistilledSpladeIndex — encoder-free SPLADE tier on the lattice (nfcorpus)"); print("=" * 84)

    dz = np.load(CACHE / f"splade_{name}_doc128.npz")        # cached doc vectors
    vocab = sorted({w for q in test_ids for w in words(queries[q])})
    vz = np.load(CACHE / f"distill_{name}_v{len(vocab)}.npz", allow_pickle=True)  # cached per-word table
    qz = np.load(CACHE / f"splade_{name}_qry.npz")           # cached full-query (for SPLADE-full ref)

    idx = DistilledSpladeIndex(); idx.doc_ids = doc_ids; idx.N = len(doc_ids)
    idx._set_docs(dz["t"], dz["w"], dz["o"])
    idx._set_table(vz["words"].tolist(), vz["t"], vz["w"], vz["o"])

    eng = EdgeRAG().build(corpus)

    def rec(top, gold): return len(set(top) & gold) / len(gold) if gold else 0.0
    L = S = D = nq = 0; lat = []
    for qi, qid in enumerate(test_ids):
        gold = {d2i[d] for d in test_q[qid] if test_q[qid].get(d, 0) > 0 and d in d2i}
        if not gold: continue
        nq += 1
        L += rec([d2i[d] for d in eng.search(queries[qid], 100)], gold)
        sf = np.zeros(idx.N)
        for tid, w in zip(qz["t"][qz["o"][qi]:qz["o"][qi+1]], qz["w"][qz["o"][qi]:qz["o"][qi+1]]):
            a, e = int(idx.indptr[tid]), int(idx.indptr[tid+1])
            if e > a: sf[idx.seg_doc[a:e]] += float(w) * idx.seg_w[a:e].astype(np.float32)
        S += rec(list(np.argsort(sf)[::-1][:100]), gold)
        t0 = time.perf_counter(); top = idx.search(queries[qid], 100); lat.append((time.perf_counter()-t0)*1000)
        D += rec([d2i[d] for d in top], gold)
    L, S, D = L/nq, S/nq, D/nq
    print(f"  Recall@100: lexical {L:.4f} | distilled(encoder-free) {D:.4f} | SPLADE-full {S:.4f}")
    print(f"  -> recovers {(D-L)/(S-L)*100:.0f}% of SPLADE's recall gain, {np.median(lat):.2f} ms/q, NO query encoder")

    p = os.path.join(CACHE, "distilled_splade_idx"); idx.save(p)
    idx2 = DistilledSpladeIndex.load(p)
    same = idx2.search(queries[test_ids[0]], 10) == idx.search(queries[test_ids[0]], 10)
    sz = sum(os.path.getsize(os.path.join(p, f)) for f in os.listdir(p)) / idx.N
    print(f"  save/load round-trip serves identically: {same} | on-disk {sz:.0f} B/doc | model NOT needed to serve")


if __name__ == "__main__":
    _selftest()
