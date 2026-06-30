#!/usr/bin/env python3
"""SPLADE'S FOUR ROLES on the lattice (Timothy): does it raise RECALL, raise nDCG, fuse with lexical, and
rerank? Encode once (cached), then measure on the SAME lattice:
  lexical (EdgeRAG)            : nDCG@10, Recall@100
  SPLADE on lattice           : nDCG@10, Recall@100  (expansion -> recall up)
  RRF fuse (lexical + SPLADE)  : nDCG@10, Recall@100  (different misses -> recall above either)
  SPLADE rerank of lexical-100 : nDCG@10             (reranker role)
  union ceiling (recall the fused pool COULD reach)."""
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
CACHE = Path(os.environ.get("TEMP", "/tmp"))


class Splade:
    def __init__(self):
        self.tok = AutoTokenizer.from_pretrained(NAME)
        self.m = AutoModelForMaskedLM.from_pretrained(NAME).eval(); torch.set_num_threads(os.cpu_count() or 4)

    @torch.no_grad()
    def encode(self, texts, bs=16, topk=None):
        terms, weights, offs = [], [], [0]
        for i in range(0, len(texts), bs):
            b = self.tok(texts[i:i + bs], padding=True, truncation=True, max_length=256, return_tensors="pt")
            v = (torch.log1p(torch.relu(self.m(**b).logits)) * b["attention_mask"].unsqueeze(-1)).max(1).values
            for row in v:
                nz = torch.nonzero(row).squeeze(-1); w = row[nz]
                if topk and nz.numel() > topk:
                    keep = torch.topk(w, topk).indices; nz, w = nz[keep], w[keep]
                terms.append(nz.numpy().astype(np.int32)); weights.append(w.numpy().astype(np.float32))
                offs.append(offs[-1] + nz.numel())
        return np.concatenate(terms), np.concatenate(weights).astype(np.float16), np.array(offs, np.int64)


def cached_encode(sp, key, texts, topk):
    f = CACHE / f"splade_{key}.npz"
    if f.exists():
        z = np.load(f); return z["t"], z["w"], z["o"]
    t, w, o = sp.encode(texts, topk=topk)
    np.savez(f, t=t, w=w, o=o); return t, w, o


def build_csr(terms, offs, V=30522):
    di = np.repeat(np.arange(offs.size - 1), np.diff(offs)).astype(np.int32)
    order = np.argsort(terms, kind="stable"); T = terms[order]
    df = np.bincount(T, minlength=V); indptr = np.concatenate([[0], np.cumsum(df)]).astype(np.int64)
    return indptr, di[order].astype(np.uint32), order


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "scifact"
    topk = 128
    corpus, queries, train_q, test_q = load(name)
    doc_ids = list(corpus.keys()); d2i = {d: i for i, d in enumerate(doc_ids)}
    test_ids = [q for q in test_q if q in queries]; N = len(corpus)
    print("=" * 92); print(f"SPLADE'S FOUR ROLES — {name}: {N:,} docs, {len(test_ids)} queries (encode cached)"); print("=" * 92)

    sp = Splade()
    t0 = time.perf_counter()
    dt, dw, do = cached_encode(sp, f"{name}_doc{topk}", [corpus[d] for d in doc_ids], topk)
    qt, qw, qo = cached_encode(sp, f"{name}_qry", [queries[q] for q in test_ids], None)
    print(f"  encoded (or cached) in {time.perf_counter()-t0:.0f}s; {dt.size/N:.0f} terms/doc")
    indptr, seg_doc, order = build_csr(dt, do); seg_w = dw[order]

    eng = EdgeRAG().build(corpus)

    def recall_at(top_idx, gold):
        return len(set(top_idx) & gold) / len(gold) if gold else 0.0

    agg = {m: [0.0, 0.0] for m in ["lexical", "splade", "rrf", "splade_rerank"]}
    union_rec = 0.0
    for qi, qid in enumerate(test_ids):
        gold = {d2i[d] for d in test_q[qid] if test_q[qid].get(d, 0) > 0 and d in d2i}
        if not gold: continue
        lex = eng.score(queries[qid])
        spl = np.zeros(N)
        for tid, w in zip(qt[qo[qi]:qo[qi + 1]], qw[qo[qi]:qo[qi + 1]]):
            a, e = int(indptr[tid]), int(indptr[tid + 1])
            if e > a: spl[seg_doc[a:e]] += float(w) * seg_w[a:e].astype(np.float32)
        lex100 = np.argsort(lex)[::-1][:100]; spl100 = np.argsort(spl)[::-1][:100]
        # rrf over union
        lr = {int(d): r for r, d in enumerate(lex100)}; sr = {int(d): r for r, d in enumerate(spl100)}
        cand = set(lr) | set(sr)
        rrf = sorted(cand, key=lambda d: -(1.0/(60+lr.get(d, 1e6)) + 1.0/(60+sr.get(d, 1e6))))
        # splade rerank of lexical pool
        rr = sorted(lex100, key=lambda d: -spl[d])
        for m, top in [("lexical", lex100), ("splade", spl100), ("rrf", np.array(rrf)), ("splade_rerank", np.array(rr))]:
            agg[m][0] += ndcg10([doc_ids[i] for i in top[:10]], test_q[qid])
            agg[m][1] += recall_at(top[:100], gold)
        union_rec += recall_at(set(lex100) | set(spl100), gold)
    nq = sum(1 for q in test_ids if any(test_q[q].get(d, 0) > 0 and d in d2i for d in test_q[q]))

    print(f"\n  {'role':<26}{'nDCG@10':>10}{'Recall@100':>12}")
    for m in ["lexical", "splade", "rrf", "splade_rerank"]:
        print(f"  {m:<26}{agg[m][0]/nq:>10.4f}{agg[m][1]/nq:>12.4f}")
    print(f"  {'union ceiling (lex|splade)':<26}{'':>10}{union_rec/nq:>12.4f}")
    L, S, R = agg["lexical"], agg["splade"], agg["rrf"]
    print(f"\n  RECALL: lexical {L[1]/nq:.3f} -> +SPLADE expansion {S[1]/nq:.3f} -> RRF-fused {R[1]/nq:.3f} "
          f"(fusion reaches what each misses)")
    print(f"  nDCG:   lexical {L[0]/nq:.3f} -> SPLADE {S[0]/nq:.3f} -> fused {R[0]/nq:.3f} | "
          f"SPLADE-as-reranker {agg['splade_rerank'][0]/nq:.3f}")


if __name__ == "__main__":
    main()
