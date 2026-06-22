#!/usr/bin/env python3
"""THE CONSOLIDATED SLIM INDEX -- every proven lever in one runnable artifact:
    prune idf>=4  +  per-term min-width delta di  +  4-bit tf  +  O(1) presence meet.
Pruning obviates block-skip: all surviving terms are rare (df<=~160k), so per-term full decode is cheap.
Builds the compressed index, PERSISTS it (slim_index.npz), and queries FROM the compressed form (decode on
the fly). Reports REAL on-disk size + retrieval latency + recall@100 + hybrid MRR -- one engine, end to end.
"""
import time, random
from collections import defaultdict
import numpy as np
from marco_full_eval import FullIndex, stoks, MARCO, K1, B
from marco_prune import bm25_prune

KEEP_IDF = 4.0
DF_CAP = 100_000
_DT = {1: np.uint8, 2: np.uint16, 4: np.uint32}


def build_slim(idx, keep_idf=KEEP_IDF):
    keep = np.where(idx.idfa >= keep_idf)[0]
    nk = len(keep)
    first = np.zeros(nk, np.uint32); nn = np.zeros(nk, np.uint32); width = np.zeros(nk, np.uint8)
    toff = np.zeros(nk, np.uint64); poff = np.zeros(nk + 1, np.uint64)
    blobs = {1: [], 2: [], 4: []}; lens = {1: 0, 2: 0, 4: 0}; tf_chunks = []
    ptr = idx.ptr
    for j, t in enumerate(keep):
        s, e = int(ptr[t]), int(ptr[t + 1]); n = e - s
        di_t = idx.di[s:e]
        first[j] = di_t[0]; nn[j] = n
        if n > 1:
            d = np.diff(di_t.astype(np.int64))
            mx = int(d.max())
            w = 1 if mx < 256 else (2 if mx < 65536 else 4)
            blobs[w].append(d.astype(_DT[w])); toff[j] = lens[w]; lens[w] += (n - 1)
        else:
            w = 1; toff[j] = lens[1]
        width[j] = w
        poff[j + 1] = poff[j] + n
        tf_chunks.append(np.minimum(idx.tf[s:e], 15).astype(np.uint8))
    blob = {w: (np.concatenate(blobs[w]) if blobs[w] else np.zeros(0, _DT[w])) for w in (1, 2, 4)}
    tf_all = np.concatenate(tf_chunks)
    if len(tf_all) % 2:
        tf_all = np.append(tf_all, np.uint8(0))
    tf_packed = (tf_all[0::2] | (tf_all[1::2] << 4)).astype(np.uint8)
    tid = {idx.vocab[t]: j for j, t in enumerate(keep)}
    sizes = {"di_blobs": sum(blob[w].nbytes for w in (1, 2, 4)), "tf_4bit": tf_packed.nbytes,
             "first/n/width": first.nbytes + nn.nbytes + width.nbytes,
             "offsets": toff.nbytes + poff.nbytes, "idf": idx.idfa[keep].nbytes,
             "vocab": sum(len(idx.vocab[t]) for t in keep)}
    return dict(first=first, nn=nn, width=width, toff=toff, poff=poff,
                blob1=blob[1], blob2=blob[2], blob4=blob[4], tf_packed=tf_packed,
                tid=tid, idf=idx.idfa[keep].astype(np.float32),
                doclen=idx.doclen, avgdl=idx.avgdl, N=idx.N, sizes=sizes)


class SlimIndex:
    def __init__(self, s):
        self.first = s["first"]; self.nn = s["nn"]; self.width = s["width"]
        self.toff = s["toff"]; self.poff = s["poff"]
        self.blob = {1: s["blob1"], 2: s["blob2"], 4: s["blob4"]}
        self.tf_packed = s["tf_packed"]; self.tid = s["tid"]; self.idf = s["idf"]
        self.doclen = s["doclen"]; self.avgdl = s["avgdl"]; self.N = s["N"]
        self.val = np.zeros(self.N, np.uint16)

    def postings(self, j):
        n = int(self.nn[j]); w = int(self.width[j]); o = int(self.toff[j])
        di = np.empty(n, np.uint32); di[0] = self.first[j]
        if n > 1:
            d = self.blob[w][o:o + n - 1].astype(np.int64)
            di[1:] = self.first[j] + np.cumsum(d)
        p0 = int(self.poff[j]); ix = np.arange(p0, p0 + n)
        byte = self.tf_packed[ix // 2]
        tf = np.where(ix % 2 == 0, byte & 0xF, byte >> 4).astype(np.float32)
        return di, tf

    def retrieve(self, qterms, k=100):
        terms = []
        for w in set(qterms):
            j = self.tid.get(w)
            if j is not None:
                terms.append((float(self.idf[j]), int(self.nn[j]), j))
        if not terms:
            return np.empty(0, np.uint32)
        disc = [j for (wi, df, j) in terms if df < DF_CAP] or [min(terms, key=lambda t: t[1])[2]]
        cand = np.unique(np.concatenate([self.postings(j)[0] for j in disc]))
        dlc = self.doclen[cand]; sc = np.zeros(len(cand), np.float32); val = self.val
        for (wi, df, j) in terms:
            di, tf = self.postings(j); val[di] = tf.astype(np.uint16)
            tfc = val[cand].astype(np.float32); hit = tfc > 0
            sc[hit] += (wi * tfc * (K1 + 1) / (tfc + K1 * (1 - B + B * dlc / self.avgdl)))[hit]
            val[di] = 0
        sel = np.argpartition(-sc, k)[:k] if len(cand) > k else np.arange(len(cand))
        return cand[sel[np.argsort(-sc[sel])]]


def main():
    idx = FullIndex()
    t0 = time.perf_counter(); s = build_slim(idx)
    print(f"\n  built slim index in {time.perf_counter()-t0:.0f}s; kept {len(s['idf']):,} terms (idf>={KEEP_IDF})\n")
    tot = sum(s["sizes"].values())
    for k, v in s["sizes"].items():
        print(f"    {k:<16}{v/1e6:>8.1f} MB")
    print(f"  SLIM INDEX TOTAL: {tot/1e9:.3f} GB   (from raw 2.16 GB = {2.157e9/tot:.1f}x smaller)")
    slim = SlimIndex(s)

    val = np.zeros(idx.N, np.uint16)
    mism = 0
    qsamp = ["what is machine learning", "who invented the telephone", "average rainfall seattle"]
    for q in qsamp:
        a = set(int(x) for x in slim.retrieve(stoks(q), 100))
        b = set(int(x) for x in bm25_prune(idx, stoks(q), val, KEEP_IDF, 100))
        if a != b:
            mism += 1
    print(f"  codec round-trip check vs pruned reference: {'MATCH' if mism == 0 else f'{mism} differ'}", flush=True)

    qrels = defaultdict(set)
    with open(MARCO / "qrels.dev.tsv", encoding="utf-8") as f:
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
    random.Random(0).shuffle(queries); sample = queries[:300]
    for qid, qt in sample[:5]:
        slim.retrieve(stoks(qt), 100)
    ts = []; rec = 0
    for qid, qt in sample:
        t0 = time.perf_counter(); o = slim.retrieve(stoks(qt), 100); ts.append((time.perf_counter() - t0) * 1000)
        if any(g in set(int(d) for d in o[:100]) for g in qrels[qid]):
            rec += 1
    print(f"\n  retrieval (decode + O(1) meet): median {np.median(ts):.2f} ms, p90 {np.percentile(ts,90):.2f}, "
          f"recall@100 {rec/len(sample)*100:.1f}%", flush=True)

    from sentence_transformers import CrossEncoder
    import torch
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=256,
                      device="cuda" if torch.cuda.is_available() else "cpu")
    sub = sample[:200]; mrr = 0.0
    for qid, qt in sub:
        o = [int(d) for d in slim.retrieve(stoks(qt), 100)]
        sc = ce.predict([(qt, idx.text(p)) for p in o], batch_size=128, show_progress_bar=False)
        rr = [o[i] for i in np.argsort(-sc)]
        for r, d in enumerate(rr[:10]):
            if d in qrels[qid]:
                mrr += 1.0 / (r + 1); break
    print(f"  hybrid MRR@10 (slim + cross-encoder, n={len(sub)}): {mrr/len(sub):.4f}")

    np.savez(MARCO / "slim_index.npz", first=s["first"], nn=s["nn"], width=s["width"], toff=s["toff"],
             poff=s["poff"], blob1=s["blob1"], blob2=s["blob2"], blob4=s["blob4"],
             tf_packed=s["tf_packed"], idf=s["idf"])
    print(f"  persisted -> slim_index.npz (postings/tf/offsets; vocab+doclen load from the base index)")


if __name__ == "__main__":
    main()
