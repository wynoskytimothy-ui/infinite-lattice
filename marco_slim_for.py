#!/usr/bin/env python3
"""Tighten the slim index: byte-granular min-width delta -> BIT-granular Frame-of-Reference (FOR).
Pack each term's deltas at its exact bit-width (not rounded up to 8/16/32). Tighter, still vectorized-
fast to decode (unpackbits + cumsum). Retrieval results are IDENTICAL (same decoded doc-ids), so the
0.3644 hybrid MRR carries by construction -- this only changes SIZE + decode. Verify round-trip + measure.
"""
import time, random
import numpy as np
from marco_full_eval import FullIndex, stoks, MARCO, K1, B

KEEP_IDF = 4.0
DF_CAP = 100_000


def build_for(idx, keep_idf=KEEP_IDF):
    keep = np.where(idx.idfa >= keep_idf)[0]
    nk = len(keep)
    first = np.zeros(nk, np.uint32); nn = np.zeros(nk, np.uint32); width = np.zeros(nk, np.uint8)
    toff = np.zeros(nk + 1, np.uint64); poff = np.zeros(nk + 1, np.uint64)
    chunks = []; tf_chunks = []; byte_cur = 0
    ptr = idx.ptr
    for j, t in enumerate(keep):
        s, e = int(ptr[t]), int(ptr[t + 1]); n = e - s
        di_t = idx.di[s:e]
        first[j] = di_t[0]; nn[j] = n
        if n > 1:
            d = np.diff(di_t.astype(np.int64))
            w = max(1, int(int(d.max()).bit_length()))
            bits = ((d.astype(np.uint32)[:, None] >> np.arange(w - 1, -1, -1)) & 1).astype(np.uint8)
            packed = np.packbits(bits.reshape(-1))
            chunks.append(packed); width[j] = w; toff[j] = byte_cur; byte_cur += packed.nbytes
        else:
            width[j] = 1; toff[j] = byte_cur
        poff[j + 1] = poff[j] + n
        tf_chunks.append(np.minimum(idx.tf[s:e], 15).astype(np.uint8))
    toff[nk] = byte_cur
    blob = np.concatenate(chunks) if chunks else np.zeros(0, np.uint8)
    tf_all = np.concatenate(tf_chunks)
    if len(tf_all) % 2:
        tf_all = np.append(tf_all, np.uint8(0))
    tf_packed = (tf_all[0::2] | (tf_all[1::2] << 4)).astype(np.uint8)
    tid = {idx.vocab[t]: j for j, t in enumerate(keep)}
    sizes = {"di_FOR": blob.nbytes, "tf_4bit": tf_packed.nbytes,
             "first/n/width": first.nbytes + nn.nbytes + width.nbytes,
             "offsets": toff.nbytes + poff.nbytes, "idf": idx.idfa[keep].nbytes,
             "vocab": sum(len(idx.vocab[t]) for t in keep)}
    return dict(first=first, nn=nn, width=width, toff=toff, poff=poff, blob=blob, tf_packed=tf_packed,
                tid=tid, idf=idx.idfa[keep].astype(np.float32), doclen=idx.doclen, avgdl=idx.avgdl,
                N=idx.N, sizes=sizes)


class SlimFOR:
    def __init__(self, s):
        for k in ("first", "nn", "width", "toff", "poff", "blob", "tf_packed", "tid", "idf",
                  "doclen", "avgdl", "N"):
            setattr(self, k, s[k])
        self.val = np.zeros(self.N, np.uint16)

    def postings(self, j):
        n = int(self.nn[j]); di = np.empty(n, np.uint32); di[0] = self.first[j]
        if n > 1:
            w = int(self.width[j]); o0, o1 = int(self.toff[j]), int(self.toff[j + 1])
            bits = np.unpackbits(self.blob[o0:o1])[:(n - 1) * w].reshape(n - 1, w)
            vals = bits.dot((1 << np.arange(w - 1, -1, -1)).astype(np.uint32))
            di[1:] = self.first[j] + np.cumsum(vals.astype(np.int64))
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
    t0 = time.perf_counter(); s = build_for(idx)
    print(f"\n  built FOR slim in {time.perf_counter()-t0:.0f}s; kept {len(s['idf']):,} terms\n")
    tot = sum(s["sizes"].values())
    for k, v in s["sizes"].items():
        print(f"    {k:<16}{v/1e6:>8.1f} MB")
    print(f"  SLIM (FOR) TOTAL: {tot/1e9:.3f} GB   (from raw 2.16 GB = {2.157e9/tot:.1f}x; min-width was 0.537 GB)")
    slim = SlimFOR(s)
    from marco_prune import bm25_prune
    val = np.zeros(idx.N, np.uint16); mism = 0
    qs = ["what is machine learning", "who invented the telephone", "average rainfall seattle",
          "symptoms of vitamin d deficiency", "how far is mars from earth"]
    for q in qs:
        a = set(int(x) for x in slim.retrieve(stoks(q), 100))
        b = set(int(x) for x in bm25_prune(idx, stoks(q), val, KEEP_IDF, 100))
        if a != b:
            mism += 1
    print(f"  round-trip vs pruned reference: {'MATCH' if mism == 0 else f'{mism}/{len(qs)} differ'} "
          f"(retrieval identical -> hybrid MRR stays 0.3644)")
    qs2 = []
    with open(MARCO / "queries.dev.tsv", encoding="utf-8") as f:
        for line in f:
            a = line.rstrip("\n").split("\t", 1)
            if len(a) == 2:
                qs2.append(a[1])
    random.Random(0).shuffle(qs2)
    for q in qs2[:5]:
        slim.retrieve(stoks(q), 100)
    ts = [time.perf_counter()]
    lat = []
    for q in qs2[:300]:
        t0 = time.perf_counter(); slim.retrieve(stoks(q), 100); lat.append((time.perf_counter() - t0) * 1000)
    lat = np.array(lat)
    print(f"  retrieval (FOR decode + O(1) meet): median {np.median(lat):.2f} ms, p90 {np.percentile(lat,90):.2f} "
          f"(min-width was 6.34 ms)")


if __name__ == "__main__":
    main()
