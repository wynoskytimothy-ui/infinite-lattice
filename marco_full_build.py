#!/usr/bin/env python3
"""Build + cache the FULL 8.8M MS MARCO stemmed inverted index as a numpy CSR.

One-time build (single sequential pass over collection.tsv, memoized conservative stem).
Peak RAM ~3-4 GB; cache written to marco_data/full_idx_* (outside the repo). Then
marco_full_eval.py loads it and runs the diagnostic-tuned ladder on the full collection,
fetching the top-100 passage texts on demand via collection.offsets.npy.

This is the magnitude test: the ladder was tuned on a 298k random-distractor pool (hit-set
74%); the full 8.8M has ~30x the distractors and ~topically-harder ones, so the miss-set
becomes the majority. Does the +11.3% relative gain hold, grow, or collapse?
"""
import re, time, math, pickle
from array import array
from pathlib import Path
import numpy as np
from stem_safe import safe

MARCO = Path(r"C:\Users\wynos\trng\marco_data")
TOK = re.compile(r"[a-z0-9]+")

_st = {}
def st(w):
    r = _st.get(w)
    if r is None:
        r = safe(w); _st[w] = r
    return r


def main():
    post_di, post_tf = {}, {}
    doclen = []
    total_tokens = 0
    t0 = time.perf_counter()
    print("building full 8.8M stemmed inverted index (single pass)...", flush=True)
    with open(MARCO / "collection.tsv", encoding="utf-8", errors="replace") as f:
        for ln, line in enumerate(f):
            tab = line.find("\t")
            if tab < 0:
                doclen.append(0); continue
            text = line[tab + 1:]
            tf = {}
            for w in TOK.findall(text.lower()):
                s = st(w)
                if s:
                    tf[s] = tf.get(s, 0) + 1
            dl = 0
            for s, c in tf.items():
                pd = post_di.get(s)
                if pd is None:
                    pd = array('I'); post_di[s] = pd; post_tf[s] = array('H')
                    pt = post_tf[s]
                else:
                    pt = post_tf[s]
                pd.append(ln)
                pt.append(c if c < 65536 else 65535)
                dl += c
            doclen.append(dl)
            total_tokens += dl
            if (ln + 1) % 1_000_000 == 0:
                rss = sum(len(v) for v in post_di.values())
                print(f"  {ln+1:,} docs | {len(post_di):,} terms | {rss:,} postings | "
                      f"{time.perf_counter()-t0:.0f}s", flush=True)
    N = len(doclen)
    doclen = np.asarray(doclen, dtype=np.uint32)
    avgdl = total_tokens / N
    print(f"  indexed {N:,} docs, {len(post_di):,} terms, {total_tokens:,} tokens "
          f"(avgdl {avgdl:.1f}) in {time.perf_counter()-t0:.0f}s", flush=True)

    # --- pack into CSR numpy ---
    t1 = time.perf_counter()
    vocab = list(post_di.keys())
    V = len(vocab)
    df = np.fromiter((len(post_di[t]) for t in vocab), dtype=np.uint64, count=V)
    P = int(df.sum())
    ptr = np.zeros(V + 1, dtype=np.uint64)
    np.cumsum(df, out=ptr[1:])
    di_all = np.empty(P, dtype=np.uint32)
    tf_all = np.empty(P, dtype=np.uint16)
    for tid, t in enumerate(vocab):
        s, e = int(ptr[tid]), int(ptr[tid + 1])
        di_all[s:e] = np.frombuffer(post_di[t], dtype=np.uint32)
        tf_all[s:e] = np.frombuffer(post_tf[t], dtype=np.uint16)
    idf = np.log((N - df.astype(np.float64) + 0.5) / (df.astype(np.float64) + 0.5) + 1.0).astype(np.float32)
    print(f"  packed CSR: {P:,} postings ({di_all.nbytes/1e9:.2f}GB di + {tf_all.nbytes/1e9:.2f}GB tf) "
          f"in {time.perf_counter()-t1:.0f}s", flush=True)

    np.save(MARCO / "full_idx_di.npy", di_all)
    np.save(MARCO / "full_idx_tf.npy", tf_all)
    np.save(MARCO / "full_idx_ptr.npy", ptr)
    np.save(MARCO / "full_idx_doclen.npy", doclen)
    np.save(MARCO / "full_idx_idf.npy", idf)
    with open(MARCO / "full_idx_meta.pkl", "wb") as f:
        pickle.dump({"vocab": vocab, "N": N, "avgdl": avgdl, "P": P}, f)
    disk = (di_all.nbytes + tf_all.nbytes + ptr.nbytes + doclen.nbytes + idf.nbytes) / 1e9
    print(f"  cached full_idx_* to {MARCO} (~{disk:.2f}GB). total {time.perf_counter()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
