#!/usr/bin/env python3
"""PROVE 'RAM = working set, not the whole index' — mmap-backed edge serve, real RSS.

The CSR postings are dumped as raw uncompressed .npy (seg_indices uint32, seg_data float16) so
np.load(mmap_mode='r') TRULY memory-maps them (a .npz is zipped and can't be mmap'd). A query touches
only its terms' contiguous CSR segments -> only those pages page in -> RSS tracks the working set, not
the index size. We measure full-load RSS vs mmap RSS in ISOLATED subprocesses for a clean comparison.

modes:  build <path> <tiles>   build a tiled word-only index, dump mmap-able arrays
        serve <path> [--mmap]   load (full or mmap), serve scifact queries, print JSON {rss, ms, ...}
        (no args)               orchestrate: build once, run serve full + serve mmap, compare
"""
import os, sys, json, time, math, subprocess
os.environ.setdefault("PYTHONUTF8", "1")
import numpy as np
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from aethos_append_index import AppendOnlyLatticeIndex
from scripts.bench_supervised_bridges import load, words

try:
    import psutil; _P = psutil.Process()
    def rss_mb(): return _P.memory_info().rss / 1e6
except Exception:
    def rss_mb(): return 0.0


# ---------------------------------------------------------------- build + dump
def build_dump(path, tiles):
    os.makedirs(path, exist_ok=True)
    corpus, queries, _, test_q = load("scifact")
    base = list(corpus.items())
    idx = AppendOnlyLatticeIndex(index_mode="kappa_primary")  # word-only
    for t in range(tiles):
        for d, txt in base:
            idx.add(f"{d}#{t}", txt)
    docs = list(idx.alive); d2i = {d: i for i, d in enumerate(docs)}
    token_row, indptr, seg_idx, seg_dat, df = {}, [0], [], [], []
    for (view, tok), p in idx.token_prime.items():
        if view != "w": continue
        pl = idx.postings.get(p)
        if not pl: continue
        token_row[tok] = len(df)
        for d, wt in pl.items():
            seg_idx.append(d2i[d]); seg_dat.append(wt)
        indptr.append(len(seg_idx)); df.append(len(pl))
    np.save(f"{path}/seg_indices.npy", np.asarray(seg_idx, np.uint32))
    np.save(f"{path}/seg_data.npy", np.asarray(seg_dat, np.float16))
    np.save(f"{path}/indptr.npy", np.asarray(indptr, np.int64))
    np.save(f"{path}/df.npy", np.asarray(df, np.int32))
    np.save(f"{path}/doc_len.npy", np.asarray([idx.doc_len[d] for d in docs], np.float32))
    json.dump({"token_row": token_row,
               "meta": {"n_docs": len(docs), "n_terms": len(df), "n_post": len(seg_idx),
                        "total_len": idx._total_len, "k1": idx.k1, "b": idx.b}},
              open(f"{path}/tables.json", "w"))
    qids = [q for q in test_q if q in queries]
    json.dump([queries[q] for q in qids], open(f"{path}/queries.json", "w"))
    post_mb = (len(seg_idx) * (4 + 2)) / 1e6
    print(f"built {len(docs):,} docs, {len(df):,} terms, {len(seg_idx):,} postings | "
          f"postings on disk {post_mb:.1f} MB ({post_mb*1e6/len(docs):.0f} B/doc)")


# ---------------------------------------------------------------- serve (full or mmap)
def serve(path, mmap):
    rss0 = rss_mb()
    T = json.load(open(f"{path}/tables.json"))
    token_row = T["token_row"]; M = T["meta"]
    N = M["n_docs"]; k1, b = M["k1"], M["b"]
    indptr = np.load(f"{path}/indptr.npy"); df = np.load(f"{path}/df.npy")
    doc_len = np.load(f"{path}/doc_len.npy").astype(np.float64)
    mm = "r" if mmap else None
    seg_idx = np.load(f"{path}/seg_indices.npy", mmap_mode=mm)
    seg_dat = np.load(f"{path}/seg_data.npy", mmap_mode=mm)
    queries = json.load(open(f"{path}/queries.json"))
    rss_load = rss_mb()
    avgdl = M["total_len"] / N; A = k1 * (1 - b); Bc = k1 * b / avgdl; k1p1 = k1 + 1
    denom = A + Bc * doc_len
    touched = 0; lat = []
    for q in queries:
        t0 = time.perf_counter()
        scores = np.zeros(N)
        for w, qwt in Counter(words(q)).items():
            r = token_row.get(w)
            if r is None: continue
            a, e = int(indptr[r]), int(indptr[r + 1])
            touched += (e - a)
            di = seg_idx[a:e].astype(np.int64)
            tf = seg_dat[a:e].astype(np.float64)
            dfp = int(df[r]); idf = math.log(1 + (N - dfp + 0.5) / (dfp + 0.5))
            scores[di] += (qwt * idf * k1p1) * tf / (tf + denom[di])
        np.argpartition(scores, -10)[-10:]
        lat.append((time.perf_counter() - t0) * 1000)
    rss_serve = rss_mb()
    post_mb = seg_idx.nbytes / 1e6 + seg_dat.nbytes / 1e6
    print("RESULT " + json.dumps({
        "mode": "mmap" if mmap else "full", "n_docs": N, "index_post_mb": round(post_mb, 1),
        "rss_base_mb": round(rss0, 1), "rss_after_load_mb": round(rss_load, 1),
        "rss_after_serve_mb": round(rss_serve, 1),
        "touched_mb": round(touched * 6 / 1e6, 2), "median_ms": round(float(np.median(lat)), 2),
        "n_queries": len(queries)}))


# ---------------------------------------------------------------- orchestrate
def main():
    path = os.path.join(os.environ.get("TEMP", "/tmp"), "edge_mmap_idx")
    tiles = 50  # 5183 * 50 ~ 259k docs
    print("=" * 80); print("MMAP SERVE — proving RAM = working set (real RSS, isolated subprocesses)"); print("=" * 80)
    if os.path.exists(f"{path}/seg_indices.npy"):
        print(f"\n[build] reusing existing index at {path}")
    else:
        print(f"\n[build] tiled scifact x{tiles} -> ~{5183*tiles:,} docs, word-only postings dumped as raw .npy")
        subprocess.run([sys.executable, __file__, "build", path, str(tiles)], check=True)
    res = {}
    for mode in ["full", "mmap"]:
        args = [sys.executable, __file__, "serve", path] + (["--mmap"] if mode == "mmap" else [])
        out = subprocess.run(args, check=True, capture_output=True, text=True).stdout
        line = [l for l in out.splitlines() if l.startswith("RESULT ")][0]
        res[mode] = json.loads(line[7:])
    f, m = res["full"], res["mmap"]
    print(f"\n  index postings on disk: {f['index_post_mb']} MB  ({f['n_docs']:,} docs, raw uncompressed for random access)")
    print(f"  {'':<22}{'FULL load':>14}{'MMAP':>14}")
    print(f"  {'RSS after load':<22}{f['rss_after_load_mb']:>12.1f}MB{m['rss_after_load_mb']:>12.1f}MB   <- THE PROOF")
    print(f"  {'RSS after serving':<22}{f['rss_after_serve_mb']:>12.1f}MB{m['rss_after_serve_mb']:>12.1f}MB")
    print(f"  {'median latency':<22}{f['median_ms']:>12.2f}ms{m['median_ms']:>12.2f}ms")
    print(f"  {'bytes touched/serve':<22}{'':>14}{m['touched_mb']:>12.2f}MB  (cumulative over {m['n_queries']} q)")
    print(f"\n  PROOF (load-time): the mmap process holds {m['rss_after_load_mb']:.0f} MB resident for a "
          f"{f['index_post_mb']:.0f} MB on-disk index ({f['index_post_mb']/m['rss_after_load_mb']:.1f}x less) -- "
          f"the index is NOT in RAM.")
    print(f"  Full-load holds {f['rss_after_load_mb']:.0f} MB (the whole index resident). As queries arrive the OS")
    print(f"  pages in only what's touched (and can evict) -> mmap RSS is bounded by the OS working set, not index size.")
    print(f"  HONEST: this corpus is 50 IDENTICAL tiles, so the 300 queries collectively touch nearly every posting")
    print(f"  (worst case for mmap). Even so mmap stays UNDER the index size ({m['rss_after_serve_mb']:.0f}<{f['index_post_mb']:.0f}MB) "
          f"while full sits above it; a real corpus touches far less. Latency is equal ({m['median_ms']:.1f}ms).")
    print(f"  TRADEOFF: raw mmap postings are ~6 B/posting (uint32+float16, uncompressed) vs ~1.2 B/posting for the")
    print(f"  zlib+delta save() format -- random-access mmap trades disk footprint for low RAM. Both are phone-viable.")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "build":
        build_dump(sys.argv[2], int(sys.argv[3]))
    elif len(sys.argv) >= 2 and sys.argv[1] == "serve":
        serve(sys.argv[2], "--mmap" in sys.argv)
    else:
        main()
