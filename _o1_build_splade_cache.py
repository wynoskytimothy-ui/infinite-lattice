#!/usr/bin/env python3
"""Build SPLADE doc + per-word distill caches (GPU) for the headroom/fusion stack, matching the cache keys
_o1_headroom.py / _o1_fuse.py expect: splade_{name}_doc128.npz (t,w,o) and distill_{name}_v{V}.npz (words,t,w,o)."""
import os, sys, time
os.environ.setdefault("PYTHONUTF8", "1")
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from aethos_splade_lattice import SpladeEncoder
from scripts.bench_supervised_bridges import load
from _fast_tok import words

CACHE = Path(os.environ.get("TEMP", "/tmp"))


def main():
    corpora = sys.argv[1].split(",") if len(sys.argv) > 1 else ["scifact", "fiqa", "trec-covid"]
    enc = SpladeEncoder(); enc.encode(["warm"], topk=128)
    print(f"device={enc.device}")
    for name in corpora:
        corpus, queries, _, test_q = load(name)
        doc_ids = list(corpus.keys()); d2i = {d: i for i, d in enumerate(doc_ids)}
        ids = [q for q in test_q if q in queries and any(test_q[q].get(d, 0) > 0 and d in d2i for d in test_q[q])]
        df = CACHE / f"splade_{name}_doc128.npz"
        if not df.exists():
            t0 = time.perf_counter(); t, w, o = enc.encode([corpus[d] for d in doc_ids], topk=128)
            np.savez(df, t=t, w=w, o=o); print(f"  {name}: doc cache {len(corpus):,} in {time.perf_counter()-t0:.0f}s")
        vocab = sorted({x for q in ids for x in words(queries[q])})
        vf = CACHE / f"distill_{name}_v{len(vocab)}.npz"
        if not vf.exists():
            t, w, o = enc.encode(vocab, topk=64)
            np.savez(vf, words=np.array(vocab, object), t=t, w=w, o=o); print(f"  {name}: distill {len(vocab)} words")
        print(f"  {name}: ready (docs={df.exists()}, distill={vf.exists()})")


if __name__ == "__main__":
    main()
