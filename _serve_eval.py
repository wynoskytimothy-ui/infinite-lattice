#!/usr/bin/env python3
"""Clean driver for the REAL full-8.8M MARCO MRR (no phantom background process).
Sets WORK to the full SPLADE index, runs serve over dev-small (all 6,980 q, or a sample
via argv[1]). On --full the index holds all 8.8M docs, so mrr_all = the true headline
(gold competes against the whole corpus, not a 50k slice)."""
import os, sys, time

os.environ["WORK"] = r"C:\Users\wynos\trng\marco_data\splade_native_full"
os.environ.setdefault("PYTHONUTF8", "1")

import marco_splade_native as m

nq = int(sys.argv[1]) if len(sys.argv) > 1 else None
t0 = time.perf_counter()
res = m.serve(tag=" full", nq=nq)
print(f"\nRESULT nq={nq} mrr_all={res['mrr_all']:.4f} recall@100={res['rec_all']:.2f}% "
      f"lat_med={res['lat_med']:.2f}ms lat_p90={res['lat_p90']:.2f}ms "
      f"wall={time.perf_counter()-t0:.0f}s", flush=True)
