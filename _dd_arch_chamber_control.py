#!/usr/bin/env python3
"""
CONTROL for _dd_arch_chamber_residual: isolate how much shrink comes from the chamber-residual
HIERARCHY vs from plain weight-bit quantization alone. Runs:
  (0) baseline                       : di FOR gaps + uint8 weights
  (1) weight-requant only (w bits)   : SAME inverted index, weights -> w bits (NO chambers, NO base)
  (1b) + di pure-octant chamber block: partition postings by the term's 5-bit chamber, FOR-pack
       each chamber block's doc gaps separately (the glass-box "hierarchy of the top node" applied
       to the DI stream directly) -- measures if chamber locality shrinks doc gaps.
The chamber-residual best point must beat (1) at equal accuracy to justify the hierarchy.
"""
import os, sys, glob, math
os.environ.setdefault("WORK", r"C:\Users\wynos\trng\marco_data\splade_native")
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import numpy as np
WORK = os.environ["WORK"]; sys.path.insert(0, r"C:\Users\wynos\New folder (3)")
import marco_splade_native as m
from collections import defaultdict
from _dd_arch_chamber_residual import (load_docs, baseline_footprint, load_queries, eval_reps,
                                       requant, term_chamber)
import pickle


def main():
    doc_ids, rows, contig = load_docs(50000)
    nd = len(doc_ids); nc = int(contig.sum())
    base = baseline_footprint(doc_ids, rows, contig)
    print(f"baseline B/doc={base['B_per_doc']:.1f} (di {base['di_b']/nc:.1f} + wt {base['wt_b']/nc:.1f})", flush=True)

    queries, qrels = load_queries()
    with open(os.path.join(WORK, "_dd_qenc_cache.pkl"), "rb") as f:
        qenc = pickle.load(f)
    dec_t0 = [r[0].astype(np.int64) for r in rows]
    dec_w0 = [r[1].astype(np.float32) for r in rows]
    bmrr, brec, _ = eval_reps(doc_ids, dec_t0, dec_w0, queries, qenc, qrels)
    print(f"baseline MRR@10={bmrr:.4f} rec@100={brec:.2f}%", flush=True)

    # contiguous inverted index for di footprint, plus chamber-blocked di
    cset = set(int(p) for p, c in zip(doc_ids, contig) if c)
    by_term = defaultdict(list)
    for pid, (ti, wt), c in zip(doc_ids, rows, contig):
        if not c: continue
        for t in ti.tolist():
            by_term[t].append(pid)
    di_flat = 0; di_cham = 0; npost = 0
    for t, docs in by_term.items():
        docs = np.array(sorted(docs), np.uint32); npost += len(docs)
        _, _, _, packed = m._for_pack_gaps(docs); di_flat += packed.nbytes
        # chamber-blocked: split this term's docs by the DOC's dominant-term chamber, pack each block
        # (here we use the TERM's chamber as the block label is constant per term -> no split;
        #  instead block by doc chamber). Build doc->chamber once.
    # doc chamber from dominant term
    dom = np.zeros(nd, np.int64)
    for i, (ti, wt) in enumerate(rows):
        dom[i] = int(ti[int(np.argmax(wt))]) if len(ti) else 0
    cham = np.array([term_chamber(t) for t in dom.tolist()], np.int32)
    cham_of = {int(p): int(cham[i]) for i, p in enumerate(doc_ids)}
    # chamber-blocked di: within each term, group docs by chamber, FOR-pack each group
    for t, docs in by_term.items():
        docs = np.array(sorted(docs), np.uint32)
        ch = np.array([cham_of[int(d)] for d in docs])
        for c in np.unique(ch):
            blk = docs[ch == c]
            _, _, _, packed = m._for_pack_gaps(blk); di_cham += packed.nbytes
        di_cham += 32  # ~ per-term chamber block offsets overhead (tiny)
    print(f"di flat   B/doc = {di_flat/nc:.1f}", flush=True)
    print(f"di chamber-blocked B/doc = {di_cham/nc:.1f}   (hierarchy on the DI stream)", flush=True)

    print("\n  weight-requant-ONLY frontier (no chambers, no base):", flush=True)
    print(f"  {'wbits':>6}{'B/doc':>9}{'shrink':>8}{'MRR@10':>9}{'ret%':>7}", flush=True)
    print(f"  {'base':>6}{base['B_per_doc']:>9.1f}{1.0:>8.2f}{bmrr:>9.4f}{100.0:>7.1f}", flush=True)
    for wb in [8, 4, 3, 2]:
        dec_t = dec_t0
        dec_w = [requant(r[1], wb) for r in rows]
        mrr, rec, _ = eval_reps(doc_ids, dec_t, dec_w, queries, qenc, qrels)
        Bdoc = (di_flat + base['n_post'] * wb / 8) / nc
        print(f"  {wb:>6}{Bdoc:>9.1f}{base['B_per_doc']/Bdoc:>8.2f}{mrr:>9.4f}{100*mrr/bmrr:>7.1f}", flush=True)
        # also chamber-di + requant weights
        Bdoc_c = (di_cham + base['n_post'] * wb / 8) / nc
        print(f"  {('+cham'):>6}{Bdoc_c:>9.1f}{base['B_per_doc']/Bdoc_c:>8.2f}{mrr:>9.4f}{100*mrr/bmrr:>7.1f}", flush=True)


if __name__ == "__main__":
    main()
