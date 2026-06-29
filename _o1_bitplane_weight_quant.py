#!/usr/bin/env python3
r"""
_o1_bitplane_weight_quant.py -- LENS: sub-4-bit SPLADE weight quantization on the lattice RAG.

The doc-side SPLADE weights (si.tloc[j][1], stored float32, but they are INTEGER-valued
quantized weights in [1..86], H(w)=4.984 b) are ~half the footprint. The proven codec floors
weights at 5-bit. This lens pushes BELOW 5-bit: 4 / 3 / 2 bit, each with three quantizer families:
  (a) UNIFORM        -- evenly spaced levels over [min,max]
  (b) LLOYD-MAX      -- k-means / density-optimal non-uniform levels (global, MSE-optimal 1-D)
  (c) PER-TERM-MAX   -- per-term-max rescale then uniform-quantize the normalized weight
                        (so each posting list uses its full dynamic range; +1 small float/term)

For EACH (bitrate, family) we DEQUANTIZE the weights, run the EXACT full-scatter rerank
(si.search-style: acc[loc] += q_w * d_w, np.unique, argpartition) over ALL 6980 cached queries
(NO GPU; real qweight = uint8*QSCALE) and measure MRR@10 / recall@100 vs the fp32 baseline.

Goal: smallest bitrate that holds MRR within ~0.005 of baseline. Does 3-bit non-uniform hold?
Report weight-B/doc savings vs 5-bit and the new TOTAL (gamma doc-ids ~122 B/doc + weights).

Footprint is bit-accounted: weight stream = ceil(n_post * bits / 8) + codebook + (per-term scale
floats for family c).  Latency is reported for the rerank but the lens is footprint+accuracy.
"""
import os, sys, time, math, json
os.environ["WORK"] = r"C:\Users\wynos\trng\marco_data\splade_native"
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import numpy as np
import pickle
from pathlib import Path
from collections import defaultdict
import marco_splade_native as m

HERE = os.path.dirname(os.path.abspath(__file__))
GAMMA_DOCID_BDOC = 122.0          # proven gamma doc-id footprint (from bitpack_v2: 118.95 .. 122)
t0 = time.perf_counter()


def main():
    si = m.ServedIndex()
    n_docs = si.n_docs
    nT = len(si.tloc)
    # flatten postings once
    lens = np.array([len(loc) for loc, _ in si.tloc], np.int64)
    toffs = np.zeros(nT + 1, np.int64); toffs[1:] = np.cumsum(lens)
    n_post = int(toffs[-1])
    all_docs = np.empty(n_post, np.int32)
    all_wf = np.empty(n_post, np.float32)       # original (fp32-stored integer) weights
    term_of = np.empty(n_post, np.int32)        # which term each posting belongs to
    for j, (loc, w) in enumerate(si.tloc):
        s, e = toffs[j], toffs[j + 1]
        all_docs[s:e] = loc.astype(np.int32)
        all_wf[s:e] = w
        term_of[s:e] = j
    print(f"[load] {n_post:,} postings, {nT:,} terms, {n_docs:,} docs ({time.perf_counter()-t0:.1f}s)", flush=True)

    # ---- weight stats / entropy floor ----
    uw, cw = np.unique(all_wf, return_counts=True)
    pw = cw / cw.sum(); H_w = float(-(pw * np.log2(pw)).sum())
    wmin, wmax = float(all_wf.min()), float(all_wf.max())
    print(f"[wt] {len(uw)} distinct levels in [{wmin:.0f},{wmax:.0f}]  H(w)={H_w:.3f} b  mean={all_wf.mean():.2f}", flush=True)

    # ---- queries (cached, NO GPU) ----
    MARCO = m.MARCO
    qrels = defaultdict(set)
    with open(MARCO / "qrels.dev.small.tsv", encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) >= 4 and int(p[3]) > 0:
                qrels[p[0]].add(int(p[2]))
    raw = pickle.load(open(Path(os.environ["WORK"]) / "_dd_qenc_cache.pkl", "rb"))
    col = si.col
    # encode to (col indices into flat term arrays, real qweight)
    qenc = {}
    for qid, (ids, w) in raw.items():
        if qid not in qrels:
            continue
        cols = []; cws = []
        for t, ww in zip(ids.tolist(), w.tolist()):
            j = col.get(int(t))
            if j is not None:
                cols.append(j); cws.append(float(ww) * m.QSCALE)
        if cols:
            qenc[qid] = (np.array(cols, np.int64), np.array(cws, np.float32))
    queries = [q for q in qenc if qrels[q]]
    print(f"[q] {len(queries):,} cached queries answerable (NO GPU)", flush=True)

    present = si.present
    acc = np.zeros(n_docs, np.float32)

    def evaluate(wf_dequant, k=100, time_it=False):
        """Exact full scatter with the supplied per-posting dequantized weights."""
        mrr = 0.0; rec = 0; lat = []
        for qid in queries:
            cols, qws = qenc[qid]
            tt = time.perf_counter() if time_it else 0.0
            touched = []
            for i in range(len(cols)):
                j = int(cols[i]); s, e = toffs[j], toffs[j + 1]
                loc = all_docs[s:e]
                acc[loc] += qws[i] * wf_dequant[s:e]
                touched.append(loc)
            cand = np.unique(np.concatenate(touched))
            sc = acc[cand]; acc[cand] = 0.0
            if len(cand) > k:
                sel = np.argpartition(-sc, k)[:k]
            else:
                sel = np.arange(len(cand))
            order = sel[np.argsort(-sc[sel])]
            ids = present[cand[order].astype(np.int64)]
            if time_it:
                lat.append((time.perf_counter() - tt) * 1000)
            top = [int(d) for d in ids[:100]]; gold = qrels[qid]
            if any(d in gold for d in top):
                rec += 1
            for r, d in enumerate(top[:10]):
                if d in gold:
                    mrr += 1.0 / (r + 1); break
        n = len(queries)
        med = float(np.median(lat)) if lat else 0.0
        return mrr / n, rec / n * 100.0, med

    # ================= QUANTIZERS =================
    def uniform_levels(bits):
        L = 1 << bits
        # L levels evenly spanning [wmin, wmax] (reconstruction points)
        return np.linspace(wmin, wmax, L).astype(np.float32)

    def lloyd_max_levels(bits, iters=40):
        """1-D k-means (Lloyd-Max) on the empirical weight distribution -> MSE-optimal non-uniform levels.
        Weighted by frequency so dense low region gets more levels."""
        L = 1 << bits
        vals = uw.astype(np.float64); freq = cw.astype(np.float64)
        # init levels at frequency-quantiles (good start for skewed data)
        cdf = np.cumsum(freq) / freq.sum()
        qs = np.linspace(0, 1, L + 2)[1:-1]
        cents = np.interp(qs, cdf, vals)
        cents = np.unique(cents)
        if len(cents) < L:
            cents = np.unique(np.concatenate([cents, np.linspace(wmin, wmax, L)]))[:L]
        cents = cents.astype(np.float64)
        for _ in range(iters):
            edges = 0.5 * (cents[:-1] + cents[1:])
            idx = np.searchsorted(edges, vals)
            new = cents.copy()
            for i in range(len(cents)):
                sel = idx == i
                if freq[sel].sum() > 0:
                    new[i] = (vals[sel] * freq[sel]).sum() / freq[sel].sum()
            if np.allclose(new, cents):
                cents = new; break
            cents = new
        return np.sort(cents).astype(np.float32)

    def quantize_to(levels):
        """Map each weight to nearest level value; return dequantized per-posting weights."""
        edges = 0.5 * (levels[:-1] + levels[1:])
        idx = np.searchsorted(edges, all_wf)
        return levels[idx]

    def per_term_max_dequant(bits):
        """Per-term-max scaling: normalize each posting's weight by its term-max into [0,1],
        uniform-quantize to L levels, then rescale by the term-max. Each term stores 1 float scale.
        Reconstruction uses level CENTERS for lower bias."""
        L = 1 << bits
        # term max via reduceat over flat arrays
        tmax = np.zeros(nT, np.float32)
        for j in range(nT):
            s, e = toffs[j], toffs[j + 1]
            if e > s:
                tmax[j] = all_wf[s:e].max()
        tmax_post = tmax[term_of]
        norm = np.where(tmax_post > 0, all_wf / tmax_post, 0.0).astype(np.float32)   # in (0,1]
        # uniform mid-rise quantizer on [0,1]: bin centers at (i+0.5)/L
        idx = np.clip((norm * L).astype(np.int64), 0, L - 1)
        cent = (idx + 0.5) / L
        return (cent * tmax_post).astype(np.float32), nT  # dequant weights, #term scales

    # ================= BASELINE (fp32-stored integer weights) =================
    base_mrr, base_rec, base_lat = evaluate(all_wf, time_it=True)
    wt_bytes_fp32 = n_post * 4
    di_bdoc = GAMMA_DOCID_BDOC
    base_wt_bdoc = wt_bytes_fp32 / n_docs
    base_total = di_bdoc + base_wt_bdoc
    print(f"\n[BASELINE fp32-stored] MRR@10={base_mrr:.4f} rec@100={base_rec:.2f}% {base_lat:.1f}ms/q"
          f"  wt={base_wt_bdoc:.1f} B/doc  total(+gamma di)={base_total:.1f} B/doc", flush=True)

    # 5-bit reference (the proven floor) -- use Lloyd-Max as the 'proven 5b' point
    lv5 = lloyd_max_levels(5)
    mrr5, rec5, _ = evaluate(quantize_to(lv5))
    wt5_bytes = math.ceil(n_post * 5 / 8) + 5 * 4  # codebook negligible (32 levels * 4B)
    wt5_bdoc = wt5_bytes / n_docs
    print(f"[5-bit ref Lloyd]      MRR@10={mrr5:.4f} ({mrr5-base_mrr:+.4f}) rec={rec5:.2f}%"
          f"  wt={wt5_bdoc:.1f} B/doc  total={di_bdoc+wt5_bdoc:.1f} B/doc", flush=True)

    # ================= SWEEP =================
    print(f"\n[SWEEP]  baseline MRR@10={base_mrr:.4f}  (target hold = within 0.005)", flush=True)
    print(f"  {'bits':>4} {'family':<14}{'MRR@10':>9}{'dMRR':>9}{'rec@100':>9}{'wt B/doc':>10}{'total B/doc':>12}{' vs5b':>7}", flush=True)
    results = []
    for bits in [4, 3, 2]:
        for fam in ["uniform", "lloyd", "perterm"]:
            if fam == "uniform":
                lv = uniform_levels(bits)
                deq = quantize_to(lv)
                extra_codebook = (1 << bits) * 4
                extra_scales = 0
            elif fam == "lloyd":
                lv = lloyd_max_levels(bits)
                deq = quantize_to(lv)
                extra_codebook = (1 << bits) * 4
                extra_scales = 0
            else:  # perterm
                deq, n_scales = per_term_max_dequant(bits)
                extra_codebook = 0
                extra_scales = n_scales * 2          # store term-max as float16 (2 B/term)
            mrr, rec, _ = evaluate(deq)
            wt_bytes = math.ceil(n_post * bits / 8) + extra_codebook + extra_scales
            wt_bdoc = wt_bytes / n_docs
            total = di_bdoc + wt_bdoc
            dmrr = mrr - base_mrr
            holds = abs(dmrr) <= 0.005
            vs5 = wt5_bdoc / wt_bdoc
            tag = "  HOLD" if holds else ""
            print(f"  {bits:>4} {fam:<14}{mrr:>9.4f}{dmrr:>+9.4f}{rec:>9.2f}{wt_bdoc:>10.1f}{total:>12.1f}{vs5:>6.2f}x{tag}", flush=True)
            results.append(dict(bits=bits, family=fam, mrr=mrr, dmrr=dmrr, rec=rec,
                                wt_bdoc=wt_bdoc, total_bdoc=total, holds=holds))

    # ================= VERDICT =================
    holders = [r for r in results if r["holds"]]
    # smallest bits among holders, then smallest wt_bdoc
    holders.sort(key=lambda r: (r["bits"], r["wt_bdoc"]))
    print("\n" + "=" * 92, flush=True)
    print(f"BASELINE  MRR@10 {base_mrr:.4f}  wt {base_wt_bdoc:.1f} B/doc  total {base_total:.1f} B/doc", flush=True)
    print(f"5-bit ref MRR@10 {mrr5:.4f}  wt {wt5_bdoc:.1f} B/doc  total {di_bdoc+wt5_bdoc:.1f} B/doc", flush=True)
    best = None
    if holders:
        best = holders[0]
        sav = wt5_bdoc - best["wt_bdoc"]
        print(f"SMALLEST BITRATE HOLDING (<=0.005):  {best['bits']}-bit {best['family']}"
              f"  MRR {best['mrr']:.4f} ({best['dmrr']:+.4f})  wt {best['wt_bdoc']:.1f} B/doc"
              f"  total {best['total_bdoc']:.1f} B/doc  (saves {sav:.1f} B/doc vs 5-bit, {wt5_bdoc/best['wt_bdoc']:.2f}x wt-shrink)", flush=True)
    else:
        print("NO sub-5-bit family holds within 0.005 of baseline.", flush=True)
    # specifically: does 3-bit non-uniform hold?
    r3 = next((r for r in results if r["bits"] == 3 and r["family"] == "lloyd"), None)
    if r3:
        print(f"3-bit NON-UNIFORM (Lloyd):  MRR {r3['mrr']:.4f} ({r3['dmrr']:+.4f})  "
              f"{'HOLDS' if r3['holds'] else 'FAILS'} the 0.005 bar  -> wt {r3['wt_bdoc']:.1f} B/doc, total {r3['total_bdoc']:.1f} B/doc", flush=True)

    out = dict(baseline=dict(mrr=base_mrr, rec=base_rec, lat_ms=base_lat,
                             wt_bdoc=base_wt_bdoc, total_bdoc=base_total),
               ref5=dict(mrr=mrr5, rec=rec5, wt_bdoc=wt5_bdoc, total_bdoc=di_bdoc + wt5_bdoc),
               H_w=H_w, n_post=n_post, n_docs=n_docs, gamma_docid_bdoc=di_bdoc,
               results=results,
               best_holder=best)
    with open(os.path.join(HERE, "_o1_bitplane_weight_quant_result.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\n[done] {time.perf_counter()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
