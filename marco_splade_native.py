#!/usr/bin/env python3
"""Native SPLADE-on-the-lattice for MS MARCO -- resumable encode, FOR/chamber pack, serve+eval.

This is "native SPLADE": every passage is encoded with the SPLADE doc-encoder into sparse
learned term weights, those weights ARE the lattice postings (term -> {doc: weight}), and
dev-small is served by the sparse dot (the meet) over the inverted index -- NO BM25 candidate
pool, NO cross-encoder. The point is to remove the recall-pool ceiling: every query scores
against the whole collection's SPLADE postings, so the gold doc can always be reached.

Three stages, each independently runnable and RESUMABLE:

  encode  : stream MARCO passages in fp16 batches through SPLADE, threshold + keep top-K terms
            per doc, write each CHUNK (e.g. 200k docs) to disk as a compressed sparse file
            (doc_ids + term_ids uint16 + quantized weights uint8). Chunks already on disk are
            skipped, so an interrupted run resumes for free.
  index   : invert the chunks into term -> postings, FOR-pack the doc-id gaps (reuse
            marco_slim_for's bit-width frame-of-reference) + store weights as uint8, measure
            on-disk footprint. Optionally chamber-pack the gap stream for the cold/archival tier.
  serve   : SPLADE-encode each dev-small query, score by sparse dot over the inverted index via
            the meet, report MRR@10 + recall + query latency.

CALIBRATE mode runs all three end-to-end on a 50,000-passage slice and PROJECTS the full 8.8M
cost (encode hours, on-disk GB, serve latency, OOM risk). It does NOT launch the full run.

  python marco_splade_native.py calibrate                 # the 50k end-to-end calibration
  python marco_splade_native.py encode  --full            # the full 8.8M streaming encode
  python marco_splade_native.py index   --full
  python marco_splade_native.py serve   --full

Env knobs: SPLADE_MODEL, BATCH (256), TOPK (200), CHUNK (200000), MINW (tiny-weight floor),
WORK (output dir, default C:\\Users\\wynos\\trng\\marco_data\\splade_native).
"""
import os, sys, time, json, math, argparse, glob
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")  # Windows-safe HF cache (per the route2 modules)
from pathlib import Path
from collections import defaultdict
import numpy as np

MARCO = Path(r"C:\Users\wynos\trng\marco_data")
WORK = Path(os.environ.get("WORK", str(MARCO / "splade_native")))
WORK.mkdir(parents=True, exist_ok=True)

MODEL = os.environ.get("SPLADE_MODEL", "naver/splade-cocondenser-ensembledistil")
BATCH = int(os.environ.get("BATCH", "256"))
TOPK = int(os.environ.get("TOPK", "200"))          # keep at most this many terms/doc
CHUNK = int(os.environ.get("CHUNK", "200000"))     # docs per on-disk shard
MINW = float(os.environ.get("MINW", "0.0"))        # drop SPLADE weights at/below this (0 = keep all nonzero)
DOC_ML = int(os.environ.get("DOC_ML", "256"))      # SPLADE doc max-len
QUERY_ML = int(os.environ.get("QUERY_ML", "64"))
N_DOCS_FULL = 8_841_823                             # len(collection.offsets.npy)

# SPLADE weights are quantized to uint8 with a fixed scale. SPLADE log1p(relu(logit)) weights
# sit in ~[0, ~6]; 0.04/level covers >5.0 before saturating, which is the standard int8 serving
# range. Score is computed in the same quantized units (q_w*d_w), monotone w.r.t. the float dot.
QSCALE = float(os.environ.get("QSCALE", "0.04"))


# ----------------------------------------------------------------------------------------------
#  light passage-text accessor: only the offsets + collection.tsv, NOT the 2GB full BM25 index
# ----------------------------------------------------------------------------------------------
class Passages:
    def __init__(self):
        self.offsets = np.load(MARCO / "collection.offsets.npy", mmap_mode="r")
        self.cf = open(MARCO / "collection.tsv", "r", encoding="utf-8", errors="replace")
        self.n = len(self.offsets)

    def text(self, pid):
        self.cf.seek(int(self.offsets[pid]))
        line = self.cf.readline()
        tab = line.find("\t")
        return line[tab + 1:].rstrip("\n") if tab >= 0 else ""

    def batch_text(self, pids):
        return [self.text(p) for p in pids]


# ----------------------------------------------------------------------------------------------
#  SPLADE doc/query encoder (fp16, cuda) -- same recipe as _route2_splade_lattice.py
# ----------------------------------------------------------------------------------------------
_tok = None; _mdl = None; _DEVICE = None
def _load_splade():
    global _tok, _mdl, _DEVICE
    if _mdl is None:
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        _tok = AutoTokenizer.from_pretrained(MODEL)
        _mdl = AutoModelForMaskedLM.from_pretrained(MODEL).half().to(_DEVICE).eval()
    return _tok, _mdl, _DEVICE


def splade_sparse(texts, max_len, topk=TOPK, minw=MINW):
    """Encode a batch -> list of (term_ids uint16, qweights uint8). Top-K + tiny-weight pruned."""
    import torch
    tok, mdl, dev = _load_splade()
    texts = [(t if (t and any(c.isalnum() for c in t)) else "unk") for t in texts]
    with torch.no_grad():
        enc = tok(texts, return_tensors="pt", truncation=True, max_length=max_len, padding=True).to(dev)
        logits = mdl(**enc).logits
        rep = torch.max(torch.log1p(torch.relu(logits)) * enc.attention_mask.unsqueeze(-1), dim=1).values
        out = []
        rep = rep.float()
        for r in rep:
            nz = torch.nonzero(r > minw, as_tuple=False).squeeze(-1)
            if nz.numel() > topk:
                vals = r[nz]
                keep = torch.topk(vals, topk).indices
                nz = nz[keep]
            ids = nz.to("cpu").numpy().astype(np.uint16)
            w = r[nz].to("cpu").numpy()
            qw = np.clip(np.round(w / QSCALE), 1, 255).astype(np.uint8)
            order = np.argsort(ids)               # sorted term ids -> stable, mergeable
            out.append((ids[order], qw[order]))
    return out


# ----------------------------------------------------------------------------------------------
#  STAGE 1 -- resumable streaming encoder
# ----------------------------------------------------------------------------------------------
def chunk_path(ci):
    return WORK / f"chunk_{ci:05d}.npz"


def _encode_pids(pas, pids, cp, t_start, done_docs0, end_for_eta, tag, label):
    """Encode an explicit list of pids into one chunk file cp (atomic). Returns docs encoded."""
    if cp.exists():
        try:
            z = np.load(cp); nd = int(z["n_docs"]); z.close()
            print(f"    chunk {label} already on disk -- skip", flush=True)
            return nd
        except Exception:
            pass
    ptr = [0]; term_ids = []; weights = []; doc_ids = []
    t_c = time.perf_counter(); done = 0
    for b0 in range(0, len(pids), BATCH):
        bp = pids[b0:b0 + BATCH]
        reps = splade_sparse(pas.batch_text(bp), DOC_ML)
        for pid, (ids, qw) in zip(bp, reps):
            doc_ids.append(pid); term_ids.append(ids); weights.append(qw)
            ptr.append(ptr[-1] + len(ids))
        done += len(bp)
        if b0 > 0 and b0 % (BATCH * 40) == 0:
            el = time.perf_counter() - t_start
            dps = (done_docs0 + done) / max(1e-9, el)
            eta = (end_for_eta - (done_docs0 + done)) / dps / 3600.0
            print(f"    {done_docs0+done:,}/{end_for_eta:,}  {dps:,.0f} docs/s  "
                  f"chunk {label} {done:,}/{len(pids):,}  ETA-full {eta:.1f}h", flush=True)
    di = np.asarray(doc_ids, dtype=np.uint32)
    ti = np.concatenate(term_ids).astype(np.uint16) if term_ids else np.zeros(0, np.uint16)
    wt = np.concatenate(weights).astype(np.uint8) if weights else np.zeros(0, np.uint8)
    pa = np.asarray(ptr, dtype=np.uint64)
    tmp = cp.with_name(cp.stem + ".tmp.npz")     # np.savez force-appends ".npz"
    np.savez(tmp, doc_ids=di, term_ids=ti, weights=wt, ptr=pa,
             n_docs=np.int64(len(di)), n_post=np.int64(len(ti)))
    os.replace(tmp, cp)                           # atomic -> a half-written chunk never looks "done"
    ppd = len(ti) / max(1, len(di))
    print(f"    wrote chunk {label}: {len(di):,} docs, {len(ti):,} postings "
          f"({ppd:.1f}/doc), {cp.stat().st_size/1e6:.1f} MB raw-shard, "
          f"{time.perf_counter()-t_c:.0f}s", flush=True)
    return len(di)


def encode(n_docs, start=0, tag="", extra_pids=None):
    """Stream docs [start, start+n_docs) through SPLADE, write CHUNK-sized shards, skip done shards.
    extra_pids: optional explicit pids appended as one extra chunk (calibration gold injection)."""
    pas = Passages()
    end = min(start + n_docs, pas.n)
    print(f"  [encode{tag}] docs [{start:,}, {end:,})  batch={BATCH} topk={TOPK} model={MODEL}", flush=True)
    t_start = time.perf_counter()
    done_docs = 0
    ci0 = start // CHUNK
    ci1 = (end - 1) // CHUNK
    for ci in range(ci0, ci1 + 1):
        cs = max(start, ci * CHUNK)
        ce = min(end, (ci + 1) * CHUNK)
        cp = chunk_path(ci)
        if cp.exists():
            try:
                z = np.load(cp)
                done_docs += int(z["n_docs"]); z.close()
                print(f"    chunk {ci:05d} [{cs:,},{ce:,}) already on disk -- skip", flush=True)
                continue
            except Exception:
                pass  # corrupt/partial -> re-encode
        # encode this chunk
        ptr = [0]; term_ids = []; weights = []; doc_ids = []
        t_c = time.perf_counter()
        for b0 in range(cs, ce, BATCH):
            b1 = min(ce, b0 + BATCH)
            pids = list(range(b0, b1))
            texts = pas.batch_text(pids)
            reps = splade_sparse(texts, DOC_ML)
            for pid, (ids, qw) in zip(pids, reps):
                doc_ids.append(pid)
                term_ids.append(ids); weights.append(qw)
                ptr.append(ptr[-1] + len(ids))
            done_docs += (b1 - b0)
            if (b0 - cs) % (BATCH * 40) == 0 and b0 > cs:
                el = time.perf_counter() - t_start
                dps = done_docs / el
                eta_full = (N_DOCS_FULL - (start + done_docs)) / dps / 3600.0
                print(f"    {start+done_docs:,}/{end:,}  {dps:,.0f} docs/s  "
                      f"chunk {ci:05d} {b1-cs:,}/{ce-cs:,}  ETA-full {eta_full:.1f}h", flush=True)
        di = np.asarray(doc_ids, dtype=np.uint32)
        ti = np.concatenate(term_ids).astype(np.uint16) if term_ids else np.zeros(0, np.uint16)
        wt = np.concatenate(weights).astype(np.uint8) if weights else np.zeros(0, np.uint8)
        pa = np.asarray(ptr, dtype=np.uint64)
        # np.savez force-appends ".npz"; write to a sibling .tmp.npz then atomically rename.
        tmp = cp.with_name(cp.stem + ".tmp.npz")
        np.savez(tmp, doc_ids=di, term_ids=ti, weights=wt, ptr=pa,
                 n_docs=np.int64(len(di)), n_post=np.int64(len(ti)))
        os.replace(tmp, cp)               # atomic -> a half-written chunk never looks "done"
        ppd = len(ti) / max(1, len(di))
        print(f"    wrote chunk {ci:05d}: {len(di):,} docs, {len(ti):,} postings "
              f"({ppd:.1f}/doc), {cp.stat().st_size/1e6:.1f} MB raw-shard, "
              f"{time.perf_counter()-t_c:.0f}s", flush=True)
    el = time.perf_counter() - t_start
    dps = done_docs / max(1e-9, el)   # throughput measured on the CONTIGUOUS slice only
    if extra_pids:
        # gold-injection chunk (calibration only) -- gives a real ranking signal on a 50k slice.
        gp = sorted(set(int(p) for p in extra_pids) - set(range(start, end)))
        gp = [p for p in gp if 0 <= p < pas.n]
        if gp:
            cp = WORK / "chunk_gold.npz"
            _encode_pids(pas, gp, cp, time.perf_counter(), 0, len(gp), tag, "gold")
    print(f"  [encode{tag}] DONE {done_docs:,} contiguous docs in {el:.0f}s = {dps:,.0f} docs/s", flush=True)
    return done_docs, dps, el


# ----------------------------------------------------------------------------------------------
#  STAGE 2 -- invert chunks -> term postings, FOR-pack the doc-id gaps + uint8 weights
# ----------------------------------------------------------------------------------------------
def _for_pack_gaps(sorted_docs):
    """Bit-width Frame-of-Reference pack of ascending doc ids (== marco_slim_for codec).
    Returns (first uint32, n, width uint8, packed bytes)."""
    n = len(sorted_docs)
    first = np.uint32(sorted_docs[0])
    if n <= 1:
        return first, n, np.uint8(1), np.zeros(0, np.uint8)
    d = np.diff(sorted_docs.astype(np.int64))
    w = max(1, int(int(d.max()).bit_length()))
    bits = ((d.astype(np.uint32)[:, None] >> np.arange(w - 1, -1, -1)) & 1).astype(np.uint8)
    packed = np.packbits(bits.reshape(-1))
    return first, n, np.uint8(w), packed


def _for_unpack_gaps(first, n, w, packed):
    di = np.empty(n, np.uint32); di[0] = first
    if n > 1:
        bits = np.unpackbits(packed)[:(n - 1) * w].reshape(n - 1, w)
        vals = bits.dot((1 << np.arange(w - 1, -1, -1)).astype(np.uint32))
        di[1:] = first + np.cumsum(vals.astype(np.int64))
    return di


VOCAB = 30522   # SPLADE/bert-base vocab; term ids fit uint16


def index(tag="", chamber=False, proj_max_pid=None, shards=None):
    """Invert all chunks into term postings, FOR-pack, write the served index + footprint report.

    Memory-bounded by VOCAB-SHARDING: terms are split into `shards` contiguous id-groups; each
    pass gathers only that group's postings into numpy arrays (peak RAM ~= n_post/shards * 5B),
    so the full 1.07B-posting inversion never materialises 1B Python ints. `shards` defaults to a
    value that keeps each group's working set under ~2 GB. All-numpy, no per-posting Python loop.
    """
    chunks = sorted(WORK.glob("chunk_*.npz"))
    if not chunks:
        print("  [index] no chunks found -- run encode first", flush=True); return None
    print(f"  [index{tag}] inverting {len(chunks)} chunks", flush=True)
    t0 = time.perf_counter()

    # ---- pass 1: per-term posting counts + totals (cheap, scans uint16 term-id arrays) ----
    term_count = np.zeros(VOCAB, np.int64)
    n_docs = 0; n_post = 0
    for cp in chunks:
        z = np.load(cp)
        ti = z["term_ids"]
        n_docs += int(z["n_docs"]); n_post += len(ti)
        if len(ti):
            term_count += np.bincount(ti.astype(np.int64), minlength=VOCAB)
        z.close()
    active = np.where(term_count > 0)[0]
    if shards is None:
        # keep each shard's gather under ~300M postings (~1.5 GB at 5 B/posting)
        shards = max(1, int(math.ceil(n_post / 300_000_000)))
    print(f"    pass1: {n_post:,} postings over {n_docs:,} docs, {len(active):,} active terms "
          f"-> {shards} vocab-shard(s) ({time.perf_counter()-t0:.0f}s)", flush=True)

    # ---- pass 2 per vocab-shard: gather (doc, weight) into numpy, sort per term, FOR-pack ----
    terms = active.tolist()
    nT = len(terms)
    first = np.zeros(nT, np.uint32); nn = np.zeros(nT, np.uint32); width = np.zeros(nT, np.uint8)
    toff = np.zeros(nT + 1, np.uint64); poff = np.zeros(nT + 1, np.uint64)
    tid_map = np.asarray(terms, np.uint16)
    blob_parts = []; wt_parts = []; byte_cur = 0
    bounds = np.linspace(0, VOCAB, shards + 1).astype(np.int64)
    term_pos = {t: j for j, t in enumerate(terms)}    # term-id -> output column
    ref_sample = {}                                   # keep a few raw lists for round-trip check
    for sh in range(shards):
        lo, hi = int(bounds[sh]), int(bounds[sh + 1])
        sh_terms = [t for t in terms if lo <= t < hi]
        if not sh_terms:
            continue
        npost_sh = int(term_count[lo:hi].sum())
        # preallocate this shard's gather buffers
        g_term = np.empty(npost_sh, np.uint16)
        g_doc = np.empty(npost_sh, np.uint32)
        g_wt = np.empty(npost_sh, np.uint8)
        fill = 0
        for cp in chunks:
            z = np.load(cp)
            di = z["doc_ids"]; ti = z["term_ids"]; wt = z["weights"]; pa = z["ptr"]
            sel = (ti >= lo) & (ti < hi)
            if sel.any():
                # expand doc ids: each posting k belongs to the doc whose ptr-range covers it
                doc_of_post = np.repeat(di, np.diff(pa).astype(np.int64))
                m = sel
                cnt = int(m.sum())
                g_term[fill:fill + cnt] = ti[m]
                g_doc[fill:fill + cnt] = doc_of_post[m]
                g_wt[fill:fill + cnt] = wt[m]
                fill += cnt
            z.close()
        # sort by (term, doc) so each term's docs are ascending and contiguous
        order = np.lexsort((g_doc[:fill], g_term[:fill]))
        st = g_term[:fill][order]; sd = g_doc[:fill][order]; sw = g_wt[:fill][order]
        # term boundaries
        uniq, starts = np.unique(st, return_index=True)
        ends = np.append(starts[1:], len(st))
        # shards are id-ordered and `uniq` is sorted -> we encounter columns j in strictly
        # increasing order, so appending here builds blob/weights already in column order.
        for t, s, e in zip(uniq.tolist(), starts.tolist(), ends.tolist()):
            j = term_pos[t]
            docs = sd[s:e]; ws = sw[s:e]
            f, n, w, packed = _for_pack_gaps(docs)
            first[j] = f; nn[j] = n; width[j] = w
            toff[j] = byte_cur; byte_cur += packed.nbytes
            poff[j + 1] = poff[j] + n
            blob_parts.append(packed); wt_parts.append(ws)
            if len(ref_sample) < 20 and (s % 7 == 0):
                ref_sample[int(t)] = docs.copy()
        del g_term, g_doc, g_wt, st, sd, sw, order
        print(f"    shard {sh+1}/{shards} terms[{lo},{hi}) {npost_sh:,} postings packed "
              f"({time.perf_counter()-t0:.0f}s)", flush=True)
    blob = np.concatenate(blob_parts) if blob_parts else np.zeros(0, np.uint8)
    wts = np.concatenate(wt_parts) if wt_parts else np.zeros(0, np.uint8)
    toff[nT] = byte_cur

    out = WORK / "splade_index_for.npz"
    np.savez(out, first=first, nn=nn, width=width, toff=toff, poff=poff,
             blob=blob, weights=wts, term_ids=tid_map,
             n_docs=np.int64(n_docs), n_post=np.int64(n_post))
    on_disk = out.stat().st_size
    sizes = {"di_FOR": blob.nbytes, "weights_u8": wts.nbytes,
             "first/n/width": first.nbytes + nn.nbytes + width.nbytes,
             "offsets": toff.nbytes + poff.nbytes, "term_ids": tid_map.nbytes}
    print(f"  [index{tag}] FOR sizes:", flush=True)
    for k, v in sizes.items():
        print(f"      {k:<16}{v/1e6:>9.2f} MB", flush=True)
    bpp = blob.nbytes * 8 / max(1, n_post)
    print(f"    postings={n_post:,}  on-disk={on_disk/1e6:.1f} MB  "
          f"({on_disk/max(1,n_docs):.1f} B/doc, di-gap {bpp:.2f} bits/posting)", flush=True)

    # round-trip a few terms: FOR decode must reproduce the sorted doc list (ref_sample captured
    # the raw ascending docs for ~20 terms during packing)
    ok = True
    for t, ref in ref_sample.items():
        j = term_pos[t]
        f, n, w = int(first[j]), int(nn[j]), int(width[j])
        o0, o1 = int(toff[j]), int(toff[j + 1])
        dec = _for_unpack_gaps(f, n, w, blob[o0:o1])
        if not np.array_equal(dec, np.sort(ref)):
            ok = False; break
    print(f"    FOR round-trip on {len(ref_sample)} sampled terms: {'MATCH' if ok else 'MISMATCH'}", flush=True)

    cham_bits = None
    if chamber:
        cham_bits = _chamber_probe(blob)

    # contiguous-only footprint for an HONEST projection (the injected scattered gold chunk
    # makes huge FOR gaps that do NOT occur in the dense full run, where every pid is present).
    # Re-scan chunks, keep only docs < proj_max_pid -- this region is small so it fits in numpy.
    proj = None
    if proj_max_pid is not None:
        cterm = []; cdoc = []
        for cp in chunks:
            z = np.load(cp)
            di = z["doc_ids"]; ti = z["term_ids"]; pa = z["ptr"]
            doc_of_post = np.repeat(di, np.diff(pa).astype(np.int64))
            m = doc_of_post < proj_max_pid
            if m.any():
                cterm.append(ti[m]); cdoc.append(doc_of_post[m])
            z.close()
        ct = np.concatenate(cterm) if cterm else np.zeros(0, np.uint16)
        cd = np.concatenate(cdoc) if cdoc else np.zeros(0, np.uint32)
        di_b = 0; wt_b = 0; n_p = len(ct)
        if n_p:
            order = np.lexsort((cd, ct)); ct = ct[order]; cd = cd[order]
            uniq, starts = np.unique(ct, return_index=True)
            ends = np.append(starts[1:], len(ct))
            for s, e in zip(starts.tolist(), ends.tolist()):
                _, _, _, packed = _for_pack_gaps(cd[s:e])
                di_b += packed.nbytes; wt_b += (e - s)
        proj = dict(di_bytes=di_b, wt_bytes=wt_b, n_docs=int(proj_max_pid), n_post=n_p)
        print(f"    contiguous-only [<{proj_max_pid:,}]: {n_p:,} postings over {proj_max_pid:,} docs, "
              f"di {di_b/1e6:.1f}MB ({di_b*8/max(1,n_p):.2f} b/posting), wt {wt_b/1e6:.1f}MB", flush=True)

    return dict(n_docs=n_docs, n_post=n_post, on_disk=on_disk, sizes=sizes,
                di_bytes=blob.nbytes, wt_bytes=wts.nbytes, roundtrip=ok, cham_bits=cham_bits,
                proj=proj)


def _chamber_probe(blob):
    """Run the native chamber codec on the FOR gap-blob to estimate the cold/archival ratio.
    (The chamber is sequential + slow-decode -- archival tier only, per the repo's headline.)"""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
        from test_chamber_mixer_v5_native import compress as cham_compress
        sample = blob[: min(len(blob), 4_000_000)].tobytes()
        if not sample:
            return None
        t0 = time.perf_counter()
        cblob, _ = cham_compress(sample)
        dt = time.perf_counter() - t0
        ratio = len(sample) / len(cblob)
        bps = len(sample) / 1e6 / dt
        print(f"    [chamber probe] {len(sample)/1e6:.1f}MB gap-blob -> {len(cblob)/1e6:.2f}MB "
              f"({len(cblob)*8/len(sample):.3f} bits/byte, {ratio:.2f}x, {bps:.2f} MB/s encode)", flush=True)
        return dict(ratio=ratio, bits_per_byte=len(cblob) * 8 / len(sample), enc_mb_s=bps)
    except Exception as e:
        print(f"    [chamber probe] skipped: {type(e).__name__}: {str(e)[:80]}", flush=True)
        return None


# ----------------------------------------------------------------------------------------------
#  STAGE 3 -- serve + eval: SPLADE-encode queries, sparse-dot meet over the FOR index
# ----------------------------------------------------------------------------------------------
class ServedIndex:
    """Load the FOR index, decode all postings once into per-term (docs, weights) arrays.
    Score = sum over query terms of q_w * d_w (the meet) -- no BM25 pool, no CE.

    Full-run RAM: decoding all 1.07B postings = ~4.3 GB docs (uint32) + ~4.3 GB weights (f32) +
    an 8.84M int64 pid->local map (~70 MB) ~= 9 GB, well within this 51 GB box. Per query the meet
    unions ONLY the query terms' posting lists (np.unique over touched), so cost is DF-bound, not
    O(8.8M). If RAM is tight at full scale, decode postings lazily per query term instead."""
    def __init__(self):
        z = np.load(WORK / "splade_index_for.npz")
        self.first = z["first"]; self.nn = z["nn"]; self.width = z["width"]
        self.toff = z["toff"]; self.poff = z["poff"]; self.blob = z["blob"]
        self.weights = z["weights"]; self.term_ids = z["term_ids"]
        self.n_docs = int(z["n_docs"]); self.n_post = int(z["n_post"])
        # tid -> column index into the term arrays
        self.col = {int(t): j for j, t in enumerate(self.term_ids)}
        # decode postings for every term up-front (calibration scale; for full, decode lazily)
        self.tdocs = [None] * len(self.term_ids)
        for j in range(len(self.term_ids)):
            f, n, w = int(self.first[j]), int(self.nn[j]), int(self.width[j])
            o0, o1 = int(self.toff[j]), int(self.toff[j + 1])
            docs = _for_unpack_gaps(f, n, w, self.blob[o0:o1])
            p0 = int(self.poff[j])
            self.tdocs[j] = (docs, self.weights[p0:p0 + n].astype(np.float32))
        # contiguous remap of doc ids actually present -> dense accumulator
        present = np.unique(np.concatenate([d for d, _ in self.tdocs])) if self.tdocs else np.zeros(0, np.uint32)
        self.present = present
        self.remap = {int(d): i for i, d in enumerate(present)}
        self.local = np.full(int(present.max()) + 1 if len(present) else 1, -1, np.int64)
        for i, d in enumerate(present):
            self.local[int(d)] = i
        # pre-remap each term's docs to local indices
        self.tloc = []
        for docs, w in self.tdocs:
            self.tloc.append((self.local[docs.astype(np.int64)], w))
        self.acc = np.zeros(len(present), np.float32)

    def search(self, qids, qw, k=10):
        acc = self.acc; acc[:] = 0.0
        touched = []
        for tid, qweight in zip(qids, qw):
            j = self.col.get(int(tid))
            if j is None:
                continue
            loc, w = self.tloc[j]
            acc[loc] += float(qweight) * w
            touched.append(loc)
        if not touched:
            return np.zeros(0, np.uint32), np.zeros(0, np.float32)
        cand = np.unique(np.concatenate(touched))
        sc = acc[cand]
        if len(cand) > k:
            sel = np.argpartition(-sc, k)[:k]
        else:
            sel = np.arange(len(cand))
        order = sel[np.argsort(-sc[sel])]
        return self.present[cand[order]], sc[order]

    def search_fast(self, qids, qw, k=10, topq=30, pool_cap=80000):
        """Rarest-address candidate pooling: the SHORT (discriminative) query-term posting lists
        build a small candidate set C; every term then refines scores by searchsorted(C) against
        its sorted posting list -- O(|C|*log|posting|), never O(|posting|). So a million-long
        common-term list costs ~|C| lookups, not a million scatter-adds. Footprint unchanged
        (query-side only); for candidates in C the score is the EXACT sparse-dot. ~35x faster than
        search() at equal MRR@10 (measured: 0.40 @ 92 ms vs 3234 ms; see MEASUREMENTS.md)."""
        qw = np.asarray(qw, np.float32)
        top = np.argsort(-qw)[:topq]
        terms = []
        for i in top:
            j = self.col.get(int(qids[i]))
            if j is None:
                continue
            loc, w = self.tloc[j]
            terms.append((loc, w, float(qw[i])))
        if not terms:
            return np.zeros(0, np.uint32), np.zeros(0, np.float32)
        terms.sort(key=lambda t: len(t[0]))                 # discriminative (short lists) first
        parts = []; tot = 0
        for loc, w, qweight in terms:
            if parts and tot + len(loc) > pool_cap:
                break
            parts.append(loc); tot += len(loc)
        C = np.unique(np.concatenate(parts))                # sorted candidate local ids
        score = np.zeros(len(C), np.float32)
        for loc, w, qweight in terms:
            pos = np.searchsorted(loc, C)
            pc = np.minimum(pos, len(loc) - 1)
            hit = loc[pc] == C
            score[hit] += qweight * w[pc[hit]]
        if len(C) > k:
            sel = np.argpartition(-score, k)[:k]
        else:
            sel = np.arange(len(C))
        order = sel[np.argsort(-score[sel])]
        return self.present[C[order]], score[order]

    def search_corr(self, qids, qw, k=10, topq=30, n_anchor=6):
        """Composite-meet pool: the candidates = docs sharing a CORRELATION with the query, i.e. the
        union of pairwise MEETS (intersections) of the query's most discriminative terms, plus the single
        rarest term as a recall floor. Recovers the FULL-scatter accuracy that the rarest-union heuristic
        loses (measured 0.398 vs 0.391 on the same 250 q = the 3.1s ceiling), at 286.9 B/doc, ~127 ms.
        This is the accuracy-optimal serve; search_fast is faster/slightly-lower; the stored-composite
        layer (composites.npz, build_composites.py) is a further speed dial (down to ~14-45 ms)."""
        qw = np.asarray(qw, np.float32)
        top = np.argsort(-qw)[:topq]
        terms = []
        for i in top:
            j = self.col.get(int(qids[i]))
            if j is None:
                continue
            loc, w = self.tloc[j]
            terms.append((loc, w, float(qw[i])))
        if not terms:
            return np.zeros(0, np.uint32), np.zeros(0, np.float32)
        terms.sort(key=lambda t: len(t[0]))
        anchors = terms[:n_anchor]
        parts = []
        for a in range(len(anchors)):
            la = anchors[a][0]
            for b in range(a + 1, len(anchors)):
                lb = anchors[b][0]
                x, y = (la, lb) if len(la) <= len(lb) else (lb, la)
                pos = np.searchsorted(y, x); pc = np.minimum(pos, len(y) - 1)
                ab = x[y[pc] == x]                      # the meet = composite (correlation) doc-list
                if len(ab):
                    parts.append(ab)
        parts.append(anchors[0][0])                    # rarest single term = recall floor
        C = np.unique(np.concatenate(parts))
        score = np.zeros(len(C), np.float32)
        for loc, w, qweight in terms:
            pos = np.searchsorted(loc, C); pc = np.minimum(pos, len(loc) - 1)
            hit = loc[pc] == C
            score[hit] += qweight * w[pc[hit]]
        sel = np.argpartition(-score, k)[:k] if len(C) > k else np.arange(len(C))
        order = sel[np.argsort(-score[sel])]
        return self.present[C[order]], score[order]


def serve(tag="", nq=None):
    MODE = os.environ.get("SERVE_MODE", "corr")   # corr=composite-meet (best acc) | fast=rarest-union | full=scatter
    print(f"  [serve{tag}] loading FOR index  (serve mode: {MODE})", flush=True)
    t0 = time.perf_counter()
    si = ServedIndex()
    print(f"    loaded {len(si.term_ids):,} terms, {si.n_post:,} postings, "
          f"{len(si.present):,} present docs ({time.perf_counter()-t0:.0f}s)", flush=True)

    qrels = defaultdict(set)
    with open(MARCO / "qrels.dev.small.tsv", encoding="utf-8") as f:
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
    if nq:
        queries = queries[:nq]

    # for a 50k SLICE only a fraction of gold docs are even in the index -> measure the
    # answerable subset honestly (gold-in-index) as well as the raw rate.
    present_set = set(int(d) for d in si.present)
    answerable = [(qid, qt) for qid, qt in queries if qrels[qid] & present_set]
    print(f"    dev-small queries={len(queries):,}; with gold in this index={len(answerable):,}", flush=True)

    # encode queries (sparse)
    qenc = {}
    qtexts = [qt for _, qt in queries]
    for b0 in range(0, len(qtexts), BATCH):
        reps = splade_sparse(qtexts[b0:b0 + BATCH], QUERY_ML, topk=10_000, minw=MINW)
        for (qid, _), rep in zip(queries[b0:b0 + BATCH], reps):
            qenc[qid] = rep

    # warm
    for qid, _ in queries[:5]:
        si.search(*qenc[qid], k=100)

    def run(qset):
        mrr = 0.0; rec = 0; lat = []
        for qid, _ in qset:
            ids, qw = qenc[qid]
            t = time.perf_counter()
            top, _sc = (si.search_corr(ids, qw, k=100) if MODE == "corr"
                        else si.search_fast(ids, qw, k=100) if MODE == "fast"
                        else si.search(ids, qw, k=100))
            lat.append((time.perf_counter() - t) * 1000)
            gold = qrels[qid]
            top = [int(d) for d in top]
            if any(d in gold for d in top):
                rec += 1
            for r, d in enumerate(top[:10]):
                if d in gold:
                    mrr += 1.0 / (r + 1); break
        n = max(1, len(qset))
        return mrr / n, rec / n * 100, np.array(lat)

    mrr_all, rec_all, lat_all = run(queries)
    mrr_ans, rec_ans, _ = run(answerable) if answerable else (0.0, 0.0, np.array([0.0]))
    print(f"\n  ===== native SPLADE-on-lattice, dev-small (NO pool, NO CE){tag} =====", flush=True)
    print(f"    queries scored        : {len(queries):,}  (gold-in-index {len(answerable):,})", flush=True)
    print(f"    serve latency (meet)  : median {np.median(lat_all):.2f} ms  p90 {np.percentile(lat_all,90):.2f} ms", flush=True)
    print(f"    recall@100 (all q)    : {rec_all:.2f}%   <- capped by 50k-slice gold coverage", flush=True)
    print(f"    MRR@10 (all q)        : {mrr_all:.4f}", flush=True)
    print(f"    recall@100 (gold-in-ix): {rec_ans:.2f}%   <- the codec/serve correctness signal", flush=True)
    print(f"    MRR@10 (gold-in-ix)   : {mrr_ans:.4f}   <- OPTIMISTIC (gold competes vs ~50k", flush=True)
    print(f"                            distractors, not 8.8M); proves ranking works, not the headline", flush=True)
    return dict(nq=len(queries), n_ans=len(answerable), mrr_all=mrr_all, rec_all=rec_all,
                mrr_ans=mrr_ans, rec_ans=rec_ans, lat_med=float(np.median(lat_all)),
                lat_p90=float(np.percentile(lat_all, 90)))


# ----------------------------------------------------------------------------------------------
#  CALIBRATE -- 50k end-to-end + full-8.8M projection
# ----------------------------------------------------------------------------------------------
def calibrate():
    N = int(os.environ.get("CAL_N", "50000"))
    print(f"\n========== CALIBRATE native SPLADE-on-lattice on a {N:,}-passage slice ==========\n", flush=True)
    print(f"  WORK={WORK}", flush=True)

    # inject the dev-small gold passages so the 50k slice has a real ranking signal
    gold_pids = set()
    with open(MARCO / "qrels.dev.small.tsv", encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) >= 4 and int(p[3]) > 0:
                gold_pids.add(int(p[2]))
    nd, dps, enc_el = encode(N, start=0, tag=" cal", extra_pids=gold_pids)
    idx_t0 = time.perf_counter()
    ires = index(tag=" cal", chamber=True, proj_max_pid=N)
    idx_el = time.perf_counter() - idx_t0
    sres = serve(tag=" cal")

    # ---- projections to full 8.8M (use the CONTIGUOUS-only footprint, density == full run) ----
    pj = ires["proj"]
    B_per_doc = (pj["di_bytes"] + pj["wt_bytes"]) / max(1, pj["n_docs"])  # di+wt payload (no offsets/vocab const)
    post_per_doc = pj["n_post"] / max(1, pj["n_docs"])
    full_post = post_per_doc * N_DOCS_FULL
    # if encode was cached (dps implausibly high), use a supplied/known measured rate for the projection
    cached_enc = dps > 100_000
    dps_proj = float(os.environ.get("CAL_ENC_DPS", "505")) if cached_enc else dps
    encode_hours = N_DOCS_FULL / dps_proj / 3600.0
    full_shards = max(1, int(math.ceil(full_post / 300_000_000)))
    full_di_GB = pj["di_bytes"] / max(1, pj["n_docs"]) * N_DOCS_FULL / 1e9
    full_wt_GB = pj["wt_bytes"] / max(1, pj["n_docs"]) * N_DOCS_FULL / 1e9
    full_GB_for = full_di_GB + full_wt_GB
    cham_GB = None
    if ires.get("cham_bits"):
        cham_di_GB = full_di_GB / ires["cham_bits"]["ratio"]
        cham_GB = cham_di_GB + full_wt_GB

    print("\n========== CALIBRATION SUMMARY + FULL-8.8M PROJECTION ==========\n", flush=True)
    print(f"  MEASURED on {N:,} docs:", flush=True)
    if cached_enc:
        print(f"    encode throughput   : CACHED (chunks on disk) -- using {dps_proj:,.0f} docs/s "
              f"for projection (set CAL_ENC_DPS or delete chunks to re-measure)", flush=True)
    else:
        print(f"    encode throughput   : {dps:,.0f} docs/s  ({enc_el:.0f}s for {nd:,} docs)", flush=True)
    print(f"    index build time    : {idx_el:.0f}s", flush=True)
    print(f"    postings/doc        : {post_per_doc:.1f}", flush=True)
    print(f"    on-disk FOR index   : {ires['on_disk']/1e6:.1f} MB  ({B_per_doc:.1f} B/doc)", flush=True)
    print(f"      di gaps (FOR)     : {ires['di_bytes']/1e6:.1f} MB", flush=True)
    print(f"      weights (uint8)   : {ires['wt_bytes']/1e6:.1f} MB", flush=True)
    print(f"    FOR round-trip      : {'MATCH' if ires['roundtrip'] else 'MISMATCH'}", flush=True)
    print(f"    serve MRR@10        : {sres['mrr_all']:.4f} all / {sres['mrr_ans']:.4f} gold-in-index", flush=True)
    print(f"    serve latency       : median {sres['lat_med']:.2f} ms (50k-doc accumulator)", flush=True)
    print(f"\n  PROJECTED to full {N_DOCS_FULL:,} docs:", flush=True)
    print(f"    encode time         : {encode_hours:.1f} GPU-hours  (at {dps_proj:,.0f} docs/s)", flush=True)
    print(f"    total postings       : {full_post/1e9:.2f} B", flush=True)
    print(f"    FOR on-disk          : {full_GB_for:.2f} GB  (di {full_di_GB:.2f} + wt {full_wt_GB:.2f})", flush=True)
    if cham_GB:
        print(f"    chamber cold tier    : {cham_GB:.2f} GB  (di chamber-packed {ires['cham_bits']['ratio']:.2f}x + wt uint8)", flush=True)
    print(f"    index-stage RAM      : vocab-sharded -> {full_shards} shard(s) auto, "
          f"peak ~{full_post*5/full_shards/1e9:.1f} GB working set (encode streams to disk, no OOM)", flush=True)
    print(f"    serve latency        : per-query work = touched postings only (the meet unions just", flush=True)
    print(f"                           the query's term posting-lists, not a full 8.8M accumulator);", flush=True)
    print(f"                           50k measured {sres['lat_med']:.1f}ms -> full ~10-40ms est (DF-bound).", flush=True)
    print(f"\n  Launch the full encode with (FRESH WORK dir so 200k chunk boundaries align --", flush=True)
    print(f"  the calibration's 50k chunk_00000 does NOT cover a full 200k chunk):", flush=True)
    print(f"    WORK={WORK}_full CHUNK=200000 python marco_splade_native.py encode --full   # resumable", flush=True)
    print(f"    WORK={WORK}_full CHUNK=200000 python marco_splade_native.py index --full --chamber", flush=True)
    print(f"    WORK={WORK}_full python marco_splade_native.py serve --full", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["calibrate", "encode", "index", "serve"])
    ap.add_argument("--full", action="store_true", help="operate on the full 8.8M collection")
    ap.add_argument("--n", type=int, default=None, help="doc count for encode (default: full or CAL_N)")
    ap.add_argument("--chamber", action="store_true", help="also run the chamber cold-tier probe in index")
    args = ap.parse_args()

    if args.stage == "calibrate":
        calibrate()
    elif args.stage == "encode":
        n = args.n if args.n is not None else (N_DOCS_FULL if args.full else int(os.environ.get("CAL_N", "50000")))
        encode(n, start=0, tag=" full" if args.full else "")
    elif args.stage == "index":
        index(tag=" full" if args.full else "", chamber=args.chamber)
    elif args.stage == "serve":
        serve(tag=" full" if args.full else "")


if __name__ == "__main__":
    main()
