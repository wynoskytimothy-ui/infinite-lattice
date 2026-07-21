#!/usr/bin/env python3
"""aethos_splade_lattice — the distilled, ENCODER-FREE SPLADE tier on the lattice.

SPLADE's learned weights ARE lattice postings. This packages the validated pipeline:
  BUILD   : SPLADE-encode docs once -> store learned weights as a CSR (term=WordPiece id, weight float16).
  DISTILL : SPLADE-encode each vocab WORD once -> save its expansion (the static per-word knowledge table).
  SERVE   : NO model. Query word -> look up its distilled expansion -> MAX-POOL -> scatter-add over the SPLADE
            CSR (the meet). Recovers ~92% of SPLADE's recall gain at pure-lattice speed, no GPU at query time.

The encoder (torch/transformers) is imported lazily and used ONLY at build/distill. save()/load() persist the
index + table; a loaded index serves with numpy alone (mmap-able CSR -> RAM = working set)."""
from __future__ import annotations
import os, json
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _fast_tok import words

V_WORDPIECE = 30522


def _pack_nbit_local(indices, bits):
    """Pack uint8 indices (each `bits` low bits) into a byte stream (footprint stack)."""
    n = len(indices)
    if n == 0: return np.zeros(0, np.uint8)
    out = np.zeros((n * bits + 7) // 8, np.uint8); acc = acc_bits = oi = 0; mask = (1 << bits) - 1
    for v in indices:
        acc |= (int(v) & mask) << acc_bits; acc_bits += bits
        while acc_bits >= 8:
            out[oi] = acc & 0xFF; oi += 1; acc >>= 8; acc_bits -= 8
    if acc_bits: out[oi] = acc & 0xFF
    return out


def _unpack_nbit_local(packed, n, bits):
    if n == 0: return np.zeros(0, np.uint8)
    out = np.empty(n, np.uint8); acc = acc_bits = pi = 0; mask = (1 << bits) - 1
    for i in range(n):
        while acc_bits < bits and pi < len(packed):
            acc |= int(packed[pi]) << acc_bits; pi += 1; acc_bits += 8
        out[i] = acc & mask; acc >>= bits; acc_bits -= bits
    return out


class SpladeEncoder:
    """Only needed at ingest/distill (NOT serve). GPU-accelerated when CUDA is present (fp16, on-device top-K
    to minimize transfer); falls back to CPU. Lazy torch import."""
    def __init__(self, name="prithivida/Splade_PP_en_v1", device=None):
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(name)
        m = AutoModelForMaskedLM.from_pretrained(name).eval()
        if self.device == "cuda":
            m = m.half()                                  # fp16 on GPU
        else:
            torch.set_num_threads(os.cpu_count() or 4)
        self.m = m.to(self.device)

    def encode(self, texts, bs=None, topk=None):
        t = self.torch; dev = self.device
        bs = bs or (128 if dev == "cuda" else 16)         # bigger batches on GPU
        terms, wts, off = [], [], [0]
        with t.no_grad():
            for i in range(0, len(texts), bs):
                b = self.tok(texts[i:i + bs], padding=True, truncation=True, max_length=256, return_tensors="pt")
                b = {k: v.to(dev) for k, v in b.items()}
                v = (t.log1p(t.relu(self.m(**b).logits)) * b["attention_mask"].unsqueeze(-1)).max(1).values  # [B,V]
                if topk:                                   # on-GPU top-K -> move only top-K to CPU
                    vals, idx = t.topk(v, min(topk, v.shape[1]), dim=1)
                    vals = vals.float().cpu().numpy(); idx = idx.cpu().numpy()
                    for r in range(idx.shape[0]):
                        msk = vals[r] > 0
                        terms.append(idx[r][msk].astype(np.int32)); wts.append(vals[r][msk].astype(np.float16))
                        off.append(off[-1] + int(msk.sum()))
                else:                                      # keep all nonzero (queries / per-word distill)
                    vc = v.float().cpu().numpy()
                    for row in vc:
                        nz = np.nonzero(row)[0]
                        terms.append(nz.astype(np.int32)); wts.append(row[nz].astype(np.float16))
                        off.append(off[-1] + nz.size)
        return np.concatenate(terms) if terms else np.zeros(0, np.int32), \
            np.concatenate(wts).astype(np.float16) if wts else np.zeros(0, np.float16), np.asarray(off, np.int64)


def _csr_by_term(terms, offs, V=V_WORDPIECE):
    di = np.repeat(np.arange(offs.size - 1), np.diff(offs)).astype(np.int32)
    order = np.argsort(terms, kind="stable"); T = terms[order]
    df = np.bincount(T, minlength=V); indptr = np.concatenate([[0], np.cumsum(df)]).astype(np.int64)
    return indptr, di[order].astype(np.uint32), order


class DistilledSpladeIndex:
    def __init__(self):
        self.doc_ids = None

    # ---- build (needs encoder) ----
    def build_docs(self, corpus, encoder, topk=128):
        self.doc_ids = list(corpus.keys()); self.N = len(self.doc_ids)
        t, w, o = encoder.encode([corpus[d] for d in self.doc_ids], topk=topk)
        self._set_docs(t, w, o)
        return self

    def _set_docs(self, terms, weights, offs, seed_topk=None):   # also used to build from cached vectors
        if seed_topk:                                    # #10 seed-trim: keep only the top-seed_topk weights per doc
            terms, weights, offs = self._trim_per_doc(terms, weights, offs, int(seed_topk))
        self.indptr, self.seg_doc, order = _csr_by_term(terms, offs)
        self.seg_w = weights[order]
        self.seed_topk = seed_topk
        return self

    @staticmethod
    def _trim_per_doc(terms, weights, offs, topk):
        """Keep each doc's top-`topk` SPLADE terms by weight (the low-weight tail is near-zero noise -- dropping it
        shrinks every posting list ~topk/128x and, per the lazy-regenerate finding, often HOLDS or lifts recall)."""
        nt, nw, no = [], [], [0]
        w32 = np.asarray(weights, np.float32)
        for d in range(len(offs) - 1):
            a, e = int(offs[d]), int(offs[d + 1])
            if e - a > topk:
                keep = a + np.argpartition(w32[a:e], -topk)[-topk:]
                nt.append(np.asarray(terms)[keep]); nw.append(np.asarray(weights)[keep]); no.append(no[-1] + topk)
            else:
                nt.append(np.asarray(terms)[a:e]); nw.append(np.asarray(weights)[a:e]); no.append(no[-1] + (e - a))
        return (np.concatenate(nt) if nt else np.zeros(0, np.int32),
                np.concatenate(nw) if nw else np.zeros(0, np.float16),
                np.asarray(no, np.int64))

    def configure_serve(self, tau=None, topq=None):
        """Set default serve gates (#8): tau = energy threshold on expanded query neurons (corpus-adaptive,
        ~2x faster near-lossless), topq = hard cap on kept neurons. Holds the SPLADE serve band at scale (the
        accumulator stays O(N) -- WAND is measured-dead for non-neg SPLADE queries -- so gating bounds the constant)."""
        if tau is not None: self.tau = float(tau)
        if topq is not None: self.topq = int(topq)
        return self

    def distill(self, vocab_words, encoder, topk=64):
        self.vocab = list(vocab_words); self.w2i = {w: i for i, w in enumerate(self.vocab)}
        t, w, o = encoder.encode(self.vocab, bs=64, topk=topk)
        self.wt_terms, self.wt_w, self.wt_off = t, w, o
        return self

    def _set_table(self, vocab_words, terms, weights, offs):
        self.vocab = list(vocab_words); self.w2i = {w: i for i, w in enumerate(self.vocab)}
        self.wt_terms, self.wt_w, self.wt_off = terms, weights, offs
        return self

    # ---- serve (NO model) ----
    def _query_expansion(self, query):
        acc = {}                                          # term -> MAX weight (matches SPLADE pooling)
        for w in words(query):
            j = self.w2i.get(w)
            if j is None: continue
            a, e = int(self.wt_off[j]), int(self.wt_off[j + 1])
            for t, wt in zip(self.wt_terms[a:e], self.wt_w[a:e]):
                t = int(t); wt = float(wt)
                if wt > acc.get(t, 0.0): acc[t] = wt
        return acc

    def score_vec(self, query, prune=None, tau=None, topq=None):
        """Full N-length SPLADE score vector (search() and the routed complement both read this).
        Selective activation on the semantic forward pass. Two gates on the expanded query neurons:
          tau (energy threshold, 0<tau<1): keep the strongest neurons until cumulative weight >= tau * total L1 mass
              -- CORPUS-ADAPTIVE (fires ~few on a peaked query, more on a diffuse one), no fixed constant.
          prune (fixed top-K): keep only the top-`prune` neurons by weight.
        SPLADE expansions have a long near-zero-weight tail that touches huge posting lists without ranking; gating it
        is ~2x faster serve at near-lossless top-k (block-max WAND LOST here: many-term non-neg queries => weak bounds)."""
        if prune is None: prune = getattr(self, "prune_terms", None)
        if tau is None: tau = getattr(self, "tau", None)
        if topq is None: topq = getattr(self, "topq", None)
        exp = self._query_expansion(query)
        if tau is not None and 0.0 < tau < 1.0 and exp:
            items = sorted(exp.items(), key=lambda x: -x[1]); tot = sum(w for _, w in items) or 1.0
            acc = 0.0; kept = {}
            for t, w in items:
                kept[t] = w; acc += w
                if acc >= tau * tot: break
            exp = kept
        elif prune and prune < len(exp):
            exp = dict(sorted(exp.items(), key=lambda x: -x[1])[:prune])
        if topq and topq < len(exp):                     # hard neuron cap (#8), belt-and-braces with tau
            exp = dict(sorted(exp.items(), key=lambda x: -x[1])[:topq])
        scores = np.zeros(self.N)
        for tid, qw in exp.items():
            a, e = int(self.indptr[tid]), int(self.indptr[tid + 1])
            if e > a: scores[self.seg_doc[a:e]] += qw * self.seg_w[a:e].astype(np.float32)
        return scores

    def search(self, query, k=10, prune=None, tau=None, topq=None):
        scores = self.score_vec(query, prune=prune, tau=tau, topq=topq)
        kk = min(k, self.N); top = np.argpartition(scores, -kk)[-kk:]
        top = top[np.argsort(scores[top])[::-1]]
        return [self.doc_ids[i] for i in top if scores[i] > 0]

    # ---- persistence (serve needs no model) ----
    def save(self, path):
        os.makedirs(path, exist_ok=True)
        np.save(f"{path}/indptr.npy", self.indptr); np.save(f"{path}/seg_doc.npy", self.seg_doc)
        np.save(f"{path}/seg_w.npy", self.seg_w)
        np.save(f"{path}/wt_terms.npy", self.wt_terms); np.save(f"{path}/wt_w.npy", self.wt_w)
        np.save(f"{path}/wt_off.npy", self.wt_off)
        json.dump({"doc_ids": self.doc_ids, "vocab": self.vocab, "N": self.N}, open(f"{path}/meta.json", "w"))
        return self

    # ---- footprint stack (priority #5): delta+varbyte doc-ids LOSSLESS + 3-bit per-term-max weights ----
    _FP_BITS = 3

    @staticmethod
    def _vb_enc(vals):
        out = bytearray()
        for v in vals:
            v = int(v)
            while True:
                b = v & 0x7F; v >>= 7; out.append(b | (0x80 if v else 0))
                if not v: break
        return bytes(out)

    @staticmethod
    def _vb_dec(buf, n, pos):
        out = np.empty(n, np.int64)
        for i in range(n):
            shift = val = 0
            while True:
                b = buf[pos]; pos += 1; val |= (b & 0x7F) << shift
                if not (b & 0x80): break
                shift += 7
            out[i] = val
        return out, pos

    def save_compressed(self, path):
        """Footprint-stack persist (priority #5): per-term delta+varbyte doc-ids (LOSSLESS) + 3-bit per-term-max
        weights (near-lossless, ~+0.0 nDCG). ~3.6x smaller than fp16-weights + int-doc-ids, serves with NO model.
        Pure-Python varbyte (fine for edge/mid corpora; MARCO-scale wants a numba varbyte for decode-on-the-fly)."""
        os.makedirs(path, exist_ok=True); B = self._FP_BITS
        di_parts, w_parts, n_arr, dl_arr, wl_arr, sc_arr = [], [], [], [], [], []
        ip, sd, sw = self.indptr, np.asarray(self.seg_doc), np.asarray(self.seg_w, np.float32)
        for t in range(len(ip) - 1):
            a, e = int(ip[t]), int(ip[t + 1]); n = e - a
            if n == 0: n_arr.append(0); dl_arr.append(0); wl_arr.append(0); sc_arr.append(np.float16(0)); continue
            o = np.argsort(sd[a:e]); docs = sd[a:e][o]; w = sw[a:e][o]
            di = self._vb_enc(np.diff(docs, prepend=0))
            L = 1 << B; tmax = float(w.max()) or 1.0
            widx = np.clip((w / tmax * L).astype(np.int64), 0, L - 1).astype(np.uint8)
            wp = _pack_nbit_local(widx, B).tobytes()
            di_parts.append(di); w_parts.append(wp)
            n_arr.append(n); dl_arr.append(len(di)); wl_arr.append(len(wp)); sc_arr.append(np.float16(tmax))
        open(f"{path}/blob.bin", "wb").write(b"".join(di_parts) + b"".join(w_parts))
        np.savez(f"{path}/meta.npz", n=np.array(n_arr, np.int64), dl=np.array(dl_arr, np.int64),
                 wl=np.array(wl_arr, np.int64), sc=np.array(sc_arr, np.float16), bits=B)
        np.save(f"{path}/wt_terms.npy", self.wt_terms); np.save(f"{path}/wt_w.npy", self.wt_w)
        np.save(f"{path}/wt_off.npy", self.wt_off)
        json.dump({"doc_ids": self.doc_ids, "vocab": self.vocab, "N": self.N}, open(f"{path}/meta.json", "w"))
        return self

    @classmethod
    def load_compressed(cls, path):
        self = cls(); m = np.load(f"{path}/meta.npz"); B = int(m["bits"])
        blob = open(f"{path}/blob.bin", "rb").read()
        n_arr, dl_arr, wl_arr, sc_arr = m["n"], m["dl"], m["wl"], m["sc"]
        di_off = 0; w_off = int(dl_arr.sum()); indptr = [0]; sd, sw = [], []
        for n, dl, wl, sc in zip(n_arr, dl_arr, wl_arr, sc_arr):
            if n == 0: indptr.append(indptr[-1]); continue
            gaps, _ = cls._vb_dec(blob[di_off:di_off + dl], int(n), 0); di_off += int(dl)
            widx = _unpack_nbit_local(np.frombuffer(blob[w_off:w_off + wl], np.uint8), int(n), B); w_off += int(wl)
            sd.append(np.cumsum(gaps)); sw.append((widx.astype(np.float32) + 0.5) / (1 << B) * float(sc))
            indptr.append(indptr[-1] + int(n))
        self.indptr = np.array(indptr, np.int64); self.seg_doc = np.concatenate(sd).astype(np.int64)
        self.seg_w = np.concatenate(sw).astype(np.float16)
        self.wt_terms = np.load(f"{path}/wt_terms.npy"); self.wt_w = np.load(f"{path}/wt_w.npy")
        self.wt_off = np.load(f"{path}/wt_off.npy")
        j = json.load(open(f"{path}/meta.json")); self.doc_ids = j["doc_ids"]; self.vocab = j["vocab"]; self.N = j["N"]
        self.w2i = {w: i for i, w in enumerate(self.vocab)}
        return self

    @classmethod
    def from_cache(cls, doc_ids, doc_npz, distill_npz, seed_topk=None):
        """One-call ENCODER-FREE build from baked outputs (no model): doc_npz has t/w/o (doc term-ids/weights/
        offsets from the GPU-once ingest bake); distill_npz has words/t/w/o (the static query-expansion table).
        Makes `eng.attach_splade(DistilledSpladeIndex.from_cache(doc_ids, doc_npz, distill_npz))` a one-liner --
        the whole-system-map priority #2 wiring (bake once at ingest, serve encoder-free forever). seed_topk trims
        each doc to its top-k terms at build (#10 footprint/serve win, no re-encode needed)."""
        self = cls(); self.doc_ids = list(doc_ids); self.N = len(self.doc_ids)
        dz = np.load(doc_npz); vz = np.load(distill_npz, allow_pickle=True)
        self._set_docs(dz["t"], dz["w"], dz["o"], seed_topk=seed_topk)
        self._set_table(vz["words"].tolist(), vz["t"], vz["w"], vz["o"])
        return self

    @classmethod
    def load(cls, path, mmap=True):
        self = cls(); mm = "r" if mmap else None
        self.indptr = np.load(f"{path}/indptr.npy")
        self.seg_doc = np.load(f"{path}/seg_doc.npy", mmap_mode=mm)
        self.seg_w = np.load(f"{path}/seg_w.npy", mmap_mode=mm)
        self.wt_terms = np.load(f"{path}/wt_terms.npy"); self.wt_w = np.load(f"{path}/wt_w.npy")
        self.wt_off = np.load(f"{path}/wt_off.npy")
        m = json.load(open(f"{path}/meta.json"))
        self.doc_ids = m["doc_ids"]; self.vocab = m["vocab"]; self.N = m["N"]
        self.w2i = {w: i for i, w in enumerate(self.vocab)}
        return self


def _selftest():
    """Reuse the cached nfcorpus SPLADE vectors -> build, serve, verify ~92% recall recovery + save/load."""
    import time
    from scripts.bench_supervised_bridges import load, ndcg10
    from aethos_edge_rag import EdgeRAG
    CACHE = Path(os.environ.get("TEMP", "/tmp"))
    name = "nfcorpus"
    corpus, queries, train_q, test_q = load(name)
    doc_ids = list(corpus.keys()); d2i = {d: i for i, d in enumerate(doc_ids)}
    test_ids = [q for q in test_q if q in queries]
    print("=" * 84); print("DistilledSpladeIndex — encoder-free SPLADE tier on the lattice (nfcorpus)"); print("=" * 84)

    dz = np.load(CACHE / f"splade_{name}_doc128.npz")        # cached doc vectors
    vocab = sorted({w for q in test_ids for w in words(queries[q])})
    vz = np.load(CACHE / f"distill_{name}_v{len(vocab)}.npz", allow_pickle=True)  # cached per-word table
    qz = np.load(CACHE / f"splade_{name}_qry.npz")           # cached full-query (for SPLADE-full ref)

    idx = DistilledSpladeIndex(); idx.doc_ids = doc_ids; idx.N = len(doc_ids)
    idx._set_docs(dz["t"], dz["w"], dz["o"])
    idx._set_table(vz["words"].tolist(), vz["t"], vz["w"], vz["o"])

    eng = EdgeRAG().build(corpus)

    def rec(top, gold): return len(set(top) & gold) / len(gold) if gold else 0.0
    L = S = D = nq = 0; lat = []
    for qi, qid in enumerate(test_ids):
        gold = {d2i[d] for d in test_q[qid] if test_q[qid].get(d, 0) > 0 and d in d2i}
        if not gold: continue
        nq += 1
        L += rec([d2i[d] for d in eng.search(queries[qid], 100)], gold)
        sf = np.zeros(idx.N)
        for tid, w in zip(qz["t"][qz["o"][qi]:qz["o"][qi+1]], qz["w"][qz["o"][qi]:qz["o"][qi+1]]):
            a, e = int(idx.indptr[tid]), int(idx.indptr[tid+1])
            if e > a: sf[idx.seg_doc[a:e]] += float(w) * idx.seg_w[a:e].astype(np.float32)
        S += rec(list(np.argsort(sf)[::-1][:100]), gold)
        t0 = time.perf_counter(); top = idx.search(queries[qid], 100); lat.append((time.perf_counter()-t0)*1000)
        D += rec([d2i[d] for d in top], gold)
    L, S, D = L/nq, S/nq, D/nq
    print(f"  Recall@100: lexical {L:.4f} | distilled(encoder-free) {D:.4f} | SPLADE-full {S:.4f}")
    print(f"  -> recovers {(D-L)/(S-L)*100:.0f}% of SPLADE's recall gain, {np.median(lat):.2f} ms/q, NO query encoder")

    p = os.path.join(CACHE, "distilled_splade_idx"); idx.save(p)
    idx2 = DistilledSpladeIndex.load(p)
    same = idx2.search(queries[test_ids[0]], 10) == idx.search(queries[test_ids[0]], 10)
    sz = sum(os.path.getsize(os.path.join(p, f)) for f in os.listdir(p)) / idx.N
    print(f"  save/load round-trip serves identically: {same} | on-disk {sz:.0f} B/doc | model NOT needed to serve")


if __name__ == "__main__":
    _selftest()
