#!/usr/bin/env python3
"""aethos_rag_api — the deployable AETHOS RAG service (Andrea's December ask: demo + API, OCI free tier).

WHAT THIS IS
    A FastAPI app that packages the MEASURED AETHOS RAG engine as a drop-in Pitagora-contract service.
    It is a thin, honest wrapper over two already-built, already-measured pieces of the repo:

      1. DENSE COMPRESSION (the index footprint claim).  Product Quantization of dense embeddings, reusing
         the exact primitives measured in `_dense_compress_scifact.py`:
            * make_free_rotation  -- the FREE octahedral+permutation orthogonal rotation (0 stored bytes),
            * pq_train / pq_reconstruct / code_entropy_bytes_per_doc.
         MEASURED (real bge-large, 1024-d): scifact fp32 0.7463@4096B -> 0.7223@330B (12x, R@100 beats full);
         nfcorpus 0.3814@4096B -> 0.3664@66B (62x); 96-97% accuracy retained; NO GPU at serve; clears <4KB/doc.
         HONEST SCOPE: the compression is Product Quantization -- competitive and standard, NOT a proprietary
         ratio. The AETHOS moat is the *packaging*: glass-box explain, no-GPU-at-serve, invertible/regenerable
         codec (the rotation is 0 bytes, replayed from a seed).

      2. GLASS-BOX EXPLAIN (the differentiator).  A no-GPU lexical lattice (aethos_rag_pipeline.AethosPipeline)
         built over the SAME corpus, so /explain returns named, traceable signal contributions and learned
         bridges -- provenance a generic reranker cannot produce. This tier is entirely optional and never
         touches a GPU; if its heavy deps are missing the service still serves dense results and a dense-only
         explanation.

THE NO-GPU-AT-SERVE BOUNDARY (enforced, not just claimed)
    * The ONLY place a model/GPU is ever touched is EMBEDDING THE QUERY.  Three honest modes:
        - precomputed  (default contract): the caller sends the query vector -> ZERO model at serve.
        - cpu_encoder  (optional): a lightweight sentence-transformers model pinned to device="cpu".
        - none         : text-only queries are rejected with a clear 400 unless one of the above is set.
    * We hard-disable CUDA at import (`CUDA_VISIBLE_DEVICES=""`) so even the optional encoder cannot reach a
      GPU; the codec + ADC search path is pure numpy and imports no deep-learning framework at all.
    * /stats always reports gpu_at_serve=false, and reports which query-embedding mode is active so the
      boundary is auditable from the outside.
    * INGEST embeddings may be computed anywhere (even a GPU box, offline); they are frozen into PQ codes.
      Serving only *decodes* PQ codes with numpy dot-tables. Nothing at serve can allocate a GPU.

PERSISTENCE
    The codec (doc ids, texts, PQ codes, codebooks, and the rotation *seed* -- the rotation itself is 0 bytes,
    regenerated deterministically) is saved to a single directory and RELOADED on startup, so a restarted
    container serves immediately with no re-ingest.

ENDPOINTS (see `openapi` / the README block below for signatures)
    POST /ingest          docs -> build & persist the compressed index (+ optional glass-box lattice)
    POST /search          {query|query_embedding, k} -> ranked [{doc_id, score, text}]
    POST /explain         {query|query_embedding, doc_id} -> {signals, bridges_fired, factors}
    GET  /stats           -> {n_docs, bytes_per_doc, gpu_at_serve:false, compression_vs_fp32, ...}
    GET  /healthz         -> liveness
    Pitagora drop-in aliases (same handlers): POST /create_rag, POST /add_documents, POST /retrieve
"""
# NOTE: intentionally NO `from __future__ import annotations` -- FastAPI must be able to resolve the
# Pydantic request models from real (non-stringized) annotations at route-analysis time.

# ----- HARD no-GPU boundary: disable CUDA before anything can import a DL framework -----
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")          # nothing at serve may see a GPU
os.environ.setdefault("PYTHONUTF8", "1")

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

try:                                                       # allow `python aethos_rag_api.py` from anywhere
    sys.stdout.reconfigure(encoding="utf-8")               # type: ignore[attr-defined]
except Exception:
    pass

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# --- reuse the MEASURED compression primitives verbatim (no re-implementation) ---
from _dense_compress_scifact import (                      # noqa: E402
    make_free_rotation,
    pq_train,
    pq_reconstruct,                                        # kept for exact-recon / debug parity
    code_entropy_bytes_per_doc,
)

# Persisted index location (override with AETHOS_INDEX_DIR); the codec is reloaded from here on startup.
INDEX_DIR = Path(os.environ.get("AETHOS_INDEX_DIR", str(HERE / "aethos_index")))


# ------------------------------------------------------------------ request models (module level)
# Defined at module scope (not inside build_app) so FastAPI can resolve them as request bodies.
try:
    from pydantic import BaseModel, Field
    _HAVE_PYDANTIC = True
except Exception:                                          # module still importable without fastapi/pydantic
    _HAVE_PYDANTIC = False
    BaseModel = object                                     # type: ignore

    def Field(default=None, **_):                          # type: ignore
        return default


class DocIn(BaseModel):
    doc_id: str
    text: str = ""
    embedding: Optional[List[float]] = None


class IngestIn(BaseModel):
    docs: List[DocIn]
    M: int = Field(32, description="PQ subquantizers (must divide the embedding dim)")
    Kp: int = 256
    seed: int = 1234
    rounds: int = 3
    queries: Optional[Dict[str, str]] = None               # optional: enable named bridges in /explain
    qrels: Optional[Dict[str, Dict[str, int]]] = None


class SearchIn(BaseModel):
    query: Optional[str] = None
    query_embedding: Optional[List[float]] = None
    k: int = 10


class ExplainIn(BaseModel):
    query: Optional[str] = None
    query_embedding: Optional[List[float]] = None
    doc_id: str


# =============================================================================================
# Dense PQ codec  --  the compressed, no-GPU-at-serve index (reuses _dense_compress_scifact.py)
# =============================================================================================
class DensePQCodec:
    """Two-tier Product-Quantized dense index with the FREE octahedral rotation, no GPU at serve.

    MEASURED (2026-07-11 finish-rag, _dense_harness reproduces scifact fp32 0.7463/0.9483 @ 4096 B/doc):
      * ADC-ONLY (M=32) does NOT hold the .74/.96 gate (M=32 ~0.61, M=64 ~0.70) -- rerank is MANDATORY.
      * PQ-ADC recall (all N) -> exact int8 rerank of the top-C shortlist RESTORES the baseline:
        M=32 / C=500 -> scifact 0.7473 / 0.9483 at 0.60 ms/q, and holds the complemented stack (0.7668/0.9533).

    Footprint model (amortized):
        rotation      = 0 stored bytes  (regenerated from seed -- invertible/regenerable)
        codebook      = Kp * D * 4 bytes, shared  -> Kp*D*4 / N per doc (->0 at scale)
        hot ADC codes = M bytes/doc  (M=32 -> 32 B/doc searchable index; ~194x smaller HOT than fp32)
        int8 rerank   = D bytes/doc  (verified ~lossless; can live cold/mmap and page only the top-C rows)
        in-RAM total  = ~32 + 1024 = ~1056 B/doc = ~3.9x smaller than fp32 at matched full accuracy.
    Honest vs a naive fp32 store: 13.5-16.7x MEASURED at these N (24x is asymptotic codebook-amortization).
    Serve = ADC (query rotated once, per-subquantizer dot table, sum of look-ups) then int8 dot on C candidates
    -- pure numpy, no reconstruction of the full corpus, no GPU.

    IVF (optional scale tier): auto-built at >= IVF_MIN_DOCS docs. MEASURED: on small corpora IVF must probe ~50%
    of cells to hold recall (no win -> stays off); the speedup GROWS with N (synthetic: 2.3x@10k, 6.4x@50k,
    13.8x@200k, exhaustive linear vs IVF ~flat), so it is the right lever only at scale.
    """

    IVF_MIN_DOCS = 100_000                                 # below this, exhaustive ADC holds recall & is faster (BMW pattern)

    def __init__(self, M: int = 32, Kp: int = 256, rounds: int = 3, seed: int = 1234, nrot: int = 32,
                 rerank_C: int = 500):
        self.M = int(M)
        self.Kp = int(Kp)
        self.rounds = int(rounds)
        self.seed = int(seed)
        self.nrot = int(nrot)
        self.rerank_C = int(rerank_C)                      # ADC shortlist size handed to the exact int8 rerank
        self.D: int = 0
        self.sub: int = 0
        self.doc_ids: List[str] = []
        self.codes: Optional[np.ndarray] = None            # [N, M]  -- the 32 B/doc hot ADC index
        self.codebooks: Optional[np.ndarray] = None        # [M, Kp, sub]
        self._rotate = None                                # callable X->X R (regenerated, never stored)
        self._entropy_bpd: float = 0.0
        self.v8: Optional[np.ndarray] = None               # [N, D] uint8 -- int8 rerank tier (verified lossless)
        self.v8_lo: Optional[np.ndarray] = None            # [D] per-dim min
        self.v8_scale: Optional[np.ndarray] = None         # [D] per-dim (hi-lo)/255
        # IVF coarse quantizer (optional; the SCALE tier -> O(sqrt(N)) ADC instead of O(N))
        self.coarse: Optional[np.ndarray] = None           # [nlist, D] cell centroids (rotated space)
        self.coarse_sqnorm: Optional[np.ndarray] = None    # [nlist] 0.5*||centroid||^2 for L2-consistent probe
        self.inv_docs: Optional[np.ndarray] = None         # [N] int32 doc indices, grouped by cell
        self.inv_ptr: Optional[np.ndarray] = None          # [nlist+1] CSR offsets into inv_docs
        self.nlist: int = 0
        self.nprobe: int = 0

    # ---- build ----
    def build(self, doc_ids: List[str], embeddings: np.ndarray) -> "DensePQCodec":
        X = np.ascontiguousarray(embeddings, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError("embeddings must be a 2-D array [n_docs, dim]")
        if len(doc_ids) != X.shape[0]:
            raise ValueError("doc_ids and embeddings length mismatch")
        self.D = int(X.shape[1])
        if self.D % self.M != 0:
            raise ValueError(f"dim {self.D} not divisible by M={self.M}; pick M dividing the embedding dim")
        Xn = self._l2(X)
        self._rotate = make_free_rotation(self.D, rounds=self.rounds, seed=self.seed, nrot=self.nrot)
        Yr = self._rotate(Xn)                              # orthogonal rotation preserves cosine/dot exactly
        codes, cbs, sub = pq_train(Yr, self.M, self.Kp, seed=1)
        self.doc_ids = [str(d) for d in doc_ids]
        self.codes = np.ascontiguousarray(codes)
        self.sub = int(sub)
        self.codebooks = np.stack([np.ascontiguousarray(c, np.float32) for c in cbs])  # [M, Kp, sub]
        self._entropy_bpd = float(code_entropy_bytes_per_doc(codes))
        self.v8, self.v8_lo, self.v8_scale = self._int8_encode(Xn)   # rerank tier from the pre-rotation cosine space
        return self

    @staticmethod
    def _l2(X: np.ndarray) -> np.ndarray:
        return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)

    @staticmethod
    def _int8_encode(Xn: np.ndarray):
        """Per-dim min-max int8 of L2-normalized vectors (verified effectively lossless: scifact 0.7464/0.9483)."""
        lo = Xn.min(0).astype(np.float32); hi = Xn.max(0).astype(np.float32)
        scale = ((hi - lo) / 255.0).astype(np.float32) + 1e-12
        q = np.clip(np.round((Xn - lo) / scale), 0, 255).astype(np.uint8)
        return np.ascontiguousarray(q), lo, scale

    def _rerank_dot(self, qn: np.ndarray, idx: np.ndarray) -> np.ndarray:
        """Exact-ish cosine of the int8-reconstructed candidate rows against the normalized query."""
        rec = self.v8[idx].astype(np.float32) * self.v8_scale + self.v8_lo    # [C, D]
        return rec @ qn

    @staticmethod
    def _nn(X: np.ndarray, C: np.ndarray, chunk: int = 8192) -> np.ndarray:
        """Nearest-centroid assignment by L2 (argmin ||x-c||^2 = argmin |c|^2 - 2 x.c)."""
        cn = (C * C).sum(1)
        out = np.empty(len(X), np.int64)
        for i in range(0, len(X), chunk):
            Xi = X[i:i + chunk]
            out[i:i + chunk] = (cn[None, :] - 2.0 * (Xi @ C.T)).argmin(1)
        return out

    def build_ivf(self, nlist: Optional[int] = None, nprobe: Optional[int] = None,
                  iters: int = 12, seed: int = 1) -> "DensePQCodec":
        """Fit the IVF coarse quantizer over the (int8-reconstructed) rotated vectors so serve becomes sublinear.
        nlist ~ 4*sqrt(N) cells; at query, probe the nprobe nearest cells and ADC only their docs. IVF is a WORK
        lever, not a bytes lever: adds ~cell-id/doc + coarse codebook (nlist*D*4/N -> ~0 at scale). Rerank unchanged."""
        if self.codes is None or self.v8 is None or self._rotate is None:
            raise RuntimeError("build the codec (with int8 tier) before build_ivf")
        n = self.codes.shape[0]
        Xn = self.v8.astype(np.float32) * self.v8_scale + self.v8_lo          # near-lossless recon for coarse fit
        Yr = self._rotate(Xn)                                                 # rotated space = the ADC/query space
        nl = int(nlist or max(1, min(n, round(4.0 * np.sqrt(n)))))
        rng = np.random.default_rng(seed)
        C = Yr[rng.choice(n, min(nl, n), replace=False)].astype(np.float32).copy()
        for _ in range(iters):
            a = self._nn(Yr, C)
            for k in range(len(C)):
                m = a == k
                if m.any(): C[k] = Yr[m].mean(0)
        a = self._nn(Yr, C)
        self.coarse = np.ascontiguousarray(C)
        self.coarse_sqnorm = (0.5 * (C * C).sum(1)).astype(np.float32)        # L2-consistent probe offset
        order = np.argsort(a, kind="stable")
        self.inv_docs = order.astype(np.int32)
        counts = np.bincount(a, minlength=len(C))
        self.inv_ptr = np.zeros(len(C) + 1, np.int64); self.inv_ptr[1:] = np.cumsum(counts)
        self.nlist = int(len(C))
        self.nprobe = int(nprobe or max(1, round(np.sqrt(self.nlist))))
        return self

    # ---- serve (ADC dot-product; no reconstruction, no GPU) ----
    def score(self, q_emb: np.ndarray) -> np.ndarray:
        if self.codes is None or self._rotate is None:
            raise RuntimeError("codec is empty -- ingest first")
        q = np.ascontiguousarray(q_emb, dtype=np.float32).reshape(-1)
        if q.shape[0] != self.D:
            raise ValueError(f"query embedding dim {q.shape[0]} != index dim {self.D}")
        q = q / (np.linalg.norm(q) + 1e-9)
        qr = self._rotate(q[None, :])[0]                   # rotate query the SAME way (dot preserved)
        # per-subquantizer dot table T[m, code] = <qr_sub_m, centroid_code>, then sum look-ups over docs
        scores = np.zeros(self.codes.shape[0], dtype=np.float32)
        for m in range(self.M):
            qsub = qr[m * self.sub:(m + 1) * self.sub]
            table = self.codebooks[m] @ qsub               # [Kp]
            scores += table[self.codes[:, m]]
        return scores

    # ---- serve: (optional IVF gather) -> PQ-ADC recall -> exact int8 rerank of the top-C shortlist ----
    def search(self, q_emb: np.ndarray, k: int = 10, C: Optional[int] = None, rerank: bool = True,
               use_ivf: Optional[bool] = None, nprobe: Optional[int] = None):
        """Return (top_indices, top_scores). ADC alone fails the .74/.96 gate; the int8 rerank of the top-C
        shortlist restores the full fp32 baseline (M=32/C=500 -> scifact 0.7473/0.9483, 0.60 ms/q). If the IVF
        coarse quantizer is built (and use_ivf is not False), ADC runs on only the nprobe-cell candidates -> O(sqrt N)."""
        if self.codes is None or self._rotate is None:
            raise RuntimeError("codec is empty -- ingest first")
        q = np.ascontiguousarray(q_emb, dtype=np.float32).reshape(-1)
        qn = q / (np.linalg.norm(q) + 1e-9)
        n = self.codes.shape[0]; k = max(1, min(int(k), n))
        do_ivf = (self.coarse is not None) if use_ivf is None else bool(use_ivf)
        if do_ivf and self.coarse is not None:
            qr = self._rotate(q[None, :] / (np.linalg.norm(q) + 1e-9))[0]
            probe_score = self.coarse @ qr - self.coarse_sqnorm            # L2-consistent cell probe
            npb = min(int(nprobe or self.nprobe), self.nlist)
            cells = np.argpartition(-probe_score, npb - 1)[:npb]
            cand = np.concatenate([self.inv_docs[self.inv_ptr[c]:self.inv_ptr[c + 1]] for c in cells])
            if cand.size == 0:
                do_ivf = False                                            # empty probe -> fall back to exhaustive
            else:
                codes_c = self.codes[cand]; adc = np.zeros(cand.size, np.float32)
                for m in range(self.M):
                    adc += (self.codebooks[m] @ qr[m * self.sub:(m + 1) * self.sub])[codes_c[:, m]]
                return self._finish_rank(qn, cand, adc, k, C, rerank)
        adc = self.score(q_emb)                                           # exhaustive ADC over the whole corpus
        return self._finish_rank(qn, np.arange(n), adc, k, C, rerank)

    def _finish_rank(self, qn, cand, adc, k, C, rerank):
        """Shared tail: rerank the top-C ADC candidates with the exact int8 tier, return top-k (indices, scores)."""
        m = cand.size; k = max(1, min(int(k), m))
        if rerank and self.v8 is not None:
            c = min(max(int(C or self.rerank_C), k), m)
            loc = np.argpartition(-adc, c - 1)[:c]; shortlist = cand[loc]
            ex = self._rerank_dot(qn, shortlist)
            sel = shortlist[np.argsort(-ex)[:k]]
            return sel, self._rerank_dot(qn, sel)
        loc = np.argpartition(-adc, k - 1)[:k]; loc = loc[np.argsort(-adc[loc])]
        return cand[loc], adc[loc]

    # ---- footprint ----
    def bytes_per_doc(self) -> Dict[str, float]:
        n = max(len(self.doc_ids), 1)
        codebook_total = self.Kp * self.D * 4
        code_dtype_bytes = self.M * (self.codes.itemsize if self.codes is not None else 1)
        codebook_amort = codebook_total / n
        raw_total = codebook_amort + code_dtype_bytes
        ent_total = codebook_amort + self._entropy_bpd
        int8_tier = float(self.D) if self.v8 is not None else 0.0       # rerank tier: D int8 bytes/doc (+tiny scale)
        ivf_tier = (4.0 + self.nlist * self.D * 4.0 / n) if self.coarse is not None else 0.0  # cell-id + coarse cbk amort
        fp32 = self.D * 4
        return {
            "fp32_bytes_per_doc": float(fp32),
            "codebook_amort_bytes_per_doc": round(codebook_amort, 2),
            "codes_raw_bytes_per_doc": float(code_dtype_bytes),
            "codes_entropy_bytes_per_doc": round(self._entropy_bpd, 2),
            "hot_adc_bytes_per_doc": round(raw_total, 2),              # the searchable 32 B/doc ADC index (scan tier)
            "int8_rerank_bytes_per_doc": round(int8_tier, 2),         # rerank tier (can be cold/mmap; paged for top-C)
            "ivf_bytes_per_doc": round(ivf_tier, 2),                  # IVF scale tier (cell-id + amortized coarse cbk)
            "bytes_per_doc": round(raw_total + int8_tier + ivf_tier, 2),   # honest in-RAM total (holds .74/.96)
            "bytes_per_doc_entropy_floor": round(ent_total, 2),        # gamma/range-coder floor (hot codes only)
            "compression_vs_fp32": round(fp32 / (raw_total + int8_tier + ivf_tier), 2) if (raw_total + int8_tier + ivf_tier) else 0.0,
            "compression_vs_fp32_hot_only": round(fp32 / raw_total, 2) if raw_total else 0.0,
        }

    # ---- persistence (rotation = 0 bytes, regenerated from seed) ----
    def save(self, path: Union[str, Path]) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        arrays = dict(codes=self.codes, codebooks=self.codebooks,
                      doc_ids=np.array(self.doc_ids, dtype=object))
        if self.v8 is not None:                            # persist the int8 rerank tier (regenerated-free otherwise)
            arrays.update(v8=self.v8, v8_lo=self.v8_lo, v8_scale=self.v8_scale)
        if self.coarse is not None:                        # persist the IVF scale tier
            arrays.update(coarse=self.coarse, coarse_sqnorm=self.coarse_sqnorm,
                          inv_docs=self.inv_docs, inv_ptr=self.inv_ptr)
        np.savez_compressed(p / "codec.npz", **arrays)
        (p / "codec_meta.json").write_text(json.dumps({
            "M": self.M, "Kp": self.Kp, "rounds": self.rounds, "seed": self.seed,
            "nrot": self.nrot, "D": self.D, "sub": self.sub, "rerank_C": self.rerank_C,
            "entropy_bpd": self._entropy_bpd, "n_docs": len(self.doc_ids),
            "nlist": self.nlist, "nprobe": self.nprobe,
        }, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "DensePQCodec":
        p = Path(path)
        meta = json.loads((p / "codec_meta.json").read_text(encoding="utf-8"))
        self = cls(M=meta["M"], Kp=meta["Kp"], rounds=meta["rounds"], seed=meta["seed"], nrot=meta["nrot"],
                   rerank_C=int(meta.get("rerank_C", 500)))
        z = np.load(p / "codec.npz", allow_pickle=True)
        self.codes = np.ascontiguousarray(z["codes"])
        self.codebooks = np.ascontiguousarray(z["codebooks"], dtype=np.float32)
        self.doc_ids = [str(d) for d in z["doc_ids"]]
        self.D = int(meta["D"]); self.sub = int(meta["sub"]); self._entropy_bpd = float(meta["entropy_bpd"])
        if "v8" in z.files:                                # restore the int8 rerank tier if present
            self.v8 = np.ascontiguousarray(z["v8"]); self.v8_lo = z["v8_lo"]; self.v8_scale = z["v8_scale"]
        if "coarse" in z.files:                            # restore the IVF scale tier if present
            self.coarse = np.ascontiguousarray(z["coarse"]); self.coarse_sqnorm = np.ascontiguousarray(z["coarse_sqnorm"])
            self.inv_docs = np.ascontiguousarray(z["inv_docs"]); self.inv_ptr = np.ascontiguousarray(z["inv_ptr"])
            self.nlist = int(meta.get("nlist", len(self.inv_ptr) - 1)); self.nprobe = int(meta.get("nprobe", 1))
        self._rotate = make_free_rotation(self.D, rounds=self.rounds, seed=self.seed, nrot=self.nrot)
        return self


# =============================================================================================
# Optional CPU encoder  --  the one (CPU-only) model touch, for text queries without precomputed vectors
# =============================================================================================
class CpuEncoder:
    """Lightweight sentence-transformers encoder pinned to CPU. Loaded only if AETHOS_ENCODER is set and
    sentence-transformers is installed. Never reaches a GPU (CUDA is disabled at import)."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # optional dep; import lazily
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device="cpu")

    def encode(self, texts: List[str]) -> np.ndarray:
        v = self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, device="cpu")
        return np.ascontiguousarray(v, dtype=np.float32)


# =============================================================================================
# Glass-box lexical tier (optional, no-GPU)  --  named bridges + signal contributions for /explain
# =============================================================================================
class GlassBoxTier:
    """Wraps aethos_rag_pipeline.AethosPipeline over the same corpus to produce named, traceable bridges and
    per-signal contributions. Entirely no-GPU. Fully optional: if its deps are unavailable, the service still
    serves dense results and a dense-only explanation."""

    def __init__(self):
        self.pipe = None
        self.available = False
        self.learned = False
        try:
            from aethos_rag_pipeline import AethosPipeline
            self._Pipeline = AethosPipeline
            self.available = True
        except Exception as e:                             # heavy deps missing -> degrade gracefully
            self._err = str(e)

    def ingest(self, corpus: Dict[str, str]) -> None:
        if not self.available:
            return
        try:
            self.pipe = self._Pipeline(profile="balanced", latency_budget_ms=0.5).ingest(corpus)
        except Exception as e:
            self.available = False
            self._err = str(e)

    def learn(self, queries: Dict[str, str], qrels: Dict[str, Dict[str, int]]) -> None:
        if self.pipe is None:
            return
        try:
            self.pipe.learn(queries, qrels)
            self.learned = True
        except Exception as e:
            self._err = str(e)

    def explain(self, query: str, doc_id: str) -> Optional[Dict[str, Any]]:
        if self.pipe is None or not self.learned:
            return None                                    # bridges only meaningful after learn()
        try:
            return self.pipe.explain(query, doc_id)
        except Exception:
            return None


# =============================================================================================
# The RAG service core  --  Pitagora contract (create_rag / add_documents / retrieve) over the codec
# =============================================================================================
class AethosRAG:
    """Stateful, single-index RAG service. Holds the compressed dense codec (authoritative ranking), the
    document texts, an optional CPU encoder, and an optional glass-box lexical tier for /explain."""

    def __init__(self):
        self.codec: Optional[DensePQCodec] = None
        self.texts: Dict[str, str] = {}
        self.encoder: Optional[CpuEncoder] = None
        self.glass: GlassBoxTier = GlassBoxTier()
        self.query_embed_mode: str = "precomputed"         # 'precomputed' | 'cpu_encoder' | 'none'
        self._ingest_meta: Dict[str, Any] = {}
        self._maybe_load_encoder()
        self._maybe_load_persisted()

    # ---- boundary helpers ----
    def _maybe_load_encoder(self) -> None:
        name = os.environ.get("AETHOS_ENCODER", "").strip()
        if not name:
            return
        try:
            self.encoder = CpuEncoder(name)
            self.query_embed_mode = "cpu_encoder"
        except Exception:
            self.encoder = None                            # optional -> stay in precomputed mode

    def _maybe_load_persisted(self) -> None:
        try:
            if (INDEX_DIR / "codec_meta.json").exists():
                self.codec = DensePQCodec.load(INDEX_DIR)
                tp = INDEX_DIR / "texts.json"
                if tp.exists():
                    self.texts = json.loads(tp.read_text(encoding="utf-8"))
                self._ingest_meta = {"loaded_from_disk": True, "n_docs": len(self.codec.doc_ids)}
                # rebuild the (no-GPU) glass-box lattice from persisted texts so /explain works after restart
                if self.texts:
                    self.glass.ingest(self.texts)
        except Exception as e:
            self._ingest_meta = {"load_error": str(e)}

    def _embed_query(self, query: Optional[str], query_embedding: Optional[List[float]]) -> np.ndarray:
        if query_embedding is not None:
            return np.asarray(query_embedding, dtype=np.float32)
        if query is None:
            raise ValueError("provide either `query_embedding` (precomputed) or `query` text")
        if self.encoder is None:
            raise ValueError(
                "no CPU encoder configured: send `query_embedding` (precomputed path), or set the "
                "AETHOS_ENCODER env var to a sentence-transformers model to enable the optional CPU encoder")
        return self.encoder.encode([query])[0]

    # ---- Pitagora contract ----
    def create_rag(self, M: int = 32, Kp: int = 256, seed: int = 1234, rounds: int = 3) -> None:
        """Reset/initialize an empty index with the given PQ config (Pitagora create_rag)."""
        self.codec = DensePQCodec(M=M, Kp=Kp, rounds=rounds, seed=seed)
        self.texts = {}
        self.glass = GlassBoxTier()

    def add_documents(self, docs: List[Dict[str, Any]], *, M: int = 32, Kp: int = 256,
                      seed: int = 1234, rounds: int = 3,
                      queries: Optional[Dict[str, str]] = None,
                      qrels: Optional[Dict[str, Dict[str, int]]] = None) -> Dict[str, Any]:
        """Build (or rebuild) the index from docs (Pitagora add_documents == /ingest).

        Each doc: {"doc_id": str, "text": str, "embedding": [float]?}. Embeddings may be supplied
        (precomputed anywhere, even a GPU box offline) or computed by the optional CPU encoder. The index is
        frozen into PQ codes and persisted; serving never re-embeds a document."""
        if not docs:
            raise ValueError("no documents provided")
        doc_ids = [str(d["doc_id"]) for d in docs]
        texts = [d.get("text", "") for d in docs]
        if any("embedding" in d and d["embedding"] is not None for d in docs):
            emb = np.asarray([d["embedding"] for d in docs], dtype=np.float32)
        elif self.encoder is not None:
            emb = self.encoder.encode(texts)               # CPU-only; ingest-time model touch
        else:
            raise ValueError("docs need `embedding` values, or configure a CPU encoder (AETHOS_ENCODER) to "
                             "embed `text` at ingest. Serving stays no-GPU either way.")
        self.codec = DensePQCodec(M=M, Kp=Kp, rounds=rounds, seed=seed).build(doc_ids, emb)
        if len(doc_ids) >= DensePQCodec.IVF_MIN_DOCS:      # auto-enable the sublinear IVF scan only where it wins
            self.codec.build_ivf()
        self.texts = {i: t for i, t in zip(doc_ids, texts)}
        # persist codec + texts so a restart serves with no re-ingest
        self.codec.save(INDEX_DIR)
        (INDEX_DIR / "texts.json").write_text(json.dumps(self.texts, ensure_ascii=False), encoding="utf-8")
        # optional no-GPU glass-box lattice over the same corpus (for named bridges in /explain)
        self.glass = GlassBoxTier()
        self.glass.ingest(self.texts)
        if queries and qrels:
            self.glass.learn(queries, qrels)
        self._ingest_meta = {
            "loaded_from_disk": False,
            "n_docs": len(doc_ids),
            "embedding_source": "precomputed" if any("embedding" in d for d in docs) else "cpu_encoder",
            "glass_box_available": self.glass.available,
            "glass_box_learned": self.glass.learned,
        }
        return self.stats()

    def retrieve(self, query: Optional[str] = None, k: int = 10,
                 query_embedding: Optional[List[float]] = None) -> List[Dict[str, Any]]:
        """Ranked retrieval (Pitagora retrieve == /search). Returns [{doc_id, score, text}]. No GPU."""
        if self.codec is None:
            raise RuntimeError("index is empty -- call add_documents/ingest first")
        q = self._embed_query(query, query_embedding)
        idx, scores = self.codec.search(q, k=k)            # PQ-ADC recall -> exact int8 rerank (holds .74/.96)
        out = []
        for i, s in zip(idx, scores):
            did = self.codec.doc_ids[int(i)]
            out.append({"doc_id": did, "score": float(s), "text": self.texts.get(did, "")})
        return out

    # ---- glass-box explain ----
    def explain(self, query: Optional[str], doc_id: str,
                query_embedding: Optional[List[float]] = None) -> Dict[str, Any]:
        if self.codec is None:
            raise RuntimeError("index is empty -- call add_documents/ingest first")
        doc_id = str(doc_id)
        if doc_id not in self.codec.doc_ids:
            return {"query": query, "doc_id": doc_id, "error": "doc_id not in index"}
        di = self.codec.doc_ids.index(doc_id)
        signals: Dict[str, Any] = {}
        factors: Dict[str, Any] = {}
        bridges_fired: List[Dict[str, Any]] = []
        # dense factor is available whenever we can embed the query; otherwise fall back to lexical-only glass-box
        try:
            q = self._embed_query(query, query_embedding)
            scores = self.codec.score(q)
            dense_signal = float(scores[di])
            signals["dense_pq"] = round(dense_signal, 5)
            factors.update({
                "dense_pq_score": round(dense_signal, 5),
                "dense_rank": int((scores > scores[di]).sum()),
                "dense_max": round(float(scores.max()), 5),
                "dense_share_of_max": round(dense_signal / (float(scores.max()) or 1.0), 4),
            })
        except ValueError:
            if query is None:                              # nothing to explain with
                raise
            factors["dense_pq_score"] = "unavailable (no query embedding / encoder); lexical glass-box below"
        # optional lexical glass-box tier: named, traceable bridges + lexical/drift/splade contributions
        gb = self.glass.explain(query, doc_id) if query is not None else None
        if gb is not None and "error" not in gb:
            signals.update(gb.get("signals", {}))
            bridges_fired = gb.get("bridges_fired", [])
            factors["lexical_config"] = gb.get("config", {})
            factors["mv_triangles_fired"] = gb.get("mv_triangles_fired", [])
            factors["glass_box"] = "lexical-lattice (named bridges, no GPU)"
        else:
            factors["glass_box"] = ("dense-only (lexical tier unavailable or query embedding-only); "
                                    "send `query` text and ingest with queries+qrels for named bridges")
        return {"query": query, "doc_id": doc_id, "signals": signals,
                "bridges_fired": bridges_fired, "factors": factors}

    # ---- stats ----
    def stats(self) -> Dict[str, Any]:
        if self.codec is None:
            return {"n_docs": 0, "gpu_at_serve": False, "query_embed_mode": self.query_embed_mode,
                    "index": "empty"}
        bpd = self.codec.bytes_per_doc()
        return {
            "n_docs": len(self.codec.doc_ids),
            "bytes_per_doc": bpd["bytes_per_doc"],
            "gpu_at_serve": False,                         # invariant: serve path is pure-numpy ADC
            "compression_vs_fp32": bpd["compression_vs_fp32"],
            "query_embed_mode": self.query_embed_mode,     # 'precomputed' | 'cpu_encoder'
            "footprint": bpd,
            "pq": {"M": self.codec.M, "Kp": self.codec.Kp, "dim": self.codec.D,
                   "rotation": "free octahedral+perm, 0 stored bytes (regenerated from seed)"},
            "glass_box": {"available": self.glass.available, "learned": self.glass.learned},
            "ingest": self._ingest_meta,
            "index_dir": str(INDEX_DIR),
            "honest_scope": ("compression = Product Quantization (competitive, standard). AETHOS moat = "
                             "glass-box explain + no-GPU-at-serve + invertible/regenerable codec, NOT a "
                             "proprietary ratio."),
        }


# =============================================================================================
# FastAPI app
# =============================================================================================
def build_app():
    from fastapi import FastAPI, HTTPException

    app = FastAPI(
        title="AETHOS RAG API",
        version="1.0",
        description="No-GPU-at-serve, glass-box, PQ-compressed dense RAG. Pitagora drop-in contract.",
    )
    rag = AethosRAG()

    def _guard(fn):
        try:
            return fn()
        except (ValueError, RuntimeError) as e:
            raise HTTPException(status_code=400, detail=str(e))

    # ---- endpoints ----
    @app.get("/healthz")
    def healthz():
        return {"status": "ok", "gpu_at_serve": False}

    @app.post("/ingest")
    def ingest(body: IngestIn):
        return _guard(lambda: rag.add_documents(
            [d.model_dump() for d in body.docs], M=body.M, Kp=body.Kp, seed=body.seed,
            rounds=body.rounds, queries=body.queries, qrels=body.qrels))

    @app.post("/search")
    def search(body: SearchIn):
        return {"results": _guard(lambda: rag.retrieve(body.query, body.k, body.query_embedding))}

    @app.post("/explain")
    def explain(body: ExplainIn):
        return _guard(lambda: rag.explain(body.query, body.doc_id, body.query_embedding))

    @app.get("/stats")
    def stats():
        return rag.stats()

    # ---- Pitagora drop-in aliases (distinct wrappers so each route binds its own body model) ----
    @app.post("/create_rag")                               # create_rag+add_documents == ingest in one call
    def create_rag(body: IngestIn):
        return ingest(body)

    @app.post("/add_documents")
    def add_documents(body: IngestIn):
        return ingest(body)

    @app.post("/retrieve")
    def retrieve(body: SearchIn):
        return search(body)

    return app


# module-level ASGI app for `uvicorn aethos_rag_api:app`
try:
    app = build_app()
except Exception as _e:                                    # fastapi missing -> importing the module still works
    app = None
    _APP_ERR = str(_e)


# =============================================================================================
# Self-contained smoke test (synthetic embeddings; no BEIR download, no network, no GPU)
# =============================================================================================
def _smoke():
    print("=" * 88)
    print("SMOKE — AethosRAG end-to-end on synthetic 128-d embeddings (no GPU, no network)")
    print("=" * 88)
    rng = np.random.default_rng(0)
    D, N = 128, 300
    # make 5 topical clusters so retrieval is meaningful
    centers = rng.normal(size=(5, D)).astype(np.float32)
    docs = []
    for i in range(N):
        c = centers[i % 5] + 0.35 * rng.normal(size=D).astype(np.float32)
        docs.append({"doc_id": f"doc{i}", "text": f"cluster {i % 5} document number {i}",
                     "embedding": c.tolist()})
    rag = AethosRAG()
    rag.create_rag(M=32, Kp=256)
    st = rag.add_documents(docs, M=32, Kp=256)
    print(f"  ingest: {st['n_docs']} docs, {st['bytes_per_doc']} B/doc, "
          f"{st['compression_vs_fp32']}x vs fp32, gpu_at_serve={st['gpu_at_serve']}")

    # query = a point near cluster-2 center -> should retrieve cluster-2 docs (precomputed-embedding path)
    qv = (centers[2] + 0.1 * rng.normal(size=D).astype(np.float32)).tolist()
    t = time.perf_counter()
    res = rag.retrieve(query_embedding=qv, k=5)
    ms = (time.perf_counter() - t) * 1000
    print(f"  search (precomputed q): {len(res)} hits in {ms:.3f} ms; "
          f"top texts={[r['text'] for r in res[:3]]}")

    ex = rag.explain(query=None, doc_id=res[0]["doc_id"], query_embedding=qv)
    print(f"  explain(top1): signals={ex['signals']} factors.dense_rank={ex['factors']['dense_rank']} "
          f"bridges_fired={len(ex['bridges_fired'])}")

    # persistence round-trip: reload from disk, re-serve, confirm identical top-1
    rag2 = AethosRAG()                                     # __init__ loads the persisted codec
    res2 = rag2.retrieve(query_embedding=qv, k=5)
    same = res2[0]["doc_id"] == res[0]["doc_id"]
    print(f"  persist round-trip: reloaded {rag2.stats()['n_docs']} docs from {INDEX_DIR.name}; "
          f"top-1 identical={same}")
    print("\n  SMOKE OK — create_rag/add_documents(ingest)/retrieve(search)/explain/stats + persist, "
          "encoder-free, no GPU.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        import uvicorn
        uvicorn.run("aethos_rag_api:app", host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
    else:
        _smoke()
