"""
aethos_lattice_retriever.py - drop-in retriever that exposes Timothy's CURRENT
lattice retrieval (in "C:/Users/wynos/New folder (3)") through the exact contract
Andrea's Pitagora BEIR harness expects (see beir_encoder.py, the canonical
template that sits next to this file):

    add_documents(self, documents: List[str], metadata: Optional[List[dict]] = None) -> None
    retrieve(self, query: str, top_k: int = 10, **kwargs) -> List[Tuple[str, float, dict]]
    create_*_retriever(...)                                  # factory

benchmark_beir.py calls  retriever.add_documents(doc_texts, doc_metadata)  then
retriever.retrieve(query_text, top_k=...)  and reads back  (doc_text, score, meta)
sorted descending.  The meta dicts carry {"doc_id": ..., "beir_id": ...} for
scoring; this adapter PRESERVES and RETURNS exactly the meta that was added with
each document (doc-id alignment is kept by internal ordinal, never by score).

TWO SELECTABLE BACKENDS (Timothy chose "both, selectable"):

  backend="algebraic"  (DEFAULT - self-contained, no extra model)
      Wraps aethos_algebraic_corpus.AlgebraicCorpus: the corpus-IS-a-number
      retrieval (FTA composites, primes in idf-rank order = WAND key, BM25 over
      the prime-posting lattice, warm corridor expansion).  add_documents ->
      add(ordinal, text) for each doc + build(); retrieve -> query(q, k, T) and
      return (text, score, meta).
      AlgebraicCorpus.query() returns ranked ids ONLY (no scores).  We therefore
      ALSO compute the real BM25 score off the corpus's own _bm25 primitive
      (easily available) and return that as the float so callers get a true,
      monotone score; if BM25 is somehow unavailable we fall back to
      rank-descending scores (top_k - rank).  nDCG only needs the order either
      way.  Default temperature is warm (the recall default).

  backend="splade"  (SOTA - needs the SPLADE encoder)
      Reuses the proven logic in
      "C:/Users/wynos/New folder (3)/_route2_splade_lattice.py":
      naver/splade-cocondenser-ensembledistil, HF_HUB offline, fp16, learned
      sparse term weights, an inverted index (term -> {doc: weight}) = the
      lattice sparse posting store, served by the sparse dot (= the meet), NO
      query-time cross-encoder.  retrieve returns the REAL sparse-dot scores.

Factory:  create_lattice_retriever(backend="algebraic", **kwargs)

This file ONLY ADDS an adapter; it imports Timothy's modules by sys.path-inserting
his repo, and does not modify any of his files or Andrea's files.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

# --- make Timothy's CURRENT retrieval importable -----------------------------
TIMOTHY_REPO = r"C:/Users/wynos/New folder (3)"
if TIMOTHY_REPO not in sys.path:
    sys.path.insert(0, TIMOTHY_REPO)


Hit = Tuple[str, float, Dict[str, Any]]


# =============================================================================
# Backend 1: ALGEBRAIC  (default, self-contained)
# =============================================================================
class _AlgebraicBackend:
    """Wrap aethos_algebraic_corpus.AlgebraicCorpus.

    Internal ordinals (0..N-1) are the AlgebraicCorpus doc ids, kept separate
    from whatever doc_id/beir_id the caller put in metadata.  We always return
    the original text + the original meta for each hit -> alignment is exact and
    independent of the score.
    """

    def __init__(self, warm: bool = True, temperature: Optional[float] = None, **kwargs: Any):
        from aethos_algebraic_corpus import AlgebraicCorpus  # Timothy's module

        self._AlgebraicCorpus = AlgebraicCorpus
        # temperature: None -> use the corpus warm default; warm=False -> cold (T=0)
        if temperature is not None:
            self._T: Optional[float] = float(temperature)
        elif not warm:
            self._T = 0.0
        else:
            self._T = None  # AlgebraicCorpus.query(T=None) uses its warm default

        self._corpus = None
        self._docs: List[str] = []
        self._meta: List[Dict[str, Any]] = []
        self._scored_bm25 = False  # whether retrieve returned real BM25 scores

    def add_documents(self, documents: List[str], metadata: Optional[List[Dict[str, Any]]] = None) -> None:
        if metadata is None:
            metadata = [{} for _ in documents]
        if len(metadata) != len(documents):
            raise ValueError("metadata length must match documents length")
        self._docs = list(documents)
        self._meta = [dict(m) for m in metadata]
        ac = self._AlgebraicCorpus()
        for ordinal, text in enumerate(self._docs):
            ac.add(ordinal, text or " ")  # internal ordinal == index into _docs/_meta
        ac.build()  # assign primes rarest-first, materialize the composite numbers
        self._corpus = ac

    def retrieve(self, query: str, top_k: int = 10, **kwargs: Any) -> List[Hit]:
        if self._corpus is None or not self._docs:
            return []
        T = kwargs.get("T", self._T)
        ac = self._corpus

        # AlgebraicCorpus.query returns ranked ordinals only (no scores).  We
        # ALSO pull the real BM25 scores off the corpus's own primitive so the
        # float we return is a genuine, monotone relevance score (the task asks
        # to expose real BM25 if easily available - it is).
        ranked = ac.query(query, k=top_k, T=T)  # list of internal ordinals, best-first

        bm25_scores: Dict[int, float] = {}
        try:
            qp = ac._query_primes(query)
            if qp:
                # warm path expands the query along the regenerated corridor before
                # scoring; mirror that so the score matches the ranking the user got.
                if T is None:
                    T_eff = ac.warm_T
                else:
                    T_eff = float(T)
                if T_eff > 0.0:
                    expanded = dict(qp)
                    for p in list(qp):
                        for dp, w in ac.correlated_terms(p, top=ac.warm_expand, min_pdt=ac.warm_min_pdt):
                            add = T_eff * w / (1.0 + ac._idf(p))
                            if add > 0:
                                expanded[dp] = expanded.get(dp, 0.0) + add
                    bm25_scores = dict(ac._bm25(expanded))
                else:
                    bm25_scores = dict(ac._bm25(qp))
                self._scored_bm25 = True
        except Exception:
            bm25_scores = {}
            self._scored_bm25 = False

        out: List[Hit] = []
        n = len(ranked)
        for rank, ordinal in enumerate(ranked):
            if ordinal in bm25_scores:
                score = float(bm25_scores[ordinal])
            else:
                # fallback: rank-descending score (order is all nDCG needs)
                score = float(top_k - rank) if top_k else float(n - rank)
            out.append((self._docs[ordinal], score, self._meta[ordinal]))
        # already best-first from query(); keep stable but enforce desc by score
        out.sort(key=lambda h: h[1], reverse=True)
        return out

    @property
    def scored_with_bm25(self) -> bool:
        return self._scored_bm25


# =============================================================================
# Backend 2: SPLADE on the lattice sparse index  (SOTA, needs the encoder)
# =============================================================================
class _SpladeBackend:
    """Reuse the proven logic in Timothy's _route2_splade_lattice.py:
    SPLADE-encode docs+query into sparse term weights, build the inverted index
    (term -> {doc: weight}) = the lattice sparse posting store, serve with the
    sparse dot (= the meet).  Returns REAL scores.  NO query-time cross-encoder.
    """

    def __init__(
        self,
        model: str = "naver/splade-cocondenser-ensembledistil",
        max_len_doc: int = 256,
        max_len_query: int = 64,
        batch: int = 64,
        **kwargs: Any,
    ):
        # Offline + symlink-safe HF, exactly as the route2 module expects.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
        if "SPLADE_MODEL" not in os.environ:
            os.environ["SPLADE_MODEL"] = model

        # Import Timothy's route2 module and reuse splade_batch / coo_to_arrays / search.
        import importlib.util

        route2_path = os.path.join(TIMOTHY_REPO, "_route2_splade_lattice.py")
        spec = importlib.util.spec_from_file_location("_aethos_route2_splade_lattice", route2_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"could not load route2 module from {route2_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # this triggers `import torch/transformers`
        self._route2 = mod

        self._max_len_doc = max_len_doc
        self._max_len_query = max_len_query
        self._batch = batch

        self._docs: List[str] = []
        self._meta: List[Dict[str, Any]] = []
        self._term_docs: Dict[int, Any] = {}
        self._term_wts: Dict[int, Any] = {}
        self._n_docs = 0

    def add_documents(self, documents: List[str], metadata: Optional[List[Dict[str, Any]]] = None) -> None:
        import numpy as np

        if metadata is None:
            metadata = [{} for _ in documents]
        if len(metadata) != len(documents):
            raise ValueError("metadata length must match documents length")
        self._docs = list(documents)
        self._meta = [dict(m) for m in metadata]
        texts = [d or " " for d in self._docs]

        rows_l, cols_l, vals_l = [], [], []
        for i in range(0, len(texts), self._batch):
            reps = self._route2.splade_batch(texts[i:i + self._batch], max_len=self._max_len_doc)
            for j, (idx, val) in enumerate(reps):
                row = i + j
                cols_l.append(idx)
                vals_l.append(val)
                rows_l.append(np.full(len(idx), row, dtype=np.int32))
        if rows_l:
            rows = np.concatenate(rows_l)
            cols = np.concatenate(cols_l)
            vals = np.concatenate(vals_l)
            self._term_docs, self._term_wts = self._route2.coo_to_arrays(rows, cols, vals)
        else:
            self._term_docs, self._term_wts = {}, {}
        self._n_docs = len(texts)

    def retrieve(self, query: str, top_k: int = 10, **kwargs: Any) -> List[Hit]:
        if self._n_docs == 0:
            return []
        qrep = self._route2.splade_batch([query or " "], max_len=self._max_len_query)[0]
        k = min(top_k, self._n_docs)
        top, scores = self._route2.search(qrep, self._term_docs, self._term_wts, self._n_docs, k=k)
        out: List[Hit] = []
        for ordinal, score in zip(top, scores):
            o = int(ordinal)
            out.append((self._docs[o], float(score), self._meta[o]))
        out.sort(key=lambda h: h[1], reverse=True)
        return out[:top_k]

    @property
    def scored_with_bm25(self) -> bool:
        return False  # SPLADE returns real sparse-dot scores, not BM25


# =============================================================================
# Public retriever: the contract class the harness imports
# =============================================================================
class AethosLatticeRetriever:
    """Drop-in retriever exposing Timothy's lattice retrieval to Andrea's harness.

    backend="algebraic" (default) or backend="splade".
    Same interface as BeirEncoderRetriever / Aethos13RAG:
        add_documents(documents, metadata) ; retrieve(query, top_k) -> [(text, score, meta)].
    """

    def __init__(self, backend: str = "algebraic", **kwargs: Any):
        backend = (backend or "algebraic").lower()
        if backend == "algebraic":
            self._impl: Any = _AlgebraicBackend(**kwargs)
        elif backend == "splade":
            self._impl = _SpladeBackend(**kwargs)
        else:
            raise ValueError(f"unknown backend {backend!r}; use 'algebraic' or 'splade'")
        self.backend = backend

    def add_documents(self, documents: List[str], metadata: Optional[List[Dict[str, Any]]] = None) -> None:
        self._impl.add_documents(documents, metadata)

    def retrieve(self, query: str, top_k: int = 10, **kwargs: Any) -> List[Hit]:
        return self._impl.retrieve(query, top_k=top_k, **kwargs)

    @property
    def scored_with_bm25(self) -> bool:
        return getattr(self._impl, "scored_with_bm25", False)


def create_lattice_retriever(backend: str = "algebraic", **kwargs: Any) -> AethosLatticeRetriever:
    """Create a lattice retriever for Andrea's Pitagora harness.

    backend="algebraic" (default, no extra model) or backend="splade" (SOTA,
    needs naver/splade-cocondenser-ensembledistil).  Extra kwargs are forwarded
    to the backend (algebraic: warm/temperature; splade: model/max_len/batch).
    """
    return AethosLatticeRetriever(backend=backend, **kwargs)
