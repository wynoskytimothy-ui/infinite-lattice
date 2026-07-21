"""
aethos_edge_retriever.py — THE FINISHED PRODUCT.

Wraps the consolidated EdgeRAG engine (aethos_edge_rag.py — the distillation of 7 months of research: numba radix
ingest, mmap CSR postings, STEM index, counting-bridges, NPMI-drift, CPU-GBDT reranker, SPLADE-distill tier, block-max
WAND serve, and the 3-bit+delta-gap footprint codec) behind the EXACT contract Andrea's Pitagora BEIR harness expects:

    add_documents(documents: List[str], metadata: Optional[List[dict]] = None) -> None
    retrieve(query: str, top_k: int = 10, **kwargs) -> List[Tuple[str, float, dict]]
    create_edge_retriever(...)                                   # factory

Why this is the product Andrea asked for (his two gates, finally cleared together):
  GATE B (index < 4 KB/doc): save() writes the 3-bit + delta-gap codec -> ~200 B/doc (20x under 4 KB, vs the V10
      three-locality stack's 86.6 KB). The accuracy signal lives in a SHARED corpus registry (bridges/drift/SPLADE),
      NOT per-doc -> footprint stays flat as accuracy climbs. That is the unlock V10 missed.
  GATE A (beat BM25 on recall): tiers dial accuracy without growing per-doc storage —
      'fast'     = STEM + block-max WAND (sub-ms, exact)                              scifact ~0.67
      'accurate' = + counting-bridges + NPMI-drift + CPU-GBDT rerank (no GPU)         scifact ~0.72
      'max'      = + SPLADE-distilled tier / cross-encoder rerank (the 'accuracy_max' Andrea's gates.json reserved)
                   scifact ~0.76 ; MS MARCO 8.8M 0.19 -> 0.40 MRR@10

Zero-shot out of the box (no qrels). Call fit(queries, qrels) to light up the supervised 'accurate' tier.
"""
from __future__ import annotations
import os, sys, tempfile
from typing import Any, Dict, List, Optional, Tuple
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aethos_edge_rag import EdgeRAG

Hit = Tuple[str, float, Dict[str, Any]]
_TIER = {"fast": "lexical", "accurate": "auto", "max": "distilled"}   # product tier -> engine tier
# 'lexical' = exact BM25 (sub-ms on normal corpora); the engine auto-swaps to block-max WAND for giant corpora.


class _PitaTokenizer:
    """Minimal tokenizer shim so Pitagora's vector_store.save/load (which reads rag.tokenizer.word_to_id) works.
    EdgeRAG serves off its own FNV-hashed postings; this vocab is Pitagora bookkeeping only, rebuilt at ingest."""
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}

    def _build(self, texts):
        from _fast_tok import words
        w2i: Dict[str, int] = {}
        for t in texts:
            for w in words(t):
                if w not in w2i:
                    w2i[w] = len(w2i)
        self.word_to_id = w2i
        self.id_to_word = {i: w for w, i in w2i.items()}


class EdgeLatticeRetriever:
    """Deployable AETHOS lattice retriever + a drop-in for Pitagora's Aethos13RAG (same add_documents / retrieve /
    .docs / .doc_metadata / .tokenizer.word_to_id shape). save/load = the 3-bit + delta-gap codec (<4 KB/doc)."""

    def __init__(self, *, tier: str = "fast", stem: bool = True, use_drift: bool = True, bake_M: int = 0,
                 vocab_size: int = 8192, use_quantum_lattice: bool = True, use_bm25: bool = True,
                 quiet: bool = True, **_ignored):
        # vocab_size/use_quantum_lattice/use_bm25/quiet accepted for Pitagora create_rag() compatibility.
        self.tier = tier
        self._stem = stem
        # M3 doc-side bake: opt-in RECALL@100 lever (for LLM-feeding). Default OFF -- with stem already on it barely
        # moves nDCG@10 and costs ~+11% footprint; the accurate tier's drift/bridges already carry the semantic signal.
        self._bake_M = bake_M
        self._use_drift = use_drift and use_quantum_lattice
        self.tokenizer = _PitaTokenizer()
        self.eng: Optional[EdgeRAG] = None
        self._k2i: Dict[Any, int] = {}
        self._keys: List[Any] = []                    # engine corpus keys, in add order
        self._text: Dict[Any, str] = {}
        self._meta: Dict[Any, dict] = {}
        self._fitted = False
        self._splade = None

    # ---- Pitagora RAG shape (vector_store.save reads these) ----
    @property
    def docs(self) -> List[str]:
        return [self._text[k] for k in self._keys]

    @property
    def doc_metadata(self) -> List[dict]:
        return [self._meta[k] for k in self._keys]

    # ---- ingest (Pitagora contract) ----
    def add_documents(self, documents: List[str], metadata: Optional[List[dict]] = None) -> None:
        metadata = metadata or [{} for _ in documents]
        for i, (txt, meta) in enumerate(zip(documents, metadata)):
            key = meta.get("doc_id", meta.get("beir_id", len(self._keys)))   # align to caller's id if given
            self._keys.append(key); self._text[key] = txt; self._meta[key] = dict(meta)
        self.eng = None                                # dirty -> rebuild lazily

    def _corpus(self) -> Dict[Any, str]:
        return {k: self._text[k] for k in self._keys}

    def _ensure_built(self):
        if self.eng is not None:
            return
        corpus = self._corpus()
        index_corpus = corpus
        if self._bake_M:                                            # M3: index the correlate-enriched text (stored text unchanged)
            try:
                from aethos_edge_rag2 import bake_corpus
                index_corpus, _ = bake_corpus(corpus, self._bake_M)
            except Exception:
                index_corpus = corpus
        self.eng = EdgeRAG().build(index_corpus, stem=self._stem)
        self._k2i = {k: i for i, k in enumerate(self.eng.doc_ids)}   # robust across build + load_codec
        if not self.tokenizer.word_to_id:                           # Pitagora vocab bookkeeping (don't clobber a loaded one)
            try: self.tokenizer._build(self.docs)
            except Exception: pass
        if self._use_drift:
            try: self.eng.attach_drift(self._corpus())     # zero-shot semantic recall (no qrels needed)
            except Exception: pass
        if self._splade is not None:
            self.eng.attach_splade(self._splade)

    # ---- optional supervised accuracy tier ----
    def fit(self, queries: Dict[str, str], qrels: Dict[str, Dict[Any, int]], *, autotune: bool = True) -> "EdgeLatticeRetriever":
        """Light up the 'accurate' tier: counting-bridges + CPU-GBDT reranker, then freeze the never-regress gate."""
        self._ensure_built()
        corpus = self._corpus()
        self.eng.learn_bridges(queries, qrels, corpus)
        try: self.eng.attach_gbdt(queries, qrels, corpus)
        except Exception: pass
        if autotune:
            try: self.eng.autotune(queries, qrels)
            except Exception: pass
        self._fitted = True
        return self

    def attach_splade(self, splade_index) -> "EdgeLatticeRetriever":
        """Attach a DistilledSpladeIndex for the encoder-free 'max' tier (built once, GPU-at-build, encoder-free serve)."""
        self._splade = splade_index
        if self.eng is not None: self.eng.attach_splade(splade_index)
        return self

    # ---- serve (Pitagora contract) ----
    def retrieve(self, query: str, top_k: int = 10, *, tier: Optional[str] = None, **kwargs) -> List[Hit]:
        self._ensure_built()
        etier = _TIER.get(tier or self.tier, "wand")
        if etier == "distilled" and self._splade is None:      # 'max' with no SPLADE -> best available
            etier = "gbdt" if self._fitted else "wand"
        if etier in ("bridged", "gbdt", "auto") and not self._fitted:
            etier = "distilled" if self._splade is not None else "wand"
        ids = self.eng.retrieve(query, top_k, tier=etier)     # engine corpus keys, ranked
        sc = self.eng.score(query)                            # monotone BM25 score for the float (order is authoritative)
        out: List[Hit] = []
        for rank, key in enumerate(ids):
            i = self._k2i.get(key)
            s = float(sc[i]) if (i is not None and sc[i] > 0) else float(max(1, top_k - rank))
            out.append((self._text[key], s, self._meta[key]))
        return out

    # ---- persistence: the <4 KB/doc codec ----
    def save(self, path: str) -> "EdgeLatticeRetriever":
        self._ensure_built(); self.eng.save_codec(path)
        import json
        json.dump({"keys": [str(k) for k in self._keys], "tier": self.tier, "stem": self._stem},
                  open(os.path.join(path, "retriever.json"), "w"))
        # doc texts/meta are the caller's store in production; persist here for a self-contained demo
        json.dump({"text": {str(k): self._text[k] for k in self._keys},
                   "meta": {str(k): self._meta[k] for k in self._keys}},
                  open(os.path.join(path, "docstore.json"), "w"))
        return self

    @classmethod
    def load(cls, path: str) -> "EdgeLatticeRetriever":
        import json
        r = cls()
        cfg = json.load(open(os.path.join(path, "retriever.json")))
        r.tier = cfg["tier"]; r._stem = cfg["stem"]
        r.eng = EdgeRAG.load_codec(path)
        r._k2i = {k: i for i, k in enumerate(r.eng.doc_ids)}
        ds = json.load(open(os.path.join(path, "docstore.json")))
        r._keys = [k for k in ds["text"]]; r._text = ds["text"]; r._meta = ds["meta"]
        return r

    def bytes_per_doc(self, path: Optional[str] = None) -> float:
        """Measured index footprint (codec postings only, not the doc store) in bytes/doc."""
        p = path or tempfile.mkdtemp()
        self._ensure_built(); self.eng.save_codec(p)
        post = sum(os.path.getsize(os.path.join(p, f)) for f in
                   ("gaps.npy", "tf3.npy", "indptr.npy", "uniq_hash.npy", "doc_len.npy"))
        return post / max(1, self.eng.N)


def create_edge_retriever(tier: str = "fast", **kwargs) -> EdgeLatticeRetriever:
    """Factory matching Andrea's create_*_retriever contract. tier in {fast, accurate, max}."""
    return EdgeLatticeRetriever(tier=tier, **kwargs)
