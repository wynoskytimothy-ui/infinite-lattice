# Wiring the AETHOS lattice retriever into Pitagora (Andrea's agentic RAG)

Repo: `wynoskytimothy-ui/prime_hotel`, branch `pitagora` (or `andrea`), folder `pitagora/`.

## Why it's a true drop-in
Pitagora's backend contract (`pitagora/rag_api.py`) is:
- `add_documents(documents: List[str], metadata: Optional[List[dict]]) -> None`
- `retrieve(query: str, top_k: int) -> List[Tuple[str, float, dict]]`  (text, score, metadata)

`aethos_lattice_retriever.py` implements **exactly this** (verified: `add_documents` line 90/204/271, `retrieve` line 103/232/274 returning `List[Tuple[str, float, dict]]`, factory `create_lattice_retriever(backend=...)` line 282). It already passed through Andrea's own `pitagora/benchmark_beir.py`. So it slots into the same place as `create_beir_retriever()` with no interface changes to the app.

## Step 1 — copy the retriever into the repo
Copy into `prime_hotel/` (project root, so `pitagora/rag_api.py`'s `sys.path` root import finds it):
- `aethos_lattice_retriever.py`  (the adapter)
- its backend(s): `aethos_algebraic_corpus.py` (CPU/glass-box `algebraic` backend) and/or `_route2_splade_lattice.py` + the SPLADE index for the `splade` backend.

## Step 2 — add a lattice backend to `pitagora/rag_api.py`
Append:
```python
def create_lattice_rag(backend: str = "algebraic", **kwargs):
    """AETHOS prime-lattice retriever. Same add_documents/retrieve contract as Aethos13RAG.
    backend="algebraic" = CPU, no GPU, glass-box (every score traces to terms);
    backend="splade"    = best accuracy (needs torch + the SPLADE index)."""
    from aethos_lattice_retriever import create_lattice_retriever
    return create_lattice_retriever(backend=backend, **kwargs)


def reindex_lattice(data_dir=None, backend: str = "algebraic"):
    """Mirror of reindex_beir() but backed by the prime lattice."""
    try:
        from pitagora.vector_store import get_data_dir, load as load_vector_store
    except ImportError:
        from vector_store import get_data_dir, load as load_vector_store
    from pathlib import Path
    d = Path(data_dir) if data_dir is not None else get_data_dir()
    rag = create_lattice_rag(backend=backend)
    loaded = load_vector_store(d); n = 0
    if loaded:
        documents, metadata, _ = loaded
        n = len(documents)
        rag.add_documents(documents, metadata)
    return rag, n
```

## Step 3 — expose it in the app
In `pitagora/app.py`, wherever the retrieval mode is selected (it already switches between `Aethos13RAG` and the BEIR encoder), add a `"lattice"` option that calls `reindex_lattice()`. The agentic two-stage flow (topic → rank) is unchanged — it only ever calls `add_documents` / `retrieve`.

## What this buys Andrea (vs the current Aethos13 backend)
- **287 B/doc** index on MARCO-scale (the production-gate number) — and shrinking further (footprint work in progress).
- **Glass-box**: every score decomposes into the matching terms (+ the 32-chamber region), so the agentic ranker can explain *why* a doc ranked — a capability dense encoders can't give.
- **`algebraic` backend needs no GPU** — runs in the same CPU process as the Flask app; `splade` backend for max accuracy when a GPU is present.
- Invertible / append-only index (add a doc = one update, no full reindex).

## Note
The exact backend index (codec/footprint) is being optimized right now (the architect footprint pass). The adapter above is stable regardless — it just points at whichever index that work settles on.
