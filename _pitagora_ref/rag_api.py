"""
Pitagora RAG API: create and use the app's retrieval backend.

Use this module from the Flask app and from BEIR benchmarks so both share
the same RAG config and interface.

Interface:
  - create_rag(**kwargs) -> RAG instance (add_documents, retrieve)
  - init_rag(data_dir=None) -> RAG, optionally loaded from persisted state
"""

import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple

# Ensure project root is importable (run from pitagora/ or from project root)
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

try:
    from rag.test_aethos13_rag import Aethos13RAG
except ImportError:
    from test_aethos13_rag import Aethos13RAG


def get_default_config() -> dict:
    """Return the default RAG API configuration (same values used by create_rag)."""
    return {
        "vocab_size": 8192,
        "use_quantum_lattice": True,
        "use_bm25": True,
    }


def create_rag(
    vocab_size: int = 8192,
    use_quantum_lattice: bool = True,
    use_bm25: bool = True,
    **kwargs: Any,
) -> Any:
    """
    Create the Pitagora RAG with the app's default config.

    Returns an instance with:
      - add_documents(documents: List[str], metadata: Optional[List[dict]]) -> None
      - retrieve(query: str, top_k: int = 3, ...) -> List[Tuple[str, float, dict]]

    Extra kwargs are passed to Aethos13RAG (e.g. quiet=True for benchmarks).
    """
    return Aethos13RAG(
        vocab_size=vocab_size,
        use_quantum_lattice=use_quantum_lattice,
        use_bm25=use_bm25,
        **kwargs,
    )


def init_rag(data_dir: Optional[Path] = None) -> Any:
    """
    Create RAG and optionally load persisted state from data_dir.

    Uses saved RAG config from data_dir/config.json if present; otherwise defaults.
    If data_dir has documents.json and vocab.json, restores documents and metadata.
    """
    try:
        from pitagora.vector_store import get_data_dir, load as load_vector_store, load_rag_config
    except ImportError:
        from vector_store import get_data_dir, load as load_vector_store, load_rag_config

    dir_to_use = Path(data_dir) if data_dir is not None else get_data_dir()
    rag_config = load_rag_config(dir_to_use)
    if rag_config:
        rag = create_rag(**{k: v for k, v in rag_config.items() if k in ("vocab_size", "use_quantum_lattice", "use_bm25")})
    else:
        rag = create_rag()
    loaded = load_vector_store(dir_to_use)
    if loaded:
        documents, metadata, vocab = loaded
        rag.tokenizer.word_to_id = vocab
        rag.tokenizer.id_to_word = {v: k for k, v in vocab.items()}
        rag.add_documents(documents, metadata)
    return rag


def reindex_rag(data_dir: Optional[Path] = None) -> Tuple[Any, int]:
    """
    Re-index all documents from the persisted vector store.

    Loads documents, metadata, and vocab from data_dir, builds a fresh RAG
    with those documents (rebuilding embeddings, BM25, FAISS, etc.),
    saves the state back to data_dir, and returns the new RAG instance and
    the number of documents re-indexed.

    Returns:
        (rag, n_documents). If no persisted data exists, returns (fresh rag, 0).
    """
    try:
        from pitagora.vector_store import (
            get_data_dir,
            load as load_vector_store,
            save as save_vector_store,
            load_rag_config,
        )
    except ImportError:
        from vector_store import (
            get_data_dir,
            load as load_vector_store,
            save as save_vector_store,
            load_rag_config,
        )

    dir_to_use = Path(data_dir) if data_dir is not None else get_data_dir()
    rag_config = load_rag_config(dir_to_use)
    if rag_config:
        rag = create_rag(**{k: v for k, v in rag_config.items() if k in ("vocab_size", "use_quantum_lattice", "use_bm25")})
    else:
        rag = create_rag()
    loaded = load_vector_store(dir_to_use)
    n_docs = 0
    if loaded:
        documents, metadata, vocab = loaded
        n_docs = len(documents)
        rag.tokenizer.word_to_id = vocab
        rag.tokenizer.id_to_word = {v: k for k, v in vocab.items()}
        rag.add_documents(documents, metadata)
        save_vector_store(rag, dir_to_use)
    return rag, n_docs


def reindex_beir(data_dir: Optional[Path] = None) -> Tuple[Any, int]:
    """
    Re-index all documents from the persisted vector store using BEIR (sentence-transformers).

    Loads documents and metadata from data_dir, builds a BeirEncoderRetriever,
    and returns it with the document count. Does not persist BEIR state (in-memory only).
    Use when the user selects BEIR encoding for the main app retrieval.

    Returns:
        (beir_retriever, n_documents). If no persisted data, returns (retriever with 0 docs, 0).
    """
    try:
        from pitagora.vector_store import get_data_dir, load as load_vector_store
        from pitagora.beir_encoder import create_beir_retriever
    except ImportError:
        from vector_store import get_data_dir, load as load_vector_store
        from beir_encoder import create_beir_retriever

    dir_to_use = Path(data_dir) if data_dir is not None else get_data_dir()
    loaded = load_vector_store(dir_to_use)
    retriever = create_beir_retriever()
    n_docs = 0
    if loaded:
        documents, metadata, _ = loaded
        n_docs = len(documents)
        retriever.add_documents(documents, metadata)
    return retriever, n_docs
