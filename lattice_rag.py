"""
lattice_rag - append-only multi-view lattice retrieval engine (public API).

A BM25-class retrieval engine built as pure counting on prime addresses: no
neural weights, no training loop, O(1) append, no reindex, deterministic and
verifiable. Supervised relevance bridges add an accuracy layer learned by
counting qrels. Scales to any N via champion lists (one machine) or a sharded,
distributed-exact index. See README.md for the measured numbers and FINDINGS.md
for the method and the honest negative results.

    from lattice_rag import AppendOnlyLatticeIndex, RelevanceBridges, bridge_search

    idx = AppendOnlyLatticeIndex()
    for doc_id, text in corpus.items():
        idx.add(doc_id, text)            # O(1) append, no reindex
    idx.finalize()                       # build the numpy fast path (~15x)
    hits = idx.search("query text", k=10)

    # multi-corpus brain (shared primes, isolated lattices per corpus)
    from lattice_rag import MultiCorpusBrain
    brain = MultiCorpusBrain()
    brain.stack_corpus("scifact", corpus_a, queries=q, train_qrels=train)
    brain.stack_corpus("nfcorpus", corpus_b)
    hits = brain.search("query text")          # auto-route corpus
    result = brain.search("query", corpus="scifact")

    # optional supervised accuracy layer (learned by counting qrels)
    br = RelevanceBridges(idx, len(idx.alive)).learn(queries, train_qrels, corpus)
    hits = bridge_search(idx, br, "query text", k=10)
"""

from aethos_append_index import AppendOnlyLatticeIndex, words
from aethos_bridges import RelevanceBridges, bridge_search, bridge_search_dense, choose_bridge
from aethos_gap_miner import GapReport, mine_query_gaps
from aethos_encyclopedia_teacher import (
    TeachGapResult, load_glossary, teach_full_knowledge_for_corpus, teach_gaps_for_corpus,
)
from aethos_multi_corpus import (
    CorpusBranch, LearnIteration, LearnSaturationResult, MultiCorpusBrain, SearchResult,
)
from aethos_sharded_index import ShardedIndex
from aethos_teach_store import TeachStore
from aethos_vocab_gap_router import (
    GapSignal,
    choose_expansion_mode,
    fuse_lex_expansion,
    measure_vocab_gap,
    prf_expansion,
    routed_search,
)

__all__ = [
    "AppendOnlyLatticeIndex", "words",
    "RelevanceBridges", "bridge_search", "bridge_search_dense", "choose_bridge",
    "ShardedIndex",
    "MultiCorpusBrain", "CorpusBranch", "SearchResult",
    "LearnIteration", "LearnSaturationResult",
    "GapReport", "mine_query_gaps", "TeachGapResult",
    "teach_gaps_for_corpus", "teach_full_knowledge_for_corpus", "load_glossary",
    "TeachStore",
    "GapSignal", "choose_expansion_mode", "measure_vocab_gap",
    "prf_expansion", "fuse_lex_expansion", "routed_search",
]
__version__ = "0.1.0"
