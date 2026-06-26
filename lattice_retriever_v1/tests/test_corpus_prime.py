"""Corpus prime scaffold — scoped doc ids, deterministic pool allocation."""

from lattice_retriever_v1.corpus_prime import (
    allocate_corpus_prime,
    corpus_scope,
    tag_doc_id,
)
from lattice_retriever_v1.stage04_promote import promote_from_stream

ING_CORPUS = [
    "running quickly",
    "walking slowly",
    "thinking deeply",
    "building houses",
]


def test_two_corpora_distinct_pool_primes() -> None:
    stage = promote_from_stream([])
    p_a = allocate_corpus_prime("scifact", stage)
    p_b = allocate_corpus_prime("nq", stage)
    assert p_a != p_b


def test_same_corpus_name_replay_same_prime() -> None:
    stage = promote_from_stream([])
    first = allocate_corpus_prime("scifact", stage)
    second = allocate_corpus_prime("scifact", stage)
    assert first == second


def test_tagged_doc_ids_no_cross_corpus_collision() -> None:
    a = tag_doc_id("scifact", "doc1")
    b = tag_doc_id("nq", "doc1")
    assert a != b
    assert a == "scifact:doc1"
    assert b == "nq:doc1"


def test_shared_registry_hilbert_hotel() -> None:
    """Promotions from corpus A stay put when corpus B ingests on the same registry."""
    stage = promote_from_stream(ING_CORPUS)
    ing_before = stage.promoted_subword("ing")
    assert ing_before is not None
    prime_before = ing_before.prime

    allocate_corpus_prime("scifact", stage)
    allocate_corpus_prime("nq", stage)
    stage.observe_stream(["another walking path", "extra building work"])

    ing_after = stage.promoted_subword("ing")
    assert ing_after is not None
    assert ing_after.prime == prime_before


def test_corpus_scope_matches_allocate() -> None:
    stage = promote_from_stream([])
    scope = corpus_scope("scifact", stage)
    assert scope.corpus_name == "scifact"
    assert scope.doc_id_prefix == "scifact"
    assert scope.corpus_prime == allocate_corpus_prime("scifact", stage)
