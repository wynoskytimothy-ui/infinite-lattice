"""Rare-combo triple cages — non-adjacent rare terms form C(k,3) anchors."""

from lattice_retriever_v1.doc_lattice_codec import (
    build_rare_combo_cages,
    build_doc_correlation_shells,
)
from lattice_retriever_v1.stage04_promote import promote_from_stream
from lattice_retriever_v1.stage07_semantic_light import SemanticLightIndex


def _semantic_with_corpus(texts: list[str]) -> SemanticLightIndex:
    reg = promote_from_stream(texts)
    sem = SemanticLightIndex(registry=reg)
    for text in texts:
        sem.observe_doc(text, mode="positional")
    return sem


def test_rare_combo_non_adjacent_cat_pet_purr():
    corpus = [
        "cat likes the pet and purrs loudly",
        "dog barks at the mail carrier",
        "birds sing in the morning trees",
    ]
    sem = _semantic_with_corpus(corpus)
    words = tuple("cat likes the pet and purrs loudly".split())
    triples = build_rare_combo_cages(words, sem, sem.registry, k_rare=8)
    assert ("cat", "pet", "purrs") in triples
    idx = SemanticLightIndex(registry=sem.registry)
    idx.n_docs = sem.n_docs
    idx.doc_freq = dict(sem.doc_freq)
    idx.observe_doc(
        "cat likes the pet and purrs loudly",
        mode="rare_combo",
    )
    cage = idx._cage_for_triple("cat", "pet", "purrs")
    score = idx.touch_weight(["cat", "purrs"], cage)
    assert score > 0


def test_rare_combo_mitochondria_atp_style():
    corpus = [
        "mitochondria produces atp energy in cells",
        "cells need oxygen for respiration pathways",
        "plants convert sunlight into glucose stores",
    ]
    sem = _semantic_with_corpus(corpus)
    words = tuple("mitochondria produces atp energy in cells".split())
    triples = build_rare_combo_cages(words, sem, sem.registry, k_rare=8)
    assert len(triples) >= 1
    assert all(len(t) == 3 for t in triples)
    shells = build_doc_correlation_shells(
        " ".join(words),
        sem.registry,
        semantic=sem,
        mode="rare_combo",
    )
    anchor_keys = {s.key for s in shells if s.key_kind == "anchor"}
    assert any("mitochondria" in k for k in anchor_keys)


def test_positional_mode_still_default_six_word_cap():
    sem = _semantic_with_corpus(["apple phone sells iphone tablet watch case"])
    idx = SemanticLightIndex(registry=sem.registry)
    idx.n_docs = sem.n_docs
    idx.doc_freq = dict(sem.doc_freq)
    idx.observe_doc("apple phone sells iphone tablet watch case", mode="positional")
    assert len(idx.cages) >= 4
