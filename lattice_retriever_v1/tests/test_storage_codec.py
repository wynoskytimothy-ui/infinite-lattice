"""Tests for lattice storage codec (not RAG)."""

from __future__ import annotations

from lattice_retriever_v1.storage_codec import (
    analyze_storage,
    decode_storage,
    detect_patterns,
    encode_storage,
    extrapolate_ledger,
    SymbolAlphabet,
)


def test_roundtrip_simple():
    data = b"the quick brown fox jumps over the lazy dog"
    payload, ledger, patterns = encode_storage(data)
    assert decode_storage(payload) == data
    assert ledger.n_atomic_symbols <= 27
    assert ledger.ratio >= 0.5


def test_repetitive_compresses_below_alphabet():
    """Repeating pattern → composite tokens; effective vocab can shrink."""
    chunk = b"hello world "
    data = chunk * 500
    ledger = analyze_storage(data, min_pattern_count=2, min_cohesion=0.9)
    assert ledger.n_patterns >= 1
    assert ledger.n_effective_tokens <= ledger.n_atomic_symbols + ledger.n_patterns
    assert ledger.ratio > 2.0


def test_fixed_alphabet_index_bounded():
    """Index layer (alphabet + patterns) bounded by symbol set, not walk length."""
    import random

    rng = random.Random(0)
    hot = list(range(10))

    def gen(n):
        return bytes(rng.choice(hot) for _ in range(n))

    l1 = analyze_storage(gen(2_000), min_cohesion=0.8)
    l2 = analyze_storage(gen(10_000), min_cohesion=0.8)
    assert l1.n_atomic_symbols == l2.n_atomic_symbols == 10
    assert l1.index_only_bytes == l2.index_only_bytes


def test_extrapolate_100tb():
    chunk = b"alpha beta gamma "
    data = chunk * 1000
    ledger = analyze_storage(data, min_cohesion=0.8)
    ext = extrapolate_ledger(ledger, target_bytes=100 * 1_000_000_000_000)
    assert ext["extrapolate_index_only_bytes"] < 500
    assert ext["extrapolate_ratio_x"] > 5


def test_patterns_have_lattice_quadrant():
    data = b"abcabcabc"
    alpha = SymbolAlphabet.from_bytes(data)
    pats = detect_patterns(data, alpha, min_count=2, min_cohesion=0.9)
    assert pats
    for p in pats:
        assert 1 <= p.quadrant <= 32
        assert p.lattice_L01
