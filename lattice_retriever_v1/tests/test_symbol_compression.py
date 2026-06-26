"""Pure symbol compression — lossless roundtrip, no RAG."""

import random

from lattice_retriever_v1.symbol_compression import analyze_stream, decode_bytes, encode_bytes


def _rand(n: int, alphabet: int, seed: int = 0) -> bytes:
    rng = random.Random(seed)
    return bytes(rng.randrange(alphabet) for _ in range(n))


def test_lossless_roundtrip_100_symbols():
    data = _rand(50_000, 100)
    payload, ledger = encode_bytes(data)
    assert decode_bytes(payload) == data
    assert ledger.n_symbols <= 100


def test_alphabet_table_tiny_vs_raw():
    data = _rand(1_000_000, 100)
    ledger = analyze_stream(data)
    assert ledger.alphabet_bytes < 300
    assert ledger.total_symbol_codec_bytes < len(data)


def test_unique_pairs_bounded_by_alphabet_squared():
    data = _rand(100_000, 50)
    ledger = analyze_stream(data)
    assert ledger.n_unique_pairs <= 50 * 50
