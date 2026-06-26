"""Tests for electron-model tokenizer (2 bits → 4 coin states per slot)."""

from __future__ import annotations

from aethos_electron_tokenizer import (
    CoinState,
    ElectronVocabCodec,
    bit_pair_to_state,
    decode_bit_stream,
    encode_bit_stream,
    iter_pairs,
    pack_states,
    state_to_bits,
    tier_to_state,
    tokenize_electron,
    unpack_states,
)
from aethos_pipeline import AethosPipeline


def test_two_bits_four_states_roundtrip():
    for state in CoinState:
        m, s = state_to_bits(state)
        assert bit_pair_to_state((m, s)) == state
        assert len({bit_pair_to_state(state_to_bits(st)) for st in CoinState}) == 4


def test_bit_stream_roundtrip():
    seq = [CoinState.WH, CoinState.BS, CoinState.WH, CoinState.BH]
    bits = pack_states(seq)
    assert bits == [0, 1, 1, 0, 0, 1, 1, 1]
    assert unpack_states(bits) == seq
    assert decode_bit_stream(bits) == seq


def test_iter_pairs_padding():
    assert list(iter_pairs([1, 0, 1])) == [(1, 0), (1, 0)]


def test_tier_mapping_after_ingest():
    pipe = AethosPipeline()
    pipe.ingest("the cat sat on the mat", "zebra runs fast")
    reg = pipe.registry
    assert tier_to_state(reg, "the") == CoinState.WH
    assert tier_to_state(reg, "xyzzy", species="WORD") == CoinState.WS
    toks = tokenize_electron("the cat", reg)
    assert len(toks) == 2
    assert all(len(t.bits) == 2 for t in toks)
    assert encode_bit_stream(toks) == pack_states(t.state for t in toks)


def test_vocab_codec_roundtrip():
    words = ["cat", "dog", "zebra", "mat"]
    codec = ElectronVocabCodec(words)
    assert codec.vocab_size == 4
    assert codec.dits_per_token == 1  # 4 tokens fit in one 2-bit electron read
    doc = ["cat", "zebra", "dog"]
    bits = codec.encode_document(doc)
    assert len(bits) == len(doc) * 2  # one electron read (2 bits) per token
    back = codec.decode_document(bits, n_tokens=len(doc))
    assert back == doc
