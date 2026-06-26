"""Personal lattice codec — keyed intersection sets."""

from __future__ import annotations

from lattice_retriever_v1.personal_lattice_codec import (
    PersonalKey,
    blob_entropy,
    decode_personal,
    encode_personal,
)


def test_roundtrip_with_key():
    key = PersonalKey.from_passphrase("my-secret-rail")
    data = b"the quick brown fox"
    _, _, wire = encode_personal(data, key)
    assert decode_personal(wire, key) == data


def test_wrong_key_garbage():
    key_a = PersonalKey.from_passphrase("alice")
    key_b = PersonalKey.from_passphrase("bob")
    data = b"private lattice message"
    _, _, wire = encode_personal(data, key_a)
    try:
        out = decode_personal(wire, key_b)
        assert out != data
    except ValueError:
        pass  # garbled blob without correct key


def test_blob_looks_high_entropy_without_key():
    key = PersonalKey.from_passphrase("hidden-set-771")
    data = b"hello " * 200
    _, _, wire = encode_personal(data, key)
    assert blob_entropy(wire) > 6.0


def test_different_keys_different_wires():
    data = b"same plaintext"
    w1 = encode_personal(data, PersonalKey.from_passphrase("k1"))[2]
    w2 = encode_personal(data, PersonalKey.from_passphrase("k2"))[2]
    assert w1 != w2
