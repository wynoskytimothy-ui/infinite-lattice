"""
Personal lattice codec — infinite key-sets; wrong key → random-looking blob.

Each personal key picks a unique symbol→prime assignment (one of infinitely many
valid intersection sets). The formula always finds natural 2-way meets for that
set. Without the key, dot addresses decode to high-entropy garbage.

With the key: bare lumber + dot blob → formula reconstructs exact data.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass

from aethos_sequences import SequenceKind, make_chain

from lattice_retriever_v1.dot_blob_codec import (
    DotBlob,
    DotBlobLedger,
    _encode_wire,
    reconstruct_from_blob,
)
from lattice_retriever_v1.intersection_dot_codec import (
    SymbolAlphabet,
    _oriented_pair_catalog,
    document_pair_walk,
)

MAGIC = b"LPER"


@dataclass(frozen=True)
class PersonalKey:
    """Secret — selects one intersection set from infinite personal assignments."""

    secret: bytes

    @classmethod
    def from_passphrase(cls, text: str) -> PersonalKey:
        return cls(secret=text.encode("utf-8"))

    def digest(self) -> bytes:
        return hashlib.sha256(self.secret).digest()


def personal_alphabet(symbols: tuple[int, ...], key: PersonalKey) -> SymbolAlphabet:
    """
    Key-derived prime assignment for this symbol catalog.
    Same symbols + different key → different rails → different dot geometry.
    """
    n = len(symbols)
    if n == 0:
        return SymbolAlphabet(symbols=())
    seed = key.digest()
    pool_start = int.from_bytes(seed[0:4], "little") % 5000
    pool = make_chain(SequenceKind.PRIMES, pool_start + n + 1)[pool_start : pool_start + n]
    order = list(range(n))
    state = bytearray(seed)
    for i in range(n - 1, 0, -1):
        state = hashlib.sha256(bytes(state)).digest()
        j = int.from_bytes(state[0:4], "little") % (i + 1)
        order[i], order[j] = order[j], order[i]
    prime_by_index = [0] * n
    for rank, sym_idx in enumerate(order):
        prime_by_index[sym_idx] = int(pool[rank])
    alpha = SymbolAlphabet(symbols=symbols)
    object.__setattr__(alpha, "prime_by_index", tuple(prime_by_index))
    return alpha


def _keystream(key: PersonalKey, n: int) -> bytes:
    out = bytearray()
    state = key.digest()
    while len(out) < n:
        state = hashlib.sha256(state).digest()
        out.extend(state)
    return bytes(out[:n])


def _xor_body(body: bytes, key: PersonalKey) -> bytes:
    ks = _keystream(key, len(body))
    return bytes(a ^ b for a, b in zip(body, ks))


def encode_personal(data: bytes, key: PersonalKey) -> tuple[DotBlob, DotBlobLedger, bytes]:
    """Compress with personal intersection set. Key NOT stored in wire."""
    symbols = tuple(sorted(set(data)))
    alpha = personal_alphabet(symbols, key)
    walk = document_pair_walk(data, alpha)
    catalog = _oriented_pair_catalog(alpha)
    cat_index = {k: i for i, k in enumerate(catalog)}
    dots = tuple((cat_index[d.origin], d.pair_n) for d in walk)
    blob = DotBlob(alphabet=alpha, dots=dots, raw_len=len(data))
    max_n = max((n for _, n in dots), default=1)
    inner = _encode_wire(blob, data=data, catalog_size=len(catalog), max_n=max_n)
    body = inner[len(b"LDOT") :]
    wire = MAGIC + _xor_body(body, key)
    token_bytes = 2 + alpha.n
    coords_if = len(walk) * 32 * 3 * 4
    ledger = DotBlobLedger(
        raw_bytes=len(data),
        n_tokens=alpha.n,
        n_dots=len(dots),
        n_origins_max=len(catalog),
        n_origins_used=len({d.origin for d in walk}),
        token_catalog_bytes=token_bytes,
        dot_blob_bytes=len(wire) - len(MAGIC) - 14 - token_bytes,
        formula_stored_bytes=0,
        coord_bytes_if_stored=coords_if,
        total_stored_bytes=len(wire),
    )
    return blob, ledger, wire


def decode_personal(wire: bytes, key: PersonalKey) -> bytes:
    """Reconstruct only with correct personal key."""
    if not wire.startswith(MAGIC):
        raise ValueError("bad magic")
    body = _xor_body(wire[len(MAGIC) :], key)
    inner = b"LDOT" + body
    off = len(b"LDOT")
    n_raw, n_dots, n_sym, pair_bits, n_bits = struct.unpack_from("<IIHBB", inner, off)
    off += 12
    symbols = tuple(inner[off : off + n_sym])
    off += n_sym
    alpha = personal_alphabet(symbols, key)
    if n_raw <= 1:
        return inner[-n_raw:] if n_raw else b""
    from lattice_retriever_v1.intersection_dot_codec import _unpack_pairs

    dots = tuple(_unpack_pairs(inner[off:], n_values=n_dots, pair_bits=pair_bits, n_bits=n_bits))
    blob = DotBlob(alphabet=alpha, dots=dots, raw_len=n_raw)
    return reconstruct_from_blob(blob)


def blob_entropy(wire: bytes) -> float:
    """Mean byte entropy — high when blob looks random without key context."""
    if len(wire) < 2:
        return 0.0
    counts = [0] * 256
    for b in wire:
        counts[b] += 1
    n = len(wire)
    ent = 0.0
    for c in counts:
        if c:
            p = c / n
            ent -= p * __import__("math").log2(p)
    return ent


def key_set_explain(n_symbols: int) -> dict:
    """How many personal intersection sets exist for a symbol catalog."""
    return {
        "n_symbols": n_symbols,
        "pair_origins_per_set": n_symbols**2,
        "personal_key_space_bits": 256,
        "note": "each key → unique prime assignment; formula valid for any set",
    }
