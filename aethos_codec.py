"""
AETHOS intersection codec — store data as deterministic 3D dots.

Looks random in (x,y,z); the formula witness recreates the dot and decodes meaning.
Compress payload to a minimal dot record (witness + verified coordinate), then expand.
"""

from __future__ import annotations

import base64
import json
import struct
import zlib
from dataclasses import dataclass
from typing import Any

from aethos_lattice import LatticeId, apply_vector, lattice_id_parts
from aethos_origins import OriginTree
from aethos_permutation import apply_order_offset, apply_sequence_offset, decode_order_from_dot, explain_order
from aethos_sequences import SequenceKind, canon_on_chain, make_chain

# Fixed prime menu for embedding byte chunks into anchor chains
PRIME_MENU = make_chain(SequenceKind.PRIMES, 64)


@dataclass(frozen=True)
class IntersectionWitness:
    """
    Semantic preimage of one dot — enough to recompute (x,y,z) and decode payload.
    """

    chain: tuple[int, ...]
    n: int
    lattice_id: int  # 1..32
    origin_path: str
    dim_slot: int | None  # 0,1,2 or None for root
    payload: bytes  # zlib-compressed original bytes
    prime_order: tuple[int, ...] = ()  # application order; side-offset on dot if set

    def sorted_chain(self) -> tuple[int, ...]:
        return tuple(sorted(self.chain)) if self.chain else ()

    def to_compact(self) -> str:
        """Minimal serial form (~one dot record)."""
        blob = base64.urlsafe_b64encode(self.payload).decode("ascii")
        doc = {
            "c": list(self.chain),
            "n": self.n,
            "L": self.lattice_id,
            "o": self.origin_path,
            "d": self.dim_slot,
            "p": blob,
        }
        if self.prime_order:
            doc["ord"] = list(self.prime_order)
        return base64.urlsafe_b64encode(json.dumps(doc, separators=(",", ":")).encode()).decode()

    @classmethod
    def from_compact(cls, s: str) -> IntersectionWitness:
        doc = json.loads(base64.urlsafe_b64decode(s.encode()).decode())
        ord_raw = doc.get("ord")
        return cls(
            chain=tuple(doc["c"]),
            n=int(doc["n"]),
            lattice_id=int(doc["L"]),
            origin_path=str(doc["o"]),
            dim_slot=doc["d"],
            payload=base64.urlsafe_b64decode(doc["p"].encode()),
            prime_order=tuple(ord_raw) if ord_raw else (),
        )


@dataclass(frozen=True)
class Dot:
    """One intersection in 3D — coordinate verified from witness."""

    x: float
    y: float
    z: float
    witness: IntersectionWitness

    @property
    def coord(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dot": [self.x, self.y, self.z],
            "meaning": explain_witness(self.witness),
            "compact": self.witness.to_compact(),
        }


def _pick_chain(data: bytes, max_len: int = 6) -> tuple[int, ...]:
    if not data:
        return (PRIME_MENU[0],)
    length = min(max_len, max(1, len(data) % max_len + 1))
    out: list[int] = []
    for i in range(length):
        out.append(PRIME_MENU[data[i % len(data)] % len(PRIME_MENU)])
    return tuple(sorted(set(out)))


def _pick_lattice_id(data: bytes) -> LatticeId:
    h = sum(data) + len(data) * 17
    return LatticeId((h % 32) + 1)


def _pick_n(data: bytes, compressed: bytes) -> int:
    if len(compressed) >= 4:
        (n,) = struct.unpack(">I", compressed[:4])
        return max(1, n % 100_000)
    return max(1, sum(compressed) + len(data))


def _origin_path(data: bytes) -> tuple[str, int | None]:
    tree = OriginTree.bootstrap(max_depth=2)
    origins = list(tree.walk())
    idx = sum(data) % len(origins)
    o = origins[idx]
    return o.id, int(o.dim_slot) if o.dim_slot is not None else None


def witness_from_bytes(raw: bytes) -> IntersectionWitness:
    compressed = zlib.compress(raw, level=9)
    chain = _pick_chain(compressed)
    n = _pick_n(raw, compressed)
    lid = _pick_lattice_id(raw)
    origin_path, dim_slot = _origin_path(raw)
    return IntersectionWitness(
        chain=chain,
        n=n,
        lattice_id=int(lid),
        origin_path=origin_path,
        dim_slot=dim_slot,
        payload=compressed,
    )


def coordinate_from_witness(w: IntersectionWitness, *, with_order_offset: bool = True) -> tuple[float, float, float]:
    branch, vector = lattice_id_parts(LatticeId(w.lattice_id))
    chain = w.sorted_chain()
    canon = canon_on_chain(branch, chain, w.n)
    cx, cy, cz = apply_vector(canon, vector)

    tree = OriginTree.bootstrap(max_depth=2)
    origin = next(o for o in tree.walk() if o.id == w.origin_path)
    ox, oy, oz = origin.coord
    base = (ox + cx, oy + cy, oz + cz)

    if with_order_offset and w.prime_order:
        chain = w.sorted_chain()
        if len(set(w.prime_order)) == len(w.prime_order) and len(w.prime_order) >= 2:
            return apply_order_offset(base, chain, w.prime_order)
        if len(w.prime_order) >= 1:
            return apply_sequence_offset(base, w.prime_order)
    return base


def encode_bytes(raw: bytes, prime_order: tuple[int, ...] | None = None) -> Dot:
    w = witness_from_bytes(raw)
    chain = w.chain
    order: tuple[int, ...] = ()
    if prime_order:
        order = tuple(prime_order)
        chain = tuple(sorted(set(order)))
    w = IntersectionWitness(
        chain=chain,
        n=w.n,
        lattice_id=w.lattice_id,
        origin_path=w.origin_path,
        dim_slot=w.dim_slot,
        payload=w.payload,
        prime_order=order,
    )
    x, y, z = coordinate_from_witness(w)
    return Dot(x=x, y=y, z=z, witness=w)


def verify_dot(dot: Dot) -> bool:
    rx, ry, rz = coordinate_from_witness(dot.witness)
    return (rx, ry, rz) == (dot.x, dot.y, dot.z)


def recovered_prime_order(dot: Dot) -> tuple[int, ...]:
    """Read application order from tiny side-offset (k! channel)."""
    w = dot.witness
    chain = w.sorted_chain()
    if len(chain) < 2:
        return chain
    base = coordinate_from_witness(
        IntersectionWitness(
            chain=w.chain,
            n=w.n,
            lattice_id=w.lattice_id,
            origin_path=w.origin_path,
            dim_slot=w.dim_slot,
            payload=w.payload,
            prime_order=(),
        ),
        with_order_offset=False,
    )
    if w.prime_order:
        return decode_order_from_dot(base, dot.coord, chain)
    return chain


def decode_bytes(dot: Dot | IntersectionWitness) -> bytes:
    w = dot.witness if isinstance(dot, Dot) else dot
    if isinstance(dot, Dot):
        if not verify_dot(dot):
            raise ValueError("coordinate does not match witness — dot corrupted")
    return zlib.decompress(w.payload)


def explain_witness(w: IntersectionWitness) -> str:
    branch, vector = lattice_id_parts(LatticeId(w.lattice_id))
    dim = f"D{w.dim_slot + 1}" if w.dim_slot is not None else "root"
    raw_size = len(zlib.decompress(w.payload))
    order_part = ""
    if w.prime_order and len(w.sorted_chain()) >= 2:
        order_part = explain_order(w.sorted_chain(), w.prime_order) + "; "
    return (
        f"Intersection on wing {w.lattice_id} ({branch.name}/{vector.name}), "
        f"origin {w.origin_path} [{dim}], "
        f"{order_part}"
        f"sorted anchors {w.sorted_chain()}, transgressor n={w.n}, "
        f"payload {len(w.payload)} bytes -> {raw_size} bytes decoded"
    )


def compress_to_one_dot(raw: bytes) -> str:
    """Serialize entire message to one compact dot record (preimage + formula params)."""
    return encode_bytes(raw).witness.to_compact()


def expand_from_one_dot(compact: str) -> tuple[bytes, Dot]:
    w = IntersectionWitness.from_compact(compact)
    dot = Dot(*coordinate_from_witness(w), witness=w)
    if not verify_dot(dot):
        raise ValueError("recomputed coordinate mismatch")
    return decode_bytes(w), dot


def encode_text(text: str) -> Dot:
    return encode_bytes(text.encode("utf-8"))


def decode_text(dot: Dot) -> str:
    return decode_bytes(dot).decode("utf-8")


def demo() -> None:
    samples = [
        b"Hello AETHOS",
        b"The quick brown fox " * 20,
        json.dumps({"user": "wynos", "keys": list(range(50))}).encode(),
    ]

    print("=" * 60)
    print("AETHOS INTERSECTION STORAGE — dots look random, formula decodes")
    print("=" * 60)

    for i, raw in enumerate(samples):
        dot = encode_bytes(raw)
        compact = dot.witness.to_compact()
        back, dot2 = expand_from_one_dot(compact)

        print(f"\n--- Sample {i + 1} ({len(raw)} bytes) ---")
        print(f"  Dot (looks random):  ({dot.x}, {dot.y}, {dot.z})")
        print(f"  Meaning:             {explain_witness(dot.witness)}")
        print(f"  Compact dot record:  {len(compact)} chars")
        print(f"  Ratio raw/compact:   {len(raw) / len(compact):.2f}x")
        print(f"  Verify formula:      {verify_dot(dot)}")
        print(f"  Roundtrip OK:        {back == raw}")

    print("\n--- Prime order side-channel (same set, different path) ---")
    chain = (3, 5, 7)
    for order in ((3, 5, 7), (7, 5, 3), (5, 3, 7)):
        dot = encode_bytes(b"order-test", prime_order=order)
        got = recovered_prime_order(dot)
        delta = (dot.x - coordinate_from_witness(
            IntersectionWitness(
                chain=chain, n=dot.witness.n, lattice_id=dot.witness.lattice_id,
                origin_path=dot.witness.origin_path, dim_slot=dot.witness.dim_slot,
                payload=dot.witness.payload, prime_order=(),
            ),
            with_order_offset=False,
        )[0])
        print(f"  order {order} -> dot shift dx~{delta:+.6f}  recovered {got}  ok={got==order}")

    print("\n--- Single-dot roundtrip (large text) ---")
    big = ("AETHOS compresses by intersection. " * 100).encode()
    one = compress_to_one_dot(big)
    restored, d = expand_from_one_dot(one)
    print(f"  Raw:      {len(big)} bytes")
    print(f"  1 dot:    {len(one)} chars compact record")
    print(f"  3D point: {d.coord}")
    print(f"  Match:    {restored == big}")


if __name__ == "__main__":
    demo()
