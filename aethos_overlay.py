"""
Semantic overlay between registry lattice addresses and codec/word dots.

This makes the relationship explicit:
  - registry local address (no origin offset)
  - semantic codec dot (origin + local)
  - word dot canonical base (orderless local)
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

from aethos_codec import Dot, IntersectionWitness, coordinate_from_witness
from aethos_lattice import LatticeId
from aethos_origins import OriginTree
from aethos_promotion import LatticeTier, PromotionRegistry
from aethos_words import SharedSite, canonical_base, encode_word_at_site, word_sorted_chain


def _origin_coord(origin_path: str) -> tuple[float, float, float]:
    tree = OriginTree.bootstrap(max_depth=2)
    o = next(node for node in tree.walk() if node.id == origin_path)
    return o.coord


def _semantic_chain(reg: PromotionRegistry, word: str) -> tuple[int, ...]:
    tok = reg.resolve_token(word)
    if tok.intersection_only:
        return tuple(sorted(set(tok.parent_primes)))
    return tuple(sorted(set(tok.parent_primes + (tok.prime,))))


@dataclass(frozen=True)
class SemanticOverlay:
    word: str
    tier: str
    chain: tuple[int, ...]
    n: int
    lattice_id: int
    origin_path: str
    registry_local: tuple[float, float, float]
    codec_dot: Dot
    codec_local: tuple[float, float, float]
    word_dot: Dot | None = None
    word_base: tuple[float, float, float] | None = None

    @property
    def registry_equals_codec_local(self) -> bool:
        return self.registry_local == self.codec_local

    @property
    def registry_equals_word_base(self) -> bool:
        return self.word_base is not None and self.registry_local == self.word_base

    def summary(self) -> str:
        return (
            f"Overlay[{self.word!r}] tier={self.tier} chain={self.chain} n={self.n} L={self.lattice_id}\n"
            f"  registry local: {self.registry_local}\n"
            f"  codec local:    {self.codec_local}  match={self.registry_equals_codec_local}\n"
            f"  codec dot:      {self.codec_dot.coord} origin={self.origin_path}\n"
            f"  word base:      {self.word_base}  match={self.registry_equals_word_base}"
        )


def overlay_for_word(
    registry: PromotionRegistry,
    word: str,
    *,
    n: int = 7,
    lattice_id: LatticeId = LatticeId.L01,
    origin_path: str = "O0",
    include_word_dot: bool = True,
) -> SemanticOverlay:
    """
    Build explicit relation between semantic layers for one word.

    - registry_local = registry.lattice_address(...)
    - codec_dot = coordinate_from_witness(semantic witness)
    - codec_local = codec_dot - origin_offset
    - word_base = canonical base of encode_word_at_site(...) when requested
    """
    w = word.lower()
    chain = _semantic_chain(registry, w)
    if not chain:
        raise ValueError("word has no alphabetic anchors")

    tok = registry.resolve_token(w)
    tier = "intersection_only" if tok.intersection_only else "dedicated_l3"
    registry_local = registry.lattice_address(w, LatticeTier.L3_WORD, n=n, lattice_id=lattice_id)

    witness = IntersectionWitness(
        chain=chain,
        n=n,
        lattice_id=int(lattice_id),
        origin_path=origin_path,
        dim_slot=None,
        payload=zlib.compress(f"semantic:{w}".encode("utf-8"), level=9),
        prime_order=(),
    )
    codec_dot = Dot(*coordinate_from_witness(witness, with_order_offset=False), witness=witness)
    ox, oy, oz = _origin_coord(origin_path)
    codec_local = (codec_dot.x - ox, codec_dot.y - oy, codec_dot.z - oz)

    word_dot = None
    word_base = None
    if include_word_dot:
        site = SharedSite(chain=word_sorted_chain(w), n=n, lattice_id=int(lattice_id), origin_path=origin_path, dim_slot=None)
        word_dot = encode_word_at_site(w, site)
        word_base = canonical_base(word_dot)

    return SemanticOverlay(
        word=w,
        tier=tier,
        chain=chain,
        n=n,
        lattice_id=int(lattice_id),
        origin_path=origin_path,
        registry_local=registry_local,
        codec_dot=codec_dot,
        codec_local=codec_local,
        word_dot=word_dot,
        word_base=word_base,
    )
