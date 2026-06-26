"""
Origin corridor — wrap OriginTree for corridor key regeneration with origin offset.

Corridor keys (meet_composite, quadrant, n) are origin-invariant; lattice coords
gain the origin meet offset. At depth 0 (O0) offset is zero — regeneration is
bit-identical with or without origin.
"""

from __future__ import annotations

from functools import reduce
from operator import mul
from typing import Sequence

from aethos_lattice import Coord
from aethos_origins import Origin, OriginTree

from lattice_retriever_v1.stage02_intersections import (
    DEFAULT_TRANSGRESSOR_N,
    lattice_signature,
)
from lattice_retriever_v1.stage05_free_token import (
    FreeTokenAddress,
    free_token_address,
)


def default_origin_tree(*, max_depth: int = 2) -> OriginTree:
    return OriginTree.bootstrap(max_depth=max_depth)


def resolve_origin(tree: OriginTree, origin_id: str | None) -> Origin:
    oid = origin_id if origin_id is not None else "O0"
    for node in tree.walk():
        if node.id == oid:
            return node
    raise KeyError(f"unknown origin_id {oid!r}")


def origin_offset_coord(base_coord: Coord, origin: Origin) -> Coord:
    """Shift lattice coord by origin meet point."""
    ox, oy, oz = origin.coord
    bx, by, bz = base_coord
    return (ox + bx, oy + by, oz + bz)


def offset_signature(
    signature: tuple[Coord, ...],
    origin: Origin,
) -> tuple[Coord, ...]:
    return tuple(origin_offset_coord(c, origin) for c in signature)


def corridor_key_with_origin(
    primes: Sequence[int],
    n: int = DEFAULT_TRANSGRESSOR_N,
    *,
    origin_id: str | None = None,
    quadrant: int = 1,
    tree: OriginTree | None = None,
    invoke_order: tuple[int, int] | None = None,
) -> dict:
    """
    Glass-box corridor key + origin-offset lattice readout.

    corridor_key is unchanged by origin; L01 coords pick up origin.coord offset.
    """
    ps = tuple(primes)
    if len(ps) < 2:
        raise ValueError("corridor_key_with_origin needs at least two primes")

    tree = tree or default_origin_tree()
    origin = resolve_origin(tree, origin_id)
    qid = max(1, min(32, int(quadrant)))

    if len(ps) == 2:
        p, q = ps[0], ps[1]
        order = invoke_order if invoke_order is not None else (p, q)
        addr = free_token_address(
            p,
            q,
            quadrant=qid,
            transgressor_n=n,
            invoke_order=order,
        )
        base_sig = addr.lattice_signature
        corridor_key = addr.corridor_key
        meet_composite = addr.meet_composite
        canonical_pair = [addr.p, addr.q]
        invoke = list(addr.invoke_order)
    else:
        base_sig = lattice_signature(ps, n=n)
        distinct = sorted(set(ps))
        meet_composite = reduce(mul, distinct, 1)
        corridor_key = (meet_composite, qid, n)
        canonical_pair = list(distinct)
        invoke = list(invoke_order) if invoke_order is not None else list(ps)

    off_sig = offset_signature(base_sig, origin)

    bit_identical = origin.depth == 0 and base_sig == off_sig

    return {
        "corridor_key": list(corridor_key),
        "meet_composite": meet_composite,
        "canonical_pair": canonical_pair,
        "invoke_order": invoke,
        "quadrant": qid,
        "transgressor_n": n,
        "origin_id": origin.id,
        "origin_coord": list(origin.coord),
        "origin_depth": origin.depth,
        "lattice_L01_base": list(base_sig[0]),
        "lattice_L01_offset": list(off_sig[0]),
        "corridor_key_unchanged_by_origin": True,
        "regenerate_bit_identical_depth_0": bit_identical,
        "stored_row": False,
        "k": len(ps),
    }


def explain_address_with_origin(
    addr: FreeTokenAddress,
    *,
    origin_id: str | None = None,
    tree: OriginTree | None = None,
) -> dict:
    """Attach origin offset to an existing FreeTokenAddress explain payload."""
    tree = tree or default_origin_tree()
    origin = resolve_origin(tree, origin_id)
    base = addr.explain()
    off = offset_signature(addr.lattice_signature, origin)
    base["origin_id"] = origin.id
    base["origin_coord"] = list(origin.coord)
    base["lattice_L01_offset"] = list(off[0])
    base["regenerate_bit_identical_depth_0"] = (
        origin.depth == 0 and addr.lattice_signature == off
    )
    return base
