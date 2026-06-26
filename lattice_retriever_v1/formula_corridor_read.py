"""
Formula corridor read — zero stored coordinates.

Vectors regenerate from primes. Formula branches at each n, wing×case readout
disambiguates symbols. When uniquely determined, no inverted index needed.
"""

from __future__ import annotations

from collections import Counter, defaultdict

from lattice_retriever_v1.electron_lattice_codec import wing_case_to_coin
from lattice_retriever_v1.intersection_dot_codec import (
    SymbolAlphabet,
    _oriented_pair_catalog,
    dot_on_origin,
    read_document_from_walk,
)
from lattice_retriever_v1.wing_channel_codec import wing_channel_at


def _wing_disambiguate(
    alpha: SymbolAlphabet,
    prefix: bytes,
    candidates: list[int],
) -> list[int]:
    if len(candidates) <= 1 or not prefix:
        return candidates
    current = prefix[-1]
    counts = Counter(prefix)
    rail_n = len(prefix)
    narrowed: list[int] = []
    for s in candidates:
        ch = wing_channel_at(alpha, current, s, n=rail_n, sym_counts=counts)
        coin = int(wing_case_to_coin(ch.case, ch.wing))
        if alpha.symbols[coin % alpha.n] == s:
            narrowed.append(s)
    return narrowed if narrowed else candidates


def formula_corridor_read(alpha: SymbolAlphabet, count: int) -> bytes | None:
    """Incremental formula walk — no coords stored."""
    if count <= 0:
        return b""
    syms = alpha.symbols
    if len(syms) == 1:
        return bytes([syms[0]]) * count
    catalog = _oriented_pair_catalog(alpha)
    cat_index = {k: i for i, k in enumerate(catalog)}
    out = bytearray()
    dots: list = []
    pair_counts: dict[tuple[int, int], int] = defaultdict(int)
    for pos in range(count):
        if pos == 0:
            out.append(syms[0])
            continue
        current = out[-1]
        cands: list[int] = []
        for s in syms:
            key = (current, s)
            pn = pair_counts[key] + 1
            origin = catalog[cat_index[(current, s)]] if (current, s) in cat_index else None
            if origin is None:
                continue
            trial_dots = dots + [
                dot_on_origin(origin, pair_n=pn, walk_index=len(dots))
            ]
            trial = bytes(out) + bytes([s])
            if read_document_from_walk(tuple(trial_dots)) == trial:
                cands.append(s)
        cands = _wing_disambiguate(alpha, bytes(out), cands)
        if len(cands) != 1:
            if len(cands) > 1:
                cands = [min(cands, key=lambda x: alpha.prime_for(x))]
            else:
                return None
        s = cands[0]
        key = (current, s)
        pn = pair_counts[key] + 1
        pair_counts[key] = pn
        origin = next(k for k in catalog if k.left_byte == current and k.right_byte == s)
        dots.append(dot_on_origin(origin, pair_n=pn, walk_index=len(dots)))
        out.append(s)
    return bytes(out)


def formula_can_lossless_read(alpha: SymbolAlphabet, data: bytes) -> bool:
    trial = formula_corridor_read(alpha, len(data))
    return trial == data
