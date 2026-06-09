"""
BIT 9 — Query Cell Profile

Bridges BIT 1–2 (routing) and hub scoring (aethos_hub_signature).

Contract:
  QueryCellProfile unifies the two parallel query representations:
    - QueryProfile (coord/lattice_address) used by score_document
    - query_attractor_keys (κ/cells) used by BIT 4 router
  One profile object feeds both routing AND scoring (BIT 10 wires scoring).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from aethos_physics import SpacetimeCell
from aethos_promotion import is_stopword
from aethos_tokenize import tokenize_words
from pipeline.bit_01_word_cell import (
    DEFAULT_ANCHOR_N,
    hub_formula_coord,
    word_to_spacetime_cell,
)
from pipeline.bit_02_attractor_key import AttractorKey
from pipeline.bit_04_candidate_router import (
    DEFAULT_RADIUS,
    query_attractor_keys,
    query_words_for_routing,
)
from pipeline.bit_05_z_band import band_profile_for_cell


def bm25_idf(word: str, doc_freq: dict[str, int], n_docs: int) -> float:
    """
    BM25-style IDF — must match aethos_hub_signature.build_query_profile.

    log((N - df + 0.5) / (df + 0.5) + 1)
    """
    df = max(doc_freq.get(word, 0), 0)
    return math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)


@dataclass
class QueryCellProfile:
    """Unified query geometry: cells, κ sets, and IDF-weighted z_obs."""

    words: list[str]
    word_set: frozenset[str]
    cells: dict[str, SpacetimeCell]
    kappa_q: set[AttractorKey]
    kappa_neighbor_q: set[AttractorKey]
    z_obs_q: float
    idf: dict[str, float]
    band_ids: dict[str, int]
    routed_words: list[str] = field(default_factory=list)


def build_query_cell_profile(
    registry,
    query: str,
    *,
    neighbor_map: dict[str, dict[str, float]],
    doc_freq: dict[str, int],
    n_docs: int,
    n: int = DEFAULT_ANCHOR_N,
    expand_neighbors: bool = True,
    radius: int = DEFAULT_RADIUS,
) -> QueryCellProfile:
    """
    Build unified query cell profile from tokenized query text.

    ``doc_freq`` is document frequency (# docs containing w), not term count.
    """
    words = tokenize_words(query)
    idf: dict[str, float] = {}
    for w in words:
        if not w.isalpha():
            continue
        idf[w] = bm25_idf(w, doc_freq, n_docs)

    routed = query_words_for_routing(words)
    cells: dict[str, SpacetimeCell] = {}
    band_ids: dict[str, int] = {}
    z_obs_q = 0.0

    for w in routed:
        try:
            cell = word_to_spacetime_cell(registry, w, n=n)
        except Exception:
            continue
        cells[w] = cell
        prof = band_profile_for_cell(cell)
        band_ids[w] = prof.band_id
        z_obs_q += idf.get(w, 1.0) * prof.z_obs

    kappa_q = query_attractor_keys(
        registry,
        words,
        n=n,
        radius=radius,
        neighbor_map=neighbor_map,
        expand_neighbors=False,
    )
    kappa_neighbor_q = query_attractor_keys(
        registry,
        words,
        n=n,
        radius=radius,
        neighbor_map=neighbor_map,
        expand_neighbors=expand_neighbors,
    )

    return QueryCellProfile(
        words=words,
        word_set=frozenset(w for w in words if w.isalpha()),
        cells=cells,
        kappa_q=kappa_q,
        kappa_neighbor_q=kappa_neighbor_q,
        z_obs_q=z_obs_q,
        idf=idf,
        band_ids=band_ids,
        routed_words=routed,
    )


def cells_match_hub_gate(
    registry,
    profile: QueryCellProfile,
    *,
    n: int = DEFAULT_ANCHOR_N,
    tol: float = 1e-9,
) -> list[str]:
    """Return failure messages when profile cells disagree with BIT 1 hub coords."""
    failures: list[str] = []
    for w, cell in profile.cells.items():
        try:
            x, y, z = hub_formula_coord(registry, w, n=n)
        except Exception as exc:
            failures.append(f"{w}: hub coord error: {exc}")
            continue
        if abs(cell.z.real - x) > tol or abs(cell.z.imag - y) > tol:
            failures.append(f"{w}: z mismatch cell={cell.z!r} hub=({x},{y})")
        if abs(cell.zeta - z) > tol:
            failures.append(f"{w}: zeta mismatch cell={cell.zeta} hub={z}")
    return failures


def verify_bit09_gate(
    registry,
    *,
    neighbor_map: dict[str, dict[str, float]],
    doc_freq: dict[str, int],
    n_docs: int,
    index=None,
    query: str = "phone technical software",
    tol: float = 1e-9,
) -> tuple[bool, list[str]]:
    """
    BIT 9 gate:
      - Profile builds without error
      - kappa_q ⊆ kappa_neighbor_q
      - z_obs_q is finite and positive for content queries
      - cells match BIT 1 hub gate
      - κ Jaccard ranks related doc above unrelated (when index supplied)
    """
    failures: list[str] = []

    try:
        profile = build_query_cell_profile(
            registry,
            query,
            neighbor_map=neighbor_map,
            doc_freq=doc_freq,
            n_docs=n_docs,
        )
    except Exception as exc:
        return False, [f"build error: {exc}"]

    if not profile.cells:
        failures.append("no cells built for routed query words")

    if not profile.kappa_q.issubset(profile.kappa_neighbor_q):
        failures.append("kappa_q not subset of kappa_neighbor_q")

    if not math.isfinite(profile.z_obs_q):
        failures.append(f"z_obs_q not finite: {profile.z_obs_q}")
    elif profile.cells and profile.z_obs_q <= 0.0:
        failures.append(f"z_obs_q expected positive, got {profile.z_obs_q}")

    failures.extend(cells_match_hub_gate(registry, profile, tol=tol))

    if index is not None and profile.kappa_q:
        related = index.score_doc_overlap(profile.kappa_q, "d0")
        unrelated = index.score_doc_overlap(profile.kappa_q, "d3")
        if related <= unrelated:
            failures.append(
                f"κ overlap gate: related d0={related:.3f} "
                f"<= unrelated d3={unrelated:.3f}"
            )

    return len(failures) == 0, failures
