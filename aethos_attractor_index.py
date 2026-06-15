"""
Corpus attractor index — retrieve by (z, zeta) spring clusters.

Compatibility re-exports from pipeline BIT 2–3. Prefer:
  pipeline.bit_02_attractor_key
  pipeline.bit_03_doc_attractor_set
"""

from __future__ import annotations

from aethos_complex_plane import ComplexPlane3D
from aethos_hub_signature import HubEntry, LatticeHubSignature
from aethos_lattice import BranchKind
from aethos_notch_encoder import NotchFingerprint
from aethos_physics import SpacetimeCell
from pipeline.bit_01_word_cell import word_to_spacetime_cell
from pipeline.bit_02_attractor_key import (
    AttractorKey,
    DEFAULT_QUANTIZE,
    attractor_neighbors,
    kappa,
    kappa_from_cell,
    verify_bit02_gate,
)
from pipeline.bit_03_doc_attractor_set import (
    CorpusAttractorIndex,
    DocAttractorSet,
    build_attractor_index_from_corpus,
    build_attractor_index_from_hub_signatures,
    doc_attractor_set_from_signature,
    verify_bit03_gate,
)

# Legacy names (BIT 2 aliases)
spring_attractor_key = kappa
cell_attractor_key = kappa_from_cell


def hub_entry_spacetime_cell(
    entry: HubEntry,
    *,
    n: int | None = None,
    branch: BranchKind = BranchKind.VA1,
    wing: int = 1,
) -> SpacetimeCell:
    """Legacy: rebuild cell from stored hub coord (prefer word_to_spacetime_cell)."""
    if not entry.coord:
        raise ValueError(
            "hub_entry_spacetime_cell: pin-wire entry has no coord; "
            "use word_to_spacetime_cell(registry, entry.word)"
        )
    x, y, z = entry.coord
    psi = ComplexPlane3D(z=complex(x, y), zeta=float(z))
    rail_n = float(n if n is not None else max(entry.coord))
    chain: tuple[int, ...]
    if entry.lattice_composite > 1:
        from aethos_intersection_nodes import chain_from_composite

        chain = chain_from_composite(entry.lattice_composite)
    else:
        chain = (entry.prime,) if entry.prime >= 2 else ()
    return SpacetimeCell.from_psi(
        psi,
        rail_n,
        chain=chain,
        branch=branch,
        wing=wing,
    )


def spacetime_cell_for_word(
    registry,
    word: str,
    *,
    n: int = 7,
    branch: BranchKind = BranchKind.VA1,
    wing: int = 1,
) -> SpacetimeCell:
    """Spring cell at transgressor n — BIT 1 lattice chain (hub-aligned)."""
    return word_to_spacetime_cell(registry, word, n=n, branch=branch, wing=wing)


def merge_attractor_indices(*indices: CorpusAttractorIndex) -> CorpusAttractorIndex:
    """Union multiple indices (e.g. hub + notch views)."""
    if not indices:
        return CorpusAttractorIndex()
    q = indices[0].quantize
    merged = CorpusAttractorIndex(quantize=q, anchor_n=indices[0].anchor_n)
    for idx in indices:
        for doc_id, keys in idx.doc_keys.items():
            witnesses = idx.doc_witnesses.get(doc_id, {})
            for key in keys:
                merged.add(doc_id, key, witnesses.get(key, ""))
    return merged


@classmethod
def _from_hub_signatures(
    cls,
    signatures: dict[str, LatticeHubSignature],
    registry,
    *,
    quantize: float = DEFAULT_QUANTIZE,
    n: int = 7,
    strength_tau: float = 0.0,
) -> CorpusAttractorIndex:
    return build_attractor_index_from_hub_signatures(
        registry,
        signatures,
        n=n,
        quantize=quantize,
        strength_tau=strength_tau,
    )


CorpusAttractorIndex.from_hub_signatures = _from_hub_signatures  # type: ignore[method-assign]


@classmethod
def _from_notch_fingerprints(
    cls,
    fingerprints: dict[str, NotchFingerprint],
    registry,
    *,
    quantize: float = DEFAULT_QUANTIZE,
) -> CorpusAttractorIndex:
    idx = cls(quantize=quantize)
    for doc_id, fp in fingerprints.items():
        if not fp.top_hub:
            continue
        cell = spacetime_cell_for_word(registry, fp.top_hub, n=fp.n_anchors)
        key = kappa_from_cell(cell, quantize=quantize)
        idx.add(doc_id, key, fp.top_hub)
    return idx


CorpusAttractorIndex.from_notch_fingerprints = _from_notch_fingerprints  # type: ignore[method-assign]


def demo() -> None:
    print("=" * 72)
    print("CORPUS ATTRACTOR INDEX — (z, zeta) buckets")
    print("=" * 72)

    triple = SpacetimeCell.promote_witness((3, 5, 7), (3, 5))
    key = kappa_from_cell(triple)
    print(f"\nTriple witness cell: z={triple.z} zeta={triple.zeta} key={key}")

    ok, errs = verify_bit02_gate()
    print(f"BIT 2 gate: {'PASS' if ok else 'FAIL'} {errs}")

    idx = CorpusAttractorIndex()
    idx.add("doc_a", key, "meet")
    idx.add("doc_b", key, "triple")
    idx.add("doc_c", (99, 99, 99), "other")

    hits = idx.query_by_cell(triple, radius=0)
    print(f"query triple -> docs: {hits}")
    print(f"summary: {idx.summary()}")


if __name__ == "__main__":
    demo()
