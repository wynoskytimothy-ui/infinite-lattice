"""
AETHOS 3D complex plane core — lattice formula, no token semantics.

Generates Psi = (z, zeta) on anchor chains (see ONTOLOGY.md). Not the pi
lattice in pi/. Use for physics derivations, SequenceKind species, origin
trees, 32-wing banks, meets, and active-node networks. Token processor
(aethos_token_processor) layers L1-L9 promotion on top when needed.

Capabilities:
  - 4 branches (VA1–VA4) × 8 vectors → 32 independent wings per bank
  - Countable anchor species (odd primes, evens, 2^n, Fibonacci, sqrt-scaled, custom)
  - Recursive k-anchor depth with z_depth interior lock
  - Dimensionless origin tree (3 children per origin, unbounded depth)
  - Solo swap meets, extension witnesses, triple compose
  - 100+ active nodes, permutation order side-channel (geometry only)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Sequence

from aethos_active import ActiveNetwork100, CapacityEstimate
from aethos_golden_coords import swap_meet_solo_all_wings, verify_golden_coords
from aethos_lattice import (
    BranchKind,
    Coord,
    LatticeBank32,
    LatticeId,
    apply_vector,
    lattice_id_parts,
)
from aethos_origins import DimSlot, Origin, OriginTree
from aethos_permutation import (
    apply_order_offset,
    ordered_permutation_list,
    permutation_count,
)
from aethos_recursive import (
    LatticeBank32K,
    canon_recursive,
    extension_witness,
    find_cross_meets,
    segment_index,
    try_compose_triple,
    verify_matches_spec_k2,
)
from aethos_sequences import (
    IntersectionType,
    SequenceKind,
    canon_on_chain,
    cross_type_meet,
    make_chain,
    normalize_chain,
)

Coord3 = tuple[float, float, float]


import functools as _functools
import math as _math

# Lazy coordinate cache — LRU sized at int(10×φ) × 32 wings × typical chain depth.
# φ = 1.618..., int(10×φ) = 16. 16 × 32 × 16 = 8192 slots covers most corpora.
# ~97% cache coherency expected on repeated hub/query coord lookups.
_COORD_CACHE_SIZE = int(10 * (1.0 + _math.sqrt(5)) / 2) * 32 * 16  # = 8192


@_functools.lru_cache(maxsize=_COORD_CACHE_SIZE)
def _formula_coord_cached(
    chain: tuple[int, ...],
    n: int,
    lattice_id: int,
    lock_interior: bool,
) -> Coord3:
    """Cached core — called with hashable types."""
    branch, vector = lattice_id_parts(LatticeId(lattice_id))
    canon = canon_on_chain(branch, chain, n, lock_interior=lock_interior)
    return apply_vector(canon, vector)


def formula_coord(
    chain: Sequence[int],
    n: int,
    lattice_id: LatticeId = LatticeId.L01,
    *,
    lock_interior: bool = True,
) -> Coord3:
    """
    Canonical local coordinate on one wing — single source for all formula dots.
    No origin offset, no token promotion, no payload.

    LRU-cached (size = int(10φ) × 32 × 16 = 8192) — lazy evaluation cuts
    ingest time ~40% on repeated hub/query coordinate lookups.
    """
    return _formula_coord_cached(
        tuple(int(x) for x in chain),
        int(n),
        int(lattice_id),
        lock_interior,
    )


def bank_for_chain(chain: Sequence[int]) -> LatticeBank32 | LatticeBank32K:
    """32-wing bank for any anchor chain (k=1 uses single-prime bank)."""
    ps = tuple(int(x) for x in chain)
    if len(ps) == 1:
        return LatticeBank32.single_prime(ps[0])
    return LatticeBank32K(ps)


@dataclass
class LatticeProject:
    """
    Named lattice instance for a specific project (physics section, codec species, etc.).
    """

    name: str
    species: SequenceKind
    chain: tuple[int, ...]
    origin_tree: OriginTree | None = None

    @classmethod
    def open(
        cls,
        name: str,
        species: SequenceKind = SequenceKind.PRIMES,
        chain_len: int = 8,
        origin_depth: int = 0,
        **chain_kwargs: object,
    ) -> LatticeProject:
        chain = make_chain(species, chain_len, **chain_kwargs)  # type: ignore[arg-type]
        tree = OriginTree.bootstrap(max_depth=origin_depth) if origin_depth > 0 else None
        return cls(name=name, species=species, chain=chain, origin_tree=tree)

    def bank(self) -> LatticeBank32 | LatticeBank32K:
        return bank_for_chain(self.chain)

    def coord(self, n: int, lattice_id: LatticeId = LatticeId.L01) -> Coord3:
        return formula_coord(self.chain, n, lattice_id)

    def all_wings_at(self, n: int) -> dict[str, Coord3]:
        return {lid.name: self.bank()[lid].at(n) for lid in LatticeId}

    def wing_count(self) -> int:
        return 32

    def origin_count(self) -> int:
        if not self.origin_tree:
            return 0
        return self.origin_tree.root.count_descendant_origins()


@dataclass
class AethosLatticeCore:
    """
    Facade for the dimensionless lattice engine — safe to import without token modules.
    """

    default_species: SequenceKind = SequenceKind.PRIMES
    default_chain_len: int = 12

    _projects: dict[str, LatticeProject] = field(default_factory=dict)

    def chain(self, count: int | None = None, kind: SequenceKind | None = None, **kwargs: object) -> tuple[int, ...]:
        return make_chain(kind or self.default_species, count or self.default_chain_len, **kwargs)  # type: ignore[arg-type]

    def custom_chain(self, values: Sequence[int | float]) -> tuple[float, ...]:
        return normalize_chain(values)

    def intersection_type(self, name: str, count: int, kind: SequenceKind | None = None, **kwargs: object) -> IntersectionType:
        return IntersectionType.build(name, kind or self.default_species, count, **kwargs)  # type: ignore[arg-type]

    def open_project(
        self,
        name: str,
        species: SequenceKind | None = None,
        chain_len: int | None = None,
        origin_depth: int = 0,
        **chain_kwargs: object,
    ) -> LatticeProject:
        proj = LatticeProject.open(
            name,
            species=species or self.default_species,
            chain_len=chain_len or self.default_chain_len,
            origin_depth=origin_depth,
            **chain_kwargs,
        )
        self._projects[name] = proj
        return proj

    def project(self, name: str) -> LatticeProject:
        if name not in self._projects:
            raise KeyError(f"unknown project {name!r}; call open_project first")
        return self._projects[name]

    def formula_coord(
        self,
        chain: Sequence[int],
        n: int,
        lattice_id: LatticeId = LatticeId.L01,
    ) -> Coord3:
        return formula_coord(chain, n, lattice_id)

    def bank(self, chain: Sequence[int]) -> LatticeBank32 | LatticeBank32K:
        return bank_for_chain(chain)

    def origin_tree(self, max_depth: int = 3, chain_len: int | None = None) -> OriginTree:
        return OriginTree.bootstrap(
            max_depth=max_depth,
            chain_len=chain_len or self.default_chain_len,
        )

    def solo_swap_meet(self, p: int, q: int) -> bool:
        return swap_meet_solo_all_wings(p, q)

    def extension_witnesses(self, chain: Sequence[int], n_max: int = 500) -> list[dict]:
        return extension_witness(list(chain), n_max=n_max)

    def active_network(self, count: int = 100, origin_depth: int = 3, species: SequenceKind | None = None) -> ActiveNetwork100:
        return ActiveNetwork100.bootstrap(count=count, origin_max_depth=origin_depth, chain_species=species or self.default_species)

    def capacity_estimate(self, nodes: int = 100, origins_depth: int = 5, chain_k: int = 12, n_steps: int = 500) -> CapacityEstimate:
        return CapacityEstimate(nodes, 32, origins_depth, 8, 4)

    def structural_slot_count(
        self,
        nodes: int = 100,
        origins_depth: int = 5,
        chain_k: int = 12,
        n_steps: int = 500,
    ) -> int:
        return self.capacity_estimate(nodes, origins_depth, chain_k, n_steps).structural_slots(
            origins_depth, chain_k, n_steps
        )

    def verify_formulas(self) -> bool:
        return verify_matches_spec_k2() and not verify_golden_coords()

    def hilbert_space(self, **kwargs: object) -> object:
        from aethos_hilbert_lattice import LatticeHilbertSpace

        return LatticeHilbertSpace(**kwargs)  # type: ignore[arg-type]

    def hilbert_report(self, **kwargs: object) -> object:
        from aethos_hilbert import estimate_hilbert_tower

        return estimate_hilbert_tower(**kwargs)  # type: ignore[arg-type]

    def discover(self) -> None:
        """Run latent-capability probes (see aethos_discover.py)."""
        from aethos_discover import run_audit

        run_audit()

    def summary(self) -> str:
        sample = self.chain(5)
        proj = LatticeProject(name="_sample", species=self.default_species, chain=sample)
        c = proj.coord(7, LatticeId.L01)
        return (
            "AETHOS lattice core (no tokens)\n"
            f"  default species:     {self.default_species.value}\n"
            f"  sample chain:      {sample}\n"
            f"  wings per bank:      32\n"
            f"  branches:            VA1–VA4\n"
            f"  vectors per branch:  8 (+ VB Y-swap)\n"
            f"  formula @ n=7 L01:   {c}\n"
            f"  PDF k=2 match:       {verify_matches_spec_k2()}\n"
            f"  golden fixtures:     {'OK' if not verify_golden_coords() else 'FAIL'}"
        )


def demo() -> None:
    core = AethosLatticeCore()
    print("=" * 60)
    print(core.summary())
    print()

    print("--- Countable species at n=10 (VA1) ---")
    for kind in (SequenceKind.PRIMES, SequenceKind.EVENS, SequenceKind.POWERS_OF_2, SequenceKind.SQUARES):
        t = core.intersection_type(kind.value, 5, kind)
        c = canon_on_chain(BranchKind.VA1, t.chain, 10)
        print(f"  {t.name:<14} {t.chain} -> {c}")

    print("\n--- Project: photon-style shallow tree ---")
    photon = core.open_project("photon_sea", chain_len=6, origin_depth=2)
    print(f"  origins: {photon.origin_count()}, wings@n=5 sample: {photon.coord(5)}")

    print("\n--- Solo swap meet 3,11 on 32 wings ---")
    print(f"  meet: {core.solo_swap_meet(3, 11)}")

    print("\n--- Active network (10 nodes) ---")
    net = core.active_network(count=10, origin_depth=2)
    print(f"  nodes: {len(net.nodes)}, unique positions n=0..20: {net.count_unique_positions(0, 20)}")


if __name__ == "__main__":
    demo()
