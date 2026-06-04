"""
AETHOS → Hilbert space structure.

Maps the lattice core (32 wings, k-depth, origins, meets, correlations)
to a separable Hilbert space picture with extra features beyond R³.

State label (basis index):
  (origin_id, wing, chain_species, chain, n, branch, perm_index)

Features that enlarge plain coordinate space:
  1. Direct sum: 32 wing subspaces per bank (orthogonal by wing unless meet)
  2. Regime segments: k+1 sectors per chain (FSM orthogonality optional)
  3. Branch fan: VA1–VA4 as related subspaces (4 phases)
  4. Meet quotient: identify |ψ⟩ at meet coordinates (same ray)
  5. Origin tree: tensor product across 3^d rooms
  6. k! fiber: order side-channel (path degree of freedom)
  7. L4–L9: correlation inner product on token/symbol layer
  8. Species catalog: countable family of chain Hilbert spaces
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator, Sequence

from aethos_core import AethosLatticeCore, formula_coord
from aethos_lattice import BranchKind, LatticeBank32, LatticeId, apply_vector, lattice_id_parts
from aethos_origins import OriginTree
from aethos_permutation import ordered_permutation_list, side_offset
from aethos_recursive import LatticeBank32K, segment_index
from aethos_sequences import SequenceKind, canon_on_chain, make_chain


class BasisKind(Enum):
    """Which layer of the full Hilbert tower labels this basis vector."""

    WING = "wing"           # pure formula wing state
    MEET = "meet"             # identified at collision coordinate
    ORDER = "order"           # wing + permutation fiber
    ORIGIN = "origin"         # global origin offset included
    CORRELATION = "correlation"  # L4–L6 edge state
    CATEGORY = "category"     # L7–L9 subspace


@dataclass(frozen=True)
class BasisLabel:
    """One addressable basis direction in the AETHOS Hilbert tower."""

    origin_id: str = "O0"
    wing: int = 1  # LatticeId 1..32
    species: SequenceKind = SequenceKind.PRIMES
    chain: tuple[int, ...] = ()
    n: int = 0
    branch: BranchKind = BranchKind.VA1
    perm_index: int = 0
    kind: BasisKind = BasisKind.WING

    def key(self) -> str:
        ch = ",".join(map(str, self.chain))
        return f"{self.origin_id}|L{self.wing:02d}|{self.species.value}|({ch})|n={self.n}|{self.branch.name}|π={self.perm_index}|{self.kind.value}"


@dataclass
class LatticeState:
    """
    Finite support superposition in a truncated basis.
    coeffs are complex amplitudes (real for demo).
    """

    amplitudes: dict[str, complex] = field(default_factory=dict)
    labels: dict[str, BasisLabel] = field(default_factory=dict)

    def add(self, label: BasisLabel, amp: complex = 1.0) -> None:
        k = label.key()
        self.labels[k] = label
        self.amplitudes[k] = self.amplitudes.get(k, 0) + amp

    def norm(self) -> float:
        return math.sqrt(sum(abs(a) ** 2 for a in self.amplitudes.values()))

    def normalize(self) -> LatticeState:
        n = self.norm()
        if n == 0:
            return self
        for k in self.amplitudes:
            self.amplitudes[k] /= n
        return self


def formula_coord_branch(
    chain: Sequence[int],
    n: int,
    branch: BranchKind,
    wing: int = 1,
) -> tuple[float, float, float]:
    """Canonical coord on explicit branch + wing (not only wing's default branch)."""
    _, vector = lattice_id_parts(LatticeId(wing))
    canon = canon_on_chain(branch, chain, n)
    return apply_vector(canon, vector)


def coord_for_label(label: BasisLabel, tree: OriginTree | None = None) -> tuple[float, float, float]:
    """Embed basis label into R³ (local + origin + optional order fiber)."""
    local = formula_coord_branch(label.chain, label.n, label.branch, label.wing)
    if label.perm_index and len(label.chain) >= 2:
        sorted_p = tuple(sorted(label.chain))
        off = side_offset(len(sorted_p), label.perm_index)
        local = (local[0] + off[0], local[1] + off[1], local[2] + off[2])
    if tree is None:
        return local
    origin = next(o for o in tree.walk() if o.id == label.origin_id)
    ox, oy, oz = origin.coord
    return (ox + local[0], oy + local[1], oz + local[2])


def inner_product(a: LatticeState, b: LatticeState) -> complex:
    """⟨a|b⟩ on shared basis keys."""
    keys = set(a.amplitudes) & set(b.amplitudes)
    return sum(a.amplitudes[k].conjugate() * b.amplitudes[k] for k in keys)


def meet_identify(states: list[LatticeState], coords: list[tuple[float, float, float]], tol: float = 1e-9) -> LatticeState:
    """
    Quotient: merge amplitudes of states that share the same coordinate (meet).
    Returns single state on MEET-identified basis.
    """
    merged = LatticeState()
    bucket: dict[tuple[float, float, float], complex] = {}
    for st, c in zip(states, coords):
        key = (round(c[0], 6), round(c[1], 6), round(c[2], 6))
        amp = sum(st.amplitudes.values())
        bucket[key] = bucket.get(key, 0) + amp
    for i, (c, amp) in enumerate(bucket.items()):
        lbl = BasisLabel(kind=BasisKind.MEET, n=i)
        merged.add(lbl, amp)
    return merged


@dataclass
class HilbertSpaceReport:
    """Structural counts for the full tower (truncated window)."""

    wing_dim_per_bank: int = 32
    segments_per_chain_k: int = 0
    branch_dim: int = 4
    perm_fiber_k: int = 0
    origin_rooms: int = 0
    species_count: int = 0
    truncated_basis_size: int = 0
    meet_classes_same_n: int = 0
    interior_plateau_width: int = 0
    notes: tuple[str, ...] = ()

    def summary(self) -> str:
        lines = [
            "AETHOS Hilbert space structure (truncated window)",
            f"  wing subspaces per bank:     {self.wing_dim_per_bank}",
            f"  regime segments (k+1):       {self.segments_per_chain_k}",
            f"  branch fan (VA1-VA4):        {self.branch_dim}",
            f"  permutation fiber (k!):      {self.perm_fiber_k}",
            f"  origin rooms (3^d):          {self.origin_rooms}",
            f"  anchor species (countable):  {self.species_count}",
            f"  truncated |basis| estimate: {self.truncated_basis_size:,}",
            f"  same-n meet classes:         {self.meet_classes_same_n}",
            f"  interior Z plateau width:    {self.interior_plateau_width}",
            "",
            "Feature tower (beyond R³):",
        ]
        for n in self.notes:
            lines.append(f"  • {n}")
        return "\n".join(lines)


def estimate_hilbert_tower(
    *,
    chain_k: int = 5,
    n_max: int = 50,
    origin_depth: int = 3,
    species: tuple[SequenceKind, ...] = (
        SequenceKind.PRIMES,
        SequenceKind.EVENS,
        SequenceKind.POWERS_OF_2,
        SequenceKind.SQUARES,
        SequenceKind.SQRT_SCALED,
    ),
) -> HilbertSpaceReport:
    """
    Count structural dimensions. Full space is infinite (n→∞, k→∞, depth→∞);
    this estimates a finite window |basis|.
    """
    chain = make_chain(SequenceKind.PRIMES, chain_k)
    k = len(chain)
    segments = k + 1

    from math import factorial

    perm_fiber = factorial(min(k, 6))  # cap demo

    tree = OriginTree.bootstrap(max_depth=origin_depth)
    origins = tree.root.count_descendant_origins()

    # per species: origins × wings × n_window × branches × segments × perm
    per_species = origins * 32 * n_max * 4 * perm_fiber
    truncated = per_species * len(species)

    # meet degeneracy at sample point
    bank = LatticeBank32.single_prime(chain[0] if chain else 3)
    collisions = len(bank.find_same_n_collisions(min(n_max, 7)))

    # plateau width
    from aethos_recursive import canon_recursive

    plateau = [
        n
        for n in range(1, max(chain) + 5)
        if k >= 3 and canon_recursive(BranchKind.VA1, chain, n)[2] == sum(chain)
    ]

    notes = (
        "Direct sum: 32 wing subspaces per bank (H = sum_w H_w)",
        "Regime FSM: k+1 sectors; interior Z plateau = bulk subspace",
        "Branch fan: 4 VA phases (polarization / isospin without new primes)",
        "Meet quotient: coordinate collision identifies states (same-ocean)",
        "Origin tree: tensor product across 3^d rooms -- dimensionless depth",
        "k! order fiber: path history orthogonal to sorted multiset content",
        "Countable species: sum over species -- infinite lattice types",
        "L4-L9 (token layer): correlation inner product on sparse prime weights",
        "Transgressor n->inf + extend k: separable infinite-dimensional completion",
    )

    return HilbertSpaceReport(
        segments_per_chain_k=segments,
        perm_fiber_k=perm_fiber,
        origin_rooms=origins,
        species_count=len(species),
        truncated_basis_size=truncated,
        meet_classes_same_n=collisions,
        interior_plateau_width=len(plateau),
        notes=notes,
    )


def wing_subspace_states(chain: tuple[int, ...], n: int) -> list[LatticeState]:
    """Orthonormal-ish basis: one unit vector per wing at fixed (chain, n)."""
    states = []
    for lid in LatticeId:
        lbl = BasisLabel(wing=int(lid), chain=chain, n=n, branch=lattice_id_parts(lid)[0])
        st = LatticeState()
        st.add(lbl, 1.0)
        states.append(st)
    return states


def branch_fan_states(chain: tuple[int, ...], n: int, wing: int = 1) -> list[LatticeState]:
    """Four branch phases on one wing — subspace fan."""
    out = []
    for b in BranchKind:
        lbl = BasisLabel(wing=wing, chain=chain, n=n, branch=b)
        st = LatticeState()
        st.add(lbl, 1.0)
        out.append(st)
    return out


def perm_fiber_states(chain: tuple[int, ...], n: int, wing: int = 1) -> list[LatticeState]:
    """k! order directions on sorted multiset."""
    sorted_p = tuple(sorted(chain))
    perms = ordered_permutation_list(sorted_p)
    out = []
    for i, _ in enumerate(perms):
        lbl = BasisLabel(wing=wing, chain=chain, n=n, perm_index=i, kind=BasisKind.ORDER)
        st = LatticeState()
        st.add(lbl, 1.0)
        out.append(st)
    return out


def demonstrate_orthogonality() -> dict[str, float]:
    """Sample inner products showing wing vs branch structure."""
    chain = (3, 5, 7)
    n = 5
    wings = wing_subspace_states(chain, n)
    # different wings same n — generally orthogonal in basis labels
    ip_w1_w2 = inner_product(wings[0], wings[1])
    ip_w1_w1 = inner_product(wings[0], wings[0])

    fan = branch_fan_states(chain, n)
    ip_b1_b2 = inner_product(fan[0], fan[1])

    # meet-identified: two wings that collide share coordinate
    bank = LatticeBank32K(chain)
    coords = [bank[LatticeId(i + 1)].at(n) for i in range(32)]
    collision_groups: dict[tuple[float, float, float], list[int]] = {}
    for i, c in enumerate(coords):
        collision_groups.setdefault(c, []).append(i + 1)
    multi = [g for g in collision_groups.values() if len(g) > 1]

    return {
        "wing_self": abs(ip_w1_w1),
        "wing_01_02_overlap": abs(ip_w1_w2),
        "branch_VA1_VA2_overlap": abs(ip_b1_b2),
        "collision_groups_at_n": len(multi),
    }


def correlation_inner_product(
    weights_a: dict[int, float],
    weights_b: dict[int, float],
) -> float:
    """L7–L9 sparse prime-weight inner product (semantic layer)."""
    keys = set(weights_a) & set(weights_b)
    return sum(weights_a[k] * weights_b[k] for k in keys)


def full_tower_demo() -> None:
    rep = estimate_hilbert_tower()
    print("=" * 70)
    print(rep.summary())
    print()
    orth = demonstrate_orthogonality()
    print("--- Sample inner products (wing / branch basis) ---")
    for k, v in orth.items():
        print(f"  {k}: {v}")
    print()
    print("--- Branch fan coordinates (3,5,7) n=5 wing L01 ---")
    for b in BranchKind:
        c = formula_coord_branch((3, 5, 7), 5, b, wing=1)
        print(f"  {b.name}: {c}")
    print()
    print("--- Permutation fiber k=3 (6 directions) ---")
    perms = perm_fiber_states((3, 5, 7), 5)
    print(f"  fiber dimension: {len(perms)}")
    print()
    print("--- Origin tensor: rooms × wings ---")
    tree = OriginTree.bootstrap(max_depth=2)
    print(f"  origins: {tree.root.count_descendant_origins()}, wing-rooms: {tree.lattice_count_estimate()}")


if __name__ == "__main__":
    full_tower_demo()
