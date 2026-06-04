"""
100 active nodes — finite seeds, infinite deterministic positions.

Model (user specification):
  - 100 active nodes, each on one of 8 main vectors, 4-way branch (32 wings)
  - Each node carries an anchor chain (solo / pair / triple / k-chain / 4-way VA fan)
  - Anchor chains may use ANY strictly increasing countable set — not only primes
    (see aethos_sequences.SequenceKind: primes, evens, 2^n, Fibonacci, custom, …)
  - Different active sets on the same wing topology build different correlations
  - Every origin opens 3 new dimensions; each dimension gets full 8×4 on its chain set
  - Same recursive law → combinatorial address space far larger than node count
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Iterator, Sequence

from aethos_lattice import BranchKind, LatticeId, apply_vector, lattice_id_parts
from aethos_origins import CHILD_DIMENSIONS, DimSlot, Origin, OriginTree
from aethos_recursive import LatticeBank32K, canon_recursive, segment_index
from aethos_sequences import SequenceKind, canon_on_chain, make_chain

try:
    from aethos_blob import ElectronBlob, assign_species
except ImportError:  # pragma: no cover
    ElectronBlob = None  # type: ignore
    assign_species = None  # type: ignore

VECTORS_PER_NODE = 8
BRANCHES_PER_VECTOR = 4
WINGS_PER_ROOM = VECTORS_PER_NODE * BRANCHES_PER_VECTOR  # 32
DIMENSIONS_PER_ORIGIN = 3


class ActiveRole(IntEnum):
    """How an active node participates in anchor-depth branching (any chain species)."""

    SOLO = 1          # single anchor in chain
    PAIR = 2          # two-anchor chain (|chain|=2)
    TRIPLE = 3        # three-anchor chain (|chain|=3)
    K_CHAIN = 4       # general k-anchor chain
    FOUR_WAY = 5      # explicit VA1–VA4 fan on same anchor


@dataclass(frozen=True)
class ActiveNode:
    """
    One of 100 (or N) active seeds. Deterministic address = f(node, n, origin offset).
    """

    node_id: int
    origin_path: str  # e.g. "O0.D1.D2"
    vector_index: int  # 1..8
    branch: BranchKind
    chain: tuple[int, ...]
    role: ActiveRole
    dim_slot: DimSlot | None = None
    chain_species: SequenceKind = SequenceKind.PRIMES

    @property
    def wing_key(self) -> str:
        return f"v{self.vector_index}_{self.branch.name}"

    def address(self, n: int, origin_coord: tuple[float, float, float] = (0, 0, 0)) -> tuple[float, float, float]:
        from aethos_lattice import VECTORS

        v = VECTORS[self.vector_index - 1]
        canon = canon_on_chain(self.branch, self.chain, n)
        cx, cy, cz = apply_vector(canon, v)
        ox, oy, oz = origin_coord
        return (ox + cx, oy + cy, oz + cz)


@dataclass
class ActiveNetwork100:
    """
    N active nodes spread across origin tree, 8 vectors, 4 branches,
    mixed anchor depths and 3D origin slots.

    chain_species selects the default anchor set (primes, evens, custom, …);
    any strictly increasing chain is valid for correlation structure (C6).
    """

    nodes: list[ActiveNode] = field(default_factory=list)
    origin_tree: OriginTree | None = None
    _origin_index: dict[str, Origin] = field(default_factory=dict)

    @classmethod
    def bootstrap(
        cls,
        count: int = 100,
        origin_max_depth: int = 3,
        chain_species: SequenceKind = SequenceKind.PRIMES,
    ) -> ActiveNetwork100:
        tree = OriginTree.bootstrap(max_depth=origin_max_depth)
        net = cls(nodes=[], origin_tree=tree)
        net._origin_index = {o.id: o for o in tree.walk()}

        origins = list(tree.walk())
        if len(origins) < count:
            # reuse origins with different vector/branch/chain/n roles
            pass

        primes = make_chain(chain_species, 12)
        for i in range(count):
            origin = origins[i % len(origins)]
            vec = (i % 8) + 1
            branch = BranchKind((i % 4) + 1)
            role, chain = net._assign_role(i, primes)
            net.nodes.append(
                ActiveNode(
                    node_id=i,
                    origin_path=origin.id,
                    vector_index=vec,
                    branch=branch,
                    chain=chain,
                    role=role,
                    dim_slot=origin.dim_slot,
                    chain_species=chain_species,
                )
            )
        return net

    @classmethod
    def bootstrap_from_blob(
        cls,
        blob: ElectronBlob,
        *,
        count: int = 100,
        origin_max_depth: int = 3,
    ) -> ActiveNetwork100:
        """Bootstrap with per-node anchor set from material blob (C6)."""
        if assign_species is None:
            raise RuntimeError("aethos_blob required")
        tree = OriginTree.bootstrap(max_depth=origin_max_depth)
        net = cls(nodes=[], origin_tree=tree)
        net._origin_index = {o.id: o for o in tree.walk()}
        origins = list(tree.walk())
        for i in range(count):
            origin = origins[i % len(origins)]
            species = assign_species(blob, i, origin_depth=origin.depth)
            pool = make_chain(species, 12)
            vec = (i % 8) + 1
            branch = BranchKind((i % 4) + 1)
            role, chain = net._assign_role(i, pool)
            net.nodes.append(
                ActiveNode(
                    node_id=i,
                    origin_path=origin.id,
                    vector_index=vec,
                    branch=branch,
                    chain=chain,
                    role=role,
                    dim_slot=origin.dim_slot,
                    chain_species=species,
                )
            )
        return net

    def apply_blob_chains(self, blob: ElectronBlob) -> None:
        """Reassign each node's chain from blob parameters (same topology, new counting set)."""
        if assign_species is None:
            raise RuntimeError("aethos_blob required")
        updated: list[ActiveNode] = []
        for i, node in enumerate(self.nodes):
            origin = self.origin_for(node)
            species = assign_species(blob, i, origin_depth=origin.depth)
            pool = make_chain(species, 12)
            role, chain = self._assign_role(i, pool)
            updated.append(
                ActiveNode(
                    node_id=node.node_id,
                    origin_path=node.origin_path,
                    vector_index=node.vector_index,
                    branch=node.branch,
                    chain=chain,
                    role=role,
                    dim_slot=node.dim_slot,
                    chain_species=species,
                )
            )
        self.nodes = updated

    def _distinct_chain(self, pool: tuple[int, ...], i: int, k: int) -> tuple[int, ...]:
        """Pick k distinct anchors from pool (Fibonacci pools may repeat neighbors)."""
        if k <= 0:
            return ()
        n = len(pool)
        if k == 1:
            return (pool[i % n],)
        picked: list[int] = []
        seen: set[int] = set()
        for j in range(n * 2):
            v = pool[(i + j) % n]
            if v in seen:
                continue
            seen.add(v)
            picked.append(v)
            if len(picked) >= k:
                break
        if len(picked) < k:
            step = max(1, n // max(k, 1))
            picked = []
            seen = set()
            for t in range(n):
                v = pool[(i + t * step) % n]
                if v in seen:
                    continue
                seen.add(v)
                picked.append(v)
                if len(picked) >= k:
                    break
        return tuple(sorted(picked[:k]))

    def _assign_role(self, i: int, primes: tuple[int, ...]) -> tuple[ActiveRole, tuple[int, ...]]:
        r = i % 5
        if r == 0:
            return ActiveRole.SOLO, self._distinct_chain(primes, i, 1)
        if r == 1:
            return ActiveRole.PAIR, self._distinct_chain(primes, i, 2)
        if r == 2:
            return ActiveRole.TRIPLE, self._distinct_chain(primes, i, 3)
        if r == 3:
            k = 3 + (i % 4)
            return ActiveRole.K_CHAIN, self._distinct_chain(primes, i, k)
        k = 2 + (i % 3)
        return ActiveRole.FOUR_WAY, self._distinct_chain(primes, i, k)

    def origin_for(self, node: ActiveNode) -> Origin:
        return self._origin_index[node.origin_path]

    def positions_in_window(self, n_min: int, n_max: int) -> dict[tuple[float, float, float], list[int]]:
        """Map coordinate -> node ids (deterministic; collisions = natural meets)."""
        bucket: dict[tuple[float, float, float], list[int]] = {}
        for n in range(n_min, n_max + 1):
            for node in self.nodes:
                o = self.origin_for(node)
                c = node.address(n, o.coord)
                bucket.setdefault(c, []).append(node.node_id)
        return bucket

    def count_unique_positions(self, n_min: int, n_max: int) -> int:
        return len(self.positions_in_window(n_min, n_max))

    def sweep_transgression(self, n_max: int = 50) -> Iterator[tuple[int, int, tuple[float, float, float]]]:
        """All 100 nodes x each n -> stream of (node_id, n, coord)."""
        for n in range(0, n_max + 1):
            for node in self.nodes:
                o = self.origin_for(node)
                yield node.node_id, n, node.address(n, o.coord)


@dataclass(frozen=True)
class CapacityEstimate:
    """Combinatorial capacity from 100 active nodes (structural, not infinite n)."""

    active_nodes: int
    wings_per_node: int
    dimensions_per_origin: int
    vectors: int
    branches: int

    @property
    def base_wing_states(self) -> int:
        return self.active_nodes * self.wings_per_node

    def origin_rooms_at_depth(self, depth: int) -> int:
        return sum(DIMENSIONS_PER_ORIGIN**i for i in range(depth + 1))

    def structural_slots(self, origin_depth: int, max_chain_k: int, n_window: int) -> int:
        """
        Finite window estimate:
          nodes × wings × n_window × 3^depth × (k-chain segments)
        """
        segments = max_chain_k + 1
        rooms = self.origin_rooms_at_depth(origin_depth)
        return self.active_nodes * self.wings_per_node * n_window * rooms * segments

    def explain(self) -> str:
        return f"""
100 ACTIVE NODES -> DETERMINISTIC POSITION CAPACITY
==================================================
Per node:
  - 8 main vectors x 4-way branch = {self.wings_per_node} wings
  - Anchor chains (solo / pair / triple / k-chain / 4-way VA fan) on any SequenceKind
  - Sits on an origin; origin spawns {self.dimensions_per_origin} child dimensions
  - Each child dimension = new full 8 x 4 room on its anchor set

Finite window (example depth=5, k<=12, n in 0..500):
  Structural slots ~ {self.structural_slots(5, 12, 500):,}

As n -> infinity:
  Each wing produces infinitely many distinct addresses (regime changes at every anchor).

With only {self.active_nodes} active nodes:
  Base wing assignments     = {self.base_wing_states:,}
  x transgression (unbounded n)
  x chain extension (countable anchors)
  x origin tree (3^d rooms)
  => infinite deterministic positions from finite seeds.
"""


def deterministic_id(coord: tuple[float, float, float], node_id: int, n: int) -> str:
    """Stable hash id for a position witness."""
    raw = f"{coord}|{node_id}|{n}".encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def demo() -> None:
    print("=" * 60)
    print("100 ACTIVE NODES — DETERMINISTIC INFINITY FROM FINITE SEEDS")
    print("=" * 60)

    net = ActiveNetwork100.bootstrap(count=100, origin_max_depth=3)
    cap = CapacityEstimate(
        active_nodes=100,
        wings_per_node=WINGS_PER_ROOM,
        dimensions_per_origin=DIMENSIONS_PER_ORIGIN,
        vectors=VECTORS_PER_NODE,
        branches=BRANCHES_PER_VECTOR,
    )
    print(cap.explain())

    roles = {}
    for node in net.nodes:
        roles[node.role.name] = roles.get(node.role.name, 0) + 1
    print("  Role mix among 100 nodes:", dict(sorted(roles.items())))

    for n_win in (10, 50, 200):
        u = net.count_unique_positions(0, n_win)
        samples = 100 * (n_win + 1)
        print(f"\n  n in 0..{n_win}: {samples:,} deterministic samples -> {u:,} unique positions")

    # transgression sweep sample
    stream = list(net.sweep_transgression(n_max=20))
    unique_coords = len({c for _, _, c in stream})
    print(f"\n  Sweep: 100 nodes x n=0..20 = {len(stream):,} deterministic samples")
    print(f"  Unique coordinates in sweep:  {unique_coords:,}")

    # Show one node across segments (prime-by-prime depth)
    node = next(n for n in net.nodes if n.role == ActiveRole.TRIPLE)
    o = net.origin_for(node)
    print(f"\n  Sample node {node.node_id} TRIPLE chain={node.chain} @ {node.origin_path}")
    for n in (2, 3, 5, 7, 10):
        seg = segment_index(node.chain, n)
        print(f"    n={n} seg={seg}  {node.address(n, o.coord)}")

    print("\n  Origin tree + 100 nodes:")
    print(f"    Origins available: {len(net._origin_index)}")
    print(f"    Each origin: {DIMENSIONS_PER_ORIGIN} child dims x {WINGS_PER_ROOM} wings")
    print(f"    Depth-5 full tree would add 364 rooms x 32 wings without more nodes")


if __name__ == "__main__":
    demo()
