"""
aethos_recursive_lattice.py - hierarchical lattice with promoted primes.

Each prime is a node. Base primes (chain_primes) are leaves at level 0.
Promoted primes (PROMOTION_POOL) sit at higher levels: each one labels a
chain at the level below, and it participates as an anchor at the level above.

Because the same wing_transform / swap_meet / triple_equalization operators
work at every level (proven empirically by test_recursive_lattice.py:
310/310 identities pass), the hierarchy needs no special-case code per depth.

Use cases:
  - Pattern memory: winning chains promote to new primes; each promoted
    prime's sub-chain is the pattern it represents.
  - Address book: walk_down(prime) recovers the base chain; walk_up(prime)
    finds all promoted patterns containing this prime.
  - Hierarchical lattice: chambers(prime) returns 32 (branch x wing) addresses
    for any node at any level, using the same formula.

The cascade-free property holds at every level:
  - Allocating a new promoted prime never moves any existing prime.
  - All observables are computed fresh from the formula on the chain.
  - Walks are O(depth), not O(N).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from aethos_complex_plane import (
    ComplexPlane3D, swap_meet, triple_equalization, wing_transform,
)
from aethos_lattice import BranchKind
from aethos_promotion import PROMOTION_POOL


@dataclass
class RecursiveNode:
    """A single node in the recursive lattice."""

    prime: int                                # this node's prime ID
    level: int                                # 0 = base, 1+ = promoted
    sub_chain: tuple[int, ...] | None         # at level 1+: the chain this represents
    label: str = ""                           # optional human-readable name
    parents: list[int] = field(default_factory=list)  # promoted primes that include this

    @property
    def is_base(self) -> bool:
        return self.level == 0

    @property
    def is_promoted(self) -> bool:
        return self.level > 0


class RecursiveLattice:
    """Registry of all primes (base + promoted) in a recursive hierarchy."""

    def __init__(self):
        self.nodes: dict[int, RecursiveNode] = {}
        self._next_pool_idx = 0

    # ---- registration ----

    def register_base(self, prime: int, label: str = "") -> RecursiveNode:
        if prime not in self.nodes:
            self.nodes[prime] = RecursiveNode(
                prime=prime, level=0, sub_chain=None, label=label
            )
        elif label and not self.nodes[prime].label:
            self.nodes[prime].label = label
        return self.nodes[prime]

    def promote(self, chain: Iterable[int], label: str = "") -> int:
        """Promote a chain to a new pool prime; level = max(chain levels) + 1.

        Skips pool primes that collide with already-registered base primes
        (the PROMOTION_POOL overlaps with chain_primes for some choices of
        chain length, so collision-checking is required).
        """
        chain_sorted = tuple(sorted(int(p) for p in chain))
        if not chain_sorted:
            raise ValueError("cannot promote empty chain")
        if len(set(chain_sorted)) != len(chain_sorted):
            raise ValueError(f"chain must have unique primes, got {chain_sorted}")

        # find an unused pool prime (skip any already in the lattice)
        new_prime: int | None = None
        while self._next_pool_idx < len(PROMOTION_POOL):
            candidate = PROMOTION_POOL[self._next_pool_idx]
            self._next_pool_idx += 1
            if candidate not in self.nodes:
                new_prime = candidate
                break
        if new_prime is None:
            raise RuntimeError("promotion pool exhausted")

        max_level = 0
        for p in chain_sorted:
            if p not in self.nodes:
                self.register_base(p)
            node = self.nodes[p]
            max_level = max(max_level, node.level)
            if new_prime not in node.parents:
                node.parents.append(new_prime)

        new_node = RecursiveNode(
            prime=new_prime,
            level=max_level + 1,
            sub_chain=chain_sorted,
            label=label,
        )
        self.nodes[new_prime] = new_node
        return new_prime

    def resolve(self, prime: int) -> RecursiveNode:
        if prime not in self.nodes:
            return self.register_base(prime)
        return self.nodes[prime]

    # ---- walks ----

    def walk_down(self, prime: int) -> tuple[int, ...]:
        """Expand any prime down to its base-level chain."""
        node = self.resolve(prime)
        if node.is_base:
            return (prime,)
        result: list[int] = []
        for p in node.sub_chain:
            result.extend(self.walk_down(p))
        return tuple(result)

    def walk_up(self, prime: int) -> list[int]:
        """All promoted primes that transitively contain this prime."""
        seen: set[int] = set()
        result: list[int] = []

        def dfs(p: int):
            for parent in self.resolve(p).parents:
                if parent in seen:
                    continue
                seen.add(parent)
                result.append(parent)
                dfs(parent)

        dfs(prime)
        return result

    # ---- chambers ----

    def chambers(self, prime: int, n: int | None = None) -> list[tuple[BranchKind, int, tuple]]:
        """All 32 chamber addresses (branch x wing) for this prime.

        For a base prime: chain = (prime,), n defaults to prime + 1.
        For a promoted: chain = sub_chain, n defaults to prime.
        """
        node = self.resolve(prime)
        if node.is_base:
            chain = (prime,)
            n_val = n if n is not None else prime + 1
        else:
            chain = node.sub_chain
            n_val = n if n is not None else prime

        chambers: list[tuple[BranchKind, int, tuple]] = []
        for branch in BranchKind:
            for wing in range(1, 9):
                psi = wing_transform(branch, chain, n_val, wing)
                chambers.append((branch, wing, psi.coord))
        return chambers

    # ---- meet algebra at any level ----

    def swap_meet(self, a: int, b: int) -> tuple[ComplexPlane3D, ComplexPlane3D]:
        """swap_meet on any pair of primes (any levels)."""
        return swap_meet(a, b)

    def triple_meet(self, a: int, p: int, q: int) -> dict:
        """triple_equalization on any three primes."""
        return triple_equalization(a, p, q)

    # ---- diagnostics ----

    def render_tree(self, prime: int, depth: int = 0,
                    _seen: set | None = None) -> str:
        if _seen is None:
            _seen = set()
        node = self.resolve(prime)
        indent = "  " * depth
        label = f" '{node.label}'" if node.label else ""
        line = f"{indent}{node.prime}{label} [L{node.level}]"
        lines = [line]
        if node.is_promoted and prime not in _seen:
            _seen.add(prime)
            for p in node.sub_chain:
                lines.append(self.render_tree(p, depth + 1, _seen))
        return "\n".join(lines)

    def stats(self) -> dict:
        level_counts: dict[int, int] = {}
        for n in self.nodes.values():
            level_counts[n.level] = level_counts.get(n.level, 0) + 1
        return {
            "total_nodes": len(self.nodes),
            "level_counts": dict(sorted(level_counts.items())),
            "max_level": max(level_counts.keys()) if level_counts else 0,
            "pool_used": self._next_pool_idx,
            "pool_remaining": len(PROMOTION_POOL) - self._next_pool_idx,
        }
