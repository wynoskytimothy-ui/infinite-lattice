"""
AETHOS origin tree: every collision origin spawns 3 new dimensions.

At each origin O:
  - Full 8 base vectors (v1-v8)
  - 4-way branching (VA1-VA4) on each vector -> 32 wings
  - Transgressor n : 0 -> infinity with recursive anchor chain A
  - When branches meet -> new origin O' with 3 child dimension-spaces

"Dimensionless" = no fixed global dimension; depth is unbounded (origins in origins).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Iterator, Sequence

from aethos_lattice import BranchKind, Coord, LatticeId, VECTORS
from aethos_sequences import IntersectionType, SequenceKind, canon_on_chain, make_chain

# Three orthogonal child dimensions opened at every new origin
CHILD_DIMENSIONS = ("D1", "D2", "D3")


class DimSlot(IntEnum):
    D1 = 0
    D2 = 1
    D3 = 2


@dataclass
class Origin:
    """
    A collision / anchor point that hosts one full AETHOS 3D room.
    From here, three new dimensions branch (each gets 8 vectors + 4 branches).
    """

    id: str
    coord: Coord
    anchor_chain: tuple[int, ...]
    parent: Origin | None = None
    dim_slot: DimSlot | None = None  # which of 3 parent dimensions we hang from
    children: dict[DimSlot, Origin] = field(default_factory=dict)
    depth: int = 0

    def spawn_children(
        self,
        meet_coord: Coord,
        extended_chain: Sequence[int],
        child_id_prefix: str = "",
    ) -> dict[DimSlot, Origin]:
        """Each origin branches into 3 dimensions; each child is a new full room."""
        chain = tuple(extended_chain)
        for slot in DimSlot:
            cid = f"{child_id_prefix}{self.id}.{CHILD_DIMENSIONS[slot]}"
            child = Origin(
                id=cid,
                coord=meet_coord,
                anchor_chain=chain,
                parent=self,
                dim_slot=slot,
                depth=self.depth + 1,
            )
            self.children[slot] = child
        return self.children

    def wings_at(self, n: int, branch: BranchKind = BranchKind.VA1) -> dict[str, Coord]:
        """32 addresses in this origin's room (4 branches x 8 vectors)."""
        from aethos_lattice import apply_vector, lattice_id_parts

        out: dict[str, Coord] = {}
        for lid in LatticeId:
            b, v = lattice_id_parts(lid)
            canon = canon_on_chain(b, self.anchor_chain, n)
            # Offset by origin (local + global meet point)
            ox, oy, oz = self.coord
            cx, cy, cz = apply_vector(canon, v)
            out[lid.name] = (ox + cx, oy + cy, oz + cz)
        return out

    def count_descendant_origins(self) -> int:
        return 1 + sum(c.count_descendant_origins() for c in self.children.values())


@dataclass
class OriginTree:
    """
    Recursive origin tree: dimensionless = unbounded depth x 3 branches per node.
    Each node still has 32 wings and 4-way branching inside its room.
    """

    root: Origin
    max_depth: int = 4

    @classmethod
    def bootstrap(
        cls,
        species: SequenceKind = SequenceKind.PRIMES,
        chain_len: int = 3,
        max_depth: int = 3,
    ) -> OriginTree:
        chain = make_chain(species, chain_len)
        root = Origin(id="O0", coord=(0, 0, 0), anchor_chain=chain, depth=0)
        tree = cls(root=root, max_depth=max_depth)
        tree._expand_from(root)
        return tree

    def _expand_from(self, node: Origin) -> None:
        if node.depth >= self.max_depth:
            return
        # Meet witness: extend chain by next prime in species (if available)
        chain = list(node.anchor_chain)
        next_primes = make_chain(SequenceKind.PRIMES, len(chain) + 2)
        if len(next_primes) > len(chain):
            extended = next_primes
            witness_n = extended[-1]
            meet = canon_on_chain(BranchKind.VA1, chain, witness_n)
            node.spawn_children(meet, extended)
            for child in node.children.values():
                self._expand_from(child)

    def lattice_count_estimate(self) -> int:
        """
        Rough count of 'rooms': at each origin, 32 wings;
        3 child dimensions per origin, recursively.
        """
        def nodes(o: Origin) -> int:
            return 1 + sum(nodes(c) for c in o.children.values())

        n_origins = nodes(self.root)
        return n_origins * 32  # wings per origin room

    def walk(self) -> Iterator[Origin]:
        stack = [self.root]
        while stack:
            o = stack.pop()
            yield o
            stack.extend(reversed(list(o.children.values())))


def demo() -> None:
    print("=== AETHOS: dimensionless origin tree ===\n")
    print("Axiom: every origin O branches into 3 new dimensions (D1, D2, D3).")
    print("Each dimension hosts: 8 vectors x 4 branches = 32 wings + recursive n.\n")

    tree = OriginTree.bootstrap(max_depth=3)
    print(f"  Origins in tree (depth<=3): {tree.root.count_descendant_origins()}")
    print(f"  Estimated wing-addresses:  {tree.lattice_count_estimate()}\n")

    print("  Origin tree (ids):")
    for o in tree.walk():
        indent = "    " * o.depth
        slot = f" [{CHILD_DIMENSIONS[o.dim_slot]}]" if o.dim_slot is not None else ""
        print(f"{indent}{o.id}{slot}  chain={o.anchor_chain}  @ {o.coord}")

    print("\n  Sample: O0 room at n=5 (VA1, 32 wings summed as count):")
    wings = tree.root.wings_at(5)
    print(f"    wings: {len(wings)}  e.g. L01={wings['L01']}")

    print("\n=== Growth law (why 'dimensionless') ===\n")
    print("  depth d  ->  up to 3^d origin rooms (each a full 3D+8+4 engine)")
    print("  each room -> 32 independent lattice wings")
    print("  each wing -> infinite n and extendable anchor chain A")
    print("  => no fixed dimension: always 3 more at every origin, forever.\n")

    for d in range(1, 6):
        rooms = 3**d
        wings = rooms * 32
        print(f"  depth {d}: ~{rooms} origins, ~{wings} wing-rooms (before n x |A|)")


if __name__ == "__main__":
    demo()
