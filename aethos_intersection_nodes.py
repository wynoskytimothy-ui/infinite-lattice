"""
Intersection nodes — meet-activated branching on the 3D complex plane.

Every meet (solo, pair, triple, composite, cross-species) can **activate a node**.
An activated node **follows** its rail (transgressor n), **branches** on the 4-way i-fan
and 8-wing corridors, and **discovers** natural intersections with other active nodes.

Meet taxonomy (all use the same canon_on_chain + wing machinery)
-----------------------------------------------------------------
  SOLO_SOLO       bank(p) @ n=q  =  bank(q) @ n=p
  PAIR_PAIR       bank(A) @ n=m  =  bank(B) @ n=w   (missing-variable witnesses)
  TRIPLE          three pair rails equalize -> chain (a,p,q)
  COMPOSITE       C = prod(factors) -> chain = sorted unique factors
  COMPOSITE_SOLO  composite chain @ n=p  vs  solo(p) @ n=?
  CROSS_SPECIES   same formula, different SequenceKind chains

Node activation does not invent new formulas — it records witness (Psi, path, chain)
and queues further meet probes against the live node registry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from functools import reduce
from operator import mul
from typing import Iterator, Sequence

from aethos_complex_plane import ComplexPlane3D, equalize_witness, wing_transform
from aethos_lattice import BranchKind, LatticeId, lattice_id_parts
from aethos_physics import SpacetimeCell
from aethos_recursive import LatticeBank32, LatticeBank32K, find_cross_meets, try_compose_triple
from aethos_sequences import canon_on_chain, normalize_chain


class MeetKind(Enum):
    SOLO_SWAP = "solo_swap"
    PAIR_CROSS = "pair_cross"
    TRIPLE = "triple"
    COMPOSITE_CHAIN = "composite_chain"
    COMPOSITE_SOLO = "composite_solo"
    SAME_N_COLLISION = "same_n_collision"
    CUSTOM = "custom"


def prime_factors(n: int) -> tuple[int, ...]:
    """Unique prime factors of n > 1 (sorted)."""
    if n < 2:
        return ()
    x = n
    factors: list[int] = []
    d = 2
    while d * d <= x:
        if x % d == 0:
            factors.append(d)
            while x % d == 0:
                x //= d
        d = 3 if d == 2 else d + 2
    if x > 1:
        factors.append(x)
    return tuple(factors)


def chain_from_composite(composite: int) -> tuple[int, ...]:
    """Composite integer -> sorted unique prime-factor chain."""
    return prime_factors(composite)


def bank_for_chain(chain: Sequence[int]) -> LatticeBank32 | LatticeBank32K:
    c = normalize_chain(tuple(int(x) for x in chain))
    if len(c) == 1:
        return LatticeBank32.single_prime(int(c[0]))
    if len(c) == 2:
        return LatticeBank32.prime_pair(int(c[0]), int(c[1]))
    return LatticeBank32K(c)


@dataclass(frozen=True)
class MeetWitness:
    """One equalization event in C x R."""

    kind: MeetKind
    coord: tuple[float, float, float]
    psi: ComplexPlane3D
    wing: int
    branch: BranchKind
    left: tuple[int, ...]  # chain or (p,)
    right: tuple[int, ...]
    n_left: int
    n_right: int
    label: str = ""

    @property
    def z(self) -> complex:
        return self.psi.z

    def spacetime_cell(self, *, chain: tuple[int, ...] | None = None) -> SpacetimeCell:
        """Witness as SpacetimeCell on the left rail."""
        c = chain if chain is not None else self.left
        return SpacetimeCell.from_psi(
            self.psi,
            self.n_left,
            chain=c,
            branch=self.branch,
            wing=self.wing,
        )


@dataclass(frozen=True)
class EntanglementMeetPair:
    """
    Two meet witnesses sharing spring z but differing path/chain — entanglement address.

    Same (z, zeta) with different left/right chains is the lattice nonlocality channel (Sec 6).
    """

    cell_a: SpacetimeCell
    cell_b: SpacetimeCell
    witness_a: MeetWitness
    witness_b: MeetWitness

    @property
    def spring_match(self) -> bool:
        return self.cell_a.z == self.cell_b.z

    @property
    def depth_match(self) -> bool:
        return self.cell_a.zeta == self.cell_b.zeta

    @property
    def path_distinct(self) -> bool:
        return (
            self.witness_a.left != self.witness_b.left
            or self.witness_a.right != self.witness_b.right
            or self.witness_a.n_left != self.witness_b.n_left
        )


def witness_to_spacetime_cell(
    witness: MeetWitness,
    *,
    chain: tuple[int, ...] | None = None,
) -> SpacetimeCell:
    return witness.spacetime_cell(chain=chain)


def find_entangled_meet_pairs(
    network: IntersectionNetwork,
    *,
    require_depth_match: bool = False,
) -> list[EntanglementMeetPair]:
    """
    Pair active nodes sharing spring z with distinct meet paths.

    Default matches on z only (same spring, different path/depth) — Sec 6 nonlocality.
    Set require_depth_match=True to require identical zeta as well.
    """
    nodes = list(network.walk_active())
    out: list[EntanglementMeetPair] = []
    for i, na in enumerate(nodes):
        wa = na.witness
        ca = wa.spacetime_cell(chain=na.chain)
        for nb in nodes[i + 1 :]:
            wb = nb.witness
            cb = wb.spacetime_cell(chain=nb.chain)
            if wa.psi.z != wb.psi.z:
                continue
            if require_depth_match and wa.psi.zeta != wb.psi.zeta:
                continue
            if wa.left == wb.left and wa.right == wb.right and wa.n_left == wb.n_left:
                continue
            out.append(
                EntanglementMeetPair(
                    cell_a=ca,
                    cell_b=cb,
                    witness_a=wa,
                    witness_b=wb,
                )
            )
    return out


@dataclass
class IntersectionNode:
    """
    Activated node: a meet point that hosts further branching.

    Carries enough label to distinguish same (X,Y,zeta) meets (path, chain, wing).
    """

    node_id: int
    witness: MeetWitness
    chain: tuple[int, ...]
    depth: int = 0
    children: list[int] = field(default_factory=list)
    active: bool = True

    @property
    def psi(self) -> ComplexPlane3D:
        return self.witness.psi


class IntersectionNetwork:
    """
    Registry of meet-activated nodes; probes natural intersections.
    """

    def __init__(self) -> None:
        self._nodes: dict[int, IntersectionNode] = {}
        self._next_id = 0
        self._seen: set[tuple] = set()

    def __len__(self) -> int:
        return len(self._nodes)

    def activate(self, witness: MeetWitness, chain: tuple[int, ...]) -> IntersectionNode | None:
        key = (
            witness.kind,
            witness.coord,
            witness.wing,
            witness.branch,
            chain,
            witness.n_left,
            witness.n_right,
            witness.left,
            witness.right,
        )
        if key in self._seen:
            return None
        self._seen.add(key)
        nid = self._next_id
        self._next_id += 1
        node = IntersectionNode(node_id=nid, witness=witness, chain=chain)
        self._nodes[nid] = node
        return node

    def node(self, node_id: int) -> IntersectionNode:
        return self._nodes[node_id]

    def probe_solo_swap(
        self,
        p: int,
        q: int,
        *,
        wing: int = 1,
        branch: BranchKind = BranchKind.VA1,
    ) -> MeetWitness | None:
        left = wing_transform(branch, (p,), q, wing)
        right = wing_transform(branch, (q,), p, wing)
        if left.coord != right.coord:
            return None
        return MeetWitness(
            kind=MeetKind.SOLO_SWAP,
            coord=left.coord,
            psi=left,
            wing=wing,
            branch=branch,
            left=(p,),
            right=(q,),
            n_left=q,
            n_right=p,
            label=f"solo {p}@n={q} = solo {q}@n={p}",
        )

    def probe_pair_cross(
        self,
        chain_a: Sequence[int],
        chain_b: Sequence[int],
        *,
        wing: int = 1,
        branch: BranchKind = BranchKind.VA1,
        n_max: int = 200,
    ) -> list[MeetWitness]:
        ca = normalize_chain(chain_a)
        cb = normalize_chain(chain_b)
        ba = bank_for_chain(ca)
        bb = bank_for_chain(cb)
        lid = LatticeId((int(branch) - 1) * 8 + wing)
        hits = find_cross_meets(ba, bb, lid, n_max)  # type: ignore[arg-type]
        out: list[MeetWitness] = []
        for coord, na, nb in hits:
            psi = ComplexPlane3D.from_coord(coord)
            out.append(
                MeetWitness(
                    kind=MeetKind.PAIR_CROSS,
                    coord=coord,
                    psi=psi,
                    wing=wing,
                    branch=branch,
                    left=ca,
                    right=cb,
                    n_left=na,
                    n_right=nb,
                    label=f"{ca}@n={na} x {cb}@n={nb}",
                )
            )
        return out

    def probe_triple(
        self,
        a: int,
        p: int,
        q: int,
        *,
        wing: int = 1,
        branch: BranchKind = BranchKind.VA1,
    ) -> MeetWitness | None:
        if not (a < p < q):
            return None
        full = (a, p, q)
        n_ap, psi_ap = equalize_witness(full, (a, p), branch, wing)
        n_aq, psi_aq = equalize_witness(full, (a, q), branch, wing)
        n_pq, psi_pq = equalize_witness(full, (p, q), branch, wing)
        if not (psi_ap.coord == psi_aq.coord == psi_pq.coord):
            return None
        return MeetWitness(
            kind=MeetKind.TRIPLE,
            coord=psi_ap.coord,
            psi=psi_ap,
            wing=wing,
            branch=branch,
            left=(a, p),
            right=(a, q),
            n_left=int(n_ap),
            n_right=int(n_aq),
            label=f"triple ({a},{p},{q}) witnesses n={int(n_ap)},{int(n_aq)},{int(n_pq)}",
        )

    def probe_composite(
        self,
        composite: int,
        *,
        wing: int = 1,
        branch: BranchKind = BranchKind.VA1,
    ) -> tuple[tuple[int, ...], MeetWitness] | None:
        chain = chain_from_composite(composite)
        if len(chain) < 2:
            return None
        if len(chain) == 2:
            w = self.probe_solo_swap(chain[0], chain[1], wing=wing, branch=branch)
            kind = MeetKind.COMPOSITE_CHAIN
        else:
            w = self.probe_triple(chain[0], chain[1], chain[2], wing=wing, branch=branch)
            kind = MeetKind.COMPOSITE_CHAIN
        if w is None:
            return None
        wc = MeetWitness(
            kind=kind,
            coord=w.coord,
            psi=w.psi,
            wing=wing,
            branch=branch,
            left=chain,
            right=(),
            n_left=w.n_left,
            n_right=w.n_right,
            label=f"composite {composite} -> chain {chain}",
        )
        return chain, wc

    def follow_and_branch(
        self,
        seeds: Sequence[tuple[int, ...] | int],
        *,
        wing: int = 1,
        branch: BranchKind = BranchKind.VA1,
        max_nodes: int = 64,
    ) -> list[IntersectionNode]:
        """
        Activate nodes from seeds; probe pairwise meets; grow network.
        """
        activated: list[IntersectionNode] = []
        chains: list[tuple[int, ...]] = []
        for s in seeds:
            if isinstance(s, int):
                if s > 1 and prime_factors(s) and reduce(mul, prime_factors(s), 1) == s:
                    chains.append(chain_from_composite(s))
                else:
                    chains.append((s,))
            else:
                chains.append(normalize_chain(tuple(int(x) for x in s)))

        # solo swaps and pair crosses
        for i, ca in enumerate(chains):
            if len(ca) == 1:
                for cb in chains[i + 1 :]:
                    if len(cb) == 1:
                        w = self.probe_solo_swap(ca[0], cb[0], wing=wing, branch=branch)
                        if w:
                            node = self.activate(w, ca + cb)
                            if node:
                                activated.append(node)
            for cb in chains[i + 1 :]:
                if len(ca) >= 1 and len(cb) >= 1 and ca != cb:
                    for w in self.probe_pair_cross(ca, cb, wing=wing, branch=branch)[:3]:
                        merged = tuple(sorted(set(ca) | set(cb)))
                        node = self.activate(w, merged)
                        if node and len(activated) < max_nodes:
                            activated.append(node)

        # triple if three solo primes present
        solo_primes = sorted({x for c in chains if len(c) == 1 for x in c})
        if len(solo_primes) >= 3:
            a, p, q = solo_primes[0], solo_primes[1], solo_primes[2]
            w = self.probe_triple(a, p, q, wing=wing, branch=branch)
            if w:
                node = self.activate(w, (a, p, q))
                if node:
                    activated.append(node)

        return activated[:max_nodes]

    def entangled_pairs(
        self,
        *,
        require_depth_match: bool = False,
    ) -> list[EntanglementMeetPair]:
        """Entanglement candidates among active meet nodes."""
        return find_entangled_meet_pairs(self, require_depth_match=require_depth_match)

    def walk_active(self) -> Iterator[IntersectionNode]:
        for node in self._nodes.values():
            if node.active:
                yield node


def demo() -> None:
    print("=" * 72)
    print("INTERSECTION NODES — meet activates, follow, branch")
    print("=" * 72)

    net = IntersectionNetwork()

    print("\n--- Solo swap 3 x 5 ---")
    w = net.probe_solo_swap(3, 5)
    assert w
    print(f"  {w.label}  z={w.z.real:.0f}{w.z.imag:+.0f}i")
    net.activate(w, (3, 5))

    print("\n--- Triple 3,5,7 ---")
    w3 = net.probe_triple(3, 5, 7)
    assert w3
    print(f"  {w3.label}  z={w3.z.real:.0f}{w3.z.imag:+.0f}i")
    net.activate(w3, (3, 5, 7))

    print("\n--- Pair cross (3,5) x (3,7) ---")
    for w in net.probe_pair_cross((3, 5), (3, 7))[:2]:
        print(f"  {w.label}  z={w.z.real:.0f}{w.z.imag:+.0f}i")
        net.activate(w, (3, 5, 7))

    print("\n--- Composite 3*5=15 -> chain (3,5) ---")
    comp = 3 * 5
    got = net.probe_composite(comp)
    assert got
    chain, wc = got
    print(f"  composite {comp} chain={chain}")
    net.activate(wc, chain)

    print("\n--- Composite 2*3*5=30 (3-way factor meet) ---")
    chain30 = chain_from_composite(30)
    print(f"  factors chain={chain30}")
    if len(chain30) == 3:
        a, b, c = chain30
        w30 = net.probe_triple(a, b, c)
        if w30:
            print(f"  {w30.label}  z={w30.z.real:.0f}{w30.z.imag:+.0f}i")
            net.activate(w30, chain30)

    print("\n--- follow_and_branch from seeds [3,5,7,15,30] ---")
    net2 = IntersectionNetwork()
    nodes = net2.follow_and_branch([3, 5, 7, 15, 30])
    print(f"  activated {len(nodes)} nodes")
    for node in nodes[:8]:
        w = node.witness
        print(
            f"    id={node.node_id} {w.kind.value:16} chain={node.chain} "
            f"z={w.z.real:.0f}{w.z.imag:+.0f}i"
        )

    print("\n--- try_compose_triple (reference) ---")
    r = try_compose_triple(3, 5, 7)
    print(f"  confirmations: {len(r['triple_confirmations'])}")


if __name__ == "__main__":
    demo()
