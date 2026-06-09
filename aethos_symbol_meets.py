"""
Symbol prime meets — 2-way solo swaps and 3-way triple equalization.

When L1 symbol primes intersect on the imaginary line, each pair can solo-swap
(bank(p)@n=q = bank(q)@n=p).  When three or more distinct primes appear in a
subword or text, every sorted triple also equalizes to one locked node Psi:

    (a,p)@n=q  =  (a,q)@n=p  =  (p,q)@n=a   on full chain (a,p,q)

This wires ``aethos_symbol_map`` ICN chains into ``aethos_intersection_nodes``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

from aethos_complex_plane import ComplexPlane3D, triple_equalization
from aethos_intersection_nodes import (
    IntersectionNetwork,
    IntersectionNode,
    MeetKind,
    MeetWitness,
)
from aethos_lattice import BranchKind
from aethos_symbol_map import (
    prime_to_symbol,
    text_icn,
    text_icn_chain,
    text_symbol_chain,
)


@dataclass(frozen=True)
class TwoWayMeet:
    """Solo swap between two symbol primes."""

    left_prime: int
    right_prime: int
    left_symbol: str
    right_symbol: str
    witness: MeetWitness
    psi: ComplexPlane3D

    @property
    def coord(self) -> tuple[float, float, float]:
        return self.witness.coord


@dataclass(frozen=True)
class ThreeWayMeet:
    """Triple equalization — all three pair rails lock to one node."""

    primes: tuple[int, int, int]  # sorted a < p < q
    symbols: tuple[str, str, str]
    witnesses: dict[str, tuple[float, ComplexPlane3D]]  # ap, aq, pq
    witness: MeetWitness
    psi: ComplexPlane3D

    @property
    def coord(self) -> tuple[float, float, float]:
        return self.witness.coord


@dataclass
class TextMeetDiscovery:
    """All 2-way and 3-way meets discovered from a text's symbol primes."""

    text: str
    chain: tuple[int, ...]  # sorted unique ICN factors
    icn: int
    order_chain: tuple[int, ...]  # left-to-right symbol primes
    two_way: list[TwoWayMeet] = field(default_factory=list)
    three_way: list[ThreeWayMeet] = field(default_factory=list)
    composite: MeetWitness | None = None
    nodes: list[IntersectionNode] = field(default_factory=list)

    @property
    def has_triple(self) -> bool:
        return bool(self.three_way)

    @property
    def locked_coord(self) -> tuple[float, float, float] | None:
        if self.three_way:
            return self.three_way[0].coord
        if self.two_way:
            return self.two_way[0].coord
        return None


def _symbol_for_prime(p: int) -> str:
    try:
        return prime_to_symbol(p)
    except KeyError:
        return f"p{p}"


def discover_pair_meet(
    left_prime: int,
    right_prime: int,
    *,
    wing: int = 1,
    branch: BranchKind = BranchKind.VA1,
) -> TwoWayMeet | None:
    """2-way solo swap between two L1 symbol primes."""
    net = IntersectionNetwork()
    p, q = min(left_prime, right_prime), max(left_prime, right_prime)
    w = net.probe_solo_swap(p, q, wing=wing, branch=branch)
    if w is None:
        return None
    return TwoWayMeet(
        left_prime=p,
        right_prime=q,
        left_symbol=_symbol_for_prime(p),
        right_symbol=_symbol_for_prime(q),
        witness=w,
        psi=w.psi,
    )


def discover_triple_meet(
    a: int,
    p: int,
    q: int,
    *,
    wing: int = 1,
    branch: BranchKind = BranchKind.VA1,
) -> ThreeWayMeet | None:
    """3-way equalization for sorted primes a < p < q."""
    primes = tuple(sorted((a, p, q)))
    if len(set(primes)) != 3:
        return None
    a, p, q = primes  # type: ignore[assignment]
    net = IntersectionNetwork()
    w = net.probe_triple(a, p, q, wing=wing, branch=branch)
    if w is None:
        return None
    eq = triple_equalization(a, p, q, branch, wing)
    coords = {psi.coord for _, psi in eq.values()}
    if len(coords) != 1:
        return None
    return ThreeWayMeet(
        primes=(a, p, q),
        symbols=tuple(_symbol_for_prime(x) for x in (a, p, q)),
        witnesses=eq,
        witness=w,
        psi=w.psi,
    )


def discover_text_meets(
    text: str,
    *,
    wing: int = 1,
    branch: BranchKind = BranchKind.VA1,
    grow_network: bool = True,
    max_nodes: int = 64,
) -> TextMeetDiscovery:
    """
    Discover all 2-way solo swaps and 3-way triple locks from text symbol primes.

    For k distinct primes in text:
      - C(k,2) pairwise solo swaps
      - C(k,3) triple equalizations (each sorted triple)
      - composite ICN witness when k >= 2
    """
    chain = text_icn_chain(text)
    icn = text_icn(text)
    order = text_symbol_chain(text)
    out = TextMeetDiscovery(text=text, chain=chain, icn=icn, order_chain=order)

    net = IntersectionNetwork()

    for p, q in combinations(chain, 2):
        meet = discover_pair_meet(p, q, wing=wing, branch=branch)
        if meet:
            out.two_way.append(meet)

    for triple in combinations(chain, 3):
        a, p, q = triple
        meet = discover_triple_meet(a, p, q, wing=wing, branch=branch)
        if meet:
            out.three_way.append(meet)

    if len(chain) >= 2 and icn > 1:
        comp = net.probe_composite(icn, wing=wing, branch=branch)
        if comp:
            _, wc = comp
            out.composite = wc

    if grow_network:
        seeds: list[int | tuple[int, ...]] = [icn] if icn > 1 else list(chain)
        if not seeds and chain:
            seeds = [chain]
        out.nodes = net.follow_and_branch(
            seeds,
            wing=wing,
            branch=branch,
            max_nodes=max_nodes,
        )

    return out


def discover_subword_meets(subword: str, **kwargs) -> TextMeetDiscovery:
    """Alias for subword ICN meet discovery."""
    return discover_text_meets(subword, **kwargs)


def demo() -> None:
    print("=" * 60)
    print("SYMBOL PRIME MEETS — 2-way swaps + 3-way triple locks")
    print("=" * 60)

    for text in ("th", "ing", "the", "cats"):
        d = discover_text_meets(text, grow_network=False)
        print(f"\n  text {text!r}  chain={d.chain}  ICN={d.icn}")
        print(f"    2-way meets: {len(d.two_way)}")
        for m in d.two_way[:3]:
            print(
                f"      {m.left_symbol}+{m.right_symbol}  "
                f"primes ({m.left_prime},{m.right_prime})  "
                f"z={m.psi.z}  coord={m.coord}"
            )
        print(f"    3-way meets: {len(d.three_way)}")
        for m in d.three_way[:2]:
            ap_n, _ = m.witnesses["ap"]
            aq_n, _ = m.witnesses["aq"]
            pq_n, _ = m.witnesses["pq"]
            print(
                f"      ({m.symbols[0]},{m.symbols[1]},{m.symbols[2]})  "
                f"primes {m.primes}  "
                f"witnesses n=({int(ap_n)},{int(aq_n)},{int(pq_n)})  "
                f"z={m.psi.z}  coord={m.coord}"
            )
        if d.composite:
            print(f"    composite witness: {d.composite.label}")


if __name__ == "__main__":
    demo()
