"""
Recursive k-prime formulas for AETHOS (4 canonical branches × 32 vector wings).

Only four core recurrences (VA1–VA4) are needed. Depth grows by sorted prime tuple
P = (p1 <= p2 <= ... <= pk) and transgressor segment s in {0,...,k} as n crosses each pi.

Each of the 32 lattices applies the same canonical (X,Y,Z) from its branch, then
vector sign / Y-over swap — depth does not multiply the formula count.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Sequence

from aethos_lattice import (
    BranchKind,
    Coord,
    Lattice,
    LatticeBank32,
    LatticeId,
    VECTORS,
    apply_vector,
    lattice_id_parts,
    prime_pair_canon,
    single_prime_canon,
)

# ---------------------------------------------------------------------------
# Segment: which velocity regime transgressor n is in
# ---------------------------------------------------------------------------

def normalize_primes(primes: Sequence[int]) -> tuple[int, ...]:
    ps = tuple(sorted(set(primes)))
    if len(ps) != len(primes):
        raise ValueError("primes must be distinct")
    return ps


def segment_index(primes: tuple[int, ...], n: int) -> int:
    """
    s = 0        when n < p1
    s = i        when pi <= n < p_{i+1}  for i = 1 .. k-1
    s = k        when n >= pk
    """
    k = len(primes)
    if k == 0:
        return 0
    s = 0
    for i, p in enumerate(primes):
        if n < p:
            return s
        s = i + 1
    return k


def sum_primes(primes: tuple[int, ...]) -> int:
    return sum(primes)


def z_depth(primes: tuple[int, ...], n: int, seg: int) -> int:
    """
    Z axis with composition lock: between interior primes (not at ends),
    Z = sum(P) so pairwise branches that meet at transgressor witnesses share
  the same depth address when promoted to |P|=k. End segments use S+n (spec k=2).
    """
    k = len(primes)
    if k <= 2:
        return sum_primes(primes) + n
    if 0 < seg < k:
        return sum_primes(primes)
    return sum_primes(primes) + n


# ---------------------------------------------------------------------------
# Four recursive canonical formulas (general k)
# Inferred from spec §4 (k=1) and §5.1 (k=2); middle segments extend by induction.
# ---------------------------------------------------------------------------

def canon_va1(primes: tuple[int, ...], n: int) -> Coord:
    """VA1 / VA1A family."""
    k = len(primes)
    if k == 0:
        return (n, n, n)
    if k == 1:
        return single_prime_canon(BranchKind.VA1, primes[0], n)
    p1, pk = primes[0], primes[-1]
    seg = segment_index(primes, n)
    z = z_depth(primes, n, seg)
    if seg == 0:
        return (p1 + pk, p1, z)
    if seg < k:
        return (pk + n, n, z)
    return (pk + n, pk, z)


def canon_va2(primes: tuple[int, ...], n: int) -> Coord:
    """VA2 / VA1B family."""
    k = len(primes)
    if k == 1:
        return single_prime_canon(BranchKind.VA2, primes[0], n)
    p1, pk = primes[0], primes[-1]
    seg = segment_index(primes, n)
    z = z_depth(primes, n, seg)
    if seg == 0:
        return (2 * n + p1 + pk, -p1, z)
    if seg < k:
        return (2 * p1 + n + pk, -n, z)
    return (2 * p1 + n + pk, -pk, z)


def canon_va3(primes: tuple[int, ...], n: int) -> Coord:
    """VA3 / VA1C family."""
    k = len(primes)
    if k == 1:
        return single_prime_canon(BranchKind.VA3, primes[0], n)
    p1, pk = primes[0], primes[-1]
    seg = segment_index(primes, n)
    z = z_depth(primes, n, seg)
    if seg == 0:
        return (n + pk, -n, z)
    if seg < k:
        return (p1 + pk, -p1, z)
    return (p1 + n, -p1, z)


def canon_va4(primes: tuple[int, ...], n: int) -> Coord:
    """VA4 / VA1D family."""
    k = len(primes)
    if k == 1:
        return single_prime_canon(BranchKind.VA4, primes[0], n)
    p1, pk = primes[0], primes[-1]
    seg = segment_index(primes, n)
    z = z_depth(primes, n, seg)
    if seg == 0:
        return (2 * p1 + pk + n, n, z)
    if seg < k:
        return (2 * n + p1 + pk, p1, z)
    return (2 * pk + p1 + n, p1, z)


CANON = {
    BranchKind.VA1: canon_va1,
    BranchKind.VA2: canon_va2,
    BranchKind.VA3: canon_va3,
    BranchKind.VA4: canon_va4,
}


def canon_recursive(branch: BranchKind, primes: Sequence[int], n: int) -> Coord:
    ps = normalize_primes(primes)
    return CANON[branch](ps, n)


def verify_matches_spec_k2() -> bool:
    """Recursive k=2 must equal PDF tables."""
    tests = [
        (BranchKind.VA1, "VA1A"),
        (BranchKind.VA2, "VA1B"),
        (BranchKind.VA3, "VA1C"),
        (BranchKind.VA4, "VA1D"),
    ]
    a, p = 3, 11
    for branch, sub in tests:
        for n in (1, 3, 5, 11, 12, 20):
            got = canon_recursive(branch, (a, p), n)
            exp = prime_pair_canon(sub, a, p, n)
            if got != exp:
                print(f"FAIL {sub} n={n}: {got} != {exp}")
                return False
    return True


# ---------------------------------------------------------------------------
# k-depth lattice bank (still 32 wings; same 4 formulas)
# ---------------------------------------------------------------------------

@dataclass
class LatticeK(Lattice):
  primes: tuple[int, ...] = ()

  @classmethod
  def from_id_k(cls, lid: LatticeId, primes: Sequence[int]) -> LatticeK:
    ps = normalize_primes(primes)
    branch, vector = lattice_id_parts(lid)
    base = cls(
      id=lid,
      branch=branch,
      vector=vector,
      mode="single",
      p=ps[0] if ps else 0,
      a=None,
      pair_p=None,
    )
    base.primes = ps
    return base

  def _canon(self, n: int) -> Coord:
    if not self.primes:
      return canon_recursive(self.branch, (self.p,), n)
    return canon_recursive(self.branch, self.primes, n)

  def velocity_boundaries(self) -> list[int]:
    return list(self.primes)


class LatticeBank32K:
    """32 independent lattices for arbitrary prime depth k."""

    def __init__(self, primes: Sequence[int]) -> None:
        self.primes = normalize_primes(primes)
        self.k = len(self.primes)
        self.lattices = [LatticeK.from_id_k(LatticeId(i), self.primes) for i in range(1, 33)]

    def __getitem__(self, lid: LatticeId | int) -> LatticeK:
        return self.lattices[int(lid) - 1]

    def at_all(self, n: int) -> dict[LatticeId, Coord]:
        return {lat.id: lat.at(n) for lat in self.lattices}


# ---------------------------------------------------------------------------
# Natural composition: pairwise banks meet -> triple/quadruple address
# ---------------------------------------------------------------------------

def cross_bank_meet(
    primes_a: tuple[int, ...],
    primes_b: tuple[int, ...],
    lid: LatticeId = LatticeId.L01,
    n_max: int = 800,
) -> list[tuple[Coord, int, int]]:
    """Same wing, two prime-sets, find (n_a, n_b) with same coordinate."""
    ba = LatticeBank32K(primes_a)
    bb = LatticeBank32K(primes_b)
    la, lb = ba[lid], bb[lid]
    hits: list[tuple[Coord, int, int]] = []
    for na in range(1, n_max + 1):
        ca = la.at(na)
        for nb in range(1, n_max + 1):
            if la.at(nb) == ca and nb != na:
                continue
            if lb.at(nb) == ca:
                hits.append((ca, na, nb))
    # cleaner: standard swap meet for single primes
    return hits


def find_cross_meets(
    bank_left: LatticeBank32K | LatticeBank32,
    bank_right: LatticeBank32K | LatticeBank32,
    lid: LatticeId,
    n_max: int = 400,
) -> list[tuple[Coord, int, int]]:
    la = bank_left[lid]
    rb = bank_right[lid]
    hits: list[tuple[Coord, int, int]] = []
    for na in range(1, n_max):
        ca = la.at(na)
        for nb in range(1, n_max):
            if rb.at(nb) == ca:
                hits.append((ca, na, nb))
    return hits


def extension_witness(
    primes: Sequence[int],
    lid: LatticeId = LatticeId.L01,
    n_max: int = 2000,
) -> list[dict]:
    """
    Find how a shallower bank meets a deeper bank on one wing.
    Rule observed for |P|=2: solo p at n=q meets (p,q) at n=p (swap).
    For |P|=k: search (p1,...,pk-1) vs (p1,...,pk).
    """
    ps = normalize_primes(primes)
    if len(ps) < 2:
        return []
    shallow = LatticeBank32K(ps[:-1])
    deep = LatticeBank32K(ps)
    lat_s, lat_d = shallow[lid], deep[lid]
    p1, pk = ps[0], ps[-1]
    witnesses = []
    for na in range(1, n_max):
        cs = lat_s.at(na)
        for nd in range(1, n_max):
            if lat_d.at(nd) == cs:
                witnesses.append(
                    {
                        "coord": cs,
                        "n_shallow": na,
                        "n_deep": nd,
                        "swap_like": na == pk and nd == p1,
                    }
                )
    return witnesses[:20]


def try_compose_triple(a: int, b: int, c: int, lid: LatticeId = LatticeId.L01) -> dict:
    """
    Hypothesis: meetings of (a,b) and (a,c) banks force the (a,b,c) recursive
    formula at a witness n (often n=c or n=b when extending by transgression).
    """
    bank_ab = LatticeBank32K((a, b))
    bank_ac = LatticeBank32K((a, c))
    bank_abc = LatticeBank32K((a, b, c))
    lat_ab, lat_ac, lat_3 = bank_ab[lid], bank_ac[lid], bank_abc[lid]

  # Pairwise cross meets (swap endpoints)
    meet_ab_ac: list[tuple[int, int, Coord]] = []
    for n_ab in range(1, 600):
        cab = lat_ab.at(n_ab)
        for n_ac in range(1, 600):
            if lat_ac.at(n_ac) == cab:
                meet_ab_ac.append((n_ab, n_ac, cab))

    # Find n where triple formula matches a pairwise meeting point
    confirmations = []
    for n_ab, n_ac, coord in meet_ab_ac[:50]:
        for n3 in (b, c, n_ab, n_ac, max(b, c)):
            if lat_3.at(n3) == coord:
                confirmations.append(
                    {"coord": coord, "n_ab": n_ab, "n_ac": n_ac, "n_triple": n3}
                )

    return {
        "primes": (a, b, c),
        "pair_meetings_sample": meet_ab_ac[:5],
        "triple_confirmations": confirmations[:10],
    }


def recursive_formula_doc() -> str:
    return """
RECURSIVE AETHOS (forever) — four cores only
============================================
Let P = (p1,...,pk) sorted, S = sum(P), p1 = min, pk = max, segment s = segment_index(P,n).

VA1 (wing family 1):  Z = z_depth(P,n,s)  [S+n at ends; S locked between interior primes]
  s=0:     (p1+pk,  p1,  Z)
  0<s<k:  (pk+n,   n,   Z)
  s=k:    (pk+n,   pk,  Z)

VA2/VA3/VA4: same segment rules as VA1; Z = z_depth(...)

32 lattices:  canon_va*(P,n) -> apply_vector(., v_i)  for i=1..8 per branch.

Depth extension: P -> P union {p} when pairwise banks naturally meet at transgressor
witnesses; triple/k-tuple uses same four formulas with longer segment chain (k+1
velocity changes). No fifth formula — only more segments.
"""


def demo() -> None:
    print(recursive_formula_doc())

    print("=== Verify k=2 recursive == PDF tables ===")
    print(f"  Match: {verify_matches_spec_k2()}\n")

    print("=== k=3 primes (3,5,7) — L01 transgression & regimes ===\n")
    bank = LatticeBank32K((3, 5, 7))
    lat = bank[LatticeId.L01]
    for n in (2, 3, 4, 5, 6, 7, 8):
        print(f"  n={n}  seg={segment_index((3,5,7), n)}  {lat.at(n)}")

    print("\n=== k=4 primes (3,5,7,11) — velocity boundaries on all 32 wings ===\n")
    bank4 = LatticeBank32K((3, 5, 7, 11))
    assert all(
        lat.velocity_boundaries() == [3, 5, 7, 11] for lat in bank4.lattices
    )
    print("  All 32 lattices share boundaries [3,5,7,11]")

    print("\n=== Compose (3,5) + (3,7) -> (3,5,7) on L01 ===\n")
    r = try_compose_triple(3, 5, 7)
    print(f"  Pair meetings (sample): {r['pair_meetings_sample'][:3]}")
    print(f"  Triple hits: {r['triple_confirmations'][:5]}")

    print("\n=== Extension witness (3,5,7): shallow (3,5) vs deep (3,5,7) ===\n")
    w = extension_witness((3, 5, 7), LatticeId.L01, 400)
    for row in w[:6]:
        print(f"  {row}")

    print("\n=== Extension witness (3,5,7,11): (3,5,7) vs full set ===\n")
    w4 = extension_witness((3, 5, 7, 11), LatticeId.L01, 600)
    swap_hits = [r for r in w4 if r["swap_like"]]
    print(f"  swap-like (n_shallow=11, n_deep=3): {len(swap_hits)} hits")
    if swap_hits:
        print(f"  example: {swap_hits[0]}")

    print("\n=== k=1 vs k=2 cross: single 3 vs (3,11) all wings ===\n")
    b3 = LatticeBank32.single_prime(3)
    b11 = LatticeBank32.single_prime(11)
    b311 = LatticeBank32K((3, 11))
    m = sum(1 for lid in LatticeId if b3[lid].at(11) == b11[lid].at(3))
    print(f"  solo 3@11 = solo 11@3 (swap extension): {m}/32 wings")
    m2 = sum(1 for lid in LatticeId if b3[lid].at(11) == b311[lid].at(3))
    print(f"  solo 3@11 = tuple (3,11)@3: {m2}/32 (different object; pair is k=2 bank)")

    print("\n=== Pairwise meet promotes to triple (3,5,7) with z_depth lock ===\n")
    for branch in BranchKind:
        c_pair = canon_recursive(branch, (3, 5), 7)
        c_triple = canon_recursive(branch, (3, 5, 7), 5)
        print(f"  {branch.name}: (3,5)@7={c_pair}  (3,5,7)@5={c_triple}  match={c_pair == c_triple}")


if __name__ == "__main__":
    demo()
