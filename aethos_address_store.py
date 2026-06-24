"""
aethos_address_store.py -- a HIGH-DIMENSIONAL ADDRESS STORE on the AETHOS
prime-lattice ("billions of 90-degree-away axes").

Idea
----
The verified atom of the lattice is the 3-way meet: a triple {a, p, q} co-meets
on ONE node whose conserved depth is

    zeta = a + p + q        (the LOCK -- invariant of the transgressor n)

and whose 2-way meet readout (X, Y, Z) = (p+q, p, p+q) is INVERTIBLE
(small = Y, large = X - Y). Crucially, any 2 of the 3 anchors recover the 3rd
(missing = zeta - pair), and all three drop-one witnesses collide on the SAME
node. So a single triple is a *self-checking address*: a coordinate value plus
its own erasure code.

This module turns that atom into a key->vector store. Each coordinate axis gets
its own DISJOINT integer band, so a triple on axis i never shares an anchor with
a triple on axis j: the axes are mutually "90 degrees away" (orthogonal --
perturbing one axis moves exactly one meet-node). A key maps to a dict of
{axis_id: value}; each (axis, value) is encoded as one triple on that axis's
band, and the store keeps the meet-node per triple.

What is real vs. what is bookkeeping (two-sided)
------------------------------------------------
REAL (carried by the lattice node):
  - zeta = sum is read off the lattice meet (wing_transform .zeta), and IS the
    coordinate carrier; recovery is sum-based, exactly the verified rule.
  - the 3-of-3 drop-one collision is a genuine erasure self-check (verify()).
  - orthogonality is structural: disjoint bands => disjoint anchors => one node
    per axis moves under a one-axis perturbation. Measured, not asserted.
HONEST LIMITS (see __main__ "HONEST LIMITS" section, all measured):
  - "billions of axes" is an ADDRESSING claim: bands are disjoint by
    construction so the axis count is unbounded *in principle*, but each stored
    value costs O(1) python; this is a structured store, not a compressor.
  - zeta alone is partition-invariant: it locks ONE triple's interior, it does
    not by itself disambiguate which triple produced a bare sum. The store keys
    every coordinate by its (axis, band) so this is never relied upon; a
    shared-pool regime (no bands) DOES collide, and we measure exactly when.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

from aethos_complex_plane import (
    ComplexPlane3D,
    equalize_witness,
    missing_member,
    wing_transform,
)
from aethos_lattice import BranchKind


# ---------------------------------------------------------------------------
# The triple atom: one axis-coordinate, encoded + self-checking.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TripleNode:
    """One axis-coordinate as a verified 3-way meet.

    The triple is {a, p, q} (sorted). Its meet-node lives at transgressor
    n = a (the smallest, dropped) with conserved depth zeta = a + p + q.
    """

    axis_id: int
    band: int              # private integer-band base for this axis
    a: int                 # = band
    p: int                 # = band + 1
    q: int                 # = band + 2 + value   (so zeta encodes value)
    z: complex             # meet readout X + iY (lattice node)
    zeta: float            # conserved depth = a + p + q  (the lock)

    @property
    def triple(self) -> Tuple[int, int, int]:
        return (self.a, self.p, self.q)

    @property
    def value(self) -> int:
        """Recover the coordinate value from the conserved depth (sum-based)."""
        # zeta = a + p + q = band + (band+1) + (band+2+v) = 3*band + 3 + v
        return int(round(self.zeta)) - (3 * self.band + 3)


def _meet(a: int, p: int, q: int,
          branch: BranchKind = BranchKind.VA1, wing: int = 1) -> ComplexPlane3D:
    """The lattice meet-node of triple {a,p,q}, read off the lattice itself.

    Uses the missing-variable rule: transgress the (p,q) sub-chain until n hits
    the missing anchor a; the resulting Psi is the co-meet node. zeta == a+p+q.
    """
    chain = sorted((a, p, q))
    _, psi = equalize_witness(chain, [chain[1], chain[2]], branch=branch, wing=wing)
    return psi


def make_triple(axis_id: int, band: int, value: int,
                branch: BranchKind = BranchKind.VA1, wing: int = 1) -> TripleNode:
    """Encode (axis, value) as a triple on the axis's private band."""
    if value < 0:
        raise ValueError("coordinate value must be >= 0")
    a, p, q = band, band + 1, band + 2 + value
    psi = _meet(a, p, q, branch=branch, wing=wing)
    return TripleNode(axis_id=axis_id, band=band, a=a, p=p, q=q,
                      z=psi.z, zeta=psi.zeta)


# ---------------------------------------------------------------------------
# The store.
# ---------------------------------------------------------------------------

@dataclass
class AddressStore:
    """High-dimensional address store: key -> {axis_id: value}.

    Each (axis, value) is one self-checking triple on a disjoint band, so axes
    are mutually orthogonal. Coordinates recover independently from their own
    triple's conserved meet-depth.

    Parameters
    ----------
    band_size : int
        Width reserved per axis band. Axis i owns integers
        [base + i*band_size, base + (i+1)*band_size). Values must satisfy
        value + 3 < band_size so a triple never spills into the next band.
    base : int
        First band start (keep > 0; anchors are positive countable).
    branch, wing : the chamber the meet is read in (any of the 32; the
        coordinate lock zeta is conserved across all 32).
    """

    band_size: int = 1 << 20
    base: int = 1 << 20
    branch: BranchKind = BranchKind.VA1
    wing: int = 1
    _store: Dict[object, Dict[int, TripleNode]] = field(default_factory=dict)

    # -- band geometry ------------------------------------------------------
    def _band(self, axis_id: int) -> int:
        if axis_id < 0:
            raise ValueError("axis_id must be >= 0")
        return self.base + axis_id * self.band_size

    def _check_fits(self, value: int) -> None:
        if value < 0 or value + 3 >= self.band_size:
            raise ValueError(
                f"value {value} does not fit band_size {self.band_size} "
                f"(need 0 <= value < {self.band_size - 3})"
            )

    # -- public API ---------------------------------------------------------
    def put(self, key: object, vec: Dict[int, int]) -> None:
        """Store key -> {axis_id: value} as one triple per axis."""
        nodes: Dict[int, TripleNode] = {}
        for axis_id, value in vec.items():
            self._check_fits(value)
            nodes[axis_id] = make_triple(
                axis_id, self._band(axis_id), value,
                branch=self.branch, wing=self.wing,
            )
        self._store[key] = nodes

    def get(self, key: object) -> Dict[int, int]:
        """Recover {axis_id: value}, each coordinate independently from its meet."""
        nodes = self._store[key]
        return {axis_id: node.value for axis_id, node in nodes.items()}

    def node(self, key: object, axis_id: int) -> TripleNode:
        return self._store[key][axis_id]

    def nodes(self, key: object) -> Dict[int, TripleNode]:
        return self._store[key]

    def keys(self) -> Iterable[object]:
        return self._store.keys()

    def __len__(self) -> int:
        return len(self._store)

    # -- erasure self-check -------------------------------------------------
    def verify_node(self, node: TripleNode) -> bool:
        """Erasure self-check on ONE axis: any 2 of 3 recover the 3rd, and all
        three drop-one witnesses must collide on the same meet-node.

        - recover-the-missing: for each dropped anchor, missing == zeta - pair.
        - collision: the three drop-one witnesses (each a 2-way meet that
          transgresses to the missing anchor) land on the SAME (z, zeta).
        """
        chain = sorted(node.triple)
        s = node.zeta
        # 1. sum-based erasure: any pair recovers the dropped third exactly.
        for drop in range(3):
            pair = [chain[i] for i in range(3) if i != drop]
            recovered = s - sum(pair)            # missing = zeta - pair
            if int(round(recovered)) != chain[drop]:
                return False
            # also exercise the lattice's own missing-member rule
            if int(round(missing_member(chain, pair))) != chain[drop]:
                return False
        # 2. all three drop-one witnesses collide on one node.
        coords = set()
        zetas = set()
        for drop in range(3):
            pair = [chain[i] for i in range(3) if i != drop]
            psi = _meet(pair[0], pair[1], chain[drop],
                        branch=self.branch, wing=self.wing)
            coords.add(psi.coord)
            zetas.add(round(psi.zeta, 9))
        if len(coords) != 1 or len(zetas) != 1:
            return False
        # 3. the stored node must agree with the recomputed meet (tamper check).
        if round(next(iter(zetas)), 9) != round(s, 9):
            return False
        return True

    def verify(self, key: object) -> bool:
        """Erasure self-check across every axis of a stored key."""
        return all(self.verify_node(n) for n in self._store[key].values())

    def corrupt(self, key: object, axis_id: int, *, flip: int = 1) -> None:
        """DEMO ONLY: tamper with one stored coordinate's depth so its triple
        no longer closes (zeta inconsistent with its anchors)."""
        old = self._store[key][axis_id]
        bad = TripleNode(
            axis_id=old.axis_id, band=old.band,
            a=old.a, p=old.p, q=old.q,
            z=old.z, zeta=old.zeta + flip,        # break the lock
        )
        self._store[key][axis_id] = bad


# ---------------------------------------------------------------------------
# Demo + MEASURE  (run: python aethos_address_store.py)
# ---------------------------------------------------------------------------

def _hr(title: str) -> None:
    print("=" * 72)
    print(title)
    print("=" * 72)


def demo() -> dict:
    import random
    random.seed(7)
    results: dict = {}

    _hr("(1) ENCODE a D=12 vector, RECOVER all 12 coords independently")
    D = 12
    store = AddressStore()
    true_vec = {i: random.randint(0, 5000) for i in range(D)}
    store.put("item", true_vec)
    rec = store.get("item")
    exact = (rec == true_vec)
    print("  true :", [true_vec[i] for i in range(D)])
    print("  recov:", [rec[i] for i in range(D)])
    print(f"  ALL {D} coords recovered exactly: {exact}")
    print("  (each axis decoded ONLY from its own triple's conserved meet-depth)")
    results["recover_all_12_exact"] = exact

    _hr("(2) ORTHOGONALITY: perturb ONE axis -> exactly ONE meet-node moves")
    base_nodes = {i: (store.node("item", i).z, store.node("item", i).zeta)
                  for i in range(D)}
    bumped = dict(true_vec)
    bumped[5] += 137
    store.put("item2", bumped)
    moved = []
    for i in range(D):
        z0, zeta0 = base_nodes[i]
        z1, zeta1 = store.node("item2", i).z, store.node("item2", i).zeta
        if (z0, zeta0) != (z1, zeta1):
            moved.append(i)
    print("  perturbed axis 5 by +137")
    print("  axes whose meet-node MOVED:", moved)
    one_moved = (moved == [5])
    print(f"  orthogonal (exactly axis 5 moved): {one_moved}")
    print(f"  axis-5 coord still correct: {store.get('item2')[5] == bumped[5]}")
    results["orthogonal_one_axis_moves"] = one_moved

    _hr("(3) INTEGRITY: corrupt one coordinate -> verify() catches it")
    clean = store.verify("item")
    print(f"  verify(clean 'item'): {clean}")
    store.put("victim", {i: random.randint(0, 5000) for i in range(D)})
    before = store.verify("victim")
    store.corrupt("victim", 4, flip=1)        # flip one coordinate's lock
    after = store.verify("victim")
    print(f"  verify(before corruption): {before}")
    print(f"  flipped axis-4 depth by +1")
    print(f"  verify(after corruption):  {after}   (corruption detected: {not after})")
    detected = clean and before and (not after)
    results["corruption_detected"] = detected

    _hr("(4) ERASURE: any 2 of 3 recover the 3rd, on a real stored axis")
    node = store.node("item", 0)
    chain = sorted(node.triple)
    print(f"  axis-0 triple {tuple(chain)}  zeta(lock)={node.zeta:.0f}")
    all_ok = True
    for drop in range(3):
        pair = [chain[i] for i in range(3) if i != drop]
        recovered = node.zeta - sum(pair)
        ok = int(round(recovered)) == chain[drop]
        all_ok = all_ok and ok
        print(f"    drop {chain[drop]:>8} | keep {tuple(pair)} -> recover "
              f"{int(round(recovered)):>8}  {'OK' if ok else 'FAIL'}")
    results["erasure_any2_recover3"] = all_ok

    _hr("(5) CAPACITY -- collision-free node count (two regimes, MEASURED)")
    # REGIME A: disjoint bands (how the store actually works). One node per
    # (axis, value); distinct by construction -> 0 collisions, always.
    nA = 200
    bands_store = AddressStore()
    bands_store.put("cap", {i: (i * 7) % 4000 for i in range(nA)})
    coords = set()
    for i in range(nA):
        nd = bands_store.node("cap", i)
        coords.add((nd.z, nd.zeta))
    print(f"  REGIME A (disjoint bands, the store): {nA} axes -> "
          f"{len(coords)} DISTINCT meet-nodes, "
          f"{nA - len(coords)} collisions "
          f"({100.0*len(coords)/nA:.1f}% unique)")
    results["bands_axes"] = nA
    results["bands_distinct_nodes"] = len(coords)
    results["bands_collisions"] = nA - len(coords)

    # REGIME B: shared small pool (NO bands). Draw every triple from 1..M and
    # ask which READOUT disambiguates. Two readouts, opposite verdicts:
    #   full meet-node (z, zeta) = (p+q, p, a+p+q)  -> X,Y disambiguate the
    #       triple, so it stays collision-free even on a tiny shared pool.
    #   bare zeta (the SUM) alone -> partition-invariant, collapses hard.
    print("  REGIME B (shared pool, no bands) -- which readout disambiguates:")
    regime_b = {}
    for M in (40, 80, 160):
        pool = range(1, M + 1)
        seen_full: Dict[Tuple[complex, float], int] = {}
        seen_sum: Dict[float, int] = {}
        ntr = 0
        for a, p, q in itertools.combinations(pool, 3):
            psi = _meet(a, p, q)
            ntr += 1
            seen_full[(psi.z, psi.zeta)] = 1
            seen_sum[round(psi.zeta, 9)] = 1
        df, ds = len(seen_full), len(seen_sum)
        print(f"    pool 1..{M:<4}: {ntr:7d} triples | "
              f"full node (z,zeta): {df:7d} distinct ({100.0*df/ntr:5.1f}% uniq) | "
              f"bare sum zeta: {ds:5d} distinct ({100.0*ds/ntr:4.1f}% uniq)")
        regime_b[M] = {"triples": ntr, "full_distinct": df, "sum_distinct": ds}
    results["shared_pool"] = regime_b

    _hr("(6) 32-CHAMBER FAN-OUT -- each node x (4 branch x 8 wing), MEASURED")
    # Same triple through all 32 chambers: distinct z (sub-addresses), zeta
    # conserved (one lock). Confirms each meet-node carries a 32-fold address.
    chain = (7, 11, 19)
    zs, zts = set(), set()
    for br in BranchKind:
        for wing in range(1, 9):
            psi = _meet(chain[0], chain[1], chain[2], branch=br, wing=wing)
            zs.add(psi.z)
            zts.add(round(psi.zeta, 9))
    print(f"  triple {chain}: distinct z across 32 chambers = {len(zs)}, "
          f"distinct |zeta| = {len(set(abs(z) for z in zts))} (conserved)")
    results["chamber_distinct_z"] = len(zs)

    _hr("HONEST LIMITS (two-sided, all measured above)")
    print("""
  WORKS:
    - all 12 coords recover exactly from independent triple meets (test 1)
    - orthogonality is real & structural: 1 axis perturbed -> exactly 1 node
      moves, because disjoint bands share no anchors (test 2)
    - corruption of a single coordinate's lock is caught by the erasure
      self-check (test 3); any-2-of-3 recovery is exact (test 4)
    - in the band regime the store is collision-free by construction (test 5A)
    - each node legitimately fans to a 32-chamber sub-address, zeta conserved
      across all 32 (test 6)
  LIMITS:
    - the full meet-node (z,zeta)=(p+q, p, sum) stays collision-free even on a
      shared pool, BUT the bare sum zeta alone collapses to ~0.1-1%% unique
      (test 5B): zeta is partition-invariant -- it locks ONE triple's interior,
      a bare sum does not say which triple made it. Recovery therefore needs the
      (axis,band) key, which the store always carries; never decode bare zeta.
    - bands are load-bearing for the disjoint-anchor / orthogonality property;
      "billions of 90-deg axes" is an addressing/construction claim, not magic.
    - this is a STRUCTURED self-checking store (erasure code + orthogonal axes),
      O(1) python per stored value -- not a compressor and not "free" capacity.
""")
    return results


if __name__ == "__main__":
    demo()
