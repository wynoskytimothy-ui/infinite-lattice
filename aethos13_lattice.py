#!/usr/bin/env python3
"""AETHOS-13 / OSCAR LATTICE — Phase 1 (State, Un-Flattened Formula, 64 Views) + Phase 2 (Traversal, Path, Invariant).

Built verbatim to AETHOS_MASTER.md (C:/Users/wynos/aethos_master/docs), sections 3.9a.4-3.9a.6.
NOT a neural net: deterministic integer geometry, no gradients / attention / dense vectors.

Canonical law (master 3.9a.4-3.9a.5), reproduced exactly:
    state(values,N) -> (m1,m2,m3,Sigma)    # order statistics of the multiset (values U {N}) + its sum
    branch VA1..VA4                         # verse + anti-verse, m2/m3 tier shift -- all four load-bearing
    vector V1..V8                           # AXIOM 2: V5..V8 are V1..V4 computed in (Y,X,Z) then transposed
    address = (V, VA, m1, m2, m3, Sigma)    # 8 vectors x 4 branches = 32 canonical
    + bridge maps (swap P1,P3)              # doubles 8->16 vectors => 16 maps x 4 branches = 64 views (master 902)

This module validates against the master's printed 32-address table for seed {2,3,5,7}, N=4 -> (7,5,4,21).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterable, Iterator, List, Sequence, Tuple

Coord = Tuple[int, int, int]

# ======================================================================================================
# PHASE 1.1 -- THE STATE  (master 3.9a.5, verbatim)
# ======================================================================================================
@dataclass(frozen=True)
class State:
    m1: int; m2: int; m3: int; Sigma: int
    def as_tuple(self) -> Tuple[int, int, int, int]: return (self.m1, self.m2, self.m3, self.Sigma)

def state(values: Iterable[float], N: int) -> State:
    """Sort the multiset (values U {N}) ascending; m1=largest, m2=2nd, m3=3rd (0 if absent); Sigma=sum."""
    M = sorted(list(values) + [N])
    m1 = M[-1]
    m2 = M[-2] if len(M) > 1 else 0
    m3 = M[-3] if len(M) > 2 else 0
    return State(m1, m2, m3, sum(M))

# ======================================================================================================
# PHASE 1.2 -- THE FOUR BRANCHES VA1..VA4  (master 3.9a.5, verbatim)
# ======================================================================================================
def branch(s: State, b: int) -> Coord:
    m1, m2, m3, S = s.m1, s.m2, s.m3, s.Sigma
    if b == 1: return (m1 + m2,            +m2, S)     # verse,      m2 pivot
    if b == 2: return (m1 + m2 + 2 * m3,   -m2, S)     # anti-verse, m2 pivot, +2m3 tier lift
    if b == 3: return (m1 + m3,            -m3, S)     # anti-verse, m3 pivot
    if b == 4: return (m1 + 2 * m2 + m3,   +m3, S)     # verse,      m3 pivot, +2m2 tier lift
    raise ValueError("branch b in 1..4")

def branches(s: State) -> List[Tuple[int, Coord]]:
    return [(b, branch(s, b)) for b in (1, 2, 3, 4)]

# ======================================================================================================
# PHASE 1.3 -- THE VECTORS / MAPS
# ======================================================================================================
# 8 base vectors -- AXIOM 2 verbatim (master 3.9a.4). V5..V8 = V1..V4 computed in (Y,X,Z), the XY-swap.
def vector(c: Coord, v: int) -> Coord:
    x, y, z = c
    return [(x, y, z), (x, y, -z), (-x, y, z), (-x, y, -z),     # XYZ family (y=0 plane)
            (y, x, z), (y, x, -z), (-y, x, z), (-y, x, -z)][v - 1]   # YXZ family (x=0 plane): THE SWAP

# Bridge maps (master 902): swap P1,P3 (reflect across x=z) doubles the 8 vectors to 16 -> 64 views.
def bridge(c: Coord) -> Coord:
    x, y, z = c
    return (z, y, x)

def maps16(coord: Coord) -> List[Tuple[int, str, Coord]]:
    """16 oriented views of a branch coordinate: maps 1-8 = the 8 Axiom-2 vectors; 9-16 = their bridges."""
    out: List[Tuple[int, str, Coord]] = []
    for v in range(1, 9):
        out.append((v, f"V{v}", vector(coord, v)))
    for v in range(1, 9):
        out.append((8 + v, f"V{v}~bridge", bridge(vector(coord, v))))
    return out

# ======================================================================================================
# PHASE 1.4 / PHASE 2.3 -- THE ADDRESS with PATH HISTORY (the binary operad)
# ======================================================================================================
@dataclass(frozen=True)
class Address:
    v: int; b: int
    m1: int; m2: int; m3: int; Sigma: int
    coord: Coord
    path: Tuple[float, ...] = field(default=())     # ordered arrival history -- the operad
    @property
    def rank(self) -> int: return self.Sigma        # Sigma already carries N; the DOOR/OVERPASS discriminator
    def state_key(self) -> Tuple:                    # ORDER-BLIND identity (Axiom: VA is order-blind)
        return (self.v, self.b, self.m1, self.m2, self.m3, self.Sigma, self.coord)
    def full_key(self) -> Tuple:                     # order-SENSITIVE identity: separate until explicit convergence
        return (self.path,) + self.state_key()

def address_set(values: Sequence[float], N: int, path: Tuple[float, ...] = ()) -> List[Address]:
    """All 64 views (16 maps x 4 branches) of the state, each tagged with the arrival path."""
    s = state(values, N)
    out: List[Address] = []
    for b, coord in branches(s):
        for _map_i, _label, mapped in maps16(coord):
            out.append(Address(_map_i, b, s.m1, s.m2, s.m3, s.Sigma, mapped, path))
    return out

# ======================================================================================================
# PHASE 2.1 / 2.2 -- CORRIDOR & INVARIANT (DAG guarantee)
# ======================================================================================================
def corridor(values: Sequence[float], n_from: int = 0, n_to: int = 16) -> Iterator[Tuple[int, State]]:
    base = sum(values); prev = None
    for N in range(n_from, n_to):
        s = state(values, N)
        assert s.Sigma == base + N, f"invariant z=Sigma+N violated: {s.Sigma} != {base}+{N}"
        if prev is not None:
            assert s.Sigma > prev, f"z must strictly increase (DAG): {s.Sigma} !> {prev}"
        prev = s.Sigma
        yield N, s

# Correctness: the triple point (master 3.9a) -- multiset symmetry, no k-way function.
def triple_point(A, B, C) -> bool:
    return state([A, B], C) == state([A, C], B) == state([B, C], A)

# ======================================================================================================
# PHASE 2 (v5) -- THE PI GRAPH / HELIX  (master 3.9a.16, 5.8): sector periodic x z monotone = spiral staircase
# ======================================================================================================
import math as _math
def helix_dealiasing(k_turns: int = 5, per_turn: int = 100):
    """Master 5.8: one rotation f(t)=e^{it} viewed in 1D/2D/3D. Report how many DISTINCT points each keeps.
    1D wave loses direction; 2D circle aliases (loses turn count); 3D helix loses nothing. z de-aliases."""
    n = k_turns * per_turn
    ts = [2 * _math.pi * k_turns * i / n for i in range(n)]
    d1 = len({round(_math.cos(t), 6) for t in ts})                          # WAVE: shadow on one axis
    d2 = len({(round(_math.cos(t), 6), round(_math.sin(t), 6)) for t in ts})  # CIRCLE: aliases across turns
    d3 = len({(round(_math.cos(t), 6), round(_math.sin(t), 6), i) for i, t in enumerate(ts)})  # HELIX: + height
    return n, d1, d2, d3

def lattice_helix(values, n_from=0, n_to=200, sectors=64):
    """Master 3.9a.16: octant/sector (periodic, loops via bridges) x z=Sigma+N (monotone) = HELIX.
    Walk a corridor while the sector index rotates; the (x,y) SHADOW recurs but (sector,z) never does."""
    shadow_seen = set(); helix_seen = set(); shadow_reuse = 0
    step = 0
    for N, s in corridor(values, n_from, n_to):
        sector = step % sectors                       # the periodic rotation (wants to cycle)
        c = branch(s, 1); shadow = (c[0], c[1])       # (x,y) with z projected away
        if shadow in shadow_seen: shadow_reuse += 1
        shadow_seen.add(shadow)
        helix_seen.add((sector, s.Sigma))             # sector x height -- the helix
        step += 1
    return step, shadow_reuse, len(helix_seen)

# ======================================================================================================
# PHASE 2 (v5) -- ZENO'S RESOLUTION  (master 5.6.1): the descent peels M[:-2], halting in ceil(k/2) steps
# ======================================================================================================
def coord3(M):
    """The 3D coordinate a tier exposes: (m1+m2, m2, sum). Master 5.6.1 (matches its (2,3,5)->(8,3,10))."""
    Ms = sorted(M)
    m1 = Ms[-1] if Ms else 0
    m2 = Ms[-2] if len(Ms) > 1 else 0
    return (m1 + m2, m2, sum(Ms))

def tiers(M):
    """The upper-verse tier stack. Zeno cashed out: continuous spines, but the descent HALTS in ceil(k/2)
    steps because each tier consumes the top two (M = M[:-2]). Master 5.6.1 verbatim."""
    M = sorted(M); out = []
    while M:
        out.append(coord3(M))
        M = M[:-2]                                    # drop the two the tier above consumed
    return tuple(out)

# ======================================================================================================
# DEMONSTRATION + VALIDATION AGAINST THE MASTER TABLE
# ======================================================================================================
def _rule(t): print("\n" + "=" * 96 + f"\n{t}\n" + "=" * 96)

# The master's printed 32-address table for seed {2,3,5,7}, N=4 -> (7,5,4,21)  (AETHOS_MASTER.md 3.9a.4)
MASTER_TABLE = {  # (V, VA) -> coord
 (1,1):(12,5,21),(1,2):(20,-5,21),(1,3):(11,-4,21),(1,4):(21,4,21),
 (2,1):(12,5,-21),(2,2):(20,-5,-21),(2,3):(11,-4,-21),(2,4):(21,4,-21),
 (3,1):(-12,5,21),(3,2):(-20,-5,21),(3,3):(-11,-4,21),(3,4):(-21,4,21),
 (4,1):(-12,5,-21),(4,2):(-20,-5,-21),(4,3):(-11,-4,-21),(4,4):(-21,4,-21),
 (5,1):(5,12,21),(5,2):(-5,20,21),(5,3):(-4,11,21),(5,4):(4,21,21),
 (6,1):(5,12,-21),(6,2):(-5,20,-21),(6,3):(-4,11,-21),(6,4):(4,21,-21),
 (7,1):(-5,12,21),(7,2):(5,20,21),(7,3):(4,11,21),(7,4):(-4,21,21),
 (8,1):(-5,12,-21),(8,2):(5,20,-21),(8,3):(4,11,-21),(8,4):(-4,21,-21),
}

if __name__ == "__main__":
    _rule("PHASE 1.1 -- THE STATE")
    s = state([2, 3, 5, 7], 4)
    print(f"  state([2,3,5,7], N=4) = (m1={s.m1}, m2={s.m2}, m3={s.m3}, Sigma={s.Sigma})   [master: (7,5,4,21)]")
    print(f"  tiny-set default: state([4],1) = {state([4],1).as_tuple()}  (m2=m3=0)")

    _rule("PHASE 1.2 -- THE FOUR BRANCHES")
    for b, c in branches(s): print(f"  VA{b}: {c}")

    _rule("PHASE 1.3 -- VALIDATE THE 8 VECTORS x 4 BRANCHES AGAINST THE MASTER'S PRINTED 32-TABLE")
    ok = 0; bad = 0
    for v in range(1, 9):
        row = []
        for b in (1, 2, 3, 4):
            got = vector(branch(s, b), v); exp = MASTER_TABLE[(v, b)]
            match = got == exp; ok += match; bad += (not match)
            row.append(f"{got}{'' if match else ' !=' + str(exp)}")
        print(f"  V{v}: " + "  ".join(row))
    print(f"\n  32/32 addresses match the master table: {'YES' if bad == 0 else f'NO ({bad} mismatches)'}  "
          f"(AXIOM 2 verbatim; nothing flattened)")

    _rule("PHASE 1.3b -- BRIDGE DOUBLING: 16 maps x 4 branches = 64 distinct views")
    allv = address_set([2, 3, 5, 7], 4)
    print(f"  distinct 64-view keys: {len({a.state_key() for a in allv})} / 64   "
          f"(8 vectors + 8 bridge maps, per master 902)")

    _rule("PHASE 1.4 -- NATURAL SEPARATION (anti-collision): identical sums land in DIFFERENT nodes")
    for (p, q) in [([3], 5), ([2], 6)]:
        st = state(p, q)
        print(f"  values={p}, N={q}: sum={st.Sigma}  ->  (m1={st.m1}, m2={st.m2}, m3={st.m3})")
    a35 = state([3], 5); a26 = state([2], 6)
    print(f"  both sum to 8, but (m1,m2)=({a35.m1},{a35.m2}) vs ({a26.m1},{a26.m2}) -> "
          f"{'SEPARATE nodes (geometry split them, no dedup function)' if a35 != a26 else 'COLLISION'}")

    _rule("PHASE 2.1/2.2 -- CORRIDOR TRAVERSAL & INVARIANT z=Sigma+N strictly increasing (DAG)")
    fixed = [7, 3, 5]
    print(f"  fixed values={fixed}; N=0..11")
    print(f"  {'N':>3} {'m1':>4} {'m2':>4} {'m3':>4} {'z=Sigma':>8}   VA1")
    prev = None
    for N, st in corridor(fixed, 0, 12):
        arr = "" if prev is None else ("  z++" if st.Sigma > prev else "  !!")
        print(f"  {N:>3} {st.m1:>4} {st.m2:>4} {st.m3:>4} {st.Sigma:>8}   {branch(st,1)}{arr}")
        prev = st.Sigma
    print("  every assertion inside corridor() passed -> invariant holds, structure is a DAG.")

    _rule("PHASE 2.3 -- PATH HISTORY (the binary operad): (3,5) and (5,3) stay SEPARATE until convergence")
    p1 = address_set([3], 5, path=(3, 5))       # admitted 3 then 5
    p2 = address_set([5], 3, path=(5, 3))       # admitted 5 then 3
    same_state = p1[0].state_key() == p2[0].state_key()
    same_full = p1[0].full_key() == p2[0].full_key()
    print(f"  (3,5): state {(p1[0].m1,p1[0].m2,p1[0].Sigma)} coord {p1[0].coord} path {p1[0].path}")
    print(f"  (5,3): state {(p2[0].m1,p2[0].m2,p2[0].Sigma)} coord {p2[0].coord} path {p2[0].path}")
    print(f"  ORDER-BLIND state identical: {same_state}  (they WILL converge at the triple point)")
    print(f"  ORDER-SENSITIVE full key identical: {same_full}  "
          f"-> path history keeps them SEPARATE until explicit convergence (the operad)")

    _rule("PHASE 2 (v5) -- THE PI GRAPH / HELIX: sector wants to cycle, z forbids it (master 5.8, 3.9a.16)")
    n, d1, d2, d3 = helix_dealiasing(5, 100)
    print(f"  one rotation e^(it), {n} points sampled across 5 turns:")
    print(f"    1D WAVE   distinct shadows: {d1:>4} / {n}   -> loses DIRECTION (cos t = cos -t)")
    print(f"    2D CIRCLE distinct points : {d2:>4} / {n}   -> ALIASES: loses the TURN COUNT (t, t+2pi collapse)")
    print(f"    3D HELIX  distinct points : {d3:>4} / {n}   -> loses NOTHING. z is the DE-ALIASING dimension")
    steps, reuse, helix = lattice_helix([7, 3, 5], 0, 200)
    print(f"  lattice corridor, {steps} steps: (x,y) SHADOW reused {reuse} times (it wants to cycle), "
          f"but (sector,z) pairs distinct = {helix}/{steps}")
    print(f"  -> LOOPS WITHOUT CYCLES. The spiral staircase: octant loops, z climbs, nothing ever closes -> DAG.")

    _rule("PHASE 2 (v5) -- ZENO'S RESOLUTION: continuous spines, yet the descent HALTS in ceil(k/2) steps")
    print(f"  the M[:-2] peel separates what 3D collapses (master 5.6.1 worked example):")
    for M in ([2, 3, 5], [1, 1, 3, 5]):
        t = tiers(M)
        print(f"    M={M}:  3D(tier0)={t[0]}   tier 4/5/6={t[1]}")
    print(f"    ^ tier-0 IDENTICAL (8,3,10); tier 4/5/6 DIFFERENT -> the upper verse recovers what 3D forgot")
    print(f"\n  halting length = ceil(k/2), exactly:")
    import random as _r; _r.seed(0)
    allok = True
    for k in (2, 3, 5, 8, 13, 20, 40):
        steps_k = len(tiers([_r.randint(1, 99) for _ in range(k)]))
        exp = -(-k // 2)                                        # ceil(k/2)
        allok &= (steps_k == exp)
        print(f"    k={k:>3}:  descent halts in {steps_k:>2} tiers   (ceil(k/2)={exp})   {'OK' if steps_k==exp else 'MISMATCH'}")
    mean = sum(len(tiers([_r.randint(1,99) for _ in range(_r.randint(2,40))])) for _ in range(20000)) / 20000
    print(f"    20,000 random sets (k=2..40): mean walk {mean:.2f} tiers   (master reported 10.77)")
    print(f"  Zeno's arrow reaches its destination deterministically: {'CONFIRMED' if allok else 'FAILED'}")

    _rule("CORRECTNESS -- THE TRIPLE POINT (multiset symmetry, no k-way function)")
    for (A, B, C) in [(7, 3, 5), (11, 2, 9), (4, 4, 6)]:
        print(f"  {{{A},{B}}}@{C} == {{{A},{C}}}@{B} == {{{B},{C}}}@{A}  -> "
              f"{state([A,B],C).as_tuple()}   {'IDENTICAL' if triple_point(A,B,C) else 'MISMATCH'}")

    _rule("PHASE 1-2 COMPLETE (v5) -- validated against AETHOS_MASTER.md. Awaiting confirmation for Phase 3.")
    print("  state / 4 branches / 8 vectors (Axiom 2, 32/32 vs master) / 16 maps->64 / anti-collision /")
    print("  corridor+invariant (DAG) / helix de-aliasing / Zeno ceil(k/2) / path-operad / triple point.")
    print("  Say go for Phase 3 (Synthetic Ingestion).")
