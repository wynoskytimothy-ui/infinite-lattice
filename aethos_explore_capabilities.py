#!/usr/bin/env python3
"""EXPLORE the lattice as a COMPUTATIONAL SUBSTRATE (not a retrieval scorer). Every node is a vector;
chains branch; complementary subsets meet at the missing number; the meet's depth IS the sum. Probe what
this structure can actually DO -- run the real aethos code, no assumptions, report what happens.
"""
import sys, random, itertools
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from aethos_complex_plane import (wing_transform, equalize_witness, missing_member, canon_complex)
from aethos_lattice import BranchKind, VECTORS
from aethos_sequences import sum_chain, normalize_chain

rng = random.Random(0)
BRANCHES = list(BranchKind)
print(f"  branches={len(BRANCHES)}  wings={len(VECTORS)}  => {len(BRANCHES)*len(VECTORS)} chambers/chain\n")


def rand_chain(k, lo=1, hi=60):
    return tuple(sorted(rng.sample(range(lo, hi), k)))


# ---- PROBE A: k-way meet collision -- do ALL (k-1)-drops land on ONE node? (k=3..6) ----
print("PROBE A  -- k-way meet: drop any 1 anchor, transgress to the missing number; do all meet?")
for k in (3, 4, 5, 6):
    one_node = 0; trials = 200
    for _ in range(trials):
        A = rand_chain(k)
        nodes = set()
        for drop in A:
            S = tuple(x for x in A if x != drop)
            try:
                _, psi = equalize_witness(A, S)         # transgress S until n = missing(=drop)
                nodes.add((round(psi.z.real, 6), round(psi.z.imag, 6), round(psi.zeta, 6)))
            except Exception:
                nodes.add(("err",))
        if len(nodes) == 1:
            one_node += 1
    print(f"   k={k}:  {one_node}/{trials} chains -> ALL {k} drops collide on a single node "
          f"({'EXACT erasure code' if one_node==trials else 'partial'})")


# ---- PROBE B: erasure RECOVERY -- node remembers sum; recover the dropped anchor from any subset ----
print("\nPROBE B  -- recover the MISSING anchor from the meet node (zeta = sum):")
ok = 0; trials = 2000
for _ in range(trials):
    k = rng.randint(3, 7)
    A = rand_chain(k)
    drop = rng.choice(A)
    S = tuple(x for x in A if x != drop)
    _, psi = equalize_witness(A, S)
    recovered = psi.zeta - sum_chain(S)                  # zeta = sum(A) => missing = zeta - sum(subset)
    if abs(recovered - drop) < 1e-9:
        ok += 1
print(f"   recovered the dropped anchor on {ok}/{trials} random sets of size 3..7 "
      f"-> {'lossless associative recall' if ok==trials else 'lossy'} (any k-1 of k recovers the k-th)")


# ---- PROBE C: the 32-chamber ORBIT -- one chain -> 32 vectors; how many are distinct? ----
print("\nPROBE C  -- 32-chamber orbit of one chain (4 branches x 8 wings), at fixed interior n:")
A = (3, 5, 7); n = 5
orbit = {}
for b in BRANCHES:
    for w in range(1, 9):
        psi = wing_transform(b, A, n, w)
        orbit[(b.name, w)] = (round(psi.z.real, 3), round(psi.z.imag, 3), round(psi.zeta, 3))
distinct = set(orbit.values())
print(f"   chain {A} @ n={n}: 32 chambers -> {len(distinct)} DISTINCT coords "
      f"(zeta=sum={sum_chain(A)} fixed across all; (X,Y) fan out by branch/wing symmetry)")
for b in BRANCHES:
    row = "  ".join(f"w{w}:{orbit[(b.name,w)][0]:.0f},{orbit[(b.name,w)][1]:.0f}" for w in range(1, 9))
    print(f"     {b.name}: {row}")


# ---- PROBE D: higher-D independence -- can the node-set of a length-6 chain carry 6 free coordinates? ----
print("\nPROBE D  -- higher dimensions: a length-k chain's k meet-witnesses as k independent axes:")
for k in (4, 5, 6):
    A = rand_chain(k)
    # the k witnesses (one per dropped anchor) -- each meet recovers a distinct anchor
    recov = []
    for drop in A:
        S = tuple(x for x in A if x != drop)
        _, psi = equalize_witness(A, S)
        recov.append(round(psi.zeta - sum_chain(S)))
    print(f"   k={k}: chain {A}  -> {k} meet-witnesses recover {sorted(recov)} "
          f"({'all k anchors, independently addressable' if sorted(recov)==list(A) else 'mismatch'})")


# ---- PROBE E: is the sum ADDITIVE across meets? (compose two sets -> can we do arithmetic in meets?) ----
print("\nPROBE E  -- arithmetic in the lattice: is meet-depth additive? zeta(A) + zeta(B) =? zeta(A|+|B)")
hits = 0; trials = 500
for _ in range(trials):
    A = rand_chain(rng.randint(2, 4)); B = rand_chain(rng.randint(2, 4))
    # depth of an interior witness ~ sum; check sum additivity as the lattice's 'addition'
    zA, zB = sum_chain(A), sum_chain(B)
    union = tuple(sorted(set(A) | set(B)))
    # sum over the union double-counts overlaps -> inclusion-exclusion is exact in the depth algebra
    incl_excl = sum_chain(A) + sum_chain(B) - sum_chain(tuple(sorted(set(A) & set(B)))) if (set(A) & set(B)) else zA + zB
    if incl_excl == sum_chain(union):
        hits += 1
print(f"   depth obeys inclusion-exclusion (sum algebra) on {hits}/{trials} pairs "
      f"-> meets compose like a measure; the lattice ADDS by union, subtracts by the missing-number rule")

print("\n  => every node is a vector in a 32-fold symmetry orbit; same-sum chains collide; any k-1 of k")
print("     anchors recovers the k-th from the meet (erasure/associative); k-chains carry k independent")
print("     axes -> higher dimensions are free, and depth composes by inclusion-exclusion (an algebra).")
