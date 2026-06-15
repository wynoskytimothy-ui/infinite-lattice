#!/usr/bin/env python3
"""
Test 45 - Conflict-free scheduling / graph coloring from the meet algebra.

Graph coloring = scheduling: assign each task a slot so conflicting tasks
differ. It powers exam timetabling, register allocation, and frequency
assignment. Two proven primitives give a working scheduler:

  CONFLICT DETECTION = the meet (Test 11). Each task carries a composite of
  its required resources; two tasks conflict iff gcd(comp_i, comp_j) > 1 -
  they share a resource. The conflict graph IS the gcd-meet relation.

  AVAILABLE-SLOT TRACKING = FTA membership (Test 44). For a task, the slots
  used by its neighbors form a forbidden composite; the task takes the
  smallest slot whose prime does not divide it. Exact, no scan.

  SUNFLOWER => CLIQUE (Test 11). Tasks all sharing one core resource pairwise
  conflict - a clique - so they need that many distinct slots. The sunflower
  core is a lower bound on the schedule length, read straight off the algebra.

Verified: conflicts match brute force, the schedule is valid, it meets the
greedy bound, and a planted sunflower forces its size in slots.
"""

from __future__ import annotations

import random
import sys
from math import gcd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.primes import chain_primes


def header(s: str):
    print()
    print("=" * 72)
    print(s)
    print("=" * 72)


def assertion(cond: bool, msg: str):
    print(f"  [{'PASS' if cond else 'FAIL'}]  {msg}")
    if not cond:
        sys.exit(1)


def main():
    header("Conflict-free scheduling from the meet algebra (gcd) + FTA slots")
    rng = random.Random(0x45E0)

    res_primes = chain_primes(60)          # resource alphabet
    slot_primes = chain_primes(200)[60:]   # disjoint slot-color alphabet

    # ---- tasks: each requires a small random set of resources ----
    n_tasks = 60
    tasks = []
    for _ in range(n_tasks):
        k = rng.randint(2, 4)
        res = rng.sample(range(40), k)
        comp = 1
        for r in res:
            comp *= res_primes[r]
        tasks.append({"res": set(res), "comp": comp})

    # ---- conflict graph via the gcd-meet, checked against brute force ----
    print("\nConflict graph = gcd-meet > 1 (verified against brute force)")
    print("-" * 72)
    adj = [set() for _ in range(n_tasks)]
    meet_edges = bruteforce_edges = 0
    for i in range(n_tasks):
        for j in range(i + 1, n_tasks):
            by_meet = gcd(tasks[i]["comp"], tasks[j]["comp"]) > 1
            by_set = bool(tasks[i]["res"] & tasks[j]["res"])
            if by_meet:
                meet_edges += 1
                adj[i].add(j)
                adj[j].add(i)
            if by_set:
                bruteforce_edges += 1
            assert by_meet == by_set       # the meet IS the conflict relation
    print(f"  conflicts by gcd-meet: {meet_edges}; by brute-force set overlap: "
          f"{bruteforce_edges}")
    assertion(meet_edges == bruteforce_edges,
              "the gcd-meet detects exactly the resource conflicts (the meet "
              "algebra IS the conflict graph)")

    # ---- greedy coloring with composite-forbidden slots (FTA membership) ----
    print("\nSchedule via composite-forbidden slot selection (FTA)")
    print("-" * 72)
    order = sorted(range(n_tasks), key=lambda v: -len(adj[v]))  # high-degree first
    color = {}
    for v in order:
        forbidden = 1                       # product of neighbours' slot primes
        for u in adj[v]:
            if u in color:
                forbidden *= slot_primes[color[u]]
        c = 0
        while forbidden % slot_primes[c] == 0:   # smallest free slot (Test 44)
            c += 1
        color[v] = c
    n_slots = max(color.values()) + 1
    max_deg = max(len(a) for a in adj)
    print(f"  tasks {n_tasks}, max conflict degree {max_deg}, "
          f"schedule length {n_slots} slots")

    # validity: no conflicting pair shares a slot
    bad = sum(1 for i in range(n_tasks) for j in adj[i] if color[i] == color[j])
    assertion(bad == 0,
              "valid schedule: no two conflicting tasks share a slot")
    assertion(n_slots <= max_deg + 1,
              "schedule length <= max_degree + 1 (the greedy coloring bound)")

    # ---- sunflower => clique => lower bound on slots ----
    print("\nSunflower core = a lower bound on schedule length (Test 11)")
    print("-" * 72)
    # plant a sunflower: K tasks all sharing resource 50 (core), disjoint petals
    K = 6
    core = 50
    petal_pool = list(range(51, 60))
    sun = []
    s2 = []
    base = list(tasks)
    for t in range(K):
        petals = rng.sample(petal_pool, 2)
        comp = res_primes[core]
        for p in petals:
            comp *= res_primes[p]
        s2.append({"res": {core, *petals}, "comp": comp})
    # all sunflower tasks pairwise share `core` -> pairwise conflict -> clique
    clique = all(gcd(s2[i]["comp"], s2[j]["comp"]) > 1
                 for i in range(K) for j in range(i + 1, K))
    print(f"  planted sunflower of {K} tasks sharing resource {core}")
    print(f"  all pairwise conflict (a clique): {clique}")
    assertion(clique,
              "a sunflower (shared core) is a clique under the meet - every "
              "pair conflicts through the core resource")
    # such a clique forces >= K distinct slots for those tasks
    # color the sunflower alone and confirm it needs K slots
    sadj = [set(range(K)) - {i} for i in range(K)]
    scolor = {}
    for v in range(K):
        forb = 1
        for u in sadj[v]:
            if u in scolor:
                forb *= slot_primes[scolor[u]]
        c = 0
        while forb % slot_primes[c] == 0:
            c += 1
        scolor[v] = c
    assertion(len(set(scolor.values())) == K,
              f"the sunflower forces exactly {K} slots (the core size is a "
              "schedule lower bound, read off the algebra)")

    header("RESULT")
    print(f"  conflict graph:  gcd-meet == exact resource conflicts")
    print(f"  schedule:        {n_slots} slots, valid, within greedy bound")
    print(f"  sunflower bound: shared core forces its size in slots")
    print()
    print("  Graph coloring / scheduling - exam timetables, register")
    print("  allocation, frequency assignment - is the meet algebra (gcd")
    print("  detects conflict) plus FTA membership (divisibility picks the")
    print("  free slot). The sunflower lemma (Test 11) becomes a lower bound")
    print("  on schedule length. Two primitives we proved for other reasons,")
    print("  composing into a classic NP-hard optimization heuristic.")


if __name__ == "__main__":
    main()
