"""
miniverse_probe.py - RUN the recursive sub-lattice (miniverse) capability.

Tests:
 1. A node spawns its OWN lattice: a chain whose anchors are themselves
    meet/promoted nodes, recursively. Build a deep tower.
 2. Branching factor: how many children per node, and how the node count
    grows per level.
 3. Depth: how deep the recursion addresses cleanly (level grows, walk_down
    still recovers the base chain, zeta invariants still hold).
 4. Decodability / provenance: is a deep address (path through miniverses)
    UNIQUELY decodable back to its origin base chain? Test for collisions.
 5. "Billions of addresses for free?" - count distinct chamber addresses
    reachable and compare to pool cost.
"""
from __future__ import annotations

import itertools
from aethos_recursive_lattice import RecursiveLattice
from aethos_promotion import PROMOTION_POOL
from aethos_complex_plane import wing_transform, triple_equalization
from aethos_lattice import BranchKind


def primes_upto(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i, v in enumerate(sieve) if v]


def main():
    print("=" * 70)
    print("EXPERIMENT 1: a node spawns its OWN lattice (recursive tower)")
    print("=" * 70)
    L = RecursiveLattice()

    # Base primes (level 0 leaves), excluding any that collide with the pool.
    base = [p for p in primes_upto(105) if p >= 3]  # 3..103
    for p in base:
        L.register_base(p)
    print(f"Base leaves (level 0): {len(base)} primes  {base[:6]}...{base[-3:]}")

    # Build a TOWER: group leaves into triples -> promote each to a L1 prime.
    # Then group L1 primes into triples -> L2 primes. Recurse until pool runs out.
    level_nodes = {0: list(base)}
    level = 0
    GROUP = 3  # triple is the verified atom (k=3 meet)
    while True:
        cur = level_nodes[level]
        if len(cur) < GROUP:
            break
        nxt = []
        for i in range(0, len(cur) - GROUP + 1, GROUP):
            chain = cur[i:i + GROUP]
            try:
                newp = L.promote(chain, label=f"L{level+1}_g{i//GROUP}")
            except RuntimeError:
                # pool exhausted
                break
            nxt.append(newp)
        if not nxt:
            break
        level += 1
        level_nodes[level] = nxt
        print(f"  level {level}: spawned {len(nxt)} promoted nodes "
              f"(each anchors a chain of {GROUP} level-{level-1} nodes)")
        if level_nodes[level] and len(level_nodes[level]) < GROUP:
            break
        if L.stats()["pool_remaining"] < GROUP:
            print("  (pool nearly exhausted, stopping tower)")
            break

    st = L.stats()
    print(f"\nTower stats: max_level={st['max_level']}  total_nodes={st['total_nodes']}")
    print(f"  level_counts={st['level_counts']}")
    print(f"  pool_used={st['pool_used']}  pool_remaining={st['pool_remaining']}")

    print("\n" + "=" * 70)
    print("EXPERIMENT 2: branching factor (children/node) + growth per level")
    print("=" * 70)
    # Effective branching = GROUP (each promoted node has GROUP children).
    # Measure node-count ratio level to level.
    counts = st["level_counts"]
    levels = sorted(counts)
    for i in range(1, len(levels)):
        lo, hi = levels[i-1], levels[i]
        ratio = counts[lo] / counts[hi] if counts[hi] else float("inf")
        print(f"  level {lo} ({counts[lo]}) -> level {hi} ({counts[hi]}): "
              f"contraction {ratio:.2f}x  (branching factor {GROUP})")

    print("\n" + "=" * 70)
    print("EXPERIMENT 3: how deep does the address decode CLEANLY?")
    print("=" * 70)
    # Take the top-most node and walk_down to base; check zeta-lock holds at
    # each promoted node (zeta == sum of its sub_chain's expanded leaves? we
    # check the immediate sub_chain meet via triple_equalization for k=3).
    top = level_nodes[max(level_nodes)][0]
    print(f"Top node: {top} at level {L.resolve(top).level}")
    base_chain = L.walk_down(top)
    print(f"walk_down(top) -> {len(base_chain)} base leaves: {base_chain[:8]}...")
    # Verify uniqueness of leaves in the recovered chain (tree property)
    print(f"  leaves unique under walk_down: {len(set(base_chain)) == len(base_chain)}")

    # zeta-lock check: for each promoted node whose sub_chain has length 3,
    # triple_equalization should give a coherent missing/lock structure.
    clean_levels = 0
    for lv in sorted(level_nodes):
        if lv == 0:
            continue
        ok = True
        for node_p in level_nodes[lv]:
            sc = L.resolve(node_p).sub_chain
            if sc and len(sc) == 3:
                try:
                    res = triple_equalization(sc[0], sc[1], sc[2])
                    # just require it returns the structure without error
                    if res is None:
                        ok = False
                except Exception as e:
                    ok = False
            # also require walk_down round-trips (provenance)
            wd = L.walk_down(node_p)
            if len(set(wd)) != len(wd):
                ok = False
        if ok:
            clean_levels += 1
        print(f"  level {lv}: {len(level_nodes[lv])} nodes, "
              f"triple-meet + provenance clean = {ok}")
    print(f"Clean recursive levels (above base): {clean_levels}/{max(level_nodes)}")

    print("\n" + "=" * 70)
    print("EXPERIMENT 4: DECODABILITY / PROVENANCE — unique back to origin?")
    print("=" * 70)
    # For EVERY promoted node, walk_down to base leaves. Two different promoted
    # nodes must map to DIFFERENT leaf-sets (else address is not unique).
    decode_map = {}
    collisions = 0
    for p, node in L.nodes.items():
        if node.is_promoted:
            leaves = tuple(sorted(L.walk_down(p)))
            if leaves in decode_map:
                collisions += 1
            else:
                decode_map[leaves] = p
    promoted_total = sum(1 for n in L.nodes.values() if n.is_promoted)
    print(f"Promoted nodes: {promoted_total}")
    print(f"Distinct leaf-set decodings: {len(decode_map)}")
    print(f"Provenance COLLISIONS (two nodes, same leaf set): {collisions}")
    print(f"  -> every deep address uniquely decodes to its origin: "
          f"{collisions == 0}")

    # And walk_up: from a base leaf, can we find the full provenance chain up?
    leaf = base[0]
    ancestors = L.walk_up(leaf)
    print(f"walk_up(leaf {leaf}) -> {len(ancestors)} ancestor nodes "
          f"(the miniverses containing it): {ancestors[:6]}...")

    print("\n" + "=" * 70)
    print("EXPERIMENT 5: 'billions of addresses for free?'")
    print("=" * 70)
    # Each node (any level) exposes 32 chambers (4 branch x 8 wing).
    # Count distinct (X,Y) chamber coords across the WHOLE tower, and the
    # combinatorial address space: nodes * 32 chambers.
    all_coords = set()
    chamber_rows = 0
    for p, node in L.nodes.items():
        for (branch, wing, coord) in L.chambers(p):
            all_coords.add((coord))  # (X,Y,Z) tuple
            chamber_rows += 1
    print(f"Nodes total: {len(L.nodes)}")
    print(f"Chamber rows (nodes x 32): {chamber_rows}")
    print(f"Distinct chamber coords: {len(all_coords)}")

    # The "for free" claim: pool primes CONSUMED vs ADDRESSES created.
    pool_used = st["pool_used"]
    print(f"Pool primes consumed: {pool_used}")
    print(f"Addresses (chamber coords) per pool prime consumed: "
          f"{len(all_coords)/max(pool_used,1):.1f}")

    # Combinatorial reach if we DIDN'T cap at GROUP=3 but allowed all triples
    # of base leaves to promote (theoretical address space, not built):
    from math import comb
    n_base = len(base)
    theoretical_triples = comb(n_base, 3)
    print(f"\nTheoretical: distinct base triples C({n_base},3) = "
          f"{theoretical_triples} promotable miniverses from {n_base} leaves")
    print(f"  each x 32 chambers = {theoretical_triples*32} addresses, "
          f"but only {len(PROMOTION_POOL)} pool primes exist to name them")
    print(f"  -> ADDRESS SPACE is combinatorial (C(n,3)) but NAMING is "
          f"pool-bounded ({len(PROMOTION_POOL)} promoted-prime IDs)")

    print("\n" + "=" * 70)
    print("EXPERIMENT 6: pool-free addressing? Can a path-tuple BE the address")
    print("(no pool prime needed) and still uniquely decode?")
    print("=" * 70)
    # Idea: address a miniverse node by its PATH (tuple of base leaves) rather
    # than a promoted prime ID. Then there is no pool cap. Test uniqueness of
    # path -> chamber-coord and whether chamber-coord -> path inverts.
    # Build deep paths: nested triples of triples up to depth 3.
    leaves = base[:27]  # 27 = 3^3 leaves -> one depth-3 tower
    # depth1: 9 triples
    d1 = [tuple(leaves[i:i+3]) for i in range(0, 27, 3)]
    # represent each d1 by its SUM (the zeta-lock invariant = sum of chain)
    d1_keys = [sum(t) for t in d1]
    # depth2: 3 triples of the depth1 keys
    d2 = [tuple(d1_keys[i:i+3]) for i in range(0, 9, 3)]
    d2_keys = [sum(t) for t in d2]
    # depth3: 1 triple of depth2 keys
    d3 = tuple(d2_keys)
    d3_key = sum(d3)
    print(f"depth-1 (9 triples) zeta keys: {d1_keys}")
    print(f"depth-2 (3 triples) zeta keys: {d2_keys}")
    print(f"depth-3 root zeta key: {d3_key}  (== sum of all 27 leaves "
          f"= {sum(leaves)}: {d3_key == sum(leaves)})")
    # The zeta-lock means the ROOT key = total sum, which is NOT unique to the
    # tree shape (any partition sums the same). Test: does zeta alone decode?
    # Counter: a DIFFERENT grouping of same 27 leaves gives same root zeta.
    import random
    sh = leaves[:]
    random.seed(0)
    random.shuffle(sh)
    alt_d1 = [tuple(sh[i:i+3]) for i in range(0, 27, 3)]
    alt_root = sum(sum(t) for t in alt_d1)
    print(f"Alt grouping root zeta: {alt_root}  -> same as original: "
          f"{alt_root == d3_key}")
    print("  => zeta (depth) ALONE does NOT decode the tree (sum is "
          "partition-invariant). The PATH/sub_chain tuples are what carry "
          "provenance, and walk_down on those IS injective (Exp 4).")


if __name__ == "__main__":
    main()
