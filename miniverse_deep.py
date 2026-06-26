"""
miniverse_deep.py - push depth to pool exhaustion; stress provenance + address
uniqueness at extreme recursion. Two-sided: find where it BREAKS.
"""
from __future__ import annotations
from aethos_recursive_lattice import RecursiveLattice
from aethos_promotion import PROMOTION_POOL


def primes_upto(n):
    s = [True]*(n+1); s[0]=s[1]=False
    for i in range(2,int(n**0.5)+1):
        if s[i]:
            for j in range(i*i,n+1,i): s[j]=False
    return [i for i,v in enumerate(s) if v]


def build_tower(group, n_base_primes):
    L = RecursiveLattice()
    base = [p for p in primes_upto(100000) if p >= 3][:n_base_primes]
    # avoid pool collisions: pool starts at 107; keep base below 107 OR above pool max.
    base = [p for p in base if p < 107 or p > 3673]
    base = base[:n_base_primes]
    for p in base:
        L.register_base(p)
    level_nodes = {0: list(base)}
    level = 0
    while True:
        cur = level_nodes[level]
        if len(cur) < group:
            break
        nxt = []
        exhausted = False
        for i in range(0, len(cur)-group+1, group):
            try:
                newp = L.promote(cur[i:i+group], label=f"L{level+1}.{i//group}")
            except RuntimeError:
                exhausted = True
                break
            nxt.append(newp)
        if not nxt:
            break
        level += 1
        level_nodes[level] = nxt
        if exhausted:
            break
    return L, level_nodes


print("="*70)
print("DEEP TOWER A: narrow base (group=3) -> how DEEP before it ends?")
print("="*70)
L, lv = build_tower(group=3, n_base_primes=3**9)  # 19683 leaves: 9 clean levels if pool allowed
st = L.stats()
print(f"max_level={st['max_level']}  total_nodes={st['total_nodes']}")
print(f"level_counts={st['level_counts']}")
print(f"pool_used={st['pool_used']}/{len(PROMOTION_POOL)}  remaining={st['pool_remaining']}")

# Provenance at the deepest reachable node
deepest = max(lv)
top = lv[deepest][0]
wd = L.walk_down(top)
print(f"\nDeepest node {top} @L{deepest}: walk_down -> {len(wd)} leaves, "
      f"unique={len(set(wd))==len(wd)}")

# Provenance collision check across ALL promoted nodes
seen = {}
coll = 0
for p,node in L.nodes.items():
    if node.is_promoted:
        key = tuple(sorted(L.walk_down(p)))
        if key in seen: coll += 1
        else: seen[key] = p
print(f"Promoted nodes={sum(1 for n in L.nodes.values() if n.is_promoted)}  "
      f"provenance collisions={coll}  (0 => every address decodes uniquely)")

print("\n" + "="*70)
print("ADDRESS UNIQUENESS ACROSS LEVELS: do L1/L2/.. chamber coords collide?")
print("="*70)
# Map each distinct chamber coord to the set of LEVELS that produce it.
coord_levels = {}
for p,node in L.nodes.items():
    for (_b,_w,coord) in L.chambers(p):
        coord_levels.setdefault(coord, set()).add(node.level)
cross = sum(1 for s in coord_levels.values() if len(s) > 1)
print(f"distinct chamber coords={len(coord_levels)}  "
      f"coords shared across >1 level={cross}")
print("  (cross-level sharing is EXPECTED: chamber coord is value-based, not "
      "identity-based; provenance lives in walk_down/sub_chain, not the coord)")

print("\n" + "="*70)
print("DEPTH LIMIT (the honest wall): pool=486 names. With group g, level L")
print("needs sum_{i=1..L} (#nodes at i) promoted IDs. Max clean depth:")
print("="*70)
for g in (2,3,4,5):
    # geometric: to reach level L from a base wide enough, total promoted =
    # ceil over levels; with infinite base, nodes at level i = base/g^i.
    # Pool caps TOTAL promoted nodes at 486. Deepest level with a full g-ary
    # tree using <=486 internal nodes: internal nodes of complete g-ary tree
    # with L levels above base and 1 root = (g^L - 1)/(g-1) ... approximate.
    import math
    L_max = 0
    total = 0
    nodes_at = 1  # root
    # count internal (promoted) nodes for a complete g-ary tree of height h
    for h in range(1, 40):
        internal = (g**h - 1)//(g-1)  # promoted nodes for height h
        if internal <= len(PROMOTION_POOL):
            L_max = h
            total = internal
    print(f"  group g={g}: deepest COMPLETE tower height={L_max} "
          f"(uses {total} promoted IDs, base width={g**L_max} leaves)")
