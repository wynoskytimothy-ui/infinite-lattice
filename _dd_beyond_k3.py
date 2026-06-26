"""
_dd_beyond_k3.py -- ADVERSARIAL deep-dive on the two structural walls.

WALL A: k>=4 meets do NOT collapse to one node (triple is the atom).
  A1. Raw k-way meet: does triple_equalization generalize to k anchors? Measure
      "all (k) drop-one rails agree on ONE node" rate for k=3..8.
  A2. Nested triples / Merkle-of-meets: build a balanced ternary tree of triple
      meets over k leaves. Measure any-(k-1)-of-k RECOVERY (erase 1 leaf, rebuild)
      + cost (nodes stored, bytes) for k=4..81.
  A3. A DIFFERENT operator for direct any-(k-1)-of-k on one node: the sum-lock
      zeta = sum(members) is a (k,k-1) erasure code by itself. Measure recovery
      and where the readout (z,zeta) stops disambiguating the SET.

WALL B: depth capped at 486 (fixed PROMOTION_POOL).
  B1. LazyLattice: replace the fixed pool with an on-demand prime generator.
      Re-run the deep tower; show height > 8 with 0 provenance collisions.
  B2. bytes-per-node vs depth curve (sub_chain storage = the real cost).
  B3. composite-ID promotion (no prime gen at all): promote -> a fresh composite
      ID = product-free counter; show identical provenance with O(1) alloc.

Run: PYTHONUTF8=1 python _dd_beyond_k3.py  (writes UTF-8 report to _dd_beyond_k3_out.txt)
"""
from __future__ import annotations
import sys, itertools, math, time
from dataclasses import dataclass, field

from aethos_complex_plane import (
    wing_transform, equalize_witness, missing_member, triple_equalization,
)
from aethos_lattice import BranchKind

OUT = []
def pr(*a):
    s = " ".join(str(x) for x in a)
    OUT.append(s)
    print(s)

def hr(t):
    pr("=" * 72); pr(t); pr("=" * 72)


# ---------------------------------------------------------------------------
# WALL A
# ---------------------------------------------------------------------------

def kway_meet_node(members, branch=BranchKind.VA1, wing=1):
    """The 'meet node' of a k-set via the missing-variable rail.
    For each dropped member, transgress the remaining (k-1) subset to the
    dropped anchor. For k=3 all 3 rails agree (the verified atom). For k>3 we
    MEASURE whether they still agree."""
    chain = sorted(members)
    nodes = []
    for drop in range(len(chain)):
        sub = [chain[i] for i in range(len(chain)) if i != drop]
        m, psi = equalize_witness(chain, sub, branch=branch, wing=wing)
        nodes.append((psi.coord, round(psi.zeta, 9)))
    return nodes


def test_A1_raw_kway():
    hr("A1. RAW k-way meet: do all k drop-one rails land on ONE node?")
    pr("  k | trials | rails-agree(all k same node) | distinct-nodes typical | zeta-agree")
    import random
    random.seed(11)
    results = {}
    for k in range(3, 9):
        agree = 0
        zeta_agree = 0
        trials = 300
        distinct_hist = {}
        for _ in range(trials):
            members = random.sample(range(2, 5000), k)
            nodes = kway_meet_node(members)
            ncoords = len(set(c for c, z in nodes))
            nzetas = len(set(z for c, z in nodes))
            distinct_hist[ncoords] = distinct_hist.get(ncoords, 0) + 1
            if ncoords == 1:
                agree += 1
            if nzetas == 1:
                zeta_agree += 1
        results[k] = (agree, zeta_agree, trials)
        typ = max(distinct_hist, key=distinct_hist.get)
        pr(f"  {k} | {trials:6d} | {agree:6d} ({100*agree/trials:5.1f}%) | "
           f"{typ} | zeta-agree {zeta_agree:6d} ({100*zeta_agree/trials:5.1f}%)")
    return results


@dataclass
class MerkleMeet:
    """Balanced ternary Merkle-of-meets over leaves. Each internal node is a
    3-way meet (triple atom) of its children's *summary ids* (we use the
    conserved zeta = sum as the child's summary; the atom is invertible so the
    3 children of any node are recoverable from any 2 + the node's stored zeta).
    Stores sub_chain per internal node (the real provenance + cost)."""
    fanout: int = 3
    nodes: dict = field(default_factory=dict)   # id -> (children_ids, zeta, level)
    leaf_val: dict = field(default_factory=dict)
    _next: int = 0

    def _alloc(self):
        self._next += 1
        return -self._next   # negative ids = internal, positive given = leaves

    def build(self, leaves):
        for v in leaves:
            self.leaf_val[v] = v
        level = list(leaves)
        lv = 0
        while len(level) > 1:
            nxt = []
            # pad last group if needed (allow ragged final group >=2)
            for i in range(0, len(level), self.fanout):
                grp = level[i:i+self.fanout]
                if len(grp) == 1:
                    nxt.append(grp[0]); continue
                nid = self._alloc()
                # zeta-lock = sum of children's summaries (the conserved depth)
                zeta = sum(self._summary(c) for c in grp)
                self.nodes[nid] = (tuple(grp), zeta, lv+1)
                nxt.append(nid)
            level = nxt; lv += 1
        return level[0]

    def _summary(self, nid):
        if nid in self.leaf_val:
            return self.leaf_val[nid]
        return self.nodes[nid][1]   # zeta

    def walk_down(self, nid):
        if nid in self.leaf_val:
            return (self.leaf_val[nid],)
        out = []
        for c in self.nodes[nid][0]:
            out.extend(self.walk_down(c))
        return tuple(out)

    def recover_one_erased(self, root, erase_leaf):
        """Erase one leaf value; rebuild it from the sum-lock of its parent group
        (any-(g-1)-of-g per node). Returns recovered value or None."""
        # find the internal node whose direct children include erase_leaf
        for nid, (children, zeta, lv) in self.nodes.items():
            if erase_leaf in children:
                kept = [self._summary(c) for c in children if c != erase_leaf]
                # erased leaf is a LEAF (summary == value), so:
                return zeta - sum(kept)
        return None

    def node_count(self):
        return len(self.nodes)

    def bytes_estimate(self):
        # each internal node stores: children tuple (fanout ints) + zeta (1 int)
        # + level (1 int). Use 8 bytes per int as a uniform model.
        b = 0
        for (children, zeta, lv) in self.nodes.values():
            b += 8 * (len(children) + 2)
        return b


def test_A2_merkle():
    hr("A2. NESTED triples (Merkle-of-meets): any-(k-1)-of-k recovery + cost")
    pr("  k(leaves) | tree-height | internal-nodes | bytes | recover-1-erased rate | walk_down ok")
    import random
    random.seed(13)
    rows = {}
    for k in (4, 9, 27, 81, 243, 729):
        leaves = random.sample(range(1000, 10_000_000), k)
        mk = MerkleMeet(fanout=3)
        root = mk.build(leaves)
        # provenance: walk_down recovers exactly the leaf multiset
        wd = sorted(mk.walk_down(root))
        wd_ok = (wd == sorted(leaves))
        # erase each leaf once, recover from its parent group's sum-lock
        ok = 0
        for L in leaves:
            rec = mk.recover_one_erased(root, L)
            if rec is not None and int(round(rec)) == L:
                ok += 1
        height = max(lv for (_, _, lv) in mk.nodes.values()) if mk.nodes else 0
        rate = ok / k
        rows[k] = (height, mk.node_count(), mk.bytes_estimate(), rate, wd_ok)
        pr(f"  {k:5d} | h={height:3d} | {mk.node_count():6d} | {mk.bytes_estimate():9d} | "
           f"{ok}/{k} ({100*rate:5.1f}%) | walk_down={wd_ok}")
    return rows


def test_A3_sumlock_direct():
    hr("A3. DIRECT (k,k-1) sum-lock on ONE node: recovery + SET disambiguation")
    pr("  The conserved zeta=sum(members) is itself a (k,k-1) erasure code:")
    pr("  erase any one member -> recover = zeta - sum(rest). Always exact.")
    pr("  BUT: does the lattice READOUT (z,zeta) of a k-set still pick out the SET?")
    import random
    random.seed(17)
    pr("  k | recover-1 rate | distinct-set-readout uniqueness on shared pool")
    for k in range(3, 9):
        # recovery: trivially 100% (algebra). measure to confirm no float drift.
        rec_ok = 0
        T = 500
        for _ in range(T):
            members = random.sample(range(2, 100000), k)
            zeta = sum(members)
            drop = random.randrange(k)
            kept = [members[i] for i in range(k) if i != drop]
            r = zeta - sum(kept)
            if r == members[drop]:
                rec_ok += 1
        # SET disambiguation: enumerate all k-subsets of a small pool, ask how
        # many DISTINCT (sorted-sub-meet readout) vs distinct sets.
        M = 30
        seen_sum = {}
        nsets = 0
        for combo in itertools.combinations(range(1, M+1), k):
            nsets += 1
            seen_sum[sum(combo)] = seen_sum.get(sum(combo), 0) + 1
        sum_distinct = len(seen_sum)
        pr(f"  {k} | {rec_ok}/{T} (100%={'Y' if rec_ok==T else 'N'}) | "
           f"pool1..{M}: {nsets} sets -> {sum_distinct} distinct sums "
           f"({100*sum_distinct/nsets:.2f}% uniq)")


# ---------------------------------------------------------------------------
# WALL B
# ---------------------------------------------------------------------------

class LazyLattice:
    """Recursive lattice whose promotion ids come from an UNBOUNDED generator.
    Two id strategies:
      'prime'   : lazily-sieved primes (semantically faithful 'promoted prime')
      'counter' : composite/opaque ids (just a monotone counter) -- O(1) alloc,
                  no sieving, identical provenance via stored sub_chain.
    Provenance lives in sub_chain (walk_down), so id strategy is irrelevant to
    correctness; we MEASURE that both give 0 provenance collisions to any depth.
    """
    def __init__(self, strategy="prime", base_offset=0):
        self.strategy = strategy
        self.nodes = {}            # id -> (level, sub_chain or None, label)
        self.parents = {}          # id -> list of parent ids
        self._counter = 10**9 + base_offset
        self._prime_cur = 10**7 + base_offset   # start primes well above leaf band
        self._used = set()

    def _is_prime(self, x):
        if x < 2: return False
        if x % 2 == 0: return x == 2
        i = 3
        while i*i <= x:
            if x % i == 0: return False
            i += 2
        return True

    def _next_id(self):
        if self.strategy == "counter":
            self._counter += 1
            return self._counter
        # prime
        x = self._prime_cur + 1
        while not self._is_prime(x):
            x += 1
        self._prime_cur = x
        return x

    def register_base(self, v):
        if v not in self.nodes:
            self.nodes[v] = (0, None, "")
            self.parents.setdefault(v, [])

    def promote(self, chain, label=""):
        chain = tuple(sorted(int(p) for p in chain))
        for p in chain:
            if p not in self.nodes:
                self.register_base(p)
        nid = self._next_id()
        while nid in self.nodes:        # never collide with an existing id
            nid = self._next_id()
        lvl = max(self.nodes[p][0] for p in chain) + 1
        self.nodes[nid] = (lvl, chain, label)
        self.parents.setdefault(nid, [])
        for p in chain:
            self.parents.setdefault(p, []).append(nid)
        return nid

    def walk_down(self, nid):
        lvl, sub, _ = self.nodes[nid]
        if sub is None:
            return (nid,)
        out = []
        for p in sub:
            out.extend(self.walk_down(p))
        return tuple(out)

    def max_level(self):
        return max(l for (l, _, _) in self.nodes.values())

    def n_promoted(self):
        return sum(1 for (l, _, _) in self.nodes.values() if l > 0)

    def bytes_per_node(self):
        # storage cost of a promoted node = its sub_chain (fanout ids) + level.
        # report total provenance bytes and per-node average at each level.
        per_level = {}
        for nid, (lvl, sub, _) in self.nodes.items():
            if sub is None: continue
            b = 8 * (len(sub) + 1)
            per_level.setdefault(lvl, []).append(b)
        return {lvl: (len(v), sum(v)/len(v)) for lvl, v in per_level.items()}


def build_lazy_tower(strategy, group, height):
    """Build a COMPLETE g-ary tower of the requested HEIGHT (depth past 8)."""
    L = LazyLattice(strategy=strategy)
    n_leaves = group ** height
    leaves = list(range(2, 2 + n_leaves))   # opaque leaf ids (any number set works)
    for v in leaves:
        L.register_base(v)
    level = leaves
    while len(level) > 1:
        nxt = []
        for i in range(0, len(level), group):
            grp = level[i:i+group]
            if len(grp) < 2:
                nxt.append(grp[0]); continue
            nxt.append(L.promote(grp))
        level = nxt
    return L, level[0]


def test_B1_depth():
    hr("B1. LAZY tower: blow past height 8 with 0 provenance collisions")
    pr("  strategy | group | target-h | max_level | promoted | walk_down-leaves | prov-collisions")
    import random
    for strategy in ("counter", "prime"):
        for (group, height) in [(3, 9), (3, 12), (2, 16), (4, 8)]:
            t0 = time.time()
            L, root = build_lazy_tower(strategy, group, height)
            wd = L.walk_down(root)
            n_leaves = group ** height
            wd_ok = (len(wd) == n_leaves and len(set(wd)) == len(wd))
            # provenance collisions: do two distinct promoted nodes expand to the
            # same base multiset?
            seen = {}
            coll = 0
            for nid, (lvl, sub, _) in L.nodes.items():
                if sub is None: continue
                key = tuple(sorted(L.walk_down(nid)))
                if key in seen: coll += 1
                else: seen[key] = nid
            dt = time.time() - t0
            pr(f"  {strategy:7s} | g={group} | h={height:2d} | maxL={L.max_level():2d} | "
               f"prom={L.n_promoted():7d} | leaves={len(wd)} ok={wd_ok} | "
               f"coll={coll} | {dt:5.2f}s")
    return True


def test_B2_bytes_curve():
    hr("B2. bytes-per-node vs DEPTH curve (provenance storage = the real cost)")
    L, root = build_lazy_tower("counter", group=3, height=10)  # 59049 leaves
    bp = L.bytes_per_node()
    pr("  group=3 tower height=10 (59049 leaves):")
    pr("  level | #nodes | avg bytes/node (sub_chain+level)")
    total = 0
    for lvl in sorted(bp):
        cnt, avg = bp[lvl]
        total += cnt * avg
        pr(f"    L{lvl:2d} | {cnt:6d} | {avg:.1f}")
    n_leaves = 3 ** 10
    pr(f"  total provenance bytes={int(total)} for {n_leaves} leaves "
       f"=> {total/n_leaves:.2f} bytes/leaf amortized (flat in depth: each "
       f"node is g+1 ids regardless of level)")
    pr("  KEY: bytes/node is CONSTANT in depth (=g+1 ids). Depth is unbounded;")
    pr("  cost grows with #nodes (= (N-1)/(g-1) internal), not with height.")


def test_B3_composite():
    hr("B3. COMPOSITE-id promotion: O(1) alloc, identical provenance, no sieve")
    # head-to-head timing: prime-gen vs counter at the same tower size.
    import random
    g, h = 3, 9   # 19683 leaves -- the original miniverse target
    res = {}
    for strat in ("prime", "counter"):
        t0 = time.time()
        L, root = build_lazy_tower(strat, g, h)
        wd = L.walk_down(root)
        dt = time.time() - t0
        ok = (len(set(wd)) == g**h)
        res[strat] = (dt, L.max_level(), ok)
        pr(f"  {strat:7s}: build {dt:6.3f}s  maxLevel={L.max_level()}  "
           f"provenance-unique={ok}")
    sp = res["prime"][0] / max(res["counter"][0], 1e-9)
    pr(f"  composite/counter is {sp:.1f}x faster than prime-gen at the SAME "
       f"depth, SAME provenance. Promoted-PRIME semantics are optional sugar;")
    pr("  the address book only needs a fresh unique id per promotion.")


if __name__ == "__main__":
    hr("DEEP DIVE: BEYOND-k3 and BEYOND-486-DEPTH (adversarial, measured)")
    test_A1_raw_kway()
    test_A2_merkle()
    test_A3_sumlock_direct()
    test_B1_depth()
    test_B2_bytes_curve()
    test_B3_composite()
    hr("DONE")
    with open("_dd_beyond_k3_out.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(OUT))
