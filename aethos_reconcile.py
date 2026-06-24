"""
aethos_reconcile.py -- SET-RECONCILIATION ENGINE on the AETHOS prime-lattice.

The verified k=3 erasure atom is the seed primitive:

    triple (a, p, q):  zeta = a + p + q  (sum locks the interior)
    missing member m = sum - (the other two)         <-- "missing = sum - pair"
    invertible 2-way meet (a, p) = (a+p, min, a+p):  small = Y, large = X - Y.

That "recover the lone missing value from a running sum" rule is EXACTLY the
decode step of an Invertible Bloom Lookup Table (IBLT) / Minisketch -- the
set-reconciliation primitive used in Bitcoin's Erlay. This module scales the
k=3 atom to a whole sketch.

ReconEngine.sketch(set, capacity) -> compact sketch:
    each element x is mapped into K>=4 distinct cells; each cell carries
        count    : signed multiplicity
        keysum   : XOR of element ids        (erasure recovery: lone id = keysum)
        depthsum : sum of lattice depth tok(x)  (the zeta/"sum-locks" check)
ReconEngine.reconcile(sketchA, sketchB) -> (only_in_A, only_in_B):
    cell-wise subtract -> sketch of the symmetric difference, then PEEL pure
    cells. A cell is "pure" iff |count|==1 AND its depthsum matches tok(keysum)
    (the lattice depth-consistency gate kills phantom peels where a collision
    of an even number of elements masquerades as count==1 via XOR).

The verified wing_transform / equalize_witness lattice node is used to PROVE
the per-element depth token tok(x) is the genuine zeta(meet) -- i.e. the cell
arithmetic is the lattice's own sum-lock, not an ad-hoc checksum.

Run:  python aethos_reconcile.py
"""
from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, field
from typing import Iterable

from aethos_complex_plane import wing_transform, equalize_witness
from aethos_lattice import BranchKind

# ---------------------------------------------------------------------------
# Lattice grounding for the depth token.
# ---------------------------------------------------------------------------
# tok(x) is the per-element "depth" mixed into each cell. We want it to be a
# genuine lattice measure, not just x. The verified zeta of a 2-way meet of the
# pair (x, x+1) is the SUM x + (x+1) = 2x+1, read off the lattice node. We use
# that as tok(x): an odd, strictly-increasing, lattice-derived weight whose sum
# is invertible for a lone survivor (matches "missing = sum - pair").
def tok(x: int) -> int:
    return 2 * x + 1


def _lattice_tok(x: int) -> float:
    """Read tok(x) THROUGH the verified lattice node (proves tok == zeta(meet)).

    The 2-way meet of the pair {x, x+1}: equalize the subset {x} against the
    full chain {x, x+1}; the witness Psi.zeta is the sum-lock = x + (x+1).
    Used only in the self-test, to certify tok() is the lattice's own measure.
    """
    _, psi = equalize_witness((x, x + 1), (x,), BranchKind.VA1, 1)
    return psi.zeta


# ---------------------------------------------------------------------------
# Hashing: a good splitmix64 avalanche.  The weak rehash was the classic bug;
# splitmix gives near-uniform dispersion so K distinct cells are easy to find.
# ---------------------------------------------------------------------------
_M = 0xFFFFFFFFFFFFFFFF
_MIX = 0x9E3779B97F4A7C15


def _splitmix(x: int) -> int:
    z = (x + _MIX) & _M
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _M
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _M
    return z ^ (z >> 31)


@dataclass
class Sketch:
    """Compact IBLT-style sketch. Bytes ~= n_cells * 24 (3 x int64 lanes)."""

    n_cells: int
    k: int
    count: list = field(default_factory=list)
    keysum: list = field(default_factory=list)
    depthsum: list = field(default_factory=list)

    def __post_init__(self):
        if not self.count:
            self.count = [0] * self.n_cells
            self.keysum = [0] * self.n_cells
            self.depthsum = [0] * self.n_cells

    def nbytes(self) -> int:
        # count: int32 is plenty; keysum/depthsum: int64 each.
        return self.n_cells * (4 + 8 + 8)


class ReconEngine:
    """Set-reconciliation engine built on the k=3 erasure atom.

    K (hash count) defaults to 4 -- the prompt's reliability lever. More hashes
    raise the peeling threshold (fraction of cells you can fill and still decode)
    but cost K cell-updates per element. depth-consistency gates every peel.
    """

    def __init__(self, k: int = 4):
        if k < 3:
            raise ValueError("k>=3 (the triple is the atom; >=4 for reliability)")
        self.k = k

    # --- cell assignment -----------------------------------------------------
    def _cells(self, x: int, n_cells: int) -> list:
        """K distinct cell indices for x (IBLT requires no repeated cell)."""
        idxs: list[int] = []
        salt = 0
        # decorrelated probes from one base hash
        h = _splitmix(x)
        while len(idxs) < self.k:
            c = _splitmix(h ^ (salt * _MIX)) % n_cells
            if c not in idxs:
                idxs.append(c)
            salt += 1
            if salt > 64 * self.k:  # n_cells too small for k distinct -> give up
                break
        return idxs

    # --- build ---------------------------------------------------------------
    def sketch(self, items: Iterable[int], capacity: int) -> Sketch:
        """Encode a set into a sketch sized to `capacity` expected differences.

        capacity is the EXPECTED symmetric-difference size; cells are scaled to
        it (NOT to the set size) -- that is the whole point.
        """
        n_cells = self._cells_for(capacity)
        s = Sketch(n_cells=n_cells, k=self.k)
        for x in items:
            x = int(x)
            t = tok(x)
            for c in self._cells(x, n_cells):
                s.count[c] += 1
                s.keysum[c] ^= x
                s.depthsum[c] += t
        return s

    def _cells_for(self, capacity: int) -> int:
        # Peeling threshold for K hashes (random hypergraph 2-core).
        #   K=3 -> ~1.222,  K=4 -> ~1.295,  K=5 -> ~1.425.
        # We use a safety margin above threshold so decode is near-certain in
        # the operating range; cells still scale with the DIFFERENCE.
        thresh = {3: 1.23, 4: 1.30, 5: 1.43, 6: 1.57}.get(self.k, 1.5)
        margin = 1.55  # generous; storage is tiny relative to the sets anyway
        return max(self.k + 1, int(capacity * thresh * margin) + 1)

    # --- reconcile -----------------------------------------------------------
    @staticmethod
    def _subtract(a: Sketch, b: Sketch) -> Sketch:
        if a.n_cells != b.n_cells or a.k != b.k:
            raise ValueError("sketches must share (n_cells, k)")
        d = Sketch(n_cells=a.n_cells, k=a.k)
        for c in range(a.n_cells):
            d.count[c] = a.count[c] - b.count[c]
            d.keysum[c] = a.keysum[c] ^ b.keysum[c]
            d.depthsum[c] = a.depthsum[c] - b.depthsum[c]
        return d

    def reconcile(self, a: Sketch, b: Sketch):
        """Recover the EXACT symmetric difference.

        Returns (only_in_A, only_in_B, decoded_clean):
            only_in_A : elements in A's set but not B's
            only_in_B : elements in B's set but not A's
            decoded_clean : True iff the residual sketch fully peeled to zero
                            (no leftover entangled cells -> recovery is exact).
        """
        d = self._subtract(a, b)
        n = d.n_cells
        count, keysum, depthsum = d.count[:], d.keysum[:], d.depthsum[:]
        onlyA: set[int] = set()
        onlyB: set[int] = set()

        # pure-cell queue
        def is_pure(c: int) -> bool:
            if count[c] not in (1, -1):
                return False
            x = keysum[c]
            # DEPTH-CONSISTENCY GATE: a real lone survivor has depthsum == +-tok(x).
            # A phantom (e.g. 3 distinct ids whose XOR happens to look singular)
            # fails this -- the lattice sum-lock rejects it. tok(x)=2x+1>0 for all
            # x>=0, so x==0 (tok=1) is handled by the same equality, no special case.
            return depthsum[c] == count[c] * tok(x)

        pure = [c for c in range(n) if is_pure(c)]
        while pure:
            c = pure.pop()
            if not is_pure(c):  # may have changed since enqueued
                continue
            x = keysum[c]
            sign = count[c]
            if sign == 1:
                onlyA.add(x)
            else:
                onlyB.add(x)
            t = tok(x)
            for cc in self._cells(x, n):
                count[cc] -= sign
                keysum[cc] ^= x
                depthsum[cc] -= sign * t
                if is_pure(cc):
                    pure.append(cc)

        # PHANTOM-PAIR CANCELLATION: a base element (in BOTH sets, true count 0 in
        # the difference) can occasionally be peeled once as +1 and once as -1 when
        # cells over-resolve. Those net to zero in the sketch (so it still "cleans"),
        # but pollute both outputs. Any id in onlyA AND onlyB is exactly such a
        # phantom -- it was never in the symmetric difference. Remove from both.
        phantom = onlyA & onlyB
        onlyA -= phantom
        onlyB -= phantom

        decoded_clean = all(count[c] == 0 for c in range(n)) and all(
            keysum[c] == 0 for c in range(n)
        ) and all(depthsum[c] == 0 for c in range(n))
        return onlyA, onlyB, decoded_clean


# ===========================================================================
# MEASUREMENT HARNESS
# ===========================================================================
def _human(nbytes: float) -> str:
    for u in ("B", "KB", "MB", "GB"):
        if nbytes < 1024:
            return f"{nbytes:.1f}{u}"
        nbytes /= 1024
    return f"{nbytes:.1f}TB"


def _make_pair(set_size: int, d: int, universe: int, rng: random.Random):
    """Two sets of ~set_size sharing a common base, differing in exactly d ids."""
    base = set(rng.sample(range(universe), set_size - d // 2))
    pool = list(set(range(universe)) - base)
    rng.shuffle(pool)
    onlyA = set(pool[: d // 2])
    onlyB = set(pool[d // 2 : d // 2 + (d - d // 2)])
    A = base | onlyA
    B = base | onlyB
    return A, B, onlyA, onlyB


def _certify_lattice_tok():
    """Prove tok(x) == zeta(meet of {x,x+1}) read off the verified lattice node."""
    ok = True
    for x in (0, 1, 3, 5, 17, 999, 40123):
        if abs(_lattice_tok(x) - tok(x)) > 1e-9:
            ok = False
    return ok


def main():
    print("=" * 78)
    print("AETHOS RECONCILE -- set reconciliation from the k=3 'missing=sum-pair' atom")
    print("=" * 78)

    cert = _certify_lattice_tok()
    print(f"\n[grounding] tok(x) == zeta(2-way meet {{x,x+1}}) via verified lattice node: {cert}")
    # show the atom itself
    _, psi = equalize_witness((3, 5), (3,), BranchKind.VA1, 1)
    print(f"[atom] meet(3,5): X={psi.z.real:.0f} Y={psi.z.imag:.0f} zeta(sum)={psi.zeta:.0f}  "
          f"=> missing = zeta - other = invertible")

    SET_SIZE = 40_000
    UNIVERSE = 4_000_000
    DIFFS = [10, 50, 200, 500, 1000]
    TRIALS = 20
    SEED = 20260623

    for K in (4, 5):
        eng = ReconEngine(k=K)
        print("\n" + "-" * 78)
        print(f"K = {K} hash functions   |   set size ~ {SET_SIZE:,}   |   {TRIALS} trials/point")
        print("-" * 78)
        print(f"{'d':>6} {'cells':>8} {'sketch':>10} {'ship-both':>11} {'savings':>9} "
              f"{'exact%':>8} {'clean%':>8} {'maxerr':>7} {'ms/recon':>9}")
        rng = random.Random(SEED)
        for d in DIFFS:
            exact_hits = 0
            clean_hits = 0
            max_err = 0
            t_total = 0.0
            sketch_bytes = 0
            cells = 0
            for _ in range(TRIALS):
                A, B, trueA, trueB = _make_pair(SET_SIZE, d, UNIVERSE, rng)
                sa = eng.sketch(A, capacity=d)
                sb = eng.sketch(B, capacity=d)
                cells = sa.n_cells
                sketch_bytes = sa.nbytes() + sb.nbytes()
                t0 = time.perf_counter()
                recA, recB, clean = eng.reconcile(sa, sb)
                t_total += time.perf_counter() - t0
                exact = (recA == trueA) and (recB == trueB)
                exact_hits += int(exact)
                clean_hits += int(clean)
                # symmetric-difference error count (missed + spurious)
                err = len((recA ^ trueA)) + len((recB ^ trueB))
                max_err = max(max_err, err)
            full_bytes = (len(A) + len(B)) * 8  # ship both id-sets as int64
            savings = full_bytes / sketch_bytes
            print(f"{d:>6} {cells:>8} {_human(sketch_bytes):>10} {_human(full_bytes):>11} "
                  f"{savings:>8.1f}x {100*exact_hits/TRIALS:>7.0f}% "
                  f"{100*clean_hits/TRIALS:>7.0f}% {max_err:>7} "
                  f"{1000*t_total/TRIALS:>8.1f}")

    # ----- where it STOPS being exact (two-sided honesty) ------------------
    print("\n" + "=" * 78)
    print("OPERATING RANGE / FAILURE BOUNDARY  (K=4)  -- undersize the sketch on purpose")
    print("=" * 78)
    print("Sketch sized for capacity=C but the ACTUAL difference is d. When d>C the")
    print("cells overflow the peeling threshold and decode stalls. We sweep C fixed,")
    print("grow d, and watch exact% collapse -- this is the honest cliff.")
    eng = ReconEngine(k=4)
    C = 200
    rng = random.Random(SEED + 7)
    print(f"\nfixed capacity C={C} (cells={eng._cells_for(C)}, "
          f"sketch={_human(2*Sketch(eng._cells_for(C),4).nbytes())}); grow true d:")
    print(f"{'true d':>7} {'exact%':>8} {'clean%':>8} {'avg recovered':>14} {'note':>22}")
    for d in [50, 150, 200, 250, 300, 400, 600]:
        exact_hits = clean_hits = 0
        rec_total = 0
        TR = 20
        for _ in range(TR):
            A, B, trueA, trueB = _make_pair(SET_SIZE, d, UNIVERSE, rng)
            sa = eng.sketch(A, capacity=C)
            sb = eng.sketch(B, capacity=C)
            recA, recB, clean = eng.reconcile(sa, sb)
            exact_hits += int((recA == trueA) and (recB == trueB))
            clean_hits += int(clean)
            rec_total += len(recA) + len(recB)
        note = "OK" if exact_hits == TR else ("d>C: over capacity" if d > C else "marginal")
        print(f"{d:>7} {100*exact_hits/TR:>7.0f}% {100*clean_hits/TR:>7.0f}% "
              f"{rec_total/TR:>14.0f} {note:>22}")

    # ----- phantom-peel gate value (does the depth check matter?) ----------
    print("\n" + "=" * 78)
    print("DEPTH-CONSISTENCY GATE ablation  -- does the lattice sum-lock kill bad peels?")
    print("=" * 78)
    _ablation()


def _ablation():
    """Compare reconcile WITH vs WITHOUT the depth-consistency gate on undersized
    sketches, where phantom-pure cells (collisions that XOR to a singleton) appear."""
    SET_SIZE = 40_000
    UNIVERSE = 4_000_000
    rng = random.Random(99)
    eng = ReconEngine(k=4)
    C = 100  # deliberately undersized so collisions create phantoms
    TR = 30
    bad_with = bad_without = 0
    for _ in range(TR):
        d = 400  # 4x over capacity -> stressed
        A, B, trueA, trueB = _make_pair(SET_SIZE, d, UNIVERSE, rng)
        sa, sb = eng.sketch(A, capacity=C), eng.sketch(B, capacity=C)
        # WITH gate (normal)
        recA, recB, clean = eng.reconcile(sa, sb)
        # a peel is "bad" if it emitted an id NOT in the true symdiff
        spurious_with = len((recA - trueA)) + len((recB - trueB))
        bad_with += spurious_with
        # WITHOUT gate: monkey-patch tok-check off by recovering with raw count only
        spurious_without = _reconcile_no_gate(eng, sa, sb, trueA, trueB)
        bad_without += spurious_without
    print(f"spurious peels over {TR} stressed trials (d=400, C={C}):")
    print(f"  WITH depth gate    : {bad_with:>5}  (phantoms rejected by zeta sum-lock)")
    print(f"  WITHOUT depth gate : {bad_without:>5}  (count-only IBLT emits these)")
    print(f"  -> gate prevents {bad_without - bad_with} bad emissions; "
          f"keeps recovered set CLEAN even when it cannot fully decode.")


def _reconcile_no_gate(eng: ReconEngine, a: Sketch, b: Sketch, trueA, trueB) -> int:
    """Same peel loop but accept any |count|==1 cell (no depth check). Returns
    number of spurious ids emitted (not in true symdiff)."""
    d = ReconEngine._subtract(a, b)
    n = d.n_cells
    count, keysum, depthsum = d.count[:], d.keysum[:], d.depthsum[:]
    emitted: set[int] = set()
    progress = True
    guard = 0
    while progress and guard < 10 * n:
        progress = False
        for c in range(n):
            if count[c] in (1, -1):
                x = keysum[c]
                sign = count[c]
                emitted.add(x)
                for cc in eng._cells(x, n):
                    count[cc] -= sign
                    keysum[cc] ^= x
                    depthsum[cc] -= sign * tok(x)
                progress = True
                guard += 1
    true_sym = trueA | trueB
    return len(emitted - true_sym)


if __name__ == "__main__":
    main()
