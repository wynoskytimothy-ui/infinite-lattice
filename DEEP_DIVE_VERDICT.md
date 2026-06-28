# Deep-dive stress-test verdict (second, adversarial)

*8-agent build-and-break dive, 2026-06-25. Every number re-run against live code; red-teamed; independently re-verified. Honest, no hype.*

## Bottom line
**Elegant and correct, but REDUCIBLE.** AETHOS invents no new asymptotics, no error-correction distance, and
no capability outside `{unimodular prefix-sum bijection} × {min-plus semiring} × {content hashing}` — all
off-the-shelf. It is a good ABSTRACTION, not a new theorem.

## Survived every reduction (real, narrow)
- Meet = unimodular integer bijection `det(M)=−1` → distinct triples cannot collide. 0 collisions / 9.37M
  adversarial triples + exhaustive [0,200]³; 0 round-trip failures to 1e36. A genuine perfect-hash relabel.
- `(sum,min)` iterated = exact Floyd-Warshall (0 disagreements / 3,540 pairs).
- Single-erasure recovery `missing = ζ − sum(rest)` with **no shared dictionary** — the one property a bare
  Minisketch syndrome lacks. 50000/50000 exact.

## Broke / reduced (the bulk)
- "Edge-free fabric": FALSE — every address inverts to `(u,v,weight)` (286/286); the index IS the edge list
  under a det=−1 relabel, 3 ints/edge. Plain dict + Floyd-Warshall is byte-identical AND faster.
- "Tamper detection": does not exist — 0/50000 single-coordinate corruptions caught; no code distance.
- Merkle-of-meets erasure = plain `parent=sum(children)` single-parity; the meet adds nothing to recovery.
- k≥4 flat co-location: dead (0/300). The triple is the atom.
- Compact address: only pure mixed-radix Horner hits ≤1.1× entropy (zero AETHOS); every AETHOS-flavored
  address blows up 1.5×–7.9×H or is lossy. (Distinctness+decode DID flip to REAL via per-edge interior read;
  only the meet-FOLD was lossy.)
- Depth 486: a hard-coded list, not algebra. A lazy counter → height-20 / 1M-node tower, 0 provenance
  collisions. (Worse than thought: the original fixed pool can't build even level 1 with a wide base.)
- "32 chambers": not a closed group — a transversal half of D4×Z2 = SmallGroup(16,11). One magnitude |ζ|,
  not 32 free dims. The "32-orbit sums to 0" is FALSE on the Y axis (flip_y is never used; Y_sum=8·Y).

## The math object (pinned exactly)
The meet is a unimodular prefix-sum bijection (`G=[[0,1,1],[0,1,0],[1,1,1]]`, det=−1, integer inverse
a=Z−X, p=Y, q=X−Y) composed with the order-statistic sort (the only nonlinear content). Symmetry group =
**D4 × Z2 = SmallGroup(16,11)** (hyperoctahedral B2 × depth-sign), invariant ring `{|ζ|, X²+Y²}`. = min-plus
tropical algebra + a Coxeter sign group.

## The one defensible niche
**Dictionary-free d=1 erasure / set reconciliation:** names a missing member with no shared universe at a
constant 8 bytes regardless of set size (500× vs shipping a 1000-element set). It is Minisketch's simplest
row (d=1), but the no-dictionary property is genuinely Minisketch-lacking. Niche win iff real workloads have
symmetric difference ≤ 1; for symdiff > 1 it collapses to textbook PinSketch/Minisketch (2d power sums +
Berlekamp-Massey + rooting over a shared universe), and over-capacity decoding fails SILENTLY.

## Real bug found — FIXED (commit follows)
`aethos_address_store.AddressStore`: the `ComplexPlane3D.zeta` float path inherits the IEEE-754 2^53 wall.
In the band ~2^52.5 .. 2^53 `get()` returned WRONG values (incl. negatives: 7→6, 2→−1) with NO exception;
only at ≥2^53 did it hard-fail.
**FIX (verified by `_dd_bugfix_verify.py`):** (1) `TripleNode.value` now decodes from the EXACT integer
anchors `a+p+q`, not the float `zeta` readout — so recovery is exact regardless of the float wall;
(2) `make_triple` raises a clear `ValueError` when `a+p+q >= 2^53`, so the silent-corruption zone is
unreachable (it now errors explicitly instead of returning wrong values). Normal range unchanged; demo passes.

## What this does NOT change
The earlier MEASURED engineering wins are separate and still real: native MARCO SPLADE-on-lattice at
286.9 B/doc + MRR ~0.39 + ~88–127 ms serve (see MEASUREMENTS.md), and the no-SPLADE lattice + supervised
bridges (scifact 0.702→0.7375, no GPU). Those are good engineering on a clean substrate — they never
depended on a "new math" breakthrough. The lattice's value is a tidy invertible primitive + those measured
wins, not a new theorem.

Evidence (all in repo): `_dd_break_meet*.py`, `_dd_godel_address*.py`, `_dd_colocation_join.py`,
`_dd_join_noedge.py`, `_dd_beyond_k3*.py`, `_dd_new_caps.py`, `_dd_setrec*.py`, `_dd_math*.py`,
`_dd_redteam.py`, `_verify_now.py`, `_verify_join.py` (+ `*_out.txt`), summarized in `_dd_summary.txt`.
