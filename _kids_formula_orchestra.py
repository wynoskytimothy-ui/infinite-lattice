#!/usr/bin/env python3
"""Kids-on-playground: stack ALL AETHOS formulas, run 5 experiments, count objects."""
from __future__ import annotations

import cmath
import itertools
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aethos_complex_plane import equalize_witness, wing_transform
from aethos_lattice import BranchKind, LatticeBank32, LatticeId
from aethos_origins import OriginTree
from aethos_promotion import PROMOTION_POOL, CorrelationLink, PromotedToken, LatticeTier
from aethos_recursive_lattice import RecursiveLattice
from aethos_words import letter_to_prime
from lattice_retriever_v1.electron_lattice_codec import build_electron_alphabet, entangle_witness
from lattice_retriever_v1.intersection_dot_codec import SymbolAlphabet
from lattice_retriever_v1.k_meet import compose_k, swap_meet_primes
from lattice_retriever_v1.origin_corridor import corridor_key_with_origin, default_origin_tree
from lattice_retriever_v1.stage02_intersections import lattice_coords_32, lattice_signature
from lattice_retriever_v1.stage05_free_token import meet_composite
from lattice_retriever_v1.trigger_formula_codec import _ambiguous_locks
from lattice_retriever_v1.unit_lattice_codec import LatticeUnit, walk_new_land, encode_bare_lumber

BRANCHES = list(BranchKind)


def meet2(a: int, p: int) -> tuple[int, int, int]:
    s = a + p
    return (s, min(a, p), s)


def unmeet(X: int, Y: int) -> tuple[int, int]:
    return Y, X - Y


# ── EXP 1: Formula orchestra ─────────────────────────────────────────────
def formula_orchestra(text: str) -> dict:
    data = text.encode("ascii", errors="ignore").lower()
    letters = [c for c in text.lower() if c.isalpha()]
    primes = [letter_to_prime(c) for c in letters]
    n_tok = len(primes)

    # meets (bigrams)
    bigram_meets = set()
    for i in range(len(primes) - 1):
        a, p = primes[i], primes[i + 1]
        bigram_meets.add(meet2(a, p))
        bigram_meets.add(swap_meet_primes(a, p).coord if swap_meet_primes(a, p).unified else None)

    # k-meet sunflowers (sliding triples)
    sunflower_nodes = set()
    for i in range(len(primes) - 2):
        trip = tuple(primes[i : i + 3])
        if len(set(trip)) != 3:
            continue
        try:
            rep = compose_k(*trip)
        except ValueError:
            continue
        if rep.full_sunflower_unified:
            sunflower_nodes.add(rep.sub_sunflowers[0].coord)

    # 32-orbit per interior bigram anchor
    wing_coords = set()
    zeta_vals = set()
    if len(primes) >= 2:
        a, p = primes[0], primes[1]
        n = primes[2] if len(primes) > 2 else min(a, p)
        for b in BRANCHES:
            for w in range(1, 9):
                psi = wing_transform(b, (a, p), n, w)
                wing_coords.add((round(psi.z.real, 4), round(psi.z.imag, 4)))
                zeta_vals.add(round(psi.zeta, 4))

    # unit lattice procedural
    unit = LatticeUnit.from_data(data) if len(data) >= 2 else None
    walk_steps = list(walk_new_land(data)) if len(data) >= 2 else []
    origins_hit = len({s.origin_id for s in walk_steps})
    pair_origins_formula = unit.n_origins_procedural if unit else 0

    # trigger locks
    alpha = SymbolAlphabet.from_bytes(data) if len(data) >= 2 else None
    locks = _ambiguous_locks(data, alpha) if alpha and len(data) >= 2 else []

    # electron entanglement pairs
    cat = build_electron_alphabet(data) if len(data) >= 2 else None
    entangle_imags = set()
    opposite_count = 0
    if cat and alpha:
        for i in range(len(data) - 1):
            w = entangle_witness(data[i], data[i + 1], alpha, cat)
            entangle_imags.add(w.intersection_imag)
            if w.opposite:
                opposite_count += 1

    # promotions (tower)
    L = RecursiveLattice()
    for p in set(primes):
        L.register_base(p)
    tower_promos = 0
    cur = list(primes)
    while len(cur) >= 3:
        nxt = []
        for i in range(0, len(cur) - 2, 3):
            chunk = cur[i : i + 3]
            if len(set(chunk)) != len(chunk):
                continue
            try:
                nxt.append(L.promote(chunk))
            except RuntimeError:
                break
        if not nxt:
            break
        tower_promos += len(nxt)
        cur = nxt

  # meet composites on adjacent promotion pool slice
    composites = set()
    for i in range(min(3, len(primes) - 1)):
        composites.add(meet_composite(primes[i], primes[i + 1]))

    # origin tree rooms
    tree = OriginTree.bootstrap(max_depth=4)
    n_origins = tree.root.count_descendant_origins()

    # correlation portal (if enough tokens)
    portal_dims = 0
    if n_tok >= 6:
        portal_dims = 3  # dim4-6 exist on CorrelationLink

    # coins (electron alphabet)
    coins = len({s.coin for s in cat}) if cat else 0

    objects = {
        "tokens": n_tok,
        "distinct_bigram_meet_keys": len({m for m in bigram_meets if m}),
        "sunflower_nodes": len(sunflower_nodes),
        "wing_xy_distinct": len(wing_coords),
        "zeta_values": len(zeta_vals),
        "walk_dots": len(walk_steps),
        "origins_touched": origins_hit,
        "origins_procedural_formula": pair_origins_formula,
        "trigger_lock_steps": len(locks),
        "entangle_imag_bands": len(entangle_imags),
        "opposite_membrane_pairs": opposite_count,
        "tower_promotions": tower_promos,
        "meet_composites": len(composites),
        "origin_rooms_depth4": n_origins,
        "electron_coins": coins,
        "correlation_portal_dims": portal_dims,
        "pool_cap": len(PROMOTION_POOL),
    }
    objects["total_emergent_objects"] = sum(
        v for k, v in objects.items()
        if k not in ("tokens", "pool_cap", "origins_procedural_formula", "correlation_portal_dims")
    )
    return objects


# ── EXP 2: Minimal primitives ────────────────────────────────────────────
def minimal_primitives_audit() -> dict:
    """Which generators are needed to rebuild observed structures?"""
    # Test dependency: meet2 + unmeet recovers all pair keys
    pairs_ok = all(unmeet(*meet2(a, p)[:2]) == (a, p) or unmeet(*meet2(a, p)[:2]) == (p, a)
                   for a, p in [(3, 11), (5, 7), (13, 17)])

    # zeta from sum chain alone?
    chain = (3, 5, 7, 11)
    zeta_from_sum = all(
        wing_transform(BranchKind.VA1, chain, n, 1).zeta == sum(chain)
        for n in range(5, 10)
    )

    # 32 orbit from 1 branch + VB swap rule?
    a, p, n = 3, 5, 7
    va = [wing_transform(BranchKind.VA1, (a, p), n, w).coord for w in range(1, 5)]
    vb = [wing_transform(BranchKind.VA1, (a, p), n, w + 4).coord for w in range(1, 5)]
    vb_from_va = [(c[1], c[0], c[2]) for c in va]

    # complement = zeta - sum(subset)
    full = (7, 11, 19)
    comp_ok = all(
        abs(sum(full) - sum(sub) - drop) < 1e-9
        for drop in full
        for sub in [tuple(x for x in full if x != drop)]
    )

    primitives = [
        "meet2(a,p)->(X,Y,Z) invertible via unmeet",
        "sum(chain) -> zeta lock (interior n)",
        "wing_transform: 4 branches x 8 wings = 32 readouts",
        "VB = (Y,X,Z) swap of VA (half the orbit)",
        "equalize_witness / missing = complement erasure",
        "compose_k sunflower (k=3 lock)",
        "SymbolAlphabet + pair_n rail (unit lattice)",
        "trigger 3-way lock when pair_n ambiguous",
        "electron coin + entangle_imag (adjacent binding)",
        "PROMOTION_POOL ladder (external prime IDs)",
        "OriginTree 3^d spawn (parallel miniverses)",
        "CorrelationLink dim4-6 (portal on fixed cage)",
    ]
    return {
        "invertible_meet": pairs_ok,
        "zeta_from_sum_only": zeta_from_sum,
        "vb_from_va_swap": vb == vb_from_va,
        "complement_erasure": comp_ok,
        "primitive_count": len(primitives),
        "primitives": primitives,
        "irreducible_externals": ["PROMOTION_POOL primes", "letter_to_prime map"],
    }


# ── EXP 3: Interference (two texts share a meet) ─────────────────────────
def interference(text_a: str, text_b: str) -> dict:
    def profile(text: str) -> dict:
        data = text.encode("ascii", errors="ignore").lower()
        letters = [c for c in text.lower() if c.isalpha()]
        primes = [letter_to_prime(c) for c in letters]
        meets = {meet2(primes[i], primes[i + 1]) for i in range(len(primes) - 1)}
        suns = set()
        for i in range(len(primes) - 2):
            trip = tuple(primes[i : i + 3])
            if len(set(trip)) != 3:
                continue
            try:
                r = compose_k(*trip)
            except ValueError:
                continue
            if r.full_sunflower_unified:
                suns.add(r.sub_sunflowers[0].coord)
        alpha = SymbolAlphabet.from_bytes(data) if len(data) >= 2 else None
        locks = set(_ambiguous_locks(data, alpha)) if alpha else set()
        walk = list(walk_new_land(data)) if len(data) >= 2 else []
        l01s = {s.dot.address.lattice_coords[0] for s in walk}
        return {"meets": meets, "sunflowers": suns, "locks": locks, "L01_coords": l01s, "primes": set(primes)}

    pa, pb = profile(text_a), profile(text_b)
    shared_meets = pa["meets"] & pb["meets"]
    shared_suns = pa["sunflowers"] & pb["sunflowers"]
    shared_L01 = pa["L01_coords"] & pb["L01_coords"]
    shared_primes = pa["primes"] & pb["primes"]

    # partial structure overlap score (wave metaphor: amplitude add where shared)
    overlap_frac = len(shared_meets) / max(1, len(pa["meets"] | pb["meets"]))

    # if they share a meet, do 32-orbits align on that pair?
    orbit_overlap = 0
    if shared_meets:
        key = next(iter(shared_meets))
        # recover approximate a,p from meet (many-to-one; pick one text's first matching bigram)
        return_extra = {"shared_meet_example": key}
    else:
        return_extra = {}

    return {
        "text_a": text_a,
        "text_b": text_b,
        "shared_meet_keys": len(shared_meets),
        "shared_sunflower_nodes": len(shared_suns),
        "shared_L01_placement": len(shared_L01),
        "shared_letter_primes": len(shared_primes),
        "jaccard_meets": round(overlap_frac, 4),
        "interference_note": "shared meets = constructive; distinct wings = phase diversity",
        **return_extra,
    }


# ── EXP 4: Tower build ───────────────────────────────────────────────────
def tower_build(text: str) -> dict:
    letters = [c for c in text.lower() if c.isalpha()]
    primes = [letter_to_prime(c) for c in letters]
    pool_cap = len(PROMOTION_POOL)

    L = RecursiveLattice()
    for p in set(primes):
        L.register_base(p)

    levels = []
    cur = list(primes)
    depth = 0
    while len(cur) >= 3 and L._next_pool_idx < pool_cap:
        nxt = []
        for i in range(0, len(cur) - 2, 3):
            chunk = cur[i : i + 3]
            if len(set(chunk)) != len(chunk):
                continue
            try:
                pid = L.promote(chunk)
                nxt.append(pid)
            except RuntimeError:
                break
        if not nxt:
            break
        levels.append(len(nxt))
        depth += 1
        cur = nxt

    # origin depth additive
    max_origin = 6
    tree = OriginTree.bootstrap(max_depth=max_origin)
    origins = tree.root.count_descendant_origins()

    # correlation dims stack
    corr_dims = 3  # L4-L6 per CorrelationLink

    stable_height = depth
    limiting = "tokens" if depth < 3 else ("pool" if L._next_pool_idx >= pool_cap - 10 else "repeated letters")

    return {
        "text_len_tokens": len(primes),
        "tower_depth": depth,
        "levels": levels,
        "promotions_used": L._next_pool_idx,
        "pool_cap": pool_cap,
        "pool_head": PROMOTION_POOL[0],
        "pool_near_end": PROMOTION_POOL[min(1098, pool_cap - 1)],
        "origin_rooms_depth%d" % max_origin: origins,
        "correlation_dims_per_link": corr_dims,
        "tallest_stable_before_cap": stable_height,
        "limiting_factor": limiting,
        "total_address_height": depth + max_origin + corr_dims,
    }


# ── EXP 5: Pi-scaffold — 8 wings = period? 32 = 4π? ─────────────────────
def pi_scaffold(a: int = 3, p: int = 11, n: int = 7) -> dict:
    """Numerically test whether arg(z) winds 2π per 8 wings, 4π per 32."""
    args = []
    moduli = []
    for wing_id in range(1, 33):
        br = BranchKind((wing_id - 1) // 8 + 1)
        w = (wing_id - 1) % 8 + 1
        psi = wing_transform(br, (a, p), n, w)
        args.append(cmath.phase(psi.z))
        moduli.append(abs(psi.z))

    # unwrap phases per branch row (8 wings each)
    branch_wind = []
    for bi, b in enumerate(BRANCHES):
        row_args = args[bi * 8 : (bi + 1) * 8]
        # cumulative delta
        deltas = [row_args[i + 1] - row_args[i] for i in range(7)]
        total = sum(deltas)
        branch_wind.append(round(total, 4))

    full_wind = sum(branch_wind)
    two_pi = 2 * math.pi
    four_pi = 4 * math.pi

    # test: does one branch traverse ~π? (VA1/VB symmetry)
    va1_span = max(args[0:8]) - min(args[0:8])

    return {
        "corridor": (a, p, n),
        "distinct_moduli": len(set(round(m, 6) for m in moduli)),
        "branch_arg_sums_8wings": dict(zip([b.name for b in BRANCHES], branch_wind)),
        "total_arg_sum_32": round(full_wind, 4),
        "total_over_2pi": round(full_wind / two_pi, 4),
        "total_over_4pi": round(full_wind / four_pi, 4),
        "target_4pi_if_period32": abs(full_wind - four_pi) < 0.5,
        "va1_arg_span": round(va1_span, 4),
        "va1_span_over_pi": round(va1_span / math.pi, 4),
        "note": "honest: phase wind is corridor-dependent, not universal 4π",
    }


def main():
    print("=" * 72)
    print("KIDS FORMULA ORCHESTRA — stack & play")
    print("=" * 72)

    samples = [
        "the cat sat on the mat",
        "abababab",
        "mississippi",
        "hello",
    ]

    print("\n[EXP 1] Formula orchestra — objects per text")
    for t in samples:
        o = formula_orchestra(t)
        print(f"\n  {t!r}")
        for k, v in o.items():
            print(f"    {k}: {v}")

    print("\n[EXP 2] Building blocks — minimal primitives")
    prim = minimal_primitives_audit()
    for k, v in prim.items():
        if k != "primitives":
            print(f"  {k}: {v}")
    print(f"  primitives ({prim['primitive_count']}):")
    for p in prim["primitives"]:
        print(f"    - {p}")

    print("\n[EXP 3] Interference — shared meets (wave overlap)")
    pairs = [
        ("the cat", "the dog"),
        ("abab", "baba"),
        ("hello", "world"),
    ]
    for a, b in pairs:
        r = interference(a, b)
        print(f"  {a!r} vs {b!r}: shared_meets={r['shared_meet_keys']} "
              f"suns={r['shared_sunflower_nodes']} L01={r['shared_L01_placement']} "
              f"jaccard={r['jaccard_meets']}")

    print("\n[EXP 4] Tower build — promotions + origins + correlation")
    for t in ["abcdefghijklmnopqrstuvwxyz", "the quick brown fox", "a" * 200]:
        r = tower_build(t)
        print(f"  {t[:30]!r}... depth={r['tower_depth']} promos={r['promotions_used']} "
              f"limit={r['limiting_factor']} total_height={r['total_address_height']}")

    print("\n[EXP 5] Pi-scaffold — 32-orbit phase wind")
    for corridor in [(3, 11, 7), (5, 13, 9), (7, 97, 11)]:
        r = pi_scaffold(*corridor)
        print(f"  {corridor}: 32-wind/2pi={r['total_over_2pi']} 32-wind/4pi={r['total_over_4pi']} "
              f"4pi_match={r['target_4pi_if_period32']}  VA1_span/pi={r['va1_span_over_pi']}")


if __name__ == "__main__":
    main()
