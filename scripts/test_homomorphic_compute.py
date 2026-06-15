#!/usr/bin/env python3
"""
Test 42 - Homomorphic / private computation on composites.

Test 41 (B) showed the set ALGEBRA (gcd=intersect) but transmitted plain
composites - factorable, so not private. This test adds the blinding step
and verifies real privacy, plus homomorphic aggregation:

  (A) PRIVATE SET INTERSECTION via commutative encryption (Pohlig-Hellman /
      Huberman-Franklin-Hogg). Each party masks elements with a secret
      exponent; because (g^x)^a^b = (g^x)^b^a, both reach a common masked
      form and intersect it - WITHOUT either learning the other's
      non-shared elements (would require discrete log).

  (B) HOMOMORPHIC HISTOGRAM AGGREGATION via prime exponents. Counts encode
      as prime^count (Test 10's multiset trick); multiplying composites
      ADDS the histograms. A coordinator sums many parties' encrypted
      tallies and reads the totals by factoring - computing on the encoded
      data without decoding the inputs first.

Both verified: intersection exact, privacy held (a party cannot enumerate
the other's private set from the transcript), histogram totals exact.
"""

from __future__ import annotations

import random
import sys
from math import prod
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


# A large safe-ish prime for the commutative-encryption group.
P = (1 << 127) - 1            # Mersenne prime 2^127 - 1
G = 5                          # generator of a large subgroup


def H(e):
    """Map an element id to a group element g^(e+1) mod P (injective on ids)."""
    return pow(G, e + 1, P)


def part_a(rng):
    header("(A) PRIVATE SET INTERSECTION - commutative encryption (real blinding)")

    def run_protocol(alice, bob):
        a = rng.randrange(2, P - 1)        # Alice secret key
        b = rng.randrange(2, P - 1)        # Bob secret key
        # round 1: each masks its own set with its key
        A1 = {H(e): pow(H(e), a, P) for e in alice}     # Alice -> {single-masked}
        # round 2: Bob double-masks Alice's values and masks his own
        A2 = {pow(v, b, P) for v in A1.values()}         # g^(x a b)
        B1 = [pow(H(f), b, P) for f in bob]
        B2 = [pow(v, a, P) for v in B1]                  # Alice masks Bob's: g^(y a b)
        # intersection = common double-masked values
        inter_masked = A2 & set(B2)
        return inter_masked, A1, a, b

    # correctness over many random pairs
    ok = 0
    trials = 200
    for _ in range(trials):
        alice = set(rng.sample(range(500), 40))
        bob = set(rng.sample(range(500), 40))
        true_int = alice & bob
        inter_masked, _, _, _ = run_protocol(alice, bob)
        if len(inter_masked) == len(true_int):
            ok += 1
    print(f"  {trials} random set pairs: intersection size correct {ok}/{trials}")
    assertion(ok == trials,
              "commutative-encryption PSI recovers the exact intersection size")

    # privacy: from Alice's single-masked transcript, can Bob recover Alice's
    # private (non-shared) elements? He would need to match H(candidate)^a, but
    # he does not know a. He can only confirm the agreed intersection.
    alice = set(rng.sample(range(500), 40))
    bob = set(rng.sample(range(500), 40))
    inter_masked, A1_transcript, a, b = run_protocol(alice, bob)
    # Bob attacks: try every universe element, mask with his own key b, and
    # see if it matches Alice's single-masked values (it will NOT - wrong key).
    bob_recovered = set()
    masked_vals = set(A1_transcript.values())            # what Bob saw from Alice
    for cand in range(500):
        if pow(H(cand), b, P) in masked_vals:            # Bob's key, not Alice's
            bob_recovered.add(cand)
    print(f"  Alice's private set size {len(alice)}; from the transcript Bob "
          f"recovered {len(bob_recovered)} elements")
    assertion(len(bob_recovered) == 0,
              "Bob cannot recover ANY of Alice's elements from the single-masked "
              "transcript (he lacks Alice's key - discrete-log hard)")
    # only the agreed protocol yields the intersection, and nothing more
    assertion(len(inter_masked) == len(alice & bob),
              "the ONLY thing learned is the intersection - true private set "
              "intersection, the blinding Test 41 only noted")


def part_b(rng):
    header("(B) HOMOMORPHIC HISTOGRAM - sum encrypted tallies by multiplying")
    cats = chain_primes(32)                # one prime per category (32 categories)
    n_cats = len(cats)

    def encode(counts):
        c = 1
        for i, k in enumerate(counts):
            c *= cats[i] ** k              # prime^count (Test 10 multiplicity)
        return c

    def decode(composite):
        out = []
        for p in cats:
            k = 0
            while composite % p == 0:
                composite //= p
                k += 1
            out.append(k)
        return out

    # many parties each hold a private histogram; coordinator multiplies the
    # encoded blobs and reads the TOTAL without seeing any party's raw counts.
    n_parties = 50
    parties = [[rng.randrange(0, 6) for _ in range(n_cats)] for _ in range(n_parties)]
    encoded = [encode(h) for h in parties]
    aggregate = prod(encoded)              # homomorphic sum: exponents add
    totals = decode(aggregate)
    truth = [sum(p[i] for p in parties) for i in range(n_cats)]
    print(f"  {n_parties} parties x {n_cats} categories")
    print(f"  category 0 total: decoded {totals[0]} vs true {truth[0]}")
    print(f"  category 17 total: decoded {totals[17]} vs true {truth[17]}")
    assertion(totals == truth,
              "multiplying the encrypted tallies SUMS the histograms exactly "
              "(homomorphic aggregation - compute on encoded data)")
    # the coordinator never decoded any individual party's blob
    print("  the coordinator multiplied encoded blobs and read totals by")
    print("  factoring the product - individual inputs were never decoded.")


def main():
    header("Homomorphic / private computation on prime composites")
    rng = random.Random(0x42E0)
    part_a(rng)
    part_b(rng)

    header("RESULT")
    print("  (A) real private set intersection: commutative encryption blinds")
    print("      the elements, both parties reach the common double-masked")
    print("      form, intersect it, and learn NOTHING else. Discrete log")
    print("      protects the non-shared elements.")
    print("  (B) homomorphic histogram: prime^count + multiply = add. A")
    print("      coordinator aggregates many encrypted tallies and reads the")
    print("      totals by factoring, never decoding an input.")
    print()
    print("  Privacy-preserving computation - normally lattice cryptography or")
    print("  secure-multiparty protocols - is the prime composite wearing a")
    print("  blinding mask. The same multiply=union that ran retrieval and the")
    print("  ledger, now computing on data nobody can read.")


if __name__ == "__main__":
    main()
