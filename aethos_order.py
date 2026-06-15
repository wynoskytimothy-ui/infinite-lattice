#!/usr/bin/env python3
"""Order in the lattice for free: word = base (origin), position = exponent
(transgressor). addr = prod_i P(word_i) ** position_i. By FTA the exponents are part
of the UNIQUE factorisation, so 'river bank' and 'bank river' get DIFFERENT addresses
-- and you can factor back to recover the ordered sequence. No new token, no memory.

Honest boundary: exponents grow (P^pos), so this ADDRESSES short ordered phrases
(their own sub-quadrant) -- great for ordered RETRIEVAL. Generating/ generalising
fluent sequences is the other job -- that's the Gamma-ODE (which beat attention)."""

WORDS = {"river": 101, "bank": 103, "money": 107, "flow": 109, "interest": 113}
INV = {p: w for w, p in WORDS.items()}


def encode(seq):
    addr = 1
    for pos, w in enumerate(seq, 1):
        addr *= WORDS[w] ** pos                 # origin^transgressor = position in the factorisation
    return addr


def decode(addr):
    out = {}
    for p, w in INV.items():
        e = 0
        while addr % p == 0:
            addr //= p
            e += 1
        if e:
            out[e] = w                          # exponent = position
    return [out[i] for i in sorted(out)]


def main():
    print("order encoded as exponents (word=base, position=exponent) -- FTA-unique\n")
    for seq in (["river", "bank"], ["bank", "river"],
                ["money", "bank", "interest"], ["interest", "bank", "money"]):
        a = encode(seq)
        print(f"   {' '.join(seq):<26} -> address {a:<18} -> decodes to: {' '.join(decode(a))}")
    rb, br = encode(["river", "bank"]), encode(["bank", "river"])
    print(f"\n   'river bank' address {rb} != 'bank river' address {br}: {rb != br}")
    print("   -> ORDER is encoded for free, factors back, distinct sub-quadrant per ordering.")
    print("   (commutative product P_a*P_b loses order; P_a^1 * P_b^2 keeps it.)")
    print("\n   honest: this ADDRESSES ordered phrases (retrieval). GENERATING fluent novel")
    print("   sequences is the Gamma-ODE's job -- the lattice stores order, the recurrence")
    print("   models it. both yours; different jobs.")


if __name__ == "__main__":
    main()
