"""
Near-location word encoding: "tab" and "bat" share the same letter-prime set
(same canonical lattice address) but differ by tiny permutation offset (order).

Same multiset -> same sorted anchors -> nearly the same dot.
Different order -> different side shift -> formula tells them apart.
"""

from __future__ import annotations

import math
import unicodedata
import zlib
from dataclasses import dataclass

from aethos_codec import (
    Dot,
    IntersectionWitness,
    coordinate_from_witness,
    decode_bytes,
    verify_dot,
)
from aethos_lattice import LatticeId
from aethos_permutation import (
    PERM_EPSILON,
    apply_order_offset,
    decode_order_from_dot,
    decode_sequence_from_dot,
    explain_order,
)
from aethos_sequences import SequenceKind, make_chain

# One odd prime per letter a..z (distinct anchors; 2 is not used)
LETTER_PRIMES: tuple[int, ...] = make_chain(SequenceKind.PRIMES, 26)
# Deterministic primes for non-Latin letters (first i18n step).
_UNICODE_LETTER_PRIMES: tuple[int, ...] = make_chain(SequenceKind.PRIMES, 512)[100:228]


def letter_to_prime(c: str) -> int:
    """Map one letter to an odd prime; NFKC + Latin base fold for accented chars."""
    c = unicodedata.normalize("NFKC", c)
    if len(c) != 1:
        raise ValueError(f"single character required: {c!r}")
    cl = c.lower()
    if "a" <= cl <= "z":
        return LETTER_PRIMES[ord(cl) - ord("a")]
    if cl.isalpha():
        for ch in unicodedata.normalize("NFD", cl):
            if "a" <= ch <= "z":
                return LETTER_PRIMES[ord(ch) - ord("a")]
        idx = zlib.crc32(cl.encode("utf-8")) % len(_UNICODE_LETTER_PRIMES)
        return _UNICODE_LETTER_PRIMES[idx]
    raise ValueError(f"not a letter: {c!r}")


def prime_to_letter(p: int) -> str:
    idx = LETTER_PRIMES.index(p)
    return chr(ord("a") + idx)


def word_to_order(word: str) -> tuple[int, ...]:
    """Application order of letter-primes as read left-to-right."""
    return tuple(letter_to_prime(c) for c in word.lower() if c.isalpha())


def order_to_word(order: tuple[int, ...]) -> str:
    return "".join(prime_to_letter(p) for p in order)


def word_sorted_chain(word: str) -> tuple[int, ...]:
    return tuple(sorted(set(word_to_order(word))))


def word_order_candidates(word: str) -> tuple[tuple[int, ...], ...]:
    """All distinct letter-prime orderings for a word (handles repeated letters)."""
    from itertools import permutations

    order = word_to_order(word)
    if len(set(order)) == len(order):
        return (order,)
    return tuple(dict.fromkeys(permutations(order)))


@dataclass(frozen=True)
class SharedSite:
    """Fixed lattice site — same for all anagrams of the same letters."""

    chain: tuple[int, ...]
    n: int = 7
    lattice_id: int = int(LatticeId.L01)
    origin_path: str = "O0"
    dim_slot: int | None = None


def encode_word_at_site(word: str, site: SharedSite | None = None) -> Dot:
    """
    Place word near canonical site; order encoded in side-offset.
    tab and bat share site.chain, differ in prime_order -> nearby dots.
    """
    order = word_to_order(word)
    chain = word_sorted_chain(word)
    site = site or SharedSite(chain=chain)

    if set(order) != set(chain):
        raise ValueError("word letters must match site chain")

    w = IntersectionWitness(
        chain=chain,
        n=site.n,
        lattice_id=site.lattice_id,
        origin_path=site.origin_path,
        dim_slot=site.dim_slot,
        payload=zlib.compress(word.encode("utf-8"), level=9),
        prime_order=order,
    )
    x, y, z = coordinate_from_witness(w)
    return Dot(x=x, y=y, z=z, witness=w)


def canonical_base(dot: Dot) -> tuple[float, float, float]:
    w = dot.witness
    return coordinate_from_witness(
        IntersectionWitness(
            chain=w.chain,
            n=w.n,
            lattice_id=w.lattice_id,
            origin_path=w.origin_path,
            dim_slot=w.dim_slot,
            payload=w.payload,
            prime_order=(),
        ),
        with_order_offset=False,
    )


def distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def decode_word(dot: Dot) -> str:
    if not verify_dot(dot):
        raise ValueError("dot does not match witness")
    # Full text from payload; order from side-channel confirms path
    text = decode_bytes(dot).decode("utf-8")
    w = dot.witness
    chain = w.sorted_chain()
    if w.prime_order:
        base = canonical_base(dot)
        if len(set(w.prime_order)) == len(w.prime_order) and len(chain) >= 2:
            order = decode_order_from_dot(base, dot.coord, chain)
        else:
            order = decode_sequence_from_dot(base, dot.coord, word_order_candidates(text))
        path_word = order_to_word(order)
        if sorted(path_word) != sorted(text):
            raise ValueError("order channel disagrees with payload")
    return text


def compare_words(*words: str) -> None:
    if len(words) < 2:
        words = ("tab", "bat")

    chains = {w: word_sorted_chain(w) for w in words}
    print("=" * 60)
    print("NEAR-LOCATION WORDS — same site, different order offset")
    print("=" * 60)

    for w, ch in chains.items():
        print(f"  {w!r}  letter-primes order={word_to_order(w)}  sorted={ch}")

    if len(set(chains.values())) != 1:
        print("\n  Note: words use different letter sets; demo uses shared site from first word.")
    site = SharedSite(chain=chains[words[0]])

    dots = {w: encode_word_at_site(w, site) for w in words}
    base = canonical_base(dots[words[0]])

    print(f"\n  Shared canonical base: {base}")
    print(f"  Side epsilon:          {PERM_EPSILON}\n")

    for w, dot in dots.items():
        base_d = canonical_base(dot)
        dist = distance(dot.coord, base_d)
        order = decode_order_from_dot(base_d, dot.coord, site.chain)
        decoded = decode_word(dot)
        print(f"  {w!r}:")
        print(f"    dot      {dot.coord}")
        print(f"    offset   {dist:.6f} from base")
        print(f"    order    {order} -> {order_to_word(order)!r}")
        print(f"    decode   {decoded!r}  ok={decoded == w}")

    w1, w2 = words[0], words[1]
    sep = distance(dots[w1].coord, dots[w2].coord)
    print(f"\n  Distance {w1!r} <-> {w2!r}: {sep:.6f}  (near, but distinct)")
    print(f"  {explain_order(site.chain, word_to_order(w1))}")
    print(f"  {explain_order(site.chain, word_to_order(w2))}")


def demo() -> None:
    compare_words("tab", "bat")
    print("\n--- Also works for longer anagrams ---")
    compare_words("listen", "silent")


if __name__ == "__main__":
    demo()
