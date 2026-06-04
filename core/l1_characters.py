"""
L1 character layer (Step 2) — letters map to fixed odd primes.
"""

from __future__ import annotations

import unicodedata
import zlib

from core.primes import LETTER_PRIMES, chain_primes

_UNICODE_LETTER_PRIMES: tuple[int, ...] = chain_primes(512)[100:228]


def char_prime(c: str) -> int:
    """Map one letter to an odd prime (NFKC + Latin fold)."""
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


def prime_to_char(p: int) -> str:
    idx = LETTER_PRIMES.index(p)
    return chr(ord("a") + idx)


def word_letter_order(word: str) -> tuple[int, ...]:
    """Left-to-right letter primes."""
    return tuple(char_prime(c) for c in word.lower() if c.isalpha())


def word_letter_chain(word: str) -> tuple[int, ...]:
    """Sorted distinct letter primes (canonical multiset)."""
    return tuple(sorted(set(word_letter_order(word))))


def intersection_prime(word: str) -> int:
    """L1 anchor: sum of letter primes (not product)."""
    return sum(word_letter_order(word))
