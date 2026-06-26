"""
Stage 01 — Symbols become primes.

Output: ordered letter-primes for each symbol in a string.
Prime order IS the coordinate order (first prime = real-axis anchor in the lattice).
No words, no promotion, no rotation — symbols only.
"""

from __future__ import annotations

from dataclasses import dataclass

from aethos_words import letter_to_prime, word_to_order


@dataclass(frozen=True)
class SymbolPrimeSpan:
    """One symbol and its prime — glass-box unit for stage 01."""

    char: str
    prime: int
    index: int


@dataclass(frozen=True)
class SymbolPrimeSequence:
    """Full prime sequence for a string + inspectable reasons."""

    text: str
    spans: tuple[SymbolPrimeSpan, ...]

    @property
    def primes(self) -> tuple[int, ...]:
        return tuple(s.prime for s in self.spans)

    @property
    def prime_order(self) -> tuple[int, ...]:
        """Sorted unique primes — lattice address order (intersection chain)."""
        return word_to_order(self.text.lower())

    def explain(self) -> dict:
        return {
            "text": self.text,
            "symbols": [
                {"char": s.char, "prime": s.prime, "index": s.index}
                for s in self.spans
            ],
            "sequence": list(self.primes),
            "order": list(self.prime_order),
        }


def symbols_to_primes(text: str) -> SymbolPrimeSequence:
    """Map each alphabetic symbol to its letter-prime in left-to-right order."""
    spans: list[SymbolPrimeSpan] = []
    for i, ch in enumerate(text):
        if not ch.isalpha():
            continue
        cl = ch.lower()
        spans.append(SymbolPrimeSpan(char=cl, prime=letter_to_prime(cl), index=i))
    return SymbolPrimeSequence(text=text, spans=tuple(spans))
