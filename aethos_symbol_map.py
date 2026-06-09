"""
L1 English symbol map v2 — every English-language character → unique prime.

Three disjoint prime bands (FTA-safe for subword ICN products):
  ALPHA  a–z           → odd primes 3..103        (26, shared with aethos_words)
  DIGIT  0–9           → even primes 2..20         (10, shared with aethos_species)
  PUNCT  space+punct  → odd primes band [512:576)  (64 reserved, clear of promotion pool)

Imaginary-line placement
------------------------
Each symbol prime p is a solo anchor on the imaginary vector:

    Psi(p) = wing_transform(VA1, (p,), n, wing=1)

Layer 0 (empty chain) walks the diagonal n(1+i).  Solo anchors branch off that
rail when the transgressor crosses p — the subword ICN (product of symbol primes)
becomes a composite chain address on the plane.

Subwords
--------
  "th"  → primes (t, h)  → ICN = p_t * p_h  → chain_from_composite(ICN)
  "ing" → product of i,n,g primes → unique lattice node (meets each factor bank)

Use ``text_symbol_chain`` for left-to-right order (anagram discrimination).
Use ``text_icn`` / ``text_icn_chain`` for sorted-unique factor chains (meets).
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from enum import Enum
from functools import reduce
from operator import mul

from aethos_complex_plane import ComplexPlane3D, wing_transform
from aethos_intersection_nodes import chain_from_composite
from aethos_lattice import BranchKind
from aethos_sequences import SequenceKind, make_chain
from aethos_species import DIGIT_PRIMES, digit_to_prime, prime_to_digit
from aethos_words import LETTER_PRIMES, letter_to_prime, prime_to_letter

# Reserved high odd-prime band — indices 512..575 (64 slots), above promotion pool [26:512).
_PUNCT_PRIMES: tuple[int, ...] = make_chain(SequenceKind.PRIMES, 576)[512:576]

# Whitespace + ASCII punctuation (canonical keys; Unicode variants fold in normalize_symbol).
PUNCT_SYMBOLS: tuple[str, ...] = (
    " ",
    "\t",
    "\n",
    "\r",
    "!",
    '"',
    "#",
    "$",
    "%",
    "&",
    "'",
    "(",
    ")",
    "*",
    "+",
    ",",
    "-",
    ".",
    "/",
    ":",
    ";",
    "<",
    "=",
    ">",
    "?",
    "@",
    "[",
    "\\",
    "]",
    "^",
    "_",
    "`",
    "{",
    "|",
    "}",
    "~",
)

# NFKC-safe folds into canonical ASCII table keys.
_UNICODE_TO_ASCII: dict[str, str] = {
    "\u2013": "-",
    "\u2014": "-",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2026": ".",
}

if len(PUNCT_SYMBOLS) > len(_PUNCT_PRIMES):
    raise RuntimeError(
        f"PUNCT_SYMBOLS ({len(PUNCT_SYMBOLS)}) exceeds reserved band ({len(_PUNCT_PRIMES)})"
    )

PUNCT_PRIMES: tuple[int, ...] = _PUNCT_PRIMES[: len(PUNCT_SYMBOLS)]
_SYMBOL_TO_PRIME: dict[str, int] = {}
_PRIME_TO_SYMBOL: dict[int, str] = {}


class SymbolKind(str, Enum):
    ALPHA = "ALPHA"
    DIGIT = "DIGIT"
    PUNCT = "PUNCT"


def _register(kind: SymbolKind, symbol: str, prime: int) -> None:
    if symbol in _SYMBOL_TO_PRIME:
        raise RuntimeError(f"duplicate symbol registration: {symbol!r}")
    if prime in _PRIME_TO_SYMBOL:
        raise RuntimeError(f"duplicate prime registration: {prime}")
    _SYMBOL_TO_PRIME[symbol] = prime
    _PRIME_TO_SYMBOL[prime] = symbol


def _init_tables() -> None:
    if _SYMBOL_TO_PRIME:
        return
    for i, ch in enumerate("abcdefghijklmnopqrstuvwxyz"):
        _register(SymbolKind.ALPHA, ch, LETTER_PRIMES[i])
    for d in "0123456789":
        _register(SymbolKind.DIGIT, d, digit_to_prime(d))
    for sym, prime in zip(PUNCT_SYMBOLS, PUNCT_PRIMES):
        _register(SymbolKind.PUNCT, sym, prime)


_init_tables()

ENGLISH_SYMBOLS: frozenset[str] = frozenset(_SYMBOL_TO_PRIME)


def normalize_symbol(c: str) -> str:
    """NFKC fold; uppercase letters → lowercase; Unicode punctuation → ASCII keys."""
    if c in _UNICODE_TO_ASCII:
        c = _UNICODE_TO_ASCII[c]
    c = unicodedata.normalize("NFKC", c)
    if len(c) != 1:
        raise ValueError(f"single character required: {c!r}")
    if c in _UNICODE_TO_ASCII:
        c = _UNICODE_TO_ASCII[c]
    cl = c.lower()
    if "a" <= cl <= "z":
        return cl
    if cl.isdigit():
        return cl
    if c in _SYMBOL_TO_PRIME:
        return c
    raise ValueError(f"unmapped English symbol: {c!r}")


def _kind_for_prime(p: int) -> SymbolKind:
    if p in LETTER_PRIMES:
        return SymbolKind.ALPHA
    if p in DIGIT_PRIMES:
        return SymbolKind.DIGIT
    if p in PUNCT_PRIMES:
        return SymbolKind.PUNCT
    raise KeyError(f"not an L1 symbol prime: {p}")


def symbol_kind(c: str) -> SymbolKind:
    return _kind_for_prime(symbol_to_prime(c))


def symbol_to_prime(c: str) -> int:
    """Map one English symbol to its unique L1 prime."""
    return _SYMBOL_TO_PRIME[normalize_symbol(c)]


def prime_to_symbol(p: int) -> str:
    """Inverse map; raises KeyError for unknown primes."""
    return _PRIME_TO_SYMBOL[p]


def text_symbol_chain(text: str) -> tuple[int, ...]:
    """Left-to-right symbol primes (preserves order for subword / anagram paths)."""
    return tuple(symbol_to_prime(c) for c in text)


def text_icn(text: str) -> int:
    """ICN: product of distinct symbol primes in text (FTA-unique composite address)."""
    factors = set(text_symbol_chain(text))
    if not factors:
        return 1
    return reduce(mul, factors)


def text_icn_chain(text: str) -> tuple[int, ...]:
    """Sorted unique prime factors — chain for wing_transform / meets."""
    return chain_from_composite(text_icn(text))


def text_intersection(text: str) -> int:
    """Sum of symbol primes (parallel to letter intersection for mixed strings)."""
    return sum(text_symbol_chain(text))


def symbol_on_imaginary_line(
    c: str,
    *,
    n: int = 7,
    branch: BranchKind = BranchKind.VA1,
    wing: int = 1,
) -> ComplexPlane3D:
    """Solo-anchor Psi for one symbol on the imaginary vector."""
    p = symbol_to_prime(c)
    return wing_transform(branch, (p,), n=n, wing=wing)


def subword_on_imaginary_line(
    subword: str,
    *,
    n: int = 7,
    branch: BranchKind = BranchKind.VA1,
    wing: int = 1,
) -> ComplexPlane3D:
    """Composite ICN chain for a subword (product of its symbol primes)."""
    chain = text_icn_chain(subword)
    if not chain:
        raise ValueError("empty subword")
    return wing_transform(branch, chain, n=n, wing=wing)


@dataclass(frozen=True)
class SymbolTableEntry:
    symbol: str
    kind: SymbolKind
    prime: int
    index: int  # position within kind band


def symbol_table() -> tuple[SymbolTableEntry, ...]:
    """Full English L1 table for inspection / export."""
    rows: list[SymbolTableEntry] = []
    alpha_i = digit_i = punct_i = 0
    for sym in sorted(_SYMBOL_TO_PRIME, key=lambda s: (_SYMBOL_TO_PRIME[s], s)):
        p = _SYMBOL_TO_PRIME[sym]
        kind = _kind_for_prime(p)
        if kind == SymbolKind.ALPHA:
            idx = alpha_i
            alpha_i += 1
        elif kind == SymbolKind.DIGIT:
            idx = digit_i
            digit_i += 1
        else:
            idx = punct_i
            punct_i += 1
        rows.append(SymbolTableEntry(symbol=sym, kind=kind, prime=p, index=idx))
    return tuple(rows)


def demo() -> None:
    print("=" * 60)
    print("L1 ENGLISH SYMBOL MAP v2 — primes on imaginary line")
    print("=" * 60)
    print(f"  ALPHA: 26  DIGIT: 10  PUNCT: {len(PUNCT_SYMBOLS)}  TOTAL: {len(ENGLISH_SYMBOLS)}")
    print()

    for label, ch in (("letter", "t"), ("digit", "7"), ("punct", "."), ("space", " ")):
        p = symbol_to_prime(ch)
        psi = symbol_on_imaginary_line(ch, n=7)
        print(f"  {label:6} {ch!r:6}  prime={p:5}  z={psi.z}  zeta={psi.zeta}")

    print()
    for sw in ("th", "ing", "the"):
        icn = text_icn(sw)
        chain = text_icn_chain(sw)
        psi = subword_on_imaginary_line(sw, n=7)
        print(f"  subword {sw!r}: ICN={icn}  chain={chain}  z={psi.z}  zeta={psi.zeta}")

    print()
    print("  2-way / 3-way meets (see aethos_symbol_meets.py):")
    from aethos_symbol_meets import discover_text_meets

    for sw in ("th", "the"):
        d = discover_text_meets(sw, grow_network=False)
        print(f"    {sw!r}: {len(d.two_way)} solo swaps, {len(d.three_way)} triple locks")

    print()
    print("  Sample table (first 8 punct):")
    for entry in symbol_table():
        if entry.kind == SymbolKind.PUNCT and entry.index < 8:
            disp = repr(entry.symbol)
            print(f"    {disp:8} kind=PUNCT prime={entry.prime}")


if __name__ == "__main__":
    demo()
