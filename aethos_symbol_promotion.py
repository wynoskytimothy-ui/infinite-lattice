"""
Standalone ordered-subword promotion — path-sensitive nodes on the imaginary line.

L1 collapse
-----------
``ca`` / ``ac`` share ICN and Ψ.  ``the`` / ``het`` / ``eth`` (3-way L1 triple)
share one triple-lock node but took different paths.

L2 promotion
------------
Every **ordered** 1/2/3-gram subword → dedicated pool prime → distinct solo Ψ.

For length-3 with three distinct symbols, **all 6 path permutations** promote
even if only one ordering appeared in corpus (``the`` ⇒ ``teh``, ``het``, …).

Corpus layer: ``aethos_symbol_corpus.py`` scans docs, counts frequency, promotes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations

from aethos_complex_plane import ComplexPlane3D, wing_transform
from aethos_lattice import BranchKind
from aethos_sequences import SequenceKind, make_chain
from aethos_symbol_map import normalize_symbol, text_icn, text_icn_chain, text_symbol_chain

# Standalone L2 band — odd primes after L1 punct band [512:576).
_L2_ORDERED_PRIMES: tuple[int, ...] = make_chain(SequenceKind.PRIMES, 5000)[576:]

MIN_SUBWORD_LEN = 1
MAX_SUBWORD_LEN = 3


def all_path_permutations(text: str) -> tuple[str, ...]:
    """
    Unique ordered permutations of a subword's symbol multiset.

    ``the`` → 6 paths; ``see`` → 3 paths; ``aaa`` → 1 path.
    """
    if not text:
        return ()
    return tuple(dict.fromkeys("".join(p) for p in permutations(text, len(text))))


def validate_subword(text: str, *, min_len: int = MIN_SUBWORD_LEN, max_len: int = MAX_SUBWORD_LEN) -> str:
    if not (min_len <= len(text) <= max_len):
        raise ValueError(f"subword length must be {min_len}..{max_len}, got {len(text)!r}")
    for ch in text:
        normalize_symbol(ch)
    return text


@dataclass(frozen=True)
class OrderedSubword:
    """One promoted ordered subword — path is identity."""

    text: str
    length: int
    prime: int
    path: tuple[int, ...]
    icn: int
    icn_chain: tuple[int, ...]
    frequency: int = 0


@dataclass
class SymbolPromotionRegistry:
    """One pool prime per distinct ordered subword (length 1–3)."""

    min_len: int = MIN_SUBWORD_LEN
    max_len: int = MAX_SUBWORD_LEN
    promote_trigram_siblings: bool = True
    _cursor: int = 0
    promoted: dict[str, OrderedSubword] = field(default_factory=dict)
    _freq: dict[str, int] = field(default_factory=dict)

    def record_frequency(self, text: str, count: int = 1) -> None:
        self._freq[text] = self._freq.get(text, 0) + count

    def frequency(self, text: str) -> int:
        return self._freq.get(text, 0)

    def promote(self, text: str, *, frequency: int | None = None) -> OrderedSubword:
        """Allocate a pool prime for this exact ordered subword."""
        text = validate_subword(text, min_len=self.min_len, max_len=self.max_len)
        if text in self.promoted:
            tok = self.promoted[text]
            if frequency is not None and frequency > tok.frequency:
                self.promoted[text] = OrderedSubword(
                    text=tok.text,
                    length=tok.length,
                    prime=tok.prime,
                    path=tok.path,
                    icn=tok.icn,
                    icn_chain=tok.icn_chain,
                    frequency=frequency,
                )
            return self.promoted[text]
        if self._cursor >= len(_L2_ORDERED_PRIMES):
            raise RuntimeError("standalone L2 promotion pool exhausted")
        prime = _L2_ORDERED_PRIMES[self._cursor]
        self._cursor += 1
        freq = frequency if frequency is not None else self._freq.get(text, 0)
        tok = OrderedSubword(
            text=text,
            length=len(text),
            prime=prime,
            path=text_symbol_chain(text),
            icn=text_icn(text),
            icn_chain=text_icn_chain(text),
            frequency=freq,
        )
        self.promoted[text] = tok
        return tok

    def promote_with_siblings(self, text: str, *, frequency: int | None = None) -> list[OrderedSubword]:
        """
        Promote ``text`` and, for length-3, all path permutations (6 for distinct letters).
        """
        text = validate_subword(text, min_len=self.min_len, max_len=self.max_len)
        targets = all_path_permutations(text) if len(text) == 3 and self.promote_trigram_siblings else (text,)
        out: list[OrderedSubword] = []
        for t in targets:
            freq = frequency if (frequency is not None and t == text) else self._freq.get(t, 0)
            out.append(self.promote(t, frequency=freq))
        return out

    def by_length(self, n: int) -> tuple[OrderedSubword, ...]:
        return tuple(sorted(
            (t for t in self.promoted.values() if t.length == n),
            key=lambda t: (-t.frequency, t.text),
        ))

    def promoted_node(
        self,
        text: str,
        *,
        n: int = 7,
        branch: BranchKind = BranchKind.VA1,
        wing: int = 1,
    ) -> tuple[OrderedSubword, ComplexPlane3D]:
        tok = self.promote(text)
        psi = wing_transform(branch, (tok.prime,), n=n, wing=wing)
        return tok, psi


def path_collision_report(a: str, b: str) -> dict[str, object]:
    """Why two anagrams collide at L1 and separate after L2 promotion."""
    from aethos_symbol_map import subword_on_imaginary_line

    reg = SymbolPromotionRegistry()
    l1_a = subword_on_imaginary_line(a, n=7)
    l1_b = subword_on_imaginary_line(b, n=7)
    tok_a, p_a = reg.promoted_node(a, n=7)
    tok_b, p_b = reg.promoted_node(b, n=7)
    return {
        "a": a,
        "b": b,
        "l1_order_a": text_symbol_chain(a),
        "l1_order_b": text_symbol_chain(b),
        "l1_icn_same": text_icn(a) == text_icn(b),
        "l1_psi_same": l1_a.z == l1_b.z and l1_a.zeta == l1_b.zeta,
        "l2_prime_a": tok_a.prime,
        "l2_prime_b": tok_b.prime,
        "l2_psi_same": p_a.z == p_b.z and p_a.zeta == p_b.zeta,
        "l2_psi_a": p_a.z,
        "l2_psi_b": p_b.z,
    }


def trigram_path_report(seed: str = "the") -> dict[str, object]:
    """All path permutations of a trigram → distinct L2 primes."""
    from aethos_symbol_map import subword_on_imaginary_line

    paths = all_path_permutations(seed)
    reg = SymbolPromotionRegistry()
    l1_coords: set[tuple[float, float, float]] = set()
    l2_primes: list[int] = []
    l2_z: list[complex] = []
    for p in paths:
        psi_l1 = subword_on_imaginary_line(p, n=7)
        l1_coords.add(psi_l1.coord)
        tok, psi_l2 = reg.promoted_node(p, n=7)
        l2_primes.append(tok.prime)
        l2_z.append(psi_l2.z)
    return {
        "seed": seed,
        "paths": paths,
        "n_paths": len(paths),
        "l1_unique_coords": len(l1_coords),
        "l2_unique_primes": len(set(l2_primes)),
        "l2_unique_z": len(set(l2_z)),
        "promoted": {t.text: t.prime for t in reg.by_length(3)},
    }


def demo() -> None:
    print("=" * 60)
    print("ORDERED SUBWORD PROMOTION")
    print("=" * 60)
    print("\n  ca vs ac:")
    for k, v in path_collision_report("ca", "ac").items():
        print(f"    {k}: {v}")
    print("\n  trigram paths (the → 6 L2 nodes):")
    r = trigram_path_report("the")
    for k, v in r.items():
        print(f"    {k}: {v}")


if __name__ == "__main__":
    demo()
