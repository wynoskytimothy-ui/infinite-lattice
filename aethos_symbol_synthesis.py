"""
Word synthesis on the imaginary number line — sum of meeting primes.

Model (standalone)
------------------
Each symbol has an L1 prime (``a``→3, ``b``→5, …).  When primes **meet** on the
imaginary line, their address is the **sum**:

    prime 3 meets prime 5  →  imaginary position **8**

The system **knows all corpus words** and discovers what meetings are needed:

  • **2-letter word**  → 2-way meet of two L1 primes  →  imag = p₁ + p₂
  • **3-letter word**  → 3-way meet of three L1 primes →  imag = p₁ + p₂ + p₃
  • **longer word**    → 2-way meet of two already-synthesized parts
                         →  imag = imag(left) + imag(right)

Only **unique** meetings promote to a composite pool prime (same imag + same
factorization must not ambiguously map to two words).

This replaces span-counting; the imaginary line uses **prime sums**, not symbol length.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from aethos_complex_plane import ComplexPlane3D, imaginary_start, wing_transform
from aethos_intersection_nodes import IntersectionNetwork
from aethos_lattice import BranchKind
from aethos_sequences import SequenceKind, make_chain
from aethos_symbol_map import symbol_to_prime, text_symbol_chain
from aethos_symbol_promotion import validate_subword

# L5 synthesis composites — pool primes tagged to imaginary sum witnesses.
_SYNTHESIS_PRIMES: tuple[int, ...] = make_chain(SequenceKind.PRIMES, 30_000)[20_000:28_000]


@dataclass(frozen=True)
class WordComposite:
    """Promoted word composite — unique meeting on the imaginary line."""

    text: str
    composite_prime: int
    imaginary_position: int  # sum of primes that met (e.g. 3+5=8)
    meet_arity: int  # 2 or 3
    meeting_primes: tuple[int, ...]  # operands at this meet (L1 or child composite primes)
    parts: tuple[str, ...]  # child word texts (("",) for pure L1 arity-3)


@dataclass
class SynthesisRegistry:
    """Bottom-up word synthesis from corpus vocabulary."""

    _cursor: int = 0
    composites: dict[str, WordComposite] = field(default_factory=dict)
    by_imag: dict[int, list[str]] = field(default_factory=dict)

    def _alloc(self) -> int:
        if self._cursor >= len(_SYNTHESIS_PRIMES):
            raise RuntimeError("synthesis prime pool exhausted")
        p = _SYNTHESIS_PRIMES[self._cursor]
        self._cursor += 1
        return p

    def register(self, comp: WordComposite) -> WordComposite:
        self.composites[comp.text] = comp
        self.by_imag.setdefault(comp.imaginary_position, []).append(comp.text)
        return comp

    def imaginary_position(self, text: str) -> int | None:
        if text in self.composites:
            return self.composites[text].imaginary_position
        if len(text) == 1:
            return symbol_to_prime(text)
        return None

    def is_unique_imag(self, imag: int, word: str) -> bool:
        others = [w for w in self.by_imag.get(imag, []) if w != word]
        return len(others) == 0


def imaginary_sum(*primes: int) -> int:
    """Position on the imaginary number line when these primes meet."""
    return sum(primes)


def l1_primes(word: str) -> tuple[int, ...]:
    return text_symbol_chain(word)


def meet_2way_on_line(p: int, q: int) -> int:
    """prime p meets prime q → imaginary position p+q."""
    return p + q


def meet_3way_on_line(a: int, p: int, q: int) -> int:
    return a + p + q


def probe_l1_meet(primes: tuple[int, ...]) -> bool:
    """Geometric witness exists for this L1 prime tuple."""
    net = IntersectionNetwork()
    if len(primes) == 2:
        p, q = sorted(primes)
        return net.probe_solo_swap(p, q) is not None
    if len(primes) == 3:
        a, p, q = sorted(primes)
        return net.probe_triple(a, p, q) is not None
    return False


def psi_at_imaginary_position(position: int, *, n: int | None = None) -> ComplexPlane3D:
    """
    Readout on the imaginary line at sum position.

    Layer-0 diagonal uses n; when n equals the imaginary sum, z = n + n·i.
    """
    rail = float(position if n is None else n)
    return imaginary_start(rail)


def synthesize_len2(word: str, reg: SynthesisRegistry) -> WordComposite | None:
    """Two L1 primes meet → imag = p₁+p₂."""
    if len(word) != 2:
        return None
    primes = l1_primes(word)
    if len(primes) != 2:
        return None
    imag = imaginary_sum(*primes)
    if not probe_l1_meet(primes):
        return None
    if not reg.is_unique_imag(imag, word) and word not in reg.composites:
        return None
    if word in reg.composites:
        return reg.composites[word]
    comp = WordComposite(
        text=word,
        composite_prime=reg._alloc(),
        imaginary_position=imag,
        meet_arity=2,
        meeting_primes=primes,
        parts=(word[0], word[1]),
    )
    return reg.register(comp)


def synthesize_len3(word: str, reg: SynthesisRegistry) -> WordComposite | None:
    """Three L1 primes meet (3-way) → imag = p₁+p₂+p₃."""
    if len(word) != 3:
        return None
    primes = l1_primes(word)
    if len(primes) != 3:
        return None
    imag = imaginary_sum(*primes)
    if not probe_l1_meet(primes):
        return None
    if not reg.is_unique_imag(imag, word) and word not in reg.composites:
        return None
    if word in reg.composites:
        return reg.composites[word]
    comp = WordComposite(
        text=word,
        composite_prime=reg._alloc(),
        imaginary_position=imag,
        meet_arity=3,
        meeting_primes=primes,
        parts=(word[0], word[1], word[2]),
    )
    return reg.register(comp)


def synthesize_len2_split(word: str, reg: SynthesisRegistry) -> WordComposite | None:
    """Longer word: 2-way meet of two synthesized parts → imag = imag(L)+imag(R)."""
    if len(word) < 2:
        return None
    best: WordComposite | None = None
    for i in range(1, len(word)):
        left, right = word[:i], word[i:]
        try:
            validate_subword(left)
            validate_subword(right)
        except ValueError:
            continue
        imag_l = reg.imaginary_position(left)
        imag_r = reg.imaginary_position(right)
        if imag_l is None or imag_r is None:
            continue
        imag = imag_l + imag_r
        left_c = reg.composites.get(left)
        right_c = reg.composites.get(right)
        meet_primes = (
            (left_c.composite_prime if left_c else imag_l),
            (right_c.composite_prime if right_c else imag_r),
        )
        if not reg.is_unique_imag(imag, word) and word not in reg.composites:
            continue
        if word in reg.composites:
            return reg.composites[word]
        comp = WordComposite(
            text=word,
            composite_prime=reg._alloc(),
            imaginary_position=imag,
            meet_arity=2,
            meeting_primes=meet_primes,
            parts=(left, right),
        )
        best = reg.register(comp)
        break
    return best


def synthesize_word(word: str, reg: SynthesisRegistry) -> WordComposite | None:
    """Try shortest construction: len2 L1, len3 L1, then binary split."""
    word = word.lower()
    if not word or not word.isalpha():
        return None
    if word in reg.composites:
        return reg.composites[word]
    if len(word) == 1:
        return None
    if len(word) == 2:
        return synthesize_len2(word, reg)
    if len(word) == 3:
        return synthesize_len3(word, reg)
    return synthesize_len2_split(word, reg)


def build_vocabulary(
    words: set[str] | list[str],
    *,
    max_len: int = 27,
) -> SynthesisRegistry:
    """
    Know all words; synthesize bottom-up by length so parts exist before merges.

    Order: length 2 → 3 → 4 → … so each 2-way split finds child composites.
    """
    reg = SynthesisRegistry()
    vocab = sorted({w.lower() for w in words if w and w.isalpha()}, key=lambda w: (len(w), w))
    for w in vocab:
        if len(w) > max_len:
            continue
        if len(w) == 2:
            synthesize_len2(w, reg)
        elif len(w) == 3:
            synthesize_len3(w, reg)
    for w in vocab:
        if len(w) > max_len or len(w) <= 3:
            continue
        synthesize_len2_split(w, reg)
    return reg


def needed_meetings(word: str, reg: SynthesisRegistry) -> dict[str, object] | None:
    """What primes / parts are needed to make this word (after build_vocabulary)."""
    comp = reg.composites.get(word.lower())
    if comp is None:
        return None
    return {
        "word": comp.text,
        "meet_arity": comp.meet_arity,
        "meeting_primes": comp.meeting_primes,
        "imaginary_position": comp.imaginary_position,
        "parts": comp.parts,
        "composite_prime": comp.composite_prime,
    }


def demo() -> None:
    print("=" * 60)
    print("WORD SYNTHESIS — imaginary line = sum of meeting primes")
    print("=" * 60)

    # User example: 3 meets 5 → 8
    pa, pb = symbol_to_prime("a"), symbol_to_prime("b")
    imag_ab = meet_2way_on_line(pa, pb)
    print(f"\n  prime {pa} meets {pb}  ->  imaginary position {imag_ab}")
    print(f"  z at layer-0 rail n={imag_ab}: {psi_at_imaginary_position(imag_ab).z}")

    words = {"ab", "ba", "cat", "the", "cats", "at", "ca"}
    reg = build_vocabulary(words)

    print("\n  corpus words -> synthesis:")
    for w in sorted(words):
        info = needed_meetings(w, reg)
        if info:
            print(
                f"    {w!r:5}  arity={info['meet_arity']}  "
                f"primes={info['meeting_primes']}  imag={info['imaginary_position']}  "
                f"parts={info['parts']}"
            )
        elif len(w) == 1:
            print(f"    {w!r:5}  L1 prime={symbol_to_prime(w)}")
        else:
            print(f"    {w!r:5}  (not uniquely synthesized)")

    print(f"\n  promoted composites: {len(reg.composites)}")


if __name__ == "__main__":
    demo()
