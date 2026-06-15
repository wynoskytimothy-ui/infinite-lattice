"""
Morphological synthesis — unique subwords, rare-word composites, shared correlations.

Example family (user model)
---------------------------
  ``diminis``  — unique subword  →  promoted L2 prime, own imag address
  ``ed``       — unique suffix   →  promoted L2 prime
  ``es``       — different suffix →  different promoted prime (not ``ed``)
  ``sub``      — unique subword  →  promoted L2 prime

  ``diminished`` — **rare** (not seen inside another vocabulary word)
                   → 2-way composite of root + ``ed``
                   → imaginary line position = imag(root) + imag(ed)
                   → e.g. when operands are 3 and 5, lives at **8** (z = 8+8j)

  ``diminishes`` — same root **correlation** prime as ``diminished``
                   → but meets ``es`` not ``ed``  →  different composite

Correlation = shared promoted root (``dimin`` / ``diminis``) links the family;
suffix primes differ so composites are unique.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from aethos_complex_plane import imaginary_start
from aethos_symbol_map import text_symbol_chain
from aethos_symbol_promotion import SymbolPromotionRegistry, validate_subword
from aethos_symbol_synthesis import (
    SynthesisRegistry,
    WordComposite,
    imaginary_sum,
    meet_2way_on_line,
)

# Re-export for morph tests / demos
__all__ = [
    "meet_2way_on_line",
    "build_morph_registry",
    "explain_diminish_family",
    "MorphRegistry",
    "canonical_morph_chain_and_imag",
    "longest_embedded_subword",
    "pick_root_suffix",
]

_MORPH_SUFFIXES = ("tion", "ment", "ness", "ing", "ious", "ular", "ed", "es", "er", "ly", "s")

_CORRELATION_PRIMES: tuple[int, ...] = ()  # filled at import from pool tail


@dataclass(frozen=True)
class UniqueSubword:
    text: str
    prime: int
    imaginary_position: int  # sum of L1 symbol primes in subword
    parent_words: frozenset[str]


@dataclass(frozen=True)
class MorphCorrelation:
    """Shared root prime linking a word family."""

    root_text: str
    root_prime: int
    root_imag: int
    words: frozenset[str]


@dataclass(frozen=True)
class MorphComposite:
    word: str
    composite_prime: int
    imaginary_position: int
    meet_arity: int
    meeting_primes: tuple[int, ...]
    parts: tuple[str, ...]
    suffix: str
    correlation: MorphCorrelation
    rare: bool


@dataclass
class MorphRegistry:
    subwords: dict[str, UniqueSubword] = field(default_factory=dict)
    correlations: dict[str, MorphCorrelation] = field(default_factory=dict)
    composites: dict[str, MorphComposite] = field(default_factory=dict)
    _promo: SymbolPromotionRegistry = field(default_factory=SymbolPromotionRegistry)
    _synth: SynthesisRegistry = field(default_factory=SynthesisRegistry)
    _cursor: int = 0

    def _alloc_composite(self) -> int:
        from aethos_sequences import SequenceKind, make_chain

        global _CORRELATION_PRIMES
        if not _CORRELATION_PRIMES:
            _CORRELATION_PRIMES = make_chain(SequenceKind.PRIMES, 90_000)[28_000:78_000]
        if self._cursor >= len(_CORRELATION_PRIMES):
            raise RuntimeError("morph composite pool exhausted")
        p = _CORRELATION_PRIMES[self._cursor]
        self._cursor += 1
        return p

    def subword_imag(self, text: str) -> int:
        return imaginary_sum(*text_symbol_chain(text))

    def promote_morph_piece(self, text: str, *, parents: frozenset[str]) -> UniqueSubword:
        """
        Unique morph subword — ``ed``/``es``/``sub`` use L2 (len<=3);
        longer pieces like ``diminis`` get a composite pool prime + imag sum.
        """
        if text in self.subwords:
            return self.subwords[text]
        imag = self.subword_imag(text)
        if len(text) <= 3:
            prime = self._promo.promote(text).prime
        else:
            prime = self._alloc_composite()
        sw = UniqueSubword(
            text=text,
            prime=prime,
            imaginary_position=imag,
            parent_words=parents,
        )
        self.subwords[text] = sw
        return sw


def subword_parent_words(vocab: set[str], subword: str) -> frozenset[str]:
    return frozenset(w for w in vocab if subword in w)


def is_rare_word(word: str, vocab: set[str]) -> bool:
    """Rare = no other vocabulary word contains this exact string as substring."""
    return not any(word != other and word in other for other in vocab)


def discover_unique_subwords(
    vocab: set[str],
    *,
    candidates: tuple[str, ...] | None = None,
) -> dict[str, frozenset[str]]:
    """
    Subwords that are morphologically distinct — appear in vocab but identify
    a family piece (suffix ``ed``/``es``, root ``dimin``, etc.).
    """
    if candidates is None:
        candidates = ()
        for w in vocab:
            for i in range(len(w)):
                for ln in (1, 2, 3):
                    if i + ln <= len(w):
                        candidates += (w[i : i + ln],)
        candidates = tuple(dict.fromkeys(candidates))

    out: dict[str, frozenset[str]] = {}
    for sw in candidates:
        try:
            validate_subword(sw)
        except ValueError:
            continue
        parents = subword_parent_words(vocab, sw)
        if parents:
            out[sw] = parents
    return out


def pick_root_suffix(word: str, vocab: set[str] | None = None) -> tuple[str, str] | None:
    """Split token into promoted root + suffix (longest suffix first)."""
    w = word.lower()
    for suffix in _MORPH_SUFFIXES:
        if not w.endswith(suffix) or len(w) <= len(suffix) + 2:
            continue
        root = w[: -len(suffix)]
        if not (3 <= len(root) <= 18):
            continue
        if vocab is not None and root not in vocab:
            # still allow when root is already a morph subword (checked by caller)
            pass
        return root, suffix
    return None


def longest_embedded_subword(
    morph: MorphRegistry,
    word: str,
    *,
    min_len: int = 3,
    min_remainder: int = 2,
) -> str | None:
    """
    Longest morph subword in *word* when the leftover suffix is not a bare plural.

    ``cellular`` → ``cell`` (remainder ``ular``, len 4).
    ``shows`` → None (remainder ``s`` only — avoids hub collapse).
    """
    w = word.lower()
    if w in morph.subwords or w in morph.composites:
        return None
    hits = [
        sw for sw in morph.subwords
        if len(sw) >= min_len and sw in w and len(w) - len(sw) >= min_remainder
    ]
    return max(hits, key=len) if hits else None


def canonical_morph_chain_and_imag(
    morph: MorphRegistry,
    word: str,
) -> tuple[tuple[int, ...], int | None]:
    """
    Morph-aware chain + imaginary line position (Concrete Plane P1).

    Priority: composite → subword → root+suffix meet → root decay → embedded root.
    Returns ``((), None)`` when caller should fall back to L1 ICN chain.
    """
    w = word.lower()
    if w in morph.composites:
        comp = morph.composites[w]
        return (
            tuple(int(p) for p in comp.meeting_primes),
            int(comp.imaginary_position),
        )
    if w in morph.subwords:
        sw = morph.subwords[w]
        return (int(sw.prime),), int(sw.imaginary_position)

    embedded = longest_embedded_subword(morph, w)
    if embedded:
        sw = morph.subwords[embedded]
        return (int(sw.prime),), int(sw.imaginary_position)

    split = pick_root_suffix(w)
    if split:
        root, suffix = split
        root_sw = morph.subwords.get(root)
        suf_sw = morph.subwords.get(suffix)
        if root_sw and suf_sw:
            return (
                (int(root_sw.prime), int(suf_sw.prime)),
                int(root_sw.imaginary_position + suf_sw.imaginary_position),
            )
        if root_sw:
            return (int(root_sw.prime),), int(root_sw.imaginary_position)

    return (), None


def build_morph_registry(
    vocab: set[str] | list[str],
    *,
    seed_subwords: tuple[str, ...] = (
        "diminis", "dimin", "ed", "es", "s", "sub", "raise", "lower", "improve", "improves",
    ),
    max_composites: int | None = None,
) -> MorphRegistry:
    reg = MorphRegistry()
    words = {w.lower() for w in vocab if w and w.isalpha()}

    discovered = discover_unique_subwords(words)

    # Promote named unique subwords (ed, es, sub, diminis, dimin, …)
    for sw in seed_subwords:
        parents = discovered.get(sw, subword_parent_words(words, sw))
        if not parents and sw not in words:
            continue
        reg.promote_morph_piece(sw, parents=parents or frozenset({sw}))

    # Correlation: longest shared root among diminish*
    roots = [sw for sw in ("diminis", "dimin", "diminish") if sw in reg.subwords or any(w.startswith(sw) for w in words)]
    root_key = "dimin" if "dimin" in reg.subwords else (roots[0] if roots else "")
    if root_key and root_key in reg.subwords:
        family = frozenset(w for w in words if w.startswith(root_key) or root_key in w)
        sw = reg.subwords[root_key]
        reg.correlations[root_key] = MorphCorrelation(
            root_text=root_key,
            root_prime=sw.prime,
            root_imag=sw.imaginary_position,
            words=family,
        )

    composite_count = 0
    for word in sorted(words):
        if max_composites is not None and composite_count >= max_composites:
            break
        split = pick_root_suffix(word, words)
        if split is None:
            continue
        root, suffix = split
        if suffix not in reg.subwords:
            reg.promote_morph_piece(suffix, parents=subword_parent_words(words, suffix))
        if root not in reg.subwords:
            try:
                reg.promote_morph_piece(root, parents=subword_parent_words(words, root))
            except RuntimeError:
                continue

        root_sw = reg.subwords[root]
        suf_sw = reg.subwords[suffix]
        imag = root_sw.imaginary_position + suf_sw.imaginary_position
        if root not in reg.correlations:
            reg.correlations[root] = MorphCorrelation(
                root_text=root,
                root_prime=root_sw.prime,
                root_imag=root_sw.imaginary_position,
                words=frozenset({word}),
            )
        else:
            existing = reg.correlations[root]
            reg.correlations[root] = MorphCorrelation(
                root_text=existing.root_text,
                root_prime=existing.root_prime,
                root_imag=existing.root_imag,
                words=existing.words | frozenset({word}),
            )
        corr = reg.correlations[root]
        try:
            cprime = reg._alloc_composite()
        except RuntimeError:
            break
        comp = MorphComposite(
            word=word,
            composite_prime=cprime,
            imaginary_position=imag,
            meet_arity=2,
            meeting_primes=(root_sw.prime, suf_sw.prime),
            parts=(root, suffix),
            suffix=suffix,
            correlation=corr,
            rare=is_rare_word(word, words),
        )
        reg.composites[word] = comp
        composite_count += 1

    return reg


def explain_diminish_family() -> dict[str, object]:
    """User example: diminished vs diminishes, ed vs es, imag 8 for 3+5."""
    vocab = {
        "diminis", "diminished", "diminishes", "diminish",
        "sub", "subscribe", "reduced",
    }
    reg = build_morph_registry(vocab)

    # Canonical 3+5=8 on imaginary line (a+b)
    demo_imag = meet_2way_on_line(3, 5)
    z8 = imaginary_start(demo_imag)

    diminished = reg.composites.get("diminished")
    diminishes = reg.composites.get("diminishes")

    return {
        "unique_subwords": {
            k: {"prime": v.prime, "imag": v.imaginary_position, "parents": sorted(v.parent_words)}
            for k, v in reg.subwords.items()
        },
        "three_meets_five": {
            "primes": (3, 5),
            "imaginary_position": demo_imag,
            "z": str(z8.z),
            "note": "operand primes sum to imag line position 8",
        },
        "diminished": _morph_dict(diminished) if diminished else None,
        "diminishes": _morph_dict(diminishes) if diminishes else None,
        "same_root_correlation": (
            diminished.correlation.root_prime == diminishes.correlation.root_prime
            if diminished and diminishes
            else None
        ),
        "different_suffix_prime": (
            reg.subwords.get("ed", UniqueSubword("ed", 0, 0, frozenset())).prime
            != reg.subwords.get("es", UniqueSubword("es", 0, 0, frozenset())).prime
            if "ed" in reg.subwords and "es" in reg.subwords
            else None
        ),
        "diminished_rare": diminished.rare if diminished else None,
    }


def _morph_dict(c: MorphComposite | None) -> dict[str, object] | None:
    if c is None:
        return None
    return {
        "word": c.word,
        "parts": c.parts,
        "suffix": c.suffix,
        "meeting_primes": c.meeting_primes,
        "imaginary_position": c.imaginary_position,
        "composite_prime": c.composite_prime,
        "rare": c.rare,
        "z_on_line": str(imaginary_start(c.imaginary_position).z),
        "shared_root_prime": c.correlation.root_prime,
    }


def demo() -> None:
    print("=" * 60)
    print("MORPH SYNTHESIS — diminish family")
    print("=" * 60)
    r = explain_diminish_family()

    print("\n  3 meets 5 -> imag 8:")
    t = r["three_meets_five"]
    print(f"    primes {t['primes']}  imag={t['imaginary_position']}  z={t['z']}")

    print("\n  unique subwords:")
    for k, v in sorted(r["unique_subwords"].items()):
        print(f"    {k!r:10}  L2 prime={v['prime']}  imag={v['imag']}  in {v['parents'][:3]}")

    for key in ("diminished", "diminishes"):
        info = r[key]
        if info:
            print(f"\n  {key}:")
            print(f"    parts={info['parts']}  rare={info['rare']}")
            print(f"    meeting L2 primes={info['meeting_primes']}")
            print(f"    imag={info['imaginary_position']}  composite={info['composite_prime']}")
            print(f"    shared root prime={info['shared_root_prime']}")
            print(f"    suffix prime differs: ed vs es = {r['different_suffix_prime']}")


if __name__ == "__main__":
    demo()
