"""
Shared tokenization — NFKC normalization, apostrophe, hyphen, species split.

Single entry for promotion ingest, natural reading, and LatticeToken emitters.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from aethos_species import TokenSpecies, is_numeric_token

# Common apostrophe and hyphen code points (NFKC may fold some of these).
APOSTROPHE_CHARS: frozenset[str] = frozenset("'\u2019`\u02bc")
HYPHEN_CHARS: frozenset[str] = frozenset("-\u2010\u2011\u2013\u2014")

# Optional numeric punctuation stripped before NUM classification.
_NUM_STRIP = re.compile(r"^[\$€£¥₹]+|[,_.]+$")


def normalize_unicode(text: str) -> str:
    """Unicode NFKC — compatibility fold before tokenization."""
    return unicodedata.normalize("NFKC", text)


def clean_word_token(raw: str) -> str:
    """
    Lowercase alphabetic token after punctuation policy.

    - Apostrophes between letters are removed: don't -> dont
    - Hyphens between letters are removed: co-operate -> cooperate
    - Non-letters are stripped; empty -> ''
    """
    w = normalize_unicode(raw).lower()
    for ch in APOSTROPHE_CHARS:
        w = w.replace(ch, "")
    for ch in HYPHEN_CHARS:
        w = w.replace(ch, "")
    return "".join(c for c in w if c.isalpha())


def _numeric_core(piece: str) -> str | None:
    """Extract pure digit run for NUM species (commas/currency stripped)."""
    s = normalize_unicode(piece).strip()
    s = _NUM_STRIP.sub("", s)
    if s.startswith("+"):
        s = s[1:]
    if not s:
        return None
    if s.isdigit():
        return s
    # Decimal: keep digits only if single dot (3.14 -> 314 policy: drop dot)
    if s.count(".") == 1:
        whole, frac = s.split(".", 1)
        if whole.isdigit() and frac.isdigit():
            return whole + frac
    return None


@dataclass(frozen=True)
class TokenSpan:
    raw: str
    text: str
    species: TokenSpecies
    position: int = 0


def tokenize_spans(text: str) -> list[TokenSpan]:
    """Whitespace tokenization with WORD / NUM species."""
    text = normalize_unicode(text)
    spans: list[TokenSpan] = []
    pos = 0
    for piece in text.split():
        if not piece.strip():
            continue
        num = _numeric_core(piece)
        if num is not None:
            spans.append(TokenSpan(raw=piece, text=num, species=TokenSpecies.NUM, position=pos))
            pos += 1
            continue
        w = clean_word_token(piece)
        if w:
            spans.append(TokenSpan(raw=piece, text=w, species=TokenSpecies.WORD, position=pos))
            pos += 1
    return spans


def tokenize_words(text: str) -> list[str]:
    """All token texts in order (WORD + NUM) for co-occurrence windows."""
    return [s.text for s in tokenize_spans(text)]


def tokenize_word_only(text: str) -> list[str]:
    """Alphabetic words only (legacy helper)."""
    return [s.text for s in tokenize_spans(text) if s.species == TokenSpecies.WORD]


def tokenize_with_raw(text: str) -> list[tuple[str, str, TokenSpecies]]:
    """Return (raw, cleaned, species) triples for diagnostics."""
    return [(s.raw, s.text, s.species) for s in tokenize_spans(text)]
