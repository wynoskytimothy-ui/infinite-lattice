"""
Corpus-driven 1/2/3-gram subword discovery + frequency + L2 promotion.

Scans every document for all ordered subwords of length 1, 2, and 3 on the
symbol stream.  Counts corpus frequency, records per-doc inventories, then
promotes each observed subword (and all 6 trigram path siblings when len=3).

Standalone v2 ingest:

    index = CorpusSubwordIndex()
    index.ingest_corpus({"d1": "the cat sat", "d2": "ether hypothesis"})
    index.promote_all()
    index.subwords_in_doc("d1")   # every distinct 1/2/3-gram in that doc
    index.top_subwords(length=3)  # by frequency
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field

from aethos_symbol_map import normalize_symbol
from aethos_symbol_promotion import (
    MAX_SUBWORD_LEN,
    MIN_SUBWORD_LEN,
    OrderedSubword,
    SymbolPromotionRegistry,
    all_path_permutations,
)

def symbol_stream(text: str) -> list[str]:
    """Normalized mappable symbols from text (lowercase letters + mapped punct/digits)."""
    text = unicodedata.normalize("NFKC", text).lower()
    out: list[str] = []
    for ch in text:
        try:
            out.append(normalize_symbol(ch))
        except ValueError:
            if ch.isspace():
                try:
                    out.append(normalize_symbol(" "))
                except ValueError:
                    pass
    return out


def extract_ordered_subwords(
    text: str,
    *,
    min_len: int = MIN_SUBWORD_LEN,
    max_len: int = MAX_SUBWORD_LEN,
) -> list[str]:
    """All contiguous ordered subwords of length min_len..max_len on symbol stream."""
    stream = symbol_stream(text)
    found: list[str] = []
    seen: set[str] = set()
    for i in range(len(stream)):
        for length in range(min_len, max_len + 1):
            if i + length > len(stream):
                break
            sw = "".join(stream[i : i + length])
            if sw not in seen:
                seen.add(sw)
                found.append(sw)
    return found


@dataclass
class SubwordStats:
    text: str
    length: int
    frequency: int
    doc_count: int
    docs: frozenset[str] = field(default_factory=frozenset)


@dataclass
class CorpusSubwordIndex:
    """
    Frequency index + per-doc subword inventory + promotion bridge.
    """

    min_len: int = MIN_SUBWORD_LEN
    max_len: int = MAX_SUBWORD_LEN
    promote_trigram_siblings: bool = True
    counts: dict[str, int] = field(default_factory=dict)
    doc_hits: dict[str, set[str]] = field(default_factory=dict)
    doc_subwords: dict[str, set[str]] = field(default_factory=dict)
    doc_texts: dict[str, str] = field(default_factory=dict)
    registry: SymbolPromotionRegistry = field(default_factory=SymbolPromotionRegistry)
    _promoted: bool = False

    def __post_init__(self) -> None:
        self.registry.min_len = self.min_len
        self.registry.max_len = self.max_len
        self.registry.promote_trigram_siblings = self.promote_trigram_siblings

    def observe_subword(self, subword: str, doc_id: str, *, count: int = 1) -> None:
        self.counts[subword] = self.counts.get(subword, 0) + count
        self.doc_hits.setdefault(subword, set()).add(doc_id)
        self.doc_subwords.setdefault(doc_id, set()).add(subword)
        self.registry.record_frequency(subword, count)

    def ingest_text(self, doc_id: str, text: str) -> int:
        """Scan one document; return number of distinct subwords found."""
        self.doc_texts[doc_id] = text
        subwords = extract_ordered_subwords(
            text, min_len=self.min_len, max_len=self.max_len,
        )
        for sw in subwords:
            self.observe_subword(sw, doc_id, count=1)
        return len(subwords)

    def ingest_corpus(self, corpus: dict[str, str]) -> dict[str, int]:
        """Ingest all docs. Returns doc_id → distinct subword count."""
        return {did: self.ingest_text(did, text) for did, text in corpus.items()}

    def stats(self, subword: str) -> SubwordStats | None:
        if subword not in self.counts:
            return None
        docs = frozenset(self.doc_hits.get(subword, set()))
        return SubwordStats(
            text=subword,
            length=len(subword),
            frequency=self.counts[subword],
            doc_count=len(docs),
            docs=docs,
        )

    def top_subwords(
        self,
        *,
        length: int | None = None,
        limit: int = 50,
    ) -> list[SubwordStats]:
        rows: list[SubwordStats] = []
        for sw, freq in self.counts.items():
            if length is not None and len(sw) != length:
                continue
            st = self.stats(sw)
            if st:
                rows.append(st)
        rows.sort(key=lambda s: (-s.frequency, -s.doc_count, s.text))
        return rows[:limit]

    def subwords_in_doc(self, doc_id: str) -> tuple[str, ...]:
        """All distinct 1/2/3-grams observed in one document."""
        return tuple(sorted(self.doc_subwords.get(doc_id, set())))

    def subwords_by_length_in_doc(self, doc_id: str) -> dict[int, tuple[str, ...]]:
        raw = self.subwords_in_doc(doc_id)
        out: dict[int, list[str]] = {1: [], 2: [], 3: []}
        for sw in raw:
            if self.min_len <= len(sw) <= self.max_len:
                out.setdefault(len(sw), []).append(sw)
        return {k: tuple(sorted(v)) for k, v in out.items() if v}

    def promote_all(self) -> SymbolPromotionRegistry:
        """
        Promote every observed subword by corpus frequency order.

        Length-3: also promotes all path permutations (6 for distinct letters).
        """
        ordered = sorted(
            self.counts.items(),
            key=lambda x: (-x[1], -len(self.doc_hits.get(x[0], set())), x[0]),
        )
        for sw, freq in ordered:
            if len(sw) == 3 and self.promote_trigram_siblings:
                self.registry.promote_with_siblings(sw, frequency=freq)
            else:
                self.registry.promote(sw, frequency=freq)
        self._promoted = True
        return self.registry

    def promoted_for_doc(self, doc_id: str) -> tuple[OrderedSubword, ...]:
        if not self._promoted:
            self.promote_all()
        sws = self.subwords_in_doc(doc_id)
        return tuple(
            self.registry.promoted[sw]
            for sw in sws
            if sw in self.registry.promoted
        )

    def branch_composites(self, **kwargs):
        """Promote 4–9 symbol composites from L2 meet branching (``aethos_symbol_composite``)."""
        from aethos_symbol_composite import CompositePromotionRegistry, branch_meets

        if not self._promoted:
            self.promote_all()
        kwargs.setdefault("min_frequency", 1)
        self._composite_registry: CompositePromotionRegistry = branch_meets(
            self.registry, **kwargs,
        )
        return self._composite_registry

    def build_cellular_entanglement(self):
        """Rare-rare correlations only; frequent subwords = membrane (no false edges)."""
        from aethos_symbol_cellular import build_cellular_entanglement
        from aethos_symbol_morph import build_morph_registry

        words: set[str] = set()
        for doc_id in self.doc_subwords:
            for sw in self.doc_subwords[doc_id]:
                if sw.strip().isalpha():
                    words.add(sw.strip())
        for sw in self.counts:
            if sw.strip().isalpha():
                words.add(sw.strip())
        corpus = dict(self.doc_texts) if self.doc_texts else {
            did: " ".join(sorted(self.doc_subwords.get(did, set())))
            for did in self.doc_subwords
        }
        morph = build_morph_registry(words)
        self._entangle_registry, self._cellular_registry = build_cellular_entanglement(
            corpus, morph,
        )
        return self._entangle_registry, self._cellular_registry

    def synthesize_words(self, *, max_len: int = 27) -> "SynthesisRegistry":
        """Build word composites: imaginary line = sum of meeting primes."""
        from aethos_symbol_synthesis import SynthesisRegistry, build_vocabulary

        words: set[str] = set()
        for doc_id in self.doc_subwords:
            for sw in self.doc_subwords[doc_id]:
                if sw.strip().isalpha():
                    words.add(sw.strip())
        for sw in self.counts:
            if sw.strip().isalpha():
                words.add(sw.strip())
        self._synthesis_registry: SynthesisRegistry = build_vocabulary(
            words, max_len=max_len,
        )
        return self._synthesis_registry

    def summary(self) -> dict[str, int | float]:
        by_len = {1: 0, 2: 0, 3: 0}
        for sw in self.counts:
            by_len[len(sw)] = by_len.get(len(sw), 0) + 1
        comp_n = (
            len(self._composite_registry.composites)
            if getattr(self, "_composite_registry", None)
            else 0
        )
        synth_n = (
            len(self._synthesis_registry.composites)
            if getattr(self, "_synthesis_registry", None)
            else 0
        )
        return {
            "docs": len(self.doc_subwords),
            "unique_subwords": len(self.counts),
            "unique_len1": by_len.get(1, 0),
            "unique_len2": by_len.get(2, 0),
            "unique_len3": by_len.get(3, 0),
            "total_observations": sum(self.counts.values()),
            "promoted_l2": len(self.registry.promoted) if self._promoted else 0,
            "promoted_composites": comp_n,
            "synthesized_words": synth_n,
        }


def demo() -> None:
    corpus = {
        "d1": "the cat sat on the mat",
        "d2": "ether hypothesis thesis",
    }
    idx = CorpusSubwordIndex()
    idx.ingest_corpus(corpus)
    idx.promote_all()

    print("=" * 60)
    print("CORPUS SUBWORD INDEX — 1/2/3 grams + frequency + promotion")
    print("=" * 60)
    print(f"  summary: {idx.summary()}")

    print("\n  doc d1 subwords by length:")
    for ln, sws in idx.subwords_by_length_in_doc("d1").items():
        print(f"    len{ln}: {len(sws)} unique  e.g. {sws[:8]}")

    print("\n  top trigrams:")
    for st in idx.top_subwords(length=3, limit=8):
        print(f"    {st.text!r}  freq={st.frequency}  docs={st.doc_count}")

    print("\n  'the' path siblings promoted:")
    siblings = all_path_permutations("the")
    for s in siblings:
        tok = idx.registry.promoted.get(s)
        if tok:
            print(f"    {s!r}  prime={tok.prime}  freq={tok.frequency}")


if __name__ == "__main__":
    demo()
