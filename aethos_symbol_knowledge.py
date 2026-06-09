"""
Persistent symbol knowledge — corpus-wide words, correlations, cross-links.

Build once on a training corpus (e.g. SciFact), save to disk, reload before
queries.  The brain already knows:

  • **direct** — rare-rare pairs that co-occurred in a document
  • **morph** — shared-root family (never touched, same structure)
  • **bridge** — non-touching via root sibling + entangled neighbor

Gap report lists signal words that still have no correlation edge — those are
what a new, more explanatory corpus should teach next.

Usage::

    from aethos_symbol_knowledge import SymbolKnowledgeIndex

    idx = SymbolKnowledgeIndex.build_from_beir("scifact", max_docs=500)
    idx.save()  # brains/symbol_knowledge/scifact.pkl

    loaded = SymbolKnowledgeIndex.load("scifact")
    loaded.correlates("diminishes", "lower")   # True via bridge
    loaded.gap_words()                           # signal words still unlinked
"""

from __future__ import annotations

import json
import pickle
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Sequence

from aethos_symbol_cellular import (
    CellularRegistry,
    _DEFAULT_MEMBRANE,
    build_cellular_entanglement,
)
from aethos_symbol_entangle import EntanglementRegistry, EntangledPair
from aethos_symbol_morph import MorphRegistry, build_morph_registry
from aethos_symbol_subjects import (
    MASTER_CHAMBER,
    chambers_for_ingest,
    normalize_subjects,
    subjects_for_dataset,
)
from beir_data_root import resolve_beir_root

_TOKEN_RE = re.compile(r"[a-z]+")
KNOWLEDGE_VERSION = 3

_BEIR_URLS: dict[str, str] = {
    "scifact": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip",
    "nfcorpus": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip",
    "fiqa": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip",
}

LinkKind = Literal["direct", "morph", "bridge"]

# Gold pretrain doc: query concepts SciFact rarely links explicitly
PRETRAIN_QUANTUM_GOLD: dict[str, str] = {
    "gold_quantum_biometrics": (
        "Quantum inductive biometrics links measurement to zero dimensional "
        "Hilbert space boundaries. In this framework quantum states projected "
        "to zero dimension carry inductive biometric signatures. The zero "
        "dimensional quantum limit explains how inductive inference collapses "
        "to a single measurable axis. Quantum mechanics inductive biometrics "
        "and zero dimension co-occur as the foundational correlation triple "
        "for biometric state reduction in quantum measurement theory."
    ),
}


def _tokens_from_text(text: str) -> list[str]:
    return [
        t
        for t in _TOKEN_RE.findall(text.lower())
        if t not in _DEFAULT_MEMBRANE and len(t) >= 2
    ]


def _single_doc_cooccurrence_pairs(
    text: str,
    *,
    window: int = 15,
) -> dict[tuple[str, str], int]:
    """Sliding-window co-occurrence for one document (membrane excluded)."""
    pairs: dict[tuple[str, str], int] = {}
    tokens = _tokens_from_text(text)
    for i, a in enumerate(tokens):
        for b in tokens[i + 1 : i + window]:
            if a == b:
                continue
            key = tuple(sorted((a, b)))
            pairs[key] = pairs.get(key, 0) + 1
    return pairs


def _doc_cooccurrence_pairs(
    corpus: dict[str, str],
    *,
    window: int = 15,
) -> dict[tuple[str, str], int]:
    """
    Sliding-window co-occurrence within each document (membrane excluded).

    Full abstract all-pairs explodes on long docs and creates weak false links;
    a local window keeps correlations semantically tighter.
    """
    pairs: dict[tuple[str, str], int] = {}
    for text in corpus.values():
        for key, count in _single_doc_cooccurrence_pairs(text, window=window).items():
            pairs[key] = pairs.get(key, 0) + count
    return pairs


def _morph_vocab_subset(vocab: set[str], *, max_words: int = 4000) -> set[str]:
    """Morph composites only for suffix-splittable + polarity seeds (not full vocab)."""
    from aethos_symbol_entangle import _POLARITY_LEXICON

    candidates: list[str] = []
    for w in vocab:
        if len(w) < 5 or len(w) > 22:
            continue
        if w in _POLARITY_LEXICON:
            candidates.append(w)
            continue
        if w.endswith(("ed", "es", "ing", "ly", "er", "tion", "ment", "ness")):
            candidates.append(w)
    # Prefer shorter rarer-looking tokens first
    candidates.sort(key=lambda x: (len(x), x))
    out = set(candidates[:max_words])
    out.update(_POLARITY_LEXICON.keys())
    return out


@dataclass(frozen=True)
class DocEvidence:
    """Per-doc co-occurrence stored once — chambers are views, not copies."""

    chambers: frozenset[int]
    pairs: dict[tuple[str, str], int]


@dataclass(frozen=True)
class CrossLink:
    """One correlation edge — touching or inferred without co-occurrence."""

    left: str
    right: str
    kind: LinkKind
    strength: float
    chamber: int = MASTER_CHAMBER
    via: str | None = None
    opposite: bool = False
    intersection_imag: int | None = None


@dataclass
class SymbolKnowledgeIndex:
    """
    Full corpus knowledge: vocabulary, direct entanglement, cross-correlations.
    """

    dataset: str
    corpus: dict[str, str]
    morph: MorphRegistry
    entangle: EntanglementRegistry
    cellular: CellularRegistry
    chamber_links: dict[int, dict[tuple[str, str], CrossLink]] = field(
        default_factory=dict,
    )
    vocab: set[str] = field(default_factory=set)
    build_ms: float = 0.0
    version: int = KNOWLEDGE_VERSION
    ingest_rare_weight: bool = True
    _chamber_cooccur: dict[int, dict[tuple[str, str], int]] = field(
        default_factory=dict, repr=False,
    )
    _cooccur_pairs: dict[tuple[str, str], int] = field(default_factory=dict, repr=False)
    _doc_evidence: dict[str, DocEvidence] = field(default_factory=dict, repr=False)
    _chamber_dirty: set[int] = field(default_factory=set, repr=False)
    _morph_links_shared: dict[tuple[str, str], CrossLink] | None = field(
        default=None, repr=False,
    )
    _family_of_cache: dict[str, set[str]] | None = field(default=None, repr=False)
    _oov_lattice: dict[str, object] = field(default_factory=dict, repr=False)

    @property
    def cross_links(self) -> dict[tuple[str, str], CrossLink]:
        """Master chamber (k=0) — backward-compatible flat view."""
        return self.chamber_links.get(MASTER_CHAMBER, {})

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    @classmethod
    def build_from_corpus(
        cls,
        corpus: dict[str, str],
        *,
        dataset: str = "custom",
        morph: MorphRegistry | None = None,
        subjects: int | Sequence[int] | set[int] | None = None,
        doc_subjects: dict[str, set[int]] | None = None,
    ) -> SymbolKnowledgeIndex:
        """Index all docs: morph + cellular entanglement + chamber cross-links."""
        t0 = time.perf_counter()
        vocab: set[str] = set()
        for text in corpus.values():
            vocab.update(_TOKEN_RE.findall(text.lower()))

        morph = morph or build_morph_registry(
            _morph_vocab_subset(vocab), max_composites=3500,
        )
        entangle, cellular = build_cellular_entanglement(
            corpus, morph, knowledge_mode=True,
        )

        idx = cls(
            dataset=dataset,
            corpus=dict(corpus),
            morph=morph,
            entangle=entangle,
            cellular=cellular,
            vocab=vocab,
        )
        default_subjects = (
            normalize_subjects(subjects)
            if subjects is not None
            else subjects_for_dataset(dataset)
        )
        idx.ingest_corpus(
            corpus,
            default_subjects,
            doc_subjects=doc_subjects,
            rebuild_morph=False,
        )
        idx.build_ms = (time.perf_counter() - t0) * 1000.0
        return idx

    @classmethod
    def build_from_beir(
        cls,
        name: str,
        *,
        max_docs: int | None = None,
        download: bool = True,
        subjects: int | Sequence[int] | set[int] | None = None,
    ) -> SymbolKnowledgeIndex:
        corpus = load_beir_corpus_text(name, max_docs=max_docs, download=download)
        tags = (
            normalize_subjects(subjects)
            if subjects is not None
            else subjects_for_dataset(name)
        )
        return cls.build_from_corpus(corpus, dataset=name, subjects=tags)

    def ingest_corpus(
        self,
        corpus: dict[str, str],
        subjects: int | Sequence[int] | set[int] | None,
        *,
        doc_subjects: dict[str, set[int]] | None = None,
        rebuild_morph: bool = True,
        lazy_chambers: bool = True,
        rare_weight_ingest: bool | None = None,
    ) -> dict[str, object]:
        """
        Train correlation chambers: each doc writes to its subject set + master (k=0).

        Fast path (default): store each doc's pairs **once**, update master
        incrementally, mark subject chambers dirty — built on first read (same
        correlations, less ingest work and memory churn).

        ``subjects`` — corpus default (1 or more of 1..31).
        ``doc_subjects`` — optional per-doc override.
        """
        from aethos_symbol_subjects import infer_doc_subjects

        t0 = time.perf_counter()
        corpus_tags = normalize_subjects(subjects)
        chambers_touched: set[int] = set()
        docs_by_chamber: dict[int, int] = {}
        master_bucket = self._chamber_cooccur.setdefault(MASTER_CHAMBER, {})
        new_master_keys: set[tuple[str, str]] = set()

        for doc_id, text in corpus.items():
            self.corpus[doc_id] = text
            self.vocab.update(_TOKEN_RE.findall(text.lower()))
            if doc_subjects and doc_id in doc_subjects:
                doc_tags = normalize_subjects(doc_subjects[doc_id])
            elif corpus_tags:
                doc_tags = corpus_tags
            else:
                doc_tags = infer_doc_subjects(
                    text, fallback=corpus_tags or subjects_for_dataset(self.dataset),
                )
            active = chambers_for_ingest(doc_tags)
            pairs = _single_doc_cooccurrence_pairs(text)
            self._doc_evidence[doc_id] = DocEvidence(
                chambers=active - {MASTER_CHAMBER},
                pairs=pairs,
            )
            for key, count in pairs.items():
                master_bucket[key] = master_bucket.get(key, 0) + count
                new_master_keys.add(key)
            for chamber in active:
                chambers_touched.add(chamber)
                if chamber != MASTER_CHAMBER:
                    self._chamber_dirty.add(chamber)
                docs_by_chamber[chamber] = docs_by_chamber.get(chamber, 0) + 1

        if rebuild_morph:
            self._patch_morph_vocab()

        rare_on = self.ingest_rare_weight if rare_weight_ingest is None else rare_weight_ingest
        self._build_cross_links_incremental(
            MASTER_CHAMBER,
            new_direct_keys=new_master_keys,
            rare_boost=rare_on,
        )
        if not lazy_chambers:
            for chamber in chambers_touched:
                if chamber != MASTER_CHAMBER:
                    self._ensure_chamber(chamber)

        self._sync_master_cooccur()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "docs_ingested": len(corpus),
            "chambers_touched": sorted(chambers_touched),
            "chambers_built": sorted(
                k for k in chambers_touched
                if k in self.chamber_links and k not in self._chamber_dirty
            ),
            "chambers_lazy": sorted(self._chamber_dirty),
            "docs_by_chamber": docs_by_chamber,
            "subject_tags": sorted(corpus_tags),
            "ingest_ms": round(elapsed_ms, 2),
            "lazy_chambers": lazy_chambers,
        }

    def _patch_morph_vocab(self) -> None:
        """Extend morph registry with new vocab only (no full rebuild)."""
        new_morph_words = _morph_vocab_subset(self.vocab)
        extra = new_morph_words - set(self.morph.composites) - set(self.morph.subwords)
        if extra:
            patch = build_morph_registry(extra, max_composites=500)
            self.morph.subwords.update(patch.subwords)
            self.morph.composites.update(patch.composites)
            self.morph.correlations.update(patch.correlations)
        self._morph_links_shared = None
        self._family_of_cache = None

    def _aggregate_chamber_cooccur(self, chamber: int) -> dict[tuple[str, str], int]:
        """Reconstruct chamber co-occurrence from per-doc evidence (one copy of truth)."""
        if self._doc_evidence:
            out: dict[tuple[str, str], int] = {}
            for ev in self._doc_evidence.values():
                if chamber == MASTER_CHAMBER or chamber in ev.chambers:
                    for key, count in ev.pairs.items():
                        out[key] = out.get(key, 0) + count
            return out
        return dict(self._chamber_cooccur.get(chamber, {}))

    def _ensure_chamber(self, chamber: int) -> None:
        """Build subject chamber links on demand (lazy rotation view)."""
        if chamber not in self._chamber_dirty and self.chamber_links.get(chamber):
            return
        cooccur = self._aggregate_chamber_cooccur(chamber)
        self._chamber_cooccur[chamber] = cooccur
        self._build_cross_links(chamber, cooccur=cooccur, rare_boost=self.ingest_rare_weight)
        self._chamber_dirty.discard(chamber)

    def _sync_master_cooccur(self) -> None:
        """Keep legacy _cooccur_pairs in sync with master chamber."""
        self._cooccur_pairs = dict(
            self._chamber_cooccur.get(MASTER_CHAMBER, {}),
        )

    def _shared_morph_links(self, chamber: int) -> dict[tuple[str, str], CrossLink]:
        """Morph family links — identical across chambers, computed once."""
        if self._morph_links_shared is not None:
            return {
                k: CrossLink(
                    left=v.left,
                    right=v.right,
                    kind=v.kind,
                    strength=v.strength,
                    chamber=chamber,
                    via=v.via,
                    opposite=v.opposite,
                    intersection_imag=v.intersection_imag,
                )
                for k, v in self._morph_links_shared.items()
            }
        base: dict[tuple[str, str], CrossLink] = {}
        _MAX_MORPH_FAMILY = 24
        by_root_prime: dict[int, set[str]] = {}
        for word, comp in self.morph.composites.items():
            by_root_prime.setdefault(comp.correlation.root_prime, set()).add(word)
        for members in by_root_prime.values():
            if len(members) > _MAX_MORPH_FAMILY:
                continue
            mlist = sorted(members)
            for i, a in enumerate(mlist):
                for b in mlist[i + 1 :]:
                    self._add_link(
                        a, b, kind="morph", strength=0.75,
                        chamber=MASTER_CHAMBER, store=base,
                    )
        self._morph_links_shared = dict(base)
        return self._shared_morph_links(chamber)

    def _build_cross_links(
        self,
        chamber: int = MASTER_CHAMBER,
        *,
        cooccur: dict[tuple[str, str], int] | None = None,
        rare_boost: bool | None = None,
    ) -> None:
        """Direct pairs + shared morph + bridge for one chamber."""
        cooccur = cooccur if cooccur is not None else self._chamber_cooccur.get(chamber, {})
        links = self._shared_morph_links(chamber)
        rare_on = self.ingest_rare_weight if rare_boost is None else rare_boost
        df_cache = self._ingest_df_cache() if rare_on else None

        for key, count in cooccur.items():
            strength = float(count)
            if rare_on:
                strength = self._boost_ingest_strength(
                    key[0], key[1], strength, df_cache=df_cache, chamber=chamber,
                )
            self._add_link(
                key[0], key[1],
                kind="direct", strength=strength,
                chamber=chamber, store=links,
            )

        self._add_bridges_for_keys(
            chamber, list(cooccur.keys()), store=links, rare_boost=rare_on, df_cache=df_cache,
        )
        self.chamber_links[chamber] = links

    def _build_cross_links_incremental(
        self,
        chamber: int,
        *,
        new_direct_keys: set[tuple[str, str]],
        rare_boost: bool | None = None,
    ) -> None:
        """Update master chamber: merge new direct pairs + bridges only for new keys."""
        rare_on = self.ingest_rare_weight if rare_boost is None else rare_boost
        if not new_direct_keys:
            if chamber not in self.chamber_links:
                self._build_cross_links(chamber, rare_boost=rare_on)
            return
        links = self.chamber_links.get(chamber)
        if links is None:
            links = self._shared_morph_links(chamber)
        cooccur = self._chamber_cooccur.get(chamber, {})
        df_cache = self._ingest_df_cache() if rare_on else None
        for key in new_direct_keys:
            count = cooccur.get(key)
            if count is None:
                continue
            strength = float(count)
            if rare_on:
                strength = self._boost_ingest_strength(
                    key[0], key[1], strength, df_cache=df_cache, chamber=chamber,
                )
            self._add_link(
                key[0], key[1],
                kind="direct", strength=strength,
                chamber=chamber, store=links,
            )
        self._add_bridges_for_keys(
            chamber, list(new_direct_keys), store=links,
            rare_boost=rare_on, df_cache=df_cache,
        )
        self.chamber_links[chamber] = links

    def _ingest_df_cache(self) -> object:
        from aethos_rare_rank import _DocFreqCache
        return _DocFreqCache(self)

    def _boost_ingest_strength(
        self,
        left: str,
        right: str,
        base: float,
        *,
        via: str | None = None,
        df_cache: object | None = None,
        chamber: int = MASTER_CHAMBER,
    ) -> float:
        from aethos_rare_rank import ingest_link_strength
        return ingest_link_strength(
            self, left, right, base, via=via, df_cache=df_cache, chamber=chamber,
        )

    def _add_bridges_for_keys(
        self,
        chamber: int,
        direct_keys: list[tuple[str, str]],
        *,
        store: dict[tuple[str, str], CrossLink],
        rare_boost: bool | None = None,
        df_cache: object | None = None,
    ) -> None:
        rare_on = self.ingest_rare_weight if rare_boost is None else rare_boost
        family_of = self._word_families()
        for left, right in direct_keys:
            for sibling in family_of.get(left, {left}):
                if sibling != left:
                    strength = 0.5
                    if rare_on:
                        strength = self._boost_ingest_strength(
                            sibling, right, strength, via=left,
                            df_cache=df_cache, chamber=chamber,
                        )
                    self._add_link(
                        sibling, right, kind="bridge", strength=strength,
                        via=left, chamber=chamber, store=store,
                    )
            for sibling in family_of.get(right, {right}):
                if sibling != right:
                    strength = 0.5
                    if rare_on:
                        strength = self._boost_ingest_strength(
                            left, sibling, strength, via=right,
                            df_cache=df_cache, chamber=chamber,
                        )
                    self._add_link(
                        left, sibling, kind="bridge", strength=strength,
                        via=right, chamber=chamber, store=store,
                    )

    def _word_families(self) -> dict[str, set[str]]:
        if self._family_of_cache is not None:
            return self._family_of_cache
        out: dict[str, set[str]] = {}
        for word, comp in self.morph.composites.items():
            fam = out.setdefault(word, set())
            fam.add(word)
            for w in comp.correlation.words:
                fam.add(w)
        for corr in self.morph.correlations.values():
            for w in corr.words:
                fam = out.setdefault(w, set())
                fam.update(corr.words)
        self._family_of_cache = out
        return out

    def _add_link(
        self,
        left: str,
        right: str,
        *,
        kind: LinkKind,
        strength: float,
        chamber: int = MASTER_CHAMBER,
        via: str | None = None,
        opposite: bool = False,
        intersection_imag: int | None = None,
        store: dict[tuple[str, str], CrossLink] | None = None,
    ) -> None:
        key = tuple(sorted((left.lower(), right.lower())))
        bucket = store if store is not None else self.chamber_links.setdefault(chamber, {})
        existing = bucket.get(key)
        if existing is not None:
            rank = {"direct": 3, "morph": 2, "bridge": 1}
            if rank.get(kind, 0) < rank.get(existing.kind, 0):
                return
            if rank.get(kind, 0) == rank.get(existing.kind, 0) and strength <= existing.strength:
                return
        bucket[key] = CrossLink(
            left=key[0],
            right=key[1],
            kind=kind,
            strength=strength,
            chamber=chamber,
            via=via,
            opposite=opposite,
            intersection_imag=intersection_imag,
        )

    # ------------------------------------------------------------------
    # Query (pre-query knowledge — no question needed)
    # ------------------------------------------------------------------

    def ensure_query_lattice(self, word: str, plane=None):
        """Build and cache lattice node for OOV query token (no corpus ingest)."""
        from aethos_query_oov import QueryLatticeNode, build_query_lattice_node

        w = word.lower()
        if not getattr(self, "_oov_lattice", None):
            self._oov_lattice = {}
        cached = self._oov_lattice.get(w)
        if isinstance(cached, QueryLatticeNode):
            return cached
        node = build_query_lattice_node(self, w, plane)
        self._oov_lattice[w] = node
        return node

    def correlates(
        self,
        a: str,
        b: str,
        *,
        chamber: int | None = None,
    ) -> CrossLink | None:
        """Return known link in chamber (default: master k=0)."""
        key = tuple(sorted((a.lower(), b.lower())))
        if chamber is not None:
            self._ensure_chamber(chamber)
            return self.chamber_links.get(chamber, {}).get(key)
        self._ensure_chamber(MASTER_CHAMBER)
        return self.chamber_links.get(MASTER_CHAMBER, {}).get(key)

    def neighbors(
        self,
        word: str,
        *,
        kinds: set[LinkKind] | None = None,
        chamber: int | None = None,
    ) -> list[CrossLink]:
        w = word.lower()
        out: list[CrossLink] = []
        if chamber is not None:
            self._ensure_chamber(chamber)
            sources = [self.chamber_links.get(chamber, {})]
        else:
            self._ensure_chamber(MASTER_CHAMBER)
            sources = [self.chamber_links.get(MASTER_CHAMBER, {})]
        for bucket in sources:
            for link in bucket.values():
                if kinds and link.kind not in kinds:
                    continue
                if link.left == w or link.right == w:
                    out.append(link)
        out.sort(key=lambda lk: (-lk.strength, lk.kind))
        return out

    def subject_chambers_active(self) -> frozenset[int]:
        """Subject chambers with doc evidence (built or lazy)."""
        out: set[int] = set()
        for ev in self._doc_evidence.values():
            out |= set(ev.chambers)
        out |= {k for k in self.chamber_links if k != MASTER_CHAMBER}
        out |= set(self._chamber_dirty)
        return frozenset(out)

    def active_chambers_for_query(
        self,
        words: Sequence[str],
    ) -> frozenset[int]:
        """Vote subject chambers from query; always includes master."""
        from aethos_symbol_subjects import vote_query_chambers

        voted = vote_query_chambers(words)
        if voted:
            return voted | {MASTER_CHAMBER}
        return frozenset({MASTER_CHAMBER})

    def gap_words(self, *, min_len: int = 4) -> list[str]:
        """
        Signal vocabulary with **no** correlation edge yet.

        These are subjects the next training corpus should explain/link.
        Membrane filler (the, and, ed, …) is excluded.
        """
        linked: set[str] = set()
        for bucket in self.chamber_links.values():
            for lk in bucket.values():
                linked.add(lk.left)
                linked.add(lk.right)

        gaps: list[str] = []
        for w in sorted(self.vocab):
            if len(w) < min_len:
                continue
            if self.cellular.role_of(w).value == "membrane":
                continue
            if w not in linked and w not in self.morph.composites:
                gaps.append(w)
            elif w not in linked and w in self.morph.composites:
                gaps.append(w)
        return gaps

    def merge_corpus(
        self,
        corpus: dict[str, str],
        *,
        dataset_suffix: str = "+",
    ) -> SymbolKnowledgeIndex:
        """Extend knowledge with a new explanatory corpus; rebuild cross-links."""
        merged = dict(self.corpus)
        merged.update(corpus)
        new_vocab = set(self.vocab)
        for text in corpus.values():
            new_vocab.update(_TOKEN_RE.findall(text.lower()))
        morph = build_morph_registry(new_vocab)
        return SymbolKnowledgeIndex.build_from_corpus(
            merged,
            dataset=self.dataset + dataset_suffix,
            morph=morph,
        )

    def gaps_after_merge(self, new_corpus: dict[str, str]) -> dict[str, object]:
        """Compare gap words before vs after merging a training corpus."""
        before = set(self.gap_words())
        extended = self.merge_corpus(new_corpus)
        after = set(extended.gap_words())
        return {
            "before": len(before),
            "after": len(after),
            "newly_linked": sorted(before - after),
            "still_gaps": sorted(after),
        }

    def stack_corpus(
        self,
        new_corpus: dict[str, str],
        *,
        name: str = "",
        subjects: int | Sequence[int] | set[int] | None = None,
        doc_subjects: dict[str, set[int]] | None = None,
    ) -> dict[str, object]:
        """
        Add a new corpus on top — compound learn without erasing prior knowledge.

        Safe to call for every downloaded dataset (SciFact, NFCorpus, …).
        Pair counts accumulate per chamber; bridge layer re-derives from union.
        """
        suffix = f"+{name}" if name else "+"
        tags = (
            normalize_subjects(subjects)
            if subjects is not None
            else (subjects_for_dataset(name) if name else frozenset())
        )
        return self.compound_learn(
            new_corpus,
            dataset_suffix=suffix,
            subjects=tags,
            doc_subjects=doc_subjects,
        )

    def compound_learn(
        self,
        new_corpus: dict[str, str],
        *,
        dataset_suffix: str = "+",
        subjects: int | Sequence[int] | set[int] | None = None,
        doc_subjects: dict[str, set[int]] | None = None,
    ) -> dict[str, object]:
        """
        Lazy deepen: add docs into subject chambers + master without full rebuild.
        """
        before_keys = set(self.cross_links.keys())
        before_count = len(before_keys)

        ingest_report = self.ingest_corpus(
            new_corpus,
            subjects,
            doc_subjects=doc_subjects,
            rebuild_morph=True,
        )
        if dataset_suffix:
            self.dataset = f"{self.dataset}{dataset_suffix}"

        after_keys = set(self.cross_links.keys())
        added = after_keys - before_keys
        master_links = self.chamber_links.get(MASTER_CHAMBER, {})
        return {
            "docs_added": len(new_corpus),
            "links_before": before_count,
            "links_after": len(after_keys),
            "links_added": len(added),
            "chambers_touched": ingest_report.get("chambers_touched", []),
            "ingest_ms": ingest_report.get("ingest_ms", 0),
            "chambers_lazy": ingest_report.get("chambers_lazy", []),
            "new_link_samples": [
                {
                    "pair": list(k),
                    "kind": master_links[k].kind,
                    "strength": master_links[k].strength,
                    "chamber": master_links[k].chamber,
                }
                for k in sorted(added)[:30]
            ],
        }

    def master_audit(self, **kwargs) -> dict[str, object]:
        """Cross-chamber mis-correlation scan (subconscious k=0)."""
        from aethos_symbol_subjects import master_audit as _audit

        return _audit(self, **kwargs)

    def remembers(self, left: str, right: str) -> bool:
        """True when a correlation edge exists (any kind: direct, morph, bridge)."""
        return self.correlates(left, right) is not None

    def query_gold_links(
        self,
        query_words: Sequence[str],
        gold_doc_id: str,
    ) -> dict[str, object]:
        """
        Check query-to-gold linkage: do query terms correlate and does gold
        doc contain the pre-trained concepts?
        """
        qw = [w.lower() for w in query_words if len(w) >= 3]
        gold_text = self.corpus.get(gold_doc_id, "").lower()
        term_hits = {w: w in gold_text for w in qw}
        pair_hits: list[dict[str, object]] = []
        for i, a in enumerate(qw):
            for b in qw[i + 1 :]:
                lk = self.correlates(a, b)
                pair_hits.append({
                    "pair": [a, b],
                    "linked": lk is not None,
                    "kind": lk.kind if lk else None,
                    "strength": lk.strength if lk else 0,
                })
        return {
            "gold_doc_id": gold_doc_id,
            "gold_present": gold_doc_id in self.corpus,
            "term_in_gold": term_hits,
            "pair_links": pair_hits,
            "all_pairs_linked": all(p["linked"] for p in pair_hits) if pair_hits else False,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, int | float | str]:
        by_kind: dict[str, int] = {"direct": 0, "morph": 0, "bridge": 0}
        for lk in self.cross_links.values():
            by_kind[lk.kind] = by_kind.get(lk.kind, 0) + 1
        chamber_counts = {
            str(k): len(bucket) for k, bucket in sorted(self.chamber_links.items())
        }
        return {
            "dataset": self.dataset,
            "version": self.version,
            "n_docs": len(self.corpus),
            "vocab": len(self.vocab),
            "direct_pairs": by_kind["direct"],
            "morph_links": by_kind["morph"],
            "bridge_links": by_kind["bridge"],
            "total_cross_links": len(self.cross_links),
            "n_chambers": len(self.chamber_links),
            "chamber_link_counts": chamber_counts,
            "gap_signal_words": len(self.gap_words()),
            "build_ms": round(self.build_ms, 1),
        }

    def save(self, path: str | Path | None = None) -> Path:
        out = Path(path) if path else knowledge_path(self.dataset)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        meta = out.with_suffix(".json")
        meta.write_text(json.dumps(self.summary(), indent=2), encoding="utf-8")
        return out

    @classmethod
    def load(cls, dataset: str, path: str | Path | None = None) -> SymbolKnowledgeIndex:
        src = Path(path) if path else knowledge_path(dataset)
        if not src.is_file():
            raise FileNotFoundError(f"symbol knowledge not found: {src}")
        with open(src, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"expected SymbolKnowledgeIndex, got {type(obj)}")
        cls._migrate_chambers(obj)
        return obj

    @staticmethod
    def _migrate_chambers(obj: SymbolKnowledgeIndex) -> None:
        """Upgrade v1 flat cross_links → v2 chamber_links."""
        existing = getattr(obj, "chamber_links", None)
        if (
            getattr(obj, "version", 1) >= KNOWLEDGE_VERSION
            and isinstance(existing, dict)
            and existing
        ):
            return
        if not isinstance(existing, dict):
            obj.chamber_links = {}
        if not getattr(obj, "_chamber_cooccur", None):
            obj._chamber_cooccur = {}
        if not getattr(obj, "_doc_evidence", None):
            obj._doc_evidence = {}
        if not getattr(obj, "_chamber_dirty", None):
            obj._chamber_dirty = set()
        obj._morph_links_shared = getattr(obj, "_morph_links_shared", None)
        obj._family_of_cache = getattr(obj, "_family_of_cache", None)
        if not getattr(obj, "_oov_lattice", None):
            obj._oov_lattice = {}
        # Read legacy flat dict from instance __dict__ only — not the property.
        legacy = obj.__dict__.get("cross_links") if hasattr(obj, "__dict__") else None
        if isinstance(legacy, dict) and legacy and not obj.chamber_links:
            migrated: dict[tuple[str, str], CrossLink] = {}
            for key, lk in legacy.items():
                if isinstance(lk, CrossLink):
                    migrated[key] = CrossLink(
                        left=lk.left,
                        right=lk.right,
                        kind=lk.kind,
                        strength=lk.strength,
                        chamber=MASTER_CHAMBER,
                        via=lk.via,
                        opposite=lk.opposite,
                        intersection_imag=lk.intersection_imag,
                    )
            obj.chamber_links = {MASTER_CHAMBER: migrated}
        cooccur = getattr(obj, "_cooccur_pairs", {})
        if cooccur and not obj._chamber_cooccur:
            obj._chamber_cooccur = {MASTER_CHAMBER: dict(cooccur)}
        obj.version = KNOWLEDGE_VERSION
        if hasattr(obj, "__dict__") and "cross_links" in obj.__dict__:
            del obj.__dict__["cross_links"]


def knowledge_path(dataset: str) -> Path:
    root = Path(__file__).resolve().parent / "brains" / "symbol_knowledge"
    safe = dataset.replace("/", "_")
    return root / f"{safe}.pkl"


def ensure_beir_corpus(name: str, *, download: bool = True) -> Path:
    """Return path to corpus.jsonl; optionally download BEIR zip."""
    root = Path(resolve_beir_root())
    corpus_path = root / name / "corpus.jsonl"
    if corpus_path.is_file():
        return corpus_path
    if not download:
        raise FileNotFoundError(f"BEIR corpus missing: {corpus_path}")
    url = _BEIR_URLS.get(name)
    if not url:
        raise ValueError(f"no download URL for dataset {name!r}")
    root.mkdir(parents=True, exist_ok=True)
    try:
        from beir import util

        util.download_and_unzip(url, str(root))
    except ImportError:
        import urllib.request
        import zipfile
        import io

        print(f"  downloading {name} from BEIR ...", flush=True)
        with urllib.request.urlopen(url, timeout=120) as resp:
            data = resp.read()
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            zf.extractall(root)
    if not corpus_path.is_file():
        raise FileNotFoundError(f"download did not produce {corpus_path}")
    return corpus_path


def load_beir_corpus_text(
    name: str,
    *,
    max_docs: int | None = None,
    download: bool = True,
) -> dict[str, str]:
    """Load BEIR corpus as doc_id -> full text (title + body)."""
    from eval_beir import load_corpus

    corpus_path = ensure_beir_corpus(name, download=download)
    raw = load_corpus(corpus_path, max_docs=max_docs)
    out: dict[str, str] = {}
    for doc_id, fields in raw.items():
        title = (fields.get("title") or "").strip()
        text = (fields.get("text") or "").strip()
        out[doc_id] = f"{title} {text}".strip() if title else text
    return out


def demo() -> None:
    corpus = {
        "d1": "quantum mechanics zero dimension Hilbert space",
        "d2": "zero dimensional quantum systems exhibit unique behavior",
        "d3": "classical mechanics three dimensions",
    }
    idx = SymbolKnowledgeIndex.build_from_corpus(corpus, dataset="demo")
    print("=" * 60)
    print("SYMBOL KNOWLEDGE — pre-query cross-correlations")
    print("=" * 60)
    print(f"  summary: {idx.summary()}")

    for a, b in (("quantum", "zero"), ("quantum", "dimension"), ("zero", "dimension")):
        lk = idx.correlates(a, b)
        if lk:
            print(f"  {a!r} <-> {b!r}  kind={lk.kind}  strength={lk.strength}")
        else:
            print(f"  {a!r} <-> {b!r}  (no link — would need richer corpus)")

    print(f"\n  gap signal words ({len(idx.gap_words())}):")
    for w in idx.gap_words()[:12]:
        print(f"    {w!r}")

    path = idx.save()
    print(f"\n  saved: {path}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Build symbol knowledge from BEIR corpus")
    p.add_argument("--dataset", default="scifact")
    p.add_argument("--max-docs", type=int, default=None)
    p.add_argument("--no-download", action="store_true")
    p.add_argument("--demo", action="store_true")
    p.add_argument("--audit", action="store_true", help="write logs/chamber_conflicts.json")
    args = p.parse_args()

    if args.demo:
        demo()
    else:
        idx = SymbolKnowledgeIndex.build_from_beir(
            args.dataset,
            max_docs=args.max_docs,
            download=not args.no_download,
        )
        path = idx.save()
        print(f"symbol knowledge built: {idx.summary()}")
        print(f"saved: {path}")
        if args.audit:
            from aethos_symbol_subjects import write_master_audit

            audit_path = write_master_audit(idx)
            print(f"chamber audit: {audit_path}")
