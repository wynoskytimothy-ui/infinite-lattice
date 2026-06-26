"""
Doc lattice codec — Layer 0 placement + per-doc L4–L6 shell materialization.

Allocates ``doc_prime`` from an append-only odd-prime pool (same pattern as
``aethos_lattice_retrieval._allocate_doc_prime``).  Encodes text into
``(doc_prime, order_stream)`` using Stage-04 token-identity primes.

Builds ``TermCorrelationShell`` per document from sliding 3-way windows,
reusing Stage 07 observe_doc correlation logic scoped to one doc.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import combinations

from aethos_sequences import chain_primes

from lattice_retriever_v1.lattice2_correlation import (
    DocLatticePlacement,
    ShellNeighbor,
    TermCorrelationShell,
    ZeroShotTwoLatticeRetriever,
)
from lattice_retriever_v1.stage04_promote import Stage04Registry, promote_from_stream
from lattice_retriever_v1.stage06_composites import meet_composite_k
from lattice_retriever_v1.stage07_semantic_light import (
    CageIngestMode,
    SemanticLightIndex,
    anchor_composite,
    correlation_dims,
    rotation_quadrant_l4,
)

_TOKEN_RE = re.compile(r"[a-z]+")


@dataclass
class DocPrimePool:
    """Append-only doc prime allocator — mirrors aethos_lattice_retrieval."""

    pool_size: int = 100_000
    _primes: tuple[int, ...] = field(init=False, repr=False)
    _next_idx: int = 0
    doc_id_to_prime: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._primes = chain_primes(self.pool_size)

    def allocate(self, doc_id: str) -> int:
        if doc_id in self.doc_id_to_prime:
            return self.doc_id_to_prime[doc_id]
        if self._next_idx >= len(self._primes):
            raise RuntimeError("doc prime pool exhausted")
        p = self._primes[self._next_idx]
        self._next_idx += 1
        self.doc_id_to_prime[doc_id] = p
        return p


def _words(text: str) -> tuple[str, ...]:
    return tuple(_TOKEN_RE.findall(text.lower()))


def encode_doc(
    doc_id: str,
    text: str,
    registry: Stage04Registry,
    pool: DocPrimePool,
    *,
    semantic: SemanticLightIndex | None = None,
    corpus_prime: int | None = None,
) -> DocLatticePlacement:
    """
    Text → Layer-0 lattice placement.

    Returns doc_prime, order_stream of word-identity primes, corridor pins.
    When *corpus_prime* is set, doc_prime is the meet under the corpus spine.
    """
    sem = semantic or SemanticLightIndex(registry=registry)
    words = _words(text)
    order_stream = tuple(sem._prime_for_term(w) for w in words)
    pins: set[int] = set()
    for w in words:
        pins |= sem.corridor_pins_for_term(w)
    raw_prime = pool.allocate(doc_id)
    if corpus_prime is not None:
        try:
            doc_prime = meet_composite_k(corpus_prime, raw_prime)
        except ValueError:
            doc_prime = corpus_prime * raw_prime
    else:
        doc_prime = raw_prime
    return DocLatticePlacement(
        doc_id=doc_id,
        doc_prime=doc_prime,
        order_stream=order_stream,
        corridor_pins=frozenset(pins),
        words=words,
    )


def _cage_composite_for_triple(
    sem: SemanticLightIndex,
    w1: str,
    w2: str,
    w3: str,
) -> tuple[int, tuple[int, ...], str]:
    """Mirror stage07 _cage_for_triple composite logic."""
    p1, p2, p3 = sem._prime_for_term(w1), sem._prime_for_term(w2), sem._prime_for_term(w3)
    primes = (p1, p2, p3)
    if len(set(primes)) == 3:
        comp = anchor_composite(*primes)
    elif len(set(primes)) == 2:
        sp = sorted(set(primes))
        comp = sp[0] * sp[1]
    else:
        comp = p1
    label = f"{w1}|{w2}|{w3}"
    return comp, primes, label


def select_rare_in_doc(
    words: tuple[str, ...] | list[str],
    semantic: SemanticLightIndex,
    *,
    k: int = 8,
    max_df_frac: float = 0.05,
) -> tuple[str, ...]:
    """Up to *k* rarest in-doc terms, returned in document order."""
    n = semantic.n_docs
    max_df = max(1, int(n * max_df_frac))
    seen: set[str] = set()
    ordered_unique: list[str] = []
    for w in words:
        wl = w.lower()
        if not wl.isalpha() or wl in seen:
            continue
        seen.add(wl)
        ordered_unique.append(wl)
    rare_candidates = [
        w for w in ordered_unique if semantic.is_rare(w, max_df=max_df)
    ]
    if len(rare_candidates) <= k:
        return tuple(rare_candidates)
    top_by_idf = sorted(rare_candidates, key=lambda w: (-semantic.idf(w), w))[:k]
    selected = set(top_by_idf)
    return tuple(w for w in ordered_unique if w in selected)


def build_rare_combo_cages(
    words: tuple[str, ...] | list[str],
    semantic: SemanticLightIndex,
    registry: Stage04Registry,
    *,
    k_rare: int = 8,
    max_df_frac: float = 0.05,
) -> list[tuple[str, str, str]]:
    """
    Rare-term triple cages in document order — C(k,3) on rare-filtered subsequence.

    Non-adjacent rare terms (e.g. cat … pet … purr) still form a cage triple.
    """
    _ = registry  # reserved for future registry-scoped rare filters
    rare_ordered = select_rare_in_doc(
        words, semantic, k=k_rare, max_df_frac=max_df_frac
    )
    if len(rare_ordered) < 3:
        return []
    return [
        (rare_ordered[i], rare_ordered[j], rare_ordered[k])
        for i, j, k in combinations(range(len(rare_ordered)), 3)
    ]


def build_rare_correlation_shells(
    text: str,
    registry: Stage04Registry,
    semantic: SemanticLightIndex,
    *,
    k: int = 8,
    max_df_frac: float = 0.05,
    max_window: int = 6,
    mode: CageIngestMode = "positional",
) -> tuple[TermCorrelationShell, ...]:
    """Shell materialization on rare-term triple cages or positional rare subsequence."""
    words = _words(text)
    if mode == "rare_combo":
        return _shells_from_rare_combo_triples(
            words, registry, semantic, k=k, max_df_frac=max_df_frac
        )
    rare = set(select_rare_in_doc(words, semantic, k=k, max_df_frac=max_df_frac))
    if not rare:
        return ()
    rare_text = " ".join(w for w in words if w in rare)
    return build_doc_correlation_shells(
        rare_text, registry, max_window=max_window, mode="positional"
    )


def _shells_from_rare_combo_triples(
    words: tuple[str, ...],
    registry: Stage04Registry,
    semantic: SemanticLightIndex,
    *,
    k: int = 8,
    max_df_frac: float = 0.05,
) -> tuple[TermCorrelationShell, ...]:
    sem = semantic
    triples = build_rare_combo_cages(
        words, sem, registry, k_rare=k, max_df_frac=max_df_frac
    )
    if not triples:
        return ()
    shells: dict[int, TermCorrelationShell] = {}
    rare_ctx = tuple(dict.fromkeys(t for triple in triples for t in triple).keys())
    for triple in triples:
        comp, primes, label = _cage_composite_for_triple(sem, *triple)
        shell = shells.get(comp)
        if shell is None:
            anchor_p = primes[0]
            d4, d5, d6 = correlation_dims(anchor_p, comp, strength=1)
            shell = TermCorrelationShell(
                key=label,
                key_kind="anchor",
                anchor_composite=comp,
                anchor_primes=primes,
                dim4=d4,
                dim5=d5,
                dim6=d6,
            )
            shells[comp] = shell
        anchor_p = shell.anchor_primes[0] if shell.anchor_primes else 3
        for t in triple:
            _add_neighbor(shell, sem, t, source_prime=anchor_p, strength=1)
        for t in rare_ctx:
            if t in triple:
                continue
            _add_neighbor(shell, sem, t, source_prime=anchor_p, strength=1)
    for w in words:
        prime = sem._prime_for_term(w)
        if w not in {s.key for s in shells.values() if s.key_kind == "term"}:
            d4, d5, d6 = correlation_dims(prime, prime, strength=1)
            shells[prime] = TermCorrelationShell(
                key=w,
                key_kind="term",
                anchor_composite=prime,
                anchor_primes=(prime,),
                dim4=d4,
                dim5=d5,
                dim6=d6,
            )
    return tuple(shells.values())


def build_doc_correlation_shells(
    text: str,
    registry: Stage04Registry,
    *,
    max_window: int = 6,
    mode: CageIngestMode = "positional",
    semantic: SemanticLightIndex | None = None,
    k_rare: int = 8,
    max_df_frac: float = 0.05,
) -> tuple[TermCorrelationShell, ...]:
    """
    3-way cage shells per document.

    *positional* — sliding triples in first *max_window* tokens (legacy).
    *rare_combo* — C(k,3) on rare terms in document order.
    """
    words = _words(text)
    if not words:
        return ()
    if mode == "rare_combo":
        sem = semantic or SemanticLightIndex(registry=registry)
        return _shells_from_rare_combo_triples(
            words, registry, sem, k=k_rare, max_df_frac=max_df_frac
        )

    sem = semantic or SemanticLightIndex(registry=registry)
    word_list = [w for w in words if w.isalpha()]
    shells: dict[int, TermCorrelationShell] = {}
    window = word_list[:max_window]

    for i in range(len(window) - 2):
        triple = window[i : i + 3]
        comp, primes, label = _cage_composite_for_triple(sem, *triple)
        shell = shells.get(comp)
        if shell is None:
            anchor_p = primes[0]
            d4, d5, d6 = correlation_dims(anchor_p, comp, strength=1)
            shell = TermCorrelationShell(
                key=label,
                key_kind="anchor",
                anchor_composite=comp,
                anchor_primes=primes,
                dim4=d4,
                dim5=d5,
                dim6=d6,
            )
            shells[comp] = shell
        anchor_p = shell.anchor_primes[0] if shell.anchor_primes else 3
        for t in triple:
            _add_neighbor(shell, sem, t, source_prime=anchor_p, strength=1)
        for t in window:
            if t in triple:
                continue
            _add_neighbor(shell, sem, t, source_prime=anchor_p, strength=1)

    for w in word_list:
        prime = sem._prime_for_term(w)
        d4, d5, d6 = correlation_dims(prime, prime, strength=1)
        term_key = w
        if term_key not in {s.key for s in shells.values() if s.key_kind == "term"}:
            shells[prime] = TermCorrelationShell(
                key=term_key,
                key_kind="term",
                anchor_composite=prime,
                anchor_primes=(prime,),
                dim4=d4,
                dim5=d5,
                dim6=d6,
            )

    return tuple(shells.values())


def _add_neighbor(
    shell: TermCorrelationShell,
    sem: SemanticLightIndex,
    term: str,
    *,
    source_prime: int,
    strength: int,
) -> None:
    prime = sem._prime_for_term(term)
    d4, d5, d6 = correlation_dims(source_prime, prime, strength)
    rotation_quadrant_l4(source_prime, prime)
    existing = shell.neighbors.get(term)
    if existing:
        strength = existing.strength + strength
        d4, d5, d6 = correlation_dims(source_prime, prime, strength)
    shell.neighbors[term] = ShellNeighbor(
        term=term,
        prime=prime,
        strength=strength,
        dim4=d4,
        dim5=d5,
        dim6=d6,
    )


def build_two_lattice_retriever(
    corpus: dict[str, str],
    *,
    registry: Stage04Registry | None = None,
) -> ZeroShotTwoLatticeRetriever:
    """Index a corpus into the two-lattice retriever."""
    reg = registry or promote_from_stream(list(corpus.values()))
    pool = DocPrimePool()
    sem = SemanticLightIndex(registry=reg)
    retriever = ZeroShotTwoLatticeRetriever(semantic=sem)
    for doc_id, text in corpus.items():
        placement = encode_doc(doc_id, text, reg, pool, semantic=sem)
        shells = build_doc_correlation_shells(text, reg)
        retriever.index_placement(placement, shells)
    return retriever
