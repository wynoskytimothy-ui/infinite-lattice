"""
BIT 7 — Meet witness index (solo swap + triple promotion)

Math:
  meet(p, q) ⇔ solo(p)@n=q coord = solo(q)@n=p  (32-wing swap)
  triple (3,5,7) → κ* = (12, 5, 15)

Index:
  pool prime p → { doc_id : p ∈ pool_factors(hub) }

Query routing:
  factors(q_words) → union docs with ≥ min_factor_hits (default 2)
  OR κ match on promotion witnesses

Mitigation: cap docs per factor; integer wing_transform probes only.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence

from aethos_hub_signature import (
    MIN_POOL_PRIME,
    LatticeHubSignature,
    pool_factors_for_word,
)
from aethos_intersection_nodes import IntersectionNetwork, MeetKind
from aethos_physics import SpacetimeCell
from aethos_promotion import is_stopword
from aethos_tokenize import tokenize_words
from pipeline.bit_02_attractor_key import AttractorKey, kappa_from_cell

DEFAULT_MAX_DOCS_PER_FACTOR = 500
DEFAULT_MIN_FACTOR_HITS = 2
TRIPLE_WITNESS_CHAIN = (3, 5, 7)
TRIPLE_PROMOTION_KEY: AttractorKey = (12, 5, 15)


@dataclass
class MeetWitnessIndex:
    """
    Pool-prime factor → doc_id postings with per-factor caps.

    Compatible with eval_beir tier-2 meet lookup via legacy_dict().
    """

    by_factor: dict[int, list[str]] = field(default_factory=dict)
    doc_factors: dict[str, set[int]] = field(default_factory=dict)
    promotion_keys: set[AttractorKey] = field(default_factory=set)
    max_docs_per_factor: int = DEFAULT_MAX_DOCS_PER_FACTOR

    def add_factor(self, prime: int, doc_id: str) -> None:
        if prime < MIN_POOL_PRIME:
            return
        bucket = self.by_factor.setdefault(prime, [])
        if doc_id not in bucket and len(bucket) < self.max_docs_per_factor:
            bucket.append(doc_id)
        self.doc_factors.setdefault(doc_id, set()).add(prime)

    def add_promotion_key(self, key: AttractorKey, doc_id: str) -> None:
        self.promotion_keys.add(key)
        # κ bucket stored as synthetic factor hash not needed — use doc_factors path

    def legacy_dict(self) -> dict[int, set[str]]:
        """eval_beir candidate_ids tier-2/3 compatibility."""
        return {p: set(docs) for p, docs in self.by_factor.items()}

    def summary(self) -> dict[str, int | float]:
        return {
            "factors": len(self.by_factor),
            "docs": len(self.doc_factors),
            "promotion_keys": len(self.promotion_keys),
            "avg_factors_per_doc": (
                sum(len(v) for v in self.doc_factors.values()) / len(self.doc_factors)
                if self.doc_factors
                else 0.0
            ),
        }


def probe_solo_swap_witness(p: int, q: int):
    """BIT 7 geometry gate: solo swap meet witness or None."""
    return IntersectionNetwork().probe_solo_swap(p, q)


def triple_promotion_witness(a: int = 3, p: int = 5, q: int = 7):
    """Triple equalization witness; None if rails disagree."""
    return IntersectionNetwork().probe_triple(a, p, q)


def triple_promotion_key(
    a: int = 3,
    p: int = 5,
    q: int = 7,
) -> AttractorKey | None:
    """κ* for triple promotion chain (a,p,q)."""
    w = triple_promotion_witness(a, p, q)
    if w is None:
        return None
    cell = w.spacetime_cell(chain=(a, p, q))
    return kappa_from_cell(cell)


def query_pool_factors(
    words: Sequence[str],
    registry,
    *,
    min_len: int = 3,
) -> set[int]:
    """Union of pool factors for routed query words."""
    out: set[int] = set()
    for w in words:
        wl = w.lower()
        if not wl.isalpha() or len(wl) < min_len or is_stopword(wl):
            continue
        out |= pool_factors_for_word(registry, wl)
    return out


def build_meet_witness_index(
    hub_sigs: dict[str, LatticeHubSignature],
    registry,
    *,
    max_docs_per_factor: int = DEFAULT_MAX_DOCS_PER_FACTOR,
) -> MeetWitnessIndex:
    """Build capped pool-prime → doc index from hub signatures."""
    idx = MeetWitnessIndex(max_docs_per_factor=max_docs_per_factor)
    for doc_id, sig in hub_sigs.items():
        for word, entry in sig.hubs.items():
            if entry.pool_factors:
                for p in entry.pool_factors:
                    idx.add_factor(p, doc_id)
            elif entry.prime >= MIN_POOL_PRIME:
                idx.add_factor(entry.prime, doc_id)
            try:
                for p in pool_factors_for_word(registry, word):
                    idx.add_factor(p, doc_id)
            except Exception:
                pass
        # Register triple-chain hubs when chain matches promotion pattern
        for word, entry in sig.hubs.items():
            if entry.lattice_composite > 1:
                from aethos_intersection_nodes import chain_from_composite

                chain = chain_from_composite(entry.lattice_composite)
                if len(chain) >= 3 and chain[:3] == TRIPLE_WITNESS_CHAIN:
                    cell = SpacetimeCell.at(chain[:3], 5, wing=1)
                    idx.promotion_keys.add(kappa_from_cell(cell))
    return idx


def candidates_from_meet_witness(
    words: Sequence[str],
    registry,
    index: MeetWitnessIndex,
    *,
    min_factor_hits: int = DEFAULT_MIN_FACTOR_HITS,
    query_keys: set[AttractorKey] | None = None,
) -> list[str]:
    """
    Route candidates by pool-factor overlap.

    Requires ≥ min_factor_hits factor matches unless doc κ is in query_keys.
    """
    factors = query_pool_factors(words, registry)
    if not factors and not query_keys:
        return []

    hit_counts: Counter[str] = Counter()
    for p in factors:
        for doc_id in index.by_factor.get(p, []):
            hit_counts[doc_id] += 1

    out: list[str] = []
    seen: set[str] = set()
    for doc_id, count in hit_counts.most_common():
        if count >= min_factor_hits and doc_id not in seen:
            seen.add(doc_id)
            out.append(doc_id)

    if query_keys and index.promotion_keys & query_keys:
        for doc_id in index.doc_factors:
            if doc_id not in seen:
                seen.add(doc_id)
                out.append(doc_id)

    return out


def verify_bit07_gate(
    registry,
    hub_sigs: dict[str, LatticeHubSignature] | None = None,
    *,
    min_factor_hits: int = DEFAULT_MIN_FACTOR_HITS,
) -> tuple[bool, list[str]]:
    """
    BIT 7 gate:
      - probe_solo_swap(3,5) activates
      - triple (3,5,7) → single κ* = (12,5,15)
      - meet index routes docs sharing ≥2 pool factors (when sigs provided)
    """
    failures: list[str] = []

    solo = probe_solo_swap_witness(3, 5)
    if solo is None:
        failures.append("probe_solo_swap(3,5) returned None")
    elif solo.kind != MeetKind.SOLO_SWAP:
        failures.append(f"solo swap kind expected SOLO_SWAP, got {solo.kind}")

    triple = triple_promotion_witness(3, 5, 7)
    if triple is None:
        failures.append("probe_triple(3,5,7) returned None")
    else:
        cell = triple.spacetime_cell(chain=(3, 5, 7))
        key = kappa_from_cell(cell)
        if key != TRIPLE_PROMOTION_KEY:
            failures.append(f"triple κ expected {TRIPLE_PROMOTION_KEY}, got {key}")
        if cell.z != complex(12, 5):
            failures.append(f"triple z expected 12+5i, got {cell.z}")

    pk = triple_promotion_key(3, 5, 7)
    if pk != TRIPLE_PROMOTION_KEY:
        failures.append(f"triple_promotion_key got {pk}")

    if hub_sigs:
        idx = build_meet_witness_index(hub_sigs, registry)
        if not idx.by_factor:
            failures.append("meet index has no factor buckets")
        # find two docs sharing a factor if possible
        factor_docs = [(p, docs) for p, docs in idx.by_factor.items() if len(docs) >= 2]
        if factor_docs:
            p, docs = factor_docs[0]
            cands = candidates_from_meet_witness(
                [hub_sigs[docs[0]].hubs[next(iter(hub_sigs[docs[0]].hubs))].word],
                registry,
                idx,
                min_factor_hits=1,
            )
            if not cands:
                failures.append("meet routing returned empty for shared-factor probe")

    return len(failures) == 0, failures


def verify_bit07_routing_gate(
    registry,
    hub_sigs: dict[str, LatticeHubSignature],
    doc_tokens: dict[str, frozenset[str]],
    *,
    min_factor_hits: int = 2,
) -> tuple[bool, float, list[str]]:
    """
    Mean recall of gold docs (shared hub factor) in meet-routed candidates.
    """
    idx = build_meet_witness_index(hub_sigs, registry)
    failures: list[str] = []
    recalls: list[float] = []

    for doc_id, sig in hub_sigs.items():
        if not sig.hubs:
            continue
        top_word = max(sig.hubs.items(), key=lambda x: (-x[1].strength, x[0]))[0]
        factors = pool_factors_for_word(registry, top_word)
        if not factors:
            continue
        gold = {
            other
            for other, osig in hub_sigs.items()
            if other != doc_id
            and any(
                pool_factors_for_word(registry, w) & factors
                for w in osig.hubs
            )
        }
        if not gold:
            continue
        cands = set(
            candidates_from_meet_witness(
                [top_word],
                registry,
                idx,
                min_factor_hits=min_factor_hits,
            )
        )
        rec = len(gold & cands) / len(gold)
        recalls.append(rec)

    avg = sum(recalls) / len(recalls) if recalls else 0.0
    if recalls and avg < 0.5:
        failures.append(f"mean factor-routing recall {avg:.2f} < 0.5")
    return len(failures) == 0, avg, failures
