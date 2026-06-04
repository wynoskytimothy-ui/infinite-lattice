"""
Recursive learning layer — bad correlations as subconscious promotion hypotheses.

The brain stores distilled primes (promoted structure), not raw co-occurrence.
Bad correlations are *unresolved* geometric mismatches: they accumulate signal
until a new pool prime explains them (shares context factors with the misfire).

King/queen arithmetic is factor-set algebra on composites, not embedding math.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from core.phi_lattice import (
    LatticeId,
    compute_coordinates,
    prime_factor_similarity,
    prime_factors,
    swap_meet_all_wings,
)
from core.primes import chain_primes

# ---------------------------------------------------------------------------
# King/queen fixture primes (promoted semantic factors for proofs)
# ---------------------------------------------------------------------------

ROYAL = 1031
MALE = 1039
FEMALE = 1049
HUMAN = 1051

KING = ROYAL * MALE
QUEEN = ROYAL * FEMALE
MAN = MALE
WOMAN = FEMALE

# Species chains for cross-lattice coordinate variation (same prime, different geometry)
PRIME_CHAIN = chain_primes(8)
FIB_CHAIN = (2, 3, 5, 8, 13, 21, 34, 55)  # illustrative second anchor species


# ---------------------------------------------------------------------------
# Factor-set analogy  (king - man + woman = queen)
# ---------------------------------------------------------------------------

def _factor_counter(n: int) -> Counter[int]:
    c: Counter[int] = Counter()
    for p in prime_factors(n):
        c[p] += 1
    return c


def counter_to_composite(c: Counter[int]) -> int:
    out = 1
    for p, exp in sorted(c.items()):
        if exp > 0:
            out *= int(p) ** int(exp)
    return out


def factor_analogy(base: int, subtract: int, add: int) -> int:
    """
    Multiset factor arithmetic: factors(base) - factors(subtract) + factors(add).

    Example: king - man + woman → queen when base=KING, subtract=MAN, add=WOMAN.
    """
    acc = _factor_counter(base)
    for p, e in _factor_counter(subtract).items():
        acc[p] -= e
    for p, e in _factor_counter(add).items():
        acc[p] += e
    acc = Counter({p: e for p, e in acc.items() if e > 0})
    return counter_to_composite(acc)


# ---------------------------------------------------------------------------
# Bad correlation store (subconscious signal queue)
# ---------------------------------------------------------------------------

def _pair_key(a: str, b: str) -> tuple[str, str]:
    la, lb = a.lower(), b.lower()
    return (la, lb) if la <= lb else (lb, la)


@dataclass
class BadCorrelation:
    """Query word ↔ hub word misfire on a false-positive retrieval."""

    word_a: str
    word_b: str
    context_primes: frozenset[int]
    signal_strength: float = 0.0
    fire_count: int = 0
    resolved: bool = False
    resolving_prime: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "word_a": self.word_a,
            "word_b": self.word_b,
            "context_primes": sorted(self.context_primes),
            "signal_strength": self.signal_strength,
            "fire_count": self.fire_count,
            "resolved": self.resolved,
            "resolving_prime": self.resolving_prime,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BadCorrelation:
        return cls(
            word_a=d["word_a"],
            word_b=d["word_b"],
            context_primes=frozenset(int(x) for x in d.get("context_primes", [])),
            signal_strength=float(d.get("signal_strength", 0.0)),
            fire_count=int(d.get("fire_count", 0)),
            resolved=bool(d.get("resolved", False)),
            resolving_prime=(
                int(d["resolving_prime"]) if d.get("resolving_prime") is not None else None
            ),
        )


@dataclass
class BadCorrelationStore:
    """
    Hypothesis queue: each false positive strengthens a (word_a, word_b) entry.

    ``signal_strength`` grows with log-scaled fire count × context mass so
    chronic misfires surface first for the next promotion pass.
    """

    entries: dict[tuple[str, str], BadCorrelation] = field(default_factory=dict)
    strength_per_fire: float = 0.35
    context_log_weight: float = 0.12

    def record(
        self,
        word_a: str,
        word_b: str,
        context_primes: Iterable[int],
        *,
        extra_strength: float = 0.0,
    ) -> BadCorrelation:
        key = _pair_key(word_a, word_b)
        ctx = frozenset(int(p) for p in context_primes if int(p) >= 107)
        if key in self.entries:
            bc = self.entries[key]
            bc.fire_count += 1
            bc.context_primes = frozenset(bc.context_primes | ctx)
        else:
            bc = BadCorrelation(word_a=key[0], word_b=key[1], context_primes=ctx)
            bc.fire_count = 1
            self.entries[key] = bc
        ctx_mass = sum(math.log1p(float(p) - 106.0) for p in bc.context_primes)
        bc.signal_strength += (
            self.strength_per_fire * math.log1p(bc.fire_count) + self.context_log_weight * ctx_mass
        )
        bc.signal_strength += extra_strength
        return bc

    def top_unresolved(self, n: int = 10) -> list[BadCorrelation]:
        pool = [e for e in self.entries.values() if not e.resolved]
        pool.sort(key=lambda e: e.signal_strength, reverse=True)
        return pool[:n]

    def try_resolve(
        self,
        new_prime: int,
        new_prime_context: Iterable[int],
        *,
        min_shared: int = 2,
    ) -> list[BadCorrelation]:
        """
        When a promoted prime shares ≥ min_shared pool factors with a bad
        correlation's context, treat the misfire as explained.
        """
        ctx_new = frozenset(int(p) for p in new_prime_context if int(p) >= 107)
        ctx_new = frozenset(ctx_new | {int(new_prime)})
        resolved: list[BadCorrelation] = []
        for bc in self.entries.values():
            if bc.resolved:
                continue
            if len(bc.context_primes & ctx_new) >= min_shared:
                bc.resolved = True
                bc.resolving_prime = int(new_prime)
                resolved.append(bc)
        return resolved

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        doc = {
            "version": 1,
            "entries": [bc.to_dict() for bc in self.entries.values()],
        }
        p.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> BadCorrelationStore:
        p = Path(path)
        if not p.exists():
            return cls()
        doc = json.loads(p.read_text(encoding="utf-8"))
        store = cls()
        for d in doc.get("entries", []):
            bc = BadCorrelation.from_dict(d)
            store.entries[_pair_key(bc.word_a, bc.word_b)] = bc
        return store


def context_primes_for_word(registry, word: str) -> frozenset[int]:
    """Pool primes (≥107) in a word's L3 prime + parent chain."""
    try:
        tok = registry.resolve_token(word.lower())
    except Exception:
        return frozenset()
    out: set[int] = set()
    if tok.prime >= 107:
        out.add(tok.prime)
    for p in tok.parent_primes:
        if p >= 107:
            out.add(p)
    return frozenset(out)


def context_primes_for_pair(registry, word_a: str, word_b: str) -> frozenset[int]:
    return context_primes_for_word(registry, word_a) | context_primes_for_word(registry, word_b)


def record_retrieval_false_positives(
    store: BadCorrelationStore,
    ranked: Sequence[str],
    relevant: set[str],
    profile,
    hub_sigs: dict,
    registry,
    *,
    top_k: int = 10,
    max_pairs_per_doc: int = 8,
) -> int:
    """
    Record bad correlations for high-ranked non-relevant docs.

    Pairs query words with hub words on the false-positive document (skipping
    direct query↔hub identity matches). Returns number of record() calls.
    """
    n_recorded = 0
    for did in ranked[:top_k]:
        if did in relevant:
            continue
        sig = hub_sigs.get(did)
        if sig is None:
            continue
        hubs = list(sig.hubs.keys())[: max_pairs_per_doc * 2]
        pairs_done = 0
        for qw in profile.word_set:
            if pairs_done >= max_pairs_per_doc:
                break
            for hw in hubs:
                if hw in profile.word_set:
                    continue
                ctx = context_primes_for_pair(registry, qw, hw)
                store.record(qw, hw, ctx)
                n_recorded += 1
                pairs_done += 1
    return n_recorded


# ---------------------------------------------------------------------------
# Distilled registry brain (promoted primes only — no raw observations)
# ---------------------------------------------------------------------------

def distilled_registry_to_dict(registry) -> dict[str, Any]:
    """Serialize promoted tokens + intersections only (no counts/correlations)."""
    from aethos_promotion import LatticeTier

    promoted = {}
    for (tier, text), tok in registry.promoted.items():
        if int(tier) < int(LatticeTier.L3_WORD):
            continue
        promoted[text] = {
            "tier": int(tier),
            "prime": tok.prime,
            "parent_primes": list(tok.parent_primes),
            "intersection_only": tok.intersection_only,
        }
    intersections = {
        w: {
            "prime": t.prime,
            "parent_primes": list(t.parent_primes),
            "intersection_only": t.intersection_only,
        }
        for w, t in registry.intersections.items()
    }
    return {"version": 1, "promoted": promoted, "intersections": intersections}


def distilled_registry_from_dict(doc: dict[str, Any], registry) -> None:
    """Load distilled primes into an existing registry (merge, no counts)."""
    from aethos_promotion import LatticeTier, PromotedToken

    for text, d in doc.get("promoted", {}).items():
        tier = LatticeTier(int(d.get("tier", LatticeTier.L3_WORD)))
        tok = PromotedToken(
            text=text,
            tier=tier,
            prime=int(d["prime"]),
            parent_primes=tuple(d.get("parent_primes", [])),
            intersection_only=bool(d.get("intersection_only", False)),
        )
        registry.promoted[(tier, text)] = tok
    for w, d in doc.get("intersections", {}).items():
        registry.intersections[w] = PromotedToken(
            text=w,
            tier=LatticeTier.L1_SYMBOL,
            prime=int(d["prime"]),
            parent_primes=tuple(d.get("parent_primes", [])),
            intersection_only=bool(d.get("intersection_only", True)),
        )


def save_distilled_registry(registry, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(distilled_registry_to_dict(registry), indent=2), encoding="utf-8")
    return p


def load_distilled_registry(registry, path: str | Path) -> bool:
    p = Path(path)
    if not p.exists():
        return False
    distilled_registry_from_dict(json.loads(p.read_text(encoding="utf-8")), registry)
    return True


def distilled_registry_path(dataset: str, mode: str = "quality") -> Path:
    root = Path(__file__).resolve().parent.parent / "brains"
    return root / f"{dataset}_{mode}.distilled.json"


def bad_correlation_path(dataset: str, mode: str = "quality") -> Path:
    root = Path(__file__).resolve().parent.parent / "brains"
    return root / f"{dataset}_{mode}.bad_corr.json"


# ---------------------------------------------------------------------------
# Cross-lattice consensus (32 wings)
# ---------------------------------------------------------------------------

def pf_similarity_all_wings(c1: int, c2: int) -> tuple[float, int]:
    """
    Factor Jaccard is wing-invariant; return (similarity, n_swap_meet_wings).

    The second count is geometric consensus via swap_meet witnesses on factor
    primes extracted from composites (when composites are pool-prime products).
    """
    sim = prime_factor_similarity(c1, c2)
    wings = 0
    shared = prime_factors(c1) & prime_factors(c2)
    for p in shared:
        if p < 107:
            continue
        for q in shared:
            if q <= p:
                continue
            hits = swap_meet_all_wings(int(p), int(q))
            if hits:
                wings = max(wings, len(hits))
    return sim, wings


def coord_with_anchor_chain(
    prime: int,
    anchor_n: int,
    species_anchor: int,
    lid: LatticeId = LatticeId.L01,
) -> tuple[int, int, int]:
    """Same pool prime, different species anchor → different wing geometry."""
    return compute_coordinates((species_anchor, prime), anchor_n, lid)


def coordinate_variation_across_species(prime: int, anchor_n: int = 7) -> int:
    """Count distinct L01 coords when species anchor prime changes."""
    coords = {
        coord_with_anchor_chain(prime, anchor_n, PRIME_CHAIN[0]),
        coord_with_anchor_chain(prime, anchor_n, FIB_CHAIN[3]),
    }
    return len(coords)


def consensus_factor_agreement(c1: int, c2: int, *, min_sim: float = 0.3) -> bool:
    sim, _ = pf_similarity_all_wings(c1, c2)
    return sim >= min_sim


# ---------------------------------------------------------------------------
# Promotion hypothesis from bad-correlation queue
# ---------------------------------------------------------------------------

def promotion_candidates_from_store(
    store: BadCorrelationStore,
    *,
    min_strength: float = 1.0,
    limit: int = 20,
) -> list[BadCorrelation]:
    """Top unresolved misfires above strength threshold (next-pass promotion hints)."""
    return [bc for bc in store.top_unresolved(limit) if bc.signal_strength >= min_strength]
