"""
Stage 08 — Lazy corridor retrieve (Stages 01–07 stack).

Phase A — inverted index on token identities (corridor-opening pins):
  rarest query term first → intersect postings → bounded pool.

Miss policy (locked — Course 1 §7 zero-recall):
  1. Primary: promoted/pool token-identity intersect (rarest first)
  2. Fallback widen: rarest single-term union if intersect empty
  3. FTA letter fallback: L1 letter-prime postings for OOV query terms
  4. Empty: no corridor opened — trace explains the miss (never full-corpus scan)

Phase B — lazy corridor score on pool only:
  identity overlap + wing-cage L4–L6 touch (Stage 07).
  Corridor keys built from primes + quadrant at query time — no stored rows.

Gate: synthetic properties first; SciFact quality baseline second (see BUILD.md).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

from aethos_words import letter_to_prime

from lattice_retriever_v1.stage03_rotation import rotate_token
from lattice_retriever_v1.stage05_free_token import (
    FreeTokenAddress,
    corridor_witness_explain,
    decode_corridor,
    decode_corridor_address,
    free_token_address,
    meet_composite,
    oriented_corridor_pin,
    oriented_corridor_key,
)
from lattice_retriever_v1.k_meet import velocity_meet
from lattice_retriever_v1.k_meet_index import (
    query_primes_from_terms,
    velocity_witness_primes,
    widen_pins_from_velocity,
)
from lattice_retriever_v1.stage06_composites import meet_composite_k
from lattice_retriever_v1.stage07_semantic_light import (
    HUB_WORDS,
    SemanticLightIndex,
    WingCage,
    build_demo_registry,
)

_TOKEN_RE = re.compile(r"[a-z]+")

RouteMode = Literal[
    "primary",
    "widen_rarest",
    "lift_pin_widen",
    "k_meet_velocity_widen",
    "fta_letter_fallback",
    "empty",
]


class MissPolicy(Enum):
    """Behavior when primary rarest-prime route lights nothing."""

    FTA_LETTER_FALLBACK = "fta_letter_fallback"
    EMPTY = "empty"
    WIDEN_RAREST_ONLY = "widen_rarest_only"


@dataclass(frozen=True)
class DocRecord:
    """Corridor pins per doc — identities computed, not float tables."""

    doc_id: str
    text: str
    words: tuple[str, ...]
    token_identities: tuple[int, ...]
    corridor_pins: tuple[int, ...] = ()


@dataclass(frozen=True)
class QueryPrimeFactor:
    """Glass-box: how each query term factored to a corridor pin."""

    term: str
    token_identity: int
    letter_primes: tuple[int, ...]
    corridor_pins: tuple[int, ...]
    doc_freq: int
    in_corpus: bool
    rare: bool


@dataclass(frozen=True)
class RetrieveHit:
    doc_id: str
    score: float
    reasons: tuple[dict, ...]

    def explain(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "score": round(self.score, 6),
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class RetrieveTrace:
    """Full glass-box retrieval trace — diagnostic surface for tuning."""

    query: str
    route_mode: RouteMode
    query_primes: tuple[QueryPrimeFactor, ...]
    filter_steps: tuple[dict, ...]
    cages_considered: tuple[dict, ...]
    rare_dots: tuple[dict, ...]
    pool_size: int
    hits: tuple[RetrieveHit, ...]
    corridor_witnesses: tuple[dict, ...] = ()
    velocity_witness: tuple[dict, ...] = ()

    def explain(self) -> dict:
        out = {
            "query": self.query,
            "route_mode": self.route_mode,
            "pool_size": self.pool_size,
            "corridor_witnesses": list(self.corridor_witnesses),
            "query_primes": [
                {
                    "term": q.term,
                    "token_identity": q.token_identity,
                    "letter_primes": list(q.letter_primes),
                    "corridor_pins": list(q.corridor_pins),
                    "doc_freq": q.doc_freq,
                    "in_corpus": q.in_corpus,
                    "rare": q.rare,
                }
                for q in self.query_primes
            ],
            "filter_steps": list(self.filter_steps),
            "cages_considered": list(self.cages_considered),
            "rare_dots": list(self.rare_dots),
            "hits": [h.explain() for h in self.hits],
        }
        if self.velocity_witness:
            out["velocity_witness"] = list(self.velocity_witness)
        return out


@dataclass
class LatticeRetriever:
    """
    End-to-end glass-box retriever on the prime lattice stack.

    Combines Stage 04 registry + Stage 07 wing cages + lazy corridor keys.
    """

    semantic: SemanticLightIndex = field(default_factory=lambda: SemanticLightIndex(registry=build_demo_registry()))
    docs: dict[str, DocRecord] = field(default_factory=dict)
    postings: dict[int, set[str]] = field(default_factory=dict)
    letter_postings: dict[int, set[str]] = field(default_factory=dict)
    pin_doc_freq: dict[int, int] = field(default_factory=dict)
    miss_policy: MissPolicy = MissPolicy.FTA_LETTER_FALLBACK
    pin_selectivity_max_df_frac: float = 0.05
    standalone_hub_max_df_frac: float = 0.10
    enable_lift_pin_fallback: bool = True
    lift_pin_max_pool_frac: float = 0.20
    enable_compound_first_routing: bool = True
    enable_k_meet_velocity_widen: bool = True
    k_meet_min_pool_size: int = 1
    enable_lexical_bridge_rerank: bool = True
    bridge_rerank_lambda: float = 1.0
    bridge_rerank_anchors: int = 1
    # When >0, bridge boost applies only if top1_score - doc_score <= frac * top1_score.
    bridge_rerank_tiebreak_frac: float = 0.0
    enable_cage_anchor_rerank: bool = True
    cage_anchor_rerank_lambda: float = 1.0
    cage_anchor_rerank_tiebreak_frac: float = 0.0

    def _max_selective_pin_df(self) -> int:
        n = max(self.semantic.n_docs, 1)
        return max(int(n * self.pin_selectivity_max_df_frac), 3)

    def _max_standalone_hub_df(self) -> int:
        n = max(self.semantic.n_docs, 1)
        return max(int(n * self.standalone_hub_max_df_frac), 10)

    def _identity_pin(self, term: str) -> int:
        return self._identity_for(term)

    def _compound_pin(self, w1: str, w2: str) -> int:
        """Oriented corridor posting pin — read-order + quadrant + invoke direction."""
        return oriented_corridor_pin(self._corridor_address(w1, w2))

    def _compound_address(self, w1: str, w2: str) -> FreeTokenAddress:
        return self._corridor_address(w1, w2)

    def _pin_selective_for_narrowing(self, pin: int, *, identity_pin: int) -> bool:
        """Hub-class pins (high df) lift only — never narrow. Identity always narrows."""
        if pin == identity_pin:
            return True
        return self.pin_doc_freq.get(pin, 0) <= self._max_selective_pin_df()

    def _term_narrows(self, term: str) -> bool:
        """Standalone high-df terms stay in library but do not narrow the pool."""
        if term in HUB_WORDS:
            return False
        return self.semantic.doc_freq.get(term, 0) <= self._max_standalone_hub_df()

    def _split_pins(self, term: str) -> tuple[int, frozenset[int], frozenset[int]]:
        """Identity + selective narrow pins + promiscuous lift-only pins."""
        pins = self._corridor_pins(term)
        identity = self._identity_pin(term)
        selective = frozenset(
            p for p in pins if self._pin_selective_for_narrowing(p, identity_pin=identity)
        )
        lift = pins - selective
        return identity, selective, frozenset(lift)

    def _routing_pin_for_term(self, term: str) -> tuple[int, frozenset[int], frozenset[int]]:
        """Min-df pin among selective set — never union of all pins."""
        identity, selective, lift = self._split_pins(term)
        if not selective:
            selective = frozenset({identity})
        routing = min(selective, key=lambda p: (self.pin_doc_freq.get(p, 0), p))
        return routing, selective, lift

    def _corridor_pins(self, word: str) -> frozenset[int]:
        return self.semantic.corridor_pins_for_term(word)

    def _postings_for_routing_pin(self, pin: int) -> set[str]:
        return set(self.postings.get(pin, set()))

    def _postings_for_term(self, term: str) -> set[str]:
        routing, _, _ = self._routing_pin_for_term(term)
        return self._postings_for_routing_pin(routing)

    def _lift_bucket_for_term(self, term: str) -> tuple[set[str], list[dict]]:
        """Union all lift pins for one term — min single lift may miss gold."""
        _, _, lift = self._split_pins(term)
        bucket: set[str] = set()
        entries: list[dict] = []
        for lp in sorted(lift, key=lambda p: (self.pin_doc_freq.get(p, 0), p)):
            b = self._postings_for_routing_pin(lp)
            bucket |= b
            entries.append(
                {
                    "term": term,
                    "lift_pin": lp,
                    "pin_doc_freq": self.pin_doc_freq.get(lp, 0),
                    "bucket_size": len(b),
                }
            )
        return bucket, entries

    def _lift_pin_widen_pool(self, query: str) -> tuple[set[str], list[dict]]:
        """
        When selective intersect fails: intersect per-term lift unions (all lift pins).
        Falls back to capped union of per-term buckets; never abort-to-empty on cap.
        """
        max_pool = max(int(self.semantic.n_docs * self.lift_pin_max_pool_frac), 50)
        terms = [t for t in self._routed_terms(self._query_terms(query)) if self._term_narrows(t)]
        meta: list[dict] = []

        term_buckets: list[tuple[str, set[str], list[dict]]] = []
        for term in terms:
            bucket, entries = self._lift_bucket_for_term(term)
            if bucket:
                term_buckets.append((term, bucket, entries))

        # Strategy 1: intersect per-term lift unions (skip terms with no lift pins)
        pool: set[str] | None = None
        for _term, bucket, entries in term_buckets:
            meta.extend(entries)
            pool = bucket if pool is None else pool & bucket
        if pool and 0 < len(pool) <= max_pool:
            return pool, meta

        # Strategy 2: capped union of per-term buckets — smallest first
        meta = []
        term_buckets.sort(key=lambda x: len(x[1]))
        union_pool: set[str] = set()
        for _term, bucket, entries in term_buckets:
            candidate = union_pool | bucket
            meta.extend(entries)
            if len(candidate) > max_pool and union_pool:
                break
            union_pool = candidate
        if union_pool and len(union_pool) <= max_pool:
            return union_pool, meta

        # Strategy 3: single smallest per-term lift union within cap
        for _term, bucket, entries in term_buckets:
            if bucket and len(bucket) <= max_pool:
                return set(bucket), entries

        return set(), meta

    def _k_meet_velocity_widen_pool(
        self,
        query: str,
        existing_pins: frozenset[int],
    ) -> tuple[set[str], list[dict], tuple[dict, ...]]:
        """Union postings for velocity-derived composite pins (conservative k≥3)."""
        terms = self._query_terms(query)
        q_primes = query_primes_from_terms(tuple(terms), self._identity_for)
        witnesses = tuple(velocity_witness_primes(q_primes))
        widen = widen_pins_from_velocity(q_primes, existing_pins=existing_pins)
        if not widen:
            return set(), [], witnesses

        pool: set[str] = set()
        meta: list[dict] = []
        for pin in sorted(widen):
            bucket = set(self.postings.get(pin, set()))
            pool |= bucket
            meta.append(
                {
                    "pin": pin,
                    "pin_doc_freq": self.pin_doc_freq.get(pin, 0),
                    "bucket_size": len(bucket),
                }
            )
        return pool, meta, witnesses

    def _words(self, text: str) -> tuple[str, ...]:
        return tuple(_TOKEN_RE.findall(text.lower()))

    def _identity_for(self, word: str) -> int:
        return self.semantic._prime_for_term(word)

    def _letter_primes(self, word: str) -> tuple[int, ...]:
        return tuple(letter_to_prime(c) for c in word.lower() if c.isalpha())

    def _factor_query_term(self, term: str) -> QueryPrimeFactor:
        df = self.semantic.doc_freq.get(term, 0)
        pins = tuple(sorted(self._corridor_pins(term)))
        return QueryPrimeFactor(
            term=term,
            token_identity=self._identity_for(term),
            letter_primes=self._letter_primes(term),
            corridor_pins=pins,
            doc_freq=df,
            in_corpus=df > 0,
            rare=self.semantic.is_rare(term),
        )

    def index_doc(
        self,
        doc_id: str,
        text: str,
        *,
        cage_ingest_mode: Literal["positional", "rare_combo"] = "positional",
        k_rare: int = 8,
        max_df_frac: float = 0.05,
    ) -> None:
        words = self._words(text)
        ids = tuple(self._identity_for(w) for w in words)
        pins_in_doc: set[int] = set()
        for w in words:
            pins_in_doc |= self._corridor_pins(w)
        for i in range(len(words) - 1):
            pins_in_doc.add(self._compound_pin(words[i], words[i + 1]))
        for i in range(len(words) - 2):
            ids = tuple(self._identity_for(words[i + j]) for j in range(3))
            ps = tuple(sorted(set(ids)))
            if len(ps) >= 3:
                vel = velocity_meet(*ps)
                if vel is not None and vel.unified:
                    try:
                        pins_in_doc.add(meet_composite_k(*ps))
                    except ValueError:
                        pass
        self.docs[doc_id] = DocRecord(
            doc_id=doc_id,
            text=text,
            words=words,
            token_identities=ids,
            corridor_pins=tuple(sorted(pins_in_doc)),
        )
        for pin in pins_in_doc:
            self.postings.setdefault(pin, set()).add(doc_id)
            self.pin_doc_freq[pin] = self.pin_doc_freq.get(pin, 0) + 1
        for w in words:
            for lp in set(self._letter_primes(w)):
                self.letter_postings.setdefault(lp, set()).add(doc_id)
        self.semantic.observe_doc(
            text,
            mode=cage_ingest_mode,
            k_rare=k_rare,
            max_df_frac=max_df_frac,
        )

    def index_corpus(self, corpus: dict[str, str]) -> None:
        for doc_id, text in corpus.items():
            self.index_doc(doc_id, text)

    def _query_terms(self, query: str) -> list[str]:
        return [w for w in self._words(query) if len(w) >= 2]

    def _ordered_query_words(self, query: str) -> list[str]:
        return [w for w in self._words(query) if len(w) >= 2]

    def _routing_units(self, query: str) -> list[dict]:
        """
        Narrowing units: selective compound bigrams (own df) then selective terms.
        Hub standalone terms are skipped for narrowing, not deleted.
        """
        words = self._ordered_query_words(query)
        units: list[dict] = []
        i = 0
        while i < len(words):
            if i + 1 < len(words):
                w1, w2 = words[i], words[i + 1]
                addr = self._compound_address(w1, w2)
                cpin = oriented_corridor_pin(addr)
                cp_df = self.pin_doc_freq.get(cpin, 0)
                if cp_df > 0 and self._pin_selective_for_narrowing(cpin, identity_pin=cpin):
                    units.append(
                        {
                            "kind": "compound",
                            "terms": (w1, w2),
                            "routing_pin": cpin,
                            "meet_composite": addr.meet_composite,
                            "corridor_key": list(addr.corridor_key),
                            "oriented_key": list(oriented_corridor_key(addr)),
                            "pin_doc_freq": cp_df,
                        }
                    )
                    i += 2
                    continue
            w = words[i]
            if not self._term_narrows(w):
                units.append({"kind": "hub_skip", "terms": (w,), "standalone_df": self.semantic.doc_freq.get(w, 0)})
            else:
                routing, selective, lift = self._routing_pin_for_term(w)
                units.append(
                    {
                        "kind": "term",
                        "terms": (w,),
                        "routing_pin": routing,
                        "pin_doc_freq": self.pin_doc_freq.get(routing, 0),
                        "selective_pins": sorted(selective),
                        "lift_pins": sorted(lift),
                    }
                )
            i += 1
        if self.enable_compound_first_routing:
            units.sort(
                key=lambda u: (
                    0 if u.get("kind") == "compound" else 1,
                    u.get("pin_doc_freq", 10**9),
                    u.get("terms", ("",))[0],
                )
            )
        else:
            units.sort(key=lambda u: (u.get("pin_doc_freq", 10**9), u.get("terms", ("",))[0]))
        return units

    def _compound_only_units(self, units: list[dict]) -> list[dict]:
        """Compound pins alone — skip constituent standalone terms that broke intersect."""
        return [u for u in units if u.get("kind") == "compound"]

    def _intersect_routing_units(
        self, units: list[dict]
    ) -> tuple[set[str], list[dict]]:
        """Apply selective intersect over routing units — glass-box steps."""
        steps: list[dict] = []
        pool: set[str] | None = None
        for unit in units:
            if unit["kind"] == "hub_skip":
                steps.append(
                    {
                        "step": "hub_skip",
                        "term": unit["terms"][0],
                        "standalone_df": unit["standalone_df"],
                        "reason": "standalone hub — lift only, no narrow",
                    }
                )
                continue
            pin = unit["routing_pin"]
            bucket = self._postings_for_routing_pin(pin)
            pool = set(bucket) if pool is None else pool & bucket
            step: dict = {
                "step": "rarest_filter",
                "kind": unit["kind"],
                "terms": list(unit["terms"]),
                "routing_pin": pin,
                "pin_doc_freq": unit["pin_doc_freq"],
                "pool_size": len(pool),
                "log_pool": round(math.log2(max(len(pool), 1)), 4),
            }
            if unit["kind"] == "term":
                w = unit["terms"][0]
                step["term"] = w
                step["token_identity"] = self._identity_for(w)
                step["corridor_pins"] = sorted(self._corridor_pins(w))
                step["selective_pins"] = unit.get("selective_pins", [])
                step["lift_pins"] = unit.get("lift_pins", [])
                step["doc_freq"] = self.semantic.doc_freq.get(w, 0)
            elif unit["kind"] == "compound":
                w1, w2 = unit["terms"]
                step["corridor_witness"] = self.corridor_witness_for_pair(w1, w2)
                step["meet_composite"] = unit.get("meet_composite")
                step["corridor_key"] = unit.get("corridor_key")
                step["oriented_key"] = unit.get("oriented_key")
            steps.append(step)
            if not pool:
                break
        return pool or set(), steps

    def _routed_terms(self, terms: list[str]) -> list[str]:
        """Rarest-first routing order — hubs deferred to end."""
        hubs = [t for t in terms if t in HUB_WORDS]
        rare = [t for t in terms if t not in HUB_WORDS]
        rare.sort(key=lambda w: (self.semantic.doc_freq.get(w, 0), w))
        hubs.sort(key=lambda w: (self.semantic.doc_freq.get(w, 0), w))
        return rare + hubs

    def route_pool(
        self, query: str
    ) -> tuple[
        list[str],
        RouteMode,
        tuple[dict, ...],
        tuple[QueryPrimeFactor, ...],
        tuple[dict, ...],
    ]:
        """
        Phase A routing with explicit miss policy — glass-box steps throughout.
        """
        terms = self._query_terms(query)
        factors = tuple(self._factor_query_term(t) for t in terms)
        velocity_witness: tuple[dict, ...] = ()
        if not terms:
            return (
                list(self.docs.keys()),
                "primary",
                ({"step": "empty_query", "pool": len(self.docs)},),
                factors,
                velocity_witness,
            )

        ordered = self._routed_terms(terms)
        units = self._routing_units(query)
        pool, steps = self._intersect_routing_units(units)

        if (
            not pool
            and self.enable_compound_first_routing
            and any(u.get("kind") == "compound" for u in units)
        ):
            compound_only = self._compound_only_units(units)
            if len(compound_only) < len([u for u in units if u.get("kind") in ("compound", "term")]):
                pool, retry_steps = self._intersect_routing_units(compound_only)
                if pool:
                    steps = [{"step": "compound_first_retry", "reason": "compound-only after full intersect empty"}]
                    steps.extend(retry_steps)
                    return sorted(pool), "primary", tuple(steps), factors, velocity_witness

        existing_pins = frozenset(
            u["routing_pin"] for u in units if u.get("kind") in ("compound", "term")
        )
        if (
            self.enable_k_meet_velocity_widen
            and (not pool or len(pool) < self.k_meet_min_pool_size)
        ):
            vel_pool, vel_meta, velocity_witness = self._k_meet_velocity_widen_pool(
                query, existing_pins
            )
            if vel_pool:
                steps.append(
                    {
                        "step": "k_meet_velocity_widen",
                        "widen_pins": vel_meta,
                        "pool_size": len(vel_pool),
                        "velocity_unified": bool(velocity_witness),
                    }
                )
                if not pool:
                    return (
                        sorted(vel_pool),
                        "k_meet_velocity_widen",
                        tuple(steps),
                        factors,
                        velocity_witness,
                    )
                pool |= vel_pool

        if pool:
            return sorted(pool), "primary", tuple(steps), factors, velocity_witness

        # Fallback 1: lift-pin widen (selective intersect empty; lift pins may still route)
        if (
            self.enable_lift_pin_fallback
            and self.miss_policy in (MissPolicy.FTA_LETTER_FALLBACK, MissPolicy.WIDEN_RAREST_ONLY)
        ):
            lift_pool, lift_meta = self._lift_pin_widen_pool(query)
            if lift_pool:
                steps.append(
                    {
                        "step": "lift_pin_widen",
                        "lift_pins": lift_meta,
                        "pool_size": len(lift_pool),
                    }
                )
                return sorted(lift_pool), "lift_pin_widen", tuple(steps), factors, velocity_witness

        # Fallback 2: rarest-only union
        if self.miss_policy in (MissPolicy.FTA_LETTER_FALLBACK, MissPolicy.WIDEN_RAREST_ONLY):
            rarest = ordered[0]
            pool = self._postings_for_term(rarest)
            steps.append(
                {
                    "step": "widen_rarest",
                    "term": rarest,
                    "routing_pin": self._routing_pin_for_term(rarest)[0],
                    "corridor_pins": sorted(self._corridor_pins(rarest)),
                    "selective_pins": sorted(self._routing_pin_for_term(rarest)[1]),
                    "pool_size": len(pool),
                }
            )
            if pool:
                return sorted(pool), "widen_rarest", tuple(steps), factors, velocity_witness

        # Fallback 3: FTA letter-prime postings for OOV / unlit terms
        if self.miss_policy == MissPolicy.FTA_LETTER_FALLBACK:
            letter_pool: set[str] = set()
            oov_terms = [t for t in ordered if not factors[[f.term for f in factors].index(t)].in_corpus]
            targets = oov_terms if oov_terms else ordered
            for term in targets:
                for lp in self._letter_primes(term):
                    letter_pool |= self.letter_postings.get(lp, set())
            steps.append(
                {
                    "step": "fta_letter_fallback",
                    "terms": targets,
                    "letter_primes": [self._letter_primes(t) for t in targets],
                    "pool_size": len(letter_pool),
                }
            )
            if letter_pool:
                return sorted(letter_pool), "fta_letter_fallback", tuple(steps), factors, velocity_witness

        steps.append({"step": "miss_empty", "reason": "no corridor opened"})
        return [], "empty", tuple(steps), factors, velocity_witness

    def lazy_pool(self, query: str) -> tuple[list[str], tuple[dict, ...]]:
        """Backward-compatible pool API."""
        pool, _, steps, _, _ = self.route_pool(query)
        return pool, steps

    def _query_cage_trace(self, terms: list[str]) -> tuple[tuple[dict, ...], tuple[dict, ...]]:
        """Cages lit + rare dots touched at query time."""
        cages_out: list[dict] = []
        dots_out: list[dict] = []
        window = terms[:6]
        qset = set(terms)
        for i in range(max(0, len(window) - 2)):
            triple = window[i : i + 3]
            cage = self.semantic.cages.get(
                self.semantic._cage_for_triple(*triple).anchor_composite
            )
            if cage is None:
                continue
            touch = self.semantic.touch_weight(qset, cage)
            cages_out.append(
                {
                    "anchor": cage.anchor_label,
                    "anchor_composite": cage.anchor_composite,
                    "touch": round(touch, 4),
                }
            )
            for term, dot in cage.correlations.items():
                if term in qset and self.semantic.is_rare(term):
                    dots_out.append(
                        {
                            "term": term,
                            "anchor": cage.anchor_label,
                            "strength": dot.strength,
                            "drift_weight": round(dot.drift_weight, 4),
                            "rotation_quadrant": dot.rotation_quadrant,
                            "dim4": round(dot.dim4, 4),
                            "dim5": round(dot.dim5, 4),
                            "dim6": round(dot.dim6, 4),
                        }
                    )
        return tuple(cages_out), tuple(dots_out)

    def _corridor_address(self, w1: str, w2: str) -> FreeTokenAddress:
        """Build corridor address from read-order pair — no hub skip (trace/explain)."""
        p, q = self._identity_for(w1), self._identity_for(w2)
        df = {c: max(1, self.semantic.doc_freq.get(c, 1)) for c in w1 + w2 if c.isalpha()}
        quadrant = rotate_token(w1 + w2, df).quadrant if w1 and w2 else 1
        return free_token_address(p, q, quadrant=quadrant, invoke_order=(p, q))

    def _lazy_corridor_address(self, w1: str, w2: str) -> FreeTokenAddress | None:
        """Open 2-way corridor for scoring — hub pairs skip lift."""
        if w1 in HUB_WORDS or w2 in HUB_WORDS:
            return None
        return self._corridor_address(w1, w2)

    def corridor_witness_for_pair(self, w1: str, w2: str) -> dict:
        """Glass-box path witness for an ordered bigram — TH vs HE separation."""
        addr = self._corridor_address(w1, w2)
        witness = corridor_witness_explain(addr, pair=(w1, w2))
        witness["decode"] = decode_corridor_address(
            addr, registry=self.semantic.registry, from_text=w1, to_text=w2
        )
        return witness

    def _query_corridor_witnesses(self, query: str) -> tuple[dict, ...]:
        words = self._ordered_query_words(query)
        out: list[dict] = []
        for i in range(len(words) - 1):
            out.append(self.corridor_witness_for_pair(words[i], words[i + 1]))
        return tuple(out)

    def _cage_score(self, query_terms: list[str], doc_words: tuple[str, ...]) -> tuple[float, list[dict]]:
        """Phase B — wing-cage touch on shared 3-way windows."""
        score = 0.0
        reasons: list[dict] = []
        qset = set(query_terms)
        window = query_terms[:6]
        for i in range(max(0, len(window) - 2)):
            triple = window[i : i + 3]
            if not all(t in doc_words for t in triple):
                continue
            cage = self.semantic.cages.get(
                self.semantic._cage_for_triple(*triple).anchor_composite
            )
            if cage is None:
                continue
            w = self.semantic.touch_weight(qset, cage)
            if w > 0:
                score += w
                dot_hits = []
                for t in qset:
                    dot = cage.correlations.get(t)
                    if dot:
                        dot_hits.append(
                            {"term": t, "drift_weight": round(dot.drift_weight, 4), "strength": dot.strength}
                        )
                reasons.append(
                    {
                        "kind": "wing_cage",
                        "anchor": cage.anchor_label,
                        "anchor_composite": cage.anchor_composite,
                        "touch": round(w, 4),
                        "rare_dots": dot_hits,
                    }
                )
        return score, reasons

    def score_doc(self, doc_id: str, query: str) -> RetrieveHit:
        doc = self.docs[doc_id]
        terms = self._query_terms(query)
        q_pins: set[int] = set()
        for t in terms:
            q_pins |= self._corridor_pins(t)
        doc_pins = set(doc.corridor_pins)
        pin_overlap = len(q_pins & doc_pins)
        word_hits = sum(1 for t in terms if t in doc.words)

        score = float(pin_overlap) + 0.5 * word_hits
        reasons: list[dict] = [
            {
                "kind": "identity_overlap",
                "matched": pin_overlap,
                "word_hits": word_hits,
                "query_terms": terms,
                "query_corridor_pins": sorted(q_pins),
                "doc_corridor_pins": list(doc.corridor_pins),
            }
        ]

        for i, w1 in enumerate(terms):
            for w2 in terms[i + 1 :]:
                witness = self.corridor_witness_for_pair(w1, w2)
                if witness is None:
                    continue
                w1_pins = self._corridor_pins(w1)
                w2_pins = self._corridor_pins(w2)
                if w1_pins & doc_pins or w2_pins & doc_pins:
                    reasons.append(
                        {
                            "kind": "corridor_open",
                            "lazy": True,
                            **witness,
                        }
                    )

        cage_score, cage_reasons = self._cage_score(terms, doc.words)
        score += cage_score
        reasons.extend(cage_reasons)

        for t in terms:
            if t not in HUB_WORDS:
                score += self.semantic.idf(t) * (1 if t in doc.words else 0)

        return RetrieveHit(doc_id=doc_id, score=score, reasons=tuple(reasons))

    def _shared_rare_terms(self, doc_a: str, doc_b: str) -> tuple[str, ...]:
        """Rare lexical overlap between two docs — blind bridge signal."""
        da, db = self.docs.get(doc_a), self.docs.get(doc_b)
        if da is None or db is None:
            return ()
        shared = set(da.words) & set(db.words)
        rare = tuple(
            sorted(
                t
                for t in shared
                if t not in HUB_WORDS and self.semantic.is_rare(t)
            )
        )
        return rare

    def _doc_cage_composites(self, doc_id: str, *, max_words: int = 32) -> frozenset[int]:
        doc = self.docs.get(doc_id)
        if doc is None:
            return frozenset()
        words = list(doc.words)[:max_words]
        comps: set[int] = set()
        for i in range(max(0, len(words) - 2)):
            triple = words[i : i + 3]
            cage = self.semantic.cages.get(
                self.semantic._cage_for_triple(*triple).anchor_composite
            )
            if cage is not None:
                comps.add(cage.anchor_composite)
        return frozenset(comps)

    def _shared_cage_anchors(self, doc_a: str, doc_b: str) -> tuple[int, ...]:
        return tuple(
            sorted(self._doc_cage_composites(doc_a) & self._doc_cage_composites(doc_b))
        )

    def _rerank_tiebreak_delta(self, top_score: float, *, frac: float) -> float:
        """Score gap within which rerank may reorder; inf disables the gate."""
        if frac <= 0:
            return float("inf")
        return frac * max(top_score, 1e-9)

    def _cage_anchor_rerank(
        self, hits: list[RetrieveHit], query_terms: list[str]
    ) -> list[RetrieveHit]:
        """
        Blind L35 rule: boost docs sharing wing-cage anchors with top anchor doc.
        Uses query-lit cage touch weights — no gold knowledge.
        """
        if not self.enable_cage_anchor_rerank or len(hits) < 2:
            return hits
        top_score = hits[0].score
        tie_delta = self._rerank_tiebreak_delta(
            top_score, frac=self.cage_anchor_rerank_tiebreak_frac
        )
        anchor_id = hits[0].doc_id
        anchor_cages = self._doc_cage_composites(anchor_id)
        if not anchor_cages:
            return hits
        boosted: list[RetrieveHit] = []
        for h in hits:
            if h.doc_id == anchor_id:
                boosted.append(h)
                continue
            if top_score - h.score > tie_delta:
                boosted.append(h)
                continue
            shared = self._shared_cage_anchors(anchor_id, h.doc_id)
            if not shared:
                boosted.append(h)
                continue
            touch = 0.0
            anchors_hit: list[dict] = []
            for comp in shared:
                cage = self.semantic.cages.get(comp)
                if cage is None:
                    continue
                w = self.semantic.touch_weight(query_terms, cage)
                if w > 0:
                    touch += w
                    anchors_hit.append(
                        {"anchor_composite": comp, "anchor_label": cage.anchor_label, "touch": round(w, 4)}
                    )
            boost = self.cage_anchor_rerank_lambda * touch
            if boost > 0:
                reasons = list(h.reasons) + [
                    {
                        "kind": "cage_anchor_bridge",
                        "anchor_doc": anchor_id,
                        "shared_anchors": anchors_hit,
                        "boost": round(boost, 4),
                    }
                ]
                boosted.append(
                    RetrieveHit(doc_id=h.doc_id, score=h.score + boost, reasons=tuple(reasons))
                )
            else:
                boosted.append(h)
        boosted.sort(key=lambda h: (-h.score, h.doc_id))
        return boosted

    def _lexical_bridge_rerank(self, hits: list[RetrieveHit]) -> list[RetrieveHit]:
        """
        Blind L40 rule: boost docs that share rare terms with top anchor doc(s).
        Uses preliminary rank — no gold knowledge.
        """
        if not self.enable_lexical_bridge_rerank or len(hits) < 2:
            return hits
        top_score = hits[0].score
        tie_delta = self._rerank_tiebreak_delta(
            top_score, frac=self.bridge_rerank_tiebreak_frac
        )
        n_anchor = max(1, min(self.bridge_rerank_anchors, len(hits)))
        anchors = [h.doc_id for h in hits[:n_anchor]]
        boosted: list[RetrieveHit] = []
        for h in hits:
            if h.doc_id not in anchors and top_score - h.score > tie_delta:
                boosted.append(h)
                continue
            best_terms: tuple[str, ...] = ()
            best_boost = 0.0
            best_anchor = ""
            for aid in anchors:
                if aid == h.doc_id:
                    continue
                shared = self._shared_rare_terms(aid, h.doc_id)
                if not shared:
                    continue
                b = self.bridge_rerank_lambda * sum(self.semantic.idf(t) for t in shared)
                if b > best_boost:
                    best_boost = b
                    best_terms = shared
                    best_anchor = aid
            if best_boost > 0:
                reasons = list(h.reasons) + [
                    {
                        "kind": "lexical_bridge",
                        "anchor_doc": best_anchor,
                        "shared_rare_terms": list(best_terms),
                        "boost": round(best_boost, 4),
                    }
                ]
                boosted.append(
                    RetrieveHit(
                        doc_id=h.doc_id,
                        score=h.score + best_boost,
                        reasons=tuple(reasons),
                    )
                )
            else:
                boosted.append(h)
        boosted.sort(key=lambda h: (-h.score, h.doc_id))
        return boosted

    def retrieve_with_trace(self, query: str, *, limit: int = 10) -> RetrieveTrace:
        """Lazy branch retrieve with full glass-box trace."""
        terms = self._query_terms(query)
        pool, route_mode, filter_steps, query_primes, velocity_witness = self.route_pool(query)
        cages, rare_dots = self._query_cage_trace(terms)

        hits = [self.score_doc(doc_id, query) for doc_id in pool]
        enriched: list[RetrieveHit] = []
        for h in hits:
            reasons = list(h.reasons)
            if reasons:
                reasons[0] = {
                    **reasons[0],
                    "filter_steps": list(filter_steps),
                    "route_mode": route_mode,
                }
            enriched.append(RetrieveHit(doc_id=h.doc_id, score=h.score, reasons=tuple(reasons)))
        enriched.sort(key=lambda h: (-h.score, h.doc_id))
        enriched = self._lexical_bridge_rerank(enriched)
        enriched = self._cage_anchor_rerank(enriched, terms)

        return RetrieveTrace(
            query=query,
            route_mode=route_mode,
            query_primes=query_primes,
            filter_steps=filter_steps,
            cages_considered=cages,
            rare_dots=rare_dots,
            pool_size=len(pool),
            corridor_witnesses=self._query_corridor_witnesses(query),
            velocity_witness=velocity_witness,
            hits=tuple(enriched[:limit]),
        )

    def retrieve(self, query: str, *, limit: int = 10) -> list[RetrieveHit]:
        """Lazy branch retrieve — bounded pool, glass-box reasons on every hit."""
        return list(self.retrieve_with_trace(query, limit=limit).hits)


# --- Synthetic 10-doc fixture (BUILD gate: 8/10 top-1) ---

FIXTURE_CORPUS: dict[str, str] = {
    "d01": "cat purrs softly at home",
    "d02": "dog barks loudly at mailman",
    "d03": "apple phone sells in store",
    "d04": "python code runs fast here",
    "d05": "cancer mutation rare variant",
    "d06": "heart blood pumps daily",
    "d07": "moon orbits earth slowly",
    "d08": "virus infects host cells",
    "d09": "gold metal shines bright",
    "d10": "fish swim in water",
}

FIXTURE_QUERIES: tuple[tuple[str, str, str], ...] = (
    ("q01", "cat purrs", "d01"),
    ("q02", "dog barks", "d02"),
    ("q03", "apple phone", "d03"),
    ("q04", "python code", "d04"),
    ("q05", "cancer mutation", "d05"),
    ("q06", "heart blood", "d06"),
    ("q07", "moon earth", "d07"),
    ("q08", "virus cells", "d08"),
    ("q09", "gold metal", "d09"),
    ("q10", "fish water", "d10"),
)


def build_fixture_retriever() -> LatticeRetriever:
    r = LatticeRetriever()
    r.index_corpus(FIXTURE_CORPUS)
    return r
