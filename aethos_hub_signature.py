"""
Lattice Hub Signature — compact per-document index for retrieval.

Section 5 framing
-----------------
Each document's observation pass (ingest) compresses co-occurrence into
pool primes and L4-L6 correlation edges.  The top-K words by
``compression_strength`` are the most strongly pinned — those are the
document's observable hubs.

Instead of storing full token sets for scoring (O(Q × D_tokens) per query),
we store only the hub entries: word + ``formula_coord`` at the anchor n +
L4-L6 neighbor set.  Scoring then runs in O(Q × K) time (K = hub count,
typically 12–16).

Step 2 will replace the inverted-index candidate pass with a MeetIndex
(swap-meet routers between prime banks) — that stays fully compatible with
this signature layout because each HubEntry already stores prime + coord.

Byte estimate (compact):
  Legacy HubEntry: prime(4) + strength(4) + coord(12) + band + z_obs ≈ 25 B + word
  Pin wire (α + critical line): 8 B pin + strength(4) + band(1) + z_obs(4) + word
  K = 12 → ~96–120 B/doc pin mode vs ~240 B/doc legacy coord index
  Coords regenerated via SpacetimeCell.at (3D complex plane = lattice).
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field, replace

from functools import reduce
from operator import mul

from aethos_composite import morph_meet_score
from aethos_hub_wire import (
    CriticalLinePin,
    LegSumMeetKey,
    WIRE_BYTES,
    hub_coord_from_word,
    leg_sum_meet_key,
    leg_sum_meet_key_from_coord,
    pin_for_query_word,
    wing_coords_for_word,
)
from aethos_lattice import LatticeId, lattice_id_parts
from aethos_promotion import LatticeTier, is_stopword
# Pool-promoted primes only for meet/composite indexing (letter primes are universal).
MIN_POOL_PRIME = 107


# ---------------------------------------------------------------------------
# 32-wing consensus constants
# ---------------------------------------------------------------------------

# Representative wings for consensus scoring: one per VA branch family.
# L01 = VA1×v1, L09 = VA2×v1, L17 = VA3×v1, L25 = VA4×v1.
# A genuine swap-meet fires on ALL wings; coincidental L01 collisions fire on fewer.
# Using 4 wings gives 4× the geometric evidence with modest compute overhead.
CONSENSUS_WINGS: tuple[int, ...] = (1, 9, 17, 25)

# Signal 7: L7-L9 cluster routing bonus weight
# When a hub word belongs to the same L7-L9 cluster as a query word, boost.
LAMBDA_CLUSTER = 0.4


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------

def _pool_factors(prime: int, parent_primes: tuple[int, ...]) -> frozenset[int]:
    """Pool-promoted primes only — letter primes (3..101) are excluded."""
    out: set[int] = set()
    if prime >= MIN_POOL_PRIME:
        out.add(prime)
    for p in parent_primes:
        if p >= MIN_POOL_PRIME:
            out.add(p)
    return frozenset(out)


def pool_factor_jaccard(a: frozenset[int], b: frozenset[int]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def lattice_composite_for_token(prime: int, parent_primes: tuple[int, ...]) -> int:
    """FTA product of pool primes only; 1 if no pool factors (Signal 5b skips)."""
    factors = _pool_factors(prime, parent_primes)
    if not factors:
        return 1
    if len(factors) == 1:
        return next(iter(factors))
    return reduce(mul, sorted(factors), 1)


def lattice_composite_for_word(registry, word: str) -> int:
    tok = registry.resolve_token(word.lower())
    return lattice_composite_for_token(tok.prime, tok.parent_primes)


def pool_factors_for_word(registry, word: str) -> frozenset[int]:
    tok = registry.resolve_token(word.lower())
    return _pool_factors(tok.prime, tok.parent_primes)


@dataclass(frozen=True)
class HubEntry:
    """Single hub word pinned to the lattice."""

    word: str
    strength: float
    prime: int
    lattice_composite: int                                  # product of pool primes in chain
    neighbors: frozenset[str]                               # L4-L6 correlated words
    coord: tuple[float, float, float] | None = None          # legacy L01; None when pin wire
    pool_factors: frozenset[int] = frozenset()              # factors for Signal 5b Jaccard
    band_id: int = 0                                        # BIT 5: |z| band 0..3
    z_obs: float = 0.0                                      # BIT 5: Re(z_VA1 + z_VA2)
    # Multi-wing coords for consensus scoring: tuple of (lattice_id_int, x, y, z).
    # Stored as a tuple of tuples so HubEntry stays frozen/hashable.
    # Populated for wings in CONSENSUS_WINGS; empty tuple = not computed.
    wing_coord_tuples: tuple[tuple[int, float, float, float], ...] = ()
    pin: CriticalLinePin | None = None
    lazy_wings: bool = False

    def wing_coords(self) -> dict[int, tuple[float, float, float]]:
        """Return {lattice_id: (x,y,z)} for fast O(1) lookup in Signal 2."""
        return {lid: (x, y, z) for lid, x, y, z in self.wing_coord_tuples}

    def resolve_wing_coords(
        self,
        registry,
        *,
        consensus_wings: tuple[int, ...] | None = None,
    ) -> dict[int, tuple[float, float, float]]:
        """Materialized wings or on-demand regen from lattice address α."""
        if self.wing_coord_tuples:
            return self.wing_coords()
        wings = consensus_wings if consensus_wings is not None else CONSENSUS_WINGS
        if registry is not None and self.lazy_wings:
            return wing_coords_for_word(
                registry,
                self.word,
                rail_n=self.pin.rail_n if self.pin else 7,
                consensus_wings=wings,
            )
        if self.coord:
            return {1: self.coord}
        return {}

    def resolve_coord(self, registry) -> tuple[float, float, float] | None:
        if self.coord:
            return self.coord
        if self.pin is not None and registry is not None:
            return hub_coord_from_word(registry, self.word, self.pin)
        return None

    def compact_bytes(self) -> int:
        """Wire size estimate: pin mode vs legacy float coord."""
        word_b = len(self.word.encode("utf-8"))
        if self.pin is not None and self.lazy_wings:
            return WIRE_BYTES + 4 + 1 + 4 + word_b  # pin + strength + band + z_obs + word
        return 25 + word_b


@dataclass
class LatticeHubSignature:
    """
    Compact per-document lattice index.

    ``hubs``       : word → HubEntry  (top-K by compression_strength)
    ``hub_coords``  : coord → word (legacy float meet, omitted when pin wire)
    ``hub_pins``    : CriticalLinePin → word (pin-keyed L01 meet)
    ``hub_leg_sum`` : LegSumMeetKey → word (S-partner leg_sum dedup on j)
    """

    doc_id: str
    hubs: dict[str, HubEntry] = field(default_factory=dict)
    hub_coords: dict[tuple[float, float, float], str] = field(default_factory=dict)
    hub_pins: dict[CriticalLinePin, str] = field(default_factory=dict)
    hub_leg_sum: dict[LegSumMeetKey, str] = field(default_factory=dict)

    def hub_words(self) -> frozenset[str]:
        return frozenset(self.hubs.keys())

    def encoded_size(self) -> int:
        return sum(e.compact_bytes() for e in self.hubs.values())


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def precompute_registry_index(
    registry,
    *,
    anchor_n: int = 7,
    lattice_id: LatticeId = LatticeId.L01,
) -> "RegistryIndex":
    """
    Precompute all per-word data in O(|words| + |correlations|) — called once,
    not once per document.  Calling compression_strength / correlations_for
    inside a per-doc loop is O(docs × words × |corr|) — too slow for real corpora.
    """
    # Build neighbor map once: word → {neighbor: total_weight}
    neighbor_strength: dict[str, float] = {}
    word_neighbors: dict[str, set[str]] = {}
    for (a, b), link in registry.correlations.items():
        w = float(link.strength) * (link.dim4 + link.dim6 + 1.0)
        neighbor_strength[a] = neighbor_strength.get(a, 0.0) + link.strength
        neighbor_strength[b] = neighbor_strength.get(b, 0.0) + link.strength
        word_neighbors.setdefault(a, set()).add(b)
        word_neighbors.setdefault(b, set()).add(a)

    # Per-word strength: correlation mass or raw count
    strength_map: dict[str, float] = {}
    for w, cnt in registry.word_counts.items():
        s = neighbor_strength.get(w)
        strength_map[w] = float(s) if s else float(cnt)
    for w, cnt in registry.number_counts.items():
        if w not in strength_map:
            strength_map[w] = float(cnt)

    # Per-word formula_coord and prime (cached lazily below)
    return RegistryIndex(
        registry=registry,
        strength_map=strength_map,
        word_neighbors=word_neighbors,
        anchor_n=anchor_n,
        lattice_id=lattice_id,
    )


class RegistryIndex:
    """
    O(1) per-word lookups after a single O(|vocab| + |corr|) build pass.
    Coord and prime are cached on first access per word.
    """

    def __init__(
        self,
        registry,
        strength_map: dict[str, float],
        word_neighbors: dict[str, set[str]],
        anchor_n: int,
        lattice_id: LatticeId,
    ) -> None:
        self.registry = registry
        self.strength_map = strength_map
        self.word_neighbors = word_neighbors
        self.anchor_n = anchor_n
        self.lattice_id = lattice_id
        # L01 single-wing cache (backward compat)
        self._coord_cache: dict[str, tuple[float, float, float]] = {}
        self._prime_cache: dict[str, int] = {}
        # Multi-wing cache: (word, lattice_id_int) → coord
        self._wing_cache: dict[tuple[str, int], tuple[float, float, float]] = {}

    def coord(self, word: str) -> tuple[float, float, float] | None:
        if word in self._coord_cache:
            return self._coord_cache[word]
        try:
            c = self.registry.lattice_address(word, LatticeTier.L3_WORD, self.anchor_n, self.lattice_id)
            self._coord_cache[word] = c
            return c
        except Exception:
            return None

    def wing_coord(self, word: str, lid: int) -> tuple[float, float, float] | None:
        """
        Return formula_coord for ``word`` on lattice ``lid`` (1–32).

        Uses a per-wing cache so each (word, lid) pair is computed at most once
        across the full hub-building pass.  lid=1 is equivalent to .coord().
        """
        key = (word, lid)
        if key in self._wing_cache:
            return self._wing_cache[key]
        try:
            c = self.registry.lattice_address(
                word, LatticeTier.L3_WORD, self.anchor_n, LatticeId(lid)
            )
            self._wing_cache[key] = c
            return c
        except Exception:
            return None

    def all_wing_coords(self, word: str) -> tuple[tuple[int, float, float, float], ...]:
        """
        Return wing_coord_tuples for all CONSENSUS_WINGS.
        Returns only the wings that successfully computed (skips exceptions).
        """
        result: list[tuple[int, float, float, float]] = []
        for lid in CONSENSUS_WINGS:
            c = self.wing_coord(word, lid)
            if c is not None:
                result.append((lid, c[0], c[1], c[2]))
        return tuple(result)

    def prime(self, word: str) -> int:
        if word in self._prime_cache:
            return self._prime_cache[word]
        tok = self.registry.resolve_token(word)
        self._prime_cache[word] = tok.prime
        return tok.prime

    def lattice_composite(self, word: str) -> int:
        tok = self.registry.resolve_token(word.lower())
        return lattice_composite_for_token(tok.prime, tok.parent_primes)

    def pool_factors(self, word: str) -> frozenset[int]:
        tok = self.registry.resolve_token(word.lower())
        return _pool_factors(tok.prime, tok.parent_primes)

    def neighbors(self, word: str) -> frozenset[str]:
        return frozenset(self.word_neighbors.get(word, set()))

    def strength(self, word: str) -> float:
        return self.strength_map.get(word, 0.0)


def build_hub_signature_from_tokens(
    doc_id: str,
    tokens: frozenset[str],
    idx: RegistryIndex,
    *,
    top_k: int = 12,
    use_pin_wire: bool = True,
) -> LatticeHubSignature:
    """
    Build a LatticeHubSignature using precomputed RegistryIndex.
    O(|tokens| × top_k) — no full correlation scan per call.
    """
    candidates: list[tuple[float, str]] = []
    for w in tokens:
        if not w.isalpha() or len(w) < 3 or is_stopword(w):
            continue
        s = idx.strength(w)
        if s > 0:
            candidates.append((s, w))

    candidates.sort(key=lambda x: (-x[0], x[1]))

    sig = LatticeHubSignature(doc_id=doc_id)
    for strength, word in candidates[:top_k]:
        c = idx.coord(word)
        if c is None:
            continue
        pf = idx.pool_factors(word)
        from pipeline.bit_05_z_band import band_profile_for_word

        band = band_profile_for_word(idx.registry, word, n=idx.anchor_n)
        branch, vec = lattice_id_parts(idx.lattice_id)
        pin = CriticalLinePin.from_coord(
            c,
            idx.prime(word),
            rail_n=idx.anchor_n,
            branch=int(branch),
            wing=vec.index,
        )
        entry = HubEntry(
            word=word,
            strength=strength,
            prime=idx.prime(word),
            lattice_composite=idx.lattice_composite(word),
            pool_factors=pf,
            band_id=band.band_id,
            z_obs=band.z_obs,
            coord=None if use_pin_wire else c,
            neighbors=idx.neighbors(word),
            wing_coord_tuples=() if use_pin_wire else idx.all_wing_coords(word),
            pin=pin if use_pin_wire else None,
            lazy_wings=use_pin_wire,
        )
        sig.hubs[word] = entry
        if use_pin_wire:
            prev = sig.hub_pins.get(pin)
            if prev is None or entry.strength > sig.hubs[prev].strength:
                sig.hub_pins[pin] = word
            lsk = leg_sum_meet_key(pin)
            prev_ls = sig.hub_leg_sum.get(lsk)
            if prev_ls is None or entry.strength > sig.hubs[prev_ls].strength:
                sig.hub_leg_sum[lsk] = word
        else:
            sig.hub_coords[c] = word

    return sig


def build_hub_signature(
    doc_id: str,
    text: str,
    registry,
    *,
    top_k: int = 12,
    anchor_n: int = 7,
    lattice_id: LatticeId = LatticeId.L01,
    use_pin_wire: bool = True,
) -> LatticeHubSignature:
    """Single-doc helper for tests.  For corpus-scale use build_all_hub_signatures."""
    from aethos_tokenize import tokenize_words
    idx = precompute_registry_index(registry, anchor_n=anchor_n, lattice_id=lattice_id)
    return build_hub_signature_from_tokens(
        doc_id,
        frozenset(tokenize_words(text)),
        idx,
        top_k=top_k,
        use_pin_wire=use_pin_wire,
    )


def materialize_lazy_hub_wings(
    sigs: dict[str, LatticeHubSignature],
    registry,
    *,
    consensus_wings: tuple[int, ...] | None = None,
) -> None:
    """
    One-shot wing regen from pin wire — query path matches legacy (O(1) lookup).

    Pin wire keeps ``encoded_size()`` small; wings live in RAM only after build.
    """
    wings = consensus_wings if consensus_wings is not None else CONSENSUS_WINGS
    for sig in sigs.values():
        new_hubs: dict[str, HubEntry] = {}
        changed = False
        for word, entry in sig.hubs.items():
            if entry.wing_coord_tuples or not entry.lazy_wings:
                new_hubs[word] = entry
                continue
            wc = wing_coords_for_word(
                registry,
                word,
                rail_n=entry.pin.rail_n if entry.pin else 7,
                consensus_wings=wings,
            )
            tuples = tuple(
                (lid, x, y, z) for lid in wings for x, y, z in [wc[lid]] if lid in wc
            )
            new_hubs[word] = replace(entry, wing_coord_tuples=tuples)
            changed = True
        if changed:
            sig.hubs = new_hubs


def build_all_hub_signatures(
    doc_ids: list[str],
    doc_tokens: dict[str, frozenset[str]],
    registry,
    *,
    top_k: int = 12,
    anchor_n: int = 7,
    use_pin_wire: bool = True,
    materialize_wings: bool = True,
) -> dict[str, LatticeHubSignature]:
    """
    Build hub signatures for all docs after ingest.
    Precomputes registry index once then runs in O(docs × words_per_doc).
    """
    idx = precompute_registry_index(registry, anchor_n=anchor_n)
    sigs: dict[str, LatticeHubSignature] = {}
    for did in doc_ids:
        sigs[did] = build_hub_signature_from_tokens(
            did,
            doc_tokens.get(did, frozenset()),
            idx,
            top_k=top_k,
            use_pin_wire=use_pin_wire,
        )
    if use_pin_wire and materialize_wings:
        materialize_lazy_hub_wings(sigs, registry)
    return sigs


# ---------------------------------------------------------------------------
# Query profile — computed once per query, not per (query, doc) pair
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Signal weight constants — IDF × saturated_overlap × λ algebra.
# Every lattice bonus must be in the same units as BM25 Signal 1.
# λ = 1.0 means the signal contributes one full IDF unit at maximum match.
# ---------------------------------------------------------------------------

LAMBDA_COORD = 0.5      # Signal 2: coord meet bonus ∝ IDF of the matching word
LAMBDA_NEIGHBOR = 0.15  # Signal 3: reduced — audit showed false-doc lift on SciFact
LAMBDA_PRIME_FACTOR = 0.0   # Signal 5b: off until pool-prime promotion sufficient
LAMBDA_KAPPA = 0.0      # Signal 8a: off until train-tuned (0.25 regressed on SciFact slice)


@dataclass
class QueryProfile:
    """
    Precomputed per-query data for fast hub scoring.

    Built once; reused across all candidate documents.

    ``flat_neighbors``    collapses per-word neighbor maps → {nb_word: max_weight}
    ``neighbor_source``   maps nb_word → the query word that triggered it
                          (needed for IDF-weighting Signal 3)
    ``max_neighbor_weight`` pre-computed denominator for [0,1] saturation
    ``wing_coords``       per-wing coord lookup for 32-wing consensus Signal 2.
                          Structure: {lattice_id_int: {coord_tuple: query_word}}
    """

    words: list[str]
    word_set: frozenset[str]
    coords: dict[tuple[float, float, float], str]    # coord → word (legacy L01)
    idf: dict[str, float]                            # word → IDF score
    flat_neighbors: dict[str, float]                 # neighbor_word → max raw weight
    neighbor_source: dict[str, str]                  # neighbor_word → query_word that fired it
    pins: dict[CriticalLinePin, str] = field(default_factory=dict)
    leg_sum_pins: dict[LegSumMeetKey, str] = field(default_factory=dict)
    max_neighbor_weight: float = 1.0                 # normalization denominator
    wing_coords: dict[int, dict[tuple[float, float, float], str]] = field(default_factory=dict)
    # {lattice_id_int: {coord: query_word}} for each lid in CONSENSUS_WINGS

    # Signal 7: L7-L9 cluster IDs that query words belong to.
    # Precomputed from pipe.reader.word_to_cluster at query time.
    # {cluster_id: idf_of_best_query_word_in_that_cluster}
    query_cluster_ids: dict[str, float] = field(default_factory=dict)
    # Signal 5b: query word → pool prime factors (≥ MIN_POOL_PRIME)
    word_pool_factors: dict[str, frozenset[int]] = field(default_factory=dict)


def build_query_profile(
    query: str,
    registry,
    *,
    neighbor_map: dict[str, dict[str, float]],
    doc_freq: dict[str, int],   # # docs containing each word (document frequency)
    n_docs: int,
    anchor_n: int = 7,
    lattice_id: LatticeId = LatticeId.L01,
) -> QueryProfile:
    """
    Precompute all per-query lookup structures.

    ``doc_freq`` must be document frequency (# docs containing w), not total
    term count — this is critical for correct IDF weighting.
    """
    from aethos_tokenize import tokenize_words

    words = tokenize_words(query)
    coords: dict[tuple[float, float, float], str] = {}
    pins: dict[CriticalLinePin, str] = {}
    leg_sum_pins: dict[LegSumMeetKey, str] = {}
    idf: dict[str, float] = {}
    word_pool_factors: dict[str, frozenset[int]] = {}
    flat_neighbors: dict[str, float] = {}
    neighbor_source: dict[str, str] = {}
    # Per-wing coord maps for 32-wing consensus Signal 2
    wing_coords: dict[int, dict[tuple[float, float, float], str]] = {
        lid: {} for lid in CONSENSUS_WINGS
    }

    for w in words:
        if not w.isalpha():
            continue
        # BM25-style IDF: log((N - df + 0.5) / (df + 0.5) + 1)
        df = max(doc_freq.get(w, 0), 0)
        idf[w] = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)

        if len(w) >= 3 and not is_stopword(w):
            pf = pool_factors_for_word(registry, w)
            if pf:
                word_pool_factors[w] = pf

        if is_stopword(w) or len(w) < 3:
            continue  # stopwords contribute to IDF scoring but not coord/neighbor signals

        # L01 coord + pin meet keys
        try:
            c = registry.lattice_address(w, LatticeTier.L3_WORD, anchor_n, lattice_id)
            coords[c] = w
            pin = pin_for_query_word(
                registry, w, c, anchor_n=anchor_n, lattice_id=lattice_id
            )
            pins[pin] = w
            leg_sum_pins[leg_sum_meet_key(pin)] = w
            leg_sum_pins[leg_sum_meet_key_from_coord(c, rail_n=anchor_n)] = w
        except Exception:
            pass

        # Multi-wing coords for consensus Signal 2
        for lid in CONSENSUS_WINGS:
            try:
                wc = registry.lattice_address(w, LatticeTier.L3_WORD, anchor_n, LatticeId(lid))
                wing_coords[lid][wc] = w
            except Exception:
                pass

        # Flatten neighbors — skip stopword neighbors to avoid noise.
        # Track which query word triggered each neighbor (for IDF weighting in Signal 3).
        for nb, weight in neighbor_map.get(w, {}).items():
            if not is_stopword(nb) and len(nb) >= 3:
                if flat_neighbors.get(nb, 0.0) < weight:
                    flat_neighbors[nb] = weight
                    neighbor_source[nb] = w

    max_nb = max(flat_neighbors.values(), default=1.0)

    return QueryProfile(
        words=words,
        word_set=frozenset(w for w in words if w.isalpha()),
        coords=coords,
        pins=pins,
        leg_sum_pins=leg_sum_pins,
        idf=idf,
        flat_neighbors=flat_neighbors,
        neighbor_source=neighbor_source,
        max_neighbor_weight=max_nb,
        wing_coords=wing_coords,
        word_pool_factors=word_pool_factors,
    )


# ---------------------------------------------------------------------------
# Scoring — O(Q × K) instead of O(Q × D_tokens)
# ---------------------------------------------------------------------------

def _l01_meet_query_word(
    profile: QueryProfile,
    hub_entry: HubEntry,
    hub_word: str,
) -> str:
    """Query word that L01-meets hub_word, or '' if none."""
    if hub_entry.pin is not None and profile.pins:
        q_word = profile.pins.get(hub_entry.pin, "")
        if q_word and q_word != hub_word:
            return q_word
        if profile.leg_sum_pins:
            lsk = leg_sum_meet_key(hub_entry.pin)
            q_word = profile.leg_sum_pins.get(lsk, "")
            if q_word and q_word != hub_word:
                return q_word
    if hub_entry.coord and hub_entry.coord in profile.coords:
        q_word = profile.coords[hub_entry.coord]
        if q_word and q_word != hub_word:
            return q_word
    return ""


def _l01_coord_meet_for_sig(profile: QueryProfile, sig: LatticeHubSignature) -> float:
    """Binary L01 coord meet when wing maps are empty (legacy path)."""
    bonus = 0.0
    if sig.hub_pins:
        for pin, hub_word in sig.hub_pins.items():
            q_word = profile.pins.get(pin, "")
            if not q_word or q_word == hub_word:
                q_word = profile.leg_sum_pins.get(leg_sum_meet_key(pin), "")
            if q_word and q_word != hub_word:
                bonus += profile.idf.get(q_word, 1.0) * LAMBDA_COORD
        return bonus
    for coord, hub_word in sig.hub_coords.items():
        if coord in profile.coords:
            q_word = profile.coords[coord]
            if q_word != hub_word:
                bonus += profile.idf.get(q_word, 1.0) * LAMBDA_COORD
    return bonus


def prime_factor_meet_score(
    profile: QueryProfile,
    sig: LatticeHubSignature,
) -> float:
    """
    Jaccard on pool prime factors only (≥ MIN_POOL_PRIME).

    Skips query/hub pairs where either side has no promoted pool primes,
    so letter-prime leakage cannot match every document.
    """
    if not profile.word_pool_factors or not sig.hubs:
        return 0.0
    score = 0.0
    for qw, q_factors in profile.word_pool_factors.items():
        if qw not in profile.word_set:
            continue
        best = 0.0
        for entry in sig.hubs.values():
            if not entry.pool_factors:
                continue
            sim = pool_factor_jaccard(q_factors, entry.pool_factors)
            if sim > best:
                best = sim
        if best > 0:
            score += profile.idf.get(qw, 1.0) * best * LAMBDA_PRIME_FACTOR
    return score


def score_document(
    profile: QueryProfile,
    doc_tokens: frozenset[str],
    sig: LatticeHubSignature | None,
    *,
    doc_tf: dict[str, int] | None = None,
    doc_len: int = 0,
    avg_dl: float = 100.0,
    k1: float = 1.5,
    b: float = 0.75,
    doc_composites: list[tuple[str, int]] | None = None,
    composite_cache: dict[str, int] | None = None,
    registry=None,
) -> float:
    """
    Four-signal score.  O(Q + K) per document.

    1. BM25 TF-IDF         — full BM25 formula (lexical backbone)
    2. Coord meet          — query coord == hub coord (geometric, novel)
    3. Neighbor expansion  — log-norm L4-L6 neighbors ∩ hub words
    4. Morphological meet  — letter prime GCD ratio ≥ threshold (composite anchors)
    5b. Prime-factor Jaccard — scale-invariant overlap on lattice composite products

    Signal 4 connects morphological variants ("autophagy"↔"autophagic") via
    shared letter prime factors — geometric stemming without heuristic suffix rules.
    """
    if not profile.words:
        return 0.0

    score = 0.0

    # Signal 1 — BM25 TF-IDF  O(Q)
    for w in profile.word_set:
        if w not in doc_tokens:
            continue
        idf = profile.idf.get(w, 1.0)
        if doc_tf is not None and doc_len > 0:
            tf = doc_tf.get(w, 0)
            norm = tf * (k1 + 1.0) / (tf + k1 * (1.0 - b + b * doc_len / avg_dl))
        else:
            norm = 1.0
        score += idf * norm

    if sig is None:
        return score

    # Signal 2 — 32-wing consensus coord meet  O(K × |CONSENSUS_WINGS|)
    # A genuine swap-meet fires on ALL wings; coincidental L01 collisions fire on fewer.
    # Score = IDF(q_word) × (matching_wings / total_wings) × LAMBDA_COORD.
    # This is a continuous signal in [0,1]×IDF×λ — not binary.
    n_wings = len(CONSENSUS_WINGS)
    if profile.wing_coords and n_wings > 0:
        for hub_word, hub_entry in sig.hubs.items():
            hub_wc = hub_entry.resolve_wing_coords(registry)
            if not hub_wc:
                q_word = _l01_meet_query_word(profile, hub_entry, hub_word)
                if q_word:
                    score += profile.idf.get(q_word, 1.0) * LAMBDA_COORD
                continue
            # Count how many wings agree: same coord for different words
            wing_matches = 0
            matched_q_word = ""
            for lid, hub_coord in hub_wc.items():
                q_wing = profile.wing_coords.get(lid, {})
                q_word = q_wing.get(hub_coord, "")
                if q_word and q_word != hub_word:
                    wing_matches += 1
                    matched_q_word = q_word  # any matching word's IDF
            if wing_matches > 0:
                score += profile.idf.get(matched_q_word, 1.0) * LAMBDA_COORD * (wing_matches / n_wings)
    else:
        score += _l01_coord_meet_for_sig(profile, sig)

    # Signal 3 — IDF-weighted neighbor expansion  O(K)
    # Normalize raw correlation weight to [0,1] so it can't dominate BM25.
    # Weight by IDF of the query word that triggered each neighbor.
    max_nb = profile.max_neighbor_weight or 1.0
    for hub_word in sig.hub_words():
        nb_weight = profile.flat_neighbors.get(hub_word, 0.0)
        if nb_weight > 0:
            q_word = profile.neighbor_source.get(hub_word, "")
            idf_q = profile.idf.get(q_word, 1.0)
            sat = min(1.0, nb_weight / max_nb)   # saturate to [0,1]
            score += idf_q * sat * LAMBDA_NEIGHBOR

    # Signal 4 — Morphological meet via composite prime GCD  O(Q × K)
    if doc_composites and composite_cache is not None:
        score += morph_meet_score(
            list(profile.word_set),
            profile.idf,
            doc_composites,
            composite_cache,
        )

    # Signal 5b — pool-prime factor Jaccard (≥ MIN_POOL_PRIME)  O(Q × K)
    if profile.word_pool_factors and sig is not None:
        score += prime_factor_meet_score(profile, sig)

    # Signal 7 — L7-L9 cluster routing bonus  O(K)
    # ``query_cluster_ids`` maps hub words that share an L7-L9 cluster with any
    # query word → the IDF of the best query word in that cluster.
    # Precomputed once per query in eval_beir.py using pipe.reader.word_to_cluster.
    # This breaks ranking ties in PARTIAL queries where BM25 is similar but the
    # correct doc shares a topical cluster with the query.
    if profile.query_cluster_ids and sig is not None:
        for hub_word in sig.hub_words():
            cluster_idf = profile.query_cluster_ids.get(hub_word, 0.0)
            if cluster_idf > 0:
                score += cluster_idf * LAMBDA_CLUSTER

    return score


def score_hub_signature(
    profile: QueryProfile,
    sig: LatticeHubSignature,
) -> float:
    """Compatibility shim — delegates to score_document with hub-only tokens."""
    return score_document(profile, sig.hub_words(), sig)


# ---------------------------------------------------------------------------
# Fast rank — O(candidates × K)
# ---------------------------------------------------------------------------

# Gate threshold: suppress lattice signals for docs whose BM25 is below this
# fraction of the best BM25 score in the candidate set.  Prevents neighbor /
# correlation noise from lifting zero-lexical-overlap docs above well-matching
# ones (primary FiQA failure mode).
BM25_GATE_THRESHOLD = 0.12

# BM25 dominance cap: when a query has strong lexical signal (max_bm25 >= this),
# a doc's total score can't exceed (cap_multiplier × its BM25 score).
# This prevents weak-BM25 docs from overtaking strong-BM25 docs purely on
# neighbor/cluster noise — the primary cause of SCORE_MISS failures.
BM25_DOMINANCE_THRESHOLD = 5.0   # queries with max_bm25 >= this use the cap
BM25_DOMINANCE_CAP = 4.5          # weak-BM25 docs capped at 4.5× their BM25

# Two-tier rank: cheap BM25+κ on all capped candidates; full lattice on top-N only.
FULL_SCORE_LIMIT = 120


def _bm25_for_doc(
    profile: QueryProfile,
    did: str,
    *,
    doc_tokens: dict[str, frozenset[str]] | None,
    doc_tf: dict[str, dict[str, int]] | None,
    doc_len: dict[str, int] | None,
    avg_dl: float,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    tokens = doc_tokens[did] if doc_tokens and did in doc_tokens else frozenset()
    tf_d = doc_tf[did] if doc_tf and did in doc_tf else None
    dl_d = doc_len[did] if doc_len and did in doc_len else 0
    s1 = 0.0
    for w in profile.word_set:
        if w not in tokens:
            continue
        idf = profile.idf.get(w, 1.0)
        if tf_d and dl_d:
            tf = tf_d.get(w, 0)
            norm = tf * (k1 + 1.0) / (tf + k1 * (1.0 - b + b * dl_d / avg_dl))
        else:
            norm = 1.0
        s1 += idf * norm
    return s1


def _select_full_score_pool(
    candidates: list[str],
    bm25_scores: dict[str, float],
    *,
    attractor_index,
    query_kappa_keys,
    limit: int,
    protect: Iterable[str] = (),
) -> set[str]:
    """Top ``limit`` docs by BM25 + κ overlap; protected docs always included."""
    protect_set = set(protect) & set(candidates)
    if len(candidates) <= limit:
        return set(candidates)
    if attractor_index is not None and query_kappa_keys:
        pre_ranked = attractor_index.rank_docs_by_overlap(
            query_kappa_keys,
            candidate_doc_ids=candidates,
        )
        overlap_rank = {doc_id: i for i, (_, doc_id) in enumerate(pre_ranked)}
        order = sorted(
            candidates,
            key=lambda did: (
                -bm25_scores.get(did, 0.0),
                overlap_rank.get(did, len(candidates)),
                did,
            ),
        )
    else:
        order = sorted(
            candidates,
            key=lambda did: (-bm25_scores.get(did, 0.0), did),
        )
    picked = set(order[:limit]) | protect_set
    if len(picked) <= limit:
        return picked
    # Too many protected — keep all protected + top BM25 from rest
    extra = limit - len(protect_set)
    if extra <= 0:
        return protect_set
    rest = [d for d in order if d not in protect_set][:extra]
    return protect_set | set(rest)


def rank_with_hub_signatures(
    profile: QueryProfile,
    candidate_ids: list[str],
    hub_sigs: dict[str, LatticeHubSignature],
    all_ids: list[str],
    *,
    doc_tokens: dict[str, frozenset[str]] | None = None,
    doc_tf: dict[str, dict[str, int]] | None = None,
    doc_len: dict[str, int] | None = None,
    avg_dl: float = 100.0,
    composite_index=None,
    sub_comp_idx=None,
    registry=None,
    phrase_idx=None,
    anchor_idx=None,
    query_anchor_comps: "dict[int, float] | None" = None,
    query_phrase_comps: "dict[int, float] | None" = None,
    attractor_index=None,
    query_kappa_keys=None,
    kappa_candidate_cap: int = 0,
    protect_doc_ids: Iterable[str] | None = None,
    top_k: int = 100,
) -> list[str]:
    """
    Score candidate docs; return top_k ranked ids.

    BM25 gating: lattice Signals 2–8 are suppressed for docs whose BM25
    (Signal 1) score is below BM25_GATE_THRESHOLD × max_bm25_in_candidates.
    This prevents correlation/neighbor noise from reordering docs that have
    no lexical overlap with the query.

    Signal 8a (BIT 10): when ``attractor_index`` and ``query_kappa_keys`` are
    set, adds κ Jaccard bonus inside the BM25-gated branch. Optional
    ``kappa_candidate_cap`` pre-filters candidates by overlap before Pass A.
    """
    from pipeline.bit_10_score_fusion import (
        cap_candidates_by_kappa_overlap,
        signal_8a_kappa_jaccard,
    )

    cache = composite_index.word_cache if composite_index else {}

    candidates = list(candidate_ids)
    protect = list(protect_doc_ids or ())
    if kappa_candidate_cap > 0 and attractor_index is not None and query_kappa_keys:
        candidates = cap_candidates_by_kappa_overlap(
            candidates,
            attractor_index,
            query_kappa_keys,
            cap=kappa_candidate_cap,
            protect=protect,
        )

    # --- Pass A: BM25-only scores for gating threshold ---
    bm25_scores: dict[str, float] = {}
    for did in candidates:
        bm25_scores[did] = _bm25_for_doc(
            profile,
            did,
            doc_tokens=doc_tokens,
            doc_tf=doc_tf,
            doc_len=doc_len,
            avg_dl=avg_dl,
        )
    max_bm25 = max(bm25_scores.values(), default=0.0)

    # Gate only when the query HAS some lexical signal (max_bm25 above a floor).
    # If the entire candidate set has near-zero BM25 (zero-vocabulary-overlap query),
    # the lattice signals (neighbors, composites) are the ONLY path to the correct doc.
    # Suppressing them when BM25=0 would eliminate all non-lexical retrieval.
    BM25_QUERY_FLOOR = 1.0   # below this, treat query as zero-lexical; skip gating
    lattice_gating_active = max_bm25 >= BM25_QUERY_FLOOR
    bm25_gate = max_bm25 * BM25_GATE_THRESHOLD if lattice_gating_active else 0.0

    full_score_ids = _select_full_score_pool(
        candidates,
        bm25_scores,
        attractor_index=attractor_index,
        query_kappa_keys=query_kappa_keys,
        limit=FULL_SCORE_LIMIT,
        protect=protect,
    )

    # --- Pass B: two-tier scoring with BM25 gating ---
    scored: list[tuple[float, str]] = []
    for did in candidates:
        s1 = bm25_scores.get(did, 0.0)
        if lattice_gating_active and s1 < bm25_gate:
            if s1 > 0:
                scored.append((s1, did))
            continue

        if did not in full_score_ids:
            s = s1 + signal_8a_kappa_jaccard(
                profile,
                did,
                attractor_index,
                query_kappa_keys,
                lambda_kappa=LAMBDA_KAPPA,
            )
            if s > 0:
                scored.append((s, did))
            continue

        sig = hub_sigs.get(did)
        tokens = (doc_tokens[did] if doc_tokens and did in doc_tokens
                  else (sig.hub_words() if sig else frozenset()))
        tf = doc_tf[did] if doc_tf and did in doc_tf else None
        dl = doc_len[did] if doc_len and did in doc_len else 0
        dc = composite_index.doc_hub_composites.get(did) if composite_index else None
        s = score_document(
            profile, tokens, sig,
            doc_tf=tf, doc_len=dl, avg_dl=avg_dl,
            doc_composites=dc, composite_cache=cache,
            registry=registry,
        )
        # Signal 4: subword composite origin score
        if sub_comp_idx is not None and registry is not None:
            from aethos_subword_composite import subword_composite_score
            s += subword_composite_score(
                list(profile.word_set), profile.idf, registry, sub_comp_idx, did,
                word_cache=sub_comp_idx.word_composites,
            )
        # Signal 5: phrase composite score — fast path uses precomputed per-query dict
        if query_phrase_comps is not None and phrase_idx is not None:
            from aethos_phrase_composite import phrase_composite_score_fast
            s += phrase_composite_score_fast(query_phrase_comps, did, phrase_idx)
        elif phrase_idx is not None and registry is not None:
            from aethos_phrase_composite import phrase_composite_score
            s += phrase_composite_score(
                list(profile.word_set), profile.idf, registry, phrase_idx, did,
            )
        # Signal 6: discriminative heavy anchor score (precomputed per-query)
        if query_anchor_comps is not None and anchor_idx is not None:
            from aethos_discriminative import score_with_heavy_anchors
            s += score_with_heavy_anchors(query_anchor_comps, did, anchor_idx)

        # Signal 8a: κ Jaccard (BIT 10) — BM25-gated branch only
        s += signal_8a_kappa_jaccard(
            profile,
            did,
            attractor_index,
            query_kappa_keys,
            lambda_kappa=LAMBDA_KAPPA,
        )

        # Lattice noise cap: non-BM25 signals cannot dominate when doc has lexical match
        if lattice_gating_active and s1 > 0 and s > s1:
            max_extra = s1 * 3.0
            if s - s1 > max_extra:
                s = s1 + max_extra

        # BM25 dominance cap (Fix 2a): when the query has strong lexical signal,
        # prevent weak-BM25 docs from overtaking strong-BM25 docs purely via
        # lattice noise.  The cap is: total ≤ BM25_DOMINANCE_CAP × doc_bm25.
        # This targets the SCORE_MISS pattern: wrong doc BM25=22, correct BM25=6.6,
        # but wrong doc also gets more neighbor score.
        if max_bm25 >= BM25_DOMINANCE_THRESHOLD and s1 > 0:
            max_allowed = s1 * BM25_DOMINANCE_CAP
            if s > max_allowed:
                s = max_allowed

        if s > 0:
            scored.append((s, did))

    scored.sort(key=lambda x: (-x[0], x[1]))
    ranked = [d for _, d in scored]

    if len(ranked) < top_k:
        seen = set(ranked)
        for did in all_ids:
            if did not in seen:
                ranked.append(did)
            if len(ranked) >= top_k:
                break

    return ranked[:top_k]


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def signature_report(sig: LatticeHubSignature) -> str:
    lines = [f"LatticeHubSignature  doc={sig.doc_id!r}  hubs={len(sig.hubs)}  ~{sig.encoded_size()} B"]
    for word, entry in sorted(sig.hubs.items(), key=lambda x: -x[1].strength):
        nb_str = ",".join(sorted(entry.neighbors)[:4])
        lines.append(
            f"  {word:<16} strength={entry.strength:.1f}  prime={entry.prime}"
            f"  coord={entry.coord}  nb=[{nb_str}]"
        )
    return "\n".join(lines)
