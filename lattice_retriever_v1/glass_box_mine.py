"""
Glass-box mine — 25+ diagnostic lenses on lattice_retriever_v1 traces.

Each lens answers: "how many gold docs would this signal recover?"
Used for correlation/routing tuning — not for peeking at gold during training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from lattice_retriever_v1.stage07_semantic_light import HUB_WORDS
from lattice_retriever_v1.stage08_retrieve import LatticeRetriever


@dataclass(frozen=True)
class LensResult:
    id: str
    name: str
    description: str
    gold_hits: int
    n_queries: int
    query_ids: tuple[str, ...] = ()

    @property
    def rate(self) -> float:
        return self.gold_hits / self.n_queries if self.n_queries else 0.0

    def explain(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "gold_hits": self.gold_hits,
            "n_queries": self.n_queries,
            "rate": round(self.rate, 4),
        }


@dataclass
class QueryMineRecord:
    qid: str
    query: str
    gold_ids: tuple[str, ...]
    pool: tuple[str, ...]
    ranked: tuple[str, ...]
    route_mode: str
    pool_size: int
    gold_in_pool: bool
    gold_ranks: dict[str, int | None]
    terms_by_rarity: tuple[tuple[str, int], ...]
    rarest_term: str
    second_rarest: str
    routing_pins: tuple[tuple[str, int, int], ...]
    compound_pairs: tuple[tuple[str, str, int, int], ...]
    top10_ids: tuple[str, ...]
    buckets: dict[str, bool] = field(default_factory=dict)
    bridge_terms: tuple[str, ...] = ()
    lift_pins_hit_gold: tuple[int, ...] = ()
    cage_bridge_terms: tuple[str, ...] = ()
    subword_bridge_terms: tuple[str, ...] = ()


def _query_words(r: LatticeRetriever, query: str) -> list[str]:
    return [w for w in r._words(query) if len(w) >= 2]


def terms_by_rarity(r: LatticeRetriever, query: str) -> list[tuple[str, int]]:
    terms = _query_words(r, query)
    hubs = [t for t in terms if t in HUB_WORDS]
    rare = [t for t in terms if t not in HUB_WORDS]
    rare.sort(key=lambda w: (r.semantic.doc_freq.get(w, 0), w))
    hubs.sort(key=lambda w: (r.semantic.doc_freq.get(w, 0), w))
    return [(w, r.semantic.doc_freq.get(w, 0)) for w in rare + hubs]


def docs_for_term(r: LatticeRetriever, term: str) -> set[str]:
    return r._postings_for_routing_pin(r._routing_pin_for_term(term)[0])


def docs_for_pin(r: LatticeRetriever, pin: int) -> set[str]:
    return set(r.postings.get(pin, set()))


def docs_for_compound(r: LatticeRetriever, w1: str, w2: str) -> set[str]:
    pin = r._compound_pin(w1, w2)
    return docs_for_pin(r, pin)


def doc_has_term(r: LatticeRetriever, doc_id: str, term: str) -> bool:
    doc = r.docs.get(doc_id)
    return doc is not None and term in doc.words


def doc_pin_overlap(r: LatticeRetriever, doc_id: str, query: str) -> int:
    doc = r.docs.get(doc_id)
    if doc is None:
        return 0
    q_pins: set[int] = set()
    for t in _query_words(r, query):
        q_pins |= r._corridor_pins(t)
    return len(q_pins & set(doc.corridor_pins))


def query_lit_cages(r: LatticeRetriever, query: str) -> list:
    """3-way cages lit by query sliding window (Phase B surface)."""
    terms = r._query_terms(query)
    window = terms[:6]
    out = []
    seen: set[int] = set()
    for i in range(max(0, len(window) - 2)):
        triple = window[i : i + 3]
        comp = r.semantic._cage_for_triple(*triple).anchor_composite
        if comp in seen:
            continue
        cage = r.semantic.cages.get(comp)
        if cage is not None and cage.correlations:
            out.append(cage)
            seen.add(comp)
    return out


def doc_lit_cages(r: LatticeRetriever, doc_id: str, *, max_words: int = 32) -> list:
    doc = r.docs.get(doc_id)
    if doc is None:
        return []
    words = list(doc.words)[:max_words]
    out = []
    seen: set[int] = set()
    for i in range(max(0, len(words) - 2)):
        triple = words[i : i + 3]
        comp = r.semantic._cage_for_triple(*triple).anchor_composite
        if comp in seen:
            continue
        cage = r.semantic.cages.get(comp)
        if cage is not None and cage.correlations:
            out.append(cage)
            seen.add(comp)
    return out


def cage_rare_neighbors_for_gold(
    r: LatticeRetriever,
    cages: list,
    gold_words: set[str],
    query_words: set[str],
    *,
    exclude_query_lexical: bool = True,
) -> list[str]:
    """
    Rare correlated terms in lit cages that appear in gold.
    exclude_query_lexical: neighbor not already in query (non-lexical bridge).
    """
    hits: list[str] = []
    for cage in cages:
        for term in cage.correlations:
            if not r.semantic.is_rare(term):
                continue
            if term not in gold_words:
                continue
            if exclude_query_lexical and term in query_words:
                continue
            hits.append(term)
    return sorted(set(hits))


def promoted_subword_mismatch_terms(
    r: LatticeRetriever, query: str, gold_id: str
) -> list[str]:
    """Gold surface words sharing promoted subword with query term but not in query."""
    doc = r.docs.get(gold_id)
    if doc is None:
        return []
    q_words = set(_query_words(r, query))
    gold_words = set(doc.words)
    hits: list[str] = []
    for qt in _query_words(r, query):
        for ln in range(2, min(5, len(qt) + 1)):
            for sw in (qt[:ln], qt[-ln:]):
                if r.semantic.registry.promoted_subword(sw) is None:
                    continue
                for gw in gold_words:
                    if gw in q_words or gw == qt:
                        continue
                    if sw in gw or gw.startswith(sw) or gw.endswith(sw):
                        hits.append(gw)
    return sorted(set(hits))


def shared_rare_terms(r: LatticeRetriever, doc_a: str, doc_b: str) -> list[str]:
    da, db = r.docs.get(doc_a), r.docs.get(doc_b)
    if da is None or db is None:
        return []
    wa, wb = set(da.words), set(db.words)
    shared = wa & wb
    return sorted(
        t for t in shared
        if r.semantic.is_rare(t) and t not in HUB_WORDS
    )


def mine_query(
    r: LatticeRetriever,
    qid: str,
    query: str,
    gold_ids: Iterable[str],
) -> QueryMineRecord:
    gold = tuple(sorted(gold_ids))
    trace = r.retrieve_with_trace(query, limit=10)
    pool, route_mode, steps, _, _ = r.route_pool(query)
    ranked = tuple(h.doc_id for h in trace.hits)
    top10 = ranked[:10]

    rarity = terms_by_rarity(r, query)
    rarest = rarity[0][0] if rarity else ""
    second = rarity[1][0] if len(rarity) > 1 else ""

    routing_pins: list[tuple[str, int, int]] = []
    for s in steps:
        if s.get("step") == "rarest_filter":
            term = s.get("term") or (s.get("terms") or [""])[0]
            routing_pins.append(
                (term, s.get("routing_pin", 0), s.get("pin_doc_freq", 0))
            )

    words = [w for w, _ in rarity]
    compounds: list[tuple[str, str, int, int]] = []
    for i in range(len(words) - 1):
        w1, w2 = words[i], words[i + 1]
        cpin = r._compound_pin(w1, w2)
        compounds.append((w1, w2, cpin, r.pin_doc_freq.get(cpin, 0)))

    gold_ranks = {g: (ranked.index(g) + 1 if g in ranked else None) for g in gold}
    pool_set = set(pool)
    buckets: dict[str, bool] = {}

    def any_gold(pred) -> bool:
        return any(pred(g) for g in gold)

    buckets["L01_gold_in_pool"] = any_gold(lambda g: g in pool_set)
    buckets["L02_gold_in_top10"] = any_gold(lambda g: g in top10)
    buckets["L03_gold_rank1"] = any_gold(lambda g: gold_ranks.get(g) == 1)

    if rarest:
        buckets["L04_gold_has_rarest_term"] = any_gold(lambda g: doc_has_term(r, g, rarest))
        buckets["L05_rarest_pin_hits_gold"] = bool(set(gold) & docs_for_term(r, rarest))
    if second:
        buckets["L06_gold_has_2nd_rarest"] = any_gold(lambda g: doc_has_term(r, g, second))
        buckets["L07_2nd_rarest_pin_hits_gold"] = bool(set(gold) & docs_for_term(r, second))

    for i, (term, _) in enumerate(rarity[:5]):
        buckets[f"L{8 + i:02d}_rank{i + 1}_term_in_gold"] = any_gold(
            lambda g, t=term: doc_has_term(r, g, t)
        )

    buckets["L13_gold_via_compound_bigram"] = False
    buckets["L14_compound_reaches_gold"] = False
    for w1, w2, cpin, _cdf in compounds:
        if set(gold) & docs_for_pin(r, cpin):
            buckets["L13_gold_via_compound_bigram"] = True
            buckets["L14_compound_reaches_gold"] = True

    buckets["L15_gold_identity_pin_overlap"] = any_gold(
        lambda g: doc_pin_overlap(r, g, query) >= len(_query_words(r, query)) // 2 + 1
    )
    buckets["L16_gold_any_pin_overlap"] = any_gold(lambda g: doc_pin_overlap(r, g, query) > 0)

    buckets["L17_rescuable_or_rarest_2nd_pin"] = False
    if rarest and second:
        union = docs_for_term(r, rarest) | docs_for_term(r, second)
        buckets["L17_rescuable_or_rarest_2nd_pin"] = bool(set(gold) & union) and not buckets.get(
            "L01_gold_in_pool", False
        )

    buckets["L18_gold_missed_empty_pool"] = not pool_set and any(gold)
    buckets["L19_gold_in_pool_bad_rank"] = any(
        g in pool_set and (gold_ranks.get(g) is None or gold_ranks.get(g) > 10) for g in gold
    )

    buckets["L20_gold_wing_cage_touch"] = any(
        h.doc_id in gold and any(x.get("kind") == "wing_cage" for x in h.reasons)
        for h in trace.hits
    )
    buckets["L21_gold_corridor_open"] = any(
        h.doc_id in gold and any(x.get("kind") == "corridor_open" for x in h.reasons)
        for h in trace.hits
    )

    buckets["L22_top1_shares_rare_with_gold"] = False
    bridge: list[str] = []
    if top10:
        t0 = top10[0]
        for g in gold:
            bridge.extend(shared_rare_terms(r, t0, g))
        bridge = sorted(set(bridge))
        buckets["L22_top1_shares_rare_with_gold"] = bool(bridge)

    buckets["L23_top10_bridge_to_gold"] = False
    for td in top10[:5]:
        if td in gold:
            continue
        for g in gold:
            if shared_rare_terms(r, td, g):
                buckets["L23_top10_bridge_to_gold"] = True
                break

    buckets["L24_gold_would_enter_widen_only"] = route_mode == "widen_rarest" and buckets.get(
        "L01_gold_in_pool", False
    )
    buckets["L25_gold_hub_term_blocks_intersect"] = False
    hub_in_query = [t for t, _ in rarity if t in HUB_WORDS or not r._term_narrows(t)]
    if hub_in_query and not buckets.get("L01_gold_in_pool") and any(gold):
        selective_only = [t for t, _ in rarity if r._term_narrows(t)]
        if selective_only:
            pool_sel: set[str] | None = None
            for t in selective_only:
                b = docs_for_term(r, t)
                pool_sel = b if pool_sel is None else pool_sel & b
            if pool_sel and set(gold) & pool_sel:
                buckets["L25_gold_hub_term_blocks_intersect"] = True

    lift_hits: list[int] = []
    for t in _query_words(r, query):
        _, selective, lift = r._split_pins(t)
        for lp in lift:
            if set(gold) & docs_for_pin(r, lp):
                lift_hits.append(lp)
    buckets["L26_gold_via_lift_pin_only"] = bool(lift_hits) and not buckets.get("L01_gold_in_pool")

    buckets["L27_morph_prefix_stem_shared"] = False
    for g in gold:
        dg = r.docs.get(g)
        if dg is None:
            continue
        for qt in _query_words(r, query):
            for ln in range(3, min(5, len(qt) + 1)):
                stem = qt[:ln]
                if stem in dg.words or any(w.startswith(stem) for w in dg.words):
                    if r.semantic.registry.promoted_subword(stem):
                        buckets["L27_morph_prefix_stem_shared"] = True

    buckets["L28_oracle_any_selective_pin"] = False
    for g in gold:
        for t in _query_words(r, query):
            _, selective, _ = r._split_pins(t)
            for pin in selective:
                if g in docs_for_pin(r, pin):
                    buckets["L28_oracle_any_selective_pin"] = True
                    break

    buckets["L29_gold_same_cage_anchor_as_query"] = bool(trace.cages_considered) and any_gold(
        lambda g: any(t in r.docs[g].words for t in _query_words(r, query)) if g in r.docs else False
    )

    buckets["L30_rescore_headroom_in_pool"] = buckets.get("L01_gold_in_pool") and buckets.get(
        "L19_gold_in_pool_bad_rank", False
    )

    q_words = set(_query_words(r, query))
    lit_cages = query_lit_cages(r, query)
    cage_bridge: list[str] = []
    buckets["L31_cage_rare_neighbor_in_gold"] = False
    buckets["L32_gold_cage_lights_query_rare"] = False
    buckets["L36_nonlexical_cage_path"] = False
    for g in gold:
        gw = set(r.docs[g].words) if g in r.docs else set()
        neighbors = cage_rare_neighbors_for_gold(r, lit_cages, gw, q_words, exclude_query_lexical=True)
        if neighbors:
            buckets["L31_cage_rare_neighbor_in_gold"] = True
            cage_bridge.extend(neighbors)
        if rarest and any(
            rarest in c.correlations and r.semantic.is_rare(rarest) for c in doc_lit_cages(r, g)
        ):
            buckets["L32_gold_cage_lights_query_rare"] = True
        lexical_shared = {t for t in q_words if t in gw and r.semantic.is_rare(t)}
        if neighbors and not lexical_shared:
            buckets["L36_nonlexical_cage_path"] = True

    subword_bridge: list[str] = []
    buckets["L33_subword_surface_mismatch"] = False
    buckets["L34_gold_via_subword_pin"] = False
    for g in gold:
        sw_hits = promoted_subword_mismatch_terms(r, query, g)
        if sw_hits:
            buckets["L33_subword_surface_mismatch"] = True
            subword_bridge.extend(sw_hits)
        for qt in _query_words(r, query):
            for ln in range(2, min(5, len(qt) + 1)):
                for sw in (qt[:ln], qt[-ln:]):
                    tok = r.semantic.registry.promoted_subword(sw)
                    if tok and g in docs_for_pin(r, tok.prime):
                        buckets["L34_gold_via_subword_pin"] = True

    buckets["L35_shared_cage_anchor_top10"] = False
    gold_cage_comps: set[int] = set()
    for g in gold:
        for c in doc_lit_cages(r, g):
            gold_cage_comps.add(c.anchor_composite)
    for td in top10[:5]:
        if td in gold:
            continue
        for c in doc_lit_cages(r, td):
            if c.anchor_composite in gold_cage_comps:
                buckets["L35_shared_cage_anchor_top10"] = True
                break

    buckets["L37_compound_constituents_in_gold"] = False
    for w1, w2, cpin, _cdf in compounds:
        if cpin and any(
            g in r.docs and w1 in r.docs[g].words and w2 in r.docs[g].words for g in gold
        ):
            buckets["L37_compound_constituents_in_gold"] = True

    buckets["L38_rare_query_term_cage_linked_gold"] = False
    if rarest:
        for g in gold:
            gw = set(r.docs[g].words) if g in r.docs else set()
            for cage in lit_cages:
                if rarest not in cage.correlations:
                    continue
                for term in cage.correlations:
                    if term != rarest and r.semantic.is_rare(term) and term in gw:
                        buckets["L38_rare_query_term_cage_linked_gold"] = True
                        if term not in cage_bridge:
                            cage_bridge.append(term)

    buckets["L39_scoring_fixable_by_cage"] = buckets.get("L30_rescore_headroom_in_pool") and (
        buckets.get("L31_cage_rare_neighbor_in_gold")
        or buckets.get("L32_gold_cage_lights_query_rare")
        or buckets.get("L38_rare_query_term_cage_linked_gold")
    )
    buckets["L40_scoring_fixable_by_lexical_bridge"] = buckets.get(
        "L30_rescore_headroom_in_pool"
    ) and buckets.get("L23_top10_bridge_to_gold")

    return QueryMineRecord(
        qid=qid,
        query=query,
        gold_ids=gold,
        pool=tuple(pool),
        ranked=ranked,
        route_mode=route_mode,
        pool_size=len(pool),
        gold_in_pool=buckets.get("L01_gold_in_pool", False),
        gold_ranks=gold_ranks,
        terms_by_rarity=tuple(rarity),
        rarest_term=rarest,
        second_rarest=second,
        routing_pins=tuple(routing_pins),
        compound_pairs=tuple(compounds),
        top10_ids=top10,
        buckets=buckets,
        bridge_terms=tuple(bridge),
        lift_pins_hit_gold=tuple(sorted(set(lift_hits))),
        cage_bridge_terms=tuple(sorted(set(cage_bridge))),
        subword_bridge_terms=tuple(sorted(set(subword_bridge))),
    )


def bucket_delta(before: QueryMineRecord, after: QueryMineRecord) -> dict[str, str]:
    """Per-bucket change: drained (True->False), gained, unchanged."""
    out: dict[str, str] = {}
    keys = set(before.buckets) | set(after.buckets)
    for k in keys:
        b, a = before.buckets.get(k, False), after.buckets.get(k, False)
        if b and not a:
            out[k] = "drained"
        elif not b and a:
            out[k] = "gained"
        elif b and a:
            out[k] = "unchanged"
        else:
            out[k] = "still_false"
    return out


def mine_diff_summary(
    before: list[QueryMineRecord],
    after: list[QueryMineRecord],
    *,
    target_buckets: tuple[str, ...] = ("L26_gold_via_lift_pin_only", "L01_gold_in_pool"),
) -> dict:
    """Aggregate before/after mine for experiment regression."""
    after_map = {r.qid: r for r in after}
    pool_before = sum(1 for r in before if r.gold_in_pool)
    pool_after = sum(1 for r in after if after_map.get(r.qid, r).gold_in_pool)
    drained: dict[str, list[str]] = {b: [] for b in target_buckets}
    for b in before:
        a = after_map.get(b.qid)
        if a is None:
            continue
        for bucket in target_buckets:
            if b.buckets.get(bucket) and not a.buckets.get(bucket):
                drained[bucket].append(b.qid)
    return {
        "n_queries": len(before),
        "pool_recall_before": round(pool_before / max(len(before), 1), 4),
        "pool_recall_after": round(pool_after / max(len(after), 1), 4),
        "drained": {k: {"count": len(v), "qids": v} for k, v in drained.items()},
    }


LENS_CATALOG: tuple[tuple[str, str, str], ...] = (
    ("L01_gold_in_pool", "Gold in route pool", "Current pool recall — gold entered Phase A pool"),
    ("L02_gold_in_top10", "Gold in top 10", "End-to-end R@10 success"),
    ("L03_gold_rank1", "Gold at rank 1", "Top-1 accuracy"),
    ("L04_gold_has_rarest_term", "Gold contains rarest query term", "Lexical overlap with rarest word"),
    ("L05_rarest_pin_hits_gold", "Rarest routing pin retrieves gold", "Current rarest-pin path reaches gold"),
    ("L06_gold_has_2nd_rarest", "Gold contains 2nd-rarest term", "Second anchor term overlap"),
    ("L07_2nd_rarest_pin_hits_gold", "2nd-rarest pin retrieves gold", "Alternate pin would route to gold"),
    ("L08_rank1_term_in_gold", "Rank-1 rarest term in gold doc", "Same as L04 for explicit rank slot"),
    ("L13_gold_via_compound_bigram", "Compound bigram pin hits gold", "Adjacent query pair compound reaches gold"),
    ("L14_compound_reaches_gold", "Compound pin reaches gold", "Bigram compound posting overlaps gold doc"),
    ("L15_gold_identity_pin_overlap", "High identity pin overlap with gold", "Strong corridor pin match"),
    ("L16_gold_any_pin_overlap", "Any corridor pin overlap with gold", "Weakest pin sharing"),
    ("L17_rescuable_or_rarest_2nd_pin", "Gold via OR of top-2 term pins, missed pool", "Headroom: widen to union of 2 rarest"),
    ("L18_gold_missed_empty_pool", "Empty pool with gold existing", "Total routing miss"),
    ("L19_gold_in_pool_bad_rank", "Gold in pool but ranked below 10", "Scoring headroom — routing OK"),
    ("L20_gold_wing_cage_touch", "Gold hit has wing-cage lift", "Drift layer contributed to gold score"),
    ("L21_gold_corridor_open", "Gold hit has lazy corridor open", "Corridor key fired on gold"),
    ("L22_top1_shares_rare_with_gold", "Top-1 doc shares rare term with gold", "Bridge doc pattern for rerank"),
    ("L23_top10_bridge_to_gold", "Top-5 non-gold shares rare with gold", "Cross-doc rare bridge in top results"),
    ("L24_gold_would_enter_widen_only", "Gold only after widen step", "Intersect too strict; widen saved it"),
    ("L25_gold_hub_term_blocks_intersect", "Hub term blocked intersect; selective terms OK", "Hub-skip would recover"),
    ("L26_gold_via_lift_pin_only", "Gold only on promiscuous lift pin", "Selectivity excluded gold pin"),
    ("L27_morph_prefix_stem_shared", "Promoted prefix stem in gold", "Morphological recovery possible"),
    ("L28_oracle_any_selective_pin", "Some selective pin reaches gold", "Upper bound on pin routing"),
    ("L29_gold_same_cage_anchor_as_query", "Query lit a cage gold doc sits in", "Semantic cage connection"),
    ("L30_rescore_headroom_in_pool", "Gold in pool but not top 10", "Tune scoring not routing"),
    (
        "L31_cage_rare_neighbor_in_gold",
        "Rare cage neighbor in gold, not in query",
        "Semantic correlation bridge — query lit cage, gold has correlated rare term",
    ),
    (
        "L32_gold_cage_lights_query_rare",
        "Gold cage correlates query rarest term",
        "Gold doc's wing cage holds query rare word as dot",
    ),
    (
        "L33_subword_surface_mismatch",
        "Promoted subword in query, different surface in gold",
        "Morph/subword bridge — run/running class",
    ),
    (
        "L34_gold_via_subword_pin",
        "Gold on promoted subword pin of query term",
        "Subword pin routing/scoring headroom",
    ),
    (
        "L35_shared_cage_anchor_top10",
        "Top-10 doc shares cage anchor with gold",
        "Cage-level cross-doc bridge for rerank",
    ),
    (
        "L37_compound_constituents_in_gold",
        "Gold has both compound constituents",
        "Compound split — constituents present, compound pin unused",
    ),
    (
        "L38_rare_query_term_cage_linked_gold",
        "Rarest query term cage-linked to rare gold term",
        "Direct rare-to-rare semantic edge via shared cage",
    ),
    (
        "L36_nonlexical_cage_path",
        "Cage path with no lexical rare overlap",
        "Pure correlation — no shared query/gold rare word",
    ),
    (
        "L39_scoring_fixable_by_cage",
        "In-pool miss fixable by cage correlation boost",
        "L30 + cage bridge — drift scoring lever",
    ),
    (
        "L40_scoring_fixable_by_lexical_bridge",
        "In-pool miss fixable by top-doc lexical bridge",
        "L30 + L23 — rerank through bridge doc",
    ),
)


def aggregate_lenses(records: list[QueryMineRecord]) -> list[LensResult]:
    n = len(records)
    catalog = {k: (name, desc) for k, name, desc in LENS_CATALOG}
    all_keys = set(catalog) | {k for rec in records for k in rec.buckets}
    out: list[LensResult] = []
    for key in sorted(all_keys):
        hits = sum(1 for rec in records if rec.buckets.get(key))
        name, desc = catalog.get(key, (key, ""))
        qids = tuple(rec.qid for rec in records if rec.buckets.get(key))
        out.append(LensResult(key, name, desc, hits, n, qids))
    return out


def headroom_summary(records: list[QueryMineRecord]) -> dict:
    """Queries recoverable by each fix class (not already in pool)."""
    n = len(records)
    missed = [r for r in records if not r.gold_in_pool]
    return {
        "n_queries": n,
        "pool_recall": round(sum(1 for r in records if r.gold_in_pool) / max(n, 1), 4),
        "top10_recall": round(sum(1 for r in records if r.buckets.get("L02_gold_in_top10")) / max(n, 1), 4),
        "missed_count": len(missed),
        "rescuable_or_2nd_pin": sum(1 for r in missed if r.buckets.get("L17_rescuable_or_rarest_2nd_pin")),
        "rescuable_hub_skip": sum(1 for r in missed if r.buckets.get("L25_gold_hub_term_blocks_intersect")),
        "rescuable_lift_pin": sum(1 for r in missed if r.buckets.get("L26_gold_via_lift_pin_only")),
        "rescuable_compound_route": sum(
            1 for r in missed if r.buckets.get("L13_gold_via_compound_bigram")
        ),
        "rescuable_widen_only": sum(1 for r in missed if r.buckets.get("L24_gold_would_enter_widen_only")),
        "scoring_only_in_pool": sum(1 for r in records if r.buckets.get("L30_rescore_headroom_in_pool")),
        "bridge_rerank_candidates": sum(1 for r in records if r.buckets.get("L23_top10_bridge_to_gold")),
        "cage_neighbor_bridge": sum(1 for r in records if r.buckets.get("L31_cage_rare_neighbor_in_gold")),
        "nonlexical_cage_path": sum(1 for r in records if r.buckets.get("L36_nonlexical_cage_path")),
        "subword_surface_bridge": sum(1 for r in records if r.buckets.get("L33_subword_surface_mismatch")),
        "scoring_fixable_cage": sum(1 for r in records if r.buckets.get("L39_scoring_fixable_by_cage")),
        "scoring_fixable_lexical_bridge": sum(
            1 for r in records if r.buckets.get("L40_scoring_fixable_by_lexical_bridge")
        ),
    }
