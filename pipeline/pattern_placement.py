"""
Pattern placement — place every retrieval signal + pipeline bit on gold vs false docs.

Runs across all queries at once and reports:
  - failure pattern taxonomy (candidate miss, score miss, false positive win, …)
  - per-signal lift: mean(gold) − mean(false top-1) where ranking failed
  - routing coverage: BIT 4 tier, |C|, gold-in-pool recall
  - tuner hints: which λ / BIT to adjust next

Companion CLI: ``pattern_audit.py``
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Sequence

from aethos_discriminative import query_anchor_composites, score_with_heavy_anchors
from aethos_hub_signature import (
    CONSENSUS_WINGS,
    QueryProfile,
    _l01_coord_meet_for_sig,
    _l01_meet_query_word,
    build_query_profile,
    prime_factor_meet_score,
)
from aethos_phrase_composite import phrase_composite_score_fast
from aethos_subword_composite import subword_composite_score
from eval_beir import (
    DEFAULT_KAPPA_CANDIDATE_CAP,
    _query_phrase_composites,
    _score_one_query,
    ndcg_at_k,
    recall_at_k,
)
from pipeline.bit_09_query_cell_profile import build_query_cell_profile
from pipeline.bit_10_score_fusion import signal_8a_kappa_jaccard

import aethos_hub_signature as _hs

SIGNAL_NAMES: tuple[str, ...] = (
    "s1_bm25",
    "s2_coord",
    "s3_neighbors",
    "s4_subword",
    "s5_phrase",
    "s5b_prime_factor",
    "s6_anchors",
    "s7_cluster",
    "s8a_kappa",
)

FAILURE_PATTERNS: tuple[str, ...] = (
    "PERFECT",
    "PARTIAL",
    "MISSED_CANDIDATE",
    "ZERO_BM25",
    "SCORE_MISS",
    "RANK_LOW",
    "NO_GOLD_IN_CORPUS",
)


@dataclass
class PatternBreakdown:
    """All scoring + pipeline signals for one (query, doc) pair."""

    doc_id: str = ""
    s1_bm25: float = 0.0
    s2_coord: float = 0.0
    s3_neighbors: float = 0.0
    s4_subword: float = 0.0
    s5_phrase: float = 0.0
    s5b_prime_factor: float = 0.0
    s6_anchors: float = 0.0
    s7_cluster: float = 0.0
    s8a_kappa: float = 0.0
    kappa_jaccard: float = 0.0
    rank: int | None = None
    in_candidates: bool = False
    bm25_word_overlap: int = 0
    neighbor_word_overlap: int = 0
    total: float = 0.0

    def signal_dict(self) -> dict[str, float]:
        return {name: getattr(self, name) for name in SIGNAL_NAMES}

    def recompute_total(self) -> None:
        self.total = sum(self.signal_dict().values())


@dataclass
class QueryPatternRecord:
    qid: str
    query_text: str
    pattern: str
    ndcg10: float
    recall10: float
    route_tier: str
    n_candidates: int
    n_kappa_keys: int
    z_obs_q: float
    gold_ids: list[str]
    gold_in_candidates: bool
    gold_best_rank: int | None
    top1_id: str
    top1_is_gold: bool
    gold: PatternBreakdown | None = None
    false_top1: PatternBreakdown | None = None
    signal_delta: dict[str, float] = field(default_factory=dict)
    score_gap: float = 0.0
    fix_hint: str = ""


@dataclass
class PatternPlacementReport:
    n_queries: int
    mean_ndcg10: float
    mean_recall10: float
    gold_recall_at_c: float
    pattern_counts: dict[str, int]
    tier_counts: dict[str, int]
    signal_lift: dict[str, float]
    signal_win_rate: dict[str, float]
    tuner_hints: list[tuple[str, float, str]]
    records: list[QueryPatternRecord]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_queries": self.n_queries,
            "mean_ndcg10": self.mean_ndcg10,
            "mean_recall10": self.mean_recall10,
            "gold_recall_at_c": self.gold_recall_at_c,
            "pattern_counts": self.pattern_counts,
            "tier_counts": self.tier_counts,
            "signal_lift": self.signal_lift,
            "signal_win_rate": self.signal_win_rate,
            "tuner_hints": [
                {"signal": s, "lift": lift, "hint": hint}
                for s, lift, hint in self.tuner_hints
            ],
        }


def _bm25_overlap(profile: QueryProfile, doc_tokens: frozenset[str]) -> int:
    return len(profile.word_set & doc_tokens)


def _neighbor_overlap(profile: QueryProfile, doc_tokens: frozenset[str]) -> int:
    return len(frozenset(profile.flat_neighbors.keys()) & doc_tokens)


def compute_pattern_breakdown(
    profile: QueryProfile,
    doc_id: str,
    *,
    cidx,
    hub_sigs: dict,
    comp_idx,
    sub_comp_idx,
    phrase_idx,
    anchor_idx,
    q_anchor_comps: dict[int, float] | None,
    q_phrase_comps: dict[int, float] | None,
    registry,
    attractor_index=None,
    query_kappa_keys=None,
    cell_profile=None,
    rank: int | None = None,
    in_candidates: bool = False,
) -> PatternBreakdown:
    """Full per-signal replay for one doc — mirrors rank loop + pipeline bits."""
    bd = PatternBreakdown(doc_id=doc_id, rank=rank, in_candidates=in_candidates)

    doc_tokens = cidx.doc_tokens.get(doc_id, frozenset())
    sig = hub_sigs.get(doc_id)
    tf = cidx.doc_tf.get(doc_id) if cidx.doc_tf else None
    dl = cidx.doc_len.get(doc_id, 0)
    avg_dl = cidx.avg_dl
    k1, b_param = 1.5, 0.75

    bd.bm25_word_overlap = _bm25_overlap(profile, doc_tokens)
    bd.neighbor_word_overlap = _neighbor_overlap(profile, doc_tokens)

    for w in profile.word_set:
        if w not in doc_tokens:
            continue
        idf = profile.idf.get(w, 1.0)
        if tf and dl > 0:
            tf_val = tf.get(w, 0)
            norm = tf_val * (k1 + 1.0) / (tf_val + k1 * (1.0 - b_param + b_param * dl / avg_dl))
        else:
            norm = 1.0
        bd.s1_bm25 += idf * norm

    if sig:
        n_wings = len(CONSENSUS_WINGS)
        if profile.wing_coords and n_wings > 0:
            for hub_word, hub_entry in sig.hubs.items():
                hub_wc = hub_entry.wing_coords()
                if not hub_wc:
                    q_word = _l01_meet_query_word(profile, hub_entry, hub_word)
                    if q_word:
                        bd.s2_coord += profile.idf.get(q_word, 1.0) * _hs.LAMBDA_COORD
                    continue
                wing_matches = 0
                matched_q_word = ""
                for lid, hub_coord in hub_wc.items():
                    q_wing = profile.wing_coords.get(lid, {})
                    q_word = q_wing.get(hub_coord, "")
                    if q_word and q_word != hub_word:
                        wing_matches += 1
                        matched_q_word = q_word
                if wing_matches > 0:
                    bd.s2_coord += (
                        profile.idf.get(matched_q_word, 1.0)
                        * _hs.LAMBDA_COORD
                        * (wing_matches / n_wings)
                    )
        else:
            bd.s2_coord += _l01_coord_meet_for_sig(profile, sig)

        max_nb = profile.max_neighbor_weight or 1.0
        for hub_word in sig.hub_words():
            nb_weight = profile.flat_neighbors.get(hub_word, 0.0)
            if nb_weight > 0:
                q_word = profile.neighbor_source.get(hub_word, "")
                idf_q = profile.idf.get(q_word, 1.0)
                sat = min(1.0, nb_weight / max_nb)
                bd.s3_neighbors += idf_q * sat * _hs.LAMBDA_NEIGHBOR

        if sub_comp_idx is not None and registry is not None:
            bd.s4_subword = subword_composite_score(
                list(profile.word_set),
                profile.idf,
                registry,
                sub_comp_idx,
                doc_id,
                word_cache=sub_comp_idx.word_composites,
            )

        bd.s5b_prime_factor = prime_factor_meet_score(profile, sig)

        if profile.query_cluster_ids:
            for hub_word in sig.hub_words():
                cluster_idf = profile.query_cluster_ids.get(hub_word, 0.0)
                if cluster_idf > 0:
                    bd.s7_cluster += cluster_idf * _hs.LAMBDA_CLUSTER

    if q_phrase_comps is not None and phrase_idx is not None:
        bd.s5_phrase = phrase_composite_score_fast(q_phrase_comps, doc_id, phrase_idx)

    if q_anchor_comps and anchor_idx is not None:
        bd.s6_anchors = score_with_heavy_anchors(q_anchor_comps, doc_id, anchor_idx)

    if attractor_index is not None and query_kappa_keys:
        bd.kappa_jaccard = attractor_index.score_doc_overlap(query_kappa_keys, doc_id)
        bd.s8a_kappa = signal_8a_kappa_jaccard(
            profile,
            doc_id,
            attractor_index,
            query_kappa_keys,
            lambda_kappa=_hs.LAMBDA_KAPPA,
        )

    bd.recompute_total()
    return bd


def classify_failure_pattern(
    *,
    ndcg10: float,
    gold_ids: set[str],
    gold_in_candidates: bool,
    gold_best_rank: int | None,
    gold_bm25_overlap: int,
    top1_is_gold: bool,
    gold_in_corpus: bool,
) -> str:
    if not gold_in_corpus:
        return "NO_GOLD_IN_CORPUS"
    if ndcg10 >= 1.0 - 1e-9:
        return "PERFECT"
    if ndcg10 > 0:
        return "PARTIAL"
    if not gold_in_candidates:
        return "MISSED_CANDIDATE"
    if gold_bm25_overlap == 0:
        return "ZERO_BM25"
    if gold_best_rank is not None and gold_best_rank <= 10:
        return "RANK_LOW"
    return "SCORE_MISS"


def _fix_hint(
    pattern: str,
    delta: dict[str, float],
    *,
    route_tier: str,
    gold_in_candidates: bool,
) -> str:
    if pattern == "MISSED_CANDIDATE":
        if "trimmed" in route_tier:
            return "Raise MAX_ROUTE_CANDIDATES or κ recall; gold trimmed from pool"
        return "BIT 4/7 routing: expand κ radius, meet supplement, or tier1 fallback"
    if pattern == "ZERO_BM25":
        best = max(
            ((delta.get("s8a_kappa", 0), "λ_κ / BIT 4 κ overlap"),
             (delta.get("s3_neighbors", 0), "λ_neighbor L4-L6"),
             (delta.get("s5b_prime_factor", 0), "λ_pf pool Jaccard"),
             (delta.get("s2_coord", 0), "λ_coord meet")),
            key=lambda x: x[0],
        )
        if best[0] > 0:
            return f"Lattice-only path — increase {best[1]}"
        return "Lattice-only: no signal separates gold from false; check BIT 7 meet"
    if pattern in ("SCORE_MISS", "PARTIAL", "RANK_LOW"):
        ranked = sorted(
            ((delta.get(s, 0.0), s) for s in SIGNAL_NAMES),
            reverse=True,
        )
        top_sig, top_lift = ranked[0][1], ranked[0][0]
        if top_lift > 0.01:
            return f"Gold leads on {top_sig} (+{top_lift:.3f}); false doc wins elsewhere — tune that λ"
        if ranked[0][0] < 0:
            return f"False doc leads on {ranked[0][1]} — cap or gate that signal"
        return "Score tie — check BM25 dominance cap and FULL_SCORE_LIMIT pool"
    return ""


def audit_query_patterns(
    bundle,
    *,
    qids: Sequence[str] | None = None,
    enable_kappa_scoring: bool = True,
    kappa_candidate_cap: int = DEFAULT_KAPPA_CANDIDATE_CAP,
    progress_every: int = 0,
) -> PatternPlacementReport:
    """Score every query; place all patterns on gold vs false top-1."""
    pipe = bundle.pipe
    cidx = bundle.cidx
    hub_sigs = bundle.hub_sigs
    neighbor_map = bundle.neighbor_map
    meet_index = bundle.meet_index
    sub_comp_idx = bundle.sub_comp_idx
    comp_idx = bundle.comp_idx
    phrase_idx = bundle.phrase_idx
    anchor_idx = bundle.anchor_idx
    attractor_index = bundle.attractor_index
    queries = bundle.queries
    qrels = bundle.qrels
    loaded = set(cidx.doc_ids)

    if qids is None:
        qids = bundle.qids

    records: list[QueryPatternRecord] = []
    pattern_counts: Counter[str] = Counter()
    tier_counts: Counter[str] = Counter()
    lift_acc: dict[str, list[float]] = defaultdict(list)
    win_acc: dict[str, list[float]] = defaultdict(list)
    gold_at_c = 0
    gold_total = 0
    ndcgs: list[float] = []
    r10s: list[float] = []

    for qi, qid in enumerate(qids):
        query = queries[qid]
        rel = qrels.get(qid, {})
        gold_all = {d for d, r in rel.items() if r > 0}
        gold = gold_all & loaded

        result = _score_one_query(
            query,
            pipe=pipe,
            cidx=cidx,
            hub_sigs=hub_sigs,
            neighbor_map=neighbor_map,
            meet_index=meet_index,
            sub_comp_idx=sub_comp_idx,
            comp_idx=comp_idx,
            phrase_idx=phrase_idx,
            anchor_idx=anchor_idx,
            attractor_index=attractor_index,
            kappa_candidate_cap=kappa_candidate_cap,
            enable_kappa_scoring=enable_kappa_scoring,
        )

        ranked = result.ranked
        cand_set: set[str] = set()

        profile = build_query_profile(
            query,
            pipe.registry,
            neighbor_map=neighbor_map,
            doc_freq=cidx.doc_freq,
            n_docs=len(cidx.doc_ids),
        )
        if pipe.reader.word_to_cluster:
            query_clusters: dict[str, float] = {}
            for w in profile.word_set:
                cid = pipe.reader.word_to_cluster.get(w)
                if cid:
                    query_clusters[cid] = max(
                        query_clusters.get(cid, 0.0), profile.idf.get(w, 0.0),
                    )
            if query_clusters:
                qc_ids: dict[str, float] = {}
                for w2, cid2 in pipe.reader.word_to_cluster.items():
                    if cid2 in query_clusters and w2 not in profile.word_set:
                        if qc_ids.get(w2, 0.0) < query_clusters[cid2]:
                            qc_ids[w2] = query_clusters[cid2]
                profile.query_cluster_ids = qc_ids

        cell = None
        query_kappa_keys = None
        z_obs_q = 0.0
        if attractor_index is not None:
            cell = build_query_cell_profile(
                pipe.registry,
                query,
                neighbor_map=neighbor_map,
                doc_freq=cidx.doc_freq,
                n_docs=len(cidx.doc_ids),
            )
            query_kappa_keys = cell.kappa_neighbor_q
            z_obs_q = cell.z_obs_q

        if attractor_index is not None:
            from pipeline.bit_04_candidate_router import route_query_candidates

            meet_arg = (
                meet_index.legacy_dict()
                if meet_index is not None and hasattr(meet_index, "legacy_dict")
                else meet_index
            )
            route = route_query_candidates(
                profile.words,
                pipe.registry,
                attractor_index,
                cidx.inv,
                neighbor_map,
                cidx.doc_ids,
                meet_index=meet_index if hasattr(meet_index, "by_factor") else meet_arg,
                doc_freq=cidx.doc_freq,
                n_docs=len(cidx.doc_ids),
            )
            cand_set = set(route.doc_ids)

        q_anchor = None
        if anchor_idx is not None and anchor_idx.n_anchors > 0:
            q_anchor = query_anchor_composites(
                list(profile.word_set), anchor_idx, pipe.registry, idf=profile.idf,
            )
        q_phrase = None
        if phrase_idx is not None:
            q_phrase = _query_phrase_composites(
                profile.words, phrase_idx, pipe.registry, profile.idf,
            )

        rank_of = {did: i + 1 for i, did in enumerate(ranked)}
        top1 = ranked[0] if ranked else ""
        top1_is_gold = top1 in gold

        gold_ranks = [rank_of[g] for g in gold if g in rank_of]
        gold_best_rank = min(gold_ranks) if gold_ranks else None
        gold_in_c = bool(gold & cand_set) if gold else False

        if gold:
            gold_total += len(gold)
            if gold_in_c:
                gold_at_c += sum(1 for g in gold if g in cand_set)

        ndcg = ndcg_at_k(ranked, rel, 10)
        r10 = recall_at_k(ranked, rel, 10)
        ndcgs.append(ndcg)
        r10s.append(r10)

        best_gold_id = None
        best_gold_bd = None
        if gold:
            best_gold_id = min(
                gold,
                key=lambda g: rank_of.get(g, 10_000),
            )
            best_gold_bd = compute_pattern_breakdown(
                profile,
                best_gold_id,
                cidx=cidx,
                hub_sigs=hub_sigs,
                comp_idx=comp_idx,
                sub_comp_idx=sub_comp_idx,
                phrase_idx=phrase_idx,
                anchor_idx=anchor_idx,
                q_anchor_comps=q_anchor,
                q_phrase_comps=q_phrase,
                registry=pipe.registry,
                attractor_index=attractor_index,
                query_kappa_keys=query_kappa_keys,
                cell_profile=cell,
                rank=rank_of.get(best_gold_id),
                in_candidates=best_gold_id in cand_set,
            )

        false_bd = None
        if top1 and not top1_is_gold:
            false_bd = compute_pattern_breakdown(
                profile,
                top1,
                cidx=cidx,
                hub_sigs=hub_sigs,
                comp_idx=comp_idx,
                sub_comp_idx=sub_comp_idx,
                phrase_idx=phrase_idx,
                anchor_idx=anchor_idx,
                q_anchor_comps=q_anchor,
                q_phrase_comps=q_phrase,
                registry=pipe.registry,
                attractor_index=attractor_index,
                query_kappa_keys=query_kappa_keys,
                cell_profile=cell,
                rank=1,
                in_candidates=top1 in cand_set,
            )

        delta: dict[str, float] = {}
        score_gap = 0.0
        if best_gold_bd and false_bd:
            for name in SIGNAL_NAMES:
                d = best_gold_bd.signal_dict()[name] - false_bd.signal_dict()[name]
                delta[name] = d
                if not top1_is_gold:
                    lift_acc[name].append(d)
                    win_acc[name].append(1.0 if d > 0 else 0.0)
            score_gap = false_bd.total - best_gold_bd.total

        gold_bm25 = best_gold_bd.bm25_word_overlap if best_gold_bd else 0
        pattern = classify_failure_pattern(
            ndcg10=ndcg,
            gold_ids=gold,
            gold_in_candidates=gold_in_c,
            gold_best_rank=gold_best_rank,
            gold_bm25_overlap=gold_bm25,
            top1_is_gold=top1_is_gold,
            gold_in_corpus=bool(gold),
        )
        pattern_counts[pattern] += 1
        tier_counts[result.route_tier] += 1

        hint = _fix_hint(
            pattern,
            delta,
            route_tier=result.route_tier,
            gold_in_candidates=gold_in_c,
        )

        records.append(
            QueryPatternRecord(
                qid=qid,
                query_text=query[:120],
                pattern=pattern,
                ndcg10=ndcg,
                recall10=r10,
                route_tier=result.route_tier,
                n_candidates=result.n_candidates,
                n_kappa_keys=result.n_kappa_keys,
                z_obs_q=z_obs_q,
                gold_ids=sorted(gold),
                gold_in_candidates=gold_in_c,
                gold_best_rank=gold_best_rank,
                top1_id=top1,
                top1_is_gold=top1_is_gold,
                gold=best_gold_bd,
                false_top1=false_bd,
                signal_delta=delta,
                score_gap=score_gap,
                fix_hint=hint,
            )
        )

        if progress_every > 0 and (qi + 1) % progress_every == 0:
            print(
                f"  pattern audit {qi + 1}/{len(qids)}  "
                f"avg_ndcg={sum(ndcgs) / len(ndcgs):.4f}",
                flush=True,
            )

    signal_lift = {
        s: (sum(v) / len(v) if v else 0.0)
        for s, v in lift_acc.items()
    }
    signal_win_rate = {
        s: (sum(v) / len(v) if v else 0.0)
        for s, v in win_acc.items()
    }

    tuner_hints: list[tuple[str, float, str]] = []
    for sig in SIGNAL_NAMES:
        lift = signal_lift.get(sig, 0.0)
        win = signal_win_rate.get(sig, 0.0)
        if lift > 0.05 and win > 0.5:
            tuner_hints.append((
                sig,
                lift,
                f"Gold beats false on {sig} in {win * 100:.0f}% of misses — consider raising λ",
            ))
        elif lift < -0.05 and win < 0.5:
            tuner_hints.append((
                sig,
                lift,
                f"False beats gold on {sig} — tighten gate or lower λ",
            ))
    tuner_hints.sort(key=lambda x: -abs(x[1]))

    return PatternPlacementReport(
        n_queries=len(qids),
        mean_ndcg10=sum(ndcgs) / max(len(ndcgs), 1),
        mean_recall10=sum(r10s) / max(len(r10s), 1),
        gold_recall_at_c=gold_at_c / max(gold_total, 1),
        pattern_counts=dict(pattern_counts),
        tier_counts=dict(tier_counts),
        signal_lift=signal_lift,
        signal_win_rate=signal_win_rate,
        tuner_hints=tuner_hints,
        records=records,
    )


def format_pattern_report(report: PatternPlacementReport, *, top_n: int = 15) -> str:
    """Human-readable summary for terminal output."""
    lines: list[str] = []
    w = 72
    lines.append("=" * w)
    lines.append("  PATTERN PLACEMENT AUDIT — all queries, gold vs false, all signals")
    lines.append("=" * w)
    lines.append(
        f"  queries={report.n_queries}  NDCG@10={report.mean_ndcg10:.4f}  "
        f"R@10={report.mean_recall10:.4f}  gold-in-C={report.gold_recall_at_c:.1%}"
    )
    lines.append("")
    lines.append("  Failure patterns:")
    for pat, cnt in sorted(report.pattern_counts.items(), key=lambda x: -x[1]):
        pct = 100.0 * cnt / max(report.n_queries, 1)
        lines.append(f"    {pat:20s}  {cnt:5d}  ({pct:5.1f}%)")
    lines.append("")
    lines.append("  BIT 4 routing tiers:")
    for tier, cnt in sorted(report.tier_counts.items(), key=lambda x: -x[1]):
        lines.append(f"    {tier:28s}  {cnt:5d}")
    lines.append("")
    lines.append("  Signal lift (gold − false top-1, on wrong-top-1 queries):")
    for sig in SIGNAL_NAMES:
        lift = report.signal_lift.get(sig, 0.0)
        win = report.signal_win_rate.get(sig, 0.0)
        bar = "+" if lift > 0 else ""
        lines.append(f"    {sig:18s}  {bar}{lift:7.3f}   gold-wins {win * 100:5.1f}%")
    lines.append("")
    if report.tuner_hints:
        lines.append("  Tuner hints (ranked by |lift|):")
        for sig, lift, hint in report.tuner_hints[:top_n]:
            lines.append(f"    [{sig}] lift={lift:+.3f}  {hint}")
        lines.append("")
    lines.append(f"  Worst gaps (false #1 beats gold, top {top_n}):")
    worst = sorted(
        (r for r in report.records if r.score_gap > 0 and r.gold),
        key=lambda r: -r.score_gap,
    )[:top_n]
    for r in worst:
        lines.append(
            f"    q={r.qid:<6} gap={r.score_gap:6.2f}  {r.pattern:16s}  "
            f"|C|={r.n_candidates:<4}  {r.query_text[:40]!r}"
        )
        if r.fix_hint:
            lines.append(f"             → {r.fix_hint}")
    lines.append("=" * w)
    return "\n".join(lines)
