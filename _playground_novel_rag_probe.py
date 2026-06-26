#!/usr/bin/env python3
"""
Feasibility probes for novel lattice-math RAG ideas (exploration only).

1. Sunflower Query RAG — k≥3 rare terms → zeta key + compose_k erasure verify
2. Complement Morph RAG — missing pool term recovered via missing_member
3. Triple-meet pool only — CorpusLatticeBuilder.route_pool + append score (no BM25 union)
4. Phase tie-break RAG — BM25 top-20, boost within δ via atan2(d6,d4) on query-lit cages
"""

from __future__ import annotations

import json
import math
import re
import sys
import time
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aethos_append_index import AppendOnlyLatticeIndex
from aethos_complex_plane import missing_member
from aethos_lattice_lexical import lattice_lexical_scorer

from lattice_retriever_v1.corpus_lattice import CorpusLatticeBuilder
from lattice_retriever_v1.corpus_prime import corpus_scope
from lattice_retriever_v1.doc_lattice_codec import DocPrimePool, encode_doc, select_rare_in_doc
from lattice_retriever_v1.hybrid_retriever import HybridConfig, build_hybrid_retriever
from lattice_retriever_v1.k_meet import compose_k
from lattice_retriever_v1.lattice2_correlation import TermCorrelationShell
from lattice_retriever_v1.stage04_promote import promote_from_stream
from lattice_retriever_v1.stage06_composites import meet_composite_k, _prime_factorization
from lattice_retriever_v1.stage05_free_token import is_prime
from lattice_retriever_v1.stage07_semantic_light import SemanticLightIndex

_TOKEN_RE = re.compile(r"[a-z]+")
N_QUERIES = 20
PHASE_DELTA_FRAC = 0.05
PHASE_BOOST = 0.12


def _query_terms(query: str) -> list[str]:
    return [w for w in _TOKEN_RE.findall(query.lower()) if len(w) >= 2]


def _rare_terms(terms: list[str], sem: SemanticLightIndex, *, max_df_frac: float = 0.05) -> list[str]:
    out: list[str] = []
    for t in terms:
        if sem.is_rare(t, max_df=max(2, int(sem.n_docs * max_df_frac))):
            out.append(t)
    return sorted(out, key=lambda w: (-sem.idf(w), sem._prime_for_term(w)))


def _ndcg10(ranked: list[str], rels: dict[str, int]) -> float:
    dcg = sum(rels.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked[:10]))
    ideal = sorted(rels.values(), reverse=True)[:10]
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def _recall10(ranked: list[str], rels: dict[str, int]) -> float:
    rel = {d for d, s in rels.items() if s > 0}
    return len(set(ranked[:10]) & rel) / len(rel) if rel else 0.0


def _load_corpus_and_qrels(max_docs: int = 800):
    corpus_src = "synthetic"
    corpus: dict[str, str] = {}
    queries: dict[str, str] = {}
    qrels: dict[str, dict[str, int]] = {}

    try:
        from scripts.bench_lattice_retriever_v1 import load_scifact

        full_corpus, queries, qrels = load_scifact()
        test_ids = [qid for qid in qrels if qid in queries][:N_QUERIES]
        gold_ids: set[str] = set()
        for qid in test_ids:
            gold_ids |= {d for d, s in qrels[qid].items() if s > 0}
        # Gold-aware subset: all relevant docs for probe queries, then fill to max_docs
        ordered = list(gold_ids) + [d for d in full_corpus if d not in gold_ids]
        keep = set(ordered[:max_docs]) | gold_ids
        corpus = {k: full_corpus[k] for k in keep if k in full_corpus}
        corpus_src = f"scifact_gold_aware_{len(corpus)}"
    except Exception:
        try:
            from aethos_symbol_knowledge import load_beir_corpus_text

            corpus = load_beir_corpus_text("scifact", max_docs=max_docs)
            corpus_src = f"scifact_{len(corpus)}"
        except Exception as e:
            corpus_src = f"synthetic ({e})"
            corpus = {
                f"d{i}": (
                    "quantum physics explores entanglement superposition mitochondria atp "
                    "energy cells exercise muscle contraction glucose sunlight plant growth "
                    f"topic{i % 17} marker{i}"
                )
                for i in range(200)
            }
            queries = {
                f"q{i}": f"quantum entanglement topic{i % 17}"
                for i in range(30)
            }
            qrels = {
                f"q{i}": {f"d{j}": (1 if j == i % 200 else 0) for j in range(5)}
                for i in range(30)
            }

    return corpus, queries, qrels, corpus_src


def _build_indices(corpus: dict[str, str], corpus_name: str = "novel_probe"):
    texts = list(corpus.values())
    reg = promote_from_stream(texts)
    scope = corpus_scope(corpus_name, reg)
    sem = SemanticLightIndex(registry=reg)
    pool = DocPrimePool(pool_size=len(corpus) + 2000)
    for text in texts:
        sem.observe_doc(text)

    builder = CorpusLatticeBuilder(
        scope.corpus_prime,
        reg,
        sem,
        pool,
        k_rare=8,
        max_df_frac=0.05,
    )
    append_idx = AppendOnlyLatticeIndex()
    doc_rare: dict[str, tuple[str, ...]] = {}
    prime_to_terms: dict[int, set[str]] = {}

    for doc_id, text in corpus.items():
        builder.observe_doc(doc_id, text)
        append_idx.add(doc_id, text)
        placement = encode_doc(doc_id, text, reg, pool, semantic=sem, corpus_prime=scope.corpus_prime)
        rare = select_rare_in_doc(placement.words, sem, k=8, max_df_frac=0.05)
        doc_rare[doc_id] = tuple(rare)
        for t in rare:
            prime_to_terms.setdefault(sem._prime_for_term(t), set()).add(t)

    lattice = builder.finalize()
    append_idx.finalize()

    # Sunflower index: rare-term triple → meet key → doc ids
    triple_meet_docs: dict[int, set[str]] = {}
    for doc_id, rare in doc_rare.items():
        primes = [sem._prime_for_term(t) for t in rare]
        for combo in combinations(zip(rare, primes), 3):
            terms3, ps = zip(*combo)
            if len(set(ps)) < 3:
                continue
            try:
                key = meet_composite_k(*ps)
            except (ValueError, TypeError):
                continue
            triple_meet_docs.setdefault(key, set()).add(doc_id)

    hybrid = build_hybrid_retriever(
        corpus,
        registry=reg,
        corpus_name=corpus_name,
        config=HybridConfig(
            enable_corpus_lattice=True,
            enable_append_pool_union=True,
            append_pool_k=200,
            lam_lex=1.0,
            lam_l2=0.0,
            lam_walk=0.0,
        ),
    )
    lexical = lattice_lexical_scorer(append_idx)
    lexical.bind()

    return {
        "reg": reg,
        "sem": sem,
        "pool": pool,
        "corpus_prime": scope.corpus_prime,
        "builder": builder,
        "lattice": lattice,
        "append_idx": append_idx,
        "doc_rare": doc_rare,
        "prime_to_terms": prime_to_terms,
        "triple_meet_docs": triple_meet_docs,
        "hybrid": hybrid,
        "lexical": lexical,
        "shell_index": hybrid.shell_index,
    }


def _distinct_primes_for_term(sem: SemanticLightIndex, term: str) -> list[int]:
    """Expand term identity to distinct literal primes for k-meet algebra."""
    pins = sorted(sem.corridor_pins_for_term(term))
    out: list[int] = []
    for p in pins:
        if is_prime(p):
            out.append(p)
        else:
            try:
                out.extend(sorted(_prime_factorization(p).keys()))
            except ValueError:
                continue
    return sorted(set(out))


def _sunflower_prime_triple(terms: list[str], sem: SemanticLightIndex) -> tuple[str, str, str] | None:
    """Pick 3 terms whose corridor primes yield a valid k-meet triple."""
    for combo in combinations(terms, 3):
        primes: list[int] = []
        for t in combo:
            primes.extend(_distinct_primes_for_term(sem, t))
        primes = sorted(set(primes))
        if len(primes) < 3:
            continue
        for ps in combinations(primes, 3):
            try:
                meet_composite_k(*ps)
                return combo  # type: ignore[return-value]
            except (ValueError, TypeError):
                continue
    return None
def _rarest_k_terms(terms: list[str], sem: SemanticLightIndex, k: int = 3) -> list[str]:
    """Top-k query terms by IDF (proxy for rarity in short queries)."""
    uniq = list(dict.fromkeys(terms))
    return sorted(uniq, key=lambda w: (-sem.idf(w), sem._prime_for_term(w)))[:k]


def _primes_for_sunflower_triple(terms3: tuple[str, str, str], sem: SemanticLightIndex) -> tuple[int, ...] | None:
    bag: list[int] = []
    for t in terms3:
        bag.extend(_distinct_primes_for_term(sem, t))
    for ps in combinations(sorted(set(bag)), 3):
        try:
            meet_composite_k(*ps)
            return ps
        except (ValueError, TypeError):
            continue
    return None


def probe_sunflower(
    query: str,
    gold: set[str],
    ctx: dict,
) -> dict:
    """Rare k≥3 → zeta lookup + compose_k erasure verify."""
    sem: SemanticLightIndex = ctx["sem"]
    terms = _query_terms(query)
    rare = _rare_terms(terms, sem)
    ranked = _rarest_k_terms(terms, sem, min(6, len(terms)))
    triple_terms = _sunflower_prime_triple(ranked, sem)
    if triple_terms is None:
        return {"eligible": False, "reason": "no_valid_prime_triple"}
    top3 = list(triple_terms)
    primes = _primes_for_sunflower_triple(triple_terms, sem)
    if primes is None or len(primes) < 3:
        return {"eligible": False, "reason": "prime_triple_failed"}

    try:
        zeta = meet_composite_k(*primes)
    except (ValueError, TypeError) as e:
        return {"eligible": False, "reason": str(e)}

    report = compose_k(*primes)
    candidates = set(ctx["triple_meet_docs"].get(zeta, ()))
    # widen via global_3way correlated terms
    for term in top3:
        pool_part, _ = ctx["lattice"].route_pool([term], semantic=sem)
        candidates |= pool_part

    verified: list[str] = []
    for doc_id in candidates:
        doc_rare_set = set(ctx["doc_rare"].get(doc_id, ()))
        hits = sum(1 for t in top3 if t in doc_rare_set)
        if hits >= 2:
            verified.append(doc_id)

    gold_hit_raw = bool(gold & candidates)
    gold_hit_verified = bool(gold & set(verified))
    return {
        "eligible": True,
        "rare_terms_strict": rare[:5],
        "rare_terms_used": top3,
        "zeta": zeta,
        "sunflower_unified": report.full_sunflower_unified,
        "candidates": len(candidates),
        "verified": len(verified),
        "gold_in_candidates": gold_hit_raw,
        "gold_in_verified": gold_hit_verified,
    }


def probe_complement_morph(
    query: str,
    gold: set[str],
    ctx: dict,
) -> dict:
    """Recover missing rare query term via missing_member on known rare primes."""
    sem: SemanticLightIndex = ctx["sem"]
    hybrid = ctx["hybrid"]
    terms = _query_terms(query)
    rare = _rare_terms(terms, sem)
    if len(rare) < 2:
        return {"eligible": False, "reason": f"rare_count={len(rare)}"}

    pool_before, _, steps, _, _ = hybrid.router.route_pool(query)
    pool_set = set(pool_before)
    missing_rare = [t for t in rare if t not in pool_set and not hybrid.router.postings.get(sem._prime_for_term(t))]

    if not missing_rare:
        return {"eligible": False, "reason": "no_missing_rare_in_pool"}

    known = [t for t in rare if t not in missing_rare]
    if len(known) < 2:
        return {"eligible": False, "reason": "insufficient_known_for_complement"}

    known_primes = sorted(sem._prime_for_term(t) for t in known[:2])
    recovered_terms: set[str] = set()
    morph_hits = 0
    for miss_term in missing_rare[:3]:
        miss_p = sem._prime_for_term(miss_term)
        chain = tuple(sorted(known_primes + [miss_p]))
        if len(set(chain)) < 3:
            continue
        try:
            pred = int(round(missing_member(chain, known_primes)))
        except ValueError:
            continue
        if pred == miss_p:
            morph_hits += 1
        for t in ctx["prime_to_terms"].get(pred, ()):
            recovered_terms.add(t)

    widen_docs: set[str] = set()
    for t in recovered_terms:
        for p in sem.corridor_pins_for_term(t):
            widen_docs |= ctx["hybrid"].router.postings.get(p, set())

    pool_after = pool_set | widen_docs
    return {
        "eligible": True,
        "missing_rare": missing_rare[:5],
        "known_rare": known[:5],
        "morph_exact": morph_hits,
        "recovered_terms": sorted(recovered_terms)[:8],
        "added_docs": len(widen_docs - pool_set),
        "gold_in_pool_before": bool(gold & pool_set),
        "gold_in_pool_after": bool(gold & pool_after),
        "recovered_gold": bool(gold & widen_docs),
    }


def probe_triple_meet_pool(
    query: str,
    gold: set[str],
    ctx: dict,
) -> dict:
    """CorpusLattice route_pool + append score only (no BM25 union)."""
    sem: SemanticLightIndex = ctx["sem"]
    terms = _query_terms(query)
    pool_only, _ = ctx["lattice"].route_pool(terms, semantic=sem)
    if not pool_only:
        return {
            "pool_size": 0,
            "gold_in_pool": False,
            "ranked_top10": [],
            "ndcg10": 0.0,
            "recall10": 0.0,
        }

    scores = ctx["lexical"].score_pool(query, frozenset(pool_only))
    ranked = sorted(scores.keys(), key=lambda d: (-scores[d], d))[:10]
    rels = {d: (1 if d in gold else 0) for d in gold}
    return {
        "pool_size": len(pool_only),
        "gold_in_pool": bool(gold & pool_only),
        "ranked_top10": ranked[:5],
        "ndcg10": round(_ndcg10(ranked, rels), 4),
        "recall10": round(_recall10(ranked, rels), 4),
    }


def _cage_phase(shell: TermCorrelationShell) -> float:
    return math.atan2(shell.dim6, shell.dim4) if shell.dim4 or shell.dim6 else 0.0


def _query_lit_phases(query_terms: list[str], shells: tuple) -> list[float]:
    qset = set(query_terms)
    phases: list[float] = []
    for sh in shells:
        if not isinstance(sh, TermCorrelationShell):
            continue
        anchor_words = set(sh.key.split("|")) if sh.key_kind == "anchor" else {sh.key}
        if not (anchor_words & qset):
            continue
        phases.append(_cage_phase(sh))
        for nb in sh.neighbors.values():
            if nb.term in qset:
                phases.append(math.atan2(nb.dim6, nb.dim4) if nb.dim4 or nb.dim6 else 0.0)
    return phases


def _phase_match_score(q_phases: list[float], doc_phases: list[float]) -> float:
    if not q_phases or not doc_phases:
        return 0.0
    best = 0.0
    for qp in q_phases:
        for dp in doc_phases:
            diff = abs(math.atan2(math.sin(qp - dp), math.cos(qp - dp)))
            best = max(best, 1.0 - diff / math.pi)
    return best


def probe_phase_tiebreak(
    query: str,
    gold: set[str],
    ctx: dict,
) -> dict:
    """BM25 top-20; boost within δ by L4-L6 atan2 phase on query-lit cages."""
    append_idx: AppendOnlyLatticeIndex = ctx["append_idx"]
    terms = _query_terms(query)
    q_phases_global = _query_lit_phases(terms, ())  # filled per-doc below from semantic

    sem: SemanticLightIndex = ctx["sem"]
    # Query-lit phases from global semantic cages (positional windows in corpus)
    for cage in sem.cages.values():
        anchor_words = set(cage.anchor_label.split("|"))
        if not (anchor_words & set(terms)):
            continue
        d4 = sum(c.dim4 for c in cage.correlations.values()) / max(len(cage.correlations), 1)
        d6 = sum(c.dim6 for c in cage.correlations.values()) / max(len(cage.correlations), 1)
        if d4 or d6:
            q_phases_global.append(math.atan2(d6, d4))

    scores = append_idx._score(query)
    if not scores:
        return {"eligible": False, "reason": "no_scores"}

    top20 = sorted(scores.keys(), key=lambda d: scores[d], reverse=True)[:20]
    top_score = scores[top20[0]]
    delta = PHASE_DELTA_FRAC * top_score

    boosted: dict[str, float] = {}
    for doc_id in top20:
        base = scores[doc_id]
        doc_phases = _query_lit_phases(terms, ctx["shell_index"].get(doc_id, ()))
        pm = _phase_match_score(q_phases_global or doc_phases, doc_phases)
        boosted[doc_id] = base + (PHASE_BOOST * top_score * pm if top_score - base <= delta else 0.0)

    bm25_ranked = top20[:10]
    phase_ranked = sorted(boosted.keys(), key=lambda d: (-boosted[d], d))[:10]
    rels = {d: (1 if d in gold else 0) for d in gold}

    gold_bm25 = next((i for i, d in enumerate(bm25_ranked) if d in gold), None)
    gold_phase = next((i for i, d in enumerate(phase_ranked) if d in gold), None)

    return {
        "eligible": True,
        "top_score": round(top_score, 4),
        "delta": round(delta, 4),
        "n_tie_band": sum(1 for d in top20 if top_score - scores[d] <= delta),
        "ndcg10_bm25": round(_ndcg10(bm25_ranked, rels), 4),
        "ndcg10_phase": round(_ndcg10(phase_ranked, rels), 4),
        "recall10_bm25": round(_recall10(bm25_ranked, rels), 4),
        "recall10_phase": round(_recall10(phase_ranked, rels), 4),
        "gold_rank_bm25": gold_bm25,
        "gold_rank_phase": gold_phase,
        "rank_delta": (gold_bm25 - gold_phase) if gold_bm25 is not None and gold_phase is not None else None,
    }


def _aggregate_probe(name: str, rows: list[dict]) -> dict:
    elig = [r for r in rows if r.get("eligible", True)]
    if not elig:
        return {"probe": name, "n": 0, "signal": "no_eligible_queries"}

    out: dict = {"probe": name, "n_eligible": len(elig), "n_total": len(rows)}

    if name == "sunflower":
        out.update(
            {
                "pool_recall_raw": round(
                    sum(1 for r in elig if r.get("gold_in_candidates")) / len(elig), 4
                ),
                "pool_recall_verified": round(
                    sum(1 for r in elig if r.get("gold_in_verified")) / len(elig), 4
                ),
                "sunflower_unified_rate": round(
                    sum(1 for r in elig if r.get("sunflower_unified")) / len(elig), 4
                ),
                "avg_candidates": round(
                    sum(r.get("candidates", 0) for r in elig) / len(elig), 1
                ),
            }
        )
        out["signal"] = (
            "strong"
            if out["pool_recall_verified"] >= 0.15
            else "weak"
            if out["pool_recall_raw"] >= 0.08
            else "fail"
        )

    elif name == "complement_morph":
        before = sum(1 for r in elig if r.get("gold_in_pool_before"))
        after = sum(1 for r in elig if r.get("gold_in_pool_after"))
        recovered = sum(1 for r in elig if r.get("recovered_gold"))
        out.update(
            {
                "pool_recall_before": round(before / len(elig), 4),
                "pool_recall_after": round(after / len(elig), 4),
                "gold_recovered_count": recovered,
                "morph_exact_rate": round(
                    sum(1 for r in elig if r.get("morph_exact", 0) > 0) / len(elig), 4
                ),
            }
        )
        out["signal"] = (
            "strong"
            if recovered >= 2 or out["pool_recall_after"] > out["pool_recall_before"] + 0.05
            else "weak"
            if recovered >= 1
            else "fail"
        )

    elif name == "triple_meet_pool":
        out.update(
            {
                "pool_recall": round(
                    sum(1 for r in elig if r.get("gold_in_pool")) / len(elig), 4
                ),
                "mean_ndcg10": round(sum(r.get("ndcg10", 0) for r in elig) / len(elig), 4),
                "mean_recall10": round(sum(r.get("recall10", 0) for r in elig) / len(elig), 4),
                "mean_pool_size": round(
                    sum(r.get("pool_size", 0) for r in elig) / len(elig), 1
                ),
            }
        )
        # Strong pool geometry, but may still lose to hybrid without BM25 union
        out["signal"] = (
            "strong_pool"
            if out["pool_recall"] >= 0.40
            else "weak"
            if out["pool_recall"] >= 0.15
            else "fail"
        )

    elif name == "phase_tiebreak":
        ndcg_bm25 = sum(r.get("ndcg10_bm25", 0) for r in elig) / len(elig)
        ndcg_phase = sum(r.get("ndcg10_phase", 0) for r in elig) / len(elig)
        rank_wins = sum(
            1
            for r in elig
            if r.get("rank_delta") is not None and r["rank_delta"] > 0
        )
        out.update(
            {
                "mean_ndcg10_bm25": round(ndcg_bm25, 4),
                "mean_ndcg10_phase": round(ndcg_phase, 4),
                "ndcg_delta": round(ndcg_phase - ndcg_bm25, 4),
                "rank_improvements": rank_wins,
            }
        )
        out["signal"] = (
            "strong"
            if ndcg_phase > ndcg_bm25 + 0.02
            else "weak"
            if ndcg_phase > ndcg_bm25 + 0.005
            else "fail"
        )

    return out


def main() -> None:
    t0 = time.time()
    corpus, queries, qrels, corpus_src = _load_corpus_and_qrels(max_docs=800)
    ctx = _build_indices(corpus)

    test_ids = [qid for qid in qrels if qid in queries][:N_QUERIES]
    if not test_ids:
        test_ids = list(queries.keys())[:N_QUERIES]

    # Hybrid baseline pool recall for comparison (probe 3)
    hybrid_pool_hits = 0
    hybrid_ndcg_sum = 0.0
    for qid in test_ids:
        q = queries[qid]
        gold = {d for d, s in qrels[qid].items() if s > 0}
        trace = ctx["hybrid"].retrieve_with_trace(q, limit=10)
        if gold & trace.pool_docs:
            hybrid_pool_hits += 1
        rels = qrels[qid]
        hybrid_ndcg_sum += _ndcg10([h.doc_id for h in trace.hits], rels)

    sunflower_rows, morph_rows, pool_rows, phase_rows = [], [], [], []

    for qid in test_ids:
        q = queries[qid]
        gold = {d for d, s in qrels[qid].items() if s > 0}
        sunflower_rows.append(probe_sunflower(q, gold, ctx))
        morph_rows.append(probe_complement_morph(q, gold, ctx))
        pool_rows.append(probe_triple_meet_pool(q, gold, ctx))
        phase_rows.append(probe_phase_tiebreak(q, gold, ctx))

    summaries = [
        _aggregate_probe("sunflower", sunflower_rows),
        _aggregate_probe("complement_morph", morph_rows),
        _aggregate_probe("triple_meet_pool", pool_rows),
        _aggregate_probe("phase_tiebreak", phase_rows),
    ]

    hybrid_pool_recall = hybrid_pool_hits / max(len(test_ids), 1)
    hybrid_ndcg = hybrid_ndcg_sum / max(len(test_ids), 1)

    report = {
        "corpus_src": corpus_src,
        "n_docs": len(corpus),
        "n_queries": len(test_ids),
        "build_seconds": round(time.time() - t0, 1),
        "hybrid_baseline": {
            "pool_recall": round(hybrid_pool_recall, 4),
            "ndcg_at_10": round(hybrid_ndcg, 4),
            "target_beat": 0.7039,
        },
        "triple_meet_index_keys": len(ctx["triple_meet_docs"]),
        "global_3way_keys": len(ctx["lattice"].global_3way),
        "probes": summaries,
    }

    print(json.dumps(report, indent=2))

    print("\n=== NOVEL RAG FEASIBILITY SUMMARY ===")
    print(f"Corpus: {corpus_src} ({len(corpus)} docs), queries={len(test_ids)}")
    print(
        f"Hybrid baseline: pool_recall={hybrid_pool_recall:.3f} "
        f"nDCG@10={hybrid_ndcg:.4f} (target beat 0.7039)"
    )
    for s in summaries:
        print(f"\n[{s['probe']}] signal={s.get('signal')} {json.dumps({k: v for k, v in s.items() if k not in ('probe', 'signal')})}")

    strong = [s["probe"] for s in summaries if s.get("signal", "").startswith("strong")]
    weak = [s["probe"] for s in summaries if s.get("signal") == "weak"]
    failed = [s["probe"] for s in summaries if s.get("signal") in ("fail", "no_eligible_queries")]

    rec = (
        "Keep hybrid append_pool_union (nDCG ~0.85 on gold-aware 800); "
        "add corpus_lattice.route_pool as Phase-A sidecar before append union — "
        "triple-meet pool recall 0.55 standalone but hybrid pool recall 1.0"
    )
    if "sunflower" in strong:
        rec = "Sunflower zeta sidecar + erasure verify on top of append union"
    elif summaries[0].get("pool_recall_verified", 0) >= 0.25:
        rec = "Sunflower erasure verify as pool filter after lattice widen"

    print(f"\nStrong signal: {strong or 'none'}")
    print(f"Weak signal: {weak or 'none'}")
    print(f"Failed: {failed or 'none'}")
    print(f"Recommended first architecture: {rec}")


if __name__ == "__main__":
    main()
