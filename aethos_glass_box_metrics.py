"""
Cross-corpus glass-box metric memory — learn audit signals from any stacked corpus.

Metrics learned (append-only, merge across corpora):
  - hub diluters: high-df query words that appear more in false tops than gold
  - query bucket priors: compound_pair / rarest_only / kappa_only / missed
  - bridge qt terms that correlate with gold hits

Applied at query time on ANY corpus in the same MultiCorpusBrain (shared vocab):
  - block learned hubs from kappa fan + bridge expansion
  - boost kappa overlap fusion when query matches high-yield bucket profile
"""

from __future__ import annotations

import itertools
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from aethos_append_index import words
from aethos_bridges import bridge_expansion
from aethos_promotion import _chunk_subwords
from pipeline.bit_04_candidate_router import query_words_for_routing

if TYPE_CHECKING:
    from aethos_multi_corpus import MultiCorpusBrain


@dataclass
class GlassBoxMemory:
    """Learned cross-corpus retrieval metrics (white-box, no reindex)."""

    hub_diluters: Counter = field(default_factory=Counter)
    bucket_counts: Counter = field(default_factory=Counter)
    bridge_gold_qt: Counter = field(default_factory=Counter)
    polluter_docs: Counter = field(default_factory=Counter)
    demotion_rules: dict = field(default_factory=dict)
    corpus_summaries: dict[str, dict] = field(default_factory=dict)
    enabled: bool = True
    hub_min_queries: int = 3
    compound_kappa_boost: float = 1.25

    def ingest_summary(self, corpus_name: str, summary: dict, rows: list[dict] | None = None) -> None:
        """Merge audit summary (+ optional per-query rows) from one corpus."""
        self.corpus_summaries[corpus_name] = summary
        for w, c in summary.get("top_hub_diluters", []):
            self.hub_diluters[w] += c
        for bucket, c in summary.get("bucket_counts", {}).items():
            self.bucket_counts[bucket] += c
        if rows:
            for row in rows:
                for g in row.get("gold", []):
                    for bp in g.get("bridge_paths", []):
                        self.bridge_gold_qt[bp["qt"]] += 1

    def is_learned_hub(self, word: str) -> bool:
        if not self.enabled:
            return False
        return self.hub_diluters.get(word, 0) >= self.hub_min_queries

    def filter_routing_words(self, routed: list[str], idf: Callable[[str], float] | None = None) -> list[str]:
        """Drop cross-corpus learned hub diluters from kappa routing."""
        if not self.enabled:
            return routed
        return [w for w in routed if not self.is_learned_hub(w)]

    def kappa_lam_scale(self, query: str, idf: Callable[[str], float]) -> float:
        """Boost kappa fusion when query profile matches learned compound-heavy corpora."""
        if not self.enabled or not self.bucket_counts:
            return 1.0
        rare = [w for w in words(query) if idf(w) >= 3.0]
        if len(rare) < 2:
            return 1.0
        total = sum(self.bucket_counts.values()) or 1
        compound_frac = self.bucket_counts.get("compound_pair", 0) / total
        if compound_frac >= 0.45:
            return self.compound_kappa_boost
        return 1.0

    def ingest_polluters(self, polluters: list[dict], *, min_score: int = 5) -> None:
        for p in polluters:
            if p.get("pollution_score", 0) >= min_score:
                self.polluter_docs[p["doc_id"]] += p["pollution_score"]

    def ingest_demotion_rules(self, rules: list[dict]) -> None:
        self.demotion_rules = {r["feature"]: r for r in rules if r.get("rule")}

    def rerank_adjustment(
        self,
        query: str,
        doc_gid: str,
        branch,
        *,
        idf,
        profile: dict | None = None,
    ) -> float:
        """Score delta from learned noise-separation rules (glass-box rerank)."""
        if not self.enabled:
            return 0.0
        local = doc_gid.split("/", 1)[1] if "/" in doc_gid else doc_gid
        adj = 0.0
        if self.polluter_docs.get(local, 0) >= 8:
            adj -= 0.15

        if profile is None:
            return adj

        rules = self.demotion_rules
        if not rules:
            return adj

        # hub-only noise: many hub hits, few rare hits
        if profile.get("n_hub_query_hits", 0) > profile.get("n_rare_query_hits", 0):
            if "n_hub_query_hits" in rules:
                adj -= 0.08

        if profile.get("rare_pair_in_doc", 0) >= 1 and "rare_pair_in_doc" in rules:
            adj += 0.06

        if profile.get("has_rarest_1") and "has_rarest_1" in rules:
            adj += 0.05

        if profile.get("n_bridge_paths", 0) >= 2:
            adj += 0.04

        # long unfocused docs (high len, low density) — noise signature
        if profile.get("doc_len", 0) > 120 and profile.get("query_density", 1) < 0.25:
            adj -= 0.05

        return adj

    def memory_bytes(self) -> int:
        return (
            len(self.hub_diluters) * 16
            + len(self.bucket_counts) * 16
            + len(self.bridge_gold_qt) * 16
        )

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "hub_min_queries": self.hub_min_queries,
            "compound_kappa_boost": self.compound_kappa_boost,
            "hub_diluters": dict(self.hub_diluters),
            "bucket_counts": dict(self.bucket_counts),
            "bridge_gold_qt": dict(self.bridge_gold_qt),
            "polluter_docs": dict(self.polluter_docs),
            "demotion_rules": self.demotion_rules,
            "corpus_summaries": self.corpus_summaries,
        }

    @classmethod
    def from_dict(cls, data: dict) -> GlassBoxMemory:
        mem = cls(
            enabled=data.get("enabled", True),
            hub_min_queries=data.get("hub_min_queries", 3),
            compound_kappa_boost=data.get("compound_kappa_boost", 1.25),
        )
        mem.hub_diluters = Counter(data.get("hub_diluters", {}))
        mem.bucket_counts = Counter(data.get("bucket_counts", {}))
        mem.bridge_gold_qt = Counter(data.get("bridge_gold_qt", {}))
        mem.polluter_docs = Counter(data.get("polluter_docs", {}))
        mem.demotion_rules = dict(data.get("demotion_rules", {}))
        mem.corpus_summaries = dict(data.get("corpus_summaries", {}))
        return mem

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> GlassBoxMemory:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# ---- audit helpers (shared with scripts/audit_glass_box_gold_bridge.py) ----


def _doc_words(text: str) -> set[str]:
    return set(words(text))


def build_scale_pool(brain, branch, query: str, *, rare_df_cap: int = 256):
    from aethos_multi_corpus import IdfCache

    idx = branch.idx
    qws = query_words_for_routing(words(query))
    pool: set[str] = set()
    reasons: dict[str, set[str]] = defaultdict(set)
    keys = frozenset()

    if branch.kappa_index is not None and qws:
        kdocs, keys = brain._attractor_route(branch, qws)
        for d in kdocs[:600]:
            pool.add(d)
            reasons[d].add("kappa")

    idf = IdfCache(branch.idx, branch.n_docs)
    for w in set(words(query)):
        p = idx.token_prime.get(("w", w))
        if p is None:
            continue
        dfp = idx.df.get(p, 0)
        if 0 < dfp <= rare_df_cap:
            pl = idx.postings.get(p)
            if pl:
                for d in pl:
                    if d in idx.alive:
                        pool.add(d)
                        reasons[d].add(f"rare:{w}")

    if branch.pair_bridges is not None:
        for d in bridge_expansion(
            idx, branch.pair_bridges, query,
            idf=idf,
            hub_idf_gate=brain.HUB_IDF_GATE if brain.HUB_IDF_GATE < 50 else 0.0,
        ):
            if d in idx.alive:
                pool.add(d)
                reasons[d].add("bridge")

    return pool, keys, len(keys), {d: sorted(v) for d, v in reasons.items()}


def _rarest_terms(qterms: list[str], idf) -> list[str]:
    uniq = list(dict.fromkeys(qterms))
    return sorted(uniq, key=lambda w: (idf(w), w), reverse=True)


def _prefix_hits(doc_toks: set[str], rarest: list[str], n: int) -> tuple[bool, list[str]]:
    want = rarest[:n]
    hit = [w for w in want if w in doc_toks]
    return len(hit) >= n, hit


def _subword_hits(rare_terms: list[str], doc_toks: set[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for w in rare_terms[:5]:
        pieces = [p for p in _chunk_subwords(w) if len(p) >= 3]
        hits = [p for p in pieces if p in doc_toks and p != w]
        if hits:
            out[w] = hits
    return out


def _pair_triple_hits(rare_sorted: list[str], doc_toks: set[str]) -> dict[str, object]:
    top = rare_sorted[:6]
    pairs_ok = []
    triples_ok = []
    for a, b in itertools.combinations(top, 2):
        if a in doc_toks and b in doc_toks:
            pairs_ok.append((a, b))
    for a, b, c in itertools.combinations(top, 3):
        if a in doc_toks and b in doc_toks and c in doc_toks:
            triples_ok.append((a, b, c))
    return {
        "n_rare_pairs_in_doc": len(pairs_ok),
        "n_rare_triples_in_doc": len(triples_ok),
        "sample_pairs": pairs_ok[:4],
        "sample_triples": triples_ok[:2],
    }


def _bridge_paths(br, query: str, doc_toks: set[str]) -> list[dict]:
    if br is None:
        return []
    paths = []
    for qt in set(words(query)):
        for dt, wt in br.bridge.get(qt, ()):
            if dt in doc_toks:
                paths.append({"qt": qt, "dt": dt, "weight": round(wt, 4)})
    paths.sort(key=lambda x: -x["weight"])
    return paths[:12]


def _teach_paths(teach, query: str, doc_toks: set[str]) -> list[dict]:
    if teach is None:
        return []
    paths = []
    for qt in set(words(query)):
        partners = teach.edges.get(qt)
        if not partners:
            continue
        for dt, c in partners.most_common(12):
            if dt in doc_toks:
                paths.append({"qt": qt, "dt": dt, "count": c})
    paths.sort(key=lambda x: -x["count"])
    return paths[:12]


def _kappa_overlap(branch, keys: frozenset, gid: str) -> int:
    if branch.kappa_index is None or not keys:
        return 0
    return branch.kappa_index.score_doc_overlap(keys, gid)


def _compound_pool_sim(brain, branch, query: str, rarest: list[str], gold_ids: set[str]) -> dict:
    if branch.kappa_index is None or len(rarest) < 2:
        return {"n_intersect_2": 0, "gold_in_intersect_2": 0, "n_intersect_3": 0, "gold_in_intersect_3": 0}
    from pipeline.bit_01_word_cell import word_to_spacetime_cell
    from pipeline.bit_02_attractor_key import kappa_from_cell

    def docs_for_word(w: str) -> set[str]:
        cell = word_to_spacetime_cell(brain._registry, w)
        if cell is None:
            return set()
        k = kappa_from_cell(cell)
        return set(branch.kappa_index.by_key.get(k, ()))

    d1 = docs_for_word(rarest[0])
    d2 = docs_for_word(rarest[1]) if len(rarest) > 1 else set()
    inter2 = d1 & d2
    inter3 = inter2
    if len(rarest) > 2:
        inter3 = inter2 & docs_for_word(rarest[2])
    return {
        "rarest_for_compound": rarest[:3],
        "n_bucket_1": len(d1),
        "n_intersect_2": len(inter2),
        "gold_in_intersect_2": len(inter2 & gold_ids),
        "n_intersect_3": len(inter3),
        "gold_in_intersect_3": len(inter3 & gold_ids),
    }


def audit_one(
    brain,
    branch,
    qid: str,
    query: str,
    gold_local: dict[str, int],
    *,
    idf,
    rare_df_cap: int = 256,
) -> dict[str, object]:
    from scripts.bench_supervised_bridges import ndcg10

    gold_gids = {
        branch.global_id(d)
        for d, s in gold_local.items()
        if s > 0 and d in branch.texts
    }
    qterms = words(query)
    rarest = _rarest_terms(qterms, idf)
    rare_gate = 3.0
    rare_terms = [w for w in rarest if idf(w) >= rare_gate]

    pool, keys, n_keys, pool_reasons = build_scale_pool(brain, branch, query, rare_df_cap=rare_df_cap)
    res = brain.search(query, corpus=branch.name, k=10)
    ranked = res.ranked
    top10 = set(ranked[:10])
    false_top = [d for d in ranked[:10] if d not in gold_gids]

    search_ndcg = ndcg10(res.local_ids, {d: s for d, s in gold_local.items() if s > 0})

    gold_profiles = []
    for local_id, score in gold_local.items():
        if score <= 0 or local_id not in branch.texts:
            continue
        gid = branch.global_id(local_id)
        text = branch.texts[local_id]
        toks = _doc_words(text)
        h1, w1 = _prefix_hits(toks, rarest, 1)
        h2, w2 = _prefix_hits(toks, rarest, 2)
        h3, w3 = _prefix_hits(toks, rarest, 3)
        bp = _bridge_paths(branch.pair_bridges, query, toks)
        tp = _teach_paths(branch.teach, query, toks)
        compound = _pair_triple_hits(rare_terms or rarest[:6], toks)
        rank = ranked.index(gid) + 1 if gid in ranked else None
        gold_profiles.append({
            "local_id": local_id,
            "has_rarest_1": h1, "has_rarest_2": h2, "has_rarest_3": h3,
            "hit_words_1": w1, "hit_words_2": w2, "hit_words_3": w3,
            "in_pool": gid in pool,
            "pool_reasons": pool_reasons.get(gid, []),
            "in_top10": gid in top10,
            "rank": rank,
            "kappa_overlap": _kappa_overlap(branch, keys, gid),
            "bridge_paths": bp, "teach_paths": tp,
            "n_bridge_paths": len(bp), "n_teach_paths": len(tp),
            "subword_hits": _subword_hits(rare_terms or rarest[:5], toks),
            **compound,
        })

    false_profiles = []
    for gid in false_top[:5]:
        local = gid.split("/", 1)[1]
        toks = _doc_words(branch.texts.get(local, ""))
        h1, w1 = _prefix_hits(toks, rarest, 1)
        h2, _ = _prefix_hits(toks, rarest, 2)
        false_profiles.append({
            "local_id": local,
            "rank": ranked.index(gid) + 1,
            "has_rarest_1": h1, "has_rarest_2": h2, "hit_rarest": w1,
            "bridge_paths": _bridge_paths(branch.pair_bridges, query, toks),
            "pool_reasons": pool_reasons.get(gid, []),
        })

    hub_diluters = []
    for w in sorted(qterms, key=idf)[:8]:
        if idf(w) >= 2.5:
            continue
        in_false = sum(
            1 for fp in false_profiles
            if w in _doc_words(branch.texts.get(fp["local_id"], ""))
        )
        in_gold = sum(
            1 for gp in gold_profiles
            if w in _doc_words(branch.texts.get(gp["local_id"], ""))
        )
        hub_diluters.append({
            "word": w, "idf": round(idf(w), 2),
            "in_false_top5": in_false, "in_gold": in_gold,
        })

    any_gold_in_pool = any(g["in_pool"] for g in gold_profiles)
    any_gold_top10 = any(g["in_top10"] for g in gold_profiles)
    if not any_gold_top10:
        bucket = "missed_top10"
    elif not any_gold_in_pool:
        bucket = "rank_miss_not_in_pool"
    elif any(g["has_rarest_1"] and not g["has_rarest_2"] for g in gold_profiles if g["in_pool"]):
        bucket = "rarest_only"
    elif any(g["n_rare_pairs_in_doc"] >= 1 for g in gold_profiles if g["in_pool"]):
        bucket = "compound_pair"
    else:
        bucket = "kappa_only"

    return {
        "query_id": qid, "query": query[:120],
        "rarest_query_words": rarest[:8], "rare_terms_idf3": rare_terms[:6],
        "pool_size": len(pool), "n_kappa_keys": n_keys,
        "gold_in_pool": any_gold_in_pool, "gold_in_top10": any_gold_top10,
        "bucket": bucket, "ndcg10": round(search_ndcg, 4),
        "gold": gold_profiles, "false_top": false_profiles,
        "hub_diluters": hub_diluters,
        "compound_sim": _compound_pool_sim(brain, branch, query, rarest, gold_gids),
    }


def summarize_audit_rows(rows: list[dict]) -> dict[str, object]:
    n_q = len(rows)
    n_gold_inst = sum(len(r["gold"]) for r in rows)
    agg = Counter()
    bucket = Counter()
    pool_hit_q = top10_hit_q = 0
    bridge_gold = bridge_false = 0
    teach_gold = compound_pair_gold = subword_gold = 0
    hub_counter: Counter = Counter()
    compound2_gold = compound3_gold = compound2_q = compound3_q = 0

    for row in rows:
        bucket[row["bucket"]] += 1
        if row["gold_in_pool"]:
            pool_hit_q += 1
        if row["gold_in_top10"]:
            top10_hit_q += 1
        cs = row["compound_sim"]
        if cs.get("n_intersect_2", 0) > 0:
            compound2_q += 1
            compound2_gold += cs.get("gold_in_intersect_2", 0)
        if cs.get("n_intersect_3", 0) > 0:
            compound3_q += 1
            compound3_gold += cs.get("gold_in_intersect_3", 0)
        for g in row["gold"]:
            if g["has_rarest_1"]:
                agg["gold_rarest_1"] += 1
            if g["has_rarest_2"]:
                agg["gold_rarest_2"] += 1
            if g["has_rarest_3"]:
                agg["gold_rarest_3"] += 1
            if g["in_pool"]:
                agg["gold_in_pool"] += 1
            if g["in_top10"]:
                agg["gold_in_top10"] += 1
            if g["n_bridge_paths"]:
                bridge_gold += 1
            if g["n_teach_paths"]:
                teach_gold += 1
            if g["n_rare_pairs_in_doc"]:
                compound_pair_gold += 1
            if g["subword_hits"]:
                subword_gold += 1
        for f in row["false_top"]:
            if f["has_rarest_1"]:
                agg["false_rarest_1"] += 1
            if f.get("bridge_paths"):
                bridge_false += 1
        for h in row["hub_diluters"]:
            if h["in_false_top5"] > h["in_gold"]:
                hub_counter[h["word"]] += 1

    n_false = sum(len(r["false_top"]) for r in rows)

    def pct(a, b):
        return round(100.0 * a / max(b, 1), 1)

    return {
        "queries": n_q,
        "gold_doc_instances": n_gold_inst,
        "false_doc_instances": n_false,
        "queries_gold_in_pool_pct": pct(pool_hit_q, n_q),
        "queries_gold_in_top10_pct": pct(top10_hit_q, n_q),
        "gold_has_rarest_1_pct": pct(agg["gold_rarest_1"], n_gold_inst),
        "gold_has_rarest_2_pct": pct(agg["gold_rarest_2"], n_gold_inst),
        "gold_has_rarest_3_pct": pct(agg["gold_rarest_3"], n_gold_inst),
        "gold_in_pool_pct": pct(agg["gold_in_pool"], n_gold_inst),
        "gold_in_top10_pct": pct(agg["gold_in_top10"], n_gold_inst),
        "gold_with_bridge_path_pct": pct(bridge_gold, n_gold_inst),
        "gold_with_teach_path_pct": pct(teach_gold, n_gold_inst),
        "false_with_bridge_path_pct": pct(bridge_false, n_false),
        "gold_rare_pair_cooccur_pct": pct(compound_pair_gold, n_gold_inst),
        "gold_subword_hit_pct": pct(subword_gold, n_gold_inst),
        "false_has_rarest_1_pct": pct(agg["false_rarest_1"], n_false),
        "bucket_counts": dict(bucket),
        "top_hub_diluters": hub_counter.most_common(15),
        "compound_intersect_2": {
            "queries_with_intersection": compound2_q,
            "gold_docs_in_intersect": compound2_gold,
        },
        "compound_intersect_3": {
            "queries_with_intersection": compound3_q,
            "gold_docs_in_intersect": compound3_gold,
        },
    }


def run_corpus_audit(
    brain: MultiCorpusBrain,
    corpus_name: str,
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    *,
    qids: list[str] | None = None,
    rare_df_cap: int = 256,
) -> tuple[list[dict], dict]:
    """Audit one corpus; returns (per-query rows, aggregate summary)."""
    from aethos_multi_corpus import IdfCache

    branch = brain._corpora[corpus_name]
    idf = IdfCache(branch.idx, branch.n_docs)
    if qids is None:
        qids = [q for q in qrels if q in queries]
    rows = [
        audit_one(brain, branch, qid, queries[qid], qrels[qid], idf=idf, rare_df_cap=rare_df_cap)
        for qid in qids
    ]
    return rows, summarize_audit_rows(rows)
