#!/usr/bin/env python3
"""
Multi-dimensional speed benchmark — build, routing, scoring, micro-ops, scale.

Speed types measured:
  1. ingest throughput (docs/s, tokens/s, ms/doc)
  2. index build (hubs, meet, attractor, notch)
  3. query profile + cell profile
  4. candidate routing (BIT 4 vs legacy cascade)
  5. hub ranking (|C|=cap vs full corpus)
  6. micro-ops (cell, κ, meet probe, critical line)
  7. corpus scale curve (scientific prose N docs)
  8. seven data types (from compression corpora)

Output: stdout table + logs/speed_multi_benchmark.json
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aethos_hub_signature import (
    build_all_hub_signatures,
    build_query_profile,
    rank_with_hub_signatures,
)
from aethos_intersection_nodes import IntersectionNetwork
from aethos_spring_complex import verify_critical_line_rotation
from eval_beir import (
    build_meet_index,
    build_neighbor_weights,
    candidate_ids,
    ingest_corpus,
    make_pipeline,
    _tune_registry_for_beir,
)
from pipeline.bit_01_word_cell import word_to_spacetime_cell
from pipeline.bit_02_attractor_key import kappa_from_cell
from pipeline.bit_03_doc_attractor_set import build_attractor_index_from_hub_signatures
from pipeline.bit_04_candidate_router import route_query_candidates
from pipeline.bit_06_notch_bind import build_all_notch_fingerprints
from pipeline.bit_09_query_cell_profile import build_query_cell_profile
from scripts.compression_seven_types import build_corpora, TypeCorpus

QUERY_BY_TYPE: dict[str, list[str]] = {
    "scientific_prose": [
        "vitamin D inflammatory cytokine macrophages",
        "mRNA vaccines neutralizing antibody titers",
        "CRISPR double strand breaks NHEJ repair",
    ],
    "technical_vocabulary": [
        "autophagy lysosome proteasome ubiquitin",
        "kinase phosphatase methyltransferase",
        "hypoxia angiogenesis vasculogenesis",
    ],
    "repetitive_logs": [
        "worker task complete status OK",
        "METRIC cpu mem disk shard",
        "latency_ms task complete",
    ],
    "random_incompressible": [
        "xK9mQ2pL8nR4vT7wY1zA",
        "base64 random payload decode",
        "incompressible entropy stream",
    ],
    "structured_json": [
        "batch readings sensor temp value",
        "json batch id sensor ok",
        "readings value sensor batch",
    ],
    "morphological_variants": [
        "autophagy autophagic autophagosome",
        "phosphorylate phosphorylation phosphatase",
        "inflammation inflammatory immunomodulatory",
    ],
    "long_phrase_correlations": [
        "mitochondrial oxidative stress response pathway",
        "blood brain barrier neuroinflammatory episodes",
        "single cell RNA sequencing tumor microenvironment",
    ],
}


@dataclass
class TimingStats:
    n: int = 0
    samples_ms: list[float] = field(default_factory=list)

    def add(self, ms: float) -> None:
        self.samples_ms.append(ms)
        self.n += 1

    def p50(self) -> float:
        if not self.samples_ms:
            return 0.0
        return statistics.median(self.samples_ms)

    def p95(self) -> float:
        if not self.samples_ms:
            return 0.0
        xs = sorted(self.samples_ms)
        i = min(len(xs) - 1, int(0.95 * (len(xs) - 1)))
        return xs[i]

    def mean(self) -> float:
        return statistics.mean(self.samples_ms) if self.samples_ms else 0.0

    def per_sec(self) -> float:
        m = self.mean()
        return 1000.0 / m if m > 0 else 0.0


@dataclass
class CorpusSpeedState:
    pipe: object
    cidx: object
    hub_sigs: dict
    neighbor_map: dict
    meet_index: dict
    attractor_index: object
    notch_fps: dict
    build_ms: dict[str, float]
    n_docs: int
    n_tokens: int


def _time_ms(fn) -> tuple[float, object]:
    t0 = time.perf_counter()
    out = fn()
    return (time.perf_counter() - t0) * 1000.0, out


def build_state(corpus: TypeCorpus, *, mode: str = "scale") -> CorpusSpeedState:
    pipe = make_pipeline(mode)
    _tune_registry_for_beir(pipe.registry)

    beir = {did: {"text": txt} for did, txt in corpus.docs.items()}
    ingest_ms, (metrics, cidx) = _time_ms(lambda: ingest_corpus(pipe, beir, mode=mode))

    hub_ms, hub_sigs = _time_ms(
        lambda: build_all_hub_signatures(
            cidx.doc_ids, cidx.doc_tokens, pipe.registry, top_k=12,
        )
    )
    meet_ms, meet_index = _time_ms(lambda: build_meet_index(hub_sigs, pipe.registry))
    attr_ms, attractor_index = _time_ms(
        lambda: build_attractor_index_from_hub_signatures(pipe.registry, hub_sigs)
    )
    notch_ms, notch_fps = _time_ms(
        lambda: build_all_notch_fingerprints(hub_sigs, pipe.registry)
    )
    nb_ms, neighbor_map = _time_ms(lambda: build_neighbor_weights(pipe.registry))

    n_tokens = sum(
        sum(cidx.doc_tf.get(d, {}).values()) for d in cidx.doc_ids
    )

    return CorpusSpeedState(
        pipe=pipe,
        cidx=cidx,
        hub_sigs=hub_sigs,
        neighbor_map=neighbor_map,
        meet_index=meet_index,
        attractor_index=attractor_index,
        notch_fps=notch_fps,
        build_ms={
            "ingest": ingest_ms,
            "hubs": hub_ms,
            "meet": meet_ms,
            "attractor": attr_ms,
            "notch": notch_ms,
            "neighbor_map": nb_ms,
        },
        n_docs=len(cidx.doc_ids),
        n_tokens=n_tokens,
    )


def bench_queries(state: CorpusSpeedState, queries: list[str]) -> dict[str, object]:
    pipe = state.pipe
    cidx = state.cidx
    n_docs = len(cidx.doc_ids)

    route_bit4 = TimingStats()
    route_legacy = TimingStats()
    profile_only = TimingStats()
    cell_profile = TimingStats()
    rank_cap350 = TimingStats()
    rank_cap100 = TimingStats()
    cand_sizes_bit4: list[int] = []
    cand_sizes_legacy: list[int] = []
    tiers: dict[str, int] = {}

    meet_arg = state.meet_index

    for q in queries:
        def do_profile():
            return build_query_profile(
                q, pipe.registry,
                neighbor_map=state.neighbor_map,
                doc_freq=cidx.doc_freq,
                n_docs=n_docs,
            )

        ms, profile = _time_ms(do_profile)
        profile_only.add(ms)

        ms, _ = _time_ms(
            lambda: build_query_cell_profile(
                pipe.registry,
                q,
                neighbor_map=state.neighbor_map,
                doc_freq=cidx.doc_freq,
                n_docs=n_docs,
            )
        )
        cell_profile.add(ms)

        def do_bit4():
            return route_query_candidates(
                profile.words,
                pipe.registry,
                state.attractor_index,
                cidx.inv,
                state.neighbor_map,
                cidx.doc_ids,
                meet_index=meet_arg,
                doc_freq=cidx.doc_freq,
                n_docs=n_docs,
            )

        ms, route = _time_ms(do_bit4)
        route_bit4.add(ms)
        cand_sizes_bit4.append(len(route.doc_ids))
        tiers[route.tier] = tiers.get(route.tier, 0) + 1

        def do_legacy():
            return candidate_ids(
                profile.words,
                cidx.inv,
                state.neighbor_map,
                cidx.doc_ids,
                meet_index=meet_arg,
                registry=pipe.registry,
            )

        ms, legacy_cands = _time_ms(do_legacy)
        route_legacy.add(ms)
        cand_sizes_legacy.append(len(legacy_cands))

        cell = build_query_cell_profile(
            pipe.registry,
            q,
            neighbor_map=state.neighbor_map,
            doc_freq=cidx.doc_freq,
            n_docs=n_docs,
        )

        for cap, bucket in ((350, rank_cap350), (100, rank_cap100)):
            def do_rank(c=cap, r=route):
                return rank_with_hub_signatures(
                    profile,
                    r.doc_ids,
                    state.hub_sigs,
                    cidx.doc_ids,
                    doc_tokens=cidx.doc_tokens,
                    doc_tf=cidx.doc_tf,
                    doc_len=cidx.doc_len,
                    avg_dl=cidx.avg_dl,
                    registry=pipe.registry,
                    attractor_index=state.attractor_index,
                    query_kappa_keys=cell.kappa_neighbor_q,
                    kappa_candidate_cap=c,
                    top_k=10,
                )

            ms, _ = _time_ms(do_rank)
            bucket.add(ms)

    return {
        "route_bit4_ms": {"p50": route_bit4.p50(), "p95": route_bit4.p95(), "mean": route_bit4.mean()},
        "route_legacy_ms": {"p50": route_legacy.p50(), "p95": route_legacy.p95(), "mean": route_legacy.mean()},
        "profile_ms": {"p50": profile_only.p50(), "mean": profile_only.mean()},
        "cell_profile_ms": {"p50": cell_profile.p50(), "mean": cell_profile.mean()},
        "rank_cap350_ms": {"p50": rank_cap350.p50(), "p95": rank_cap350.p95()},
        "rank_cap100_ms": {"p50": rank_cap100.p50(), "p95": rank_cap100.p95()},
        "cand_bit4_p50": int(statistics.median(cand_sizes_bit4)) if cand_sizes_bit4 else 0,
        "cand_bit4_p95": sorted(cand_sizes_bit4)[int(0.95 * (len(cand_sizes_bit4) - 1))] if cand_sizes_bit4 else 0,
        "cand_legacy_p50": int(statistics.median(cand_sizes_legacy)) if cand_sizes_legacy else 0,
        "route_tiers": tiers,
        "queries": len(queries),
    }


def bench_micro(registry, *, n: int = 500) -> dict[str, float]:
    words = [
        "autophagy", "mitochondrial", "vitamin", "kinase", "inflammation",
        "phosphorylation", "vaccine", "cytokine", "sequencing", "hypoxia",
    ]
    net = IntersectionNetwork()

    def cell_loop():
        for i in range(n):
            word_to_spacetime_cell(registry, words[i % len(words)])

    def kappa_loop():
        for i in range(n):
            c = word_to_spacetime_cell(registry, words[i % len(words)])
            kappa_from_cell(c)

    def meet_loop():
        for _ in range(n):
            net.probe_solo_swap(3, 5)

    def critical_loop():
        for _ in range(n // 10):
            verify_critical_line_rotation()

    out = {}
    for name, fn in (
        ("cell_per_1k", lambda: cell_loop()),
        ("kappa_per_1k", lambda: kappa_loop()),
        ("solo_meet_per_1k", lambda: meet_loop()),
        ("critical_line_per_100", lambda: critical_loop()),
    ):
        t0 = time.perf_counter()
        fn()
        elapsed = (time.perf_counter() - t0) * 1000.0
        if "per_1k" in name:
            out[name] = elapsed / (n / 1000.0)
        else:
            out[name] = elapsed / (n / 100.0)
    return out


def bench_scale() -> list[dict]:
    corpora = build_corpora()
    sci = next(c for c in corpora if c.name == "scientific_prose")
    docs = list(sci.docs.items())
    sizes = [10, 20, 50, min(100, len(docs))]
    rows = []
    for n in sizes:
        sub = TypeCorpus(
            name=f"scientific_n{n}",
            description=f"first {n} scientific docs",
            docs=dict(docs[:n]),
        )
        state = build_state(sub)
        q = bench_queries(state, QUERY_BY_TYPE["scientific_prose"])
        total_build = sum(state.build_ms.values())
        rows.append({
            "n_docs": n,
            "ingest_ms_per_doc": state.build_ms["ingest"] / max(n, 1),
            "total_build_ms": total_build,
            "route_bit4_p50_ms": q["route_bit4_ms"]["p50"],
            "rank_cap350_p50_ms": q["rank_cap350_ms"]["p50"],
            "cand_p50": q["cand_bit4_p50"],
        })
    return rows


def main() -> int:
    print("=" * 88)
    print("  AETHOS multi-speed benchmark")
    print("=" * 88)

    # Micro-ops on a warmed registry
    print("\n[1] Micro-ops (500 iters, warmed registry)...", flush=True)
    warm = build_state(build_corpora()[0])
    micro = bench_micro(warm.pipe.registry, n=500)
    for k, v in micro.items():
        print(f"  {k:<28} {v:8.3f} ms")

    # Seven data types
    print("\n[2] Seven data types — build + query speed...", flush=True)
    type_rows = []
    for corpus in build_corpora():
        print(f"  {corpus.name}...", flush=True)
        state = build_state(corpus)
        queries = QUERY_BY_TYPE.get(corpus.name, QUERY_BY_TYPE["scientific_prose"])
        qstats = bench_queries(state, queries)
        ingest_ms = state.build_ms["ingest"]
        n = state.n_docs
        type_rows.append({
            "type": corpus.name,
            "n_docs": n,
            "n_tokens": state.n_tokens,
            "ingest_ms_total": ingest_ms,
            "ingest_ms_per_doc": ingest_ms / max(n, 1),
            "ingest_docs_per_s": n / (ingest_ms / 1000.0) if ingest_ms > 0 else 0,
            "build_ms": state.build_ms,
            "query": qstats,
        })

    print()
    hdr = (
        f"{'type':<22} {'docs':>5} {'ing/ms':>8} {'rt4':>7} {'leg':>7} "
        f"{'rk350':>7} {'rk100':>7} {'|C|50':>7} {'tier':>12}"
    )
    print(hdr)
    print("-" * len(hdr))
    for row in type_rows:
        q = row["query"]
        tier = max(q["route_tiers"], key=q["route_tiers"].get) if q["route_tiers"] else "—"
        print(
            f"{row['type']:<22} {row['n_docs']:>5} "
            f"{row['ingest_ms_per_doc']:>8.1f} "
            f"{q['route_bit4_ms']['p50']:>7.2f} "
            f"{q['route_legacy_ms']['p50']:>7.2f} "
            f"{q['rank_cap350_ms']['p50']:>7.2f} "
            f"{q['rank_cap100_ms']['p50']:>7.2f} "
            f"{q['cand_bit4_p50']:>7} "
            f"{tier:>12}"
        )

    # Scale curve
    print("\n[3] Scale curve (scientific prose)...", flush=True)
    scale = bench_scale()
    print(f"  {'n_docs':>6} {'ing/ms':>8} {'build':>8} {'route':>8} {'rank350':>8} {'|C|':>6}")
    for r in scale:
        print(
            f"  {r['n_docs']:>6} {r['ingest_ms_per_doc']:>8.1f} "
            f"{r['total_build_ms']:>8.0f} {r['route_bit4_p50_ms']:>8.2f} "
            f"{r['rank_cap350_p50_ms']:>8.2f} {r['cand_p50']:>6}"
        )

    # Speed utilization guide
    print("\n" + "=" * 88)
    print("SPEED TYPES & HOW TO USE THEM")
    print("=" * 88)
    print("""
  A. INGEST THROUGHPUT     — batch corpus build; scale mode + defer_l2
  B. INDEX BUILD           — one-time hubs/meet/attractor; amortize over queries
  C. BIT4 ROUTING          — route_bit4 p50; use when |C| << corpus (repetitive/morph)
  D. LEGACY CASCADE        — fallback tiers; slower when BIT4 hits bit4_fallback
  E. RANK w/ CAP 350       — default eval path; rank_cap350 dominates query time
  F. RANK w/ CAP 100       — faster scoring; use when latency-critical
  G. MICRO CELL/KAPPA      — per-token geometry; cache cells per registry word
  H. CANDIDATE TRIM        — |C| p50/p95; κ-hit trim + lexical protect (BIT4)
""")
    print("Columns: ing/ms = ingest ms/doc; rt4 = BIT4 route p50 ms; leg = legacy route;")
    print("         rk350/rk100 = rank_with_hub_signatures p50 ms; |C|50 = median candidates")
    print("=" * 88)

    out = ROOT / "logs" / "speed_multi_benchmark.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"micro": micro, "types": type_rows, "scale": scale}, indent=2),
        encoding="utf-8",
    )
    print(f"JSON: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
