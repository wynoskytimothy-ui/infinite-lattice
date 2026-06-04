#!/usr/bin/env python3
"""
Small-corpus diagnostic — trace tokens + semantic layers for explicit queries.

Usage:
  python diagnose_corpus.py
  python diagnose_corpus.py --save fixtures/diag_reader.json
"""

from __future__ import annotations

import argparse
import textwrap
from dataclasses import dataclass
from typing import Iterable

from aethos_hilbert_lattice import build_robust_space_from_corpus
from aethos_pipeline import AethosPipeline, check_promotion_invariants
from aethos_promotion import LatticeTier


# ---------------------------------------------------------------------------
# Tiny "real" corpus — enough co-occurrence for clusters, small enough to read
# ---------------------------------------------------------------------------

SMALL_CORPUS: tuple[str, ...] = (
    "the phone has a fast chip and runs technical software",
    "phone technical support called about network hardware",
    "apple released a new phone with a better chip",
    "apple fruit pie recipe from the orchard",
    "fresh apple and banana fruit salad for dessert",
    "the cat sat on the mat",
    "bat and tab are anagram words",
)

QUERIES: tuple[tuple[str, list[str]], ...] = (
    ("apple", ["phone", "chip"]),
    ("apple", ["fruit", "pie"]),
    ("phone", ["technical", "software"]),
    ("phone", ["fruit", "pie"]),  # wrong-context probe
    ("bat", ["tab", "cat"]),
    ("cat", []),
    ("zebra", ["animal"]),  # OOV probe — not in corpus
)


@dataclass
class QueryDiag:
    word: str
    context: list[str]
    tier: str
    intersection_only: bool
    prime: int
    parent_primes: tuple[int, ...]
    lattice_local: tuple[float, float, float]
    cluster_id: str
    cluster_score: float
    cluster_hub: str
    neighbors_l46: list[tuple[str, float]]
    corr_inner_vs_context: list[tuple[str, float]]
    overlay_registry_codec_match: bool | None
    flags: list[str]


def _section(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def _sub(title: str) -> None:
    print(f"\n--- {title} ---")


def diagnose_queries(
    pipe: AethosPipeline,
    queries: Iterable[tuple[str, list[str]]],
    *,
    hs=None,
) -> list[QueryDiag]:
    reg = pipe.registry
    reader = pipe.reader
    out: list[QueryDiag] = []

    for word, ctx in queries:
        flags: list[str] = []
        r = pipe.resolve(word, ctx)
        w = word.lower()

        if w not in reg.word_counts:
            flags.append("OOV: word never seen in corpus")
            if r.get("cluster_id"):
                flags.append(f"OOV should not map to cluster (got {r.get('cluster_id')!r})")

        neighbors: list[tuple[str, float]] = []
        for link in reg.correlations_for(w):
            other = link.target if link.source == w else link.source
            score = float(link.strength) * (link.dim4 + link.dim6 + 1.0)
            neighbors.append((other, score))
        neighbors.sort(key=lambda x: -x[1])

        corr_ctx: list[tuple[str, float]] = []
        if hs is not None:
            for c in ctx:
                corr_ctx.append((c, hs.correlation_inner_words(w, c)))

        overlay_match = None
        if w in reg.word_counts or w in reg.intersections or (LatticeTier.L3_WORD, w) in reg.promoted:
            try:
                ov = pipe.semantic_overlay(w, include_word_dot=True)
                overlay_match = ov.registry_equals_codec_local
                if not overlay_match:
                    flags.append("overlay mismatch: registry_local != codec_local")
            except ValueError:
                flags.append("overlay failed (no alphabetic anchors?)")

        cid = str(r.get("cluster_id", ""))
        hub = reader.cluster_hubs.get(cid, cid)
        if cid.startswith("theme_") and hub == cid:
            flags.append(f"suspicious cluster hub equals cluster id ({cid})")
        if ctx:
            ctx_clusters = {reader.word_to_cluster.get(c.lower()) for c in ctx}
            ctx_clusters.discard(None)
            if len(ctx_clusters) > 1:
                flags.append(f"context words span multiple clusters: {ctx_clusters}")
            if cid and ctx_clusters and cid not in ctx_clusters and not flags:
                # only flag if none of context shares cluster with result
                if not any(reader.word_to_cluster.get(c.lower()) == cid for c in ctx):
                    flags.append(f"query cluster {cid!r} not aligned with context clusters {ctx_clusters}")

        # bridge pollution: food query pulling tech words in related list
        if ctx and any(x in {"fruit", "pie", "orchard", "dessert", "salad"} for x in ctx):
            related = reader.related_in_cluster(cid, 8) if cid else []
            tech_hits = [t for t, _ in related if t in {"phone", "chip", "technical", "software", "hardware"}]
            if tech_hits:
                flags.append(f"food-context cluster still lists tech neighbors: {tech_hits}")

        out.append(
            QueryDiag(
                word=w,
                context=list(ctx),
                tier=str(r.get("tier", "")),
                intersection_only=bool(r.get("intersection_only")),
                prime=int(r.get("prime") or 0),
                parent_primes=tuple(r.get("parent_primes") or ()),
                lattice_local=tuple(r.get("lattice_local") or (0, 0, 0)),
                cluster_id=cid,
                cluster_score=float(r.get("cluster_score") or 0),
                cluster_hub=str(hub),
                neighbors_l46=neighbors[:8],
                corr_inner_vs_context=corr_ctx,
                overlay_registry_codec_match=overlay_match,
                flags=flags,
            )
        )
    return out


def print_corpus_overview(pipe: AethosPipeline, corpus: tuple[str, ...]) -> None:
    _section("CORPUS INGEST")
    print(f"  documents: {len(corpus)}")
    for i, doc in enumerate(corpus, 1):
        print(f"  [{i}] {doc}")
    print()
    print(pipe.report().summary())
    inv = check_promotion_invariants(pipe.registry)
    if inv:
        print("\n  INVARIANT VIOLATIONS:")
        for e in inv:
            print(f"    ! {e}")


def print_word_table(pipe: AethosPipeline) -> None:
    reg = pipe.registry
    _sub("Word promotion table (sorted by count)")
    rows = sorted(reg.word_counts.items(), key=lambda x: (-x[1], x[0]))
    print(f"  {'word':14s} {'cnt':>4s}  {'tier':18s}  {'prime':>6s}  contexts_differ")
    for w, cnt in rows:
        dedicated = (LatticeTier.L3_WORD, w) in reg.promoted
        tier = "dedicated_l3" if dedicated else "intersection"
        tok = reg.resolve_token(w)
        diff = reg.contexts_differ(w)
        print(f"  {w:14s} {cnt:4d}  {tier:18s}  {tok.prime:6d}  {diff}")


def print_clusters(pipe: AethosPipeline) -> None:
    reader = pipe.reader
    _sub("Emergent clusters")
    if not reader.cluster_hubs:
        print("  (none discovered)")
        return
    for cid, hub in sorted(reader.cluster_hubs.items()):
        cat = reader.cross.categories.get(cid)
        members = sorted(w for w, c in reader.word_to_cluster.items() if c == cid)
        print(f"\n  {cid}  hub={hub!r}")
        print(f"    members ({len(members)}): {members}")
        if cat:
            print(f"    top words:  {cat.top_words(6)}")
            print(f"    top primes: {cat.top_primes(4)}")


def print_correlation_graph(pipe: AethosPipeline, hubs: tuple[str, ...] = ("phone", "apple", "fruit")) -> None:
    reg = pipe.registry
    _sub("L4-L6 neighbor graph (selected hubs)")
    for w in hubs:
        links = reg.correlations_for(w)
        if not links:
            print(f"  {w}: (no edges)")
            continue
        parts = []
        for link in links[:6]:
            other = link.target if link.source == w else link.source
            parts.append(f"{other}(s={link.strength})")
        print(f"  {w}: {', '.join(parts)}")


def print_query_results(diags: list[QueryDiag]) -> None:
    _section("QUERY TRACE")
    for d in diags:
        ctx_s = ", ".join(d.context) if d.context else "(none)"
        print(f"\n  QUERY: {d.word!r}  context=[{ctx_s}]")
        print(f"    promotion:     {d.tier}  prime={d.prime}  parents={d.parent_primes}")
        print(f"    lattice local: {d.lattice_local}")
        print(f"    cluster:       {d.cluster_id!r}  hub={d.cluster_hub!r}  score={d.cluster_score:.4f}")
        if d.neighbors_l46:
            nb = ", ".join(f"{w}({s:.2f})" for w, s in d.neighbors_l46[:5])
            print(f"    L4-L6 neighbors: {nb}")
        if d.corr_inner_vs_context:
            ci = ", ".join(f"{w}={v:.3f}" for w, v in d.corr_inner_vs_context)
            print(f"    Hilbert corr:    {ci}")
        if d.overlay_registry_codec_match is not None:
            print(f"    overlay match:   {d.overlay_registry_codec_match}")
        if d.flags:
            print("    FLAGS:")
            for f in d.flags:
                print(f"      >> {f}")


def print_issue_summary(diags: list[QueryDiag], pipe: AethosPipeline) -> None:
    _section("ISSUE SUMMARY (candidates to fix)")
    issues: list[str] = []

    flagged = [d for d in diags if d.flags]
    if flagged:
        issues.append(f"{len(flagged)} queries raised flags (see QUERY TRACE)")

    reg = pipe.registry
    dedicated = sum(1 for k in reg.promoted if k[0] == LatticeTier.L3_WORD)
    intersection = len(reg.intersections)
    if dedicated > intersection:
        issues.append(
            f"more dedicated L3 ({dedicated}) than intersection-only ({intersection}) "
            f"— pool primes may be spent aggressively on small corpus"
        )

    reader = pipe.reader
    bad_hubs = [cid for cid, hub in reader.cluster_hubs.items() if hub == cid or hub.startswith("theme_theme")]
    if bad_hubs:
        issues.append(f"cluster hub naming looks broken: {bad_hubs}")

    bridges = []
    for w in reg.word_counts:
        if reg.contexts_differ(w):
            clusters = {reader.word_to_cluster.get(w)}
            clusters.discard(None)
            # apple should map to multiple cluster attachments as bridge
            nbr_clusters = {
                reader.word_to_cluster.get(other)
                for link in reg.correlations_for(w)
                for other in (
                    link.target if link.source == w else link.source,
                )
            }
            nbr_clusters.discard(None)
            if len(nbr_clusters) > 1 and len(clusters) <= 1:
                bridges.append(w)
    if bridges:
        issues.append(f"bridge words may not be multi-attached cleanly: {bridges}")

    apple_food = next((d for d in diags if d.word == "apple" and "fruit" in d.context), None)
    apple_tech = next((d for d in diags if d.word == "apple" and "phone" in d.context), None)
    if apple_food and apple_tech:
        if apple_food.cluster_id == apple_tech.cluster_id:
            issues.append("apple disambiguation FAILED: same cluster for tech vs food context")
        else:
            issues.append(
                f"apple disambiguation OK: tech->{apple_tech.cluster_id}, "
                f"food->{apple_food.cluster_id}"
            )

    phone_wrong = next((d for d in diags if d.word == "phone" and "fruit" in d.context), None)
    if phone_wrong and phone_wrong.cluster_id.startswith("theme_fruit"):
        issues.append("phone wrongly routed to food cluster on unrelated context")
    elif phone_wrong and phone_wrong.cluster_id == "theme_phone":
        issues.append("phone stays in tech cluster when context is unrelated (OK)")

    if not issues:
        print("  No automatic issues detected.")
    else:
        for i, msg in enumerate(issues, 1):
            print(f"  {i}. {msg}")

    print(
        textwrap.dedent(
            """
  Likely fix areas (from architecture):
    - Bridge cluster cold-start creates theme_<first_word> hubs that absorb noise
    - word_to_cluster.setdefault for bridges keeps only ONE cluster, not multi-map
    - infer_cluster adds context cluster boost but bridge attachment may pollute food cluster
    - L4-L6 dims are hash-derived, not PMI — strength only reflects co-occurrence count
    - dedicated L3 triggers on 2+ docs with Jaccard<0.45 — may fire too early on tiny corpus
            """
        ).strip()
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose small corpus + queries")
    parser.add_argument("--save", metavar="PATH", help="save reader state after ingest")
    args = parser.parse_args()

    _section("AETHOS CORPUS DIAGNOSTIC")
    print("  Feeding small corpus, then running fixed queries layer-by-layer.")

    pipe = AethosPipeline(rebuild_every=2)
    pipe.ingest(*SMALL_CORPUS)

    if args.save:
        pipe.save(args.save)
        print(f"\n  Saved reader state -> {args.save}")

    print_corpus_overview(pipe, SMALL_CORPUS)
    print_word_table(pipe)
    print_clusters(pipe)
    print_correlation_graph(pipe)

    hs = build_robust_space_from_corpus(*SMALL_CORPUS)
    diags = diagnose_queries(pipe, QUERIES, hs=hs)
    print_query_results(diags)
    print_issue_summary(diags, pipe)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
