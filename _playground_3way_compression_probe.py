#!/usr/bin/env python3
"""Probe 3-way semantic compression + correlation chain (exploration only)."""

from __future__ import annotations

import json
import math
import struct
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aethos_promotion import PromotionRegistry
from aethos_words import letter_to_prime
from lattice_retriever_v1.doc_lattice_codec import (
    DocPrimePool,
    build_doc_correlation_shells,
    build_rare_correlation_shells,
    encode_doc,
    select_rare_in_doc,
    _cage_composite_for_triple,
    _words,
)
from lattice_retriever_v1.k_meet import compose_k
from lattice_retriever_v1.stage04_promote import Stage04Registry, promote_from_stream
from lattice_retriever_v1.stage07_semantic_light import (
    SemanticLightIndex,
    anchor_composite,
    word_path_identities,
)
from lattice_retriever_v1.walker_maxsim_retriever import word_pair_walk, OrientedPairIndex


def _pair_key(a: str, b: str, sem: SemanticLightIndex) -> int:
    pa, pb = sem._prime_for_term(a), sem._prime_for_term(b)
    return min(pa, pb) * max(pa, pb)


def exp1_synthetic_chain() -> dict:
    """Trace symbol→letter→subword→word→3-way cage→correlation for cat/pet/purr."""
    corpus = [
        "cat purrs loudly",
        "pet purrs softly",
        "cat and pet purr",
        "dog barks loudly",
    ]
    reg = promote_from_stream(corpus)
    sem = SemanticLightIndex(registry=reg)
    for t in corpus:
        sem.observe_doc(t)

    terms = ("cat", "pet", "purr", "dog")
    chain: dict[str, dict] = {}
    for term in terms:
        t = term.lower()
        entry: dict = {"term": t}
        # L1 letters
        entry["letters"] = {c: letter_to_prime(c) for c in t if c.isalpha()}
        # L2 promoted subwords (prefix/suffix scan)
        sw_hits = []
        for ln in range(2, min(5, len(t) + 1)):
            for sw in (t[:ln], t[-ln:]):
                tok = reg.promoted_subword(sw)
                if tok:
                    sw_hits.append({"subword": sw, "prime": tok.prime})
        entry["promoted_subwords"] = sw_hits
        # L3 word identity prime
        entry["word_prime"] = sem._prime_for_term(t)
        entry["corridor_pins"] = sorted(sem.corridor_pins_for_term(t))
        # k-meet sunflower witness on word prime factors (if composite)
        wp = entry["word_prime"]
        if wp > 100:  # likely composite
            try:
                report = compose_k(*sorted(set(entry["corridor_pins"]))[:3])
                entry["k_meet_unified"] = report.full_sunflower_unified
                entry["k_meet_k"] = report.k
            except Exception as e:
                entry["k_meet_error"] = str(e)
        chain[t] = entry

    # 3-way cages from cat-pet-purr docs
    cages = []
    for w1, w2, w3 in [("cat", "pet", "purr"), ("cat", "purrs", "loudly"), ("pet", "purrs", "softly")]:
        comp, primes, label = _cage_composite_for_triple(sem, w1, w2, w3)
        cage = sem._cage_for_triple(w1, w2, w3)
        cages.append(
            {
                "triple": label,
                "anchor_composite": comp,
                "anchor_primes": list(primes),
                "neighbors": sorted(cage.correlations.keys()),
                "cat_purr_touch": sem.touch_weight(["cat", "purr"], cage),
                "dog_touch": sem.touch_weight(["dog", "bark"], cage),
            }
        )

    # Pairwise misses: cat↔purr co-occur but not adjacent in "cat and pet purr"
    pair_keys = set()
    triple_keys = set()
    for doc in corpus:
        words = [w for w in doc.split() if w.isalpha()]
        for i in range(len(words) - 1):
            pair_keys.add(_pair_key(words[i], words[i + 1], sem))
        window = words[:6]
        for i in range(len(window) - 2):
            comp, _, _ = _cage_composite_for_triple(sem, *window[i : i + 3])
            triple_keys.add(comp)

    # Semantic link pairwise registry misses: cat-purr not in adjacent pairs
    inner = reg.registry if hasattr(reg, "registry") else PromotionRegistry()
    for doc in corpus:
        inner.observe_cooccurrence(doc.split())
    pair_corr = set()
    for (a, b), link in inner.correlations.items():
        if "cat" in (a, b) and "purr" in (a, b):
            pair_corr.add((a, b, link.strength))
    triple_cat_purr = sem._cage_for_triple("cat", "pet", "purr")
    triple_has_both = "cat" in triple_cat_purr.correlations and "purr" in triple_cat_purr.correlations

    return {
        "chain": chain,
        "cages": cages,
        "unique_pair_keys_in_corpus": len(pair_keys),
        "unique_triple_keys_in_corpus": len(triple_keys),
        "pairwise_cat_purr_links": list(pair_corr),
        "triple_cage_links_cat_and_purr": triple_has_both,
        "triple_score_cat_purr": sem.touch_weight(["cat", "purr"], triple_cat_purr),
        "pair_adjacent_only_misses": "cat and pet purr" in corpus,
    }


def _idf_sorted_window(words: list[str], sem: SemanticLightIndex, k: int = 6) -> list[str]:
    uniq = list(dict.fromkeys(words))
    uniq.sort(key=lambda w: (-sem.idf(w), w))
    return uniq[:k]


def exp4_idf_vs_positional(corpus: dict[str, str], reg: Stage04Registry) -> dict:
    """Compare cage quality: positional 6-word window vs idf-sorted 6-word window."""
    sem = SemanticLightIndex(registry=reg)
    for text in corpus.values():
        sem.observe_doc(text)

    pos_hits = pos_total = 0
    idf_hits = idf_total = 0
    samples = []

    for doc_id, text in list(corpus.items())[:200]:
        words = list(_words(text))
        if len(words) < 3:
            continue
        rare_in_doc = set(select_rare_in_doc(words, sem, k=8, max_df_frac=0.05))
        if len(rare_in_doc) < 2:
            continue

        pos_window = words[:6]
        idf_window = _idf_sorted_window(words, sem, 6)

        def cage_rare_coverage(window: list[str]) -> tuple[int, int]:
            covered: set[str] = set()
            for i in range(max(0, len(window) - 2)):
                triple = window[i : i + 3]
                comp, _, _ = _cage_composite_for_triple(sem, *triple)
                cage = sem.cages.get(comp)
                if cage:
                    covered |= rare_in_doc & set(cage.correlations.keys())
            return len(covered), len(rare_in_doc)

        ph, pt = cage_rare_coverage(pos_window)
        ih, it = cage_rare_coverage(idf_window)
        pos_hits += ph
        pos_total += pt
        idf_hits += ih
        idf_total += it
        if len(samples) < 5 and pt > 0:
            samples.append(
                {
                    "doc_id": doc_id,
                    "rare_in_doc": sorted(rare_in_doc),
                    "pos_window": pos_window,
                    "idf_window": idf_window,
                    "pos_coverage": f"{ph}/{pt}",
                    "idf_coverage": f"{ih}/{it}",
                }
            )

    return {
        "docs_sampled": len(samples),
        "positional_rare_coverage": pos_hits / pos_total if pos_total else 0,
        "idf_sorted_rare_coverage": idf_hits / idf_total if idf_total else 0,
        "idf_wins": idf_hits > pos_hits,
        "samples": samples,
    }


def exp2_cooccur_shared_anchor(corpus: dict[str, str], reg: Stage04Registry) -> dict:
    """Docs sharing rare terms: do they share anchor_composite? Count keys."""
    sem = SemanticLightIndex(registry=reg)
    for text in corpus.values():
        sem.observe_doc(text)

    # Build doc → triple keys and pair keys
    doc_triples: dict[str, set[int]] = defaultdict(set)
    doc_pairs: dict[str, set[tuple[str, str]]] = defaultdict(set)
    global_triple: dict[int, set[str]] = defaultdict(set)
    global_pair: dict[tuple[str, str], set[str]] = defaultdict(set)

    for doc_id, text in corpus.items():
        words = list(_words(text))
        window = words[:6]
        for i in range(len(window) - 2):
            comp, _, _ = _cage_composite_for_triple(sem, *window[i : i + 3])
            doc_triples[doc_id].add(comp)
            global_triple[comp].add(doc_id)
        walk = word_pair_walk(text, sem)
        for dot in walk:
            doc_pairs[doc_id].add(dot.origin.key)
            global_pair[dot.origin.key].add(doc_id)

    # Find doc pairs that share ≥2 rare terms
    doc_rare: dict[str, set[str]] = {}
    for doc_id, text in corpus.items():
        words = list(_words(text))
        doc_rare[doc_id] = set(select_rare_in_doc(words, sem, k=8, max_df_frac=0.05))

    shared_triple_anchor = 0
    shared_pair_key = 0
    pairs_checked = 0
    examples = []

    doc_ids = list(corpus.keys())
    for i, d1 in enumerate(doc_ids):
        for d2 in doc_ids[i + 1 : i + 51]:  # sample pairs
            shared_rare = doc_rare[d1] & doc_rare[d2]
            if len(shared_rare) < 2:
                continue
            pairs_checked += 1
            triple_overlap = doc_triples[d1] & doc_triples[d2]
            pair_overlap = doc_pairs[d1] & doc_pairs[d2]
            if triple_overlap:
                shared_triple_anchor += 1
            if pair_overlap:
                shared_pair_key += 1
            if len(examples) < 8 and (triple_overlap or pair_overlap):
                examples.append(
                    {
                        "d1": d1,
                        "d2": d2,
                        "shared_rare": sorted(shared_rare)[:6],
                        "shared_triple_anchors": len(triple_overlap),
                        "shared_pair_keys": len(pair_overlap),
                        "sample_triple": next(iter(triple_overlap), None),
                    }
                )

    return {
        "n_docs": len(corpus),
        "unique_triple_anchors_global": len(global_triple),
        "unique_oriented_pair_keys_global": len(global_pair),
        "compression_ratio_pair_to_triple": len(global_pair) / max(1, len(global_triple)),
        "doc_pairs_with_2plus_shared_rare": pairs_checked,
        "fraction_with_shared_triple_anchor": shared_triple_anchor / max(1, pairs_checked),
        "fraction_with_shared_pair_key": shared_pair_key / max(1, pairs_checked),
        "triple_only_links": shared_triple_anchor - shared_pair_key,
        "examples": examples,
    }


def exp3_minimal_doc_repr(corpus: dict[str, str], reg: Stage04Registry) -> dict:
    """Can doc = doc_prime + ordered rare triple meet keys (no full text)?"""
    pool = DocPrimePool()
    sem = SemanticLightIndex(registry=reg)
    for text in corpus.values():
        sem.observe_doc(text)

    bytes_triple_only = []
    bytes_full_shells = []
    bytes_order_stream = []
    recoverable = 0
    total = 0

    for doc_id, text in list(corpus.items())[:500]:
        placement = encode_doc(doc_id, text, reg, pool, semantic=sem)
        words = list(_words(text))
        rare = select_rare_in_doc(words, sem, k=8, max_df_frac=0.05)
        rare_set = set(rare)

        # Minimal: doc_prime + rare triple keys from idf-sorted window
        idf_window = _idf_sorted_window(words, sem, 6)
        triple_keys: list[int] = []
        for i in range(max(0, len(idf_window) - 2)):
            triple = idf_window[i : i + 3]
            if not any(t in rare_set for t in triple):
                continue
            comp, _, _ = _cage_composite_for_triple(sem, *triple)
            triple_keys.append(comp)

        # Dedupe preserving order
        seen: set[int] = set()
        ordered_keys = []
        for k in triple_keys:
            if k not in seen:
                seen.add(k)
                ordered_keys.append(k)

        b_min = 4 + 8 * len(ordered_keys)  # doc_prime u32 + keys u64
        b_full = 4 + 4 * len(placement.order_stream)  # doc_prime + order_stream u32
        shells = build_rare_correlation_shells(text, reg, sem, k=8)
        b_shell = 4 + sum(8 + 4 * len(s.neighbors) for s in shells)

        bytes_triple_only.append(b_min)
        bytes_order_stream.append(b_full)
        bytes_full_shells.append(b_shell)

        # Recoverability: can we reconstruct rare terms from triple keys?
        full_shell_comps = {s.anchor_composite for s in build_doc_correlation_shells(text, reg)}
        if rare_set:
            total += 1
            if set(ordered_keys) & full_shell_comps:
                recoverable += 1

    return {
        "docs_sampled": len(bytes_triple_only),
        "avg_bytes_triple_keys_only": sum(bytes_triple_only) / max(1, len(bytes_triple_only)),
        "avg_bytes_order_stream": sum(bytes_order_stream) / max(1, len(bytes_order_stream)),
        "avg_bytes_full_rare_shells": sum(bytes_full_shells) / max(1, len(bytes_full_shells)),
        "triple_vs_order_stream_ratio": (
            sum(bytes_triple_only) / max(1, sum(bytes_order_stream))
        ),
        "rare_term_recoverable_via_triple_keys": recoverable / max(1, total),
    }


def exp5_footprint(corpus: dict[str, str], reg: Stage04Registry) -> dict:
    """Bytes per doc: pair postings vs triple meet keys only."""
    sem = SemanticLightIndex(registry=reg)
    pair_idx = OrientedPairIndex()

    for doc_id, text in corpus.items():
        sem.observe_doc(text)
        walk = word_pair_walk(text, sem)
        pair_idx.index_doc(doc_id, walk)

    pair_bytes_per_doc = []
    triple_bytes_per_doc = []

    for doc_id, text in corpus.items():
        words = list(_words(text))
        window = words[:6]
        triples: set[int] = set()
        for i in range(len(window) - 2):
            comp, _, _ = _cage_composite_for_triple(sem, *window[i : i + 3])
            triples.add(comp)
        # posting: doc_id (~8B str ref) + u64 key per entry; amortize doc_id once
        triple_bytes_per_doc.append(8 * len(triples))

        pairs: set[tuple[str, str]] = set()
        walk = word_pair_walk(text, sem)
        for dot in walk:
            pairs.add(dot.origin.key)
        # oriented pair key: two term refs (~16B) + doc ref
        pair_bytes_per_doc.append(16 * len(pairs) + 8)

    n = len(corpus)
    return {
        "n_docs": n,
        "avg_pair_posting_bytes_per_doc": sum(pair_bytes_per_doc) / max(1, n),
        "avg_triple_key_bytes_per_doc": sum(triple_bytes_per_doc) / max(1, n),
        "pair_to_triple_byte_ratio": sum(pair_bytes_per_doc) / max(1, sum(triple_bytes_per_doc)),
        "total_unique_triple_anchors": len(
            {k for doc_id, text in corpus.items() for k in _doc_triple_keys(text, sem)}
        ),
        "total_unique_pair_keys": len(pair_idx.pair_postings),
    }


def _doc_triple_keys(text: str, sem: SemanticLightIndex) -> set[int]:
    words = list(_words(text))
    window = words[:6]
    out: set[int] = set()
    for i in range(len(window) - 2):
        comp, _, _ = _cage_composite_for_triple(sem, *window[i : i + 3])
        out.add(comp)
    return out


def exp_pairwise_vs_triple_semantic_proof(reg: Stage04Registry) -> dict:
    """Explicit proof case: 3-way links terms pairwise window misses."""
    sem = SemanticLightIndex(registry=reg)
    # Doc layout: rare terms separated by hub words — adjacent pairs miss A↔C
    docs = [
        "mitochondria produces energy in cells",
        "mitochondria the organelle generates cellular energy",
        "cells require mitochondria for atp synthesis",
    ]
    for d in docs:
        sem.observe_doc(d)

    terms = ("mitochondria", "energy", "cells", "atp")
    # Pair keys fromHint only adjacent
    pair_docs: dict[tuple[str, str], set[int]] = defaultdict(set)
    triple_docs: dict[int, set[int]] = defaultdict(set)
    for i, d in enumerate(docs):
        words = [w for w in d.split() if w.isalpha()]
        for j in range(len(words) - 1):
            pair_docs[(words[j], words[j + 1])].add(i)
        window = words[:6]
        for j in range(len(window) - 2):
            comp, _, _ = _cage_composite_for_triple(sem, *window[j : j + 3])
            triple_docs[comp].add(i)

    # Query: mitochondria + atp (never adjacent in doc 0)
    q = ("mitochondria", "atp")
    pair_hit_docs = set()
    for i in range(len(q) - 1):
        pair_hit_docs |= pair_docs.get((q[i], q[i + 1]), set())
        pair_hit_docs |= pair_docs.get((q[i + 1], q[i]), set())

    triple_hit_docs: set[int] = set()
    for comp, doc_idxs in triple_docs.items():
        cage = sem.cages.get(comp)
        if not cage:
            continue
        if all(t in cage.correlations for t in q):
            triple_hit_docs |= doc_idxs

    # Also check cage bridge: mitochondria cage neighbors include atp?
    mito_cages = [
        c for c in sem.cages.values() if "mitochondria" in c.anchor_label.split("|")
    ]
    bridge_terms = set()
    for c in mito_cages:
        for t in q:
            if t in c.correlations:
                bridge_terms.add(t)

    return {
        "query": q,
        "pair_routing_hit_docs": sorted(pair_hit_docs),
        "triple_cage_hit_docs": sorted(triple_hit_docs),
        "triple_finds_semantic_link_pair_misses": len(triple_hit_docs) > len(pair_hit_docs),
        "mito_cage_bridge_terms": sorted(bridge_terms),
        "doc0_words": docs[0].split(),
        "mitochondria_atp_adjacent_in_any_doc": any(
            ("mitochondria", "atp") in zip(words, words[1:])
            or ("atp", "mitochondria") in zip(words, words[1:])
            for words in ([w for w in d.split() if w.isalpha()] for d in docs)
        ),
    }


def load_scifact_subset(max_docs: int = 5183) -> dict[str, str]:
    try:
        from scripts.bench_lattice_retriever_v1 import load_scifact

        corpus, _, _ = load_scifact()
        return dict(list(corpus.items())[:max_docs])
    except Exception as e:
        return {"error": str(e)}


def main() -> int:
    print("=" * 60)
    print("EXP 1: Synthetic cat/pet/purr chain")
    print("=" * 60)
    e1 = exp1_synthetic_chain()
    print(json.dumps(e1, indent=2, default=str))

    print("\n" + "=" * 60)
    print("SEMANTIC PROOF: pairwise vs 3-way (mitochondria/atp)")
    print("=" * 60)

    # Build registry from larger synthetic + scifact if available
    scifact = load_scifact_subset()
    if "error" in scifact:
        print(f"SciFact unavailable: {scifact['error']}")
        scifact_corpus = {}
    else:
        print(f"SciFact loaded: {len(scifact)} docs")
        scifact_corpus = scifact

    synth_big = [
        "cat purrs loudly",
        "pet purrs softly",
        "mitochondria produces energy in cells",
        "mitochondria the organelle generates cellular energy",
    ]
    reg = promote_from_stream(list(scifact_corpus.values())[:800] if scifact_corpus else synth_big)
    proof = exp_pairwise_vs_triple_semantic_proof(reg)
    print(json.dumps(proof, indent=2, default=str))

    corpus_for_scale = scifact_corpus if scifact_corpus else {
        f"d{i}": t for i, t in enumerate(synth_big * 50)
    }
    reg_scale = promote_from_stream(list(corpus_for_scale.values()))

    print("\n" + "=" * 60)
    print("EXP 2: Co-occur rare terms -> shared anchor_composite")
    print("=" * 60)
    e2 = exp2_cooccur_shared_anchor(corpus_for_scale, reg_scale)
    print(json.dumps({k: v for k, v in e2.items() if k != "examples"}, indent=2))
    print("examples:", json.dumps(e2["examples"][:4], indent=2, default=str))

    print("\n" + "=" * 60)
    print("EXP 3: Minimal doc repr (prime + triple keys)")
    print("=" * 60)
    e3 = exp3_minimal_doc_repr(corpus_for_scale, reg_scale)
    print(json.dumps(e3, indent=2))

    print("\n" + "=" * 60)
    print("EXP 4: IDF-sorted vs positional 3-window")
    print("=" * 60)
    e4 = exp4_idf_vs_positional(corpus_for_scale, reg_scale)
    print(json.dumps(e4, indent=2, default=str))

    print("\n" + "=" * 60)
    print("EXP 5: Footprint pair vs triple")
    print("=" * 60)
    e5 = exp5_footprint(corpus_for_scale, reg_scale)
    print(json.dumps(e5, indent=2))

    # Minimal formula rule set summary
    print("\n" + "=" * 60)
    print("MINIMAL INGEST FORMULA (derived)")
    print("=" * 60)
    rules = [
        "1. token_prime(w) = promoted_subword(w).prime OR meet_composite_k(distinct letter primes)",
        "2. doc_prime = next_odd_from_append_pool(doc_id)",
        "3. rare(w) = df(w) <= max(1, n_docs * max_df_frac) AND w not in HUB_WORDS",
        "4. window = first 6 alpha tokens OR idf-descending 6 unique tokens (better rare coverage)",
        "5. for each sliding triple (t1,t2,t3) in window:",
        "     anchor_composite = product(sorted unique token_primes)  [2-way if duplicate]",
        "6. neighbors = triple terms + (window \\ triple); CorrelationLink dims from (anchor_p, term_p, strength)",
        "7. invert: anchor_composite → doc_id set; term → anchor_composite set (RareShellLatticeIndex)",
        "8. query route: rarest term → union/intersect anchor postings → rescore touch_weight on shared cages",
    ]
    for r in rules:
        print(r)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
