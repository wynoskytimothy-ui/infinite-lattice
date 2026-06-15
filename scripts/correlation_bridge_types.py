#!/usr/bin/env python3
"""
Cross-type correlation bridge analysis — 7 data types, one shared registry.

Shows how lattice layers link disparate data:
  - pool-prime factors (L3 ICN)
  - κ attractor buckets (BIT 2/3)
  - critical-line leg_sum pins
  - morph letter-prime overlap
  - meet-factor postings (BIT 7)
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aethos_composite import letter_composite_gcd_ratio
from aethos_hub_signature import (
    MIN_POOL_PRIME,
    build_all_hub_signatures,
    pool_factors_for_word,
)
from aethos_tokenize import tokenize_words
from eval_beir import build_meet_index, ingest_corpus, make_pipeline, _tune_registry_for_beir
from pipeline.bit_01_word_cell import word_to_spacetime_cell
from pipeline.bit_02_attractor_key import kappa_from_cell
from pipeline.bit_03_doc_attractor_set import build_attractor_index_from_hub_signatures
from scripts.compression_seven_types import build_corpora

CRITICAL_PIN_BYTES = 8


def leg_sum_side(coord: tuple[float, float, float]) -> tuple[int, bool]:
    x, y = int(round(coord[0])), int(round(coord[1]))
    return x + y, y > x


def unified_corpora():
    """All 7 types in one ingest with type tags on doc_id."""
    corpora = build_corpora()
    merged: dict[str, str] = {}
    doc_type: dict[str, str] = {}
    for corp in corpora:
        for did, text in corp.docs.items():
            uid = f"{corp.name}::{did}"
            merged[uid] = text
            doc_type[uid] = corp.name
    return merged, doc_type, [c.name for c in corpora]


def bridge_matrix(
    doc_type: dict[str, str],
    factor_docs: dict[int, set[str]],
    kappa_docs: dict[tuple, set[str]],
    sum_docs: dict[int, set[str]],
) -> dict[str, dict[str, int]]:
    """Count shared bridge keys between type pairs."""
    types = sorted(set(doc_type.values()))
    types_set = set(types)
    out: dict[str, dict[str, int]] = {t: {u: 0 for u in types} for t in types}

    def pair_count(bucket: dict, key_fn):
        type_keys: dict[str, set] = {t: set() for t in types}
        for key, docs in bucket.items():
            tags = {doc_type[d] for d in docs if d in doc_type}
            if len(tags) < 2:
                continue
            k = key_fn(key)
            for t in tags:
                type_keys[t].add(k)
        for a in types:
            for b in types:
                if a != b:
                    out[a][b] = len(type_keys[a] & type_keys[b])

    pair_count(factor_docs, lambda x: x)
    return out


def main() -> int:
    merged, doc_type, type_names = unified_corpora()
    beir = {did: {"text": txt} for did, txt in merged.items()}

    pipe = make_pipeline("scale")
    _tune_registry_for_beir(pipe.registry)
    _, cidx = ingest_corpus(pipe, beir, mode="scale")

    hub_sigs = build_all_hub_signatures(
        cidx.doc_ids, cidx.doc_tokens, pipe.registry, top_k=12,
    )
    attractor = build_attractor_index_from_hub_signatures(pipe.registry, hub_sigs)
    meet_index = build_meet_index(hub_sigs, pipe.registry)

    # Per-type vocabulary
    type_vocab: dict[str, set[str]] = defaultdict(set)
    for did, toks in cidx.doc_tokens.items():
        type_vocab[doc_type[did]] |= set(toks)

    # Shared words across types
    all_types = type_names
    print("=" * 88)
    print("  Cross-type correlation bridges (7 data types, unified registry)")
    print(f"  docs={len(cidx.doc_ids)}  types={len(all_types)}")
    print("=" * 88)

    print("\n[1] LEXICAL OVERLAP (shared tokens between types)")
    for i, a in enumerate(all_types):
        for b in all_types[i + 1 :]:
            inter = type_vocab[a] & type_vocab[b]
            if inter:
                sample = sorted(inter)[:8]
                print(f"  {a:<22} ↔ {b:<22}  {len(inter):3d} shared  e.g. {sample}")

    # Pool-factor bridges
    factor_to_docs: dict[int, set[str]] = defaultdict(set)
    factor_to_types: dict[int, set[str]] = defaultdict(set)
    word_factors: dict[str, frozenset[int]] = {}
    for did, sig in hub_sigs.items():
        t = doc_type[did]
        for word, entry in sig.hubs.items():
            word_factors.setdefault(word, pool_factors_for_word(pipe.registry, word))
            for p in entry.pool_factors or word_factors[word]:
                if p >= MIN_POOL_PRIME:
                    factor_to_docs[p].add(did)
                    factor_to_types[p].add(t)

    cross_factor = sum(1 for ts in factor_to_types.values() if len(ts) >= 2)
    print(f"\n[2] POOL-PRIME BRIDGES (ICN / L3 meet factors spanning ≥2 types)")
    print(f"  factors with cross-type reach: {cross_factor} / {len(factor_to_types)}")
    top = sorted(
        ((p, factor_to_types[p], len(factor_to_docs[p])) for p in factor_to_types),
        key=lambda x: (-len(x[1]), -x[2]),
    )[:12]
    for p, ts, nd in top:
        if len(ts) >= 2:
            print(f"  prime {p:<6}  types={sorted(ts)}  docs={nd}")

    # κ bridges
    kappa_to_docs: dict[tuple, set[str]] = defaultdict(set)
    kappa_to_types: dict[tuple, set[str]] = defaultdict(set)
    for did in cidx.doc_ids:
        t = doc_type[did]
        for key in attractor.doc_keys.get(did, set()):
            kappa_to_docs[key].add(did)
            kappa_to_types[key].add(t)
    cross_kappa = sum(1 for ts in kappa_to_types.values() if len(ts) >= 2)
    print(f"\n[3] κ ATTRACTOR BRIDGES (same geometry bucket, different data types)")
    print(f"  κ buckets spanning ≥2 types: {cross_kappa} / {attractor.summary()['buckets']}")
    for key, ts in sorted(kappa_to_types.items(), key=lambda x: -len(x[1]))[:10]:
        if len(ts) >= 2:
            print(f"  κ={key}  types={sorted(ts)}  docs={len(kappa_to_docs[key])}")

    # Critical-line leg_sum bridges
    sum_to_types: dict[int, set[str]] = defaultdict(set)
    sum_to_words: dict[int, set[str]] = defaultdict(set)
    for did, sig in hub_sigs.items():
        t = doc_type[did]
        for word, entry in sig.hubs.items():
            s, _ = leg_sum_side(entry.coord)
            sum_to_types[s].add(t)
            sum_to_words[s].add(f"{t}:{word}")

    cross_sum = sum(1 for ts in sum_to_types.values() if len(ts) >= 2)
    print(f"\n[4] CRITICAL-LINE SUM BRIDGES (same leg_sum s on j, different types)")
    print(f"  leg_sum values spanning ≥2 types: {cross_sum}")
    for s, ts in sorted(sum_to_types.items(), key=lambda x: -len(x[1]))[:8]:
        if len(ts) >= 2:
            words = sorted(sum_to_words[s])[:6]
            print(f"  s={s:<4}  types={sorted(ts)}  witnesses={words}")

    # Morph bridges: cross-type word pairs with morph meet
    print("\n[5] MORPH MEETS (letter-prime GCD across types — no shared surface form)")
    morph_hits: list[tuple[str, str, str, str, float]] = []
    type_words = {t: sorted(type_vocab[t]) for t in all_types}
    pairs_checked = 0
    for i, ta in enumerate(all_types):
        for tb in all_types[i:]:
            if ta == tb and tb != "morphological_variants":
                continue
            wa = [w for w in type_words[ta] if len(w) >= 5][:15]
            wb = [w for w in type_words[tb] if len(w) >= 5][:15]
            for w1 in wa:
                for w2 in wb:
                    if w1 == w2:
                        continue
                    pairs_checked += 1
                    sc = letter_composite_gcd_ratio(w1, w2)
                    if sc >= 0.25:
                        morph_hits.append((ta, tb, w1, w2, sc))
    morph_hits.sort(key=lambda x: -x[4])
    print(f"  pairs sampled: {pairs_checked}  hits≥0.25: {len(morph_hits)}")
    for ta, tb, w1, w2, sc in morph_hits[:10]:
        print(f"  {ta:<18} ↔ {tb:<18}  {w1} ~ {w2}  score={sc:.2f}")

    # Meet index cross-type (from eval meet_index dict)
    meet_types: dict[int, set[str]] = defaultdict(set)
    for p, docs in meet_index.items():
        for d in docs:
            meet_types[p].add(doc_type.get(d, "?"))
    cross_meet = sum(1 for ts in meet_types.values() if len(ts) >= 2)
    print(f"\n[6] MEET INDEX BRIDGES (BIT 7 factor postings)")
    print(f"  meet factors spanning ≥2 types: {cross_meet} / {len(meet_index)}")

    # Bridge recipe
    print("\n" + "=" * 88)
    print("HOW BRIDGES WORK ACROSS DATA TYPES")
    print("=" * 88)
    print("""
  Layer 0 — CRITICAL LINE j
    Same leg_sum s pins different types to one meet on j (halver).
    Repetitive "status" and scientific "cytokine" rarely share s — different rails.

  Layer 1 — POOL PRIMES / ICN
    Promoted L3 primes are the main CROSS-TYPE bridge when vocab overlaps
    (scientific ↔ technical ↔ morph ↔ phrases). Random/JSON barely promote.

  Layer 2 — κ BUCKETS
    Geometry collision links docs with similar (Re,Im,ζ) even if words differ.
    Works when chains promote; fails on random (1 bucket).

  Layer 3 — MORPH MEET
    Letter-prime GCD links surface variants across types WITHOUT same token
    (morph_variants ↔ scientific_prose: autophagy/autophagic).

  Layer 4 — MEET INDEX
    Factor ∪ routes zero-lexical queries across types sharing pool primes.

  Layer 5 — DEEPER RECURSION
    Longer phrase ICNs localize critical line per depth — bridge narrows but
    correlation strengthens within semantic neighborhood.
""")
    print("=" * 88)

    payload = {
        "n_docs": len(cidx.doc_ids),
        "cross_pool_factors": cross_factor,
        "cross_kappa": cross_kappa,
        "cross_leg_sum": cross_sum,
        "cross_meet_factors": cross_meet,
        "morph_hits": len(morph_hits),
        "shared_vocab": {
            f"{a}|{b}": len(type_vocab[a] & type_vocab[b])
            for i, a in enumerate(all_types)
            for b in all_types[i + 1 :]
        },
    }
    out = ROOT / "logs" / "correlation_bridge_types.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"JSON: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
