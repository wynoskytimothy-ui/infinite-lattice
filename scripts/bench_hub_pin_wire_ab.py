#!/usr/bin/env python3
"""A/B: hub pin wire vs legacy — NDCG@10 and P50 must match."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRNG = Path(r"c:\Users\wynos\trng")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _resolve_beir_root() -> Path:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "beir_data_root", TRNG / "beir_data_root.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return Path(mod.resolve_beir_root())

from aethos_composite import build_composite_index
from aethos_discriminative import build_heavy_anchor_index
from aethos_hub_signature import build_all_hub_signatures
from aethos_iterative import build_multi_pass
from aethos_subword_composite import build_subword_composite_index
from eval_beir import (
    _score_one_query,
    build_meet_index,
    build_neighbor_weights,
    doc_text,
    ingest_corpus,
    load_corpus,
    load_paths,
    load_qrels,
    load_queries,
    make_pipeline,
    ndcg_at_k,
    recall_at_k,
)
from pipeline.bit_03_doc_attractor_set import build_attractor_index_from_hub_signatures


def run_arm(
    *,
    label: str,
    use_pin_wire: bool,
    pipe,
    cidx,
    queries,
    qids,
    qrels,
    neighbor_map,
    sub_comp_idx,
    phrase_idx,
    anchor_idx,
) -> dict:
    hub_sigs = build_all_hub_signatures(
        cidx.doc_ids,
        cidx.doc_tokens,
        pipe.registry,
        top_k=12,
        use_pin_wire=use_pin_wire,
        materialize_wings=use_pin_wire,
    )
    meet_index = build_meet_index(hub_sigs, pipe.registry)
    attractor_index = build_attractor_index_from_hub_signatures(
        pipe.registry, hub_sigs
    )
    comp_idx = build_composite_index(hub_sigs)
    bpd = sum(s.encoded_size() for s in hub_sigs.values()) / max(len(hub_sigs), 1)

    ndcgs: list[float] = []
    r10s: list[float] = []
    times: list[float] = []

    for qid in qids:
        t0 = time.perf_counter()
        result = _score_one_query(
            queries[qid],
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
            kappa_candidate_cap=350,
            enable_kappa_scoring=False,
        )
        times.append((time.perf_counter() - t0) * 1000)
        rel = qrels[qid]
        ndcgs.append(ndcg_at_k(result.ranked, rel, 10))
        r10s.append(recall_at_k(result.ranked, rel, 10))

    st = sorted(times)
    n = max(len(qids), 1)
    out = {
        "label": label,
        "ndcg10": sum(ndcgs) / n,
        "r10": sum(r10s) / n,
        "p50_ms": st[len(st) // 2] if st else 0.0,
        "mean_ms": sum(times) / n,
        "hub_bpd": bpd,
    }
    print(
        f"  {label}: NDCG@10={out['ndcg10']:.4f}  R@10={out['r10']:.4f}  "
        f"P50={out['p50_ms']:.2f}ms  mean={out['mean_ms']:.2f}ms  hub={bpd:.0f} B/doc"
    )
    return out


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="scifact")
    ap.add_argument("--max-queries", type=int, default=100)
    ap.add_argument("--mode", default="quality")
    args = ap.parse_args()

    paths = load_paths(_resolve_beir_root(), args.dataset)
    corpus = load_corpus(paths.corpus, max_docs=None)
    queries = load_queries(paths.queries)
    qrels = load_qrels(paths.qrels_test)
    qids = [q for q in qrels if q in queries][: args.max_queries]

    print(f"Dataset: {args.dataset}  docs={len(corpus)}  queries={len(qids)}")
    print("=" * 72)

    pipe = make_pipeline(args.mode)
    t0 = time.perf_counter()
    _metrics, cidx = ingest_corpus(pipe, corpus, mode=args.mode)
    try:
        pipe.flush()
    except Exception:
        pass
    corpus_texts = [doc_text(doc) for doc in corpus.values()]
    mp = build_multi_pass(
        pipe,
        corpus_texts,
        cidx.doc_tokens,
        n_passes=2,
        verbose=False,
        max_l2_promote=320 if args.mode == "quality" else 160,
    )
    phrase_idx = mp.phrase_idx
    neighbor_map = build_neighbor_weights(pipe.registry)
    sub_comp_idx = build_subword_composite_index(
        pipe.registry, cidx.doc_tokens, max_composites=500
    )
    anchor_idx = build_heavy_anchor_index(
        pipe.registry,
        cidx.doc_tokens,
        cidx.doc_freq,
        max_doc_count=5,
        rarity_threshold=0.018,
    )
    print(f"  Shared build: {(time.perf_counter() - t0):.1f}s")
    print()

    legacy = run_arm(
        label="LEGACY",
        use_pin_wire=False,
        pipe=pipe,
        cidx=cidx,
        queries=queries,
        qids=qids,
        qrels=qrels,
        neighbor_map=neighbor_map,
        sub_comp_idx=sub_comp_idx,
        phrase_idx=phrase_idx,
        anchor_idx=anchor_idx,
    )
    pin = run_arm(
        label="PIN_WIRE",
        use_pin_wire=True,
        pipe=pipe,
        cidx=cidx,
        queries=queries,
        qids=qids,
        qrels=qrels,
        neighbor_map=neighbor_map,
        sub_comp_idx=sub_comp_idx,
        phrase_idx=phrase_idx,
        anchor_idx=anchor_idx,
    )

    d_ndcg = pin["ndcg10"] - legacy["ndcg10"]
    d_r10 = pin["r10"] - legacy["r10"]
    d_p50 = pin["p50_ms"] - legacy["p50_ms"]
    print()
    print("=" * 72)
    print(f"  dNDCG@10 (pin - legacy) = {d_ndcg:+.4f}")
    print(f"  dR@10    (pin - legacy) = {d_r10:+.4f}")
    print(f"  dP50 ms  (pin - legacy) = {d_p50:+.2f}")
    print(f"  Hub wire B/doc saved      = {legacy['hub_bpd'] - pin['hub_bpd']:.0f}")

    ndcg_ok = abs(d_ndcg) <= 0.002
    r10_ok = abs(d_r10) <= 0.005
    p50_ok = abs(d_p50) <= 8.0  # quality-mode ~90ms P50; pin-keyed meet within noise
    if ndcg_ok and r10_ok and p50_ok:
        print("  GATE: PASS — accuracy and speed unchanged within tolerance")
        return 0
    print("  GATE: FAIL — review tolerances or fix pin regen path")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
