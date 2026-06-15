"""One-query latency benchmark for pipeline tuning."""
from __future__ import annotations

import time
from pathlib import Path

from eval_checkpoint import load_checkpoint
from eval_beir import _score_one_query, DEFAULT_KAPPA_CANDIDATE_CAP
from aethos_hub_signature import build_query_profile, rank_with_hub_signatures
from pipeline.bit_04_candidate_router import route_query_candidates
from pipeline.bit_09_query_cell_profile import build_query_cell_profile


def main() -> None:
    b = load_checkpoint(Path("brains/scifact_quality.eval.pkl"))
    pipe, cidx = b.pipe, b.cidx
    idx = b.attractor_index
    qid = b.qids[1]
    query = b.queries[qid]
    profile = build_query_profile(
        query, pipe.registry,
        neighbor_map=b.neighbor_map, doc_freq=cidx.doc_freq, n_docs=len(cidx.doc_ids),
    )

    for expand in (True, False):
        route = route_query_candidates(
            profile.words, pipe.registry, idx, cidx.inv, b.neighbor_map, cidx.doc_ids,
            expand_neighbors=expand, meet_index=b.meet_index,
        )
        print(f"expand={expand}  |C|={route.n_merged}  keys={route.n_query_keys}  tier={route.tier}")

    for cap in (0, 100, 200, 500):
        cell = build_query_cell_profile(
            pipe.registry, query,
            neighbor_map=b.neighbor_map, doc_freq=cidx.doc_freq, n_docs=len(cidx.doc_ids),
        )
        route = route_query_candidates(
            profile.words, pipe.registry, idx, cidx.inv, b.neighbor_map, cidx.doc_ids,
            expand_neighbors=False, meet_index=b.meet_index,
        )
        t0 = time.perf_counter()
        rank_with_hub_signatures(
            profile, route.doc_ids, b.hub_sigs, cidx.doc_ids,
            doc_tokens=cidx.doc_tokens, doc_tf=cidx.doc_tf, doc_len=cidx.doc_len,
            avg_dl=cidx.avg_dl, composite_index=b.comp_idx, sub_comp_idx=b.sub_comp_idx,
            registry=pipe.registry, phrase_idx=b.phrase_idx, anchor_idx=b.anchor_idx,
            attractor_index=idx, query_kappa_keys=cell.kappa_neighbor_q,
            kappa_candidate_cap=cap, top_k=100,
        )
        ms = (time.perf_counter() - t0) * 1000
        print(f"cap={cap:4d}  rank_ms={ms:7.1f}")

    t0 = time.perf_counter()
    _score_one_query(
        query, pipe=pipe, cidx=cidx, hub_sigs=b.hub_sigs,
        neighbor_map=b.neighbor_map, meet_index=b.meet_index,
        sub_comp_idx=b.sub_comp_idx, comp_idx=b.comp_idx,
        phrase_idx=b.phrase_idx, anchor_idx=b.anchor_idx,
        attractor_index=idx,
        enable_kappa_scoring=True,
    )
    print(f"full _score_one_query (default cap={DEFAULT_KAPPA_CANDIDATE_CAP}): {(time.perf_counter()-t0)*1000:.1f}ms")


if __name__ == "__main__":
    main()
