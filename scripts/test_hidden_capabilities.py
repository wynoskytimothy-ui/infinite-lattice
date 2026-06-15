#!/usr/bin/env python3
"""
Exercise hidden / underwired capabilities from the audit.

Runs pipeline gates, module smoke tests, and offline benchmarks.
Writes logs/hidden_capabilities_report.json and prints a summary table.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LOG_PATH = ROOT / "logs" / "hidden_capabilities_report.json"


@dataclass
class CapResult:
    name: str
    tier: str
    status: str  # PASS | FAIL | SKIP | PARTIAL
    detail: str = ""
    ms: float = 0.0


@dataclass
class Report:
    started: str
    elapsed_s: float = 0.0
    passed: int = 0
    failed: int = 0
    partial: int = 0
    skipped: int = 0
    results: list[CapResult] = field(default_factory=list)


def _run(name: str, tier: str, fn) -> CapResult:
    t0 = time.perf_counter()
    try:
        detail = fn()
        ms = (time.perf_counter() - t0) * 1000.0
        status = "PARTIAL" if detail.startswith("PARTIAL:") else "PASS"
        return CapResult(name=name, tier=tier, status=status, detail=detail, ms=ms)
    except Exception as e:
        ms = (time.perf_counter() - t0) * 1000.0
        return CapResult(
            name=name,
            tier=tier,
            status="FAIL",
            detail=f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}",
            ms=ms,
        )


def _small_corpus_setup():
    from aethos_token_processor import TokenProcessor
    from aethos_tokenize import tokenize_words
    from diagnose_corpus import SMALL_CORPUS
    from aethos_hub_signature import build_all_hub_signatures

    pipe = TokenProcessor()
    pipe.ingest(*SMALL_CORPUS)
    doc_tokens = {
        f"d{i}": frozenset(tokenize_words(text))
        for i, text in enumerate(SMALL_CORPUS)
    }
    doc_ids = list(doc_tokens.keys())
    sigs = build_all_hub_signatures(doc_ids, doc_tokens, pipe.registry, top_k=12)
    return pipe, doc_tokens, doc_ids, sigs


# ---------------------------------------------------------------------------
# Pipeline gates
# ---------------------------------------------------------------------------

def test_bit01_gate() -> str:
    from pipeline.bit_01_word_cell import verify_bit01_gate

    pipe, _, _, _ = _small_corpus_setup()
    passed, total, failures = verify_bit01_gate(pipe.registry, max_words=100)
    if failures:
        raise AssertionError(failures)
    return f"{passed}/{total} checks"


def test_bit02_gate() -> str:
    from pipeline.bit_02_attractor_key import verify_bit02_gate, DEFAULT_QUANTIZE

    ok, failures = verify_bit02_gate(quantize=DEFAULT_QUANTIZE)
    if not ok:
        raise AssertionError(failures)
    return "κ quantize gate ok"


def test_bit03_gate() -> str:
    from pipeline.bit_03_doc_attractor_set import verify_bit03_gate

    pipe, _, doc_ids, sigs = _small_corpus_setup()
    passed, total, failures = verify_bit03_gate(pipe.registry, sigs)
    if failures:
        raise AssertionError(failures)
    return f"{passed}/{total} sampled; {len(doc_ids)} docs"


def test_bit04_gate() -> str:
    from pipeline.bit_04_candidate_router import verify_bit04_gate_legacy_tuple
    from pipeline.bit_03_doc_attractor_set import build_attractor_index_from_hub_signatures
    from pipeline.bit_07_meet_witness import build_meet_witness_index
    from eval_beir import build_neighbor_weights

    pipe, doc_tokens, doc_ids, sigs = _small_corpus_setup()
    att = build_attractor_index_from_hub_signatures(pipe.registry, sigs)
    meet = build_meet_witness_index(sigs, pipe.registry)
    nw = build_neighbor_weights(pipe.registry)
    inv: dict[str, set[str]] = {}
    for did, toks in doc_tokens.items():
        for w in toks:
            inv.setdefault(w, set()).add(did)
    queries = {
        "q1": "phone technical software",
        "q2": "apple fruit pie",
    }
    qrels = {
        "q1": {"d0": 1, "d2": 1},
        "q2": {"d3": 1, "d4": 1},
    }
    ok, avg, failures = verify_bit04_gate_legacy_tuple(
        pipe.registry,
        queries,
        qrels,
        doc_ids,
        doc_tokens,
        sigs,
        inv,
        nw,
        index=att,
        meet_index=meet,
        min_candidates=1,
        target=0.5,
    )
    if not ok:
        raise AssertionError(failures)
    return f"router recall avg={avg:.3f}"


def test_bit05_gate() -> str:
    from pipeline.bit_05_z_band import verify_bit05_gate

    ok, failures = verify_bit05_gate(chain=(3, 5, 7), n=5)
    if not ok:
        raise AssertionError(failures)
    return "band profiles on chain (3,5,7)"


def test_bit06_gate() -> str:
    from pipeline.bit_06_notch_bind import verify_bit06_gate, score_bound_notch_pair

    pipe, _, _, sigs = _small_corpus_setup()
    ok, failures = verify_bit06_gate(pipe.registry, sigs)
    if not ok:
        raise AssertionError(failures)
    from pipeline.bit_06_notch_bind import build_all_notch_fingerprints

    fps = build_all_notch_fingerprints(sigs, pipe.registry)
    if fps:
        did = next(iter(fps))
        s = score_bound_notch_pair(["apple"], fps[did], pipe.registry)
        return f"gate ok; score_bound_notch_pair smoke={s:.4f}"
    return "gate ok; no fingerprints built"


def test_bit07_gate() -> str:
    from pipeline.bit_07_meet_witness import (
        verify_bit07_gate,
        verify_bit07_routing_gate,
    )

    pipe, doc_tokens, _, sigs = _small_corpus_setup()
    ok, failures = verify_bit07_gate(pipe.registry, sigs)
    if not ok:
        raise AssertionError(failures)
    ok2, avg, failures2 = verify_bit07_routing_gate(
        pipe.registry, sigs, doc_tokens,
    )
    if not ok2:
        raise AssertionError(failures2)
    return f"meet index + routing avg_recall={avg:.3f}"


def test_bit08_entangled_pairs() -> str:
    from aethos_intersection_nodes import IntersectionNetwork, find_entangled_meet_pairs

    net = IntersectionNetwork()
    pairs = find_entangled_meet_pairs(net, require_depth_match=True)
    return f"PARTIAL: BIT 8 module missing; find_entangled_meet_pairs returned {len(pairs)} pairs on empty net"


def test_bit09_gate() -> str:
    from pipeline.bit_09_query_cell_profile import verify_bit09_gate
    from pipeline.bit_03_doc_attractor_set import build_attractor_index_from_hub_signatures
    from eval_beir import build_neighbor_weights

    pipe, doc_tokens, doc_ids, sigs = _small_corpus_setup()
    att = build_attractor_index_from_hub_signatures(pipe.registry, sigs)
    doc_freq: dict[str, int] = {}
    for toks in doc_tokens.values():
        for w in toks:
            doc_freq[w] = doc_freq.get(w, 0) + 1
    ok, failures = verify_bit09_gate(
        pipe.registry,
        neighbor_map=build_neighbor_weights(pipe.registry),
        doc_freq=doc_freq,
        n_docs=len(doc_ids),
        index=att,
    )
    if not ok:
        raise AssertionError(failures)
    return "query cell profile gate ok"


def test_bit10_gate() -> str:
    from pipeline.bit_10_score_fusion import verify_bit10_gate, signal_8a_kappa_jaccard
    from pipeline.bit_03_doc_attractor_set import build_attractor_index_from_hub_signatures
    from pipeline.bit_09_query_cell_profile import build_query_cell_profile
    from aethos_hub_signature import build_query_profile
    from eval_beir import build_neighbor_weights

    pipe, doc_tokens, doc_ids, sigs = _small_corpus_setup()
    att = build_attractor_index_from_hub_signatures(pipe.registry, sigs)
    nw = build_neighbor_weights(pipe.registry)
    n_docs = len(doc_ids)
    doc_freq: dict[str, int] = {}
    for toks in doc_tokens.values():
        for w in toks:
            doc_freq[w] = doc_freq.get(w, 0) + 1

    query = "phone technical software"
    profile = build_query_profile(
        query, pipe.registry,
        neighbor_map=nw, doc_freq=doc_freq, n_docs=n_docs,
    )
    cell = build_query_cell_profile(
        pipe.registry, query,
        neighbor_map=nw, doc_freq=doc_freq, n_docs=n_docs,
    )

    ok, failures = verify_bit10_gate(
        pipe.registry, profile, cell, att, sigs, doc_ids, doc_ids,
        doc_tokens=doc_tokens,
    )
    if not ok:
        raise AssertionError(failures)
    s8 = signal_8a_kappa_jaccard(
        profile, "d0", att, cell.kappa_neighbor_q, lambda_kappa=0.25,
    )
    return f"8a gate ok; smoke 8a d0 λ=0.25 → {s8:.4f}"


def test_bit11_compression_ledger() -> str:
    from scripts.compression_seven_types import main as compression_main  # noqa: F401
    import scripts.compression_seven_types as c7

    pipe, _, doc_ids, sigs = _small_corpus_setup()
    rows = []
    for did in doc_ids[:3]:
        sig = sigs[did]
        pin_b = c7.critical_pin_bytes_for_sig(sig)
        hub_b = sig.encoded_size()
        rows.append(f"{did}: hubs={hub_b}B pins={pin_b}B")
    return f"PARTIAL: no bit_11 module; smoke: {'; '.join(rows)}"


# ---------------------------------------------------------------------------
# Hidden modules
# ---------------------------------------------------------------------------

def test_critical_line_rotation() -> str:
    from aethos_spring_complex import verify_critical_line_rotation

    checks = verify_critical_line_rotation()
    bad = [k for k, v in checks.items() if not v]
    if bad:
        raise AssertionError(f"failed: {bad}")
    return f"{len(checks)} rotation axioms proven"


def test_pi_bridge() -> str:
    from aethos_pi_bridge import bridge_report, pi_k0_is_spring_i, pi_layer0_direction_matches

    report = bridge_report()
    k0 = pi_k0_is_spring_i()
    l0 = pi_layer0_direction_matches()
    return f"PARTIAL: k0=i {k0}; layer0 π/4 {l0}; report {len(report)} chars"


def test_hilbert_inner_product() -> str:
    from aethos_hilbert import inner_product, wing_subspace_states
    from aethos_hilbert_lattice import build_robust_space_from_corpus
    from diagnose_corpus import SMALL_CORPUS

    space = build_robust_space_from_corpus(*SMALL_CORPUS)
    states = wing_subspace_states(chain=(3, 5), n=5)
    ip = inner_product(states[0], states[0])
    n_basis = len(space.basis_labels())
    return f"basis={n_basis}; self-IP={ip:.4f}"


def test_discriminative_score() -> str:
    from aethos_discriminative import discriminative_score, build_heavy_anchor_index
    from aethos_hub_signature import lattice_composite_for_word

    pipe, doc_tokens, doc_ids, _ = _small_corpus_setup()
    doc_freq: dict[str, int] = {}
    for toks in doc_tokens.values():
        for w in toks:
            doc_freq[w] = doc_freq.get(w, 0) + 1
    idx = build_heavy_anchor_index(pipe.registry, doc_tokens, doc_freq)
    w = "apple"
    comp = lattice_composite_for_word(pipe.registry, w)
    s = discriminative_score(comp, idx, len(doc_ids))
    return f"orphan fn smoke: discriminative_score(apple)={s:.4f}"


def test_meta_bridges_pass3() -> str:
    from aethos_iterative import build_multi_pass
    from aethos_pipeline import AethosPipeline
    from diagnose_corpus import SMALL_CORPUS

    pipe = AethosPipeline()
    pipe.ingest(*SMALL_CORPUS)
    doc_tokens = {
        f"d{i}": frozenset(__import__("aethos_tokenize", fromlist=["tokenize_words"]).tokenize_words(text))
        for i, text in enumerate(SMALL_CORPUS)
    }
    mp = build_multi_pass(
        pipe, list(SMALL_CORPUS), doc_tokens,
        n_passes=3, verbose=False,
    )
    n_meta = len(mp.meta_bridges)
    n_bridge = len(mp.bridges)
    n_phrase = mp.phrase_idx.n_composites if mp.phrase_idx else 0
    return (
        f"PARTIAL: built but not scored — "
        f"L4={n_phrase} L5={n_bridge} L6={n_meta}"
    )


def test_bad_correlation_store() -> str:
    import tempfile
    from core.learning_engine import BadCorrelationStore, record_retrieval_false_positives
    from aethos_hub_signature import build_query_profile
    from eval_beir import build_neighbor_weights

    pipe, doc_tokens, doc_ids, sigs = _small_corpus_setup()
    store = BadCorrelationStore()
    nw = build_neighbor_weights(pipe.registry)
    doc_freq = {}
    for toks in doc_tokens.values():
        for w in toks:
            doc_freq[w] = doc_freq.get(w, 0) + 1
    profile = build_query_profile(
        "apple phone", pipe.registry,
        neighbor_map=nw, doc_freq=doc_freq, n_docs=len(doc_ids),
    )
    ranked = doc_ids[:3]
    rel = {"d2"}
    record_retrieval_false_positives(
        store, ranked, rel, profile, sigs, pipe.registry, top_k=3,
    )
    n = len(store.entries)
    with tempfile.TemporaryDirectory() as td:
        p = store.save(Path(td) / "bad.json")
        loaded = BadCorrelationStore.load(p)
    return (
        f"PARTIAL: write path ok ({n} entries); "
        f"read-during-rank NOT wired; reload={len(loaded.entries)}"
    )


def test_factor_analogy() -> str:
    from core.learning_engine import factor_analogy

    KING, MAN, WOMAN, QUEEN = 67367, 33701, 67339, 67373  # example composites
    # use small primes for smoke
    result = factor_analogy(30, 6, 10)  # 30 - 6 + 10 = 34
    return f"factor_analogy(30,6,10)={result} (expected 34)"


def test_phi_lattice_plateau() -> str:
    from aethos_sequences import canon_on_chain
    from aethos_recursive import segment_index
    from aethos_lattice import BranchKind

    chain = (3, 5, 7, 11)
    zs = []
    for n in range(3, 11):
        seg = segment_index(chain, n)
        if 0 < seg < len(chain):
            x, y, z = canon_on_chain(BranchKind.VA1, chain, n, lock_interior=True)
            zs.append(z)
    plateau = len(set(zs)) == 1 if zs else False
    return f"interior Z plateau constant={plateau} (n=3..10, Z={zs[0] if zs else '?'})"


def test_wing_collisions() -> str:
    from aethos_lattice import LatticeBank32

    bank = LatticeBank32.single_prime(5)
    groups = bank.find_same_n_collisions(n=7)
    return f"{len(groups)} collision groups at p=5 n=7"


def test_use_core_l2() -> str:
    from aethos_iterative import build_multi_pass
    from aethos_pipeline import AethosPipeline
    from aethos_tokenize import tokenize_words
    from diagnose_corpus import SMALL_CORPUS

    pipe = AethosPipeline()
    pipe.ingest(*SMALL_CORPUS)
    doc_tokens = {
        f"d{i}": frozenset(tokenize_words(text))
        for i, text in enumerate(SMALL_CORPUS)
    }
    mp = build_multi_pass(
        pipe, list(SMALL_CORPUS), doc_tokens,
        n_passes=1, use_core_l2=True, verbose=False,
    )
    d1 = mp.passes[0]
    return f"PARTIAL: use_core_l2=True pass1 L2={d1.total_l2} (not in eval path)"


def test_disabled_weights() -> str:
    from aethos_phrase_composite import PHRASE_WEIGHT
    from aethos_hub_signature import LAMBDA_KAPPA, LAMBDA_PRIME_FACTOR, CONSENSUS_WINGS
    from aethos_composite import MORPH_WEIGHT

    parts = [
        f"PHRASE_WEIGHT={PHRASE_WEIGHT}",
        f"LAMBDA_KAPPA={LAMBDA_KAPPA}",
        f"LAMBDA_PRIME_FACTOR={LAMBDA_PRIME_FACTOR}",
        f"MORPH_WEIGHT={MORPH_WEIGHT}",
        f"CONSENSUS_WINGS={len(CONSENSUS_WINGS)}/32",
    ]
    return "PARTIAL: " + "; ".join(parts)


def test_hub_band_fields_populated() -> str:
    _, _, _, sigs = _small_corpus_setup()
    with_band = sum(
        1 for s in sigs.values()
        for h in s.hubs.values()
        if h.band_id >= 0
    )
    with_z = sum(1 for s in sigs.values() for h in s.hubs.values() if h.z_obs != 0.0)
    total_hubs = sum(len(s.hubs) for s in sigs.values())
    return f"PARTIAL: {with_band}/{total_hubs} hubs have band_id; {with_z} have z_obs≠0 (stored, not scored)"


def test_codec_roundtrip() -> str:
    from aethos_codec import encode_text, decode_text

    payload = "apple phone technical"
    dot = encode_text(payload)
    back = decode_text(dot)
    if back != payload:
        raise AssertionError(f"roundtrip {payload!r} → {back!r}")
    return f"1 dot coord={dot.coord} roundtrip ok"


def test_overlay_semantic() -> str:
    from aethos_overlay import overlay_for_word
    from aethos_token_processor import TokenProcessor

    pipe = TokenProcessor()
    ov = overlay_for_word(pipe.registry, "phone")
    return f"overlay chain_len={len(ov.chain) if ov else 0}"


def test_find_entangled_meet_pairs() -> str:
    from aethos_intersection_nodes import IntersectionNetwork

    net = IntersectionNetwork()
    pairs = net.entangled_pairs()
    return f"{len(pairs)} entangled pairs on fresh network"


def test_pattern_placement() -> str:
    from pipeline.pattern_placement import classify_failure_pattern

    pat = classify_failure_pattern(
        ndcg10=1.0,
        gold_ids={"d0"},
        gold_in_candidates=True,
        gold_best_rank=1,
        gold_bm25_overlap=2,
        top1_is_gold=True,
        gold_in_corpus=True,
    )
    return f"classify perfect retrieval → {pat}"


def test_signal_8b_not_in_rank() -> str:
    import inspect
    from aethos_hub_signature import rank_with_hub_signatures

    src = inspect.getsource(rank_with_hub_signatures)
    has_8b = "notch" in src.lower() or "8b" in src
    return f"PARTIAL: rank_with_hub_signatures has notch ref={has_8b} (expected False until wired)"


# ---------------------------------------------------------------------------
# Offline scripts (subprocess)
# ---------------------------------------------------------------------------

def _run_script(rel: str, timeout: int = 120) -> str:
    proc = subprocess.run(
        [sys.executable, str(ROOT / rel)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"exit {proc.returncode}\nstdout:{proc.stdout[-800:]}\nstderr:{proc.stderr[-800:]}"
        )
    tail = (proc.stdout or "").strip().splitlines()
    return tail[-1] if tail else "ok"


def test_script_compression() -> str:
    return _run_script("scripts/compression_seven_types.py")


def test_script_speed() -> str:
    return _run_script("scripts/speed_multi_benchmark.py", timeout=180)


def test_script_correlation_bridge() -> str:
    return _run_script("scripts/correlation_bridge_types.py", timeout=180)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

TESTS: list[tuple[str, str, object]] = [
    # Pipeline gates
    ("BIT 1 word→cell gate", "pipeline", test_bit01_gate),
    ("BIT 2 κ attractor key gate", "pipeline", test_bit02_gate),
    ("BIT 3 doc attractor set gate", "pipeline", test_bit03_gate),
    ("BIT 4 candidate router gate", "pipeline", test_bit04_gate),
    ("BIT 5 z-band gate", "pipeline", test_bit05_gate),
    ("BIT 6 notch bind gate + 8b smoke", "pipeline", test_bit06_gate),
    ("BIT 7 meet witness gate", "pipeline", test_bit07_gate),
    ("BIT 8 path fiber (entangled pairs)", "pipeline", test_bit08_entangled_pairs),
    ("BIT 9 query cell profile gate", "pipeline", test_bit09_gate),
    ("BIT 10 score fusion / 8a gate", "pipeline", test_bit10_gate),
    ("BIT 11 compression ledger smoke", "pipeline", test_bit11_compression_ledger),
    # Geometry / addressing
    ("Critical line rotation proof", "geometry", test_critical_line_rotation),
    ("π bridge partial functor", "geometry", test_pi_bridge),
    ("Hilbert inner product", "geometry", test_hilbert_inner_product),
    ("φ-lattice Z plateau", "geometry", test_phi_lattice_plateau),
    ("Wing collision groups", "geometry", test_wing_collisions),
    # Hidden retrieval assets
    ("discriminative_score orphan fn", "retrieval", test_discriminative_score),
    ("L5/L6 meta-bridges Pass 3 build", "retrieval", test_meta_bridges_pass3),
    ("BadCorrelationStore write path", "retrieval", test_bad_correlation_store),
    ("factor_analogy composite math", "retrieval", test_factor_analogy),
    ("Disabled weight snapshot", "retrieval", test_disabled_weights),
    ("Hub band_id/z_obs populated", "retrieval", test_hub_band_fields_populated),
    ("Signal 8b not in rank loop", "retrieval", test_signal_8b_not_in_rank),
    ("use_core_l2 pass smoke", "retrieval", test_use_core_l2),
    ("pattern_placement classify", "retrieval", test_pattern_placement),
    # Codec / overlay
    ("Codec encode/decode roundtrip", "codec", test_codec_roundtrip),
    ("Semantic overlay for word", "codec", test_overlay_semantic),
    ("find_entangled_meet_pairs", "codec", test_find_entangled_meet_pairs),
    # Offline benchmarks
    ("Script: compression_seven_types", "benchmark", test_script_compression),
    ("Script: speed_multi_benchmark", "benchmark", test_script_speed),
    ("Script: correlation_bridge_types", "benchmark", test_script_correlation_bridge),
]


def main() -> int:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    report = Report(started=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()))

    print("=" * 72)
    print("HIDDEN CAPABILITIES TEST MATRIX")
    print("=" * 72)

    for name, tier, fn in TESTS:
        r = _run(name, tier, fn)
        report.results.append(r)
        icon = {"PASS": "OK", "FAIL": "FAIL", "PARTIAL": "PARTIAL", "SKIP": "SKIP"}[r.status]
        print(f"  [{icon:7}] {name} ({r.ms:.0f}ms)")
        if r.status == "FAIL":
            print(f"           {r.detail[:200]}")
        elif r.status == "PARTIAL":
            print(f"           {r.detail[:160]}")

    for r in report.results:
        if r.status == "PASS":
            report.passed += 1
        elif r.status == "FAIL":
            report.failed += 1
        elif r.status == "PARTIAL":
            report.partial += 1
        else:
            report.skipped += 1

    report.elapsed_s = time.perf_counter() - t0

    print()
    print("-" * 72)
    print(
        f"SUMMARY: {report.passed} PASS | {report.partial} PARTIAL | "
        f"{report.failed} FAIL | {report.skipped} SKIP | {report.elapsed_s:.1f}s"
    )

    out = {
        "summary": {
            "passed": report.passed,
            "partial": report.partial,
            "failed": report.failed,
            "skipped": report.skipped,
            "elapsed_s": round(report.elapsed_s, 2),
        },
        "results": [asdict(r) for r in report.results],
    }
    LOG_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Report: {LOG_PATH}")

    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
