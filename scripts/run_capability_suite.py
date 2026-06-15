#!/usr/bin/env python3
"""
Run the full AETHOS capability suite (Tests 1-33) and report a pass/fail board.

Usage:
    python scripts/run_capability_suite.py          # fast tier (skip >30s)
    python scripts/run_capability_suite.py --all     # everything, full corpus

Each entry maps a test number to its script. A test "passes" if the script
exits 0 (every script self-asserts and exits 1 on any failed check). Results
mirror derivations/formula_capability_tests_results.md.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (number, script, args, heavy?) - heavy tests skipped unless --all
SUITE = [
    (1, "test_russell_impossibility.py", [], False),
    (2, "test_wing_reversibility.py", [], False),
    (3, "test_perfect_hash_fta.py", [], False),
    (4, "test_lattice_dependent_types.py", [], False),
    (5, "test_provenance.py", [], False),
    (6, "test_self_organizing_graph.py", [], False),
    (7, "test_hyperbolic_correspondence.py", [], False),
    (8, "test_distributed_id.py", [], False),
    (9, "test_compositional_crdt.py", [], False),
    (10, "test_compression_optimality.py", [], False),
    (11, "test_sunflower_meets.py", [], False),
    (12, "test_information_preservation.py", [], False),
    (13, "test_compression_shannon_boundary.py", [], False),
    (14, "test_few_rules_reconstruction.py", [], False),
    (15, "test_lattice_context_compressor.py", [], True),
    (16, "test_promotion_subword_codec.py", [], True),
    (17, "test_chamber_blend_codec.py", [], True),
    (18, "test_paq_chamber_mixer.py", [], True),
    (19, "test_chamber_mixer_v2.py", [], True),
    (20, "test_token_alphabet_at_scale.py", [], True),
    (21, "test_chamber_mixer_v3.py", [], True),
    (22, "test_chamber_mixer_v4_speed.py", [], True),
    (23, "test_chamber_mixer_v5_native.py", [], True),
    (24, "test_quadrant_lanes_v6.py", [], True),
    (25, "test_halting_boundary_supervision.py", [], False),
    (26, "test_gear_engine.py", [], True),
    (27, "test_streaming_halt_continue.py", [], True),
    (28, "test_halting_predictor.py", [], False),
    (29, "test_ground_zero_recycling.py", [], False),
    (30, "test_electron_qubit_lattice.py", [], False),
    (30, "test_electron_qubit_chambers.py", [], False),
    (31, "test_qubit_node_ghz.py", [], False),
    (32, "test_zeno_kernel.py", [], False),
    (33, "test_zeno_gated_recycling.py", [], False),
    (34, "test_aethos_game_engine.py", [], False),
    (35, "test_checkers_lattice.py", [], False),
    (36, "test_chess_lattice.py", [], True),
    (37, "test_latent_monitoring.py", [], False),
    (38, "test_adaptive_and_entangle_monitor.py", [], False),
    (39, "test_fused_channel_monitor.py", [], False),
    (40, "test_autonomic_loop.py", [], False),
    (41, "test_emergent_capabilities.py", [], False),
    (42, "test_homomorphic_compute.py", [], False),
    (43, "test_glassbox_lm.py", [], False),
    (44, "test_exact_membership.py", [], False),
    (45, "test_conflict_scheduling.py", [], False),
    (46, "test_proof_checker.py", [], False),
    (47, "test_reversible_computing.py", [], False),
    (48, "test_tunneling_doubleslit.py", [], False),
    (49, "test_analogical_reasoning.py", [], False),
    (50, "test_atom_spectrum.py", [], False),
    (51, "test_causal_inference.py", [], False),
    (52, "test_interpretable_classifier.py", [], False),
    (53, "test_rag_signals.py", [], False),
    (54, "test_continual_learning.py", [], False),
    (55, "test_counting_sets.py", [], False),
    (56, "test_multiview_tokens.py", [], False),
    (57, "test_append_only_index.py", [], False),
    (58, "test_deterministic_semantics.py", [], False),
]


def main():
    run_all = "--all" in sys.argv
    here = Path(__file__).resolve().parent
    env_note = "ALL (full corpus)" if run_all else "FAST tier (heavy tests skipped)"
    print(f"AETHOS capability suite - {env_note}")
    print("=" * 64)
    print(f"  {'#':>3} | {'test':<40} | {'time':>7} | result")
    print(f"  {'-'*3} | {'-'*40} | {'-'*7} | ------")

    passed = failed = skipped = 0
    failures = []
    for num, script, args, heavy in SUITE:
        path = here / script
        name = script.replace("test_", "").replace(".py", "")
        if not path.exists():
            print(f"  {num:>3} | {name:<40} | {'--':>7} | MISSING")
            failed += 1
            failures.append((num, name, "missing file"))
            continue
        if heavy and not run_all:
            print(f"  {num:>3} | {name:<40} | {'--':>7} | skip (heavy)")
            skipped += 1
            continue
        t0 = time.time()
        try:
            r = subprocess.run([sys.executable, str(path), *args],
                               capture_output=True, cwd=str(ROOT),
                               timeout=600)
            dt = time.time() - t0
            ok = r.returncode == 0
            tag = "PASS" if ok else "FAIL"
            print(f"  {num:>3} | {name:<40} | {dt:>6.1f}s | {tag}")
            if ok:
                passed += 1
            else:
                failed += 1
                tail = r.stdout.decode("utf-8", "replace").strip().splitlines()
                failures.append((num, name, tail[-1] if tail else "exit 1"))
        except subprocess.TimeoutExpired:
            print(f"  {num:>3} | {name:<40} | {'>600':>6}s | TIMEOUT")
            failed += 1
            failures.append((num, name, "timeout"))

    print("=" * 64)
    print(f"  passed {passed}   failed {failed}   skipped {skipped}")
    if failures:
        print("\n  failures:")
        for num, name, why in failures:
            print(f"    #{num} {name}: {why}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
