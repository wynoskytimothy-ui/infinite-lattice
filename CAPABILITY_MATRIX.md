# AETHOS capability matrix — the systematic CS sweep

*The campaign Timothy asked for: every aspect of computer science, the prime-lattice/meet formula applied
20+ ways, hundreds of measured tests, honest verdicts, documented as we go. Started 2026-06-29. Each row is
BUILT + MEASURED, not asserted. Verdicts: **PROVEN** (captured, reproducible) · **PARTIAL** (works but
exact-not-space-optimal, or corpus-specific) · **FAILED** (measured negative) · **TODO** (planned).*

## Meta-pattern (the honest through-line, updated each wave)
The formula is an **exact addressing + algebra engine**: it wins wherever the job is EXACTNESS,
INVERTIBILITY, MERGEABILITY, COORDINATION-FREEDOM, or ORDER-INDEPENDENCE — and it ties/loses wherever the
job is SPACE COMPRESSION (multiplicative encodings grow as Σ log p) or SEMANTIC MEANING (geometry is
by-value, ~7 measured nulls). Map every capability against that line.

## Patent claims (the spec — `trng/PROVISIONAL_PATENT.md`, 5 independent + dependent)
1. System (prime-coordinate retrieval) · 2. Multi-locality rank-N tuple channel · 3. Partial-match scoring ·
4. Contrastive-lift tuple-weight training · 5. **Exact set-algebra operations** (the load-bearing novelty) ·
+ dependent 6–14 (prime assignment, hierarchical address P_c×P_d×P_w, 4-byte fingerprint, 32-corridor Ω, …)
· §10 whitebox multi-domain RCA subsystem (independently claimable). The campaign tests these + hunts more.

## WAVE 1 — 10 domains (`_cap_probe_wave1.py`, 2026-06-29) — 8 PROVEN, 2 PARTIAL
| # | domain | capability | verdict | measured |
|---|---|---|---|---|
| 1 | indexing | O(1) invertible content-address (meet det=−1) | PROVEN | 0 coll/200k, invert exact, 227 ns/addr |
| 2 | provenance | commutative O(1)-update set hash (multiply-to-merge) | PROVEN | order-indep + O(1) insert + mergeable |
| 3 | sketching | exact multiset frequency (prime exponents) | PARTIAL | exact but 6.4× larger than a counter array |
| 4 | privacy | private set intersection via GCD of prime-products | PROVEN | exact ∩ recovery (keyed) |
| 5 | distributed | coordination-free collision-proof IDs | PROVEN | 16×100k ids, 0 coll, 0 coord msgs |
| 6 | graph | min-plus closure = exact APSP (Floyd-Warshall) | PROVEN | max|Δ|=0 vs scipy |
| 7 | data-structures | 0-false-positive membership (divisibility) | PARTIAL | 0 FP/FN but 1.9× larger than Bloom |
| 8 | coding | 32-orbit repetition code (conserved Σ) | PROVEN | 2000/2000 recovered, 15/32 corrupted |
| 9 | storage | content-addressed exact dedup (product fingerprint) | PROVEN | 300/300 dup-groups, 0 FN |
| 10 | number-theory | independence via coprimality (gcd=1) | PROVEN | 3000/3000 gcd=1 ⇔ disjoint |

## Carried-forward PROVEN (from earlier this session — measured, reproducible)
| domain | capability | measured | file |
|---|---|---|---|
| retrieval | beats BM25 CPU-only, no CE | scifact nDCG 0.7023 > 0.665 | `_prove_retrieval_cpu.py` |
| serve | composite-meet pooling, full 8.8M | 123 ms, MRR 0.3986, beats WAND 7× | `_o1_serve_shootout.py` |
| footprint | 3-bit per-term weights | 168 B/doc, ΔMRR −0.0014 | `_o1_bitplane_weight_quant.py` |
| compression | V27 lossless postings | 3 B/posting, NDCG unchanged | `trng/bench_v27_compressed.py` |
| multi-hop | meet = free second hop | +47 bridge docs (0/60→45/60) | `_prove_multihop.py` |
| dedup | embedding-free near-dup | F1 99.58% @ Jaccard≥0.8 | `_prove_compliance_dedup.py` |
| completeness | exact-match candidate recall | 100.0000%, 0 false-neg | `_prove_compliance_dedup.py` |
| allocation | φ low-discrepancy distributor | 1037× less clustering than integer | `_poc_phi_distributor.py` |
| graph | (max,+) critical path + Viterbi | exact vs networkx | `_o1_minplus_free_graph.py` |
| determinism | bit-identical across processes | sha256-stable, 0/7 GPU libs | `_prove_structural.py` |

## Measured NEGATIVES / walls (don't re-pitch — honest)
| capability | result | file |
|---|---|---|
| free relevance from geometry (π-drift) | +0.0000 (geometry=addressing not meaning) | `_poc_pi_drift.py` |
| unsupervised clustering for accuracy | ~7 tests neutral; needs supervision | `_o1_recursive_lattice_revive.py` |
| corpus-as-a-number compression | not compression (Σlog p > Σlog gap) | `_o1_algebraic_number_coldstore.py` |
| chamber routing | no recall-safe candidate reduction | `_o1_chamber_routing.py` |
| φ / lattice coords as semantic embedding | NDCG 0.001–0.0000 | `_vec_lattice_coord_features.py` |

## The 20 domains — coverage tracker (campaign target: hundreds of tests)
1. ✅ indexing/search · 2. ✅ provenance/audit · 3. ◐ sketching/streaming · 4. ✅ privacy/PSI · 5. ✅ distributed/coordination-free · 6. ✅ graph algorithms · 7. ◐ data-structures/membership · 8. ✅ coding/ECC · 9. ✅ storage/dedup · 10. ✅ number-theory ·
11. ⬜ compression (lossless/grammar) · 12. ⬜ cryptography (commitments/ZK/homomorphic) · 13. ⬜ databases (joins/indexes) · 14. ⬜ machine-learning (VSA net/classification/clustering) · 15. ⬜ anomaly/monitoring (RCA residuals) · 16. ⬜ prediction/forecasting (RUL/time-series) · 17. ⬜ scheduling/optimization (assignment/bin-packing) · 18. ⬜ language-modeling (PLMC generative) · 19. ⬜ random-number-generation (TRNG/NIST) · 20. ⬜ security (PSI/secure-dedup/tamper-evidence) ·
*(+ more as they surface: error-correcting codes beyond repetition, succinct data structures, range/successor queries, set reconciliation/Minisketch, CRDTs, Merkle/blockchain, computational geometry, SAT/constraint, automata, type theory.)*

## Campaign plan
- **Wave N** = one runnable `_cap_probe_waveN.py` (≈10 measured probes) + agent waves (4–5, when the rate
  limit eases) for breadth. Record every result here. Target: 100+ measured capabilities, honestly mapped.
- After each wave, update the meta-pattern + the coverage tracker, and commit.
- The deliverable is a CAPABILITY MAP: the definitive, measured list of what the formula can and cannot do —
  the real basis for both the patent and the product.
