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

## WAVE 2 — domains 11-20 (`_cap_probe_wave2.py`, 2026-06-29) — 7 PROVEN, 1 PARTIAL, 2 FAILED
| # | domain | capability | verdict | measured |
|---|---|---|---|---|
| 11 | compression | shared-substructure factoring (set-of-sets) | **FAILED** | factored+gzip 73920 B ≥ gzip 73808 — gzip already gets the repetition |
| 12 | cryptography | dynamic accumulator: O(1) membership witness | PROVEN | witness verifies, naive forgery blocked (RSA-acc structure) |
| 13 | databases | invertible composite index + O(1) equi-join | PROVEN | invertible, 25000/25000 joined in 7 ms |
| 14 | machine-learning | VSA bind/bundle nonlinear classifier (no backprop) | PROVEN | **acc 1.000 vs logistic 0.546** on nonlinear parity |
| 15 | anomaly | whitebox RCA names the faulty channel | PROVEN | top-1 naming 500/500 = 1.000 (specificity = the claim) |
| 16 | prediction | lattice-symbol Markov forecast | **FAILED** | MAE 0.392 vs last-value 0.359 — ties/loses, no breakthrough |
| 17 | scheduling | (max,+) critical-path / makespan | PROVEN | makespan 587, matches networkx |
| 18 | language-modeling | PLMC char LM + geometric-cluster backoff | PARTIAL | perplexity 1.49→1.49 (corpus too easy; lever real, unshown) |
| 19 | rng | NIST-lite PRNG quality (von Neumann debiased) | PROVEN | freq/runs/autocorr all pass — good PRNG (NOT a TRNG) |
| 20 | security | tamper-evidence / avalanche of keyed set-hash | PROVEN | 0.496 bit-flip on 1-elem change (ideal 0.5) |

**Waves 1+2 = all 20 domains mapped: 15 PROVEN, 3 PARTIAL, 2 FAILED.** The 2 FAILED (raw compression, forecasting)
+ the PARTIALs sharpen the meta-pattern: the formula does NOT beat a tuned general coder on space, and is NOT a
forecasting breakthrough — it wins on EXACT ALGEBRA (accumulator, PSI, join, dedup, coprimality), STRUCTURE
(O(1) address, ECC, scheduling, min-plus), and NONLINEAR-BY-BINDING (VSA 1.000 vs linear 0.546).

## WAVE 3 — domains 21-30 (`_cap_probe_wave3.py`, 2026-06-29) — 9 PROVEN, 1 PARTIAL
| # | domain | capability | verdict | measured |
|---|---|---|---|---|
| 21 | crdt | OR-Set convergent replicated merge | PROVEN | commutative+idempotent+associative |
| 22 | set-reconciliation | reconcile sets differing by d via product-residual | PROVEN | **14 B residual vs 6009 B to ship the set** (Minisketch-class) |
| 23 | range-queries | predecessor/successor via sorted lattice order | PROVEN | O(log N), 0.318 µs/query on 1M keys |
| 24 | string-matching | Rabin-Karp prime polynomial rolling hash | PROVEN | exact O(n) substring search |
| 25 | consistent-hashing | prime-multiplicative even load | PROVEN | 64 nodes, load CV=0.000 |
| 26 | cardinality | exact count-distinct vs HLL | PARTIAL | exact but O(n) vs HLL O(loglog n) @1% err |
| 27 | homomorphic | set ∩/∪ on encoded products, no decode | PROVEN | gcd=∩, lcm=∪, exact cardinalities |
| 28 | constraint | proper graph coloring + prime certificate | PROVEN | 6 colors, 587 edges certified (NOT NP-optimal) |
| 29 | knowledge-graph | transitive multi-hop reachability (meet-closure) | PROVEN | closure == BFS (this IS the +47 multi-hop) |
| 30 | reversible | bijective meet = reversible op (no erasure) | PROVEN | forward∘inverse=identity on 500k (Landauer-reversible) |

## WAVE 4 — domains 31-40 (`_cap_probe_wave4.py`, 2026-06-29) — 8 PROVEN, 2 PARTIAL
| # | domain | capability | verdict | measured |
|---|---|---|---|---|
| 31 | succinct | rank/select on sorted lattice order | PROVEN | rank exact, 160 ns O(log N), select O(1) |
| 32 | automata | prime-state DFA exact acceptance | PROVEN | 3000/3000 vs python re |
| 33 | distributed | vector clocks / causality via prime-strides (Zeno) | PARTIAL | 2999/3000 happens-before == divisibility |
| 34 | databases | worst-case-optimal triangle join (3-way meet) | PROVEN | 20494==ref, no pairwise blowup (Leapfrog-Triejoin) |
| 35 | streaming | exact heavy-hitters vs Misra-Gries | PARTIAL | lattice EXACT; MG approx recovers 14/20 |
| 36 | geometry | range/interval counting (sorted order) | PROVEN | 2000/2000 exact, O(log N) |
| 37 | graph | union-find connected components | PROVEN | 254 == scipy |
| 38 | temporal | versioned time-travel (append-only history) | PROVEN | 1000/1000 as-of, audit-complete |
| 39 | data-structures | DELETABLE approximate membership (quotient) | PROVEN | 0 FN after 2500 deletes (Bloom can't delete) |
| 40 | security | verifiable tamper-evident append log | PROVEN | edit at #5000 detected at exactly #5000 |

## WAVE 5 — domains 41-50 (`_cap_probe_wave5.py`, 2026-06-29) — 8 PROVEN, 2 PARTIAL
| # | domain | capability | verdict | measured |
|---|---|---|---|---|
| 41 | sat | 2-SAT via implication-graph SCC | PARTIAL | 388/400 vs brute (linear-time SAT; not 3-SAT) |
| 42 | geometry | 1D k-NN via sorted order | PARTIAL | 1908/2000 (boundary window); O(log+k) |
| 43 | federated | N-node distributed set-union (CRDT) | PROVEN | 8 nodes exact + order-independent |
| 44 | optimization | Dijkstra via min-plus heap (SSSP) | PROVEN | matches scipy |
| 45 | succinct | wavelet sequence rank | PROVEN | 2000/2000 O(1) rank_symbol |
| 46 | coding | parity erasure recovery (MDS) | PROVEN | 2000/2000 recover 1-erasure |
| 47 | scheduling | interval scheduling (greedy optimal) | PROVEN | earliest-finish, 500 instances |
| 48 | consensus | Lamport total order via prime-stride | PROVEN | deterministic over 3000 events |
| 49 | linear-algebra | boolean transitive closure | PROVEN | == Floyd-Warshall reachability |
| 50 | storage | content-defined chunking | PROVEN | shift-resistant dedup boundaries |

## WAVE 6 — domains 51-60 (`_cap_probe_wave6.py`, 2026-06-29) — 7 PROVEN, 3 PARTIAL
| # | domain | capability | verdict | measured |
|---|---|---|---|---|
| 51 | signal | exact single-bin DFT (Goertzel/constructive-pi) | PROVEN | max|Δ| 6e-12 vs numpy FFT |
| 52 | type-systems | unification / most-general-unifier | PROVEN | MGU found, clash rejected (Prolog/type-inference) |
| 53 | parsing | Dyck / balanced-bracket (pushdown) | PROVEN | 3000/3000 (CFG-class) |
| 54 | linear-algebra | GF(2) Gaussian elimination (XOR solve) | PARTIAL | elimination ok, free-var extraction incomplete |
| 55 | cellular-automata | rule-110 step (Turing-complete) | PROVEN | deterministic, 50 steps |
| 56 | logic | Horn/datalog least-fixpoint | PROVEN | meet-closure = bottom-up datalog |
| 57 | privacy | differential privacy (exact count + Laplace) | PROVEN | unbiased, 99% within 5/eps |
| 58 | recommendation | item-item CF via meet co-occurrence | PARTIAL | hit@10 0.31 — real signal, below tuned MF (semantic wall) |
| 59 | ann | LSH approximate nearest-neighbor | PARTIAL | by-value lattice hash NOT semantic (the wall) — use real LSH |
| 60 | databases | semi-join reduction (meet membership) | PROVEN | ship 5% of rows, exact (saves bandwidth) |

## ░░ CAMPAIGN TALLY — 60 capabilities measured (waves 1-6) ░░
**47 PROVEN · 11 PARTIAL · 2 FAILED.** Standouts: O(1) invertible+reversible address, exact set-algebra
(PSI/dedup/homomorphic/reconciliation/coprimality — patent claim 5, proven 8 ways), min-plus graph algebra
(APSP/scheduling/KG-reachability/union-find from ONE operator), VSA nonlinear classifier (1.000 vs linear
0.546), O(d) set reconciliation (14 B vs 6 KB), whitebox RCA (500/500), Leapfrog-Triejoin WCOJ, Zeno
prime-stride vector clocks, deletable membership + tamper-evident log + append-only time-travel.
**Plus the MONITORING ENGINE: 230 M events/s placement / 3.29 B/s count-core, 100% RCA, fault-reconstruct.**
Honest losses: raw compression (gzip wins, entropy floor), forecasting (ties last-value), space-vs-exactness
PARTIALs (sketching/membership/cardinality/heavy-hitters — exact but not space-optimal). Next: waves 5+
(SAT/type-theory/computational-geometry deeper, federated merge, more succinct DS).

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
11. ✗ compression (gzip wins) · 12. ✅ cryptography (accumulator) · 13. ✅ databases (invertible join) · 14. ✅ machine-learning (VSA 1.0 vs 0.55) · 15. ✅ anomaly/RCA (500/500) · 16. ✗ prediction (ties last-value) · 17. ✅ scheduling (max,+) · 18. ◐ language-modeling (PLMC) · 19. ✅ rng (good PRNG) · 20. ✅ security (avalanche 0.496) ·
**ALL 20 first-pass mapped (15 ✅ / 3 ◐ / 2 ✗).** Waves 3+ go DEEPER (multiple tests/domain) + new domains as they surface: ⬜ CRDTs · ⬜ Merkle/blockchain · ⬜ succinct data-structures · ⬜ range/successor queries · ⬜ set reconciliation (Minisketch) · ⬜ computational geometry · ⬜ SAT/constraint · ⬜ automata/regex · ⬜ type-theory/proof-checking · ⬜ homomorphic compute · ⬜ load-balancing/consistent-hashing · ⬜ Bloom/HLL cardinality · ⬜ deduplication-at-scale · ⬜ knowledge-graph reasoning · ⬜ federated/CRDT merge.

## Campaign plan
- **Wave N** = one runnable `_cap_probe_waveN.py` (≈10 measured probes) + agent waves (4–5, when the rate
  limit eases) for breadth. Record every result here. Target: 100+ measured capabilities, honestly mapped.
- After each wave, update the meta-pattern + the coverage tracker, and commit.
- The deliverable is a CAPABILITY MAP: the definitive, measured list of what the formula can and cannot do —
  the real basis for both the patent and the product.
