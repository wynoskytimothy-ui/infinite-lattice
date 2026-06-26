"""Probe: top-prime corpus lattice vision vs append/hybrid footprint."""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations

corpus = None
try:
    from aethos_symbol_knowledge import load_beir_corpus_text

    corpus = load_beir_corpus_text("scifact", max_docs=800)
    src = "scifact_800"
except Exception as e:
    src = f"synthetic ({e})"
    corpus = {
        f"d{i}": "quantum physics explores entanglement and superposition states deeply " * (1 + i % 3)
        for i in range(200)
    }

from aethos_append_index import AppendOnlyLatticeIndex
from lattice_retriever_v1.corpus_prime import corpus_scope
from lattice_retriever_v1.doc_lattice_codec import (
    DocPrimePool,
    encode_doc,
    select_rare_in_doc,
)
from lattice_retriever_v1.formula_index_codec import FormulaWalkIndex, encode_formula_index
from lattice_retriever_v1.hybrid_retriever import HybridConfig, build_hybrid_retriever
from lattice_retriever_v1.k_meet import velocity_meet
from lattice_retriever_v1.stage04_promote import promote_from_stream
from lattice_retriever_v1.stage06_composites import meet_composite_k
from lattice_retriever_v1.stage07_semantic_light import SemanticLightIndex
from lattice_retriever_v1.unit_lattice_codec import encode_bare_lumber

reg = promote_from_stream(list(corpus.values()))
scope = corpus_scope("scifact_probe", reg)
pool = DocPrimePool(pool_size=len(corpus) + 1000)
sem = SemanticLightIndex(registry=reg)
for text in corpus.values():
    sem.observe_doc(text)

corpus_prime = scope.corpus_prime
doc_records: dict[str, tuple[int, list[int], list[int]]] = {}
rare_term_sets: dict[str, set[str]] = {}
triple_meet_keys: dict[int, set[str]] = defaultdict(set)
global_3way: dict[int, dict] = {}

for doc_id, text in corpus.items():
    placement = encode_doc(doc_id, text, reg, pool, semantic=sem)
    rare = select_rare_in_doc(placement.words, sem, k=8)
    rare_primes = [sem._prime_for_term(w) for w in rare]
    try:
        doc_prime_under = meet_composite_k(corpus_prime, placement.doc_prime)
    except Exception:
        doc_prime_under = corpus_prime * placement.doc_prime
    doc_records[doc_id] = (doc_prime_under, rare_primes, [])
    rare_term_sets[doc_id] = set(rare)

term_to_docs: dict[str, set[str]] = defaultdict(set)
for doc_id, rare in rare_term_sets.items():
    for t in rare:
        term_to_docs[t].add(doc_id)

witness_triples = 0
for term, docset in term_to_docs.items():
    if len(docset) < 2:
        continue
    tp = sem._prime_for_term(term)
    for d1, d2 in combinations(sorted(docset), 2):
        chain = tuple(sorted({corpus_prime, tp, pool.doc_id_to_prime[d1]}))
        if len(chain) < 3:
            continue
        vel = velocity_meet(*chain)
        if vel is None or not vel.unified:
            continue
        try:
            meet_key = meet_composite_k(*chain)
        except Exception:
            continue
        witness_triples += 1
        triple_meet_keys[meet_key].update([d1, d2])
        entry = global_3way.setdefault(
            meet_key, {"doc_ids": set(), "correlated_terms": set()}
        )
        entry["doc_ids"].update([d1, d2])
        entry["correlated_terms"].add(term)

ai = AppendOnlyLatticeIndex()
for doc_id, text in corpus.items():
    ai.add(doc_id, text)
posting_entries = sum(len(v) for v in ai.postings.values())
n_primes = len(ai.token_prime)
append_bytes_est = posting_entries * 12 + n_primes * 8 + sum(len(t) for t in corpus.values())

hyb = build_hybrid_retriever(
    corpus, config=HybridConfig(lam_l2=0.15, enable_rare_shell_lattice=True)
)
router_pins = sum(len(v) for v in hyb.router.postings.values())
router_docs = len(hyb.router.docs)
rare_lat = hyb.rare_lattice
shell_anchors = len(rare_lat.anchor_postings) if rare_lat else 0
shell_neighbor = sum(len(v) for v in (rare_lat.neighbor_global.values() if rare_lat else []))
hybrid_bytes_est = router_pins * 12 + router_docs * 200 + shell_anchors * 16 + shell_neighbor * 20

tally_bytes_per_doc = 0
for doc_id, text in corpus.items():
    placement = encode_doc(doc_id, text, reg, pool, semantic=sem)
    n_steps = max(0, len(placement.order_stream) - 1)
    tally_bytes_per_doc += n_steps * 3

corpus_lattice_bytes = (
    8
    + len(doc_records) * (4 + 8 * 4)
    + len(triple_meet_keys) * (8 + 16)
    + tally_bytes_per_doc
)

sample_doc = next(iter(corpus.values()))
ws = [w for w in sample_doc.lower().split() if w.isalpha()][:30]
sorted_ws = sorted(ws, key=lambda w: (-sem.idf(w), sem._prime_for_term(w)))

sample = list(corpus.values())[0].encode("ascii", "ignore")[:200]
_, wire, fp = encode_bare_lumber(sample)

print("=== token sort by rare anchor ===")
print("before", ws[:8])
print("after ", sorted_ws[:8])
print("=== footprint probe ===")
print("source", src)
print("corpus_prime", corpus_prime)
print("n_docs", len(corpus))
print("append_posting_entries", posting_entries, "est_bytes", append_bytes_est)
print("hybrid_pin_postings", router_pins, "shell_anchors", shell_anchors, "est_bytes", hybrid_bytes_est)
print("corpus_lattice est_bytes", corpus_lattice_bytes)
print("triple_meet_keys", len(triple_meet_keys), "witness_triples_scanned", witness_triples)
print("ratio corpus_lattice / append", round(corpus_lattice_bytes / max(1, append_bytes_est), 3))
print("ratio corpus_lattice / hybrid", round(corpus_lattice_bytes / max(1, hybrid_bytes_est), 3))
print("bare_lumber_sample", fp.explain())
