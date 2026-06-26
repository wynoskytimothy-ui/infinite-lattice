#!/usr/bin/env python3
r"""
scifact_agentic_demo.py
=======================

Run scifact END-TO-END through the OSCAR / Pitagora agentic RAG with Timothy's
lattice retriever as the retrieval backend, on CPU.

WHAT IS REAL vs. STAND-IN
-------------------------
  REAL  : retrieval  -> Timothy's lattice retriever, backend="algebraic"
          (aethos_algebraic_corpus: the corpus IS a number; prime-FTA postings,
           BM25 over the prime lattice, warm corridor expansion). Pure CPU,
           no torch / no SPLADE / no GPU. (A GPU job is running; we deliberately
           DO NOT touch the splade backend.)
  REAL  : context build + de-reference (doc_id -> text) + provenance chain
          -> OSCAR's AethosSynthesizer.dereference()/build_prompt() (unmodified).
  STAND-IN : the text-generation step. No Ollama and no LLM API key are
          reachable, so AethosSynthesizer.generate() is monkeypatched with a
          LOCAL extractive generator (pulls the most query-relevant sentences
          from the retrieved context). It is clearly labelled "[LOCAL-EXTRACTIVE]".

ENABLE A REAL ABSTRACTIVE LLM (one line)
----------------------------------------
  Delete (or skip) the single monkeypatch line
      AethosSynthesizer.generate = _local_extractive_generate
  and construct the synthesizer with a real provider, e.g.:
      synth = AethosSynthesizer(store, provider="ollama", model_name="llama3")
  or  synth = AethosSynthesizer(store, provider="anthropic",
                                model_name="claude-3-5-sonnet-20241022",
                                api_key="sk-ant-...")
  -> the SAME retrieve->context->provenance pipeline now yields abstractive
     answers; nothing else changes.

This file ONLY ADDS a demo. It does not modify the retriever adapter or
aethos_llm.py.
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

# scifact corpus contains non-ASCII (Greek letters, dashes); the Windows console
# defaults to cp1252 and would crash on print(). Force UTF-8 with a safe
# 'replace' fallback so the demo output never dies on a character.
try:  # py3.7+: reconfigure the live stream in place
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except Exception:
    pass

# --- paths ------------------------------------------------------------------
TIMOTHY_REPO = r"C:/Users/wynos/New folder (3)"
PITAGORA_DIR = r"C:/Users/wynos/final-build-aethos-13/vendor/pitagora_andrea"
OSCAR_DIR = r"C:/Users/wynos/final-build-aethos-13/vendor/oscar"
for p in (TIMOTHY_REPO, PITAGORA_DIR, OSCAR_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

# Point the scifact loader at the OneDrive BEIR datasets (used by find_ds()).
os.environ.setdefault("BEIR_DATA_DIR", r"C:/Users/wynos/OneDrive/BEIR_datasets")

# Real retriever (CPU, algebraic) ...
from aethos_lattice_retriever import create_lattice_retriever          # noqa: E402
# ... real OSCAR synthesizer + doc store ...
from aethos_llm import AethosSynthesizer, DocumentStore                # noqa: E402
# ... scifact loader (corpus, queries, qrels_train, qrels_test).
from scripts.bench_supervised_bridges import load as load_scifact      # noqa: E402


# ===========================================================================
# LOCAL EXTRACTIVE STAND-IN for AethosSynthesizer.generate()
# ===========================================================================
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_WORD = re.compile(r"[a-z0-9]+")
_STOP = {
    "the", "a", "an", "of", "to", "in", "is", "are", "was", "were", "and",
    "or", "for", "on", "with", "as", "by", "that", "this", "these", "those",
    "it", "its", "be", "been", "has", "have", "had", "we", "our", "their",
    "can", "may", "which", "from", "at", "into", "than", "such", "does", "do",
}


def _content_words(text: str) -> List[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 2]


def _local_extractive_generate(self: "AethosSynthesizer", prompt: str) -> str:
    """LOCAL stand-in for an LLM: extractive synthesis.

    The prompt that OSCAR's build_prompt() hands us embeds the de-referenced
    fact blocks ("[1] ...\\n---\\n[2] ...") and the original "Question: ...".
    We parse those back out, score every sentence in the facts by overlap with
    the query's content words, and stitch the top few into a grounded answer.
    No network, no model weights. Clearly labelled so it is never mistaken for
    a real abstractive answer.
    """
    # Recover the question.
    qm = re.search(r"Question:\s*(.+?)\s*\nAnswer:", prompt, flags=re.S)
    query = qm.group(1).strip() if qm else ""
    if "(No verified facts found.)" in prompt or not query:
        return "[LOCAL-EXTRACTIVE] No verified facts were retrieved for this query."

    # Recover the fact blocks: everything between "Lattice:\n" and "\n\nQuestion:".
    fm = re.search(r"Verified facts from the Lattice:\n(.+?)\n\nQuestion:", prompt, flags=re.S)
    facts_blob = fm.group(1) if fm else ""
    # Strip the "[i] " block markers; keep block index for citation.
    blocks: List[Tuple[int, str]] = []
    for chunk in facts_blob.split("\n---\n"):
        m = re.match(r"\s*\[(\d+)\]\s*(.*)", chunk, flags=re.S)
        if m:
            blocks.append((int(m.group(1)), m.group(2).strip()))
        elif chunk.strip():
            blocks.append((len(blocks) + 1, chunk.strip()))

    qset = set(_content_words(query))
    scored: List[Tuple[float, int, str]] = []
    for bi, btext in blocks:
        for sent in _SENT_SPLIT.split(btext):
            sent = sent.strip()
            if len(sent) < 20:
                continue
            sw = _content_words(sent)
            if not sw:
                continue
            overlap = sum(1 for w in sw if w in qset)
            if overlap == 0:
                continue
            # overlap, normalised lightly by sentence length to avoid run-ons
            score = overlap + 0.25 * (overlap / len(sw))
            scored.append((score, bi, sent))

    if not scored:
        return ("[LOCAL-EXTRACTIVE] Retrieved facts did not contain a sentence "
                "overlapping the query terms; cannot extract a grounded answer.")

    scored.sort(key=lambda x: x[0], reverse=True)
    picked: List[Tuple[int, str]] = []
    seen_sent = set()
    for _score, bi, sent in scored:
        key = sent.lower()
        if key in seen_sent:
            continue
        seen_sent.add(key)
        picked.append((bi, sent))
        if len(picked) >= 3:
            break

    body = " ".join(s for _bi, s in picked)
    cites = sorted({bi for bi, _s in picked})
    cite_str = ", ".join(f"[{c}]" for c in cites)
    return f"[LOCAL-EXTRACTIVE] {body}  (supported by facts {cite_str})"


# Install the stand-in. THIS is the single line to remove to use a real LLM.
AethosSynthesizer.generate = _local_extractive_generate                 # noqa: E305


# ===========================================================================
# scifact slice
# ===========================================================================
def build_scifact_slice(n_docs: int = 800, n_queries: int = 5):
    """Load scifact and carve a slice that is guaranteed to contain the gold
    docs for the chosen queries (so retrieval can actually hit them)."""
    corpus, queries, qrels_train, qrels_test = load_scifact("scifact")

    # Prefer test queries that have a single, clear gold doc present in corpus.
    candidates = []
    for qid, rels in qrels_test.items():
        gold = [d for d, s in rels.items() if s > 0 and d in corpus]
        if gold and qid in queries:
            candidates.append((qid, gold))
    candidates.sort(key=lambda x: x[0])           # deterministic
    chosen = candidates[:n_queries]

    # Slice: all gold docs for chosen queries + fill to n_docs with other docs.
    slice_ids: List[str] = []
    seen = set()
    for _qid, gold in chosen:
        for d in gold:
            if d not in seen:
                seen.add(d)
                slice_ids.append(d)
    for d in corpus:                              # corpus order = jsonl order
        if len(slice_ids) >= n_docs:
            break
        if d not in seen:
            seen.add(d)
            slice_ids.append(d)

    docs = [corpus[d] for d in slice_ids]
    meta = [{"doc_id": d, "beir_id": d} for d in slice_ids]
    return docs, meta, slice_ids, corpus, queries, qrels_test, chosen


def _title_of(text: str, n: int = 70) -> str:
    """scifact corpus text is "title + ' ' + body"; the title is roughly the
    first sentence. Use a short prefix as a display title."""
    first = _SENT_SPLIT.split(text.strip())[0] if text.strip() else ""
    if len(first) > n:
        first = first[:n].rstrip() + "..."
    return first


# ===========================================================================
# main
# ===========================================================================
def main() -> None:
    print("=" * 78)
    print("scifact  ->  lattice retriever (algebraic, CPU)  ->  OSCAR synthesizer")
    print("=" * 78)

    docs, meta, slice_ids, corpus, queries, qrels_test, chosen = build_scifact_slice(
        n_docs=800, n_queries=5
    )
    print(f"\nLoaded scifact slice: {len(docs)} docs, {len(chosen)} queries "
          f"(BEIR_DATA_DIR={os.environ.get('BEIR_DATA_DIR')})")

    # --- 1. index the slice with the lattice retriever (CPU algebraic) -------
    retriever = create_lattice_retriever(backend="algebraic")
    retriever.add_documents(docs, metadata=meta)
    print(f"Indexed with backend={retriever.backend!r} "
          f"(scored_with_bm25={retriever.scored_with_bm25}); "
          f"GPU/SPLADE backend deliberately NOT used.")

    # --- 2. OSCAR synthesizer over the SAME slice ---------------------------
    # DocumentStore maps doc_id -> text (the de-reference target). Use a temp
    # store so we don't write into the repo.
    store_path = Path(tempfile.gettempdir()) / "scifact_agentic_docstore.json"
    if store_path.exists():
        store_path.unlink()
    store = DocumentStore(store_path)
    for d, text in zip(slice_ids, docs):
        store.put(d, text)
    # provider stays "ollama" by default, but generate() is the local stand-in.
    synth = AethosSynthesizer(store, provider="ollama", model_name="llama3")

    # --- 3. run the full agentic loop per query -----------------------------
    n_gold_hits = 0
    for qi, (qid, gold) in enumerate(chosen, 1):
        query = queries[qid]
        gold_set = set(gold)

        # RETRIEVE (real lattice retrieval) -> top-5
        hits = retriever.retrieve(query, top_k=5)        # [(text, score, meta)]
        retrieved_ids = [h[2].get("doc_id") for h in hits]
        gold_retrieved = bool(gold_set & set(retrieved_ids))
        n_gold_hits += int(gold_retrieved)

        print("\n" + "-" * 78)
        print(f"Q{qi}  [qid={qid}]  {query}")
        print(f"     gold doc_id(s): {sorted(gold_set)}   "
              f"GOLD RETRIEVED: {'YES' if gold_retrieved else 'no'}")
        print("     top-5 retrieved (doc_id, score, title):")
        for rank, (text, score, m) in enumerate(hits, 1):
            did = m.get("doc_id")
            star = " <-- GOLD" if did in gold_set else ""
            print(f"       {rank}. {did:>8}  score={score:8.4f}  "
                  f"{_title_of(text)}{star}")

        # CONTEXT + SYNTHESIZE (real de-reference / build_prompt; local generate)
        # We feed the synthesizer the doc_ids the lattice returned -> it builds
        # the context from the DocumentStore and runs generate(). Provenance is
        # exactly the doc_ids that survived de-referencing, in retrieved order.
        deref = synth.dereference(retrieved_ids)         # real: doc_id -> text
        prompt = synth.build_prompt(query, deref)        # real: traceable context
        answer = synth.generate(prompt)                  # LOCAL extractive stand-in

        # provenance: which doc_ids actually backed the context (order preserved)
        provenance = [d for d in retrieved_ids if store.get(d)]
        gold_in_prov = [d for d in provenance if d in gold_set]

        print("     synthesized answer:")
        print(f"       {answer}")
        print(f"     provenance chain (doc_ids the answer drew from): {provenance}")
        if gold_in_prov:
            print(f"     -> provenance INCLUDES gold: {gold_in_prov}")

    # --- summary ------------------------------------------------------------
    print("\n" + "=" * 78)
    print(f"END-TO-END OK: {len(chosen)} queries ran retrieve -> context -> "
          f"synthesize -> answer+provenance.")
    print(f"Gold doc retrieved in top-5 for {n_gold_hits}/{len(chosen)} queries "
          f"(lattice algebraic backend, CPU).")
    print("Retrieval + provenance are REAL; text-gen is the LOCAL extractive "
          "stand-in ([LOCAL-EXTRACTIVE]).")
    print("To enable a real abstractive LLM: remove the single line "
          "'AethosSynthesizer.generate = _local_extractive_generate'")
    print("and build the synthesizer with provider='ollama' (or "
          "provider='anthropic'/'openai' + api_key).")
    print("=" * 78)


if __name__ == "__main__":
    main()
