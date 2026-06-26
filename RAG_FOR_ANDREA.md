# AETHOS RAG — for Andrea

**The index-size gate you named on May 29 is cleared — and I want to be precise about exactly what is
measured, what pipeline it came from, and what is still open. Every number here traces to a captured run in
[`MEASUREMENTS.md`](MEASUREMENTS.md); I re-audited the earlier draft of this doc and corrected several
numbers that were misattributed or unbacked.**

> Your ask: *"accuracy is aligned with best of breed… query response time will match or beat best of breed.
> However the size of the index is a problem. Best of breed use 1024-dim vectors = 4 KB/doc. We are at 86.6 KB.
> Can you get below 4 KB while maintaining accuracy?"*

---

## The headline: index size, solved — at full accuracy

On the **full MS MARCO 8.8M-doc** corpus, the native SPLADE-on-lattice index is **286.9 B/doc**
(2.54 GB total, lossless, round-trip verified):

- **~14× under your 4 KB target, ~300× smaller than the 86.6 KB** we started at.
- Accuracy holds at the small size: **MRR@10 ≈ 0.40** (0.3989 on a 200-query sample of dev-small,
  full corpus, **no cross-encoder**) — squarely in the SPLADE++/ColBERT band, where dense RAG needs a
  reranker to get.

That is the real win, and it is exactly the "≈12 hub pins, not a 1024-d float vector" pipeline from my
June 8 email, now measured at scale.

## The honest caveat before we pitch: serve speed

The native SPLADE serve is currently **3.2 s/query** (median 3234 ms, p90 4336 ms) — **not** milliseconds.
The compression and the accuracy are solved; **serving speed is the open engineering item.** SPLADE queries
fan out to hundreds of expansion terms, and the serve scores them without early termination yet. The known
next step is a **WAND / block-max early-termination pass** (already proven ~18× on the lexical lattice) —
until that is applied, this index is *smallest + most accurate, not yet fastest.*

(The 31 ms you measured was the **lexical lattice** branch — a different, faster, less-compressed pipeline.
I had conflated the two in the first draft; they are separated below.)

## The two pipelines, kept separate (no conflation)

| | **A. Lexical multi-view lattice** (what you tested) | **B. Native SPLADE-on-lattice** (the 287 B/doc result) |
|---|---|---|
| MARCO 8.8M footprint | larger (the 86.6 KB/doc you flagged, slimmed) | **286.9 B/doc** ✅ |
| MARCO MRR@10 | — | **0.3989** (200-q sample, no CE) |
| Serve speed | ~31 ms (your measurement) | **3.2 s/q** (needs WAND) |
| BEIR scifact nDCG@10 | **0.7023** lattice-only (> BM25 0.665), no CE | not run on BEIR |
| BEIR nfcorpus / fiqa | lattice-only loses to BM25; **0.349 / 0.352 with a CE rerank** | not run on BEIR |

So: pipeline **A** is fast and BM25-class-to-better on scifact; pipeline **B** is the tiny, SOTA-accuracy
index whose serve still needs the speed pass. Neither is "small **and** fast **and** SOTA" *today* — but the
two halves are each real, and joining them (SPLADE index + WAND serve) is the concrete next milestone.

## Receipts (each traces to MEASUREMENTS.md)

| Metric | Value | Source |
|---|---|---|
| MARCO index footprint | **286.9 B/doc** (2.54 GB, 1.06B postings, lossless) | `_splade_index_serve.log`, stat of `splade_index_for.npz` |
| MARCO MRR@10 (full corpus, no CE) | **0.3989** (200-q sample; full run pending) | `_serve_sample.log` |
| MARCO recall@100 | 92.0% (200-q sample) | `_serve_sample.log` |
| MARCO serve latency | median **3234 ms** (open item) | `_serve_sample.log` |
| BEIR scifact nDCG@10 (lexical lattice, no CE) | **0.7023** (> BM25 0.665) | `_r1_beir.out` |
| BEIR nfcorpus / fiqa (lexical lattice **+ CE**) | 0.3489 / 0.3522 | `_r1_beir.out` |
| Lossless FOR codec | **4.97×** byte-exact (not the 6.2× I wrote before) | FOR round-trip logs |

## What is genuinely novel (architectural — true by construction)

These are properties of the design, not benchmark numbers:

- **The document index is a tiny self-describing certificate** (prime hubs + posting gaps), not a dense
  float vector — that is why it is 287 B/doc.
- **Invertible / append-only**: a doc is a product of prime addresses → decode = factor the number, add a
  doc = one multiply (no full reindex). (`aethos_algebraic_corpus.py`)
- **Correlations regenerated from the meet at query time** — not a separately stored matrix.
- **No cross-encoder** in pipeline B's 0.40 — most systems need a CE to reach that.

## How to import / run it

One interface, two backends (matches the Pitagora `add_documents` / `retrieve` contract):

```python
from aethos_lattice_retriever import create_lattice_retriever

# self-contained, CPU, no model deps (BM25-class, good for dev/offline):
r = create_lattice_retriever(backend="algebraic")
r.add_documents(docs, metadata=[{"doc_id": i} for i in range(len(docs))])
hits = r.retrieve("your query", top_k=10)      # -> [(text, score, {"doc_id": ...}), ...]

# the SPLADE index (needs torch + naver/splade-cocondenser-ensembledistil):
r = create_lattice_retriever(backend="splade")
```

- Native MARCO build (the 287 B/doc index): `marco_splade_native.py` — `encode` → `index --full --chamber` → `serve --full`.
- Agentic demo on scifact: `python scifact_agentic_demo.py` (LLM step is a local stand-in until an API key/Ollama is wired).

## Bottom line for the pitch

**The index-size blocker is genuinely gone: 287 B/doc on full MARCO at MRR ≈ 0.40, no cross-encoder —
smallest index at SOTA-band accuracy.** The one honest gap is serve speed (3.2 s/q today); closing it with the
WAND pass is the next milestone, and I'd like to prioritize that with you. I am deliberately *not* claiming
"faster than best-of-breed" for this pipeline until it's measured — the numbers above are what the runs
actually show.
