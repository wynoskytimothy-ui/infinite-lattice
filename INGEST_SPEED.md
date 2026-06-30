# Ingest speed — the unleveraged axis (now leveraged)

*We optimized query (binary-reader, mmap) and footprint, but ingest was still pure-Python dict-of-dicts, one
doc at a time. This is the ingest dive: profile the real bottleneck, then break it — keeping the index
bit-identical (so accuracy + footprint are untouched; only speed moves). Every number is a captured run.*

## Where the time actually goes (profile, `_o1_ingest_profile.py`)

For the word-only edge champion, incremental `add()` splits into: **tokenize ~32%** (regex), **dict-of-dicts
placement ~35–48%**, rest = bookkeeping. Placement ran at only **2–3M postings/s** — while the lattice monitor's
numba route does **230M/s**. The dict-of-dicts was the unleveraged bottleneck; tokenization the hidden floor.

## The fix, in two levers (both keep the index BIT-IDENTICAL)

### 1. Columnar build — remove the dict-of-dicts (`_o1_fast_ingest.py`)
Tokenize → per-doc bag → emit `(term, doc, tf)` rows → ONE vectorized `lexsort` → CSR per term. No
dict-of-dicts, and the sorted CSR is exactly what the mmap / binary-reader serve consumes (so it also skips the
separate `finalize()` step). **~2× faster end-to-end to serve-ready**, verified identical.

### 2. Numba byte-scan tokenizer — take tokenize+vocab out of Python (`_o1_ingest_numba.py`)
A `@njit` scanner walks the lowercased ASCII buffer, finds `[a-z][a-z0-9]{2,}` tokens, computes a **64-bit
FNV-1a hash per token inline** (no Python `str` objects ever), filters stopwords by hash (binary search), and
emits `(doc, hash, positional-weight)` as native arrays. Vocab = unique uint64 hashes. The whole tokenize+vocab
stage becomes a native scan + integer sort.

## Measured (word-only, vs incremental `add()+finalize()` = serve-ready baseline)

| corpus | baseline (add+finalize) | columnar | **numba** | numba speedup | identical? |
|---|---|---|---|---|---|
| scifact (5,183 docs, 724k tok) | 619 ms (1.2M tok/s) | 304 ms (2.0×) | **106 ms** | **5.9×** | ✓ PASS |
| fiqa (57,638 docs, 4.24M tok) | 4,223 ms (1.0M tok/s) | 2,257 ms (1.9×) | **1,213 ms** | **3.5×** | ✓ PASS |

**The tokenize wall is gone.** The numba scan+hash runs at **27–29M tok/s** (vs ~1M for the Python regex — a
~25× jump on the stage that was the floor). End-to-end ingest→serve-ready is **3.5–5.9× faster**, and the index
is verified identical to the last posting (the hash-collision probability among ~70k terms in 2⁶⁴ is ~1e-10).

Stage breakdown (numba, fiqa): `prep_buffer` 59 ms | **`scan+hash` 160 ms (27M tok/s)** | `group_csr` 994 ms.
The bottleneck is now the **numpy group-by sort** (the lexsort that builds CSR), not tokenization.

## Honest accounting

- **Bit-identical, verified.** Both levers reproduce the incremental index exactly (`verify_identical` /
  `verify`), so accuracy (nDCG) and footprint are unchanged — this is a pure *speed* win, nothing traded.
- **The numba output IS the mmap serve format** (`seg_doc` uint32, `seg_tf` float16, `indptr` int64) — ingest
  feeds the mmap serve directly, no dict-of-dicts and no `finalize()` anywhere in the pipeline.
- **Parallel tokenize on Windows is marginal** (`_fast_tok.py` lightweight workers): ~2.3× on fiqa, *slower* on
  small corpora (process-spawn overhead). It helps big corpora; on Linux (fork) it would help more. The numba
  path beats it without any multiprocessing.
- **The new bottleneck is the group-by sort** (~3–4M postings/s, numpy lexsort). That's the next lever: a radix
  / parallel sort would push further. The hash-keyed vocab also needs a `hash→token` map (~70k entries, built by
  decoding unique spans) for the bridge layer — cheap, not yet wired.

## Projection to the "100M tokens" target

- Tokenize+hash (the former wall): **27M tok/s → 100M tokens in ~3.7 s**, single core.
- Full ingest→serve-ready: sort-bound at ~3.5M tok/s → 100M tokens in ~30 s today. The tokenizer no longer
  limits it; a parallel/radix group-by sort is the remaining lever to approach the 1-second vision.

## Files
- `_o1_ingest_profile.py` — the tokenize/placement profile (the bottleneck).
- `_o1_fast_ingest.py` — columnar build (2×, identical) + lightweight parallel tokenize.
- `_o1_ingest_numba.py` — numba byte-scan tokenize+hash (3.5–5.9×, 27M tok/s, identical).
- `_fast_tok.py` — re-only lightweight tokenizer for the parallel workers.
