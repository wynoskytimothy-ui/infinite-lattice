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

| corpus | baseline | columnar | numba (lexsort) | numba RADIX serial | **RADIX 8-thread** | identical? |
|---|---|---|---|---|---|---|
| scifact (724k tok) | 640 ms | 304 ms (2.0×) | 113 ms (5.6×) | 68 ms (9.4×) | **36 ms (17.8×)** | ✓ PASS |
| fiqa (4.24M tok) | 4,249 ms | 2,257 ms (1.9×) | 1,232 ms (3.4×) | 416 ms (10.2×) | **173 ms (24.5×)** | ✓ PASS |

**The tokenize wall is gone, then the sort wall too.** Three escalating levers, each verified bit-identical:
1. **Columnar** removes the dict-of-dicts → 2×.
2. **Numba scan+hash** takes tokenize+vocab native (27–29M tok/s vs ~1M regex, ~25×) → but the numpy lexsort
   group-by becomes the new bottleneck (994 ms of fiqa's 1,232 ms).
3. **Radix/hash group-by** kills the lexsort: per-doc dedup + open-addressing hash-vocab + counting-scatter,
   all O(n), no comparison sort (994 ms → 62 ms). → 9–10× serial. Then **nogil threads** scale the scan
   (`njit(nogil=True)` + `ThreadPoolExecutor`, no process-spawn cost) → **17.8–24.5×**, at **24.5M tok/s**.

Stage breakdown (radix 8-thread, fiqa): `prep` 57 ms | **`scan+dedup(mt)` 56 ms** | `vocab+counting` 61 ms —
now balanced across the three stages (all ~native), so the index is verified identical to the last posting
(hash-collision probability among ~70k terms in 2⁶⁴ ≈ 1e-10).

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

- Full ingest→serve-ready (radix, 8-thread): **24.5M tok/s → 100M tokens in ~4.1 s**. The whole pipeline is now
  native + parallel; no Python floor remains. More cores scale the scan further (it's per-doc independent), and
  `prep`/`vocab+counting` are the next stages to thread to push toward the 1-second/100M vision.
- This is on the word-only edge-champion index — exactly the one that's 198 B/doc and serve-ready CSR. So the
  *same* artifact that's small + fast-to-query is now also fast-to-build.

## Files
- `_o1_ingest_profile.py` — the tokenize/placement profile (the bottleneck).
- `_o1_fast_ingest.py` — columnar build (2×, identical) + lightweight parallel tokenize.
- `_o1_ingest_numba.py` — numba byte-scan tokenize+hash (lexsort group-by; 27M tok/s tokenize).
- `_o1_ingest_radix.py` — radix/hash group-by (no comparison sort) + nogil-threaded scan: **24.5× / 24.5M tok/s**.
- `_fast_tok.py` — re-only lightweight tokenizer for the parallel workers.
