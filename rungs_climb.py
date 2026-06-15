#!/usr/bin/env python3
"""Ground-up climb, rungs 1->2. Same fixed engine + pool; swap only the tokenizer.
  rung1  words            -- the floor (= BM25, 0.5419)
  rung2  stem             -- normalize morphological variants (diabetic/diabetes -> diabet)
  rung2  word+stem        -- MULTI-VIEW: keep the exact word (precision) AND its stem (recall)
Measured vs the floor with the hit/miss split, so we see if subwords add cleanly.
Climb only if a rung beats the one below it.
"""
import re, gc
from marco_lab import load_pool, Index, evaluate

WORD = re.compile(r"[a-z0-9]+")
SUF = ("ization", "ational", "tion", "ment", "ness", "ing", "ed", "es", "ly", "s")


def words(s):
    return WORD.findall(s.lower())


def _stem(w):
    for suf in SUF:
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[:-len(suf)]
    return w


def stems(s):
    return [_stem(w) for w in WORD.findall(s.lower())]


def word_plus_stem(s):
    out = []
    for w in WORD.findall(s.lower()):
        out.append(w)
        st = _stem(w)
        if st != w:
            out.append("§" + st)        # stem in its own namespace so it doesn't collide with words
    return out


if __name__ == "__main__":
    qids, queries, qrels, texts = load_pool()
    print("\nRUNG CLIMB 1->2 (MRR@10, lower-engine fixed, only the tokenizer changes):\n")
    ref = None
    for label, tok in [("rung1 words (floor)", words),
                       ("rung2 stem", stems),
                       ("rung2 word+stem", word_plus_stem)]:
        idx = Index(tok).build(texts)
        res = evaluate(idx, qids, queries, qrels, ref_rr=ref, label=label)
        if ref is None:
            ref = res["rrs"]
        del idx
        gc.collect()
    print("\n  beat the floor + miss>0 with NO drift = subwords are a real rung. then add corridor on top.")
