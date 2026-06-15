#!/usr/bin/env python3
"""Ground-up climb, rung 3: COMPOSITES on the locked stem base (rung2 = 0.5693).
Test phrase-composites (P×Q free tokens) -- do stem-bigrams beat stemming alone? idf
already weights the rare/discriminative composites highest (= "biggest primes carry meaning").
Keep a rung only if it beats the one below it (0.5693).
"""
import re, gc
from marco_lab import load_pool, Index, evaluate

WORD = re.compile(r"[a-z0-9]+")
SUF = ("ization", "ational", "tion", "ment", "ness", "ing", "ed", "es", "ly", "s")


def _stem(w):
    for suf in SUF:
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[:-len(suf)]
    return w


def stems(s):
    return [_stem(w) for w in WORD.findall(s.lower())]


def stem_bigram(s):
    st = [_stem(w) for w in WORD.findall(s.lower())]
    out = list(st)
    for i in range(len(st) - 1):
        out.append(st[i] + "~" + st[i + 1])        # adjacent phrase composite (P x Q)
    return out


def stem_skipbigram(s):
    """phrase composites within a small window (order-free pairs of nearby rare-ish stems)."""
    st = [_stem(w) for w in WORD.findall(s.lower())]
    out = list(st)
    for i in range(len(st)):
        for j in range(i + 1, min(i + 4, len(st))):
            a, b = st[i], st[j]
            out.append((a + "~" + b) if a <= b else (b + "~" + a))   # commutative composite
    return out


if __name__ == "__main__":
    qids, queries, qrels, texts = load_pool()
    print("\nRUNG 3 -- composites on the stem base (MRR@10; base stem = 0.5693):\n")
    ref = None
    for label, tok in [("stem (base)", stems),
                       ("stem + bigram", stem_bigram),
                       ("stem + skip-bigram", stem_skipbigram)]:
        idx = Index(tok).build(texts)
        res = evaluate(idx, qids, queries, qrels, ref_rr=ref, label=label)
        if ref is None:
            ref = res["rrs"]
        del idx
        gc.collect()
    print("\n  beat 0.5693 = composites are a real rung -> lock it -> add corridor next.")
