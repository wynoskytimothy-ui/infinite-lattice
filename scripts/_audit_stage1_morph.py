"""STAGE 1 morphology/typo stress test.

Corrupt query words (typo: swap two adjacent chars; or inflect: +s/+ed/+ing) so
the WORD gear can no longer match, then ask: does the char-trigram gear still
reach the gold doc? Compares word-only vs word+tri recall on corrupted queries.
"""
import sys, math, time, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.bench_supervised_bridges import load
from scripts._audit_stage1_gears import build_index, recall_at_k
from aethos_append_index import words

random.seed(7)


def corpus_vocab(corpus):
    v = {}
    for txt in corpus.values():
        for w in words(txt):
            v[w] = v.get(w, 0) + 1
    return v


def typo(w):
    """Swap two adjacent interior chars (classic transposition typo)."""
    if len(w) < 4:
        return w
    i = random.randint(1, len(w) - 2)
    return w[:i] + w[i + 1] + w[i] + w[i + 2:]


def inflect(w):
    """Append a plausible inflection so the surface form differs from corpus form."""
    if w.endswith("s"):
        return w + "es"
    if w.endswith("e"):
        return w + "d"
    return random.choice([w + "s", w + "ed", w + "ing"])


def corrupt_query(qtext, vocab, mode):
    """Corrupt the RAREST content word in the query (the one carrying the signal).
    Returns (new_query, original_word, corrupted_word) or None if nothing to do."""
    qws = [w for w in words(qtext) if len(w) >= 5 and w in vocab]
    if not qws:
        return None
    # rarest = lowest corpus frequency = highest idf = most load-bearing
    target = min(qws, key=lambda w: vocab[w])
    new = typo(target) if mode == "typo" else inflect(target)
    if new == target or new in vocab:
        return None  # corruption must actually leave the vocab to be a real test
    # replace first occurrence in the surface query text
    out = qtext.replace(target, new, 1)
    return out, target, new


def main():
    corpus, queries, train_q, test_q = load("scifact")
    test_ids = [q for q in test_q if q in queries and any(v > 0 for v in test_q[q].values())]
    vocab = corpus_vocab(corpus)

    idx_w = build_index(corpus, ["word"])
    idx_wt = build_index(corpus, ["word", "tri"])
    idx_full = build_index(corpus, ["word", "tri", "prefix"])

    for mode in ("typo", "inflect"):
        corrupted = {}
        for qid in test_ids:
            c = corrupt_query(queries[qid], vocab, mode)
            if c:
                corrupted[qid] = c[0]
        # measure on the SUBSET of queries we actually corrupted
        def avg_recall(idx, k=100):
            vals = []
            for qid, qtext in corrupted.items():
                ranked = idx.search(qtext, k=k)
                r = recall_at_k(ranked, test_q[qid], k)
                if r is not None:
                    vals.append(r)
            return sum(vals) / len(vals) if vals else 0.0

        # baseline: clean query (uncorrupted) recall on the same subset, word-only
        def avg_recall_clean(idx, k=100):
            vals = []
            for qid in corrupted:
                ranked = idx.search(queries[qid], k=k)
                r = recall_at_k(ranked, test_q[qid], k)
                if r is not None:
                    vals.append(r)
            return sum(vals) / len(vals) if vals else 0.0

        print(f"\n=== mode={mode}  ({len(corrupted)} queries corrupted, rarest content word) ===")
        print(f"  CLEAN query, word-only        R@100 = {avg_recall_clean(idx_w):.4f}   (ceiling)")
        print(f"  CORRUPTED, word-only          R@100 = {avg_recall(idx_w):.4f}")
        print(f"  CORRUPTED, word+tri           R@100 = {avg_recall(idx_wt):.4f}")
        print(f"  CORRUPTED, word+tri+prefix    R@100 = {avg_recall(idx_full):.4f}")
        # also R@10 to show ranking quality not just pool reach
        print(f"  --- R@10 ---")
        print(f"  CORRUPTED, word-only          R@10  = {avg_recall(idx_w,10):.4f}")
        print(f"  CORRUPTED, word+tri           R@10  = {avg_recall(idx_wt,10):.4f}")
        print(f"  CORRUPTED, word+tri+prefix    R@10  = {avg_recall(idx_full,10):.4f}")


if __name__ == "__main__":
    main()
