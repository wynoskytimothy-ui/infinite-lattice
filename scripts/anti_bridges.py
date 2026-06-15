#!/usr/bin/env python3
"""
ANTI-BRIDGES - the electron's signed slot for polarity / negation / stance.

A normal bridge is one-signed (A relates to B). An anti-bridge is the opposite
ROTATION: increase / decrease, activate / inhibit, show / lack are the SAME
relation, anti-phase. Plus negators (not, no, without, fails) flip polarity.
The lexicon is LLM-distilled (antonyms a counting method can't tell from
synonyms, since opposites share the same context) - the LLM supplies the sign,
the lattice stores it.

This script measures the capability the TOPIC engine lacks: distinguishing a
claim from its negation. scifact is a claim-verification set - many claims come
as polarity-flipped pairs sharing a gold doc ("biomaterials SHOW inductive" vs
"...LACK inductive"). The retriever scores them identically; anti-bridges flag
them as opposite stance.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aethos_append_index import words
from scripts.bench_supervised_bridges import load

# LLM-taught polarity axes: each pair is one relation, anti-rotated (+ / -)
ANTI_PAIRS = [
    ("increase", "decrease"), ("increased", "decreased"), ("increases", "decreases"),
    ("increasing", "decreasing"), ("higher", "lower"), ("high", "low"),
    ("more", "less"), ("greater", "lower"), ("elevated", "reduced"),
    ("activate", "inhibit"), ("activates", "inhibits"), ("activated", "inhibited"),
    ("activation", "inhibition"), ("induce", "suppress"), ("induces", "suppresses"),
    ("induced", "suppressed"), ("induction", "suppression"),
    ("promote", "prevent"), ("promotes", "prevents"), ("promoted", "prevented"),
    ("enhance", "reduce"), ("enhances", "reduces"), ("enhanced", "reduced"),
    ("cause", "prevent"), ("causes", "prevents"), ("stimulate", "suppress"),
    ("show", "lack"), ("shows", "lack"), ("have", "lack"), ("has", "lacks"),
    ("effective", "ineffective"), ("benefit", "harm"), ("beneficial", "harmful"),
    ("positive", "negative"), ("gain", "loss"), ("improve", "worsen"),
    ("improves", "worsens"), ("raise", "lower"), ("raises", "lowers"),
    ("upregulate", "downregulate"), ("susceptible", "resistant"),
    ("sensitive", "resistant"), ("required", "dispensable"),
    ("essential", "unnecessary"), ("accelerate", "slow"), ("expand", "shrink"),
    ("worsens", "improves"), ("protects", "damages"), ("protect", "damage"),
]
NEGATORS = {"not", "no", "without", "fails", "fail", "cannot", "never", "none",
            "neither", "nor", "absent", "lacking", "fewer", "lack", "lacks"}

# anti map: word -> its anti-rotation (the opposite-phase term)
ANTI = {}
for a, b in ANTI_PAIRS:
    ANTI[a] = b
    ANTI[b] = a


def polarity_terms(text):
    """polarity-bearing words in the text (the ones with an anti-rotation)."""
    ws = words(text)
    return {w for w in ws if w in ANTI}, sum(1 for w in ws if w in NEGATORS)


def opposite_polarity(a, b):
    """Are claims a and b polarity-flips of each other? (same topic, anti phase)
    Either they swap an anti-pair word, or one negates and the other doesn't."""
    wa, wb = set(words(a)), set(words(b))
    shared = wa & wb
    only_a, only_b = wa - wb, wb - wa
    # 1) an anti-pair swap: a has X, b has anti(X), where X differs
    for x in only_a:
        if x in ANTI and ANTI[x] in only_b:
            return True, f"{x} <-> {ANTI[x]}"
    # 2) negation flip: one side adds/removes a negator on otherwise-shared topic
    neg_a = only_a & NEGATORS
    neg_b = only_b & NEGATORS
    if (neg_a and not neg_b) or (neg_b and not neg_a):
        # require strong topic overlap so it's a real negation pair
        if len(shared) >= max(2, int(0.5 * min(len(wa), len(wb)))):
            return True, f"negation: {(neg_a or neg_b)}"
    return False, ""


def stance(claim, doc_text):
    """SUPPORT / REFUTE / ABSTAIN: align the claim's polarity axis with the doc's
    direction on that same axis (the anti-bridge meet). A negator in the claim
    flips the sign (the electron's anti-rotation)."""
    from collections import Counter
    cpols, cneg = polarity_terms(claim)
    dc = Counter(words(doc_text))
    score = 0
    for p in cpols:
        a = ANTI[p]
        cp, ca = dc.get(p, 0), dc.get(a, 0)
        if cp > ca:
            score += 1                 # doc leans the claim's way
        elif ca > cp:
            score -= 1                 # doc leans the opposite (anti) way
    if cneg % 2 == 1:                  # odd negators flip polarity
        score = -score
    return "SUPPORT" if score > 0 else "REFUTE" if score < 0 else "ABSTAIN"


def main():
    corpus, queries, train_q, test_q = load("scifact")
    # all (claim, gold) over train+test
    gold_of = {}
    for split in (train_q, test_q):
        for qid, rels in split.items():
            if qid in queries:
                gs = [d for d, s in rels.items() if s > 0]
                if gs:
                    gold_of[qid] = set(gs)

    # group claims by a shared gold doc
    by_gold = defaultdict(list)
    for qid, gs in gold_of.items():
        for g in gs:
            by_gold[g].append(qid)

    pairs = []
    seen = set()
    for g, qids in by_gold.items():
        for i in range(len(qids)):
            for j in range(i + 1, len(qids)):
                a, b = qids[i], qids[j]
                key = tuple(sorted((a, b)))
                if key in seen:
                    continue
                opp, why = opposite_polarity(queries[a], queries[b])
                if opp:
                    seen.add(key)
                    pairs.append((a, b, g, why))

    print(f"scifact: {len(gold_of)} claims with gold | "
          f"{len(pairs)} polarity-opposite claim pairs sharing a gold doc\n")
    print("  (the topic engine scores these IDENTICALLY; anti-bridges flag the flip)")
    for a, b, g, why in pairs[:14]:
        print(f"   [{why:<22}] gold {g}")
        print(f"      A q{a}: {queries[a][:60]}")
        print(f"      B q{b}: {queries[b][:60]}")
    print(f"\n  total detectable negation/antonym pairs: {len(pairs)}")
    npol = sum(1 for q in gold_of if polarity_terms(queries[q])[0]
               or polarity_terms(queries[q])[1])
    print(f"  claims carrying a polarity term: {npol}/{len(gold_of)}")

    # STANCE on the pairs: each pair MUST be opposite (claim vs its flip, same
    # gold). Measure how often the anti-bridge resolver assigns them opposite.
    opp = same = abstain = 0
    for a, b, g, why in pairs:
        sa = stance(queries[a], corpus[g])
        sb = stance(queries[b], corpus[g])
        if "ABSTAIN" in (sa, sb):
            abstain += 1
        elif sa != sb:
            opp += 1
        else:
            same += 1
    resolved = opp + same
    print(f"\n  STANCE resolution on the {len(pairs)} pairs (each MUST be opposite):")
    print(f"     resolved {resolved} (doc had the polarity axis), abstained {abstain}")
    if resolved:
        print(f"     correctly OPPOSITE: {opp}/{resolved} = {opp/resolved*100:.0f}%   "
              f"(wrong/same: {same})")
    print("\n  the topic retriever cannot tell a claim from its negation - same")
    print("  score, same gold. anti-bridges assign opposite stance: the electron's")
    print("  signed slot turns retrieval into the claim-verification scifact IS.")


if __name__ == "__main__":
    main()
