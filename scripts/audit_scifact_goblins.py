#!/usr/bin/env python3
"""GLASS-BOX GOBLIN HUNT on BEIR scifact.

A "goblin" = a NON-gold doc that the unified multi-view+corridor engine ranks at
position 1 while a GOLD doc sits in the top-20 -- i.e. it beat the right answer
for the WRONG reason. We diagnose WHY by comparing the rank-1 wrong doc vs the
top gold doc on cheap, O(pool) features:

  - length (#word tokens)
  - repetitiveness  distinct_words / total_words   (low = padded/repetitive)
  - did it win on a COMMON (low-idf) word vs a true RARE word
  - a NUMBER token match (digits) that the query also has
  - a char-trigram / 4-prefix COLLISION: the multi-view gear matched fragments
    ("^tio", "ing$") of a query word WITHOUT the wrong doc containing that word
  - which query words the GOLD has that the wrong doc LACKS (and vice versa)

Reuses unified.py (build_csr / doc_bag / bm25 / search / train_corridors / toks /
widf) verbatim -- same engine the bench scores. Run:

  BEIR_DATA_DIR=C:/Users/wynos/OneDrive/BEIR_datasets python scripts/audit_scifact_goblins.py
"""
import sys
from pathlib import Path
from collections import Counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import unified as U
from unified import (build_csr, train_corridors, search, bm25, doc_bag, toks,
                     widf, WORD)
from scripts.bench_supervised_bridges import load

NAME = "scifact"
MV = True            # multi-view engine (the one with gear collisions)
USE_CORR = True      # corridor pool-expansion + fuse (the full engine)
TOPK = 20            # gold must sit within top-K for a "ranking failure"
TARGET = 30          # collect up to this many failures


def word_set(text):
    return set(toks(text))


def word_list(text):
    return toks(text)


def distinct_ratio(text):
    wl = word_list(text)
    if not wl:
        return 1.0, 0
    return len(set(wl)) / len(wl), len(wl)


def raw_number_tokens(text):
    """Pure-digit (or digit-leading numeric) raw tokens, pre-stem."""
    return set(t for t in WORD.findall(text.lower()) if any(ch.isdigit() for ch in t))


def per_term_bm25_contrib(eng, query, doc_row, mv):
    """Decompose: for each query (view,token) term, how much BM25 mass it put on
    this doc_row, plus whether the term is a real WORD the doc has vs a gear
    fragment (trigram/prefix). Returns list of (key, contrib, idf)."""
    qbag = doc_bag(query, mv)
    contribs = []
    for k, qwt in qbag.items():
        t = eng["tid"].get(k)
        if t is None:
            continue
        wi = float(eng["idf"][t])
        if wi <= 0:
            continue
        s, e = int(eng["ptr"][t]), int(eng["ptr"][t + 1])
        dis = eng["di"][s:e]
        tfs = eng["tf"][s:e]
        # find this doc in the postings
        pos = np.where(dis == doc_row)[0]
        if pos.size == 0:
            continue
        tf = float(tfs[pos[0]])
        dl = float(eng["doclen"][doc_row])
        c = qwt * wi * tf * (U.K1 + 1) / (tf + U.K1 * (1 - U.B + U.B * dl / eng["avgdl"]))
        contribs.append((k, c, wi))
    contribs.sort(key=lambda x: -x[1])
    return contribs


def main():
    print(f"loading {NAME} ...")
    corpus, queries, train_q, test_q = load(NAME)
    test_ids = [q for q in test_q if q in queries]
    print(f"{NAME}: {len(corpus):,} docs | train {len(train_q):,} q | test {len(test_ids):,} q")

    eng = build_csr(corpus, MV)
    corr = train_corridors(eng, corpus, queries, train_q) if train_q else {}
    cids = eng["cids"]
    id2row = eng["id2row"]
    # median word-IDF threshold to call a term "rare" vs "common/hub"
    word_idfs = [float(eng["idf"][i]) for k, i in eng["tid"].items() if k[0] == "w" and eng["idf"][i] > 0]
    word_idfs.sort()
    RARE_IDF = float(np.percentile(word_idfs, 60))   # top-40% rarest = "rare"
    print(f"rare-word idf threshold (p60): {RARE_IDF:.2f}")

    failures = []
    for qid in test_ids:
        gold = {d for d, s in test_q[qid].items() if s > 0 and d in id2row}
        if not gold:
            continue
        # FULL engine ranking, deep enough to find gold within top-20
        ranked = search(eng, corr, queries[qid], USE_CORR, MV, k=TOPK)
        if not ranked:
            continue
        top = ranked[0]
        if top in gold:
            continue                          # got it right
        gold_in_top = [d for d in ranked[:TOPK] if d in gold]
        if not gold_in_top:
            continue                          # gold not even in top-20: a recall miss, not a goblin
        failures.append((qid, top, gold_in_top[0], ranked))
        if len(failures) >= TARGET:
            break

    print(f"\ncollected {len(failures)} ranking failures (wrong@1, gold in top-{TOPK})\n")

    # ---- per-failure feature extraction ----
    stats = Counter()
    rows = []
    for qid, wrong, goldd, ranked in failures:
        q = queries[qid]
        qwords = word_set(q)
        qraw_nums = raw_number_tokens(q)

        wtext, gtext = corpus[wrong], corpus[goldd]
        wwords, gwords = word_set(wtext), word_set(gtext)
        wdr, wlen = distinct_ratio(wtext)
        gdr, glen = distinct_ratio(gtext)

        # query words each doc contains
        wq = qwords & wwords
        gq = qwords & gwords
        # gold has these query words the wrong doc lacks (the "missed match")
        gold_only_q = gq - wq
        wrong_only_q = wq - gq

        # rare vs common matched query words (on word view)
        w_rare = {w for w in wq if widf(eng, w) >= RARE_IDF}
        w_common = wq - w_rare
        g_rare = {w for w in gq if widf(eng, w) >= RARE_IDF}

        # BM25 decomposition on the WRONG doc: what carried it?
        contribs = per_term_bm25_contrib(eng, q, id2row[wrong], MV)
        total_c = sum(c for _, c, _ in contribs) or 1e-9
        word_c = sum(c for k, c, _ in contribs if k[0] == "w")
        gear_c = sum(c for k, c, _ in contribs if k[0] in ("3", "p"))
        rare_word_c = sum(c for k, c, wi in contribs if k[0] == "w" and wi >= RARE_IDF)
        common_word_c = sum(c for k, c, wi in contribs if k[0] == "w" and 0 < wi < RARE_IDF)
        top_term = contribs[0] if contribs else (None, 0, 0)

        # number-token match: query digit-token appears in wrong doc
        wrong_nums = raw_number_tokens(wtext)
        num_match = bool(qraw_nums & wrong_nums)

        # gear-collision goblin: a sizeable share of the wrong doc's score came
        # from trigram/prefix gears, AND the wrong doc lacks query words the gold has
        gear_frac = gear_c / total_c
        gear_collision = gear_frac >= 0.30 and len(gold_only_q) >= 1

        # repetition goblin: wrong doc more repetitive (lower distinct ratio) AND longer
        repetition = (wdr < gdr - 0.03) and (wlen > glen)
        # length goblin: wrong doc much longer (BM25 length-norm under-penalizes)
        longer = wlen > 1.3 * max(glen, 1)
        # common-word goblin: wrong doc won mostly on low-idf hub words
        common_win = (common_word_c > rare_word_c) and (rare_word_c < 0.5 * total_c)
        # rare-miss goblin: gold matched a rare query word the wrong doc lacks
        rare_miss = len(g_rare - w_rare) >= 1

        if gear_collision:
            stats["gear_collision"] += 1
        if repetition:
            stats["repetition"] += 1
        if longer:
            stats["longer"] += 1
        if common_win:
            stats["common_word_win"] += 1
        if rare_miss:
            stats["rare_word_miss"] += 1
        if num_match:
            stats["number_token_match"] += 1

        rows.append(dict(
            qid=qid, q=q, wrong=wrong, gold=goldd,
            wlen=wlen, glen=glen, wdr=round(wdr, 3), gdr=round(gdr, 3),
            gear_frac=round(gear_frac, 3), word_frac=round(word_c / total_c, 3),
            rare_word_frac=round(rare_word_c / total_c, 3),
            common_word_frac=round(common_word_c / total_c, 3),
            top_term=top_term[0], top_term_idf=round(top_term[2], 2),
            qwords_in_wrong=sorted(wq), qwords_in_gold=sorted(gq),
            gold_only_qwords=sorted(gold_only_q), wrong_only_qwords=sorted(wrong_only_q),
            gold_rare_missed=sorted(g_rare - w_rare),
            num_match=num_match,
            flags=[f for f in ("gear_collision", "repetition", "longer",
                               "common_word_win", "rare_word_miss", "number_token_match")
                   if {"gear_collision": gear_collision, "repetition": repetition,
                       "longer": longer, "common_word_win": common_win,
                       "rare_word_miss": rare_miss, "number_token_match": num_match}[f]],
        ))

    # ---- aggregate report ----
    n = len(failures)
    print("=" * 70)
    print(f"GOBLIN PATTERN FREQUENCIES over {n} failures")
    print("=" * 70)
    for pat, c in stats.most_common():
        print(f"  {pat:22s} {c:3d}/{n}  ({100*c/n:4.1f}%)")

    # aggregate numeric contrasts
    arr = lambda key: np.array([r[key] for r in rows], float)
    print("\nMEAN CONTRASTS (wrong vs gold):")
    print(f"  len   wrong {arr('wlen').mean():6.1f}  gold {arr('glen').mean():6.1f}")
    print(f"  distinct-ratio  wrong {arr('wdr').mean():.3f}  gold {arr('gdr').mean():.3f}")
    print(f"  wrong-doc score: gear_frac {arr('gear_frac').mean():.3f}  "
          f"word_frac {arr('word_frac').mean():.3f}  "
          f"rare_word_frac {arr('rare_word_frac').mean():.3f}  "
          f"common_word_frac {arr('common_word_frac').mean():.3f}")
    longer_cnt = int((arr('wlen') > arr('glen')).sum())
    print(f"  wrong doc LONGER than gold: {longer_cnt}/{n}")
    morerep = int((arr('wdr') < arr('gdr')).sum())
    print(f"  wrong doc MORE repetitive (lower distinct ratio): {morerep}/{n}")

    print("\n" + "=" * 70)
    print("CONCRETE EXAMPLES")
    print("=" * 70)
    # show a spread: pick examples illustrating the top patterns
    shown_pats = set()
    for r in sorted(rows, key=lambda r: -len(r["flags"])):
        if not r["flags"]:
            continue
        key = tuple(r["flags"][:1])
        if key in shown_pats and len(shown_pats) >= 3:
            continue
        shown_pats.add(key)
        print(f"\n[{r['qid']}] flags={r['flags']}")
        print(f"  Q: {r['q'][:110]}")
        print(f"  WRONG@1 {r['wrong']}: len={r['wlen']} distinct={r['wdr']} "
              f"gear_frac={r['gear_frac']} rare_word_frac={r['rare_word_frac']}")
        print(f"     qwords_in_wrong: {r['qwords_in_wrong']}")
        print(f"  GOLD    {r['gold']}: len={r['glen']} distinct={r['gdr']}")
        print(f"     qwords_in_gold:  {r['qwords_in_gold']}")
        print(f"     gold has these query-words wrong LACKS: {r['gold_only_qwords']}")
        print(f"     gold rare-words wrong missed: {r['gold_rare_missed']}")
        if len([1 for x in shown_pats]) >= 6:
            break

    # dump rows for the report
    print("\n--- per-failure table (qid wlen glen wdr gdr gear word rare common flags) ---")
    for r in rows:
        print(f"{r['qid']:>4} {r['wlen']:4d} {r['glen']:4d} {r['wdr']:.2f} {r['gdr']:.2f} "
              f"g{r['gear_frac']:.2f} w{r['word_frac']:.2f} r{r['rare_word_frac']:.2f} "
              f"c{r['common_word_frac']:.2f}  {','.join(r['flags'])}")


if __name__ == "__main__":
    main()
