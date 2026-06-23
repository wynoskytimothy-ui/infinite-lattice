#!/usr/bin/env python3
"""Glass-box goblin hunt on BEIR 'arguana'.

A GOBLIN = a non-gold doc ranked #1 while a gold doc sits in the top-20, that won
for the WRONG reason (length pile-up, repetition, common/hub-word match, number
collision, char-trigram/prefix fragment collision rather than the real rare word).

We reuse unified.py's multi-view engine (build_csr / bm25 / doc_bag / search).
For each ranking failure we compute CHEAP features on the rank-1 wrong doc vs the
gold doc and quantify which goblin caused the inversion. Then we report patterns
+ exact O(pool) demotion rules.
"""
import os, sys, math, re
from collections import Counter
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unified as U
from scripts.bench_supervised_bridges import load

WORD = U.WORD


def words_raw(s):
    return WORD.findall(s.lower())


def feats(text):
    w = words_raw(text)
    n = len(w)
    distinct = len(set(w))
    return dict(n=n, distinct=distinct, ratio=(distinct / n if n else 0.0), bag=set(U.toks(text)))


def per_term_contrib(eng, qbag, row):
    """How much each query (view,token) term contributed to doc `row`'s bm25 score,
    so we can attribute the win to common vs rare, word vs trigram/prefix terms."""
    out = []
    for k, qwt in qbag.items():
        t = eng["tid"].get(k)
        if t is None:
            continue
        wi = float(eng["idf"][t])
        if wi <= 0:
            continue
        s, e = int(eng["ptr"][t]), int(eng["ptr"][t + 1])
        dis = eng["di"][s:e]; tfs = eng["tf"][s:e]
        hit = np.where(dis == row)[0]
        if len(hit) == 0:
            continue
        tf = float(tfs[hit[0]])
        dl = float(eng["doclen"][row])
        contrib = qwt * wi * tf * (U.K1 + 1) / (tf + U.K1 * (1 - U.B + U.B * dl / eng["avgdl"]))
        out.append((k, wi, tf, contrib))
    return out


NUMTOK = re.compile(r"^\d")


def main():
    name = "arguana"
    corpus, queries, train_q, test_q = load(name)
    test_ids = [q for q in test_q if q in queries]
    mv = True  # multi-view (word + char-trigram + 4-char prefix) -- where goblins live
    eng = U.build_csr(corpus, mv)
    corr = U.train_corridors(eng, corpus, queries, train_q) if train_q else {}
    id2row = eng["id2row"]; cids = eng["cids"]; idf = eng["idf"]

    # idf threshold separating "hub/common" from "rare" WORD terms (median of word idfs)
    word_idfs = np.array([idf[i] for i, k in enumerate(eng["tid"]) ])  # not used; compute below
    w_idf_vals = sorted(U.widf(eng, w) for w in eng["wtid"])
    RARE_IDF = float(np.percentile([v for v in w_idf_vals if v > 0], 60))  # top 40% rare-ish

    failures = []  # (qid, wrongrow, goldrow, goldrank)
    for q in test_ids:
        ranked = U.search(eng, corr, queries[q], bool(corr), mv, k=20)
        gold = {d for d, s in test_q[q].items() if s > 0 and d in id2row}
        if not gold or not ranked:
            continue
        if ranked[0] in gold:
            continue  # gold already #1, no goblin
        # gold present in top-20?
        gpos = next((i for i, d in enumerate(ranked) if d in gold), None)
        if gpos is None:
            continue
        goldcid = ranked[gpos]
        failures.append((q, ranked[0], goldcid, gpos))

    print(f"arguana: {len(corpus):,} docs | test {len(test_ids):,} q | "
          f"rank-1 inversions with gold in top-20: {len(failures)}")
    print(f"RARE_IDF threshold (word, 60th pct of positive idf) = {RARE_IDF:.2f}\n")

    # ---- quantify goblins across (up to) all failures, detail first 30 ----
    stats = Counter()
    examples = []
    DETAIL = 30
    for idx, (qid, wcid, gcid, gpos) in enumerate(failures):
        qbag = U.doc_bag(queries[qid], mv)
        wrow, grow = id2row[wcid], id2row[gcid]
        wf, gf = feats(corpus[wcid]), feats(corpus[gcid])
        qset = wf["bag"] | gf["bag"]  # not used directly
        qwords = set(U.toks(queries[qid]))

        wc = per_term_contrib(eng, qbag, wrow)
        gc = per_term_contrib(eng, qbag, grow)
        wtot = sum(c for *_, c in wc); gtot = sum(c for *_, c in gc)

        def split(contribs):
            word_rare = word_hub = tri = pre = num = 0.0
            for k, wi, tf, c in contribs:
                view, tok = k
                if view == "3":
                    tri += c
                elif view == "p":
                    pre += c
                else:  # word
                    if NUMTOK.match(tok):
                        num += c
                    elif wi >= RARE_IDF:
                        word_rare += c
                    else:
                        word_hub += c
            return dict(word_rare=word_rare, word_hub=word_hub, tri=tri, pre=pre, num=num)

        ws, gs = split(wc), split(gc)

        # rare query WORD terms each doc actually contains
        qw = set(words_raw(queries[qid]))
        qw_rare = {U.st(w) for w in qw if U.widf(eng, U.st(w)) >= RARE_IDF}
        wrong_has_rare = qw_rare & wf["bag"]
        gold_has_rare = qw_rare & gf["bag"]
        gold_only_rare = gold_has_rare - wrong_has_rare   # rare query words gold has, wrong lacks
        wrong_only_rare = wrong_has_rare - gold_has_rare

        # ---- goblin classification (a failure can trip several) ----
        tripped = []
        # G1 LENGTH: wrong doc materially longer than gold
        if wf["n"] >= 1.15 * gf["n"]:
            stats["G1_longer"] += 1; tripped.append("longer")
        # G2 REPETITIVE: wrong doc lower distinct-ratio (pumps tf) than gold
        if wf["ratio"] < gf["ratio"] - 0.03:
            stats["G2_repetitive"] += 1; tripped.append("repetitive")
        # G3 HUB-WIN: wrong doc's score is more hub/common-word driven than gold's
        whub_frac = (ws["word_hub"] / wtot) if wtot else 0
        ghub_frac = (gs["word_hub"] / gtot) if gtot else 0
        if whub_frac > ghub_frac + 0.10:
            stats["G3_hub_driven"] += 1; tripped.append("hub")
        # G4 FRAGMENT (trigram/prefix) win: wrong leans on char-fragment views more
        wfrag = (ws["tri"] + ws["pre"]) / wtot if wtot else 0
        gfrag = (gs["tri"] + gs["pre"]) / gtot if gtot else 0
        if wfrag > gfrag + 0.05:
            stats["G4_fragment"] += 1; tripped.append("fragment")
        # G5 NUMBER match: wrong got non-trivial score from a number token gold lacks
        if ws["num"] > 0 and ws["num"] >= gs["num"] + 1e-6 and ws["num"] / max(wtot, 1e-9) > 0.02:
            stats["G5_number"] += 1; tripped.append("number")
        # G6 RARE-WORD DEFICIT: gold contains rare query words the wrong doc lacks,
        #    AND wrong doc has fewer matched rare query words than gold
        if len(gold_only_rare) >= 1 and len(wrong_has_rare) < len(gold_has_rare):
            stats["G6_rare_deficit"] += 1; tripped.append("rare-deficit")
        # NEAR-DUP probe: arguana-specific. wrong doc shares almost all words with QUERY
        # (it's the same argument / near-restatement), not a true answer.
        qbagset = set(U.toks(queries[qid]))
        wjac = len(wf["bag"] & qbagset) / max(len(wf["bag"] | qbagset), 1)
        gjac = len(gf["bag"] & qbagset) / max(len(gf["bag"] | qbagset), 1)
        if wjac > gjac + 0.05:
            stats["G7_query_echo"] += 1; tripped.append("query-echo")

        stats["TOTAL"] += 1
        if len(examples) < 6 and tripped:
            examples.append(dict(qid=qid, tripped=tripped, gpos=gpos,
                wlen=wf["n"], glen=gf["n"], wratio=round(wf["ratio"], 3), gratio=round(gf["ratio"], 3),
                whub=round(whub_frac, 2), ghub=round(ghub_frac, 2),
                wfrag=round(wfrag, 2), gfrag=round(gfrag, 2),
                gold_only_rare=sorted(gold_only_rare)[:6], wrong_only_rare=sorted(wrong_only_rare)[:6],
                wjac=round(wjac, 2), gjac=round(gjac, 2),
                q=queries[qid][:70].replace("\n", " ")))

        if idx < DETAIL:
            print(f"[{idx:02d}] q={qid[:34]:34s} gold@{gpos:2d} trip={','.join(tripped)}")
            print(f"     len  wrong={wf['n']:4d} gold={gf['n']:4d} | ratio wrong={wf['ratio']:.3f} gold={gf['ratio']:.3f}")
            print(f"     score wrong={wtot:6.2f} (hub {whub_frac:.0%} frag {wfrag:.0%} num {ws['num']:.2f}) "
                  f"gold={gtot:6.2f} (hub {ghub_frac:.0%} frag {gfrag:.0%})")
            print(f"     rare q-words: gold-only={sorted(gold_only_rare)[:5]} wrong-only={sorted(wrong_only_rare)[:5]} "
                  f"| query-jaccard wrong={wjac:.2f} gold={gjac:.2f}")

    n = max(stats["TOTAL"], 1)
    print(f"\n{'='*64}\nGOBLIN FREQUENCY across {stats['TOTAL']} rank-1 inversions (gold in top-20):")
    for key in ["G1_longer", "G2_repetitive", "G3_hub_driven", "G4_fragment",
                "G5_number", "G6_rare_deficit", "G7_query_echo"]:
        print(f"  {key:16s} {stats[key]:4d}  ({stats[key]/n:5.1%})")

    # ---- counterfactual rule sims: how many failures each rule would FIX, and
    #      collateral (how many CURRENTLY-CORRECT rank-1s the rule would wrongly demote)
    print(f"\n{'='*64}\nRULE COUNTERFACTUALS (fix = wrong#1 demoted below gold; collateral = correct#1 demoted):")
    # gather currently-correct rank-1 docs for collateral check
    correct = []  # (qid, goldrow)
    for q in test_ids:
        ranked = U.search(eng, corr, queries[q], bool(corr), mv, k=1)
        gold = {d for d, s in test_q[q].items() if s > 0 and d in id2row}
        if ranked and ranked[0] in gold:
            correct.append((q, id2row[ranked[0]]))

    def rule_demotes(qid, wrow, grow, rule):
        wf, gf = feats(corpus[cids[wrow]]), feats(corpus[cids[grow]])
        qbagset = set(U.toks(queries[qid]))
        wjac = len(wf["bag"] & qbagset) / max(len(wf["bag"] | qbagset), 1)
        gjac = len(gf["bag"] & qbagset) / max(len(gf["bag"] | qbagset), 1)
        if rule == "ratio":   # demote low distinct-ratio
            return wf["ratio"] < gf["ratio"] - 0.03
        if rule == "echo":    # demote high query-jaccard (near-restatement of query)
            return wjac > gjac + 0.08
        if rule == "len":     # demote much longer
            return wf["n"] > 1.25 * gf["n"]
        return False

    # for collateral we compare the correct rank-1 against its closest runner-up gold-less doc.
    # cheap proxy: a rule is "collateral" if applied to the correct doc it would penalize it
    # relative to the typical doc -- approximate by absolute thresholds.
    def rule_penalizes_abs(row, rule):
        f = feats(corpus[cids[row]])
        if rule == "ratio":
            return f["ratio"] < 0.55          # absolute low-diversity flag
        if rule == "len":
            return f["n"] > 600               # absolute long flag
        if rule == "echo":
            return False                      # echo is relative-to-query, needs the query; skip abs
        return False

    for rule in ["ratio", "echo", "len"]:
        fixed = sum(1 for (qid, wrow, gcid, gp) in failures
                    if rule_demotes(qid, id2row[wrow] if not isinstance(wrow, (int, np.integer)) else wrow,
                                    id2row[gcid], rule))
        coll = sum(1 for (qid, grow) in correct if rule_penalizes_abs(grow, rule))
        print(f"  rule={rule:6s}: would demote wrong#1 in {fixed:4d}/{stats['TOTAL']} failures "
              f"({fixed/n:.0%}) | abs-flags {coll}/{len(correct)} correct rank-1 golds ({coll/max(len(correct),1):.0%})")

    print(f"\n(correct rank-1 count for collateral baseline: {len(correct)})")


if __name__ == "__main__":
    main()
