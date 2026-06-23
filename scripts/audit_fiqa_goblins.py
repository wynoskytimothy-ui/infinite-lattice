#!/usr/bin/env python3
"""GLASS-BOX failure analysis on BEIR 'fiqa' -- hunt GOBLINS: a NON-gold doc at
rank-1 while a gold doc sits in the top-20, won for the WRONG reason.

Reuses unified.py's multi-view BM25 engine + supervised corridors. For each such
ranking failure we DECOMPOSE the rank-1 wrong doc's BM25 score by (view,token)
contribution and compare it head-to-head with the best-ranked gold doc on:
  - length (#word tokens), repetitiveness (distinct/total word ratio)
  - did the win come from a COMMON/low-idf word or a true RARE word
  - a NUMBER-token match
  - a char-trigram / 4-char-prefix COLLISION (matched a fragment, not the real word)
  - which query WORDS the gold has that the wrong doc lacks (and vice versa)

Run: BEIR_DATA_DIR=C:/Users/wynos/OneDrive/BEIR_datasets python scripts/audit_fiqa_goblins.py
"""
import sys, os, re, math
from collections import Counter
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unified as U
from unified import toks, doc_bag, build_csr, train_corridors, bm25, st, WORD, K1, B

NUM = re.compile(r"^\d[\d.,]*$")


def word_contribs(eng, qbag_words, drow):
    """Per WORD-view query term: BM25 contribution to doc row `drow` + that term's idf.
    Returns {word: (contrib, idf, tf)}."""
    out = {}
    for (view, w), qwt in qbag_words.items():
        if view != "w":
            continue
        t = eng["tid"].get(("w", w))
        if t is None:
            continue
        wi = float(eng["idf"][t])
        if wi <= 0:
            continue
        s, e = int(eng["ptr"][t]), int(eng["ptr"][t + 1])
        dis = eng["di"][s:e]; tfs = eng["tf"][s:e]
        pos = np.searchsorted(dis, drow)
        if pos < len(dis) and dis[pos] == drow:
            tf = float(tfs[pos]); dl = float(eng["doclen"][drow])
            c = qwt * wi * tf * (K1 + 1) / (tf + K1 * (1 - B + B * dl / eng["avgdl"]))
            out[w] = (c, wi, tf)
    return out


def frag_contribs(eng, qbag, drow):
    """Total BM25 from char-trigram + prefix views (the 'fragment-collision' channel)."""
    tri = pre = 0.0
    for (view, tok_), qwt in qbag.items():
        if view == "w":
            continue
        t = eng["tid"].get((view, tok_))
        if t is None:
            continue
        wi = float(eng["idf"][t])
        if wi <= 0:
            continue
        s, e = int(eng["ptr"][t]), int(eng["ptr"][t + 1])
        dis = eng["di"][s:e]; tfs = eng["tf"][s:e]
        pos = np.searchsorted(dis, drow)
        if pos < len(dis) and dis[pos] == drow:
            tf = float(tfs[pos]); dl = float(eng["doclen"][drow])
            c = qwt * wi * tf * (K1 + 1) / (tf + K1 * (1 - B + B * dl / eng["avgdl"]))
            if view == "3":
                tri += c
            else:
                pre += c
    return tri, pre


def doc_stats(text):
    ws = toks(text)
    n = len(ws)
    distinct = len(set(ws))
    return n, distinct, (distinct / n if n else 0.0)


def main():
    name = "fiqa"
    corpus, queries, train_q, test_q = U.load(name)
    test_ids = [q for q in test_q if q in queries]
    print(f"{name}: {len(corpus):,} docs | train {len(train_q):,} q | test {len(test_ids):,} q")

    mv = True
    eng = build_csr(corpus, mv)
    corr = train_corridors(eng, corpus, queries, train_q) if train_q else {}
    id2row = eng["id2row"]; cids = eng["cids"]

    # pre-store word tokens / stats per doc lazily via cache
    failures = []
    for qid in test_ids:
        q = queries[qid]
        qbag = doc_bag(q, mv)
        lex = bm25(eng, qbag)
        order = np.argsort(-lex)
        rank1_row = int(order[0])
        rank1_cid = cids[rank1_row]
        gold = {d for d, s in test_q[qid].items() if s > 0 and d in id2row}
        if not gold:
            continue
        if rank1_cid in gold:
            continue  # rank-1 is already gold, no goblin
        # find best gold in top-20
        top20 = order[:20]
        top20_set = set(int(r) for r in top20)
        best_gold_row = None; best_gold_rank = None
        for rk, r in enumerate(top20):
            if cids[int(r)] in gold:
                best_gold_row = int(r); best_gold_rank = rk
                break
        if best_gold_row is None:
            continue  # gold not in top-20: a recall miss, not a ranking goblin
        failures.append((qid, q, qbag, rank1_row, best_gold_row, best_gold_rank, lex))

    print(f"\nRANKING GOBLINS (non-gold@1, gold in top-20): {len(failures)} / {len(test_ids)} test queries\n")

    # how many of ALL failures would FLIP (gold word-score > wrong word-score) if we drop frags?
    flip = 0
    for (qid, q, qbag, wr, gr, grank, lex) in failures:
        ww = sum(c for c, _, _ in word_contribs(eng, qbag, wr).values())
        gw = sum(c for c, _, _ in word_contribs(eng, qbag, gr).values())
        if gw > ww:
            flip += 1
    print(f"   of these, gold's WORD-ONLY score already beats wrong's word-only score in "
          f"{flip}/{len(failures)} ({100*flip/len(failures):.0f}%) -> frag channel caused the inversion\n")

    # analyze up to 30
    SAMPLE = failures[:30]
    qwords_only = lambda b: {k[1] for k in b if k[0] == "w"}

    agg = Counter()
    examples = []
    rows = []
    for (qid, q, qbag, wr, gr, grank, lex) in SAMPLE:
        wrong_text = corpus[cids[wr]]
        gold_text = corpus[cids[gr]]
        qw = qwords_only(qbag)

        # word contributions
        wc_wrong = word_contribs(eng, qbag, wr)
        wc_gold = word_contribs(eng, qbag, gr)
        # fragment (tri/prefix) channel
        tri_w, pre_w = frag_contribs(eng, qbag, wr)
        tri_g, pre_g = frag_contribs(eng, qbag, gr)

        wrong_word_score = sum(c for c, _, _ in wc_wrong.values())
        gold_word_score = sum(c for c, _, _ in wc_gold.values())
        wrong_total = float(lex[wr]); gold_total = float(lex[gr])

        # length / repetitiveness
        lw, dw, rw = doc_stats(wrong_text)
        lg, dg, rg = doc_stats(gold_text)

        # which query words each matched (word view)
        wrong_matched = set(wc_wrong); gold_matched = set(wc_gold)
        gold_has_wrong_lacks = (gold_matched - wrong_matched) & qw
        wrong_has_gold_lacks = (wrong_matched - gold_matched) & qw

        # rare vs hub split of the wrong doc's winning words
        wrong_rare = {w for w, (c, idf, tf) in wc_wrong.items() if idf >= 4.0}
        wrong_hub = {w for w, (c, idf, tf) in wc_wrong.items() if idf < 4.0}
        rare_score = sum(c for w, (c, idf, tf) in wc_wrong.items() if idf >= 4.0)
        hub_score = sum(c for w, (c, idf, tf) in wc_wrong.items() if idf < 4.0)

        # number-token match in wrong doc's winning words
        wrong_num = {w for w in wrong_matched if NUM.match(w)}

        # ---- goblin classification (a failure can hit multiple) ----
        tags = []
        # G1 longer + more repetitive than gold (BM25 length-norm leak)
        if lw > 1.4 * lg and rw < rg:
            tags.append("longer+repetitive")
        # G2 won on the fragment (tri/prefix) channel rather than full words
        frag_w = tri_w + pre_w
        if frag_w > 0.30 * wrong_total and (wrong_word_score < gold_word_score):
            tags.append("fragment-collision")
        # G3 won via common/hub words while gold matched a rarer query word the wrong doc lacks
        rare_gold_missing = {w for w in gold_has_wrong_lacks if U.widf(eng, w) >= 4.0}
        if hub_score > rare_score and rare_gold_missing:
            tags.append("hub-word-win")
        # G4 number-token match drove the win
        if wrong_num and any(wc_wrong[w][1] >= 3.0 for w in wrong_num):
            tags.append("number-match")
        # G5 gold matched MORE distinct query words but lost on score (coverage inversion)
        if len(gold_matched & qw) > len(wrong_matched & qw):
            tags.append("coverage-inversion")
        if not tags:
            tags.append("other")
        for t in tags:
            agg[t] += 1

        rows.append(dict(qid=qid, q=q, tags=tags,
                         lw=lw, lg=lg, rw=rw, rg=rg,
                         wrong_total=wrong_total, gold_total=gold_total, grank=grank,
                         wrong_word=wrong_word_score, gold_word=gold_word_score,
                         frag_w=frag_w, tri_w=tri_w, pre_w=pre_w,
                         hub_score=hub_score, rare_score=rare_score,
                         gold_has_wrong_lacks=gold_has_wrong_lacks,
                         wrong_has_gold_lacks=wrong_has_gold_lacks,
                         wrong_num=wrong_num,
                         wc_wrong=wc_wrong, wc_gold=wc_gold))

    # ---- aggregate report ----
    print(f"Analyzed {len(SAMPLE)} failures. Goblin tag frequency (a failure can have several):")
    for t, c in agg.most_common():
        print(f"   {c:3d} ({100*c/len(SAMPLE):4.0f}%)  {t}")

    # quantify the cheap features across all failures
    arr = lambda f: np.array([f(r) for r in rows], float)
    len_ratio = arr(lambda r: r["lw"] / max(r["lg"], 1))
    rep_wrong = arr(lambda r: r["rw"]); rep_gold = arr(lambda r: r["rg"])
    frag_frac = arr(lambda r: r["frag_w"] / max(r["wrong_total"], 1e-9))
    hub_frac = arr(lambda r: r["hub_score"] / max(r["hub_score"] + r["rare_score"], 1e-9))
    cov_wrong = arr(lambda r: len(r["wc_wrong"]))
    cov_gold = arr(lambda r: len(r["wc_gold"]))

    print("\nCHEAP-FEATURE DISTRIBUTIONS over failures (median):")
    print(f"   wrong/gold length ratio          {np.median(len_ratio):.2f}  (mean {len_ratio.mean():.2f})")
    print(f"   wrong-doc distinct/total ratio    {np.median(rep_wrong):.3f}")
    print(f"   gold-doc  distinct/total ratio    {np.median(rep_gold):.3f}")
    print(f"   frag(tri+pre) frac of wrong score {np.median(frag_frac):.3f}  (mean {frag_frac.mean():.3f})")
    print(f"   hub-word frac of wrong word-score {np.median(hub_frac):.3f}")
    print(f"   #query-words matched: wrong {np.median(cov_wrong):.1f} vs gold {np.median(cov_gold):.1f}")

    # ---- concrete examples (one per dominant pattern) ----
    print("\nCONCRETE EXAMPLES:")
    shown_tags = set()
    nshown = 0
    for r in rows:
        primary = r["tags"][0]
        if primary in shown_tags and nshown >= 3:
            continue
        if primary in shown_tags:
            continue
        shown_tags.add(primary)
        nshown += 1
        wc_w = sorted(r["wc_wrong"].items(), key=lambda kv: -kv[1][0])[:5]
        wc_g = sorted(r["wc_gold"].items(), key=lambda kv: -kv[1][0])[:5]
        print(f"\n  [{','.join(r['tags'])}]  q={r['q'][:80]!r}")
        print(f"    WRONG@1 total={r['wrong_total']:.2f} (word {r['wrong_word']:.2f} + frag {r['frag_w']:.2f}) "
              f"len={r['lw']} distinct_ratio={r['rw']:.2f}")
        print(f"      top words: " + ", ".join(f"{w}(c{c:.2f},idf{idf:.1f})" for w, (c, idf, tf) in wc_w))
        print(f"    GOLD@{r['grank']+1} total={r['gold_total']:.2f} (word {r['gold_word']:.2f}) "
              f"len={r['lg']} distinct_ratio={r['rg']:.2f}")
        print(f"      top words: " + ", ".join(f"{w}(c{c:.2f},idf{idf:.1f})" for w, (c, idf, tf) in wc_g))
        if r["gold_has_wrong_lacks"]:
            gl = sorted(r["gold_has_wrong_lacks"], key=lambda w: -U.widf(eng, w))
            print(f"      gold has, wrong LACKS: " + ", ".join(f"{w}(idf{U.widf(eng,w):.1f})" for w in gl[:6]))
        if r["wrong_has_gold_lacks"]:
            print(f"      wrong has, gold lacks: " + ", ".join(sorted(r['wrong_has_gold_lacks'])[:6]))

    # ---- diagnose the 'other' bucket: what do these have in common? ----
    others = [r for r in rows if r["tags"] == ["other"]]
    if others:
        gap = np.array([r["wrong_total"] - r["gold_total"] for r in others])
        lr = np.array([r["lw"] / max(r["lg"], 1) for r in others])
        fr = np.array([r["frag_w"] / max(r["wrong_total"], 1e-9) for r in others])
        miss = np.array([len(r["gold_has_wrong_lacks"]) for r in others])
        print(f"\n'OTHER' bucket ({len(others)}): median score gap {np.median(gap):.2f}, "
              f"len-ratio {np.median(lr):.2f}, frag-frac {np.median(fr):.2f}, "
              f"#qwords gold-has-wrong-lacks {np.median(miss):.0f} "
              f"-> mostly near-ties where gold lacks NO rare word (true close calls)")

    # ---- rule simulation: what would each rule do to the 30 failures + a normal-doc safety check ----
    print("\nRULE SIMULATION (would a cheap demote FIX the failure without nuking a true rank-1?):")
    simulate_rules(eng, corpus, queries, test_q, test_ids, cids, id2row, mv)


def simulate_rules(eng, corpus, queries, test_q, test_ids, cids, id2row, mv):
    """For each proposed cheap rule, measure: # of goblin failures it FIXES (gold rises
    to #1) and # of currently-correct rank-1 golds it BREAKS (false demote). O(pool)."""
    from unified import doc_bag, bm25

    # cheap per-doc features (computed once for the pool of each query)
    def feats_for_rows(text):
        ws = toks(text); n = len(ws); d = len(set(ws))
        return n, (d / n if n else 1.0)

    # precompute doc length already in eng["doclen"]; need distinct-ratio per doc on demand
    avgdl = eng["avgdl"]

    # rule A: penalize repetitive long docs: score *= rep_ratio**0.5 if len>1.5*avg
    # rule B: cap fragment channel (use word-only score for rerank of top pool)
    # rule C: require query-word coverage -- demote docs matching < gold-matchable words
    # We evaluate by MRR@10 delta on full test set with each rule applied to top-100 pool.

    def base_rank(qid):
        q = queries[qid]; qbag = doc_bag(q, mv)
        lex = bm25(eng, qbag)
        pool = np.argsort(-lex)[:100]
        return q, qbag, lex, pool

    def mrr_of(rank_fn):
        tot = 0.0
        for qid in test_ids:
            r = rank_fn(qid)
            rels = test_q[qid]
            for i, d in enumerate(r[:10]):
                if rels.get(d, 0) > 0:
                    tot += 1.0 / (i + 1); break
        return tot / len(test_ids)

    # cache distinct-ratio per doc row (pool only, but compute lazily across run)
    drcache = {}
    def distinct_ratio(row):
        v = drcache.get(row)
        if v is None:
            ws = toks(corpus[cids[row]]); n = len(ws)
            v = (len(set(ws)) / n) if n else 1.0; drcache[row] = v
        return v

    def baseline(qid):
        q, qbag, lex, pool = base_rank(qid)
        return [cids[r] for r in pool[np.argsort(-lex[pool])][:10]]

    def rule_repetition(qid):
        # score *= distinct_ratio**0.5 for docs longer than 1.5*avgdl
        q, qbag, lex, pool = base_rank(qid)
        sc = lex[pool].copy()
        for i, r in enumerate(pool):
            r = int(r)
            if eng["doclen"][r] > 1.5 * eng["avgdl"]:
                sc[i] *= distinct_ratio(r) ** 0.5
        return [cids[pool[j]] for j in np.argsort(-sc)[:10]]

    def rule_wordonly(qid):
        # rerank pool by WORD-view-only BM25 (kills fragment-collision channel)
        q = queries[qid]
        # word-only bag
        wbag = Counter()
        for w in toks(q):
            wbag[("w", w)] += 1.0
        lexw = bm25(eng, wbag)
        _, _, lex, pool = base_rank(qid)
        return [cids[r] for r in pool[np.argsort(-lexw[pool])][:10]]

    def rule_lenpenalty(qid):
        # softer global length penalty: score *= (avgdl/dl)**0.15 when dl>avgdl
        q, qbag, lex, pool = base_rank(qid)
        sc = lex[pool].copy()
        for i, r in enumerate(pool):
            r = int(r); dl = float(eng["doclen"][r])
            if dl > eng["avgdl"]:
                sc[i] *= (eng["avgdl"] / dl) ** 0.15
        return [cids[pool[j]] for j in np.argsort(-sc)[:10]]

    # word-only score cache per query for coverage / fusion rules
    def word_lex(qid):
        wbag = Counter()
        for w in toks(queries[qid]):
            wbag[("w", w)] += 1.0
        return bm25(eng, wbag), wbag

    def word_coverage(eng, wbag, row):
        """# distinct query WORDS (idf>0) that doc row actually contains."""
        n = 0
        for (view, w), _ in wbag.items():
            if view != "w":
                continue
            t = eng["tid"].get(("w", w))
            if t is None or eng["idf"][t] <= 0:
                continue
            s, e = int(eng["ptr"][t]), int(eng["ptr"][t + 1])
            dis = eng["di"][s:e]
            pos = np.searchsorted(dis, row)
            if pos < len(dis) and dis[pos] == row:
                n += 1
        return n

    def rule_frac_blend(qid):
        # rerank pool by 0.85*wordBM25 + 0.15*fullBM25 (down-weight frag without killing it)
        wl, wbag = word_lex(qid)
        _, _, lex, pool = base_rank(qid)
        sc = 0.85 * wl[pool] + 0.15 * lex[pool]
        return [cids[r] for r in pool[np.argsort(-sc)][:10]]

    def rule_coverage_bonus(qid):
        # additive bonus: + 0.04 * (#distinct query words matched) on WORD score
        wl, wbag = word_lex(qid)
        _, _, lex, pool = base_rank(qid)
        sc = lex[pool].copy()
        lmax = max(float(lex[pool].max()), 1e-9)
        for i, r in enumerate(pool):
            cov = word_coverage(eng, wbag, int(r))
            sc[i] = lex[int(r)] + 0.04 * lmax * cov
        return [cids[pool[j]] for j in np.argsort(-sc)[:10]]

    def rule_combined(qid):
        # word-only rerank + coverage bonus (the two winners stacked)
        wl, wbag = word_lex(qid)
        _, _, lex, pool = base_rank(qid)
        wmax = max(float(wl[pool].max()), 1e-9)
        sc = wl[pool].copy()
        for i, r in enumerate(pool):
            cov = word_coverage(eng, wbag, int(r))
            sc[i] = wl[int(r)] + 0.04 * wmax * cov
        return [cids[pool[j]] for j in np.argsort(-sc)[:10]]

    base = mrr_of(baseline)
    print(f"   baseline MRR@10 (multi-view, top-100 rerank) = {base:.4f}")
    for label, fn in [("rep-penalty distinct**0.5 (len>1.5avg)", rule_repetition),
                      ("word-only rerank (kill frag channel)", rule_wordonly),
                      ("soft len penalty (avgdl/dl)**0.15", rule_lenpenalty),
                      ("frac-blend 0.85*word+0.15*full", rule_frac_blend),
                      ("coverage bonus +0.04*lmax*#qwords", rule_coverage_bonus),
                      ("COMBINED word-only + coverage", rule_combined)]:
        m = mrr_of(fn)
        print(f"   {label:42s} MRR@10 {m:.4f} ({m-base:+.4f})")


if __name__ == "__main__":
    main()
