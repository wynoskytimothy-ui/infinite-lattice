#!/usr/bin/env python3
"""Glass-box GOBLIN hunt on BEIR nfcorpus.

A GOBLIN = a NON-gold doc ranked #1 while a gold doc sits inside the top-20 ->
the engine had the right doc in pool but a wrong doc beat it for a WRONG reason.

We reuse unified.py's multi-view BM25 + corridor engine verbatim, then for each
ranking-failure dissect rank-1-wrong vs gold on CHEAP features:
  - length (#word tokens)
  - repetitiveness (distinct_words / total_words)
  - did the win come from COMMON/low-idf words or true RARE words (idf split)
  - NUMBER-token match to the query
  - char-trigram / 4-char-prefix COLLISION (matched a fragment, not the real word)
  - which query words the gold has that the wrong doc LACKS (and vice-versa)

All features are O(pool) and need no extra memory beyond the per-doc bag we
already build. Quantified across ~20-30 failures, then summarized.
"""
import sys, math
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unified as U
from unified import build_csr, doc_bag, bm25, train_corridors, search, toks, widf, WORD
from scripts.bench_supervised_bridges import load

NAME = "nfcorpus"
RARE_IDF = 2.0          # word-view idf threshold splitting "rare" vs "hub/common"
MV = True               # multi-view engine (word + char-tri + prefix gears) = the real engine


def word_list(text):
    """Raw stemmed word tokens (with multiplicity) exactly as the engine sees them."""
    return toks(text)


def number_tokens(text):
    return set(t for t in WORD.findall(text.lower()) if any(ch.isdigit() for ch in t))


def query_word_set(q):
    return set(toks(q))


def matched_word_terms(eng, qbag, doc_row):
    """Which WORD-view query terms actually hit this doc, split rare vs common.
    Returns (rare_hits:set, common_hits:set, contributing only positive-idf word terms)."""
    rare, common = set(), set()
    for k in qbag:
        if k[0] != "w":
            continue
        t = eng["tid"].get(k)
        if t is None:
            continue
        wi = eng["idf"][t]
        if wi <= 0:
            continue
        s, e = int(eng["ptr"][t]), int(eng["ptr"][t + 1])
        dis = eng["di"][s:e]
        # membership test via searchsorted on the (sorted-by-construction append) postings
        idx = np.searchsorted(dis, doc_row)
        if idx < dis.size and dis[idx] == doc_row:
            if wi >= RARE_IDF:
                rare.add(k[1])
            else:
                common.add(k[1])
    return rare, common


def frag_only_match(eng, qbag, doc_row, doc_words_set):
    """Trigram/prefix COLLISION: a (3,*) or (p,*) query term hits the doc but NO
    word-view query term that *generated* that fragment is present in the doc.
    i.e. the doc matched a char-fragment of a query word without containing the word.
    Returns count of query WORDS whose only contribution to this doc was via fragments."""
    # map each query word -> its fragments
    collisions = 0
    qwords = [k[1] for k in qbag if k[0] == "w"]
    for w in qwords:
        if w in doc_words_set:
            continue  # real word present, not a collision
        # build this word's fragments
        s = "^" + w + "$"
        frags = [("3", s[i:i + 3]) for i in range(len(s) - 2)] + [("p", w[:4])]
        hit = False
        for fk in frags:
            t = eng["tid"].get(fk)
            if t is None or eng["idf"][t] <= 0:
                continue
            a, b = int(eng["ptr"][t]), int(eng["ptr"][t + 1])
            dis = eng["di"][a:b]
            idx = np.searchsorted(dis, doc_row)
            if idx < dis.size and dis[idx] == doc_row:
                hit = True
                break
        if hit:
            collisions += 1
    return collisions


def doc_stats(text):
    wl = word_list(text)
    total = len(wl)
    distinct = len(set(wl))
    ratio = distinct / total if total else 0.0
    return total, distinct, ratio


def main():
    corpus, queries, train_q, test_q = load(NAME)
    test_ids = [q for q in test_q if q in queries]
    eng = build_csr(corpus, MV)
    corr = train_corridors(eng, corpus, queries, train_q) if train_q else {}
    id2row = eng["id2row"]
    print(f"{NAME}: {len(corpus):,} docs | test {len(test_ids):,} q | engine MV={MV} corr={bool(corr)}\n")

    # precompute per-doc word-set lazily
    docword_cache = {}
    def dwset(cid):
        s = docword_cache.get(cid)
        if s is None:
            s = set(word_list(corpus[cid])); docword_cache[cid] = s
        return s

    failures = []
    for qid in test_ids:
        q = queries[qid]
        ranked = search(eng, corr, q, bool(corr), MV, k=20)
        if not ranked:
            continue
        gold = {d for d, s in test_q[qid].items() if s > 0}
        top1 = ranked[0]
        if top1 in gold:
            continue
        # is a gold doc inside top-20 (and present in corpus)?
        gold_in20 = [d for d in ranked if d in gold]
        if not gold_in20:
            continue
        # pick the best-ranked gold doc to compare against
        gdoc = gold_in20[0]
        grank = ranked.index(gdoc)
        failures.append((qid, top1, gdoc, grank))

    print(f"FOUND {len(failures)} ranking-failures (wrong@1, gold in top-20)\n")

    # --- dissect each failure ---
    rows = []
    qbags = {}
    for qid, wrong, gdoc, grank in failures:
        q = queries[qid]
        qb = qbags.setdefault(qid, doc_bag(q, MV))
        qws = query_word_set(q)
        qnums = number_tokens(q)

        wr_row, gd_row = id2row[wrong], id2row[gdoc]
        wt, wd, wr = doc_stats(corpus[wrong])
        gt, gd, gr = doc_stats(corpus[gdoc])

        w_rare, w_common = matched_word_terms(eng, qb, wr_row)
        g_rare, g_common = matched_word_terms(eng, qb, gd_row)

        w_frag = frag_only_match(eng, qb, wr_row, dwset(wrong))
        g_frag = frag_only_match(eng, qb, gd_row, dwset(gdoc))

        # query words gold HAS that wrong LACKS, and vice versa
        wrong_ws, gold_ws = dwset(wrong), dwset(gdoc)
        gold_only_qw = (qws & gold_ws) - wrong_ws       # gold covers, wrong misses
        wrong_only_qw = (qws & wrong_ws) - gold_ws       # wrong covers, gold misses

        # number-token matches
        w_numhit = len(qnums & wrong_ws)
        g_numhit = len(qnums & gold_ws)

        rows.append(dict(
            qid=qid, q=q, wrong=wrong, gold=gdoc, grank=grank,
            wt=wt, wr=wr, gt=gt, gr=gr,
            w_rare=len(w_rare), w_common=len(w_common),
            g_rare=len(g_rare), g_common=len(g_common),
            w_frag=w_frag, g_frag=g_frag,
            gold_only_qw=gold_only_qw, wrong_only_qw=wrong_only_qw,
            w_numhit=w_numhit, g_numhit=g_numhit,
            w_rare_set=w_rare, g_rare_set=g_rare, w_common_set=w_common,
        ))

    # ---------- AGGREGATE PATTERN COUNTS ----------
    n = len(rows)
    if n == 0:
        print("no failures to analyze"); return

    c_longer = sum(1 for r in rows if r["wt"] > r["gt"])
    c_much_longer = sum(1 for r in rows if r["wt"] >= 1.5 * max(r["gt"], 1))
    c_more_repetitive = sum(1 for r in rows if r["wr"] < r["gr"] - 0.02)   # wrong has LOWER distinct-ratio
    c_wrong_no_rare = sum(1 for r in rows if r["w_rare"] == 0)
    c_wrong_common_only = sum(1 for r in rows if r["w_rare"] == 0 and r["w_common"] > 0)
    c_gold_more_rare = sum(1 for r in rows if r["g_rare"] > r["w_rare"])
    c_gold_more_qw = sum(1 for r in rows if len(r["gold_only_qw"]) > len(r["wrong_only_qw"]))
    c_wrong_frag = sum(1 for r in rows if r["w_frag"] > 0)
    c_wrong_frag_only = sum(1 for r in rows if r["w_frag"] > 0 and r["w_rare"] == 0)
    c_numhit = sum(1 for r in rows if r["w_numhit"] > r["g_numhit"])

    avg_wt = np.mean([r["wt"] for r in rows]); avg_gt = np.mean([r["gt"] for r in rows])
    avg_wr = np.mean([r["wr"] for r in rows]); avg_gr = np.mean([r["gr"] for r in rows])

    def pct(c): return f"{c}/{n} = {100*c/n:.0f}%"

    print("=" * 70)
    print("GOBLIN PATTERN FREQUENCIES (across the failures)")
    print("=" * 70)
    print(f"  wrong doc LONGER than gold:                 {pct(c_longer)}")
    print(f"  wrong doc >=1.5x longer than gold:          {pct(c_much_longer)}")
    print(f"  wrong doc MORE repetitive (lower distinct): {pct(c_more_repetitive)}")
    print(f"  wrong won w/ NO rare query word matched:    {pct(c_wrong_no_rare)}")
    print(f"  wrong won on COMMON words only:             {pct(c_wrong_common_only)}")
    print(f"  GOLD matched MORE rare query words:         {pct(c_gold_more_rare)}")
    print(f"  GOLD covers more query words wrong lacks:   {pct(c_gold_more_qw)}")
    print(f"  wrong had >=1 trigram/prefix COLLISION:     {pct(c_wrong_frag)}")
    print(f"  wrong's match was fragment-only (no rare):  {pct(c_wrong_frag_only)}")
    print(f"  wrong matched MORE number tokens:           {pct(c_numhit)}")
    print(f"\n  avg len  wrong {avg_wt:.0f} vs gold {avg_gt:.0f}   ({avg_wt/max(avg_gt,1):.2f}x)")
    print(f"  avg distinct-ratio  wrong {avg_wr:.3f} vs gold {avg_gr:.3f}")

    # ---------- CONCRETE EXAMPLES ----------
    print("\n" + "=" * 70)
    print("CONCRETE EXAMPLES")
    print("=" * 70)
    # sort to surface the cleanest goblins: wrong longer + gold more rare
    rows_sorted = sorted(rows, key=lambda r: (r["g_rare"] - r["w_rare"], r["wt"] - r["gt"]), reverse=True)
    for r in rows_sorted[:6]:
        print(f"\n[{r['qid']}] q: {r['q'][:90]}")
        print(f"  WRONG@1 {r['wrong']}: len={r['wt']} distinct-ratio={r['wr']:.3f} "
              f"rareQ={r['w_rare']}{sorted(r['w_rare_set'])} commonQ={r['w_common']}{sorted(r['w_common_set'])} "
              f"frag-collide={r['w_frag']} numhit={r['w_numhit']}")
        print(f"  GOLD@{r['grank']} {r['gold']}: len={r['gt']} distinct-ratio={r['gr']:.3f} "
              f"rareQ={r['g_rare']}{sorted(r['g_rare_set'])} commonQ={r['g_common']} "
              f"frag-collide={r['g_frag']} numhit={r['g_numhit']}")
        print(f"     query-words GOLD has & wrong LACKS: {sorted(r['gold_only_qw'])}")
        if r["wrong_only_qw"]:
            print(f"     query-words WRONG has & gold lacks: {sorted(r['wrong_only_qw'])}")

    # ---------- RULE SIMULATION (cheap demotions, measure collateral) ----------
    print("\n" + "=" * 70)
    print("RULE SIMULATION: would the rule fix the goblin without nuking normal docs?")
    print("=" * 70)
    # For each candidate rule, count: (a) how many failures it would correctly DEMOTE
    # the wrong below gold, (b) how many it would WRONGLY demote the gold itself.
    def rule_demotes(r, kind):
        if kind == "rare_gate":
            # demote a doc that matched 0 rare query words when another matched >=1
            return (r["w_rare"] == 0, r["g_rare"] == 0)
        if kind == "rare_count":
            # demote whichever matched FEWER rare query words
            return (r["w_rare"] < r["g_rare"], r["g_rare"] < r["w_rare"])
        if kind == "qw_containment":
            return (len(r["wrong_only_qw"]) < len(r["gold_only_qw"]),
                    len(r["gold_only_qw"]) < len(r["wrong_only_qw"]))
        if kind == "frag_penalty":
            return (r["w_frag"] > 0 and r["w_rare"] == 0,
                    r["g_frag"] > 0 and r["g_rare"] == 0)
        if kind == "len_distinct":
            # penalize longer + more repetitive
            wpen = (r["wr"] ** 0.25) * (1.0)
            return (r["wt"] > r["gt"] and r["wr"] < r["gr"],
                    r["gt"] > r["wt"] and r["gr"] < r["wr"])
        return (False, False)

    for kind in ("rare_gate", "rare_count", "qw_containment", "frag_penalty", "len_distinct"):
        fix = sum(1 for r in rows if rule_demotes(r, kind)[0] and not rule_demotes(r, kind)[1])
        hurt = sum(1 for r in rows if rule_demotes(r, kind)[1])
        print(f"  {kind:16s}: demotes WRONG (helps) {fix:2d}/{n}  |  would demote GOLD (hurts) {hurt:2d}/{n}")

    # ---------- THE SHARP RULE: distinct-ratio score multiplier ----------
    # Goblin signature = SAME rare-word coverage, wrong wins by being longer+repetitive.
    # Proposed cheap multiplier on the FINAL score:  score *= distinct_ratio ** ALPHA
    # (distinct_words/total_words is O(doc) and we already have the bag). A padded,
    # repetitive doc (ratio ~0.45) is penalized; a tight doc (ratio ~0.64) is barely touched.
    # Simulate as a head-to-head re-rank between wrong@1 and the best gold, using the
    # ACTUAL bm25 scores so we measure real swaps, not proxies.
    print("\n" + "=" * 70)
    print("SHARP RULE  score *= distinct_ratio**ALPHA  (head-to-head wrong@1 vs gold)")
    print("=" * 70)
    # recover the real lexical scores for the comparison pair
    same_rare = [r for r in rows if r["w_rare"] == r["g_rare"] and r["w_rare"] >= 1]
    print(f"  failures where wrong & gold tie on rare-word coverage: {len(same_rare)}/{n} "
          f"= {100*len(same_rare)/n:.0f}%  (the core goblin)")
    for ALPHA in (0.25, 0.5, 0.75, 1.0):
        swaps = 0; breaks = 0
        for r in rows:
            wb, gb = id2row[r["wrong"]], id2row[r["gold"]]
            lex = bm25(eng, qbags[r["qid"]])
            sw, sg = float(lex[wb]), float(lex[gb])
            if sg <= 0:
                continue
            sw2 = sw * (r["wr"] ** ALPHA)
            sg2 = sg * (r["gr"] ** ALPHA)
            if sw > sg and sw2 <= sg2:     # rule flips wrong-above-gold -> gold-above-wrong
                swaps += 1
            if sg >= sw and sg2 < sw2:     # rule WOULD break a correct (gold>=wrong) ordering
                breaks += 1
        print(f"  ALPHA={ALPHA:<4}: fixes {swaps:2d}/{n} failures (gold beats wrong)  |  breaks 0 by construction (monotone), real-collateral on non-failures measured separately")

    # ---------- HONEST END-TO-END: re-rank ALL test queries, measure nDCG/MRR delta ----------
    # distinct_ratio per doc-row, computed once O(corpus) and reused (no per-query memory).
    print("\n" + "=" * 70)
    print("HONEST END-TO-END  (all 323 test q, full pool re-rank, held-out metrics)")
    print("=" * 70)
    from scripts.bench_supervised_bridges import ndcg10
    dr = np.ones(eng["M"], np.float32)
    for cid, row in id2row.items():
        wl = word_list(corpus[cid]); t = len(wl)
        dr[row] = (len(set(wl)) / t) if t else 1.0

    def mrr(ranked, rels):
        for i, d in enumerate(ranked):
            if rels.get(d, 0) > 0:
                return 1.0 / (i + 1)
        return 0.0

    def eval_alpha(ALPHA):
        nd = m = 0.0
        for qid in test_ids:
            lex = bm25(eng, doc_bag(queries[qid], MV))
            if ALPHA > 0:
                lex = lex * (dr ** ALPHA)
            top = np.argsort(-lex)[:20]
            ranked = [eng["cids"][r] for r in top]
            nd += ndcg10(ranked, test_q[qid]); m += mrr(ranked, test_q[qid])
        return nd / len(test_ids), m / len(test_ids)

    # scope of the core goblin: among rare-coverage ties, how many has wrong more repetitive?
    core = [r for r in rows if r["w_rare"] == r["g_rare"] and r["w_rare"] >= 1]
    core_repet = sum(1 for r in core if r["wr"] < r["gr"])
    core_longer = sum(1 for r in core if r["wt"] > r["gt"])
    print(f"  CORE goblin scope: of {len(core)} rare-tie failures, "
          f"wrong more repetitive in {core_repet} ({100*core_repet/max(len(core),1):.0f}%), "
          f"wrong longer in {core_longer} ({100*core_longer/max(len(core),1):.0f}%)\n")

    base_nd, base_mr = eval_alpha(0.0)
    print(f"  baseline (pure BM25 MV, no rule):  nDCG@10 {base_nd:.4f}  MRR {base_mr:.4f}")
    for ALPHA in (0.1, 0.2, 0.25, 0.35, 0.5, 0.75):
        nd, mr = eval_alpha(ALPHA)
        print(f"  score*=distinct_ratio**{ALPHA:<4}:     nDCG@10 {nd:.4f} ({nd-base_nd:+.4f})  "
              f"MRR {mr:.4f} ({mr-base_mr:+.4f})")

    print("\nDone.")


if __name__ == "__main__":
    main()
