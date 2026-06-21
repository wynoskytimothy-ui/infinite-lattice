#!/usr/bin/env python3
"""The genuine multi-hop test for the lattice's corridor traversal, on HotpotQA distractor-dev.

Each BRIDGE question has 10 paragraphs; 2 are the gold supporting facts forming a reasoning chain:
  question --(shares words)--> para A (the bridge entity) --(bridge entity)--> para B (the answer).
para B shares FEW words with the question -> single-hop retrieval misses it. The lattice's meet/
corridor connects A's rare bridge-entity terms to B. Test: does corridor TRAVERSAL (hop-1 anchor ->
its rare bridge terms -> hop-2 para sharing them) lift the 2nd-hop gold para that single-hop misses?

  single-hop:  rank 10 paras by BM25(question, para)
  multi-hop:   + bridge score = idf-mass of rare terms a para shares with the hop-1 anchor paras
  control:     same, but bridge via RANDOM (non-rare) shared terms -> isolates the rare-entity bridge
Metric: supporting-fact recall@2 (both gold paras in top-2) + 2nd-hop (harder gold) recall, on bridge Qs.
"""
import re, math, time, random
from collections import Counter
import numpy as np
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
from stem_safe import safe

WORD = re.compile(r"[a-z0-9]+")
K1, B = 0.9, 0.4
RARE = 5.0          # idf gate for "bridge entity" terms (rare in the para corpus)
LAM = 1.0           # weight of the hop-2 bridge score
HOP1 = 2            # # anchor paras to traverse from

_st = {}
def st(w):
    r = _st.get(w)
    if r is None:
        r = safe(w); _st[w] = r
    return r
def toks(s):
    return [st(w) for w in WORD.findall(s.lower())]


def main():
    p = hf_hub_download("hotpot_qa", "distractor/validation-00000-of-00001.parquet", repo_type="dataset")
    t = pq.read_table(p).to_pandas()
    print(f"  loaded HotpotQA distractor-dev: {len(t)} questions", flush=True)

    # corpus df/idf over all unique paragraphs (by title)
    t0 = time.time()
    para_tok = {}; df = Counter()
    for ctx in t["context"]:
        for title, sents in zip(ctx["title"], ctx["sentences"]):
            if title not in para_tok:
                tk = toks(" ".join(sents))
                para_tok[title] = Counter(tk)
                for w in set(tk):
                    df[w] += 1
    N = len(para_tok)
    idf = {w: math.log((N - c + 0.5) / (c + 0.5) + 1.0) for w, c in df.items()}
    avgdl = sum(sum(c.values()) for c in para_tok.values()) / N
    print(f"  corpus: {N} unique paras, {len(df)} terms (idf built {time.time()-t0:.0f}s)", flush=True)

    def bm25(qset, ctr):
        dl = sum(ctr.values())
        return sum(idf.get(w, 0) * ctr.get(w, 0) * (K1 + 1) / (ctr.get(w, 0) + K1 * (1 - B + B * dl / avgdl)) for w in qset)

    rng = random.Random(42)
    res = {k: [0.0, 0.0] for k in ("single", "multi", "ctrl", "chain")}   # [recall@2 sum, 2nd-hop-recall sum]
    nb = 0
    for _, r in t.iterrows():
        if r["type"] != "bridge":
            continue
        ctx = r["context"]
        titles = list(ctx["title"]); paras = [para_tok[ti] for ti in titles]
        gold = set(r["supporting_facts"]["title"]) & set(titles)
        if len(gold) < 2:
            continue
        nb += 1
        qset = set(toks(r["question"]))
        sh = np.array([bm25(qset, c) for c in paras])
        order_sh = list(np.argsort(-sh))
        # hop-1 anchors = top-HOP1 by single-hop; gather their rare bridge terms (rare, not in question)
        anchors = order_sh[:HOP1]
        bridge_terms = Counter()
        rand_terms = Counter()
        for ai in anchors:
            for w, c in paras[ai].items():
                if w in qset:
                    continue
                if idf.get(w, 0) >= RARE:
                    bridge_terms[w] += idf[w]
                else:
                    rand_terms[w] += idf.get(w, 0)
        # hop-2 score: idf-mass of bridge terms a para shares with the anchors (exclude anchors themselves)
        def hop2(termset):
            return np.array([sum(w for t2, w in termset.items() if t2 in paras[i]) if i not in anchors else 0.0
                             for i in range(len(paras))])
        multi = sh + LAM * hop2(bridge_terms)
        ctrl = sh + LAM * hop2(rand_terms)

        def recall(order):
            top2 = set(titles[i] for i in order[:2])
            r2 = len(gold & top2) / 2.0
            # 2nd-hop = the gold para single-hop ranks LOWER; did this ranking get it into top-2?
            gold_idx = [titles.index(g) for g in gold]
            harder = max(gold_idx, key=lambda gi: order_sh.index(gi))   # the one single-hop buries
            hop2_hit = 1.0 if titles[harder] in top2 else 0.0
            return r2, hop2_hit
        for k, sc in (("single", sh), ("multi", multi), ("ctrl", ctrl)):
            order = list(np.argsort(-sc))
            r2, h2 = recall(order)
            res[k][0] += r2; res[k][1] += h2
        # CHAIN: keep hop-1 anchor (best single-hop) + add hop-2 bridge para (best bridge from anchor)
        a = order_sh[0]
        bterms = Counter()
        for w, c in paras[a].items():
            if w not in qset and idf.get(w, 0) >= RARE:
                bterms[w] += idf[w]
        h2v = np.array([(sum(w for t2, w in bterms.items() if t2 in paras[i]) if i != a else -1.0)
                        for i in range(len(paras))])
        bpar = int(np.argmax(h2v))
        chain = {titles[a], titles[bpar]}
        gold_idx = [titles.index(g) for g in gold]
        harder = max(gold_idx, key=lambda gi: order_sh.index(gi))
        res["chain"][0] += len(gold & chain) / 2.0
        res["chain"][1] += 1.0 if titles[harder] in chain else 0.0

    print(f"\nMULTI-HOP TRAVERSAL on HotpotQA distractor (bridge questions, n={nb})\n")
    print(f"   {'retriever':<28}{'support-fact R@2':>18}{'2nd-hop para R@2':>18}")
    for k, name in (("single", "single-hop BM25"), ("multi", "+ corridor (re-rank)"),
                    ("chain", "CHAIN (anchor + bridge)"), ("ctrl", "+ random-term control")):
        print(f"   {name:<28}{res[k][0]/nb:>18.4f}{res[k][1]/nb:>18.4f}")
    print(f"\n   multi > single on 2nd-hop R@2, AND multi > control = corridor traversal reaches the")
    print(f"   bridge passage single-hop misses, via the rare bridge-entity link -- the multi-hop win.")


if __name__ == "__main__":
    main()
