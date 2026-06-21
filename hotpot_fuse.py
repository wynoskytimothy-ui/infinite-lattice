#!/usr/bin/env python3
"""Complementary fusion: meet-chain UNION dense-to-question, on HotpotQA distractor bridge Qs.

hotpot_iso.py showed the bridge-step fusion (meet + dense-sim-TO-ANCHOR) HURT, because dense-sim-to-
anchor is the bad signal (0.30, fooled by distractors). But single-hop dense-to-QUESTION reaches the
bridge 0.39 by a genuinely DIFFERENT route than the meet (q-semantics, not anchor-meet). This is the
BM25-union-lattice analogue: two INDEPENDENT signals -> fuse should help IFF they reach different golds.

Two things measured:
  (1) COMPLEMENTARITY ceiling -- of the bridge passages, how many does the meet reach that dense MISSES,
      and vice versa? union = the ceiling any fusion could hit. If meet-only and dense-only are both big,
      they're complementary and there's headroom; if not, fusion can't help.
  (2) BUDGET-2 FUSED retriever -- RRF (no weights) of the lattice-chain ranking (anchor=BM25-best, then
      meet-from-anchor) and the dense-to-question ranking; top-2. Apples-to-apples vs each alone at top-2.
"""
import re, math
from collections import Counter
import numpy as np
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
from stem_safe import safe

WORD = re.compile(r"[a-z0-9]+")
K1, B = 0.9, 0.4
RARE = 5.0
RRF_K = 60.0

_st = {}
def st(w):
    r = _st.get(w)
    if r is None:
        r = safe(w); _st[w] = r
    return r
def toks(s):
    return [st(w) for w in WORD.findall(s.lower())]

def rank_of(score):
    order = np.argsort(-score)
    rank = np.empty(len(order), dtype=np.int64)
    rank[order] = np.arange(len(order))
    return rank


def main():
    p = hf_hub_download("hotpot_qa", "distractor/validation-00000-of-00001.parquet", repo_type="dataset")
    t = pq.read_table(p).to_pandas()
    print(f"  loaded HotpotQA distractor-dev: {len(t)} questions", flush=True)

    para_tok = {}; para_text = {}; df = Counter()
    for ctx in t["context"]:
        for title, sents in zip(ctx["title"], ctx["sentences"]):
            if title not in para_tok:
                txt = " ".join(sents)
                tk = toks(txt)
                para_tok[title] = Counter(tk); para_text[title] = txt
                for w in set(tk):
                    df[w] += 1
    N = len(para_tok)
    idf = {w: math.log((N - c + 0.5) / (c + 0.5) + 1.0) for w, c in df.items()}
    avgdl = sum(sum(c.values()) for c in para_tok.values()) / N
    print(f"  corpus: {N} unique paras, {len(df)} terms", flush=True)

    def bm25(qset, ctr):
        dl = sum(ctr.values())
        return sum(idf.get(w, 0) * ctr.get(w, 0) * (K1 + 1) / (ctr.get(w, 0) + K1 * (1 - B + B * dl / avgdl)) for w in qset)

    from sentence_transformers import SentenceTransformer
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        enc = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1", device=dev); mname = "multi-qa-MiniLM-L6-cos-v1"
    except Exception:
        enc = SentenceTransformer("all-MiniLM-L6-v2", device=dev); mname = "all-MiniLM-L6-v2"
    print(f"  dense encoder: {mname} on {dev}", flush=True)

    titles_all = list(para_text.keys())
    pe = enc.encode([para_text[ti] for ti in titles_all], batch_size=256, convert_to_numpy=True,
                    normalize_embeddings=True, show_progress_bar=False)
    para_emb = {ti: pe[i] for i, ti in enumerate(titles_all)}

    rows = []
    for _, r in t.iterrows():
        if r["type"] != "bridge":
            continue
        titles = list(r["context"]["title"])
        gold = set(r["supporting_facts"]["title"]) & set(titles)
        if len(gold) < 2:
            continue
        rows.append(r)
    qe = enc.encode([r["question"] for r in rows], batch_size=256, convert_to_numpy=True,
                    normalize_embeddings=True, show_progress_bar=False)
    print(f"  embedded {len(titles_all)} paras + {len(rows)} bridge Qs; scoring...", flush=True)

    ret = {k: [0.0, 0.0] for k in ("meet", "dense", "fused")}   # support-fact R@2, 2nd-hop bridge R@2
    comp = Counter()   # both / meet_only / dense_only / neither  (on the 2nd-hop bridge passage)
    nb = 0
    for qi, r in enumerate(rows):
        ctx = r["context"]
        titles = list(ctx["title"]); paras = [para_tok[ti] for ti in titles]
        gold = set(r["supporting_facts"]["title"]) & set(titles)
        nb += 1
        n = len(titles)
        qset = set(toks(r["question"]))
        sh = np.array([bm25(qset, c) for c in paras])
        order_sh = list(np.argsort(-sh))
        pv = np.array([para_emb[ti] for ti in titles])
        dv = pv @ qe[qi]

        gold_idx = [titles.index(g) for g in gold]
        harder = max(gold_idx, key=lambda gi: order_sh.index(gi))
        htitle = titles[harder]

        def score2(sel):
            top2t = set(titles[i] for i in sel)
            return len(gold & top2t) / 2.0, (1.0 if htitle in top2t else 0.0)

        # meet-chain: anchor = BM25-best; bridge = argmax meet-from-anchor
        a = order_sh[0]
        bterms = Counter()
        for w in paras[a]:
            if w not in qset and idf.get(w, 0) >= RARE:
                bterms[w] += idf[w]
        meet = np.array([sum(w for t2, w in bterms.items() if t2 in paras[i]) for i in range(n)])
        meet[a] = -1e9
        b_meet = int(np.argmax(meet))
        meet_set = {a, b_meet}
        r2, h2 = score2(meet_set); ret["meet"][0] += r2; ret["meet"][1] += h2

        # dense single-hop (to question)
        dense_top2 = set(int(i) for i in np.argsort(-dv)[:2])
        r2, h2 = score2(dense_top2); ret["dense"][0] += r2; ret["dense"][1] += h2

        # complementarity on the bridge passage
        mh = htitle in {titles[i] for i in meet_set}
        dh = htitle in {titles[i] for i in dense_top2}
        comp["both" if (mh and dh) else "meet_only" if mh else "dense_only" if dh else "neither"] += 1

        # budget-2 FUSED: RRF(lattice-chain ranking, dense-to-question ranking), top-2
        order_L = [a] + sorted([i for i in range(n) if i != a], key=lambda i: -meet[i])
        rank_L = np.empty(n, dtype=np.int64)
        for pos, i in enumerate(order_L):
            rank_L[i] = pos
        rrf = 1.0 / (RRF_K + rank_L) + 1.0 / (RRF_K + rank_of(dv))
        fused_top2 = set(int(i) for i in np.argsort(-rrf)[:2])
        r2, h2 = score2(fused_top2); ret["fused"][0] += r2; ret["fused"][1] += h2

    print(f"\nCOMPLEMENTARY FUSION on HotpotQA distractor (bridge Qs, n={nb})\n")
    print(f"  COMPLEMENTARITY on the 2nd-hop bridge passage (who reaches it):")
    print(f"    meet reaches bridge:        {ret['meet'][1]/nb:.4f}")
    print(f"    dense reaches bridge:       {ret['dense'][1]/nb:.4f}")
    print(f"    both reach it:              {comp['both']/nb:.4f}")
    print(f"    meet ONLY (dense misses):   {comp['meet_only']/nb:.4f}   <- meet's unique reach")
    print(f"    dense ONLY (meet misses):   {comp['dense_only']/nb:.4f}   <- dense's unique reach")
    print(f"    neither:                    {comp['neither']/nb:.4f}")
    print(f"    UNION (either):             {(comp['both']+comp['meet_only']+comp['dense_only'])/nb:.4f}   <- fusion ceiling")
    print(f"\n  {'retriever (budget 2)':<34}{'support-fact R@2':>18}{'2nd-hop bridge R@2':>20}")
    for k, name in (("meet", "meet-chain"), ("dense", "dense single-hop"), ("fused", "RRF-fused (meet + dense-q)")):
        print(f"  {name:<34}{ret[k][0]/nb:>18.4f}{ret[k][1]/nb:>20.4f}")
    print(f"\n  dense model: {mname}")
    print(f"  meet_only & dense_only both large => complementary, headroom; UNION >> meet => budget-2 can't")
    print(f"  hold both, a 3rd slot would. RRF-fused > meet => the independent dense route lifts the headline.")


if __name__ == "__main__":
    main()
