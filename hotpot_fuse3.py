#!/usr/bin/env python3
"""Realize the complementarity ceiling with a 3rd slot, on HotpotQA distractor bridge Qs.

hotpot_fuse.py found meet & dense-to-question reach NEARLY-INDEPENDENT bridge passages (overlap 0.19,
union ceiling 0.6629 vs meet-alone 0.4642 = +43% headroom), BUT a budget-2 RRF blend couldn't capture
it -- 2 slots can't hold anchor + meet-bridge + dense-bridge (3 things), so blending displaces the
better meet bridge. Feeding a reader 3 passages is the standard multi-hop setup, so test budget-3:

  bm25 top-3                       reference
  dense top-3                      neural single-hop, 3 paras
  meet-chain +1 (anchor + 2 meet)  the lattice spending its 3rd slot on another meet bridge
  FUSED (anchor + meet + dense)    the union chain: spend the 3rd slot on the INDEPENDENT dense route
All return exactly 3 distinct paras -> support-fact R@3 + 2nd-hop bridge R@3 apples-to-apples. If FUSED
beats both meet@3 and dense@3 on the bridge, the complementary fusion works at the practical budget.
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

    ret = {k: [0.0, 0.0] for k in ("bm25", "dense", "meet", "fused")}
    meet2 = 0.0   # budget-2 meet-chain bridge, for reference
    nb = 0; small = 0
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
        order_d = [int(i) for i in np.argsort(-dv)]

        gold_idx = [titles.index(g) for g in gold]
        harder = max(gold_idx, key=lambda gi: order_sh.index(gi))
        htitle = titles[harder]

        def score3(sel):
            tt = set(titles[i] for i in sel)
            return len(gold & tt) / 2.0, (1.0 if htitle in tt else 0.0)

        a = order_sh[0]
        bterms = Counter()
        for w in paras[a]:
            if w not in qset and idf.get(w, 0) >= RARE:
                bterms[w] += idf[w]
        meet = np.array([sum(w for t2, w in bterms.items() if t2 in paras[i]) for i in range(n)])
        meet[a] = -1e9
        order_meet = [int(i) for i in np.argsort(-meet)]   # a is last
        m1, m2 = order_meet[0], order_meet[1]

        # budget-2 meet-chain reference
        meet2 += (1.0 if htitle in {titles[a], titles[m1]} else 0.0)

        # budget-3 retrievers (exactly 3 distinct paras each)
        if n < 3:
            small += 1
        bm25_3 = set(order_sh[:3])
        dense_3 = set(order_d[:3])
        meet_3 = {a, m1, m2}
        d1 = next((i for i in order_d if i not in (a, m1)), None)   # best dense para not already in chain
        fused_3 = {a, m1} if d1 is None else {a, m1, d1}

        for k, sel in (("bm25", bm25_3), ("dense", dense_3), ("meet", meet_3), ("fused", fused_3)):
            r2, h2 = score3(sel); ret[k][0] += r2; ret[k][1] += h2

    print(f"\nBUDGET-3 COMPLEMENTARY FUSION on HotpotQA distractor (bridge Qs, n={nb}; {small} had <3 paras)")
    print(f"  reference: budget-2 meet-chain bridge R@2 = {meet2/nb:.4f}; budget-4 union ceiling = 0.6629\n")
    print(f"  {'retriever (3 paras)':<36}{'support-fact R@3':>18}{'2nd-hop bridge R@3':>20}")
    for k, name in (("bm25", "bm25 top-3"),
                    ("dense", "dense top-3"),
                    ("meet", "meet-chain +1 (anchor + 2 meet)"),
                    ("fused", "FUSED (anchor + meet + dense)")):
        print(f"  {name:<36}{ret[k][0]/nb:>18.4f}{ret[k][1]/nb:>20.4f}")
    print(f"\n  dense model: {mname}")
    print(f"  FUSED > meet@3 AND > dense@3 on bridge => the 3rd slot spent on the INDEPENDENT dense route")
    print(f"  realizes the complementarity; the lattice meet and neural embedding cover different bridges.")


if __name__ == "__main__":
    main()
