#!/usr/bin/env python3
"""Airtight isolation of the BRIDGE STEP + lattice/dense fusion, on HotpotQA distractor bridge Qs.

hotpot_dense.py showed the lattice CHAIN beats dense, but chain-vs-chain let each system pick its OWN
anchor -> not a pure isolation of the bridge step. Here we FIX the anchor (= best single-hop BM25) for
BOTH bridge methods, so the ONLY thing that varies is how the 2nd para (the bridge) is chosen:

  fixed-anchor + dense-sim bridge   bridge = para most dense-cosine-similar to the anchor
  fixed-anchor + rare-entity meet   bridge = para sharing the most rare-idf bridge terms w/ the anchor

Same anchor, same fixed target (the gold BM25 buries) -> a clean meet-vs-similarity test for the bridge.
Then FUSION (budget 2, parameter-light RRF, no weights): anchor = RRF(BM25, dense) single-hop; bridge =
RRF(meet-from-anchor, dense-sim-to-anchor). Tests whether the precise meet and diffuse-but-semantic
dense reach DIFFERENT bridge paras and combine. Every method returns exactly 2 paras -> apples-to-apples.
"""
import re, math, time
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

    t0 = time.time()
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
    print(f"  corpus: {N} unique paras, {len(df)} terms (idf {time.time()-t0:.0f}s)", flush=True)

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

    keys = ("single", "dense", "fa_sim", "fa_meet", "fused")
    res = {k: [0.0, 0.0] for k in keys}
    nb = 0
    for qi, r in enumerate(rows):
        ctx = r["context"]
        titles = list(ctx["title"]); paras = [para_tok[ti] for ti in titles]
        gold = set(r["supporting_facts"]["title"]) & set(titles)
        nb += 1
        qset = set(toks(r["question"]))
        sh = np.array([bm25(qset, c) for c in paras])
        order_sh = list(np.argsort(-sh))
        pv = np.array([para_emb[ti] for ti in titles])
        dv = pv @ qe[qi]
        order_d = list(np.argsort(-dv))

        gold_idx = [titles.index(g) for g in gold]
        harder = max(gold_idx, key=lambda gi: order_sh.index(gi))   # fixed target across all methods

        def score2(sel):
            top2 = sel if isinstance(sel, set) else set(sel[:2])
            top2t = set(titles[i] for i in top2)
            return len(gold & top2t) / 2.0, (1.0 if titles[harder] in top2t else 0.0)

        def meet_from(anchor):
            bterms = Counter()
            for w in paras[anchor]:
                if w not in qset and idf.get(w, 0) >= RARE:
                    bterms[w] += idf[w]
            return np.array([sum(w for t2, w in bterms.items() if t2 in paras[i]) for i in range(len(paras))])

        # baselines
        for k, o in (("single", order_sh), ("dense", order_d)):
            r2, h2 = score2(o); res[k][0] += r2; res[k][1] += h2

        # FIXED anchor = best single-hop BM25; vary ONLY the bridge step
        a = order_sh[0]
        sim = pv @ pv[a]; sim[a] = -1e9
        b_sim = int(np.argmax(sim))
        r2, h2 = score2({a, b_sim}); res["fa_sim"][0] += r2; res["fa_sim"][1] += h2

        meet = meet_from(a); meet[a] = -1e9
        b_meet = int(np.argmax(meet))
        r2, h2 = score2({a, b_meet}); res["fa_meet"][0] += r2; res["fa_meet"][1] += h2

        # FUSION (RRF, no weights): anchor = RRF(bm25, dense); bridge = RRF(meet, dense-sim) from that anchor
        rrf1 = 1.0 / (RRF_K + rank_of(sh)) + 1.0 / (RRF_K + rank_of(dv))
        af = int(np.argmax(rrf1))
        simf = pv @ pv[af]; simf[af] = -1e9
        meetf = meet_from(af); meetf[af] = -1e9
        rrf2 = 1.0 / (RRF_K + rank_of(simf)) + 1.0 / (RRF_K + rank_of(meetf))
        rrf2[af] = -1e9
        bf = int(np.argmax(rrf2))
        r2, h2 = score2({af, bf}); res["fused"][0] += r2; res["fused"][1] += h2

    print(f"\nBRIDGE-STEP ISOLATION + FUSION on HotpotQA distractor (bridge Qs, n={nb})")
    print(f"  fixed anchor = best single-hop BM25; target = gold BM25 buries; all methods return 2 paras\n")
    print(f"   {'method':<40}{'support-fact R@2':>18}{'2nd-hop bridge R@2':>20}")
    for k, name in (("single", "single-hop BM25"),
                    ("dense", "single-hop dense"),
                    ("fa_sim", "fixed-anchor + dense-sim bridge"),
                    ("fa_meet", "fixed-anchor + rare-entity meet bridge"),
                    ("fused", "FUSED chain (RRF anchor + RRF bridge)")):
        print(f"   {name:<40}{res[k][0]/nb:>18.4f}{res[k][1]/nb:>20.4f}")
    print(f"\n  dense model: {mname}")
    print(f"  fa_meet vs fa_sim = SAME anchor, only the bridge step differs -> clean meet-vs-similarity test.")
    print(f"  FUSED > both singles AND > fa_meet => the precise meet and semantic dense reach different")
    print(f"  bridge paras and combine; FUSED ~ fa_meet => meet already dominates the bridge step.")


if __name__ == "__main__":
    main()
