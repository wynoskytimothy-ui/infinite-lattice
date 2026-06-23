#!/usr/bin/env python3
"""CORPUS-SCALE multi-hop (the real test): retrieve from the FULL 2Wiki paragraph corpus (~37k), not the
10 distractors. single-hop BM25 top-K  vs  CHAIN (anchor + rare-entity meet bridges over the whole corpus).
Does corridor traversal reach the buried bridge passage when it must be FOUND among 37k, not re-ranked among 10?
Metric: support-fact recall@K (both gold paras) + 2nd-hop bridge recall@K, single-hop vs chain.
"""
import re, math, json, random
from collections import Counter, defaultdict
import numpy as np
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
from stem_safe import safe

WORD = re.compile(r"[a-z0-9]+")
K1, B = 0.9, 0.4
RARE = 5.0
K = 10
BRIDGE_TYPES = {"compositional", "inference", "bridge_comparison"}

_st = {}
def st(w):
    r = _st.get(w)
    if r is None:
        r = safe(w); _st[w] = r
    return r
def toks(s):
    return [st(w) for w in WORD.findall(s.lower())]
def rank_of(score):
    o = np.argsort(-score); r = np.empty(len(o), np.int64); r[o] = np.arange(len(o)); return r


def main():
    p = hf_hub_download("xanhho/2WikiMultihopQA", "dev.parquet", repo_type="dataset")
    t = pq.read_table(p).to_pandas()
    para_text = {}; rows = []
    for _, r in t.iterrows():
        if r["type"] not in BRIDGE_TYPES:
            continue
        ctx = json.loads(r["context"]); sf = json.loads(r["supporting_facts"])
        titles = [c[0] for c in ctx]; gold = set(s[0] for s in sf) & set(titles)
        if len(gold) < 2:
            continue
        for c in ctx:
            para_text.setdefault(c[0], " ".join(c[1]) if isinstance(c[1], (list, tuple)) else str(c[1]))
        rows.append((r["question"], gold))
    titles = list(para_text); tidx = {ti: i for i, ti in enumerate(titles)}; M = len(titles)
    print(f"  corpus: {M} unique paras, {len(rows)} bridge-type Qs (retrieve from the FULL corpus)", flush=True)

    para_tok = [Counter(toks(para_text[ti])) for ti in titles]
    df = Counter()
    for c in para_tok:
        for w in c:
            df[w] += 1
    idf = {w: math.log((M - c + 0.5) / (c + 0.5) + 1.0) for w, c in df.items()}
    post = defaultdict(list); postf = defaultdict(list)
    for i, c in enumerate(para_tok):
        for w, f in c.items():
            post[w].append(i); postf[w].append(f)
    post = {w: np.array(v, np.int32) for w, v in post.items()}
    postf = {w: np.array(v, np.float32) for w, v in postf.items()}
    doclen = np.array([sum(c.values()) for c in para_tok], np.float32); avgdl = float(doclen.mean())

    def bm25_scores(qterms):
        sc = np.zeros(M, np.float32)
        for w in set(qterms):
            if w not in post:
                continue
            ps = post[w]; tfs = postf[w]; dl = doclen[ps]
            sc[ps] += idf[w] * tfs * (K1 + 1) / (tfs + K1 * (1 - B + B * dl / avgdl))
        return sc

    res = {"single": [0.0, 0.0], "chain": [0.0, 0.0], "fused": [0.0, 0.0]}; nb = 0
    for q, gold in rows:
        gidx = [tidx[g] for g in gold]
        sc = bm25_scores(toks(q))
        order = np.argsort(-sc)
        top_single = set(int(x) for x in order[:K])
        pos = {int(d): r for r, d in enumerate(order)}
        harder = max(gidx, key=lambda gi: pos.get(gi, M))    # gold single-hop buries
        # CHAIN: anchor = BM25-best; bridge = corpus paras sharing the anchor's RARE terms (the meet)
        a = int(order[0])
        bsc = np.zeros(M, np.float32)
        for w, f in para_tok[a].items():
            if idf.get(w, 0) >= RARE and w in post:
                ps = post[w]; bsc[ps] += idf[w]
        bsc[a] = -1.0
        bridge = np.argsort(-bsc)[:K - 1]
        chain = set([a] + [int(x) for x in bridge])
        # FUSED (slot-based): single-hop breadth for K-2 slots + 2 meet-bridge slots (dense + sparse, not blended)
        fused = set(int(x) for x in order[:K - 2]) | set(int(x) for x in np.argsort(-bsc)[:2])

        def rec(sel):
            return len(set(gidx) & sel) / 2.0, (1.0 if harder in sel else 0.0)
        for k, sel in (("single", top_single), ("chain", chain), ("fused", fused)):
            r2, h2 = rec(sel); res[k][0] += r2; res[k][1] += h2
        nb += 1
    print(f"\n  CORPUS-SCALE 2WIKI MULTI-HOP (retrieve from {M} paras, budget K={K}, n={nb})\n")
    print(f"  {'retriever':<32}{'support-fact R@K':>18}{'2nd-hop bridge R@K':>20}")
    for k, name in (("single", "single-hop BM25 top-K"), ("chain", "CHAIN (anchor + meet bridges)"),
                    ("fused", "FUSED (breadth + 2 bridge slots)")):
        print(f"  {name:<32}{res[k][0]/nb:>18.4f}{res[k][1]/nb:>20.4f}")
    print(f"\n  chain > single on 2nd-hop => corridor traversal reaches the bridge passage when it must be FOUND")
    print(f"  among {M:,} paras, not re-ranked among 10 -- the smarter-RAG retrieval at corpus scale.")


if __name__ == "__main__":
    main()
