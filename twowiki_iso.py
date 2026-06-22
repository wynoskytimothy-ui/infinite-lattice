#!/usr/bin/env python3
"""GENERALITY TEST: does the multi-hop result (the rare-entity MEET beats dense-similarity on the bridge
step) hold on 2WikiMultihop, not just HotpotQA? Same airtight isolation -- fixed anchor = BM25-best, vary
ONLY the bridge step (meet vs dense-sim), same fixed target (the gold BM25 buries). If meet wins here too,
"precise meet > diffuse similarity for multi-hop" stops being a HotpotQA artifact.
"""
import re, math, json, random
from collections import Counter
import numpy as np
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
from stem_safe import safe

WORD = re.compile(r"[a-z0-9]+")
K1, B = 0.9, 0.4
RARE = 5.0
BRIDGE_TYPES = {"compositional", "inference", "bridge_comparison"}   # 2-hop chains (not pure "comparison")

_st = {}
def st(w):
    r = _st.get(w)
    if r is None:
        r = safe(w); _st[w] = r
    return r
def toks(s):
    return [st(w) for w in WORD.findall(s.lower())]


def main():
    p = hf_hub_download("xanhho/2WikiMultihopQA", "dev.parquet", repo_type="dataset")
    t = pq.read_table(p).to_pandas()
    print(f"  loaded 2WikiMultihop dev: {len(t)} questions", flush=True)

    para_tok = {}; para_text = {}; df = Counter(); rows = []
    for _, r in t.iterrows():
        if r["type"] not in BRIDGE_TYPES:
            continue
        ctx = json.loads(r["context"]); sf = json.loads(r["supporting_facts"])
        titles = [c[0] for c in ctx]
        gold = set(s[0] for s in sf) & set(titles)
        if len(gold) < 2:
            continue
        for c in ctx:
            title, sents = c[0], c[1]
            if title not in para_tok:
                txt = " ".join(sents) if isinstance(sents, (list, tuple)) else str(sents)
                tk = toks(txt)
                para_tok[title] = Counter(tk); para_text[title] = txt
                for w in set(tk):
                    df[w] += 1
        rows.append((r["question"], titles, gold))
    N = len(para_tok)
    idf = {w: math.log((N - c + 0.5) / (c + 0.5) + 1.0) for w, c in df.items()}
    avgdl = sum(sum(c.values()) for c in para_tok.values()) / N
    print(f"  {N} unique paras, {len(rows)} bridge-type Qs (compositional/inference/bridge_comparison)", flush=True)

    def bm25(qset, ctr):
        dl = sum(ctr.values())
        return sum(idf.get(w, 0) * ctr.get(w, 0) * (K1 + 1) / (ctr.get(w, 0) + K1 * (1 - B + B * dl / avgdl)) for w in qset)

    from sentence_transformers import SentenceTransformer
    import torch
    enc = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1", device="cuda" if torch.cuda.is_available() else "cpu")
    titles_all = list(para_text)
    pe = enc.encode([para_text[ti] for ti in titles_all], batch_size=256, convert_to_numpy=True,
                    normalize_embeddings=True, show_progress_bar=False)
    pemb = {ti: pe[i] for i, ti in enumerate(titles_all)}
    qe = enc.encode([q for (q, _, _) in rows], batch_size=256, convert_to_numpy=True,
                    normalize_embeddings=True, show_progress_bar=False)
    print(f"  embedded {N} paras + {len(rows)} Qs; scoring...", flush=True)

    res = {k: [0.0, 0.0] for k in ("single", "fa_sim", "fa_meet")}; nb = 0
    for qi, (q, titles, gold) in enumerate(rows):
        paras = [para_tok[ti] for ti in titles]; n = len(titles)
        qset = set(toks(q))
        sh = np.array([bm25(qset, c) for c in paras]); order_sh = list(np.argsort(-sh))
        gi = [titles.index(g) for g in gold]; harder = max(gi, key=lambda x: order_sh.index(x)); ht = titles[harder]

        def score2(sel):
            tt = set(titles[i] for i in sel)
            return len(gold & tt) / 2.0, (1.0 if ht in tt else 0.0)

        r2, h2 = score2(set(order_sh[:2])); res["single"][0] += r2; res["single"][1] += h2
        a = order_sh[0]
        pv = np.array([pemb[ti] for ti in titles]); sim = pv @ pv[a]; sim[a] = -1e9
        r2, h2 = score2({a, int(np.argmax(sim))}); res["fa_sim"][0] += r2; res["fa_sim"][1] += h2
        bterms = Counter()
        for w in paras[a]:
            if w not in qset and idf.get(w, 0) >= RARE:
                bterms[w] += idf[w]
        meet = np.array([sum(w for t2, w in bterms.items() if t2 in paras[i]) for i in range(n)]); meet[a] = -1e9
        r2, h2 = score2({a, int(np.argmax(meet))}); res["fa_meet"][0] += r2; res["fa_meet"][1] += h2
        nb += 1
    print(f"\n  2WIKIMULTIHOP bridge-step isolation (n={nb}) -- fixed anchor=BM25-best, only the bridge varies\n")
    print(f"  {'method':<36}{'support-fact R@2':>18}{'2nd-hop bridge R@2':>20}")
    for k, name in (("single", "single-hop BM25"), ("fa_sim", "fixed-anchor + dense-sim bridge"),
                    ("fa_meet", "fixed-anchor + rare-entity meet")):
        print(f"  {name:<36}{res[k][0]/nb:>18.4f}{res[k][1]/nb:>20.4f}")
    print(f"\n  meet > dense-sim on 2nd-hop => the rare-entity bridge generalizes beyond HotpotQA (HotpotQA was 0.46 vs 0.30).")


if __name__ == "__main__":
    main()
