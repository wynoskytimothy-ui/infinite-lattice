# Probe which corpus-agnostic BM25-pool signal separates vocab-gap (nfcorpus) from aligned (scifact/fiqa).
import sys, numpy as np, collections
sys.path.insert(0,'.')
from aethos13_phase5 import LatticeIndex, content
from aethos13_shadow_rescue import ScoredBM25
from _freq_cascade_lib import load_eval
for cn in ("scifact","nfcorpus","fiqa"):
    ev = load_eval(cn)
    corpus = {d:(v.get("title","")+" "+v.get("text","")).strip() for d,v in ev["corpus"].items()}
    idx = LatticeIndex(corpus, mindf=3)
    bm = ScoredBM25(k1=1.5,b=0.75); bm.index([(d,corpus[d]) for d in corpus])
    covs=[]; fills=[]; topnorm=[]
    for q in [x for x in ev["qids"] if ev["qrels"].get(x)]:
        qt=[t for t in dict.fromkeys(content(ev["queries"][q])) if t in idx.post]
        if not qt: continue
        qmass=sum(idx.idf(t) for t in qt) or 1.0
        sc=bm.search_scored(ev["queries"][q],200)
        best=0.0
        for d,_ in sc[:50]:
            m=sum(idx.idf(t) for t in qt if t in idx.SET[d]); best=max(best,m/qmass)
        covs.append(best); fills.append(sum(1 for _,s in sc if s>0)/200.0)
        topnorm.append((sc[0][1]/qmass) if sc else 0.0)
    covs=np.array(covs);fills=np.array(fills);tn=np.array(topnorm)
    print(f"{cn:9s} n={len(covs):4d}  coverage mean={covs.mean():.3f} p25={np.percentile(covs,25):.3f} p50={np.percentile(covs,50):.3f} | fill mean={fills.mean():.3f} p50={np.percentile(fills,50):.3f} | topnorm mean={tn.mean():.3f} p50={np.percentile(tn,50):.3f}")
