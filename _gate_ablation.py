# Honest ablation of the auto-arm gate: which signal spends shadow compute only where it converts?
import sys, collections
sys.path.insert(0,'.')
from aethos13_phase5 import LatticeIndex, content
from aethos13_shadow_rescue import ScoredBM25, shadow_retrieve
from _freq_cascade_lib import load_eval
from eval_beir import recall_at_k

DEPTH, BUD = 200, 100
GATES = {
  "fill<1.0"        : lambda fill,oov: fill < 1.0,
  "fill<0.9"        : lambda fill,oov: fill < 0.9,
  "oov-only"        : lambda fill,oov: oov,
  "fill<1.0 OR oov" : lambda fill,oov: (fill < 1.0) or oov,
}
for cn in ("scifact","nfcorpus","fiqa"):
    ev = load_eval(cn)
    corpus = {d:(v.get("title","")+" "+v.get("text","")).strip() for d,v in ev["corpus"].items()}
    idx = LatticeIndex(corpus, mindf=3)
    bm = ScoredBM25(k1=1.5,b=0.75); bm.index([(d,corpus[d]) for d in corpus])
    qids=[q for q in ev["qids"] if ev["qrels"].get(q)]
    rows=[]  # per query: (fill, oov, profitable, zo_missing_set, base_ids, G)
    for q in qids:
        G=set(ev["qrels"][q])
        if not G: continue
        qterms=content(ev["queries"][q]); qset=set(qterms)
        sc=bm.search_scored(ev["queries"][q],DEPTH); base_ids=[d for d,_ in sc]; bset=set(base_ids)
        fill=sum(1 for _,s in sc if s>0)/DEPTH
        oov=any(t not in idx.post for t in dict.fromkeys(qterms) if len(t)>=3)
        missing=[g for g in G if g not in bset]
        zo=[g for g in missing if not (idx.SET.get(g,set()) & qset)]
        sh=shadow_retrieve(idx,qterms,bset,BUD) if zo else []
        zo_bridge=set(g for g in zo if g in set(sh))
        rows.append((fill,oov,base_ids,G,zo,set(sh),zo_bridge))
    prof_total=sum(1 for r in rows if r[6])
    print(f"\n=== {cn}  n={len(rows)}  profitable(zero-overlap recoverable)={prof_total} ===")
    print(f"  {'gate':16s} {'fire%':>6} {'gateP':>6} {'gateR':>6} {'ZOrec':>6} {'dRecall':>8}")
    for name,fn in GATES.items():
        fired=0; fp=0; zorec=0; sr=0.0; br=0.0
        for fill,oov,base_ids,G,zo,sh,zo_bridge in rows:
            arm=fn(fill,oov)
            pool = base_ids + [d for d in sh if d not in set(base_ids)] if arm else base_ids
            br += recall_at_k(base_ids,{g:1 for g in G},DEPTH)
            sr += recall_at_k(pool,{g:1 for g in G},DEPTH+BUD)
            if arm:
                fired+=1
                if zo_bridge: fp+=1
                zorec += len([g for g in zo if g in set(pool)])
        n=len(rows)
        gp = fp/fired if fired else 0.0
        gr = fp/prof_total if prof_total else 0.0
        print(f"  {name:16s} {100*fired/n:>5.1f} {gp:>6.3f} {gr:>6.3f} {zorec:>6} {(sr-br)/n:>+8.4f}")
