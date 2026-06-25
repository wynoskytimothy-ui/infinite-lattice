"""Route 3 ablations -- isolate WHY the operator plateaus below BM25.
Caches BM25 pools ONCE (the 349ms/q cost) then sweeps configs fast.
Trains on a MARCO train slice, evals on a fixed dev-small subset + scifact."""
import os, time, math, random, numpy as np, torch
from collections import defaultdict
import aethos_fused_learned_scorer as M
from marco_full_eval import FullIndex, stoks, MARCO

idx = FullIndex(); vs = len(idx.vocab)
phases = M.build_phases(vs).to(M.DEVICE)
NTR = int(os.environ.get('NTR','3000')); NEV = int(os.environ.get('NEV','1500'))

# ---- cache TRAIN pools ----
train = M.load_marco_train(idx, NTR)
print(f'caching {len(train)} train pools...', flush=True)
t0=time.perf_counter()
train_ex = M.build_training_set(idx, train, max_t=12, pool=24)
print(f'  train pools cached in {time.perf_counter()-t0:.0f}s ({len(train_ex)} ex)', flush=True)

# ---- cache EVAL pools (dev-small) ----
qrels=defaultdict(set)
for line in open(MARCO/'qrels.dev.small.tsv',encoding='utf-8'):
    p=line.split()
    if len(p)>=4 and int(p[3])>0: qrels[p[0]].add(int(p[2]))
queries={}
for line in open(MARCO/'queries.dev.tsv',encoding='utf-8'):
    a=line.rstrip('\n').split('\t',1)
    if len(a)==2 and a[0] in qrels: queries[a[0]]=a[1]
qids=[q for q in qrels if q in queries]; random.Random(42).shuffle(qids); qids=qids[:NEV]
print(f'caching {len(qids)} eval pools...', flush=True)
t0=time.perf_counter(); eval_pools=[]; bm_mrr=0.0
for qid in qids:
    gold=qrels[qid]; qt=stoks(queries[qid])
    order,_=idx.bm25_top([w for w in qt if idx.idf_of(w)>=0.3],k=100)
    cand=[int(d) for d in order]
    if not cand: continue
    bm_mrr+=next((1.0/r for r,d in enumerate(cand[:10],1) if d in gold),0.0)
    tids,bm,msk=M.query_pool_features(idx,qt,cand,12)
    eval_pools.append((cand,gold,tids,bm,msk))
bm_mrr/=len(eval_pools)
print(f'  eval pools cached in {time.perf_counter()-t0:.0f}s | BM25 dev-small MRR@10={bm_mrr:.4f}', flush=True)

def eval_model(model):
    s=0.0
    with torch.no_grad():
        for cand,gold,tids,bm,msk in eval_pools:
            sc=model.score_pool(torch.tensor(tids,device=M.DEVICE),torch.tensor(bm,device=M.DEVICE),torch.tensor(msk,device=M.DEVICE)).cpu().numpy()
            op=[cand[i] for i in np.argsort(-sc)]
            s+=next((1.0/r for r,d in enumerate(op[:10],1) if d in gold),0.0)
    return s/len(eval_pools)

def train_cfg(beta, use_phase, learn_beta, loss_kind, epochs=20, lr=0.05, reg=3e-4):
    model=M.FusedLearnedScorer(vs,beta=beta,phases=phases,use_phase=use_phase,learn_beta=learn_beta).to(M.DEVICE)
    opt=torch.optim.Adam(model.parameters(),lr=lr)
    ex=list(train_ex)
    for ep in range(epochs):
        random.Random(ep).shuffle(ex)
        for bs in range(0,len(ex),256):
            batch=ex[bs:bs+256]
            tids,bm,msk,pm=M._pad_pools(batch,12)
            B,P,_=tids.shape
            sc=model.score_pool(torch.tensor(tids.reshape(B*P,12),device=M.DEVICE),
                                torch.tensor(bm.reshape(B*P,12),device=M.DEVICE),
                                torch.tensor(msk.reshape(B*P,12),device=M.DEVICE)).reshape(B,P)
            pmt=torch.tensor(pm,device=M.DEVICE)
            if loss_kind=='ce':
                sc=sc.masked_fill(~pmt,float('-inf'))
                loss=torch.nn.functional.cross_entropy(sc,torch.zeros(B,dtype=torch.long,device=M.DEVICE))
            else: # hinge: gold(col0) margin over best negative
                pos=sc[:,0]
                neg=sc[:,1:].masked_fill(~pmt[:,1:],float('-inf'))
                hardneg=neg.max(dim=1).values
                loss=torch.clamp(1.0 - (pos-hardneg),min=0).mean()
            loss=loss+reg*model.l2_reg()
            opt.zero_grad(); loss.backward(); opt.step()
    return model, eval_model(model)

print(f'\n{"config":42}{"dev MRR@10":>12}{"vs BM25":>10}')
print(f'{"BM25 baseline":42}{bm_mrr:>12.4f}{0.0:>+10.4f}')
configs=[
  ('warm0.1 no-phase CE (canonical)',      dict(beta=0.1,use_phase=False,learn_beta=False,loss_kind='ce')),
  ('warm0.1 no-phase HINGE',               dict(beta=0.1,use_phase=False,learn_beta=False,loss_kind='hinge')),
  ('warm0.05 no-phase CE',                 dict(beta=0.05,use_phase=False,learn_beta=False,loss_kind='ce')),
  ('warm0.1 +PHASE CE (wave ledger on)',   dict(beta=0.1,use_phase=True,learn_beta=False,loss_kind='ce')),
  ('learn-beta no-phase CE',               dict(beta=0.1,use_phase=False,learn_beta=True,loss_kind='ce')),
]
for name,kw in configs:
    m,mrr=train_cfg(**kw)
    extra=''
    if kw['learn_beta']: extra=f' (beta->{float(torch.exp(m.log_beta)):.3f})'
    print(f'{name:42}{mrr:>12.4f}{mrr-bm_mrr:>+10.4f}{extra}',flush=True)
