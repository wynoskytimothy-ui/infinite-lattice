#!/usr/bin/env python3
r"""
aethos_fused_learned_scorer.py  --  ROUTE 3: THE DIFFERENTIABLE FUSED OPERATOR
                                     AS A LEARNED SCORER
=============================================================================
aethos_fused_meet.py defines ONE complex operator (warm-regime differentiable):

      Z(beta) = Sum_k  a_k * exp( -beta * c_k + i * phi_k )      (partition fn)
      M_beta  = -(1/beta) * log Z(beta)                          (free energy)

This file makes the chamber costs c_k AND amplitudes a_k LEARNABLE (per-term
scalars) and trains them by GRADIENT on MS MARCO qrels to rank gold over a
contrastive pool -- a SPLADE-like learned reweighting, but routed through the
lattice's OWN complex operator instead of a bolt-on transformer.

THE SCORER (per query q, candidate doc d):
    matched query terms k=1..K, each with the lattice's per-term BM25 mass b_k.
    learnable per-term weight  w_k = softplus(theta_term)   (cost gain, >=0)
    learnable per-term amp     a_k = softplus(alpha_term)   (>=0)
    cost   c_k  = - w_k * b_k        (high BM25 mass = LOW action cost = better)
    phase  phi_k = constructive-pi root keyed by term-id (the wave ledger)
    score(q,d) = - Re( M_beta(c, phi, a) )    (negative free energy; higher=better)

In the COLD limit (beta -> inf) M_beta -> min_k c_k = the single best matched term
(tropical (min,+) meet). In the WARM regime it is a smooth log-sum-exp over all
matched terms with LEARNED per-term gains -- differentiable, so gradients flow to
w_k, a_k. Training maximizes the margin of gold over pooled negatives.

FOOTPRINT: just two scalars per vocabulary term (w_k, a_k). No dense vectors, no
transformer. SPEED: a log-sum-exp over the (few) matched terms per candidate.

HONEST PROTOCOL (two-sided):
  * weights learned on a MARCO TRAIN slice ONLY; reported on held-out MARCO
    dev-small AND zero-shot transfer to scifact / nfcorpus / fiqa (no per-corpus
    tuning -- unseen terms default to w=a=1).
  * baselines: plain BM25 (= the symbolic/answer-ness wall) on the SAME pool.
  * we report whether the LEARNED M_beta beats BM25 on the pool, by how much, and
    where it plateaus / overfits.  This isolates the lattice-NATIVE learned signal
    from the cross-encoder (the CE is NOT used here -- this is the operator alone).

    cd "C:/Users/wynos/New folder (3)" && python aethos_fused_learned_scorer.py
"""
from __future__ import annotations

import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from marco_full_eval import FullIndex, stoks, MARCO  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42


# ===========================================================================
# Per-term feature extraction (the lattice's symbolic per-term BM25 mass)
# ===========================================================================
def query_pool_features(idx: FullIndex, qterms, cand, max_t: int):
    """FAST per-term BM25 mass for a pool of candidates over the query's terms.
    For each query term, scan its posting list ONCE and gather the contribution to
    the candidate docs (searchsorted into the sorted posting). Returns
        term_ids (C,max_t) int64, bm (C,max_t) float32, mask (C,max_t) bool.
    bm[c,k] is the EXACT BM25 contribution of term k to candidate c (same formula
    the FullIndex baseline uses), so the operator learns ABOVE the real BM25 mass."""
    from marco_full_eval import K1, B
    C = len(cand)
    cand_arr = np.asarray(cand, dtype=np.int64)
    cand_dl = idx.doclen[cand_arr]
    # dedupe query terms but keep order, cap at max_t
    seen = []
    for w in qterms:
        i = idx.tid.get(w)
        if i is None:
            continue
        wi = float(idx.idfa[i])
        if wi < 0.3:                      # match the QGATE used by the baseline
            continue
        seen.append(i)
        if len(seen) >= max_t:
            break
    T = max_t
    tids = np.zeros((C, T), dtype=np.int64)
    bm = np.zeros((C, T), dtype=np.float32)
    msk = np.zeros((C, T), dtype=bool)
    # map cand_id -> row for gather
    order = np.argsort(cand_arr)
    sorted_cand = cand_arr[order]
    for k, i in enumerate(seen):
        tids[:, k] = i
        s, e = int(idx.ptr[i]), int(idx.ptr[i + 1])
        dis = idx.di[s:e]
        tf = idx.tf[s:e].astype(np.float32)
        wi = float(idx.idfa[i])
        # which candidates appear in this term's posting?
        pos = np.searchsorted(dis, sorted_cand)
        pos = np.clip(pos, 0, len(dis) - 1)
        hit = dis[pos] == sorted_cand
        rows = order[hit]                 # candidate row indices that match
        ph = pos[hit]
        tfh = tf[ph]
        dlh = cand_dl[rows]
        contrib = wi * tfh * (K1 + 1.0) / (tfh + K1 * (1.0 - B + B * dlh / idx.avgdl))
        bm[rows, k] = contrib
        msk[rows, k] = True
    return tids, bm, msk


# ===========================================================================
# THE LEARNED FUSED OPERATOR (torch module)
# ===========================================================================
class FusedLearnedScorer(nn.Module):
    r"""Learnable per-term (cost-gain w_k, amplitude a_k) routed through
        M_beta = -(1/beta) log Sum_k a_k exp(-beta c_k + i phi_k),
        c_k = -w_k * b_k.  score = -Re(M_beta).

    Parameters are indexed by global vocabulary term-id; default (untrained /
    unseen term) gives w=a=1 so the operator reduces to a fixed log-sum-exp over
    raw BM25 mass -- a clean baseline to learn ABOVE."""

    def __init__(self, vocab_size: int, beta: float = 0.3, phases: torch.Tensor | None = None,
                 use_phase: bool = True, learn_beta: bool = True):
        super().__init__()
        # init so softplus(theta)=1 => theta = log(e^1 - 1) ~ 0.5413 (untrained w_k=a_k=1)
        self.init = math.log(math.e - 1.0)
        self.theta = nn.Parameter(torch.full((vocab_size,), self.init))   # -> w_k cost-gain
        self.alpha = nn.Parameter(torch.full((vocab_size,), self.init))   # -> a_k amplitude
        self.use_phase = use_phase
        # beta routes the cold(particle)<->warm(wave) regime; learnable so the
        # operator can pick its own temperature. param = log beta (keeps beta>0).
        self.log_beta = nn.Parameter(torch.tensor(math.log(beta)), requires_grad=learn_beta)
        if phases is None:
            phases = torch.zeros(vocab_size)
        self.register_buffer("phases", phases)

    def score_pool(self, term_ids: torch.Tensor, bm: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        r"""score = -Re(M_beta),  M_beta = -(1/beta) log Sum_k a_k exp(-beta c_k + i phi_k),
            c_k = -w_k b_k  (high BM25 mass -> low action cost).  So

              -Re(M_beta) = (1/beta) * log | Sum_k a_k exp(beta w_k b_k + i phi_k) |

            WARM (small beta): -> (1/beta) log K + weighted-mean-ish (sum-like, ~BM25).
            COLD (large beta): -> max_k w_k b_k (single best term, tropical meet).
        term_ids:(C,T) long  bm:(C,T) float  mask:(C,T) bool   ->  (C,) score."""
        beta = torch.exp(self.log_beta)
        w = torch.nn.functional.softplus(self.theta[term_ids])       # (C,T) >=0
        a = torch.nn.functional.softplus(self.alpha[term_ids])       # (C,T) >=0
        amp = a * mask.float()                                       # padded slots -> 0
        x = beta * (w * bm)                                          # = -beta*c_k, >=0
        # stabilize by per-row max exponent (the cold attractor)
        x_masked = x.masked_fill(~mask, float("-inf"))
        xmax = x_masked.max(dim=1, keepdim=True).values             # (C,1)
        xmax = torch.where(torch.isfinite(xmax), xmax, torch.zeros_like(xmax))
        e = amp * torch.exp(x - xmax)                               # |term|, no overflow
        if self.use_phase:
            phi = self.phases[term_ids]
            real = (e * torch.cos(phi)).sum(dim=1)
            imag = (e * torch.sin(phi)).sum(dim=1)
            log_mod = 0.5 * torch.log(real * real + imag * imag + 1e-9)
        else:
            log_mod = torch.log(e.sum(dim=1) + 1e-9)
        # -Re(M) = (1/beta)(xmax + log|sum exp(x-xmax)|)
        return (xmax.squeeze(1) + log_mod) / beta

    def l2_reg(self) -> torch.Tensor:
        return ((self.theta - self.init) ** 2).mean() + ((self.alpha - self.init) ** 2).mean()


# ===========================================================================
# constructive-pi phases keyed by term-id (the WAVE ledger)
# ===========================================================================
def build_phases(vocab_size: int) -> torch.Tensor:
    """Phase per term: e^{i * 2pi * (term_id mod 32)/32}, the 32-chamber roots of
    unity from the lattice. Wave ledger -- deterministic, no learned phase."""
    idx = np.arange(vocab_size) % 32
    ang = 2.0 * math.pi * idx / 32.0
    return torch.tensor(ang, dtype=torch.float32)


# ===========================================================================
# MARCO training-pair construction (gold + pooled negatives)
# ===========================================================================
def load_marco_train(idx: FullIndex, n_queries: int, seed: int = SEED):
    qrels_tr = defaultdict(set)
    with open(MARCO / "qrels.train.tsv", encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) >= 4 and int(p[3]) > 0:
                qrels_tr[p[0]].add(int(p[2]))
    sel = list(qrels_tr)
    random.Random(seed).shuffle(sel)
    sel = sel[:n_queries]
    sel_set = set(sel)
    qtexts = {}
    with open(MARCO / "queries.train.tsv", encoding="utf-8") as f:
        for line in f:
            a = line.rstrip("\n").split("\t", 1)
            if len(a) == 2 and a[0] in sel_set:
                qtexts[a[0]] = a[1]
    return [(q, qtexts[q], qrels_tr[q]) for q in sel if q in qtexts]


# ===========================================================================
# Build training tensors: for each query, gold doc + BM25-pool negatives.
# Each example is a fixed-size pool (gold@row0 + `pool` negatives), pre-featurized
# to (tids, bm, mask) of shape (1+pool, max_t) so a batch is one GPU op.
# ===========================================================================
def build_training_set(idx: FullIndex, train, max_t=12, pool=24):
    examples = []
    t0 = time.perf_counter()
    for qi, (qid, qtext, gold) in enumerate(train):
        qterms = stoks(qtext)
        order, _ = idx.bm25_top([w for w in qterms if idx.idf_of(w) >= 0.3], k=pool + 8)
        cand = [int(d) for d in order]
        pos = next((g for g in gold if g in cand), None)
        if pos is None:
            pos = next((g for g in gold), None)        # gold may be outside pool
            if pos is None:
                continue
        negs = [d for d in cand if d not in gold][:pool]
        if len(negs) < 4:
            continue
        pooldocs = [pos] + negs
        tids, bm, msk = query_pool_features(idx, qterms, pooldocs, max_t)
        examples.append((tids, bm, msk))               # row 0 is gold
        if (qi + 1) % 4000 == 0:
            print(f"    featurized {qi+1}/{len(train)} ({time.perf_counter()-t0:.0f}s)", flush=True)
    return examples


def _pad_pools(batch, max_t):
    """Stack a list of (tids,bm,msk) pools of possibly-different pool sizes into
    (B, P, max_t) with a pool-mask (B, P). Row 0 of each pool is the gold."""
    P = max(b[0].shape[0] for b in batch)
    B = len(batch)
    tids = np.zeros((B, P, max_t), dtype=np.int64)
    bm = np.zeros((B, P, max_t), dtype=np.float32)
    msk = np.zeros((B, P, max_t), dtype=bool)
    pool_mask = np.zeros((B, P), dtype=bool)
    for b, (t, m, k) in enumerate(batch):
        p = t.shape[0]
        tids[b, :p] = t
        bm[b, :p] = m
        msk[b, :p] = k
        pool_mask[b, :p] = True
    return tids, bm, msk, pool_mask


def train_scorer(model: FusedLearnedScorer, examples, epochs=8, lr=0.05, reg=1e-4, max_t=12):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    n = len(examples)
    BS = 256
    for ep in range(epochs):
        random.Random(SEED + ep).shuffle(examples)
        tot = 0.0
        nb = 0
        for bstart in range(0, n, BS):
            batch = examples[bstart:bstart + BS]
            tids, bm, msk, pool_mask = _pad_pools(batch, max_t)
            B, P, _ = tids.shape
            tids_t = torch.tensor(tids.reshape(B * P, max_t), device=DEVICE)
            bm_t = torch.tensor(bm.reshape(B * P, max_t), device=DEVICE)
            msk_t = torch.tensor(msk.reshape(B * P, max_t), device=DEVICE)
            sc = model.score_pool(tids_t, bm_t, msk_t).reshape(B, P)
            pm = torch.tensor(pool_mask, device=DEVICE)
            sc = sc.masked_fill(~pm, float("-inf"))      # padded pool slots -> -inf
            target = torch.zeros(B, dtype=torch.long, device=DEVICE)   # gold at col 0
            loss = torch.nn.functional.cross_entropy(sc, target) + reg * model.l2_reg()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
            nb += 1
        print(f"    epoch {ep+1}/{epochs}  loss {tot/max(1,nb):.4f}  "
              f"beta {float(torch.exp(model.log_beta)):.3f}", flush=True)
    model.eval()


# ===========================================================================
# Evaluation: re-rank the BM25 top-`pool` with the learned operator.
# ===========================================================================
@torch.no_grad()
def eval_pool_marco(idx, model, queries, qrels, max_t=12, pool=100, n_eval=2000):
    qids = [q for q in qrels if q in queries]
    random.Random(SEED).shuffle(qids)
    qids = qids[:n_eval]
    bm_mrr = op_mrr = bm_r10 = op_r10 = 0.0
    lat_bm25 = []      # BM25 candidate generation (the baseline ALSO pays this)
    lat_feat = []      # per-term feature extraction for the pool
    lat_op = []        # the operator's own score_pool (the MARGINAL re-rank cost)
    used = 0
    for qid in qids:
        gold = qrels[qid]
        qterms = stoks(queries[qid])
        t = time.perf_counter()
        order, _ = idx.bm25_top([w for w in qterms if idx.idf_of(w) >= 0.3], k=pool)
        lat_bm25.append((time.perf_counter() - t) * 1000)
        cand = [int(d) for d in order]
        if not cand:
            continue
        used += 1
        bm_order = cand
        t = time.perf_counter()
        tids, bm, msk = query_pool_features(idx, qterms, cand, max_t)
        tids_t = torch.tensor(tids, device=DEVICE)
        bm_t = torch.tensor(bm, device=DEVICE)
        msk_t = torch.tensor(msk, device=DEVICE)
        lat_feat.append((time.perf_counter() - t) * 1000)
        t = time.perf_counter()
        sc = model.score_pool(tids_t, bm_t, msk_t).cpu().numpy()
        lat_op.append((time.perf_counter() - t) * 1000)
        op_order = [cand[i] for i in np.argsort(-sc)]
        bm_mrr += next((1.0 / r for r, d in enumerate(bm_order[:10], 1) if d in gold), 0.0)
        op_mrr += next((1.0 / r for r, d in enumerate(op_order[:10], 1) if d in gold), 0.0)
        bm_r10 += 1.0 if any(d in gold for d in bm_order[:10]) else 0.0
        op_r10 += 1.0 if any(d in gold for d in op_order[:10]) else 0.0
    m = max(1, used)
    return dict(n=used, bm_mrr=bm_mrr / m, op_mrr=op_mrr / m,
                bm_r10=bm_r10 / m, op_r10=op_r10 / m,
                lat_bm25=float(np.median(lat_bm25)) if lat_bm25 else 0.0,
                lat_feat=float(np.median(lat_feat)) if lat_feat else 0.0,
                lat_op=float(np.median(lat_op)) if lat_op else 0.0)


# ===========================================================================
# BEIR ZERO-SHOT TRANSFER: build a small per-corpus BM25 index with the SAME
# tokenizer, map tokens to MARCO term-ids so the learned per-term weights apply.
# Unseen-in-MARCO tokens get a fresh local id with default weight (w=a=1).
# ===========================================================================
BEIR_ROOT = Path(r"C:\Users\wynos\.cursor\worktrees\prime_hotel\bbl\beir_datasets")


def _beir_load(name):
    import json
    root = BEIR_ROOT / name
    corpus, queries = {}, {}
    for line in open(root / "corpus.jsonl", encoding="utf-8"):
        o = json.loads(line)
        corpus[o["_id"]] = (o.get("title", "") + " " + o.get("text", "")).strip()
    for line in open(root / "queries.jsonl", encoding="utf-8"):
        o = json.loads(line)
        queries[o["_id"]] = o["text"]
    rel = defaultdict(dict)
    p = root / "qrels" / "test.tsv"
    import csv
    r = csv.reader(open(p, encoding="utf-8"), delimiter="\t")
    next(r)
    for qid, cid, sc in r:
        if int(sc) > 0:
            rel[qid][cid] = int(sc)
    return corpus, queries, rel


class BeirBM25:
    """Minimal BM25 over a BEIR corpus, tokenized with the lattice `stoks`, terms
    mapped to MARCO global term-ids where they exist (so learned w_k transfer)."""

    def __init__(self, corpus, marco_tid, k1=0.9, b=0.4):
        self.k1, self.b = k1, b
        self.docids = list(corpus.keys())
        self.doc_idx = {d: i for i, d in enumerate(self.docids)}
        N = len(self.docids)
        # local term vocab; gid = MARCO id if shared else a synthetic id (>= big base)
        self.postings = defaultdict(list)     # term -> list[(doc_row, tf)]
        self.doclen = np.zeros(N, dtype=np.float32)
        df = Counter()
        for d, txt in corpus.items():
            row = self.doc_idx[d]
            toks = stoks(txt)
            self.doclen[row] = len(toks)
            tf = Counter(toks)
            for t, c in tf.items():
                self.postings[t].append((row, c))
            for t in tf:
                df[t] += 1
        self.avgdl = float(self.doclen.mean()) if N else 1.0
        self.N = N
        self.idf = {t: math.log(1 + (N - d + 0.5) / (d + 0.5)) for t, d in df.items()}
        # global id for each local term: MARCO id if present else synthetic
        base = 10 ** 9
        self.gid = {}
        nxt = base
        for t in self.postings:
            mi = marco_tid.get(t)
            if mi is None:
                self.gid[t] = nxt
                nxt += 1
            else:
                self.gid[t] = mi
        # precompute sorted postings as arrays
        self.parr = {}
        for t, lst in self.postings.items():
            lst.sort()
            rows = np.array([x[0] for x in lst], dtype=np.int64)
            tfs = np.array([x[1] for x in lst], dtype=np.float32)
            self.parr[t] = (rows, tfs)

    def top(self, qtoks, k=100):
        acc = np.zeros(self.N, dtype=np.float32)
        seen = [t for t in dict.fromkeys(qtoks) if t in self.parr]
        for t in seen:
            rows, tfs = self.parr[t]
            wi = self.idf.get(t, 0.0)
            dl = self.doclen[rows]
            acc[rows] += wi * tfs * (self.k1 + 1.0) / (
                tfs + self.k1 * (1.0 - self.b + self.b * dl / self.avgdl))
        nz = np.nonzero(acc)[0]
        if len(nz) == 0:
            return [], seen
        if len(nz) > k:
            nz = nz[np.argpartition(-acc[nz], k)[:k]]
        order = nz[np.argsort(-acc[nz])]
        return list(order), seen

    def pool_features(self, qtoks, cand_rows, max_t):
        """(tids,bm,mask) for BEIR candidates -- gid term-ids + BM25 mass."""
        C = len(cand_rows)
        cand = np.asarray(cand_rows, dtype=np.int64)
        seen = [t for t in dict.fromkeys(qtoks) if t in self.parr and self.idf.get(t, 0) >= 0.3][:max_t]
        tids = np.zeros((C, max_t), dtype=np.int64)
        bm = np.zeros((C, max_t), dtype=np.float32)
        msk = np.zeros((C, max_t), dtype=bool)
        cand_dl = self.doclen[cand]
        cmap = {r: i for i, r in enumerate(cand_rows)}
        for k, t in enumerate(seen):
            tids[:, k] = self.gid[t]
            rows, tfs = self.parr[t]
            wi = self.idf.get(t, 0.0)
            for r, tf in zip(rows.tolist(), tfs.tolist()):
                ci = cmap.get(r)
                if ci is None:
                    continue
                dl = cand_dl[ci]
                bm[ci, k] = wi * tf * (self.k1 + 1.0) / (
                    tf + self.k1 * (1.0 - self.b + self.b * dl / self.avgdl))
                msk[ci, k] = True
        return tids, bm, msk


@torch.no_grad()
def eval_beir(name, model, marco_tid, vocab_size, max_t=12, pool=100):
    corpus, queries, rel = _beir_load(name)
    bm25 = BeirBM25(corpus, marco_tid)
    # extend the model param tables for synthetic (BEIR-only) ids by reusing the
    # default: any gid >= vocab_size maps to the default weight via a clamp.
    qids = [q for q in rel if q in queries]
    bm_nd = op_nd = bm_r = op_r = 0.0
    used = 0
    init = model.init
    for qid in qids:
        gold = rel[qid]
        qtoks = stoks(queries[qid])
        order, _ = bm25.top(qtoks, k=pool)
        if not order:
            continue
        used += 1
        cand_rows = list(order)
        cand_docids = [bm25.docids[r] for r in cand_rows]
        tids, bm, msk = bm25.pool_features(qtoks, cand_rows, max_t)
        # clamp synthetic ids (>= vocab_size) to a reserved default slot (id 0 w/ default)
        tids = np.where(tids >= vocab_size, 0, tids)
        sc = model.score_pool(torch.tensor(tids, device=DEVICE),
                              torch.tensor(bm, device=DEVICE),
                              torch.tensor(msk, device=DEVICE)).cpu().numpy()
        op_docids = [cand_docids[i] for i in np.argsort(-sc)]
        bm_nd += _ndcg10(cand_docids, gold)
        op_nd += _ndcg10(op_docids, gold)
        bm_r += _recall10(cand_docids, gold)
        op_r += _recall10(op_docids, gold)
    m = max(1, used)
    return dict(name=name, n=used, bm_ndcg=bm_nd / m, op_ndcg=op_nd / m,
                bm_r10=bm_r / m, op_r10=op_r / m)


def _ndcg10(ranked, rels):
    dcg = sum(rels.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked[:10]))
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(sorted(rels.values(), reverse=True)[:10]))
    return dcg / idcg if idcg else 0.0


def _recall10(ranked, rels):
    rel = {d for d, s in rels.items() if s > 0}
    return len(set(ranked[:10]) & rel) / len(rel) if rel else 0.0


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    n_train = int(os.environ.get("N_TRAIN", "8000"))
    n_eval = int(os.environ.get("N_EVAL", "2000"))
    # Operator regime is FIXED by the diagnostic (_diag_untrained.py):
    #   * phase OFF -- the constructive-pi WAVE ledger is destructive interference,
    #     which is ANTI-ranking (every beta: phase<no-phase). Wave hurts retrieval.
    #   * beta small (warm) -- the operator's sum-faithful regime (~BM25). Cold beta
    #     -> max-over-terms drops multi-term evidence and MRR collapses.
    # The LEARNED signal is the per-term cost-gain w_k (SPLADE-like reweighting)
    # routed through the operator, NOT the temperature.
    beta = float(os.environ.get("BETA", "0.1"))
    epochs = int(os.environ.get("EPOCHS", "10"))
    use_phase = os.environ.get("USE_PHASE", "0") == "1"
    learn_beta = os.environ.get("LEARN_BETA", "0") == "1"
    lr = float(os.environ.get("LR", "0.03"))
    reg = float(os.environ.get("REG", "3e-4"))

    print("#" * 78)
    print("# ROUTE 3 -- LEARNED FUSED OPERATOR M_beta as a lattice-native scorer")
    print(f"#   train={n_train} q | eval={n_eval} q | beta={beta}(learn={learn_beta}) | "
          f"epochs={epochs} | phase={use_phase} | lr={lr} | {DEVICE}")
    print("#" * 78, flush=True)

    idx = FullIndex()
    vocab_size = len(idx.vocab)
    phases = build_phases(vocab_size).to(DEVICE)
    model = FusedLearnedScorer(vocab_size, beta=beta, phases=phases,
                               use_phase=use_phase, learn_beta=learn_beta).to(DEVICE)

    print("\n[train] loading MARCO train pairs ...", flush=True)
    train = load_marco_train(idx, n_train)
    print(f"  {len(train)} train queries with gold", flush=True)
    print("[train] featurizing (gold + BM25 pool negatives) ...", flush=True)
    examples = build_training_set(idx, train, max_t=12, pool=24)
    print(f"  {len(examples)} training examples", flush=True)
    print("[train] gradient descent on per-term (w_k, a_k) ...", flush=True)
    t0 = time.perf_counter()
    train_scorer(model, examples, epochs=epochs, lr=lr, reg=reg)
    print(f"  trained in {time.perf_counter()-t0:.0f}s", flush=True)

    # how many term weights actually moved (footprint = nonzero deltas)
    init = math.log(math.e - 1.0)
    moved = int(((model.theta.detach().cpu() - init).abs() > 1e-3).sum())
    print(f"  term weights that moved from default: {moved} / {vocab_size}", flush=True)

    # ---- MARCO dev-small (held out: train was train.tsv, eval is dev.small) ----
    qrels_dev = defaultdict(set)
    with open(MARCO / "qrels.dev.small.tsv", encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) >= 4 and int(p[3]) > 0:
                qrels_dev[p[0]].add(int(p[2]))
    queries_dev = {}
    with open(MARCO / "queries.dev.tsv", encoding="utf-8") as f:
        for line in f:
            a = line.rstrip("\n").split("\t", 1)
            if len(a) == 2 and a[0] in qrels_dev:
                queries_dev[a[0]] = a[1]
    print("\n[eval] MARCO dev-small (held-out) -- re-rank BM25 pool ...", flush=True)
    r = eval_pool_marco(idx, model, queries_dev, qrels_dev, pool=100, n_eval=n_eval)
    print(f"  MARCO dev-small (n={r['n']}, pool=100):")
    print(f"    BM25            MRR@10 {r['bm_mrr']:.4f}  R@10 {r['bm_r10']:.4f}")
    print(f"    learned M_beta  MRR@10 {r['op_mrr']:.4f}  R@10 {r['op_r10']:.4f}  "
          f"(delta {r['op_mrr']-r['bm_mrr']:+.4f})")
    print(f"    latency median: BM25 cand-gen {r['lat_bm25']:.1f} ms (baseline pays this too) "
          f"| feat {r['lat_feat']:.2f} ms | operator score_pool {r['lat_op']:.3f} ms", flush=True)

    # ---- BEIR ZERO-SHOT TRANSFER (no per-corpus tuning) ----
    marco_tid = idx.tid
    print("\n[eval] BEIR zero-shot transfer (learned-on-MARCO weights, no retune) ...", flush=True)
    beir = {}
    for ds in ("scifact", "nfcorpus", "fiqa"):
        try:
            br = eval_beir(ds, model, marco_tid, vocab_size, pool=100)
            beir[ds] = br
            print(f"  {ds:9s} (n={br['n']:4d}): BM25 nDCG@10 {br['bm_ndcg']:.4f} -> "
                  f"learned {br['op_ndcg']:.4f}  (delta {br['op_ndcg']-br['bm_ndcg']:+.4f})", flush=True)
        except Exception as ex:
            print(f"  {ds}: FAILED {ex}", flush=True)

    print("\n" + "#" * 78)
    print("# SUMMARY")
    print("#" * 78)
    print(f"  MARCO dev-small MRR@10 : BM25 {r['bm_mrr']:.4f} -> learned {r['op_mrr']:.4f} "
          f"({r['op_mrr']-r['bm_mrr']:+.4f})")
    for ds, br in beir.items():
        print(f"  {ds:9s} nDCG@10      : BM25 {br['bm_ndcg']:.4f} -> learned {br['op_ndcg']:.4f} "
              f"({br['op_ndcg']-br['bm_ndcg']:+.4f})")
    print(f"  footprint : 2 scalars/term, {moved} moved (~{moved*8/1e6:.2f} MB)  |  "
          f"operator re-rank {r['lat_op']:.3f} ms/q on the 100-doc pool (BM25 cand-gen "
          f"{r['lat_bm25']:.0f} ms is the baseline's cost, unchanged)")
    return model, idx, r, beir


if __name__ == "__main__":
    main()
