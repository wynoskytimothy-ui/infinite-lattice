#!/usr/bin/env python3
"""The HONEST boundary test: multi-query associative recall (MQAR), the task that
favours attention (a literal key->value lookup) and stresses a recurrence (finite
state must hold every binding). Mamba/Zoology lit: pure recurrences trail attention
here once #pairs exceeds state capacity. Multi-layer, matched params, two sweeps:
  A. STATE PRESSURE  -- recall accuracy vs number of kv-pairs (fixed length)
  B. LONG RANGE      -- recall accuracy vs sequence length (fixed #pairs)

Dictionary [k0 v0 k1 v1 ...] at the front; queries (the keys) scattered in a tail of
distractor tokens; target at each query = its bound value. Loss only on queries.
keys / values / distractors live in DISJOINT vocab ranges so a query is unambiguous.
"""
import sys, time
import torch, torch.nn as nn, torch.nn.functional as F
from gpu_gamma_scan4 import GammaChunked
from gpu_gamma_scan3 import MHAttn

dev = "cuda" if torch.cuda.is_available() else "cpu"
K, VV, DD = 48, 48, 48                       # |keys| |values| |distractors|
KEY0, VAL0, DIS0 = 1, 1 + K, 1 + K + VV
VOCAB = 1 + K + VV + DD


def make_mqar(B, n_pairs, L, n_query, g):
    x = torch.randint(DIS0, DIS0 + DD, (B, L), generator=g)       # distractor filler
    y = torch.full((B, L), -100, dtype=torch.long)
    tail0 = 2 * n_pairs
    for b in range(B):
        keys = torch.randperm(K, generator=g)[:n_pairs] + KEY0
        vals = torch.randint(VAL0, VAL0 + VV, (n_pairs,), generator=g)
        x[b, 0:tail0:2] = keys
        x[b, 1:tail0:2] = vals
        qpos = torch.randperm(L - tail0, generator=g)[:n_query] + tail0
        qj = torch.randint(0, n_pairs, (n_query,), generator=g)
        x[b, qpos] = keys[qj]
        y[b, qpos] = vals[qj]
    return x.to(dev), y.to(dev)


class Block(nn.Module):
    def __init__(self, D, mixer):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.mix = mixer
        self.ff = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))

    def forward(self, x):
        x = x + self.mix(self.n1(x))
        return x + self.ff(self.n2(x))


class LM(nn.Module):
    def __init__(self, V, D, nlayer, mixer_fn):
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.pos = nn.Parameter(torch.zeros(1, 4096, D))
        self.blocks = nn.ModuleList([Block(D, mixer_fn(D)) for _ in range(nlayer)])
        self.norm = nn.LayerNorm(D)
        self.head = nn.Linear(D, V)

    def forward(self, idx):
        h = self.emb(idx) + self.pos[:, :idx.shape[1]]
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.norm(h))


def run(mixer_fn, n_pairs, L, D=128, nlayer=2, steps=1500, B=32, n_query=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    model = LM(VOCAB, D, nlayer, mixer_fn).to(dev)
    npar = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(steps):
        x, y = make_mqar(B, n_pairs, L, n_query, g)
        loss = F.cross_entropy(model(x).reshape(-1, VOCAB), y.reshape(-1), ignore_index=-100)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for _ in range(20):
            x, y = make_mqar(B, n_pairs, L, n_query, g)
            pred = model(x).argmax(-1)
            m = y != -100
            correct += (pred[m] == y[m]).sum().item()
            total += m.sum().item()
    return correct / total, npar


def sweep(title, configs, label):
    print(title)
    print(f"   {label:>10}{'gamma recall':>15}{'attn recall':>14}{'winner':>12}")
    gamma_fn = lambda D: GammaChunked(D, chunk=512)
    attn_fn = lambda D: MHAttn(D, 4)
    for cfg, npairs, L in configs:
        t0 = time.perf_counter()
        ga, gn = run(gamma_fn, npairs, L)
        aa, an = run(attn_fn, npairs, L)
        win = "gamma" if ga > aa + 0.02 else ("attn" if aa > ga + 0.02 else "tie")
        print(f"   {cfg:>10}{ga:>14.1%}{aa:>14.1%}{win:>12}   "
              f"({time.perf_counter()-t0:.0f}s, ~{gn/1e3:.0f}K params)")
    print()


if __name__ == "__main__":
    print(f"device: {torch.cuda.get_device_name(0) if dev=='cuda' else 'cpu'}  "
          f"vocab={VOCAB}, 2-layer, D=128, matched params\n")
    sweep("A. STATE PRESSURE  (recall vs #kv-pairs, length fixed at 256)",
          [(f"{p} pairs", p, 256) for p in (4, 8, 16, 32)], "pairs")
    sweep("B. LONG RANGE  (recall vs sequence length, 8 pairs fixed)",
          [(f"L={L}", 8, L) for L in (128, 512, 1024)], "length")
