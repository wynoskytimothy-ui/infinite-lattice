#!/usr/bin/env python3
"""Gamma trails attention at exact recall (28%/18% vs 100%) -- the diagonal-state
wall. Does the formula recover recall with the STANDARD fixes the field uses?
  - gamma + short causal conv  (the Mamba ingredient: local mixing feeds selection)
  - hybrid: one gamma layer + one attention layer (recurrence + a little lookup)
All matched ~params, same recall task, trained to convergence.
"""
import time
import torch, torch.nn as nn, torch.nn.functional as F
from gpu_mqar import make_mqar, Block, VOCAB, dev
from gpu_gamma_scan4 import gamma_scan_chunked
from gpu_gamma_scan3 import MHAttn


class GammaMix(nn.Module):
    def __init__(self, D, chunk=512):
        super().__init__()
        self.vp, self.gf, self.gb, self.o = (nn.Linear(D, D) for _ in range(4))
        self.chunk = chunk

    def forward(self, x):
        a = torch.exp(-F.softplus(self.gb(x)))
        u = torch.sigmoid(self.gf(x)) * self.vp(x)
        return self.o(gamma_scan_chunked(a, u, self.chunk))


class GammaConvMix(nn.Module):
    """Gamma with a causal depthwise conv first -- lets the gate at t see a short
    window (t-k..t), the local mixing Mamba uses to form induction."""
    def __init__(self, D, k=4, chunk=512):
        super().__init__()
        self.conv = nn.Conv1d(D, D, k, groups=D, padding=k - 1)
        self.k = k
        self.vp, self.gf, self.gb, self.o = (nn.Linear(D, D) for _ in range(4))
        self.chunk = chunk

    def forward(self, x):
        T = x.shape[1]
        xc = self.conv(x.transpose(1, 2))[..., :T].transpose(1, 2)
        a = torch.exp(-F.softplus(self.gb(xc)))
        u = torch.sigmoid(self.gf(xc)) * self.vp(xc)
        return self.o(gamma_scan_chunked(a, u, self.chunk))


class MultiLM(nn.Module):
    def __init__(self, V, D, mixers):
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.pos = nn.Parameter(torch.zeros(1, 4096, D))
        self.blocks = nn.ModuleList([Block(D, m) for m in mixers])
        self.norm = nn.LayerNorm(D)
        self.head = nn.Linear(D, V)

    def forward(self, idx):
        h = self.emb(idx) + self.pos[:, :idx.shape[1]]
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.norm(h))


def run(name, mixers_fn, n_pairs, L, steps=6000, D=128, B=32, n_query=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    model = MultiLM(VOCAB, D, mixers_fn(D)).to(dev)
    npar = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    t0 = time.perf_counter()
    for _ in range(steps):
        x, y = make_mqar(B, n_pairs, L, n_query, g)
        loss = F.cross_entropy(model(x).reshape(-1, VOCAB), y.reshape(-1), ignore_index=-100)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval(); c = t = 0
    with torch.no_grad():
        for _ in range(40):
            x, y = make_mqar(B, n_pairs, L, n_query, g)
            pred = model(x).argmax(-1); m = y != -100
            c += (pred[m] == y[m]).sum().item(); t += m.sum().item()
    acc = c / t
    print(f"   {name:>22}{acc:>10.0%}   ({npar/1e3:.0f}K params, {time.perf_counter()-t0:.0f}s)")
    return acc


if __name__ == "__main__":
    print(f"device: {torch.cuda.get_device_name(0) if dev=='cuda' else 'cpu'}\n")
    NP, L = 8, 128
    print(f"recall @ {NP} pairs, L={L} (trained to convergence):")
    run("attention x2 (ref)", lambda D: [MHAttn(D, 4), MHAttn(D, 4)], NP, L)
    run("gamma x2 (baseline)", lambda D: [GammaMix(D), GammaMix(D)], NP, L)
    run("gamma+conv x2", lambda D: [GammaConvMix(D), GammaConvMix(D)], NP, L)
    run("hybrid gamma->attn", lambda D: [GammaMix(D), MHAttn(D, 4)], NP, L)
    run("hybrid gamma+conv->attn", lambda D: [GammaConvMix(D), MHAttn(D, 4)], NP, L)
