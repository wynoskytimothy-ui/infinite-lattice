#!/usr/bin/env python3
"""Depth instability fix: the long-memory init made the independent-gate state blow up
through depth (train ppl > random = divergence). Make it a BOUNDED selective EMA
  C_t = a_t*C_{t-1} + (1-a_t)*v_t          (a weighted average -> magnitude-stable)
plus an output norm and gradient clipping -- the standard S4/Mamba stabilisers. Now
does a DEEP gamma stack train stably and compete with attention at word-level?
"""
import time, math, random
from collections import Counter
import torch, torch.nn as nn, torch.nn.functional as F
from gpu_gamma_scan4 import gamma_scan_chunked
from gpu_gamma_scan3 import MHAttn
from gpu_mqar import Block
from gpu_scaleup import tokenize
from gpu_scaleup3 import gather_shuffled, LM

dev = "cuda" if torch.cuda.is_available() else "cpu"


class GammaEMA(nn.Module):
    """Bounded selective EMA with multi-timescale long-memory init + output norm."""
    def __init__(self, D, chunk=256):
        super().__init__()
        self.vp, self.gf, self.gb, self.o = (nn.Linear(D, D) for _ in range(4))
        self.cnorm = nn.LayerNorm(D)
        self.chunk = chunk
        with torch.no_grad():
            self.gb.weight.mul_(0.05)
            self.gb.bias.copy_(-torch.empty(D).uniform_(2.0, 6.0))   # a in ~[0.88, 0.998]

    def forward(self, x):
        a = torch.exp(-F.softplus(self.gb(x)))
        v = torch.sigmoid(self.gf(x)) * self.vp(x)
        u = (1.0 - a) * v                              # EMA -> C bounded by v
        C = gamma_scan_chunked(a, u, self.chunk)
        return self.o(self.cnorm(C))


def fit(mk, depth, data, V, D=256, T=256, steps=2500, B=16, seed=0, clip=1.0):
    torch.manual_seed(seed)
    n = int(len(data) * 0.9); tr, va = data[:n], data[n:]
    model = LM(V, D, [mk(D) for _ in range(depth)]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    gen = torch.Generator().manual_seed(seed)

    def batch(src):
        ix = torch.randint(0, len(src) - T - 1, (B,), generator=gen)
        return (torch.stack([src[i:i + T] for i in ix]).to(dev),
                torch.stack([src[i + 1:i + T + 1] for i in ix]).to(dev))

    model.train(); t0 = time.perf_counter()
    for _ in range(steps):
        x, y = batch(tr)
        loss = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
    torch.cuda.synchronize(); tt = time.perf_counter() - t0
    model.eval()
    with torch.no_grad():
        trl = [F.cross_entropy(model(batch(tr)[0]).reshape(-1, V), batch(tr)[1].reshape(-1)).item() for _ in range(15)]
        val = [F.cross_entropy(model(batch(va)[0]).reshape(-1, V), batch(va)[1].reshape(-1)).item() for _ in range(30)]
    return math.exp(sum(trl)/len(trl)), math.exp(sum(val)/len(val)), tt


def main():
    toks = tokenize(gather_shuffled())
    vocab = ["<unk>"] + [w for w, _ in Counter(toks).most_common(5999)]
    stoi = {w: i for i, w in enumerate(vocab)}; V = len(vocab)
    data = torch.tensor([stoi.get(w, 0) for w in toks], dtype=torch.long)
    print(f"word-level, {len(toks):,} tokens, vocab {V}, D=256 T=256, 2500 steps, "
          f"grad-clip 1.0\n")
    print("DEPTH sweep -- bounded-EMA gamma vs attention (train ppl < random ~6000 = "
          "stable):")
    print(f"   {'depth':>6}{'attn val':>11}{'gamma val':>12}{'edge':>9}"
          f"{'attn train':>13}{'gamma train':>14}")
    for depth in (1, 2, 4, 6):
        a_tr, a_v, _ = fit(lambda D: MHAttn(D, 4), depth, data, V)
        g_tr, g_v, _ = fit(GammaEMA, depth, data, V)
        edge = f"{(a_v-g_v)/a_v:+.1%}"
        flag = "" if g_tr < 6000 else "  <-DIVERGED"
        print(f"   {depth:>6}{a_v:>11.1f}{g_v:>12.1f}{edge:>9}{a_tr:>13.1f}{g_tr:>14.1f}{flag}")


if __name__ == "__main__":
    main()
