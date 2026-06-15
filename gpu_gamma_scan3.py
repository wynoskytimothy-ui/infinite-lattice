#!/usr/bin/env python3
"""Two fairness checks before trusting the gamma-scan result:
  A. MULTI-HEAD  -- single-head attention is a weak baseline. Give attention its
     real form (4 and 8 heads, SAME param budget) and see if it closes the gap.
  B. 16k SPEED   -- lock the O(T logT) crossover: one more doubling past 8k.
"""
import time, math
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
from gpu_gamma_scan import (GammaMixer, LM, load_text, train_eval, bench, dev)

torch.manual_seed(0)


class MHAttn(nn.Module):
    """Causal multi-head attention, H heads. Params identical to single-head
    (q,k,v,o are all D x D regardless of H) -- the FAIR matched comparison."""
    def __init__(self, D, H):
        super().__init__()
        self.H, self.D = H, D
        self.q, self.k, self.v, self.o = (nn.Linear(D, D) for _ in range(4))

    def forward(self, x):
        B, T, D = x.shape
        sh = lambda t: t.view(B, T, self.H, D // self.H).transpose(1, 2)
        q, k, v = sh(self.q(x)), sh(self.k(x)), sh(self.v(x))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(y.transpose(1, 2).reshape(B, T, D))


def multihead_accuracy(T=512, steps=400):
    text = load_text()
    print(f"A. MULTI-HEAD accuracy  T={T}, matched params, {steps} steps, "
          f"{len(text):,} chars")
    rows = []
    gl, gp, gt, gn = train_eval(lambda D: GammaMixer(D, use_scan=True), text, T, steps=steps)
    rows.append(("gamma-scan", gp, gt, gn))
    for H in (1, 4, 8):
        al, ap, at, an = train_eval(lambda D: MHAttn(D, H), text, T, steps=steps)
        rows.append((f"attn-{H}head", ap, at, an))
    print(f"   {'model':>14}{'perplexity':>13}{'train s':>10}{'params':>11}")
    for name, ppl, tt, npar in rows:
        print(f"   {name:>14}{ppl:>13.3f}{tt:>10.1f}{npar:>11,}")
    best_attn = min(r[1] for r in rows if r[0].startswith("attn"))
    g = rows[0][1]
    verdict = ("gamma still wins" if g < best_attn else "attention closes it")
    print(f"   -> best attention ppl {best_attn:.3f} vs gamma {g:.3f}  "
          f"({best_attn - g:+.3f})  =>  {verdict}\n")


def speed_16k():
    print("B. 16k SPEED  (lock the crossover, fwd+bwd ms, B=16 D=256)")
    D = 256
    attn = MHAttn(D, 8).to(dev)            # strongest attention (8 head, flash)
    g = GammaMixer(D, use_scan=True).to(dev)
    print(f"   {'T':>7}{'attn-8head':>13}{'gamma-scan':>13}{'scan vs attn':>15}")
    for T in (4096, 8192, 16384):
        ta = bench(attn, T, D=D, iters=8)
        ts = bench(g, T, D=D, iters=8)
        print(f"   {T:>7}{ta:>11.2f}ms{ts:>11.2f}ms{ta/ts:>14.2f}x")
    print("   (>1.00x = scan faster; the gap should WIDEN each doubling)\n")


if __name__ == "__main__":
    print(f"device: {torch.cuda.get_device_name(0) if dev=='cuda' else 'cpu'}\n")
    multihead_accuracy()
    speed_16k()
