#!/usr/bin/env python3
"""Last tuning pass: the two failures bracket the answer -- a~0.5 forgets everything
(underfit), a~0.998+EMA admits nothing (underfit). The fix is a MODERATE multi-timescale
init: a in ~[0.5, 0.95] (memory 1.5..20 tokens), bounded EMA, output norm, grad clip.
Does a stable deep gamma now FIT and compete with attention at word-level?
"""
import math
from collections import Counter
import torch, torch.nn as nn, torch.nn.functional as F
from gpu_gamma_scan4 import gamma_scan_chunked
from gpu_gamma_scan3 import MHAttn
from gpu_scaleup import tokenize
from gpu_scaleup3 import gather_shuffled
from gpu_scaleup4 import fit, dev


class GammaEMA2(nn.Module):
    def __init__(self, D, chunk=256):
        super().__init__()
        self.vp, self.gf, self.gb, self.o = (nn.Linear(D, D) for _ in range(4))
        self.cnorm = nn.LayerNorm(D)
        self.chunk = chunk
        with torch.no_grad():
            self.gb.weight.mul_(0.2)
            self.gb.bias.copy_(-torch.empty(D).uniform_(0.0, 3.0))   # a in ~[0.5, 0.95]

    def forward(self, x):
        a = torch.exp(-F.softplus(self.gb(x)))
        v = torch.sigmoid(self.gf(x)) * self.vp(x)
        C = gamma_scan_chunked(a, (1.0 - a) * v, self.chunk)
        return self.o(self.cnorm(C))


def main():
    toks = tokenize(gather_shuffled())
    vocab = ["<unk>"] + [w for w, _ in Counter(toks).most_common(5999)]
    stoi = {w: i for i, w in enumerate(vocab)}; V = len(vocab)
    data = torch.tensor([stoi.get(w, 0) for w in toks], dtype=torch.long)
    print(f"word-level, {len(toks):,} tokens, vocab {V}, moderate-init bounded EMA, "
          f"grad-clip 1.0\n")
    print(f"   {'depth':>6}{'attn val':>11}{'gamma val':>12}{'edge':>9}"
          f"{'attn train':>13}{'gamma train':>14}")
    for depth in (1, 2, 4):
        a_tr, a_v, _ = fit(lambda D: MHAttn(D, 4), depth, data, V)
        g_tr, g_v, _ = fit(GammaEMA2, depth, data, V)
        flag = "" if g_tr < 6000 else "  <-still diverged"
        print(f"   {depth:>6}{a_v:>11.1f}{g_v:>12.1f}{(a_v-g_v)/a_v:>+8.0%}"
              f"{a_tr:>13.1f}{g_tr:>14.1f}{flag}")
    print("\n   gamma train < attn train ~ fits as well; val edge ~ generalises as well.")


if __name__ == "__main__":
    main()
