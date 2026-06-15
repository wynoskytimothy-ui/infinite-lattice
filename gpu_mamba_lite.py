#!/usr/bin/env python3
"""Option-2 done honestly without mamba-ssm/WSL: a faithful Mamba-style block in pure
PyTorch, reusing our correct chunked scan. The ingredients my naive gamma lacked, all
together and correctly initialised -- THIS is "the engineering that makes deep SSMs work":
  - selective param:  dt = softplus(dt_proj(h)+dt_bias),  A = -exp(A_log),  a = exp(dt*A)
                      (dt_bias init so dt in [1e-3,0.1] -> a near 1, multi-timescale)
  - short causal depthwise conv before the SSM (local mixing)
  - SiLU output gate + skip connection + out projection
Question: does a DEEP stack of these train STABLY (train ppl << random ~6000) and
compete with attention at word-level, where the naive gamma diverged?
"""
import math
from collections import Counter
import torch, torch.nn as nn, torch.nn.functional as F
from gpu_gamma_scan4 import gamma_scan_chunked
from gpu_gamma_scan3 import MHAttn
from gpu_scaleup import tokenize
from gpu_scaleup3 import gather_shuffled, LM
from gpu_scaleup4 import fit, dev


class MambaLite(nn.Module):
    def __init__(self, D, expand=1, dconv=4, chunk=256):
        super().__init__()
        E = expand * D
        self.in_proj = nn.Linear(D, 2 * E)
        self.conv = nn.Conv1d(E, E, dconv, groups=E, padding=dconv - 1)
        self.dt_proj = nn.Linear(E, E)
        self.out_proj = nn.Linear(E, D)
        self.A_log = nn.Parameter(torch.log(torch.empty(E).uniform_(1.0, 16.0)))
        self.D_skip = nn.Parameter(torch.ones(E))
        self.chunk = chunk
        with torch.no_grad():
            dt = torch.empty(E).uniform_(1e-3, 0.1)
            self.dt_proj.bias.copy_(torch.log(torch.expm1(dt)))   # softplus(bias)=dt
            self.dt_proj.weight.mul_(0.1)

    def forward(self, x):
        T = x.shape[1]
        h, gate = self.in_proj(x).chunk(2, dim=-1)
        h = self.conv(h.transpose(1, 2))[..., :T].transpose(1, 2)
        h = F.silu(h)
        dt = F.softplus(self.dt_proj(h))                 # (B,T,E) > 0
        a = torch.exp(dt * (-torch.exp(self.A_log)))     # in (0,1), near 1 for small dt
        C = gamma_scan_chunked(a, dt * h, self.chunk)    # selective scan (reused)
        y = (C + self.D_skip * h) * F.silu(gate)         # skip + output gate
        return self.out_proj(y)


def main():
    toks = tokenize(gather_shuffled())
    vocab = ["<unk>"] + [w for w, _ in Counter(toks).most_common(5999)]
    stoi = {w: i for i, w in enumerate(vocab)}; V = len(vocab)
    data = torch.tensor([stoi.get(w, 0) for w in toks], dtype=torch.long)
    pa = sum(p.numel() for p in LM(V, 256, [MHAttn(256, 4)]).parameters())
    pm = sum(p.numel() for p in LM(V, 256, [MambaLite(256)]).parameters())
    print(f"word-level, {len(toks):,} tokens, vocab {V}, D=256 T=256, 2500 steps\n"
          f"params/layer-ish: attn-block {pa:,}  mamba-block {pm:,}\n")
    print(f"   {'depth':>6}{'attn val':>11}{'mamba val':>12}{'edge':>9}"
          f"{'attn train':>13}{'mamba train':>14}")
    for depth in (1, 2, 4):
        a_tr, a_v, _ = fit(lambda D: MHAttn(D, 4), depth, data, V)
        m_tr, m_v, _ = fit(lambda D: MambaLite(D), depth, data, V)
        flag = "" if m_tr < 6000 else "  <-diverged"
        print(f"   {depth:>6}{a_v:>11.1f}{m_v:>12.1f}{(a_v-m_v)/a_v:>+8.0%}"
              f"{a_tr:>13.1f}{m_tr:>14.1f}{flag}")
    print("\n   mamba train << random ~6000 = the recipe trains stably where naive gamma diverged.")


if __name__ == "__main__":
    main()
