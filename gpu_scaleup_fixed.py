#!/usr/bin/env python3
"""CRITICAL FIX: the scale-up eval scored predictions against MISMATCHED targets --
`model(batch(src)[0])` and `batch(src)[1]` were two SEPARATE random draws, so input and
target came from different windows. A confident (well-trained) model scores WORSE than
random on mismatched targets, which faked 'divergence'. Training was always correct
(x,y from one batch() call); only eval was broken. Fix eval to use ONE matched batch and
re-run the real comparison: attention vs naive-gamma(good init) vs faithful Mamba block.
"""
import math
from collections import Counter
import torch, torch.nn as nn, torch.nn.functional as F
from gpu_gamma_scan3 import MHAttn
from gpu_scaleup import tokenize
from gpu_scaleup3 import gather_shuffled, LM, GammaInit
from gpu_mamba_lite import MambaLite

dev = "cuda" if torch.cuda.is_available() else "cpu"


def fit(mk, depth, data, V, D=256, T=256, steps=2500, B=16, seed=0, clip=1.0, wd=0.1):
    torch.manual_seed(seed)
    n = int(len(data) * 0.9); tr, va = data[:n], data[n:]
    model = LM(V, D, [mk(D) for _ in range(depth)]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=wd)
    gen = torch.Generator().manual_seed(seed)

    def batch(src):
        ix = torch.randint(0, len(src) - T - 1, (B,), generator=gen)
        return (torch.stack([src[i:i + T] for i in ix]).to(dev),
                torch.stack([src[i + 1:i + T + 1] for i in ix]).to(dev))

    model.train()
    for _ in range(steps):
        x, y = batch(tr)
        loss = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()

    model.eval()
    with torch.no_grad():
        tr_l, va_l = [], []
        for _ in range(15):
            x, y = batch(tr)                       # ONE call -> matched x,y
            tr_l.append(F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1)).item())
        for _ in range(30):
            x, y = batch(va)
            va_l.append(F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1)).item())
    return math.exp(sum(tr_l)/len(tr_l)), math.exp(sum(va_l)/len(va_l))


def main():
    toks = tokenize(gather_shuffled())
    vocab = ["<unk>"] + [w for w, _ in Counter(toks).most_common(5999)]
    stoi = {w: i for i, w in enumerate(vocab)}; V = len(vocab)
    data = torch.tensor([stoi.get(w, 0) for w in toks], dtype=torch.long)
    print(f"word-level, {len(toks):,} tokens, vocab {V}, D=256 T=256, 2500 steps "
          f"(eval FIXED: matched x,y)\n")
    models = {"attention": lambda D: MHAttn(D, 4),
              "gamma (good init)": GammaInit,
              "mamba-lite": lambda D: MambaLite(D)}
    print(f"   {'depth':>6}{'model':>20}{'train ppl':>12}{'val ppl':>11}")
    for depth in (1, 2, 4):
        for name, mk in models.items():
            tp, vp = fit(mk, depth, data, V)
            print(f"   {depth:>6}{name:>20}{tp:>12.1f}{vp:>11.1f}")
        print()
    print("   sane now if train/val are BELOW random (~6000) and train <= val. "
          "real comparison at last.")


if __name__ == "__main__":
    main()
