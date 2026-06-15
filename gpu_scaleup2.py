#!/usr/bin/env python3
"""Diagnose the reversal: at word-level+depth, gamma lost and got worse with depth.
Is that REAL capability, or (a) overfitting on a small corpus, (b) deep-gamma being
hard to optimize, (c) word-granularity (not depth) flipping it? Report TRAIN and VAL
perplexity (can't-fit vs overfit) and isolate DEPTH-1 word-level (granularity alone).
"""
import re, time, math, glob
from collections import Counter
import torch, torch.nn as nn, torch.nn.functional as F
from gpu_scaleup import gather_corpus, tokenize, LM, build_spec, dev


def fit(spec, data, V, D=256, T=256, steps=2500, B=16, seed=0, wd=0.1):
    torch.manual_seed(seed)
    n = int(len(data) * 0.9); tr, va = data[:n], data[n:]
    model = LM(V, D, spec).to(dev)
    npar = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=wd)
    gen = torch.Generator().manual_seed(seed)

    def batch(src):
        ix = torch.randint(0, len(src) - T - 1, (B,), generator=gen)
        return (torch.stack([src[i:i + T] for i in ix]).to(dev),
                torch.stack([src[i + 1:i + T + 1] for i in ix]).to(dev))

    model.train(); t0 = time.perf_counter()
    for _ in range(steps):
        x, y = batch(tr)
        loss = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    torch.cuda.synchronize(); tt = time.perf_counter() - t0

    model.eval()
    with torch.no_grad():
        trl = [F.cross_entropy(model(batch(tr)[0]).reshape(-1, V), batch(tr)[1].reshape(-1)).item() for _ in range(15)]
        val = [F.cross_entropy(model(batch(va)[0]).reshape(-1, V), batch(va)[1].reshape(-1)).item() for _ in range(30)]
    return math.exp(sum(trl)/len(trl)), math.exp(sum(val)/len(val)), tt, npar


def main():
    text = gather_corpus()
    toks = tokenize(text)
    vocab = ["<unk>"] + [w for w, _ in Counter(toks).most_common(5999)]
    stoi = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)
    data = torch.tensor([stoi.get(w, 0) for w in toks], dtype=torch.long)
    print(f"word-level, {len(toks):,} tokens, vocab {V}, D=256 T=256, 2500 steps, wd=0.1\n")
    print("TRAIN vs VAL perplexity by depth (train<<val = overfit; both high = can't fit):")
    print(f"   {'depth':>6}{'mixer':>8}{'train ppl':>12}{'val ppl':>11}{'gap':>8}{'s':>7}")
    for depth in (1, 2, 4):
        for mode in ("attn", "gamma"):
            tp, vp, tt, npar = fit(build_spec(depth, mode), data, V)
            print(f"   {depth:>6}{mode:>8}{tp:>12.1f}{vp:>11.1f}{vp/tp:>8.2f}x{tt:>6.0f}s")
        print()
    print("   depth-1 isolates GRANULARITY (no stacking): if gamma loses here too,")
    print("   word-level itself flipped it; if gamma wins here, depth/optim is the cause.")


if __name__ == "__main__":
    main()
