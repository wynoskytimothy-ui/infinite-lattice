#!/usr/bin/env python3
"""Diagnostic before trusting any MQAR verdict: is gamma's ~chance recall a real
capability wall or just under-training? Train the EASIEST recall config to
convergence, print the accuracy curve. If attention -> ~100% and gamma stays flat
at chance, the wall is real (diagonal-SSM theory). If gamma climbs, the first run
was just under-trained.
"""
import time
import torch, torch.nn as nn, torch.nn.functional as F
from gpu_mqar import make_mqar, LM, VOCAB, dev
from gpu_gamma_scan4 import GammaChunked
from gpu_gamma_scan3 import MHAttn


def train_curve(name, mixer_fn, n_pairs, L, steps=6000, evalevery=1000,
                D=128, nlayer=2, B=32, n_query=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    model = LM(VOCAB, D, nlayer, mixer_fn).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    npar = sum(p.numel() for p in model.parameters())

    def evaluate():
        model.eval(); c = t = 0
        with torch.no_grad():
            for _ in range(40):
                x, y = make_mqar(B, n_pairs, L, n_query, g)
                pred = model(x).argmax(-1); m = y != -100
                c += (pred[m] == y[m]).sum().item(); t += m.sum().item()
        model.train(); return c / t

    print(f"   {name:>12} ({n_pairs} pairs, L={L}, {npar/1e3:.0f}K params):", end="", flush=True)
    t0 = time.perf_counter()
    model.train()
    for s in range(1, steps + 1):
        x, y = make_mqar(B, n_pairs, L, n_query, g)
        loss = F.cross_entropy(model(x).reshape(-1, VOCAB), y.reshape(-1), ignore_index=-100)
        opt.zero_grad(); loss.backward(); opt.step()
        if s % evalevery == 0:
            print(f"  {s//1000}k={evaluate():.0%}", end="", flush=True)
    print(f"   ({time.perf_counter()-t0:.0f}s)")


if __name__ == "__main__":
    print(f"device: {torch.cuda.get_device_name(0) if dev=='cuda' else 'cpu'}\n")
    gamma_fn = lambda D: GammaChunked(D, chunk=512)
    attn_fn = lambda D: MHAttn(D, 4)

    print("EASIEST recall (4 pairs, L=64) -- both SHOULD solve this if trainable:")
    train_curve("attention", attn_fn, 4, 64)
    train_curve("gamma", gamma_fn, 4, 64)
    print("\nHARDER (8 pairs, L=64) -- does attention's earlier collapse recover with steps?")
    train_curve("attention", attn_fn, 8, 64)
    train_curve("gamma", gamma_fn, 8, 64)
