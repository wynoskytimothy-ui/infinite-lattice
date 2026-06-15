#!/usr/bin/env python3
"""Clean read on the recall FIXES, in the RELIABLE regime (L=64, where attention
groks 100% every time) with MULTIPLE SEEDS -- MQAR groks sharply and seed-dependently,
so a single run can't separate capability from luck. Report per-seed + best.
Question: which architectures can do EXACT key->value binding?
"""
import time
import torch, torch.nn as nn, torch.nn.functional as F
from gpu_mqar import make_mqar, Block, VOCAB, dev
from gpu_mqar3 import GammaMix, GammaConvMix, MultiLM
from gpu_gamma_scan3 import MHAttn


def run(mixers_fn, n_pairs, L, steps, seed, D=128, B=32, n_query=8):
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    model = MultiLM(VOCAB, D, mixers_fn(D)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
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
    return c / t


MODELS = {
    "attention x2 (ref)":   lambda D: [MHAttn(D, 4), MHAttn(D, 4)],
    "gamma x2":             lambda D: [GammaMix(D), GammaMix(D)],
    "gamma+conv x2":        lambda D: [GammaConvMix(D), GammaConvMix(D)],
    "hybrid gamma->attn":   lambda D: [GammaMix(D), MHAttn(D, 4)],
}

if __name__ == "__main__":
    NP, L, STEPS, SEEDS = 8, 64, 6000, (0, 1, 2)
    print(f"device: {torch.cuda.get_device_name(0) if dev=='cuda' else 'cpu'}")
    print(f"exact recall @ {NP} pairs, L={L}, {STEPS} steps, seeds {SEEDS} "
          f"(reliable regime)\n")
    print(f"   {'model':>22}{'seed0':>8}{'seed1':>8}{'seed2':>8}{'best':>8}")
    for name, fn in MODELS.items():
        t0 = time.perf_counter()
        accs = [run(fn, NP, L, STEPS, s) for s in SEEDS]
        b = max(accs)
        print(f"   {name:>22}" + "".join(f"{a:>8.0%}" for a in accs) +
              f"{b:>8.0%}   ({time.perf_counter()-t0:.0f}s)")
    print("\n   best = can it EVER learn exact binding; consistent 100% = robustly.")
