#!/usr/bin/env python3
"""Burned twice this session (MQAR grokking variance, then the eval bug) -- so VERIFY
the reversed result across multiple seeds before claiming it. With correct eval the
SSM models beat attention at word-level, margin growing with depth. Robust, or a seed?
"""
from collections import Counter
import torch
from gpu_gamma_scan3 import MHAttn
from gpu_scaleup import tokenize
from gpu_scaleup3 import gather_shuffled, GammaInit
from gpu_mamba_lite import MambaLite
from gpu_scaleup_fixed import fit

dev = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    toks = tokenize(gather_shuffled())
    vocab = ["<unk>"] + [w for w, _ in Counter(toks).most_common(5999)]
    stoi = {w: i for i, w in enumerate(vocab)}; V = len(vocab)
    data = torch.tensor([stoi.get(w, 0) for w in toks], dtype=torch.long)
    print(f"word-level, {len(toks):,} tokens, vocab {V}, correct eval, seeds 0/1/2\n")
    models = {"attention": lambda D: MHAttn(D, 4),
              "gamma": GammaInit,
              "mamba-lite": lambda D: MambaLite(D)}
    seeds = (0, 1, 2)
    for depth in (2, 4):
        print(f"   depth {depth} -- val perplexity:")
        print(f"   {'model':>14}{'seed0':>9}{'seed1':>9}{'seed2':>9}{'mean':>9}")
        res = {}
        for name, mk in models.items():
            vs = [fit(mk, depth, data, V, seed=s)[1] for s in seeds]
            res[name] = sum(vs) / len(vs)
            print(f"   {name:>14}" + "".join(f"{v:>9.1f}" for v in vs) +
                  f"{res[name]:>9.1f}")
        win = min(res, key=res.get)
        print(f"   -> lowest mean ppl: {win} ({res[win]:.1f}); "
              f"attention {res['attention']:.1f}\n")


if __name__ == "__main__":
    main()
