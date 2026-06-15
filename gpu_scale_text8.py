#!/usr/bin/env python3
"""SCALE TEST on real text: text8 (100M chars of Wikipedia prose -- the opposite of the
repo's own text). Char-level, multi-layer, ~15-19M params, matched across mixers.
Does the gamma LM win survive off repo-text and at scale?  attention vs gamma vs hybrid.
Metric: bits-per-char (bpc) on held-out val (random=log2(27)=4.75; good char models ~1.3).
Eval is the FIXED matched-(x,y) kind. Usage: python gpu_scale_text8.py [smoke|full]
"""
import sys, time, math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from gpu_gamma_scan3 import MHAttn
from gpu_scaleup3 import GammaInit
from gpu_mqar import Block

dev = "cuda" if torch.cuda.is_available() else "cpu"
LN2 = math.log(2.0)


def load_text8(ntrain=90_000_000, nval=5_000_000):
    raw = np.frombuffer(open("text8", "rb").read(ntrain + nval), dtype=np.uint8)
    ids = np.where(raw == 32, 0, raw.astype(np.int16) - 96).astype(np.uint8)  # ' '->0, a-z->1..26
    d = torch.from_numpy(ids)
    return d[:ntrain], d[ntrain:ntrain + nval], 27


def make_mixer(k, D):
    return MHAttn(D, 8) if k == "a" else GammaInit(D, chunk=256)


def build_spec(depth, mode):
    if mode == "attn":
        return ["a"] * depth
    if mode == "gamma":
        return ["g"] * depth
    return ["a" if (i + 1) % 3 == 0 else "g" for i in range(depth)]   # hybrid: attn every 3rd


class LM(nn.Module):
    def __init__(self, V, D, spec, maxT=1024):
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.pos = nn.Parameter(torch.zeros(1, maxT, D))
        self.blocks = nn.ModuleList([Block(D, make_mixer(k, D)) for k in spec])
        self.norm = nn.LayerNorm(D)
        self.head = nn.Linear(D, V)

    def forward(self, idx):
        h = self.emb(idx) + self.pos[:, :idx.shape[1]]
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.norm(h))


def batch(src, T, B, gen):
    ix = torch.randint(0, len(src) - T - 1, (B,), generator=gen)
    x = torch.stack([src[i:i + T] for i in ix]).long().to(dev)
    y = torch.stack([src[i + 1:i + T + 1] for i in ix]).long().to(dev)
    return x, y


@torch.no_grad()
def eval_bpc(model, val, V, T, B, gen, nb=40):
    model.eval()
    ls = []
    for _ in range(nb):
        x, y = batch(val, T, B, gen)                       # ONE matched call (fixed eval)
        ls.append(F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1)).item())
    model.train()
    return (sum(ls) / len(ls)) / LN2


def fit(spec, train, val, V, D, T, steps, B, lr=3e-4, log_every=1000, seed=0, clip=1.0):
    torch.manual_seed(seed)
    model = LM(V, D, spec).to(dev)
    npar = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps,
                                                pct_start=0.05)
    gen = torch.Generator().manual_seed(seed)
    name = "".join(spec)
    print(f"  [{name}] {npar/1e6:.1f}M params -- training {steps} steps", flush=True)
    model.train(); t0 = time.perf_counter()
    for step in range(1, steps + 1):
        x, y = batch(train, T, B, gen)
        loss = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip); opt.step(); sched.step()
        if step % log_every == 0:
            el = time.perf_counter() - t0
            print(f"    [{name}] step {step}/{steps}  val bpc {eval_bpc(model, val, V, T, B, gen):.4f}"
                  f"  ({el:.0f}s, {1000*el/step:.0f}ms/step)", flush=True)
    bpc = eval_bpc(model, val, V, T, B, gen, nb=80)
    print(f"  [{name}] FINAL val bpc {bpc:.4f}  ({time.perf_counter()-t0:.0f}s)\n", flush=True)
    return bpc, npar


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    if mode == "smoke":
        cfg = dict(D=256, depth=2, T=256, steps=400, B=32, ntrain=5_000_000, nval=500_000)
    else:
        cfg = dict(D=512, depth=6, T=512, steps=12000, B=32, ntrain=90_000_000, nval=5_000_000)
    print(f"text8 scale test [{mode}] -- D={cfg['D']} depth={cfg['depth']} T={cfg['T']} "
          f"steps={cfg['steps']} B={cfg['B']}\n", flush=True)
    train, val, V = load_text8(cfg["ntrain"], cfg["nval"])
    print(f"loaded text8: train {len(train)/1e6:.0f}M chars, val {len(val)/1e6:.0f}M, vocab {V}\n", flush=True)
    res = {}
    for label in ("attn", "gamma", "hybrid"):
        spec = build_spec(cfg["depth"], label)
        bpc, npar = fit(spec, train, val, V, cfg["D"], cfg["T"], cfg["steps"], cfg["B"])
        res[label] = bpc
    print("=" * 56)
    print(f"text8 RESULT (val bits-per-char, lower better) -- {mode}:")
    for k in ("attn", "gamma", "hybrid"):
        edge = "" if k == "attn" else f"  ({(res['attn']-res[k])/res['attn']:+.1%} vs attn)"
        print(f"   {k:>8}  {res[k]:.4f} bpc{edge}")
    best = min(res, key=res.get)
    print(f"   -> lowest bpc: {best}")


if __name__ == "__main__":
    main()
