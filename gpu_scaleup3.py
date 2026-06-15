#!/usr/bin/env python3
"""The reversal was an INIT BUG: gamma's decay a=exp(-softplus(gb)) starts at ~0.5
(half-life 1 token) -- catastrophic forgetting, can't fit word-level (train ppl 11k
vs attention 1.4k). Fix it the way real SSMs do: init the decay near 1, multi-timescale
(channels spanning short..long memory, like Mamba's dt). Re-run the depth sweep with a
representative val split. Does properly-initialised gamma now FIT and compete?
"""
import re, time, math, glob, random
from collections import Counter
import torch, torch.nn as nn, torch.nn.functional as F
from gpu_gamma_scan4 import gamma_scan_chunked
from gpu_gamma_scan3 import MHAttn
from gpu_mqar import Block
from gpu_scaleup import tokenize

dev = "cuda" if torch.cuda.is_available() else "cpu"


def gather_shuffled(cap=2_000_000, seed=0):
    files = sorted(glob.glob("*.md") + glob.glob("*.py") +
                   glob.glob("derivations/*.md") + glob.glob("core/*.py"))
    random.Random(seed).shuffle(files)          # mix sources so the val tail is representative
    text = ""
    for fn in files:
        try:
            with open(fn, encoding="utf-8", errors="ignore") as f:
                text += f.read() + "\n"
        except Exception:
            pass
        if len(text) >= cap:
            break
    return text[:cap]


class GammaInit(nn.Module):
    """Gamma with SSM-style decay init: a starts in ~[0.88, 0.998] across channels."""
    def __init__(self, D, chunk=256):
        super().__init__()
        self.vp, self.gf, self.gb, self.o = (nn.Linear(D, D) for _ in range(4))
        self.chunk = chunk
        with torch.no_grad():
            self.gb.weight.mul_(0.05)                              # gentle input dependence early
            self.gb.bias.copy_(-torch.empty(D).uniform_(2.0, 6.0))  # softplus(-2..-6) -> a 0.88..0.998
            self.gf.bias.fill_(-1.0)                                # input gate starts ~0.27 (stable)

    def forward(self, x):
        a = torch.exp(-F.softplus(self.gb(x)))
        u = torch.sigmoid(self.gf(x)) * self.vp(x)
        return self.o(gamma_scan_chunked(a, u, self.chunk))


class GammaOld(nn.Module):
    """The bad-init version (a~0.5) for the before/after."""
    def __init__(self, D, chunk=256):
        super().__init__()
        self.vp, self.gf, self.gb, self.o = (nn.Linear(D, D) for _ in range(4))
        self.chunk = chunk

    def forward(self, x):
        a = torch.exp(-F.softplus(self.gb(x)))
        u = torch.sigmoid(self.gf(x)) * self.vp(x)
        return self.o(gamma_scan_chunked(a, u, self.chunk))


class LM(nn.Module):
    def __init__(self, V, D, mixers):
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.pos = nn.Parameter(torch.zeros(1, 512, D))
        self.blocks = nn.ModuleList([Block(D, m) for m in mixers])
        self.norm = nn.LayerNorm(D)
        self.head = nn.Linear(D, V, bias=False); self.head.weight = self.emb.weight

    def forward(self, idx):
        h = self.emb(idx) + self.pos[:, :idx.shape[1]]
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.norm(h))


def fit(mk, depth, data, V, D=256, T=256, steps=2500, B=16, seed=0):
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
        opt.zero_grad(); loss.backward(); opt.step()
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
    print(f"word-level (shuffled sources), {len(toks):,} tokens, vocab {V}, "
          f"D=256 T=256, 2500 steps\n")

    print("BEFORE/AFTER the init fix (depth 1) -- train ppl shows if it can FIT:")
    for name, mk in [("attention", lambda D: MHAttn(D, 4)),
                     ("gamma OLD (a~0.5)", GammaOld),
                     ("gamma INIT-FIX", GammaInit)]:
        tp, vp, tt = fit(mk, 1, data, V)
        print(f"   {name:>20}  train {tp:>8.1f}   val {vp:>8.1f}   ({tt:.0f}s)")

    print("\nDEPTH sweep with the fixed gamma -- val perplexity (lower better):")
    print(f"   {'depth':>6}{'attention':>12}{'gamma-fixed':>14}{'gamma edge':>13}")
    for depth in (2, 4, 6):
        ap_t, ap, _ = fit(lambda D: MHAttn(D, 4), depth, data, V)
        gp_t, gp, _ = fit(GammaInit, depth, data, V)
        print(f"   {depth:>6}{ap:>12.1f}{gp:>14.1f}{(ap-gp)/ap:>+12.1%}   "
              f"(train a/g {ap_t:.0f}/{gp_t:.0f})")


if __name__ == "__main__":
    main()
