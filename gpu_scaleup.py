#!/usr/bin/env python3
"""Does the gamma LM win SURVIVE depth and token-level? The 1-block char-LM win is
suggestive; this scales up: word-level, multi-layer, real corpus, matched params.
  - DEPTH sweep  {2,4,6} layers: pure-attention vs pure-gamma -- does the gap hold?
  - HYBRID       gamma bulk + ~1 attention / 3 layers: best of both (LM + recall)?
Tied embeddings, pre-norm transformer blocks, identical everything but the mixer.
"""
import sys, re, time, math, glob
import torch, torch.nn as nn, torch.nn.functional as F
from gpu_gamma_scan4 import GammaChunked
from gpu_gamma_scan3 import MHAttn
from gpu_mqar import Block

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)


def gather_corpus(cap=2_000_000):
    text, files = "", sorted(glob.glob("*.md") + glob.glob("*.py") +
                             glob.glob("derivations/*.md") + glob.glob("core/*.py"))
    for fn in files:
        try:
            with open(fn, encoding="utf-8", errors="ignore") as f:
                text += f.read() + "\n"
        except Exception:
            pass
        if len(text) >= cap:
            break
    return text[:cap]


def tokenize(text):
    return re.findall(r"[A-Za-z]+|[0-9]+|[^\sA-Za-z0-9]", text)


def make_mixer(kind, D):
    return GammaChunked(D, chunk=256) if kind == "g" else MHAttn(D, 4)


def build_spec(depth, mode):
    if mode == "attn":
        return ["a"] * depth
    if mode == "gamma":
        return ["g"] * depth
    spec = ["g"] * depth
    for i in range(depth):
        if (i + 1) % 3 == 0:
            spec[i] = "a"
    if "a" not in spec:
        spec[-1] = "a"
    return spec


class LM(nn.Module):
    def __init__(self, V, D, spec):
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.pos = nn.Parameter(torch.zeros(1, 512, D))
        self.blocks = nn.ModuleList([Block(D, make_mixer(k, D)) for k in spec])
        self.norm = nn.LayerNorm(D)
        self.head = nn.Linear(D, V, bias=False)
        self.head.weight = self.emb.weight

    def forward(self, idx):
        h = self.emb(idx) + self.pos[:, :idx.shape[1]]
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.norm(h))


def train_eval(spec, data, V, D=256, T=256, steps=1500, B=16, seed=0):
    torch.manual_seed(seed)
    n = int(len(data) * 0.9)
    tr, va = data[:n], data[n:]
    model = LM(V, D, spec).to(dev)
    npar = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    gen = torch.Generator().manual_seed(seed)

    def batch(src):
        ix = torch.randint(0, len(src) - T - 1, (B,), generator=gen)
        x = torch.stack([src[i:i + T] for i in ix]).to(dev)
        y = torch.stack([src[i + 1:i + T + 1] for i in ix]).to(dev)
        return x, y

    model.train(); t0 = time.perf_counter()
    for _ in range(steps):
        x, y = batch(tr)
        loss = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    torch.cuda.synchronize(); tt = time.perf_counter() - t0

    model.eval()
    with torch.no_grad():
        ls = [F.cross_entropy(model(batch(va)[0]).reshape(-1, V),
                              batch(va)[1].reshape(-1)).item() for _ in range(30)]
    return math.exp(sum(ls) / len(ls)), tt, npar


def main():
    text = gather_corpus()
    toks = tokenize(text)
    from collections import Counter
    freq = Counter(toks)
    VOCAB_CAP = 6000
    vocab = ["<unk>"] + [w for w, _ in freq.most_common(VOCAB_CAP - 1)]
    stoi = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)
    data = torch.tensor([stoi.get(w, 0) for w in toks], dtype=torch.long)
    print(f"corpus: {len(text):,} chars -> {len(toks):,} word-tokens, vocab {V} "
          f"(<unk> {sum(1 for w in toks if w not in stoi)/len(toks):.0%}), "
          f"D=256 T=256, {1500} steps\n")

    print("DEPTH sweep -- validation perplexity (lower better), matched params:")
    print(f"   {'depth':>6}{'attention':>12}{'gamma':>11}{'gamma edge':>13}{'attn s':>9}{'gamma s':>9}")
    for depth in (2, 4, 6):
        ap, at, an = train_eval(build_spec(depth, "attn"), data, V)
        gp, gt, gn = train_eval(build_spec(depth, "gamma"), data, V)
        edge = f"{(ap-gp)/ap:+.1%}"
        print(f"   {depth:>6}{ap:>12.2f}{gp:>11.2f}{edge:>13}{at:>8.0f}s{gt:>8.0f}s")

    print("\nHYBRID (gamma bulk + ~1 attention / 3 layers) vs the two pure stacks:")
    print(f"   {'depth':>6}{'spec':>16}{'perplexity':>13}{'params':>11}")
    for depth in (4, 6):
        for mode in ("attn", "gamma", "hybrid"):
            spec = build_spec(depth, mode)
            pp, tt, npar = train_eval(spec, data, V)
            print(f"   {depth:>6}{''.join(spec):>16}{pp:>13.2f}{npar:>11,}")
        print()


if __name__ == "__main__":
    main()
