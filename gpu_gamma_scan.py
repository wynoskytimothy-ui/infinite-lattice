#!/usr/bin/env python3
"""Gamma-ODE with a PARALLEL SCAN instead of the naive O(T) Python loop.

The gate recurrence  C_t = a_t * C_{t-1} + u_t  is a composition of affine maps
f_t(C) = a_t*C + u_t, and affine-map composition is ASSOCIATIVE:
    (A_P,X_P) . (A_Q,X_Q) = (A_P*A_Q,  A_Q*X_P + X_Q)
so a Hillis-Steele scan computes every C_t in log2(T) parallel steps (each a single
elementwise op over the whole sequence) rather than T sequential kernel launches.

Three honest deliverables:
  1. CORRECTNESS  -- scan output == sequential-loop output (same math)
  2. SPEED        -- fwd+bwd wall-time vs attention as context T grows; O(T logT)
                     scan should overtake O(T^2) attention at long context
  3. ACCURACY     -- char-LM val perplexity at long context, params matched,
                     gamma-scan block vs attention block
"""
import sys, time, math
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)


# ---------------------------------------------------------------- the scan ----
def shift(x, d, fill):
    """Shift along time (dim=1) by d; pad the FRONT with the identity element."""
    B, T, D = x.shape
    return torch.cat([x.new_full((B, d, D), fill), x[:, :T - d]], dim=1)


def gamma_scan(a, u):
    """Inclusive scan of C_t = a_t*C_{t-1} + u_t. Hillis-Steele, log2(T) steps.
    A carries the running decay (identity 1), X the running state (identity 0)."""
    A, X, T, d = a, u, a.shape[1], 1
    while d < T:
        X = A * shift(X, d, 0.0) + X      # right.A * left.X + right.X  (uses OLD A)
        A = shift(A, d, 1.0) * A          # left.A * right.A
        d *= 2
    return X


def gamma_loop(a, u):
    """Reference: the naive sequential recurrence (what made it 15x slow before)."""
    B, T, D = a.shape
    C = a.new_zeros(B, D)
    out = []
    for t in range(T):
        C = a[:, t] * C + u[:, t]
        out.append(C)
    return torch.stack(out, dim=1)


# ------------------------------------------------------------- the mixers ----
class GammaMixer(nn.Module):
    """Selective gated state-space mixer. a_t = exp(-softplus(.)) in (0,1) is the
    Gamma-break decay; b_t = sigmoid(.) the Gamma-form input gate."""
    def __init__(self, D, use_scan=True):
        super().__init__()
        self.vp, self.gf, self.gb, self.o = (nn.Linear(D, D) for _ in range(4))
        self.use_scan = use_scan

    def forward(self, x):
        a = torch.exp(-F.softplus(self.gb(x)))
        u = torch.sigmoid(self.gf(x)) * self.vp(x)
        C = gamma_scan(a, u) if self.use_scan else gamma_loop(a, u)
        return self.o(C)


class AttnMixer(nn.Module):
    """Matched-parameter causal attention (single head, flash SDPA)."""
    def __init__(self, D):
        super().__init__()
        self.q, self.k, self.v, self.o = (nn.Linear(D, D) for _ in range(4))

    def forward(self, x):
        q, k, v = (t.unsqueeze(1) for t in (self.q(x), self.k(x), self.v(x)))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(y.squeeze(1))


# --------------------------------------------------------- 1. correctness ----
def check_correctness():
    print("1. CORRECTNESS  (parallel scan must equal the sequential loop)")
    B, T, D = 4, 333, 48
    a = torch.rand(B, T, D, device=dev) * 0.9 + 0.05
    u = torch.randn(B, T, D, device=dev)
    md = (gamma_scan(a, u) - gamma_loop(a, u)).abs().max().item()
    print(f"   T={T}  max|scan - loop| = {md:.2e}   ->   "
          f"{'IDENTICAL' if md < 1e-3 else 'MISMATCH'}\n")


# --------------------------------------------------------------- 2. speed ----
def bench(mod, T, B=16, D=256, iters=15):
    x = torch.randn(B, T, D, device=dev, requires_grad=True)
    for _ in range(3):                         # warmup
        mod(x).sum().backward(); x.grad = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        x.grad = None
        mod(x).sum().backward()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3   # ms / step (fwd+bwd)


def speed_table():
    print("2. SPEED  (forward+backward ms per step, B=16 D=256; lower = faster)")
    D = 256
    g_scan = GammaMixer(D, use_scan=True).to(dev)
    g_loop = GammaMixer(D, use_scan=False).to(dev)
    attn = AttnMixer(D).to(dev)
    Ts = [128, 256, 512, 1024, 2048, 4096]
    print(f"   {'T':>6}{'attention':>12}{'gamma-loop':>12}{'gamma-scan':>12}"
          f"{'scan vs attn':>14}")
    prev = {}
    for T in Ts:
        ta = bench(attn, T, D=D)
        tl = bench(g_loop, T, D=D) if T <= 1024 else None
        ts = bench(g_scan, T, D=D)
        speedup = ta / ts
        lp = f"{tl:>10.2f}ms" if tl is not None else f"{'--':>12}"
        tag = ""
        if "a" in prev:
            tag = f"  (attn x{ta/prev['a']:.1f}, scan x{ts/prev['s']:.1f} per 2x T)"
        print(f"   {T:>6}{ta:>10.2f}ms{lp}{ts:>10.2f}ms{speedup:>11.2f}x{tag}")
        prev = {"a": ta, "s": ts}
    print("   (attention ~4x per doubling = O(T^2); scan ~2x = O(T logT). "
          "crossover where scan<attn is the win.)\n")


# ------------------------------------------------------------ 3. accuracy ----
class LM(nn.Module):
    def __init__(self, V, D, mixer):
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.pos = nn.Parameter(torch.zeros(1, 4096, D))
        self.n1, self.n2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.mix = mixer
        self.ff = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))
        self.head = nn.Linear(D, V)

    def forward(self, idx):
        T = idx.shape[1]
        h = self.emb(idx) + self.pos[:, :T]
        h = h + self.mix(self.n1(h))
        h = h + self.ff(self.n2(h))
        return self.head(h)


def load_text():
    text = ""
    for fn in ["ARCHITECTURE.md", "README.md", "PARADIGM.md", "ONTOLOGY.md",
               "FINDINGS.md", "RAG_README.md", "section_01_photon_sea.md",
               "section_02_electron.md"]:
        p = Path(fn)
        if p.exists():
            text += p.read_text(encoding="utf-8", errors="ignore") + "\n"
    return text


def train_eval(mixer_fn, text, T, D=256, steps=400, B=16):
    chars = sorted(set(text)); V = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(len(data) * 0.9); tr, va = data[:n], data[n:]
    model = LM(V, D, mixer_fn(D)).to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    def batch(src):
        ix = torch.randint(0, len(src) - T - 1, (B,))
        x = torch.stack([src[i:i + T] for i in ix]).to(dev)
        y = torch.stack([src[i + 1:i + T + 1] for i in ix]).to(dev)
        return x, y

    torch.cuda.synchronize(); t0 = time.perf_counter()
    model.train()
    for _ in range(steps):
        x, y = batch(tr)
        loss = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    torch.cuda.synchronize(); train_t = time.perf_counter() - t0

    model.eval()
    with torch.no_grad():
        vl = [F.cross_entropy(model(batch(va)[0]).reshape(-1, V),
                              batch(va)[1].reshape(-1)).item() for _ in range(20)]
    # use a fixed eval batch for fairness
    with torch.no_grad():
        torch.manual_seed(123)
        ls = []
        for _ in range(30):
            x, y = batch(va)
            ls.append(F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1)).item())
    vloss = sum(ls) / len(ls)
    return vloss, math.exp(vloss), train_t, nparam


def accuracy_long(T):
    text = load_text()
    print(f"3. ACCURACY at long context  T={T}  ({len(text):,} chars of repo text, "
          f"matched params, {400} steps)")
    gl, gp, gt, gn = train_eval(lambda D: GammaMixer(D, use_scan=True), text, T)
    al, ap, at, an = train_eval(lambda D: AttnMixer(D), text, T)
    print(f"   {'model':>14}{'val loss':>11}{'perplexity':>13}{'train s':>10}{'params':>10}")
    print(f"   {'gamma-scan':>14}{gl:>11.4f}{gp:>13.3f}{gt:>10.1f}{gn:>10,}")
    print(f"   {'attention':>14}{al:>11.4f}{ap:>13.3f}{at:>10.1f}{an:>10,}")
    win = "gamma-scan" if gl < al else "attention"
    print(f"   -> lower perplexity: {win}   "
          f"(gamma {gp:.3f} vs attn {ap:.3f}, {(ap-gp):+.3f}); "
          f"gamma {at/gt:.2f}x {'faster' if gt<at else 'slower'} to train\n")


if __name__ == "__main__":
    print(f"device: {torch.cuda.get_device_name(0) if dev=='cuda' else 'cpu'}\n")
    check_correctness()
    speed_table()
    T = int(sys.argv[1]) if len(sys.argv) > 1 else 512
    accuracy_long(T)
