#!/usr/bin/env python3
"""UNIFY THE TWO THREADS: give the Gamma recurrence a VSA OUTER-PRODUCT (matrix) state.

Diagonal state  S_t = a_t*S_{t-1} + u_t  (elementwise) caps MQAR recall at ~17% -- it bundles
the value SET into a superposition but cannot unbind key->value (the documented SSM wall).
Matrix state    S_t = a_t*S_{t-1} + k_t (x) v_t ,  read  y_t = q_t . S_t  IS the bind/unbind of
the AETHOS VSA net (aethos_nn.py) dropped INSIDE the gated recurrence (= gated linear attention;
+delta rule = DeltaNet). The gate keeps the recurrence; the outer product adds the binding.

Question: does recall jump 17% -> >80% with NO attention layer? Seeds 0,1,2, L=64 (the reliable
regime where attention groks 100%). If yes, the entanglement-ODE mixer and the binding algebra
become ONE engine: sub-quadratic AND exact-recall, no attention needed.
"""
import time
import torch, torch.nn as nn, torch.nn.functional as F
from gpu_mqar import make_mqar, VOCAB, dev
from gpu_mqar3 import GammaMix
from gpu_gamma_scan3 import MHAttn
from gpu_mqar4 import run


class GammaOuter(nn.Module):
    """Gated outer-product (matrix) state = gated linear attention. bind k(x)v, unbind q.S,
    with the gamma decay gate a_t carried over from the diagonal mixer."""
    def __init__(self, D, n_heads=8, d_head=16):
        super().__init__()
        self.h, self.dh = n_heads, d_head
        self.q = nn.Linear(D, n_heads * d_head)
        self.k = nn.Linear(D, n_heads * d_head)
        self.v = nn.Linear(D, n_heads * d_head)
        self.gb = nn.Linear(D, n_heads)              # per-head decay (the gamma gate)
        nn.init.constant_(self.gb.bias, 4.0)         # RETENTION init: a=exp(-softplus(4))~=0.98 (keep keys)
        self.o = nn.Linear(n_heads * d_head, D)

    def forward(self, x):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.h, self.dh)
        k = F.normalize(self.k(x).view(B, T, self.h, self.dh), dim=-1)
        v = self.v(x).view(B, T, self.h, self.dh)
        a = torch.sigmoid(self.gb(x))                 # (B,T,h) in (0,1); +bias -> ~1 at init (retain)
        S = torch.zeros(B, self.h, self.dh, self.dh, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(T):
            S = a[:, t, :, None, None] * S + k[:, t, :, :, None] * v[:, t, :, None, :]   # k (x) v
            ys.append(torch.einsum('bhk,bhkv->bhv', q[:, t], S))                          # q . S
        return self.o(torch.stack(ys, 1).reshape(B, T, self.h * self.dh))


class GammaDelta(nn.Module):
    """+ delta rule: subtract the value currently stored at the colliding key before writing
    (DeltaNet) -- prevents key collisions, the standard strong-recall fix."""
    def __init__(self, D, n_heads=8, d_head=16):
        super().__init__()
        self.h, self.dh = n_heads, d_head
        self.q = nn.Linear(D, n_heads * d_head)
        self.k = nn.Linear(D, n_heads * d_head)
        self.v = nn.Linear(D, n_heads * d_head)
        self.gb = nn.Linear(D, n_heads)
        nn.init.constant_(self.gb.bias, 4.0)          # RETENTION init: a~=0.98 at init (keep keys)
        self.bt = nn.Linear(D, n_heads)               # write strength beta
        nn.init.constant_(self.bt.bias, 2.0)          # strong writes at init (beta~=0.88)
        self.o = nn.Linear(n_heads * d_head, D)

    def forward(self, x):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.h, self.dh)
        k = F.normalize(self.k(x).view(B, T, self.h, self.dh), dim=-1)
        v = self.v(x).view(B, T, self.h, self.dh)
        a = torch.sigmoid(self.gb(x))                 # retention gate ~1 at init
        beta = torch.sigmoid(self.bt(x))
        S = torch.zeros(B, self.h, self.dh, self.dh, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(T):
            kt = k[:, t]
            old = torch.einsum('bhk,bhkv->bhv', kt, S)                 # value currently at kt
            delta = (v[:, t] - old) * beta[:, t, :, None]
            S = a[:, t, :, None, None] * S + kt[:, :, :, None] * delta[:, :, None, :]
            ys.append(torch.einsum('bhk,bhkv->bhv', q[:, t], S))
        return self.o(torch.stack(ys, 1).reshape(B, T, self.h * self.dh))


if __name__ == "__main__":
    NP, L, STEPS, SEEDS = 8, 64, 5000, (0,)
    print(f"device: {torch.cuda.get_device_name(0) if dev == 'cuda' else 'cpu'}")
    print(f"RETENTION-INIT matrix state -- exact recall @ {NP} pairs, L={L}, {STEPS} steps, seeds {SEEDS}\n")
    MODELS = {
        "attention x2 (sanity)":   lambda D: [MHAttn(D, 4), MHAttn(D, 4)],
        "gamma-OUTER+ret x2":      lambda D: [GammaOuter(D), GammaOuter(D)],
        "gamma-DELTA+ret x2":      lambda D: [GammaDelta(D), GammaDelta(D)],
    }
    print(f"   {'model':>22}{'seed0':>8}{'seed1':>8}{'seed2':>8}{'best':>8}")
    for name, fn in MODELS.items():
        t0 = time.perf_counter()
        accs = [run(fn, NP, L, STEPS, s) for s in SEEDS]
        print(f"   {name:>22}" + "".join(f"{a:>8.0%}" for a in accs) +
              f"{max(accs):>8.0%}   ({time.perf_counter()-t0:.0f}s)", flush=True)
    print("\n   gamma-OUTER/DELTA >80% with NO attention layer = diagonal->matrix lift fixes recall")
    print("   = the entanglement-ODE mixer + the binding algebra are ONE engine (sub-quadratic + recall).")
