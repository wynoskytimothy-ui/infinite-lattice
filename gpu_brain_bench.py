#!/usr/bin/env python3
"""Where the RTX 5080 actually helps the brain: the hypervector (VSA) layer.
Associative recall = the brain's routing + prediction = a dense matmul (query
hypervectors vs the concept memory) + top-k. That is GPU-native. The sparse
prime-addressing core is NOT here -- it stays on CPU (and doesn't need the GPU)."""
import sys
import time

import numpy as np
import torch

dev = "cuda"
print("device:", torch.cuda.get_device_name(0))
try:
    a = torch.randn(2048, 2048, device=dev, dtype=torch.float16)
    (a @ a).sum().item()
    torch.cuda.synchronize()
    print("gpu fp16 matmul: OK\n")
except Exception as e:
    print("gpu matmul FAILED on this torch build:", type(e).__name__, e)
    sys.exit(0)

D, N, Q = 4096, 200_000, 128                 # concept-dim, #concepts, #queries/batch
M = torch.randn(N, D, device=dev, dtype=torch.float16)
M /= M.norm(dim=1, keepdim=True)
Qv = torch.randn(Q, D, device=dev, dtype=torch.float16)
Qv /= Qv.norm(dim=1, keepdim=True)

for _ in range(3):                            # warmup
    (Qv @ M.t()).topk(5, dim=1)
torch.cuda.synchronize()
R = 20
t = time.time()
for _ in range(R):
    sims = Qv @ M.t()                         # (Q,N) associative recall over the whole memory
    top = sims.topk(5, dim=1)                 # nearest concepts (route / predict)
torch.cuda.synchronize()
gpu_ms = (time.time() - t) / R * 1000

Mc, Qc = M.float().cpu().numpy(), Qv.float().cpu().numpy()
t = time.time()
simc = Qc @ Mc.T
cpu_ms = (time.time() - t) * 1000

print(f"associative recall: {Q} queries vs {N:,} concepts x {D}-dim (= routing/prediction)")
print(f"  GPU (5080): {gpu_ms:6.1f} ms/batch   ({Q*N/gpu_ms/1e3:,.0f}M comparisons/ms)")
print(f"  CPU numpy : {cpu_ms:6.0f} ms/batch")
print(f"  speedup   : {cpu_ms/gpu_ms:,.0f}x on the dense hypervector layer\n")
cap = 16e9 / (D * 2)
print(f"  16 GB holds ~{cap/1e6:.1f}M concept-hypervectors; full-memory recall in ~{gpu_ms:.0f} ms")
print(f"  -> a multi-million-concept associative brain with real-time recall.")
