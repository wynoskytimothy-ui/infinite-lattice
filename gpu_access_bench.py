#!/usr/bin/env python3
"""Honest hardware check: is a GPU 'blistering fast' at the lattice's access
pattern (random integer gather / modular lookup) vs the dense float matmul it was
built for? Measures both on the 5080."""
import time

import torch

dev = "cuda"
torch.cuda.synchronize()

# 1) dense fp16 matmul -- what Tensor Cores are built for (attention/Gamma-SSM)
N = 8192
A = torch.randn(N, N, device=dev, dtype=torch.float16)
B = torch.randn(N, N, device=dev, dtype=torch.float16)
for _ in range(3):
    _ = A @ B
torch.cuda.synchronize()
R, t = 20, time.time()
for _ in range(R):
    C = A @ B
torch.cuda.synchronize()
mm_ms = (time.time() - t) / R * 1000
tflops = (2 * N ** 3) / (mm_ms / 1000) / 1e12

# 2) random integer gather -- the lattice's irregular 'modular lookup across VRAM'
M = 200_000_000                                   # 800 MB int32 'crystal lattice'
table = torch.randint(0, 1 << 20, (M,), device=dev, dtype=torch.int32)
G = 16_000_000
ridx = torch.randint(0, M, (G,), device=dev)
sidx = torch.arange(G, device=dev)
for _ in range(3):
    _ = table[ridx]
torch.cuda.synchronize()
t = time.time()
for _ in range(R):
    _ = table[ridx]                               # random (lattice routing)
torch.cuda.synchronize()
rand_ms = (time.time() - t) / R * 1000
t = time.time()
for _ in range(R):
    _ = table[sidx]                               # coalesced (ideal) read, same size
torch.cuda.synchronize()
seq_ms = (time.time() - t) / R * 1000

rand_gbs = G * 4 / (rand_ms / 1000) / 1e9
seq_gbs = G * 4 / (seq_ms / 1000) / 1e9
print(f"RTX 5080 -- dense matmul (GPU-native) vs random gather (lattice pattern)\n")
print(f"  dense fp16 matmul : {tflops:6.0f} TFLOP/s   (Tensor Cores ~saturated)")
print(f"  coalesced read    : {seq_gbs:6.0f} GB/s     (near the ~960 GB/s peak)")
print(f"  RANDOM gather     : {rand_gbs:6.0f} GB/s     (the lattice's modular-lookup pattern)")
print(f"\n  random gather is {seq_gbs/rand_gbs:.0f}x SLOWER than a coalesced read and uses a")
print(f"  fraction of the bandwidth -- irregular integer routing is GPU-HOSTILE, not")
print(f"  blistering. dense matmul (attention / the Gamma-SSM) is what the GPU is for.")
