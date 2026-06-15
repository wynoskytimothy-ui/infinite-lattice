#!/usr/bin/env python3
"""Did the scale-up 'divergence' come from a NUMERICAL BUG in the scan's backward pass
(not the formula)? Forward was verified (scan==loop to 1e-6); the BACKWARD never was.
Suspect: chunked carry uses cumprod(a) over the chunk -> underflows for small a ->
exploding gradients. Test grad correctness vs the loop, and grad MAGNITUDE/NaN across
chunk sizes and sequence lengths.
"""
import torch
from gpu_gamma_scan import gamma_scan, gamma_loop
from gpu_gamma_scan4 import gamma_scan_chunked

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)


def grads(fn, a0, u0, chunk=None):
    a = a0.clone().requires_grad_(True); u = u0.clone().requires_grad_(True)
    out = fn(a, u) if chunk is None else fn(a, u, chunk)
    (out ** 2).sum().backward()
    return a.grad, u.grad


def report(tag, ga, gu, ref_a=None):
    nan = bool(torch.isnan(ga).any() or torch.isinf(ga).any() or
               torch.isnan(gu).any() or torch.isinf(gu).any())
    line = f"   {tag:>26}  |grad_a|max {ga.abs().max().item():.2e}  " \
           f"|grad_u|max {gu.abs().max().item():.2e}  nan/inf={nan}"
    if ref_a is not None:
        line += f"  vs-loop {((ga-ref_a).abs().max().item()):.2e}"
    print(line)


for T in (128, 256):
    print(f"\nT={T}, a in [0.05,0.95] (realistic multi-timescale, some small a):")
    B, D = 2, 16
    a0 = torch.rand(B, T, D, device=dev) * 0.9 + 0.05
    u0 = torch.randn(B, T, D, device=dev)
    gal, gul = grads(gamma_loop, a0, u0)
    report("loop (reference)", gal, gul)
    gah, guh = grads(gamma_scan, a0, u0)
    report("Hillis-Steele full", gah, guh, gal)
    for ch in (16, 64, 256):
        gac, guc = grads(gamma_scan_chunked, a0, u0, chunk=ch)
        report(f"chunked chunk={ch}", gac, guc, gal)

print("\n   loop grad = ground truth; nan/inf or |grad| >> loop = the backward bug.")
print("   if a small chunk is clean and big chunk explodes -> cumprod underflow confirmed.")
