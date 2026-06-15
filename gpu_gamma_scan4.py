#!/usr/bin/env python3
"""Fix the 16k cliff with a CHUNKED scan (the Mamba-2 / FlashLinearAttention trick):
parallel-scan within fixed chunks, carry the chunk-final state across chunks
sequentially. Bounds peak memory to one chunk instead of log2(T) full-length copies.

  1. correctness  -- chunked == sequential loop
  2. memory       -- peak GB at 8k/16k: attention vs eager-scan vs chunked-scan
  3. speed        -- chunked vs attention at 4k/8k/16k (does the win hold at 16k?)
"""
import torch, torch.nn as nn, torch.nn.functional as F
from gpu_gamma_scan import gamma_scan, gamma_loop, AttnMixer, bench, dev
from gpu_gamma_scan3 import MHAttn

torch.manual_seed(0)


def gamma_scan_chunked(a, u, chunk=2048):
    """One chunk live at a time. Within a chunk: parallel scan (carry_in=0) + the
    cumulative decay P, so the carry adds P*carry_in. Across chunks: sequential."""
    B, T, D = a.shape
    carry = a.new_zeros(B, D)
    outs = []
    for s in range(0, T, chunk):
        aj, uj = a[:, s:s + chunk], u[:, s:s + chunk]
        local = gamma_scan(aj, uj)                  # inclusive scan within chunk
        P = torch.cumprod(aj, dim=1)                # decay from chunk start to t
        Cj = local + P * carry.unsqueeze(1)         # inject the incoming carry
        carry = Cj[:, -1]                           # hand last state to next chunk
        outs.append(Cj)
    return torch.cat(outs, dim=1)


class GammaChunked(nn.Module):
    def __init__(self, D, chunk=2048):
        super().__init__()
        self.vp, self.gf, self.gb, self.o = (nn.Linear(D, D) for _ in range(4))
        self.chunk = chunk

    def forward(self, x):
        a = torch.exp(-F.softplus(self.gb(x)))
        u = torch.sigmoid(self.gf(x)) * self.vp(x)
        return self.o(gamma_scan_chunked(a, u, self.chunk))


def check():
    print("1. CORRECTNESS  (chunked must equal the sequential loop)")
    B, T, D = 4, 5000, 32                     # 5000 not a multiple of chunk
    a = torch.rand(B, T, D, device=dev) * 0.9 + 0.05
    u = torch.randn(B, T, D, device=dev)
    md = (gamma_scan_chunked(a, u, chunk=1024) - gamma_loop(a, u)).abs().max().item()
    print(f"   T={T} chunk=1024  max|chunked - loop| = {md:.2e}  ->  "
          f"{'IDENTICAL' if md < 1e-3 else 'MISMATCH'}\n")


def peak_mem(mod, T, B=16, D=256):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    x = torch.randn(B, T, D, device=dev, requires_grad=True)
    mod(x).sum().backward()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1e9


def memory_diag():
    from gpu_gamma_scan import GammaMixer
    print("2. MEMORY  (peak GB, fwd+bwd, B=16 D=256; 16GB card)")
    D = 256
    mods = {"attn-8head": MHAttn(D, 8).to(dev),
            "scan-eager": GammaMixer(D, use_scan=True).to(dev),
            "scan-chunked": GammaChunked(D, 2048).to(dev)}
    print(f"   {'T':>7}{'attn-8head':>13}{'scan-eager':>13}{'scan-chunked':>14}")
    for T in (8192, 16384):
        row = {n: peak_mem(m, T, D=D) for n, m in mods.items()}
        print(f"   {T:>7}{row['attn-8head']:>11.2f}GB{row['scan-eager']:>11.2f}GB"
              f"{row['scan-chunked']:>12.2f}GB")
    print()


def speed():
    print("3. SPEED  (fwd+bwd ms, B=16 D=256; chunked scan vs strongest attention)")
    D = 256
    attn = MHAttn(D, 8).to(dev)
    ch = GammaChunked(D, 2048).to(dev)
    print(f"   {'T':>7}{'attn-8head':>13}{'scan-chunked':>14}{'chunked vs attn':>17}")
    for T in (4096, 8192, 16384):
        ta = bench(attn, T, D=D, iters=8)
        tc = bench(ch, T, D=D, iters=8)
        print(f"   {T:>7}{ta:>11.2f}ms{tc:>12.2f}ms{ta/tc:>16.2f}x")
    print("   (>1.00x = chunked scan faster; should now HOLD/widen at 16k)\n")


if __name__ == "__main__":
    print(f"device: {torch.cuda.get_device_name(0) if dev=='cuda' else 'cpu'}\n")
    check()
    memory_diag()
    speed()
