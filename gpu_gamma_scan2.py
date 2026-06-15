#!/usr/bin/env python3
"""Two follow-ups on the gamma parallel scan:
  A. FUSION  -- torch.compile the scan; can a fused scan catch flash attention?
  B. CONTEXT -- does the gamma accuracy win GROW with sequence length? (the real
                long-range-memory question: SSMs should pull further ahead as T rises)
"""
import sys, time, math
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
from gpu_gamma_scan import (gamma_scan, GammaMixer, AttnMixer, LM, load_text,
                            train_eval, bench, dev)

torch.manual_seed(0)


# --------------------------------------------------- A. fused scan speed ----
def fusion_speed():
    print("A. FUSION  (can torch.compile close the wall-clock gap with flash attn?)")
    D = 256
    attn = AttnMixer(D).to(dev)
    g_eager = GammaMixer(D, use_scan=True).to(dev)
    g_comp = GammaMixer(D, use_scan=True).to(dev)
    g_comp.load_state_dict(g_eager.state_dict())
    try:
        g_comp = torch.compile(g_comp, mode="max-autotune-no-cudagraphs")
        ok = True
    except Exception as e:
        print(f"   torch.compile unavailable ({e}); reporting eager only")
        ok = False
    print(f"   {'T':>6}{'attention':>12}{'scan-eager':>12}{'scan-compiled':>15}"
          f"{'compiled vs attn':>18}")
    for T in [512, 1024, 2048, 4096, 8192]:
        ta = bench(attn, T, D=D)
        te = bench(g_eager, T, D=D)
        if ok:
            try:
                tc = bench(g_comp, T, D=D)
            except Exception:
                tc = float("nan")
        else:
            tc = float("nan")
        cv = f"{ta/tc:>15.2f}x" if tc == tc else f"{'--':>16}"
        tcs = f"{tc:>13.2f}ms" if tc == tc else f"{'--':>15}"
        print(f"   {T:>6}{ta:>10.2f}ms{te:>10.2f}ms{tcs}{cv}")
    print("   (>1.00x = the compiled scan is FASTER than flash attention at that T)\n")


# ------------------------------------------ B. accuracy vs context length ----
def context_trend():
    text = load_text()
    print(f"B. CONTEXT  (does the gamma perplexity win grow with T? matched params, "
          f"300 steps, {len(text):,} chars)")
    print(f"   {'T':>6}{'gamma ppl':>12}{'attn ppl':>11}{'gap (attn-gamma)':>18}"
          f"{'gamma train s':>15}")
    for T in [128, 256, 512, 1024, 2048]:
        gl, gp, gt, _ = train_eval(lambda D: GammaMixer(D, use_scan=True), text, T, steps=300)
        al, ap, at, _ = train_eval(lambda D: AttnMixer(D), text, T, steps=300)
        print(f"   {T:>6}{gp:>12.3f}{ap:>11.3f}{ap-gp:>+18.3f}{gt:>15.1f}")
    print("   (positive & growing gap = gamma's long-range memory pulls further "
          "ahead as context lengthens)\n")


if __name__ == "__main__":
    print(f"device: {torch.cuda.get_device_name(0) if dev=='cuda' else 'cpu'}\n")
    fusion_speed()
    context_trend()
