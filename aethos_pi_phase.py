#!/usr/bin/env python3
"""
pi-lattice defect-frequency channel: discover the bearing's fault line from the
data, then read its EXACT energy + cross-sensor phase -- new physical variables
the broadband amplitude features can't see.

An outer-race spall strikes once per ball-pass (BPFO), modulating a high-frequency
resonance. The diagnostic is a NARROW line in the ENVELOPE of the resonance band,
at a frequency that depends on the real shaft speed/geometry (textbook 236.4 Hz;
on this rig the line actually sits at ~230 Hz). So we:

  1. envelope-demodulate the resonance band (2-6 kHz Hilbert envelope),
  2. DISCOVER the fault line = the envelope frequency in the outer-race band whose
     energy grows most from healthy to failing (the bearing's "fault address"),
  3. read that EXACT line with a recursively-rotated phasor (constructive-pi: build
     the cycle by iterated rotation, drift-free) -- the line is between FFT bins, so
     reading it exactly, not bin-rounded, is the point.

NEW variables per bearing per snapshot: defect-line energy (the named outer-race
signature, not just "more vibration") and defect-line phase (cross-sensor lock).

    python aethos_pi_phase.py
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aethos_bearing_monitor import _files, _parse_ts, smooth, load4  # noqa: E402

FS = 20_000.0
RESONANCE = (2000.0, 6000.0)     # band the impacts ring in
FAULT_BAND = (100.0, 320.0)      # roller BSF ~140, outer BPFO ~230, inner BPFI ~297
GRID = np.arange(FAULT_BAND[0], FAULT_BAND[1] + 1e-9, 1.0)
N_BASE = 100


def pi_goertzel(x, freq, fs):
    """Single-frequency magnitude via recursively rotated phasor (constructive-pi,
    drift-free). Goertzel magnitude equals the DFT magnitude at this frequency."""
    w = 2.0 * math.pi * freq / fs
    coeff = 2.0 * math.cos(w)
    s1 = s2 = 0.0
    for xn in x:
        s0 = float(xn) + coeff * s1 - s2
        s2, s1 = s1, s0
    return math.hypot(s1 - s2 * math.cos(w), s2 * math.sin(w))


def bandpass(x, lo, hi):
    X = np.fft.rfft(x, axis=0)
    fr = np.fft.rfftfreq(x.shape[0], 1.0 / FS)
    X[(fr < lo) | (fr > hi)] = 0
    return np.fft.irfft(X, n=x.shape[0], axis=0)


def hilbert_env(x):
    n = x.shape[0]
    X = np.fft.fft(x, axis=0)
    h = np.zeros(n)
    h[0] = 1
    h[1:(n + 1) // 2] = 2
    if n % 2 == 0:
        h[n // 2] = 1
    return np.abs(np.fft.ifft(X * h[:, None], axis=0))


def verify_reader(folder):
    x = np.loadtxt(_files(folder)[0], dtype=np.float64)[:, 0]
    env = hilbert_env(bandpass(x - x.mean(), *RESONANCE)[:, None])[:, 0]
    env = env - env.mean()
    f = 230.0
    recursive = pi_goertzel(env, f, FS)
    k = f * len(env) / FS
    direct = abs(np.sum(env * np.exp(-2j * math.pi * k * np.arange(len(env)) / len(env))))
    print(f"pi-Goertzel (recursive rotation) vs direct DFT magnitude at {f} Hz: "
          f"rel error {abs(recursive-direct)/direct:.2e} (exact, drift-free)")


def _grid_phasors(n):
    k = np.arange(n)
    return np.exp(-2j * math.pi * np.outer(k, GRID) / FS)   # (n, G)


def featurize(folder, cache):
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return z["resp"], [_parse_ts(s) for s in z["names"]]
    files = _files(folder)
    ph = _grid_phasors(20_480)
    resp = np.empty((len(files), 4, len(GRID)), np.complex128)
    t0 = time.time()
    for i, f in enumerate(files):
        sig = load4(f, np.float64)
        env = hilbert_env(bandpass(sig - sig.mean(0), *RESONANCE))   # (N,4)
        env = env - env.mean(0)
        resp[i] = env.T @ ph                                          # (4, G) complex
        if (i + 1) % 200 == 0:
            print(f"  defect-channel {i+1}/{len(files)}  ({time.time()-t0:.0f}s)")
    names = [f.name for f in files]
    np.savez_compressed(cache, resp=resp, names=names)
    return resp, [_parse_ts(n) for n in names]


def first_sustained(mask, start=0, persist=3, window=5):
    m = mask.copy(); m[:start] = False
    for i in range(len(m)):
        if m[i] and m[i:i + window].sum() >= persist:
            return int(i)
    return None


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="2nd_test")
    ap.add_argument("--data", default=r"C:\Users\wynos\Downloads")
    ap.add_argument("--truth", default="0", help="comma-sep 0-based truth bearings")
    args = ap.parse_args()
    folder = Path(args.data) / args.test
    truth = [int(x) for x in args.truth.split(",")]
    verify_reader(folder)
    cache = Path(__file__).resolve().parent / f"bearing_defect_{args.test}.npz"
    resp, times = featurize(folder, cache)
    hours = [(t - times[0]).total_seconds() / 3600.0 for t in times]
    T = len(resp)
    eol = T - 1
    mag = np.abs(resp)                              # (T,4,G) envelope-line magnitude

    # per-bearing discovery: each bearing finds its OWN strongest emerging envelope
    # line (its fault 'address' -- roller/outer/inner differ); healthy picks noise.
    base = mag[:N_BASE].mean(0)                     # (4,G)
    late = mag[int(0.6 * T):int(0.8 * T)].mean(0)   # (4,G)
    jb = np.argmax(late - base, axis=1)             # (4,) line index per bearing
    f_fault = GRID[jb]                              # (4,) Hz per bearing
    e = smooth(np.stack([mag[:, b, jb[b]] for b in range(4)], axis=1))  # (T,4)
    z = (e - e[:N_BASE].mean(0)) / (e[:N_BASE].std(0) + 1e-9)

    print(f"\n  per-bearing fault-line discovery (textbook: roller BSF~140, "
          f"outer BPFO~230, inner BPFI~297 Hz):")
    for b in range(4):
        on = first_sustained(z[:, b] > 6.0, start=N_BASE)
        s = (f"onset hour {hours[on]:.0f} ({hours[eol]-hours[on]:.0f} h before EOL)"
             if on is not None else "-- never --")
        star = "  <-- truth" if b in truth else ""
        print(f"   Bearing {b+1}: line {f_fault[b]:5.0f} Hz  "
              f"energy {base[b,jb[b]]:5.1f}->{late[b,jb[b]]:6.1f}  "
              f"peak {z[:,b].max():6.1f}s  {s}{star}")

    out = Path(__file__).resolve().parent / f"bearing_defect_{args.test}.json"
    out.write_text(json.dumps({
        "hours": hours, "f_fault": [float(x) for x in f_fault], "truth": truth,
        "defect_z": z.tolist(), "defect_energy": e.tolist(),
    }))
    print(f"\nwrote {out.name}")


if __name__ == "__main__":
    main()
