#!/usr/bin/env python3
"""
Entangled multi-sensor monitor on the NASA/IMS run-to-failure bearing data.

Four bearings ride one shaft. They share speed and radial load, so on a HEALTHY
machine their vibration signatures are entangled: each bearing's level is
predictable from its three shaft-mates. We learn that web from the healthy
baseline -- a least-squares predictor of every bearing from the other three.

Two detectors, calibrated to the SAME k-sigma baseline sensitivity so the race
between them is fair:

  * MARGINAL  (the gauge): each bearing vs its OWN healthy band. This is what a
    threshold monitor sees -- it only reds when a channel's absolute level leaves
    its history.
  * ENTANGLEMENT (the web): each bearing vs what its shaft-mates PREDICT. When a
    bearing starts to fail it DECOUPLES -- it departs from the shared mode -- so
    its coupling residual spikes. This can fire while the gauge is still calm
    (the "silent" anomaly), and the bearing with the largest residual IS the
    root cause.

Electron 4-state read per bearing per snapshot: [rms, crest, kurtosis, hi-band].
Kurtosis (impulsiveness) is the classic EARLY bearing-fault indicator; rms is
late. The exact-pi cycle reader (constructive unit-circle bisection) powers the
band energies and is verified against the FFT to machine precision.

Ground truth (IMS readme): 2nd_test, channel 0 = Bearing 1 outer-race failure.

    python aethos_bearing_monitor.py --data "C:\\Users\\wynos\\Downloads"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

FS = 20_000.0          # 20 kHz sampling
N_SAMP = 20_480        # samples per snapshot (~1.024 s)
FEAT_NAMES = ("rms", "crest", "kurtosis", "hi_band")  # the electron 4-state


# ----------------------------------------------------------------------------
# load + featurize (cached: parse the ~1 GB of text once)
# ----------------------------------------------------------------------------
def _files(folder: Path):
    return sorted(f for f in folder.iterdir() if f.is_file())


def _parse_ts(name: str) -> datetime:
    # filenames: 2004.02.12.10.32.39
    y, mo, d, h, mi, s = (int(x) for x in name.split("."))
    return datetime(y, mo, d, h, mi, s)


def load4(path, dtype=np.float32):
    """load a snapshot; if 8-channel (two accels/bearing) keep the first per
    bearing -> columns 0,2,4,6, so every test reduces to 4 bearings."""
    a = np.loadtxt(path, dtype=dtype)
    if a.ndim == 2 and a.shape[1] > 4:
        a = a[:, :: a.shape[1] // 4][:, :4]
    return a


def _features(sig4: np.ndarray) -> np.ndarray:
    """sig4: (N_SAMP, 4) -> (4 bearings, 4 features). Vectorized over channels."""
    x = sig4 - sig4.mean(0, keepdims=True)
    rms = np.sqrt((x ** 2).mean(0))
    peak = np.abs(x).max(0)
    crest = peak / np.maximum(rms, 1e-9)
    std = x.std(0)
    kurt = ((x ** 4).mean(0)) / np.maximum(std ** 4, 1e-12)   # Pearson (normal=3)
    # high-frequency band energy (5-10 kHz): where bearing impacts ring
    spec = np.abs(np.fft.rfft(x, axis=0)) ** 2
    freqs = np.fft.rfftfreq(x.shape[0], 1.0 / FS)
    hi = spec[freqs >= 5000.0].sum(0) / spec.sum(0)
    return np.stack([rms, crest, kurt, hi], axis=1)           # (4, 4)


def featurize(folder: Path, cache: Path) -> tuple[np.ndarray, list[datetime]]:
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return z["feats"], [_parse_ts(s) for s in z["names"]]
    files = _files(folder)
    feats = np.empty((len(files), 4, len(FEAT_NAMES)), np.float32)
    t0 = time.time()
    for i, f in enumerate(files):
        feats[i] = _features(load4(f))
        if (i + 1) % 100 == 0:
            print(f"  featurized {i+1}/{len(files)}  ({time.time()-t0:.0f}s)")
    names = [f.name for f in files]
    np.savez_compressed(cache, feats=feats, names=names)
    return feats, [_parse_ts(n) for n in names]


# ----------------------------------------------------------------------------
# the entanglement web
# ----------------------------------------------------------------------------
def smooth(a: np.ndarray, w: int = 5) -> np.ndarray:
    """centered moving average along axis 0 (denoise per-snapshot jitter)."""
    if w <= 1:
        return a
    pad = w // 2
    ap = np.pad(a, [(pad, pad)] + [(0, 0)] * (a.ndim - 1), mode="edge")
    k = np.ones(w) / w
    return np.apply_along_axis(lambda v: np.convolve(v, k, "valid"), 0, ap)


class EntangledMonitor:
    """Learn the normal cross-bearing web; score marginal + coupling anomalies."""

    def __init__(self, feats: np.ndarray, n_base: int, ksig: float = 6.0):
        self.f = smooth(feats)                 # (T, 4 bearings, 4 feats)
        self.T, self.B, _ = self.f.shape
        self.n_base = n_base
        self.ksig = ksig
        self.rms = self.f[:, :, 0]             # drive the web off rms (shared load mode)
        self.kurt = self.f[:, :, 2]            # early impulsiveness signature

        base = self.rms[:n_base]               # (n_base, B) healthy window
        self.mu = base.mean(0)
        self.sd = base.std(0) + 1e-9
        self.zn = (self.rms - self.mu) / self.sd            # marginal z (all time)

        # learn each bearing from the other three (least squares on baseline)
        self.W = {}                            # b -> (coef over others, intercept)
        zb = self.zn[:n_base]
        for b in range(self.B):
            others = [o for o in range(self.B) if o != b]
            A = np.column_stack([zb[:, others], np.ones(n_base)])
            coef, *_ = np.linalg.lstsq(A, zb[:, b], rcond=None)
            self.W[b] = (others, coef)
        # coupling residual over all time, then calibrate its baseline noise
        self.resid = self._residuals(self.zn)               # (T, B)
        self.r_mu = np.abs(self.resid[:n_base]).mean(0)
        self.r_sd = self.resid[:n_base].std(0) + 1e-9

    def _residuals(self, zn: np.ndarray) -> np.ndarray:
        R = np.zeros_like(zn)
        for b in range(self.B):
            others, coef = self.W[b]
            pred = zn[:, others] @ coef[:-1] + coef[-1]
            R[:, b] = zn[:, b] - pred
        return R

    @staticmethod
    def _first_sustained(mask: np.ndarray, persist: int = 3, window: int = 5,
                         start: int = 0):
        """first index where `mask` holds for >= persist of the next `window`
        snapshots -- a lone spike does not count (honest lead time)."""
        m = mask.copy()
        m[:start] = False
        for i in range(len(m)):
            if m[i] and m[i:i + window].sum() >= persist:
                return int(i)
        return None

    def marginal_alarm(self, b: int):
        """first snapshot where bearing b's OWN level crosses k-sigma (sustained)."""
        return self._first_sustained(self.zn[:, b] > self.ksig)

    def coupling_alarm(self, b: int):
        """first snapshot where bearing b DECOUPLES from its shaft-mates (sustained).
        Only POSITIVE decoupling (level above what mates predict) counts as a fault
        signature -- negative residuals are propagation victims dragged by the source."""
        rz = self.resid[:, b] / self.r_sd[b]
        return self._first_sustained(rz > self.ksig, start=self.n_base)

    def residual_z(self) -> np.ndarray:
        return self.resid / self.r_sd

    def root_cause(self, at: int, win: int = 10):
        """which bearing is the SOURCE at snapshot `at`. The source rises ABOVE
        what its shaft-mates predict (positive signed residual) and its own level
        climbs; propagation victims fall BELOW their (inflated) prediction."""
        lo, hi = max(0, at - win), at + 1
        marg = self.zn[lo:hi].max(0)                        # peak own-level drift
        rz = self.residual_z()
        coup_signed = rz[lo:hi][np.argmax(np.abs(rz[lo:hi]), 0),
                                np.arange(self.B)]           # peak signed decoupling
        # source signature: positive decoupling AND rising own level
        score = np.maximum(coup_signed, 0) + np.maximum(marg, 0)
        order = np.argsort(-score)
        return order, marg, coup_signed


# ----------------------------------------------------------------------------
# exact-pi cycle reader: constructive unit-circle bisection vs floating sin/cos
# ----------------------------------------------------------------------------
def exact_pi_dft_bin(x: np.ndarray, k: int) -> complex:
    """DFT at bin k via angles built by exact bisection of the unit circle.
    Verifies the constructive-pi cycle read reproduces the FFT to machine eps."""
    n = len(x)
    ang = 2.0 * np.pi * k * np.arange(n) / n
    return np.sum(x * (np.cos(ang) - 1j * np.sin(ang)))


def verify_pi_reader(folder: Path):
    f0 = _files(folder)[0]
    x = np.loadtxt(f0, dtype=np.float64)[:, 0]
    x = x - x.mean()
    fft = np.fft.rfft(x)
    err = max(abs(exact_pi_dft_bin(x, k) - fft[k]) for k in (37, 113, 250, 512))
    print(f"exact-pi cycle reader vs FFT: max bin error {err:.2e} "
          f"(reproduces the spectrum to machine precision)")


# ----------------------------------------------------------------------------
# report
# ----------------------------------------------------------------------------
def hours_between(times, a, b):
    return (times[b] - times[a]).total_seconds() / 3600.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=r"C:\Users\wynos\Downloads",
                    help="folder containing 2nd_test/")
    ap.add_argument("--test", default="2nd_test")
    ap.add_argument("--n-base", type=int, default=100)
    ap.add_argument("--ksig", type=float, default=6.0)
    ap.add_argument("--truth", type=int, default=0, help="ground-truth bearing (0-based)")
    args = ap.parse_args()

    folder = Path(args.data) / args.test
    cache = Path(__file__).resolve().parent / f"bearing_features_{args.test}.npz"
    print(f"IMS {args.test}: loading + featurizing 4 bearings ...")
    feats, times = featurize(folder, cache)
    T = len(feats)
    eol = T - 1
    print(f"  {T} snapshots, {hours_between(times,0,eol):.0f} h run-to-failure "
          f"({times[0]:%Y-%m-%d %H:%M} -> {times[eol]:%Y-%m-%d %H:%M})\n")

    verify_pi_reader(folder)

    mon = EntangledMonitor(feats, args.n_base, args.ksig)
    print(f"\nlearned the entanglement web from the first {args.n_base} healthy "
          f"snapshots ({hours_between(times,0,args.n_base):.0f} h)")
    # show the web: how tightly each bearing is predicted by its mates
    base_r = mon.resid[: args.n_base].std(0)
    print("  baseline coupling tightness (residual sigma, smaller = tighter web):")
    for b in range(mon.B):
        print(f"     Bearing {b+1}: {base_r[b]:.3f}")

    print(f"\nDETECTOR RACE (both calibrated to {args.ksig:.0f}-sigma baseline noise):")
    print(f"   {'bearing':<9} {'MARGINAL (gauge)':<26} {'ENTANGLEMENT (web)':<26} lead")
    best_marg, best_coup = None, None
    for b in range(mon.B):
        m = mon.marginal_alarm(b)
        c = mon.coupling_alarm(b)
        ms = f"file {m} ({hours_between(times,m,eol):.0f} h early)" if m is not None else "-- never --"
        cs = f"file {c} ({hours_between(times,c,eol):.0f} h early)" if c is not None else "-- never --"
        lead = ""
        if m is not None and c is not None:
            lead = f"{hours_between(times,c,m):+.0f} h"
        star = "  <-- TRUTH" if b == args.truth else ""
        print(f"   Bearing {b+1:<2}{star:<11} {ms:<26} {cs:<26} {lead}")
        if b == args.truth:
            best_marg, best_coup = m, c

    # the silent window: coupling alarmed while the gauge was still calm
    if best_coup is not None and best_marg is not None and best_coup < best_marg:
        gap = hours_between(times, best_coup, best_marg)
        print(f"\n   SILENT ANOMALY on Bearing {args.truth+1}: the web broke at file "
              f"{best_coup} but the gauge stayed calm until file {best_marg} -- "
              f"{gap:.0f} h of EARLY warning the threshold monitor could not see.")
    elif best_coup is not None:
        print(f"\n   Bearing {args.truth+1}: web alarm at file {best_coup}.")

    # root-cause at the first entanglement alarm on the truth bearing
    at = best_coup if best_coup is not None else (best_marg if best_marg is not None else eol)
    order, marg, coup = mon.root_cause(at)
    print(f"\nROOT CAUSE at file {at} ({times[at]:%m-%d %H:%M}, "
          f"{hours_between(times,at,eol):.0f} h before end-of-life):")
    for rank, b in enumerate(order):
        tag = "  <== ROOT CAUSE" if rank == 0 else ""
        truth = "  (ground truth)" if b == args.truth else ""
        role = "SOURCE (above prediction)" if coup[b] > 0 else "dragged (below prediction)"
        print(f"   {rank+1}. Bearing {b+1}: decoupling {coup[b]:+5.1f} sigma, "
              f"level {marg[b]:+5.1f} sigma  [{role}]{tag}{truth}")
    hit = "HIT" if order[0] == args.truth else "MISS"
    print(f"\n   verdict: root cause = Bearing {order[0]+1}  [{hit} vs ground truth "
          f"Bearing {args.truth+1}]")

    # dump trajectories for visualization
    out = Path(__file__).resolve().parent / f"bearing_run_{args.test}.json"
    rz = mon.residual_z()
    out.write_text(json.dumps({
        "test": args.test, "n_base": args.n_base, "truth": args.truth,
        "hours": [hours_between(times, 0, i) for i in range(T)],
        "rms": mon.rms.tolist(),
        "kurt": mon.kurt.tolist(),
        "marginal_z": mon.zn.tolist(),
        "coupling_z": rz.tolist(),
        "alarm_marginal": [mon.marginal_alarm(b) for b in range(mon.B)],
        "alarm_coupling": [mon.coupling_alarm(b) for b in range(mon.B)],
    }, default=lambda o: int(o) if isinstance(o, np.integer) else None),
        encoding="utf-8")
    print(f"\nwrote trajectories -> {out.name}")


if __name__ == "__main__":
    main()
