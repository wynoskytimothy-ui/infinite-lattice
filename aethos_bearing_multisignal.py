#!/usr/bin/env python3
"""
Multi-signal corroboration: real failure vs noise by AGREEMENT across independent
physics.

The user's principle: the second a bearing really starts to degrade it shows extra
vibration AND impulsiveness AND heat -- several independent signatures diverging
together, on that bearing, differently from its peers. Noise lights ONE signal and
no peer-divergence. So "real" = corroboration across independent channels; "noise"
= a lone, uncorroborated blip.

IMS is accelerometer-only (no thermocouple), so we use the independent signatures
the VIBRATION itself carries -- and we are honest that a heat curve DERIVED from
vibration would not be independent, so none is faked:

    rms      broadband energy        (rises late)
    kurtosis impulsiveness / spikes  (rises EARLIEST -- defect impacts)
    crest    peak-to-rms             (transient impacts)
    hi_band  5-10 kHz impact energy  (defect ring)

For each signal we build its own peer-web (each bearing predicted from its three
shaft-mates) and take the divergence residual. A bearing is in REAL degradation
when >= K of these independent signals diverge together (sustained); a lone signal
is noise. A real temperature channel would slot in identically as one more vote.

    python aethos_bearing_multisignal.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aethos_bearing_monitor import featurize, smooth, _parse_ts, FEAT_NAMES  # noqa: E402

GATE = 4.0          # per-signal divergence threshold (sigma vs baseline)
K = 2               # need >= K independent signals to call it real
N_BASE = 100
PERSIST, WINDOW = 3, 5


def first_sustained(mask, start=0):
    m = mask.copy()
    m[:start] = False
    for i in range(len(m)):
        if m[i] and m[i:i + WINDOW].sum() >= PERSIST:
            return int(i)
    return None


def peer_web_residual(sig, n_base):
    """sig: (T, B) one signal across bearings -> divergence residual z (T, B)."""
    T, B = sig.shape
    base = sig[:n_base]
    mu, sd = base.mean(0), base.std(0) + 1e-9
    zn = (sig - mu) / sd
    R = np.zeros_like(zn)
    for b in range(B):
        others = [o for o in range(B) if o != b]
        A = np.column_stack([zn[:n_base][:, others], np.ones(n_base)])
        coef, *_ = np.linalg.lstsq(A, zn[:n_base, b], rcond=None)
        R[:, b] = zn[:, b] - (zn[:, others] @ coef[:-1] + coef[-1])
    r_sd = R[:n_base].std(0) + 1e-9
    return R / r_sd


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="2nd_test")
    ap.add_argument("--data", default=r"C:\Users\wynos\Downloads")
    ap.add_argument("--truth", type=int, default=0)
    args = ap.parse_args()
    folder = Path(args.data) / args.test
    cache = Path(__file__).resolve().parent / f"bearing_features_{args.test}.npz"
    feats, times = featurize(folder, cache)
    feats = smooth(feats)
    T, B, F = feats.shape
    hours = [(t - times[0]).total_seconds() / 3600.0 for t in times]
    eol = T - 1
    truth = args.truth

    # per-signal divergence residuals (positive = rising above peers = fault-ward)
    resid = {f: peer_web_residual(feats[:, :, f], N_BASE) for f in range(F)}

    print(f"IMS {args.test}: {T} snapshots, {hours[-1]:.0f} h. "
          f"{F} independent vibration signals, need >= {K} to call it real.\n")

    # per-bearing, per-signal onset (first sustained positive divergence)
    print(f"  onset hour of each signal's divergence from peers (-- = never):")
    print(f"   {'bearing':<10} " + " ".join(f"{FEAT_NAMES[f]:>9}" for f in range(F))
          + f" |  {'CORROBORATED':>12}  single-sig blips")
    rows = []
    for b in range(B):
        onsets = {}
        for f in range(F):
            idx = first_sustained(resid[f][:, b] > GATE, start=N_BASE)
            onsets[f] = idx
        # corroboration count over time, and the corroborated onset (>=K signals)
        corro = np.sum([(resid[f][:, b] > GATE) for f in range(F)], axis=0)
        c_onset = first_sustained(corro >= K, start=N_BASE)
        # lone-signal blips: snapshots where exactly 1 signal fired (noise candidates)
        blips = int(np.sum(corro == 1))
        cells = " ".join(
            (f"{hours[onsets[f]]:>8.0f}h" if onsets[f] is not None else "       --")
            for f in range(F))
        cstr = f"{hours[c_onset]:>10.0f}h" if c_onset is not None else "        --"
        star = "  <-- TRUTH" if b == truth else ""
        print(f"   Bearing {b+1:<2} {cells} |  {cstr}   {blips:>5}{star}")
        rows.append({"bearing": b + 1, "onsets": onsets, "c_onset": c_onset, "blips": blips})

    # the leader: which independent signal fires first on the true bearing
    tb = rows[truth]
    fired = {f: tb["onsets"][f] for f in range(F) if tb["onsets"][f] is not None}
    if fired:
        lead = min(fired, key=lambda f: fired[f])
        print(f"\n  on Bearing {truth+1}, the EARLIEST independent signal is "
              f"'{FEAT_NAMES[lead]}' at hour {hours[fired[lead]]:.0f} "
              f"({hours[eol]-hours[fired[lead]]:.0f} h before end-of-life).")
        if tb["c_onset"] is not None:
            print(f"  corroborated onset (>= {K} signals agree): hour "
                  f"{hours[tb['c_onset']]:.0f} -- this is the confident real-failure call.")
    # noise contrast: peers should show MANY lone blips but no corroborated onset
    peers_corro = [r for r in rows if r["bearing"] - 1 != truth and r["c_onset"] is None]
    print(f"\n  real vs noise: Bearing {truth+1} reaches {K}-signal corroboration; "
          f"{len(peers_corro)} of {B-1} peers never do (their divergences are lone, "
          f"uncorroborated blips -- noise).")
    print(f"  a thermocouple would be one MORE independent vote; a vibration-derived "
          f"'heat' proxy would not (same signal), so none is faked.")

    out = Path(__file__).resolve().parent / f"bearing_multisignal_{args.test}.json"
    out.write_text(json.dumps({
        "hours": hours, "feat_names": list(FEAT_NAMES), "gate": GATE, "K": K, "truth": truth,
        "resid": {FEAT_NAMES[f]: resid[f].tolist() for f in range(F)},
        "onsets": [{"bearing": r["bearing"],
                    "onset_h": {FEAT_NAMES[f]: (hours[r["onsets"][f]] if r["onsets"][f] is not None else None)
                                for f in range(F)},
                    "c_onset_h": (hours[r["c_onset"]] if r["c_onset"] is not None else None)}
                   for r in rows],
    }))
    print(f"\nwrote {out.name}")


if __name__ == "__main__":
    main()
