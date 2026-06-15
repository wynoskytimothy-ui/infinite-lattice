#!/usr/bin/env python3
"""
Correlating-MEET root cause: fuse every independent sensor signal at the point
where they CONVERGE, weighting the rarest/most-diagnostic signal most.

This is the AETHOS meet applied to live multi-sensor data. Each bearing carries a
set of independent anomaly views -- broadband web residual, kurtosis, crest,
high-band, and the pi-lattice defect line (230 Hz outer-race). The MEET asks: on
which bearing do MULTIPLE independent views intersect? The "rarest means most"
rule weights each view by its idf -- a view that fires everywhere (broadband rms
at a sensitive gate) is non-diagnostic; one that fires rarely and specifically
(the named defect line) is decisive. idf is the engine's own:

    idf(s) = log(1 + (N - df + 0.5) / (df + 0.5))      # aethos_append_index.py

The source is where the rare views converge. Sign separates source (views rising
ABOVE the peer prediction) from propagation (dragged below). Because it scores
EVERY bearing independently, it surfaces MULTIPLE simultaneous sources -- the
1st_test two-fault case -- not just an argmax.

    python aethos_meet_rca.py --test 2nd_test
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GATE = 4.0


def _load(test):
    run = json.loads((HERE / f"bearing_run_{test}.json").read_text())
    ms = json.loads((HERE / f"bearing_multisignal_{test}.json").read_text())
    df = json.loads((HERE / f"bearing_defect_{test}.json").read_text())
    signals = {
        "web":     np.array(run["coupling_z"]),          # broadband peer-divergence
        "kurtosis": np.array(ms["resid"]["kurtosis"]),   # impulsiveness
        "crest":   np.array(ms["resid"]["crest"]),
        "hi_band": np.array(ms["resid"]["hi_band"]),
        "defect":  np.array(df["defect_z"]),             # the named outer-race line (rare)
    }
    return run, signals, run["hours"], run.get("truth", 0)


def _idf(df_count, N):
    return math.log(1 + (N - df_count + 0.5) / (df_count + 0.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="2nd_test")
    ap.add_argument("--n-base", type=int, default=100)
    ap.add_argument("--truth", default=None, help="comma-sep 0-based truth bearings")
    args = ap.parse_args()

    run, signals, hours, truth0 = _load(args.test)
    hours = np.array(hours)
    T, B = signals["web"].shape
    nb = args.n_base
    truth = ([int(x) for x in args.truth.split(",")] if args.truth is not None
             else [truth0])

    # SPATIAL specificity per signal: when it fires somewhere, how concentrated is
    # it on ONE bearing? (1 = localized/diagnostic, 0 = smeared = propagation). This
    # is the sensor analog of idf 'rarest means most' -- spatial, not temporal.
    spec = {}
    print(f"MEET root cause on {args.test}: {T} snapshots, {B} bearings, "
          f"gate {GATE:.0f}-sigma\n")
    print(f"  signal diagnosticity (spatial specificity -- 'most localized means most'):")
    for s, z in signals.items():
        zz = np.clip(z[nb:], 0, None)
        active = zz.max(1) > GATE
        if active.any():
            top = zz[active].max(1)
            mean = zz[active].mean(1)
            spec[s] = float(np.clip((top - mean) / (top + 1e-9), 0, 1).mean())
        else:
            spec[s] = 0.0
    for s in sorted(spec, key=lambda s: -spec[s]):
        print(f"     {s:<8} specificity {spec[s]:.2f}   "
              f"(fires {100*( (signals[s][nb:]>GATE).mean()):4.1f}% of bearing-time)")

    # localized excess: how much bearing b stands ABOVE its peers on signal s
    def excess(z):
        peer = (z.sum(1, keepdims=True) - z) / (B - 1)
        return np.clip(z - peer, 0, None)                  # (T,B)

    contrib = {s: spec[s] * excess(z) for s, z in signals.items()}
    meet = sum(contrib.values())                           # (T,B)
    meet_sm = np.stack([np.convolve(meet[:, b], np.ones(7) / 7, "same")
                        for b in range(B)], 1)
    dragged = signals["web"] < -GATE                       # below peers = victim

    score = meet_sm[nb:].sum(0)
    drag_frac = dragged[nb:].mean(0)
    rank = np.argsort(-score)
    thresh = 0.25 * score[rank[0]]
    meet_sm_full = meet_sm

    print(f"\n  diagnosis (specificity-weighted localized meet):")
    sources = []
    idf = spec                                             # report key compatibility
    for b in rank:
        # decisive view = the one contributing most localized, specific excess to b
        csum = {s: float(contrib[s][nb:, b].sum()) for s in signals}
        views = [s for s in sorted(csum, key=lambda s: -csum[s]) if csum[s] > 1.0]
        decisive = views[0] if views else "-"
        is_src = score[b] >= thresh and signals["web"][nb:, b].max() > GATE
        tag = ("SOURCE" if is_src else
               ("propagation (dragged)" if drag_frac[b] > 0.2 else "quiet"))
        star = "  <- truth" if b in truth else ""
        print(f"     Bearing {b+1}: meet {score[b]:7.0f}  [{tag}]{star}")
        print(f"        converging views: {', '.join(views[:4]) or '-'}  | "
              f"decisive: {decisive} (specificity {spec.get(decisive,0):.2f})")
        if is_src:
            sources.append(b)

    hit = sorted(sources) == sorted(truth)
    print(f"\n  => sources = {sorted(b+1 for b in sources)}  "
          f"[{'HIT' if hit else 'CHECK'} vs truth {sorted(b+1 for b in truth)}]")
    if len(sources) > 1:
        print(f"     resolved {len(sources)} simultaneous sources via the meet.")
    # decisive convergence for the top source
    top = sources[0] if sources else rank[0]
    conv = sorted([(s, float(contrib[s][nb:, top].sum())) for s in signals
                   if contrib[s][nb:, top].sum() > 1.0], key=lambda x: -x[1])
    if conv:
        print(f"     Bearing {top+1} root-caused by the convergence of "
              f"{len(conv)} localized views; the most specific ({conv[0][0]}, "
              f"specificity {spec[conv[0][0]]:.2f}) names the fault.")

    out = HERE / f"bearing_meet_{args.test}.json"
    out.write_text(json.dumps({
        "hours": hours.tolist(), "truth": truth, "spec": spec,
        "meet": meet_sm.tolist(), "sources": [int(b) for b in sources],
    }))
    print(f"\nwrote {out.name}")


if __name__ == "__main__":
    main()
