"""
Fair-baseline audit for aethos_master IMS bearing detection (2nd_test).

Ground truth: Bearing 1 (channel 0) outer-race failure, 984 snapshots @ 10 min.

AETHOS analyze():
  - sustained z-score threshold detection at snapshot 545 (Bearing 1 correct)
  - "entanglement" Ghost-Window detection at 540  (0.8h earlier)

Question: does the entanglement/ghost machinery beat a DEAD-SIMPLE univariate
threshold on RMS or kurtosis, and how much earlier? Fair baseline = textbook
condition-monitoring: alarm when feature exceeds healthy_mean + k*healthy_std.

We reuse AETHOS's OWN feature extractor so features are identical; only the
DETECTOR differs.
"""
import sys, time
sys.path.insert(0, r"C:/Users/wynos/aethos_master/src")
import numpy as np
from pathlib import Path
from aethos_master.ims.bearings import load_ims_test, PrimeLatticeIMS

TESTDIR = Path(r"C:/Users/wynos/OneDrive/New folder/2nd_test")
SNAP_MIN = 10.0  # minutes per snapshot

def sustained_first(flags, k=3):
    """First index where flag stays True for k consecutive snapshots; report start."""
    run = 0
    for i, f in enumerate(flags):
        if f:
            run += 1
            if run >= k:
                return i - k + 1
        else:
            run = 0
    return None

def main():
    t0 = time.time()
    snaps = load_ims_test(TESTDIR)
    n = len(snaps)
    print(f"Loaded {n} snapshots in {time.time()-t0:.1f}s")
    nch = 4
    feats = ["rms","peak","crest","kurtosis","hf_ratio"]
    # series[ch][feat] = np.array over snapshots
    S = {ch:{f:np.array([s['channels'][ch][f] for s in snaps]) for f in feats} for ch in range(nch)}

    n_healthy = max(10, int(n*0.2))  # same healthy window AETHOS uses
    print(f"Healthy baseline window: first {n_healthy} snapshots")

    # ---- AETHOS detector ----
    det = PrimeLatticeIMS(n_channels=4)
    res = det.analyze(snaps)
    a_first = res['first_detection']; a_ghost = res['entangled_detection']
    a_bearing = res['failed_bearing']; g_bearing = res['entangled_bearing']
    print(f"\nAETHOS threshold detection : snap {a_first}  bearing B{(a_bearing or -1)+1}  "
          f"({(n-a_first)*SNAP_MIN/60:.1f}h before end)")
    print(f"AETHOS ghost(entangle) det : snap {a_ghost}  bearing B{(g_bearing if g_bearing is not None else -1)+1}  "
          f"({(n-a_ghost)*SNAP_MIN/60:.1f}h before end)")

    # ---- Simple univariate baselines on Bearing 1 (ch 0) ----
    print(f"\n{'detector':<42} {'first_snap':>10} {'h_before_end':>13} {'bearing':>8}")
    print("-"*78)
    def report(name, idx, bearing="B1"):
        if idx is None:
            print(f"{name:<42} {'(none)':>10} {'-':>13} {bearing:>8}")
        else:
            h = (n-idx)*SNAP_MIN/60
            print(f"{name:<42} {idx:>10} {h:>13.1f} {bearing:>8}")

    report("AETHOS threshold (z-score multi-feat)", a_first)
    report("AETHOS ghost / entanglement",           a_ghost)

    # For each single feature on ch0, simple k-sigma alarm
    for f in feats:
        x = S[0][f]
        mu = x[:n_healthy].mean(); sd = x[:n_healthy].std()+1e-12
        z = np.abs(x-mu)/sd
        for ksig in (3,5):
            idx = sustained_first(z>ksig, k=3)
            report(f"B1 {f}  >{ksig}sigma (sustained x3)", idx)

    # Multivariate simple: max over the 5 features of z-score on ch0 (still trivial)
    Z = np.max([np.abs(S[0][f]-S[0][f][:n_healthy].mean())/(S[0][f][:n_healthy].std()+1e-12)
                for f in feats], axis=0)
    for ksig in (3,5):
        idx = sustained_first(Z>ksig, k=3)
        report(f"B1 max-z over 5 feats  >{ksig}sigma", idx)

    # Earliest *physically meaningful* univariate: RMS doubling vs healthy median
    rms0 = S[0]['rms']; base = np.median(rms0[:n_healthy])
    idx = sustained_first(rms0 > 2*base, k=3)
    report("B1 RMS > 2x healthy median (sustained)", idx)
    idx = sustained_first(rms0 > 1.5*base, k=3)
    report("B1 RMS > 1.5x healthy median (sustained)", idx)

    # ---- Does ghost correctly NOT fire early on healthy bearings (false alarms)? ----
    # Count how many snapshots before true degradation each simple detector falsely alarms.
    # "True degradation onset" ~ first sustained RMS>1.5x. Use it as reference.
    onset = sustained_first(rms0 > 1.5*base, k=3)
    print(f"\nReference degradation onset (RMS>1.5x median): snap {onset} "
          f"({(n-onset)*SNAP_MIN/60:.1f}h before end)" if onset else "\n(no onset)")

if __name__ == "__main__":
    main()
