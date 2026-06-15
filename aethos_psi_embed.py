#!/usr/bin/env python3
"""
Psi-plane embedding -- the container that completes the five constructs.

Every sensor's full multi-variable health state becomes ONE addressable point in
the 3D complex plane Psi=(z,zeta), using the real geometry (aethos_complex_plane.
wing_transform). The address alpha=(A,b,w,n) unifies the whole stack:

    branch b  = the ELECTRON quadrant   (sign of level, sign of web residual)
    wing   w  = which VIEWS corroborate (defect / kurtosis / hi-band bits)
    n         = SEVERITY along the rail (fused anomaly magnitude)
    chain  A  = shared context (3,5,7)

So five raw variables collapse to one address (a handful of bits + a shared chain)
and one Psi point (3 reals) -- "embed dimensions for little memory cost". The 32
chambers (4 branches x 8 wings) are discrete, glass-box STATE LABELS: a healthy
bearing sits in a low-severity normal chamber; a failing one MIGRATES to a source
chamber as n grows. The meet (sources cluster in the source branch VA1, victims in
the dragged branch VA2) is the geometric home of the RCA; meets spawn origins
(aethos_origins: 3 child dimension-spaces each, 3^d at depth d -- cheap).

Honest: this is the REPRESENTATION layer, not a new detector -- |Psi| grows with
severity, so its detection power equals the signals it is built from. Its value is
unification + addressability + the glass-box chamber labels.

    python aethos_psi_embed.py --test 1st_test --truth 2,3
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from aethos_complex_plane import LatticeAddress, triple_equalization, wing_transform
from aethos_lattice import BranchKind

HERE = Path(__file__).resolve().parent
CHAIN = (3, 5, 7)
GATE = 4.0
N_BASE = 100


def _views(test):
    run = json.loads((HERE / f"bearing_run_{test}.json").read_text())
    ms = json.loads((HERE / f"bearing_multisignal_{test}.json").read_text())
    df = json.loads((HERE / f"bearing_defect_{test}.json").read_text())
    return {
        "marg": np.array(run["marginal_z"]), "web": np.array(run["coupling_z"]),
        "kurt": np.array(ms["resid"]["kurtosis"]), "hiband": np.array(ms["resid"]["hi_band"]),
        "defect": np.array(df["defect_z"]), "hours": np.array(run["hours"]),
    }


def branch_of(m, c):
    if m > 0 and c > 0:
        return BranchKind.VA1            # rising & above  = SOURCE
    if m <= 0 and c <= 0:
        return BranchKind.VA2            # below & below   = dragged (propagation)
    return BranchKind.VA3 if m > 0 else BranchKind.VA4


def address(v, t, b):
    m, c = v["marg"][t, b], v["web"][t, b]
    branch = branch_of(m, c)
    wing = 1 + int(v["defect"][t, b] > GATE) + 2 * int(v["kurt"][t, b] > GATE) \
        + 4 * int(v["hiband"][t, b] > GATE)
    sev = sum(min(max(v[k][t, b], 0.0), 12.0) for k in ("web", "kurt", "defect", "hiband"))
    n = int(min(sev, 60))
    return LatticeAddress(CHAIN, branch, wing, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="1st_test")
    ap.add_argument("--truth", default="2,3")
    args = ap.parse_args()
    truth = [int(x) for x in args.truth.split(",")]
    v = _views(args.test)
    T = v["web"].shape[0]
    hours = v["hours"]
    fault = slice(int(0.7 * T), T)

    # 1) the embedding: five variables -> one address -> one Psi point
    bt = truth[0]
    tf = int(0.75 * T)
    a = address(v, tf, bt)
    psi = wing_transform(a.branch, a.chain, a.n, a.wing)
    print(f"Psi embedding on {args.test} (real wing_transform geometry)\n")
    print(f"  one sensor-state -> one address (Bearing {bt+1} @ hour {hours[tf]:.0f}):")
    print(f"     views: web {v['web'][tf,bt]:.0f}  kurt {v['kurt'][tf,bt]:.0f}  "
          f"defect {v['defect'][tf,bt]:.0f}  hi-band {v['hiband'][tf,bt]:.0f}")
    print(f"     -> alpha = (A={a.chain}, b={a.branch.name}, w={a.wing}, n={a.n})  "
          f"= chamber {a.lattice_id.name}")
    print(f"     -> Psi: z={psi.z.real:.0f}{psi.z.imag:+.0f}i  zeta={psi.zeta:.0f}  "
          f"(5 variables in 3 reals + a {a.wing.bit_length()+2}-bit address)\n")

    # 2) chamber occupancy: baseline vs fault -- healthy stays, faulty MIGRATES
    print(f"  chamber migration (dominant chamber, baseline -> fault period):")
    base_psi = {}
    for b in range(4):
        base_ch = Counter(int(address(v, t, b).lattice_id) for t in range(N_BASE))
        fault_ch = Counter(int(address(v, t, b).lattice_id) for t in range(*fault.indices(T)))
        bc, fc = base_ch.most_common(1)[0][0], fault_ch.most_common(1)[0][0]
        # severity reached
        nmax = max(address(v, t, b).n for t in range(*fault.indices(T)))
        star = "  <- truth" if b in truth else ""
        print(f"     Bearing {b+1}: L{bc:02d} -> L{fc:02d}  (peak severity n={nmax})"
              f"{'  MIGRATED' if bc != fc else '  stayed'}{star}")

    # 3) the meet: sources cluster in the source branch VA1, victims in VA2
    print(f"\n  geometric meet (fault-period branch occupancy):")
    for b in range(4):
        brs = Counter(address(v, t, b).branch.name for t in range(*fault.indices(T)))
        tot = sum(brs.values())
        va1 = 100 * brs.get("VA1", 0) / tot
        va2 = 100 * brs.get("VA2", 0) / tot
        role = "SOURCE region" if va1 > va2 else "dragged region"
        star = "  <- truth" if b in truth else ""
        print(f"     Bearing {b+1}: VA1(source) {va1:3.0f}%  VA2(dragged) {va2:3.0f}%  "
              f"-> {role}{star}")
    src = [b for b in range(4)
           if Counter(address(v, t, b).branch for t in range(*fault.indices(T)))
           .get(BranchKind.VA1, 0) > 0.5 * (fault.indices(T)[1] - fault.indices(T)[0])]
    print(f"     => sources meet in VA1: Bearings {sorted(b+1 for b in src)} "
          f"[{'HIT' if sorted(src)==sorted(truth) else 'CHECK'} vs truth {sorted(b+1 for b in truth)}]")

    # 4) the meet PRIMITIVE is real: three anchors equalize to one node
    eq = triple_equalization(3, 5, 7)
    node = eq["ap"][1].coord
    ok = all(p.coord == node for _, p in eq.values())
    print(f"\n  meet primitive (triple_equalization 3,5,7): all rails -> {node} "
          f"({'verified' if ok else 'MISMATCH'}); meets spawn origins = 3 cheap")
    print(f"  new dimension-spaces each (aethos_origins, 3^d at depth d).")

    # 5) honest: Psi-distance is a faithful unified anomaly, not a new detector
    def dist(b):
        bp = np.mean([wing_transform(*( (lambda x: (x.branch,x.chain,x.n,x.wing))(address(v,t,b)) )).coord
                      for t in range(N_BASE)], 0)
        fp = wing_transform(*((lambda x:(x.branch,x.chain,x.n,x.wing))(address(v, int(0.8*T), b)))).coord
        return float(np.linalg.norm(np.array(fp) - bp))
    print(f"\n  Psi-distance from baseline (unified anomaly magnitude):")
    for b in range(4):
        star = "  <- truth" if b in truth else ""
        print(f"     Bearing {b+1}: {dist(b):6.1f}{star}")
    print(f"\n  honest: |Psi| tracks severity, so detection power = the signals it"
          f" embeds. Value = one addressable, glass-box state per sensor that"
          f" unifies electron(branch)+views(wing)+severity(n)+meet(chambers).")


if __name__ == "__main__":
    main()
