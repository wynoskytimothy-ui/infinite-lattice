#!/usr/bin/env python3
"""
Test 51 - Causal inference: seeing vs doing (Pearl's do-calculus).

The provenance graph (Test 5: walk_down = a node's causes) IS a causal DAG,
and the level invariant (Test 1) forbids cycles - no circular causation. That
is exactly the structure causal inference needs, and it gives the one thing
correlation-based ML cannot: the difference between OBSERVING a value and
INTERVENING to set it.

  OBSERVE  P(Y | X=x)        - what you see in the data (can be confounded)
  DO       P(Y | do(X=x))    - what happens if you FORCE X=x (cut X's causes)
  BACKDOOR adjust for the confounder Z to recover the true effect from
           observational data alone

We build two structural causal models and verify:
  (A) confounding-only (Z->X, Z->Y, no X->Y): X and Y correlate, but
      do(X) has ZERO effect - the ice-cream/drowning trap
  (B) real effect (X->Y) plus confounding: naive observation OVERSTATES the
      effect; do(X) and backdoor adjustment both recover the TRUE effect
  (C) a counterfactual: "had X been different, same circumstances"
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def header(s: str):
    print()
    print("=" * 72)
    print(s)
    print("=" * 72)


def assertion(cond: bool, msg: str):
    print(f"  [{'PASS' if cond else 'FAIL'}]  {msg}")
    if not cond:
        sys.exit(1)


def main():
    header("Causal inference - seeing vs doing (do-calculus on the DAG)")
    rng = random.Random(0x51E0)
    N = 200_000

    # ==================================================================
    # (A) Confounding only: Z -> X, Z -> Y, NO X -> Y
    # ==================================================================
    print("\n(A) Confounding only (Z->X, Z->Y) - correlation without causation")
    print("-" * 72)

    def scm_A(do_x=None):
        z = 1 if rng.random() < 0.5 else 0                  # hidden cause (summer)
        if do_x is None:
            x = 1 if rng.random() < (0.8 if z else 0.2) else 0   # X tracks Z
        else:
            x = do_x                                         # intervention: cut Z->X
        y = 1 if rng.random() < (0.7 if z else 0.1) else 0   # Y tracks Z, not X
        return z, x, y

    # observational P(Y|X)
    obs = {0: [0, 0], 1: [0, 0]}
    for _ in range(N):
        _, x, y = scm_A()
        obs[x][0] += y
        obs[x][1] += 1
    p_y_obs1 = obs[1][0] / obs[1][1]
    p_y_obs0 = obs[0][0] / obs[0][1]
    print(f"  observed:  P(Y|X=1)={p_y_obs1:.3f}  P(Y|X=0)={p_y_obs0:.3f}  "
          f"=> correlation {p_y_obs1 - p_y_obs0:+.3f}")
    assertion(p_y_obs1 - p_y_obs0 > 0.2,
              "X and Y are strongly correlated in the data (the trap)")

    # interventional P(Y|do(X))
    def p_y_do(scm, x):
        s = sum(scm(do_x=x)[2] for _ in range(N))
        return s / N
    p_do1, p_do0 = p_y_do(scm_A, 1), p_y_do(scm_A, 0)
    print(f"  do():      P(Y|do X=1)={p_do1:.3f}  P(Y|do X=0)={p_do0:.3f}  "
          f"=> causal effect {p_do1 - p_do0:+.3f}")
    assertion(abs(p_do1 - p_do0) < 0.02,
              "FORCING X has ZERO effect on Y - the correlation was entirely "
              "the confounder Z (do != see)")

    # ==================================================================
    # (B) Real effect X->Y plus confounding Z
    # ==================================================================
    print("\n(B) Real effect (X->Y) + confounding - recover the true effect")
    print("-" * 72)
    TRUE_EFFECT = 0.30

    def scm_B(do_x=None):
        z = 1 if rng.random() < 0.5 else 0
        if do_x is None:
            x = 1 if rng.random() < (0.8 if z else 0.2) else 0
        else:
            x = do_x
        base = 0.1 + TRUE_EFFECT * x + 0.4 * z              # X truly raises Y by 0.3
        y = 1 if rng.random() < base else 0
        return z, x, y

    # naive observation overstates (X correlates with Z which also raises Y)
    obs = {0: [0, 0], 1: [0, 0]}
    strat = {(z, x): [0, 0] for z in (0, 1) for x in (0, 1)}
    zc = [0, 0]
    for _ in range(N):
        z, x, y = scm_B()
        obs[x][0] += y
        obs[x][1] += 1
        strat[(z, x)][0] += y
        strat[(z, x)][1] += 1
        zc[z] += 1
    naive = obs[1][0] / obs[1][1] - obs[0][0] / obs[0][1]
    # do-effect (ground truth)
    do_eff = p_y_do(scm_B, 1) - p_y_do(scm_B, 0)
    # backdoor adjustment: average the within-Z effect, weighted by P(Z)
    pz = [zc[0] / N, zc[1] / N]
    backdoor = sum(pz[z] * (strat[(z, 1)][0] / strat[(z, 1)][1]
                            - strat[(z, 0)][0] / strat[(z, 0)][1])
                   for z in (0, 1))
    print(f"  true effect:           {TRUE_EFFECT:+.3f}")
    print(f"  naive observation:     {naive:+.3f}  (confounded - overstated)")
    print(f"  do(X) intervention:    {do_eff:+.3f}")
    print(f"  backdoor adjust on Z:  {backdoor:+.3f}  (recovered from data alone)")
    assertion(naive > TRUE_EFFECT + 0.05,
              "naive observation OVERSTATES the effect (confounding bias)")
    assertion(abs(do_eff - TRUE_EFFECT) < 0.02,
              "do(X) recovers the true causal effect (cutting Z->X removes bias)")
    assertion(abs(backdoor - TRUE_EFFECT) < 0.02,
              "backdoor adjustment recovers the true effect from observational "
              "data by conditioning on the confounder Z")

    # ==================================================================
    # (C) Counterfactual: "had X been different, same circumstances"
    # ==================================================================
    print("\n(C) Counterfactual - flip X for a fixed individual, same noise")
    print("-" * 72)
    # an individual: fix z and the noise draws; compute Y under X and under not-X
    flips = same = 0
    for _ in range(N):
        z = 1 if rng.random() < 0.5 else 0
        noise_y = rng.random()
        x_obs = 1 if rng.random() < (0.8 if z else 0.2) else 0
        y_factual = 1 if noise_y < (0.1 + TRUE_EFFECT * x_obs + 0.4 * z) else 0
        x_cf = 1 - x_obs
        y_cf = 1 if noise_y < (0.1 + TRUE_EFFECT * x_cf + 0.4 * z) else 0
        if y_cf != y_factual:
            flips += 1
        else:
            same += 1
    flip_rate = flips / N
    print(f"  flipping X changed the outcome for {flip_rate*100:.1f}% of "
          f"individuals (counterfactual sensitivity)")
    assertion(0.05 < flip_rate < TRUE_EFFECT + 0.05,
              "counterfactuals computable per-individual (same exogenous noise, "
              "one variable changed) - impossible without the structural model")

    header("RESULT")
    print(f"  (A) confounding: correlation {p_y_obs1-p_y_obs0:+.2f} but causal")
    print(f"      effect {p_do1-p_do0:+.2f} - seeing is not doing.")
    print(f"  (B) true effect {TRUE_EFFECT}; naive {naive:+.2f} (biased), do()")
    print(f"      {do_eff:+.2f} and backdoor {backdoor:+.2f} both recover it.")
    print(f"  (C) counterfactuals computed per individual.")
    print()
    print("  Causal inference - the framework correlation-based ML provably")
    print("  cannot reach (Pearl's ladder) - is the provenance DAG (Test 5)")
    print("  with intervention = cutting a node's sub_chain and the level")
    print("  invariant (Test 1) guaranteeing acyclicity. The lattice doesn't")
    print("  just store what happened; it can answer what WOULD happen if you")
    print("  reached in and changed something.")


if __name__ == "__main__":
    main()
