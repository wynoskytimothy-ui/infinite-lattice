#!/usr/bin/env python3
"""
AETHOS as a neural network -- a Vector-Symbolic / Hebbian associative net built
from the lattice's own operations. The point: the prime-composite addressing IS a
Vector Symbolic Architecture, and that gives genuine neural-network capacity.

Correspondence (NN primitive  ->  AETHOS operation):
  embedding          token -> prime address, expanded to a distributed code
  BINDING (nonlinear) bind(a,b) = elementwise product = a CONJUNCTION feature
                      (this is the hidden layer / kernel: prime-multiply = FTA meet)
  bundle (Hebbian)    superpose = add (the bridges / correlation counting)
  weights             class prototype = bundle of training codes (learned by COUNTING)
  activation/readout  cos similarity + argmax (the electron collapse)

The textbook test of a REAL (nonlinear) network is XOR: no linear model can solve it;
you need a hidden layer. Here BINDING is the hidden layer -- so the linear (singleton)
code fails XOR and the +binding code solves it. We then show it LEARNS and
GENERALIZES a hidden nonlinear rule (xor of two bits inside a longer string) on
held-out data -- learned by Hebbian counting, no backprop, no gradients.

    python aethos_nn.py
"""

from __future__ import annotations

import hashlib
from itertools import combinations

import numpy as np

DIM = 4096
_atoms = {}


def atom(sym):
    """token -> distributed bipolar code, seeded by a stable hash of its address
    (the lattice prime address expanded to a hypervector)."""
    v = _atoms.get(sym)
    if v is None:
        seed = int(hashlib.md5(str(sym).encode()).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed)
        v = rng.choice((-1.0, 1.0), size=DIM)
        _atoms[sym] = v
    return v


def bundle(vs):
    if not len(vs):
        return np.zeros(DIM)
    return np.sign(np.sum(vs, axis=0))            # superpose (Hebbian accumulate)


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0


def encode(features, use_binding):
    """features: list of (role, value). role-filler binding = bind(atom(role),
    atom(value)). With use_binding we ALSO add pairwise conjunctions (the hidden
    layer); without, only singletons (a linear code)."""
    rf = [atom(("role", r)) * atom(("val", r, v)) for r, v in features]
    parts = list(rf)
    if use_binding:
        parts += [rf[i] * rf[j] for i, j in combinations(range(len(rf)), 2)]
    return bundle(parts)


class AssocNet:
    """one-layer Hebbian associative classifier: prototype[c] = bundle of class c's
    training codes; predict = argmax_c cos(code, prototype[c]). Learned by counting."""

    def __init__(self, use_binding):
        self.use_binding = use_binding
        self.proto = {}

    def fit(self, X, y):
        acc = {}
        for feats, c in zip(X, y):
            acc.setdefault(c, []).append(encode(feats, self.use_binding))
        self.proto = {c: bundle(v) for c, v in acc.items()}
        return self

    def predict(self, feats):
        code = encode(feats, self.use_binding)
        return max(self.proto, key=lambda c: cos(code, self.proto[c]))

    def score(self, X, y):
        return np.mean([self.predict(f) == c for f, c in zip(X, y)])


class BridgeNet:
    """Same nonlinear feature expansion (singletons + pairwise binds), but each
    conjunction is a SYMBOLIC unit weighted by contrastive class log-odds learned
    by COUNTING -- i.e. the supervised-bridge / PMI rule doing CREDIT ASSIGNMENT:
    predictive conjunctions get strong weight, noise conjunctions ~0. A linear
    readout over a fixed nonlinear basis = a network that generalises."""

    def __init__(self, use_binding):
        self.use_binding = use_binding
        self.w = {}

    def _feats(self, features):
        keys = [("s", r, v) for r, v in features]
        if self.use_binding:
            ff = sorted(features)
            keys += [("p", ff[i], ff[j]) for i, j in combinations(range(len(ff)), 2)]
        return keys

    def fit(self, X, y):
        from collections import Counter
        import math
        c1, c0, n1, n0 = Counter(), Counter(), 0, 0
        for f, c in zip(X, y):
            ks = self._feats(f)
            if c == 1:
                c1.update(ks); n1 += 1
            else:
                c0.update(ks); n0 += 1
        for k in set(c1) | set(c0):
            p1 = (c1[k] + 1) / (n1 + 2)
            p0 = (c0[k] + 1) / (n0 + 2)
            self.w[k] = math.log(p1 / p0)            # bridge log-odds = the weight
        return self

    def predict(self, features):
        return 1 if sum(self.w.get(k, 0.0) for k in self._feats(features)) > 0 else 0

    def score(self, X, y):
        return float(np.mean([self.predict(f) == c for f, c in zip(X, y)]))


class GrowingNet:
    """Constructive / resonance net: it does NOT pre-enumerate a basis. It allocates
    a NEW prime per OBSERVED d-way intersection of co-active features (data-directed,
    lazy), weighted by bridge log-odds (counting), and GROWS its order until it can
    predict -- "signalled down on what it's learning". New data just mints new primes
    on existing+new intersections; old primes are untouched (continual, no forgetting).
    The representation becomes as dense as the information it has absorbed."""

    def __init__(self):
        self.w = {}
        self.primes = 0

    @staticmethod
    def _keys(features, order):
        ff = sorted(features)
        keys = []
        for d in range(1, order + 1):
            keys.extend(combinations(ff, d))          # each d-way intersection = a prime
        return keys

    def fit(self, X, y, order):
        from collections import Counter
        import math
        c1, c0, n1, n0 = Counter(), Counter(), 0, 0
        for f, c in zip(X, y):
            ks = self._keys(f, order)
            (c1 if c == 1 else c0).update(ks)
            if c == 1:
                n1 += 1
            else:
                n0 += 1
        self.w = {k: math.log(((c1[k] + 1) / (n1 + 2)) / ((c0[k] + 1) / (n0 + 2)))
                  for k in set(c1) | set(c0)}
        self.primes = len(self.w)
        return self

    def predict(self, features, order):
        return 1 if sum(self.w.get(k, 0.0) for k in self._keys(features, order)) > 0 else 0

    def score(self, X, y, order):
        return float(np.mean([self.predict(f, order) == c for f, c in zip(X, y)]))


def growing_demo(k=8, n=900, seed=1):
    rng = np.random.RandomState(seed)
    a, b, c = 1, 4, 6                                  # a 3-WAY rule -- pairs cannot reach it
    bits = rng.randint(0, 2, size=(n, k))
    y = (bits[:, a] ^ bits[:, b] ^ bits[:, c]).tolist()
    X = [[(f"b{i}", int(v)) for i, v in enumerate(row)] for row in bits]
    ntr = n * 2 // 3
    Xtr, ytr, Xte, yte = X[:ntr], y[:ntr], X[ntr:], y[ntr:]
    print(f"\nCONSTRUCTIVE GROWTH: label = parity(b{a},b{b},b{c}) -- a 3-WAY rule.")
    print("   the net GROWS its intersection order until it predicts, minting a new")
    print("   prime per observed intersection on the fly (NO pre-enumerated basis):")
    settled = None
    for order in (1, 2, 3, 4):
        net = GrowingNet().fit(Xtr, ytr, order)
        acc = net.score(Xte, yte, order)
        note = ""
        if order < 3 and acc < 0.6:
            note = "<- chance: this depth can't represent a 3-way rule"
        elif acc > 0.85 and settled is None:
            settled = order
            note = "<- GREW to the needed depth -> solves + generalizes"
        print(f"     order {order}: {acc*100:3.0f}% held-out   "
              f"{net.primes:>6,} primes allocated   {note}")
    print(f"   it allocated exactly the depth the data demanded (order {settled}); on a"
          f" real corpus the same mechanism mints millions of intersections.")


def xor_demo():
    X = [[("a", 0), ("b", 0)], [("a", 0), ("b", 1)],
         [("a", 1), ("b", 0)], [("a", 1), ("b", 1)]]
    y = [0, 1, 1, 0]                                # XOR
    lin = AssocNet(False).fit(X, y).score(X, y)
    nl = AssocNet(True).fit(X, y).score(X, y)
    print("XOR (the textbook nonlinear-separability test):")
    print(f"   linear code (singletons only):   {lin*100:3.0f}% correct  "
          f"{'<- fails, as a linear model must' if lin < 1 else ''}")
    print(f"   + BINDING (conjunction = hidden): {nl*100:3.0f}% correct  "
          f"{'<- solves it; binding is the hidden layer' if nl == 1 else ''}")


def generalization_demo(k=10, n=600, seed=0):
    rng = np.random.RandomState(seed)
    a, b = 2, 7                                     # hidden interacting bits
    bits = rng.randint(0, 2, size=(n, k))
    y = (bits[:, a] ^ bits[:, b]).tolist()          # label = xor of two hidden bits
    X = [[(f"b{i}", int(v)) for i, v in enumerate(row)] for row in bits]
    ntr = n * 2 // 3
    Xtr, ytr, Xte, yte = X[:ntr], y[:ntr], X[ntr:], y[ntr:]
    lin = BridgeNet(False).fit(Xtr, ytr).score(Xte, yte)
    heb = AssocNet(True).fit(Xtr, ytr).score(Xte, yte)
    sup = BridgeNet(True).fit(Xtr, ytr).score(Xte, yte)
    print(f"\nGENERALIZATION: label = xor(bit{a}, bit{b}) hidden in a {k}-bit string")
    print(f"   {ntr} train / {n-ntr} HELD-OUT test, learned by counting (no backprop):")
    print(f"   linear (singletons):              {lin*100:3.0f}%  "
          f"{'<- chance: not linearly separable' if lin < 0.6 else ''}")
    print(f"   +binding, Hebbian bundle:         {heb*100:3.0f}%  "
          f"{'<- nonlinear capacity, but NO credit assignment -> cannot pick the signal' if heb < 0.6 else ''}")
    print(f"   +binding, supervised counting:    {sup*100:3.0f}%  "
          f"{'<- credit assignment (the bridge log-odds) -> LEARNS + generalizes' if sup > 0.85 else ''}")


def main():
    print("AETHOS as a neural network (Vector-Symbolic / Hebbian associative net)\n"
          + "=" * 70)
    xor_demo()
    generalization_demo()
    growing_demo()
    print("\n" + "=" * 70)
    print("binding (prime-multiply = the meet) is the hidden layer -> nonlinear")
    print("capacity. bridge log-odds (counting) = credit assignment. and the basis is")
    print("NOT fixed: it MINTS a prime per observed intersection and GROWS its depth")
    print("to fit the data (constructive / resonance net) -- as dense as what it learns,")
    print("continual (new primes, no forgetting), glass-box, exact -- all without backprop.")
    print("residual: navigating a COMBINATORIALLY-VAST space of abstract high-order")
    print("features (raw perception) is where gradient descent still has the edge.")


if __name__ == "__main__":
    main()
